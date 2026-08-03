"""The motion gate: ask the model only when the desk has changed and settled.

A VLM answer costs 3-5s and the GPU serializes them, so polling it with "has
anything moved" would run at ~0.2Hz and be busy permanently. A frame diff
answers that question at camera rate for free, so it gates everything.

Waiting for the settle is what makes the pair useful -- firing on the first
changed frame catches the arm mid-sweep, or a hand still over the desk. The
threshold sits between two measured numbers: the desk camera's noise floor (max
0.00016 over 60 static frames) and the smallest event worth catching (a 25x25px
object moving, ~0.0014).
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from mt4_vision.camera import CameraError, FrameStream
from mt4_vision.console import BottomUI
from mt4_vision.qwen_panel import WatchState
from mt4_vision.qwen_prompts import IDENTIFY_PROMPT
from mt4_vision.qwen_worker import QwenWorker


# Fraction of pixels that must change (after blur, above MOTION_DELTA) to count
# as motion. Measured noise floor on the desk camera over 60 static frames:
# mean 0.00008, max 0.00016. A 25x25px object moving produces ~0.0014, so this
# sits ~3x above the worst observed noise and ~3x below the smallest real event.
MOTION_THRESHOLD = 0.0005


# Asked on every motion event until the user types something, which replaces it.
# Deliberately open-ended: at startup the harness has no idea what you care
# about, and "what's here, boxed and described" is the one question that is
# always meaningful -- same schema as /identify, since a typed target isn't
# available yet. Single frame, not before/after -- /track <object> found the
# same thing: grounding on ONE frame is the combination that actually worked
# in testing (boxes landed on the objects), whereas the before/after "did it
# move" framing repeatedly answered "it has not moved" for changes the frame
# diff had already proven. Explicit /watch <q> opts back into the pair for
# genuinely comparative questions.
DEFAULT_WATCH_QUESTION = IDENTIFY_PROMPT.format(obj="distinct object")


MOTION_DELTA = 25          # per-pixel grayscale change that counts


MOTION_SETTLE_TICKS = 3    # consecutive quiet ticks before calling it settled


MOTION_MAX_S = 8.0         # fire anyway if motion never settles


MOTION_TICK_S = 0.1


def _prep(frame: np.ndarray) -> np.ndarray:
    """Grayscale + blur: the blur is what keeps sensor noise under threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def motion_score(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that changed meaningfully between two prepped frames."""
    return float((cv2.absdiff(a, b) > MOTION_DELTA).mean())


def changed_boxes(
    before: np.ndarray, after: np.ndarray, *, max_boxes: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of what changed, largest first -- drawn on the view.

    Not sent to the model; this is so a human can see at a glance whether the
    trigger fired on the thing they care about or on a shadow.
    """
    diff = cv2.absdiff(_prep(before), _prep(after))
    mask = (diff > MOTION_DELTA).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return [(x, y, x + w, y + h) for x, y, w, h in boxes[:max_boxes]]


class MotionWatcher:
    """Fires one before/after question at the model when the desk changes.

    A VLM answer costs 3-5s and the GPU serializes them, so polling it to ask
    "has anything moved" would run at ~0.2Hz and be busy permanently. A frame
    diff answers that same question at camera rate for free, so it gates
    everything: the model is only asked once the scene has moved *and settled*,
    and is handed the last quiet frame plus the first new quiet one.

    Waiting for the settle is what makes the pair useful -- firing on the first
    changed frame would catch the arm mid-sweep, or a hand still over the desk.
    """

    def __init__(
        self,
        stream: FrameStream,
        worker: QwenWorker,
        ui: BottomUI,
        *,
        threshold: float = MOTION_THRESHOLD,
    ) -> None:
        self._stream = stream
        self._worker = worker
        self._ui = ui
        self._lock = threading.Lock()
        self.threshold = threshold
        self._prompt: str | None = None
        self._last_prompt = DEFAULT_WATCH_QUESTION
        self._send = "latest"
        self._state = "off"
        self._score = 0.0
        self._events = 0
        self._skipped = 0
        self._boxes: list[tuple[int, int, int, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="motion", daemon=True)
        self._thread.start()

    def arm(self, prompt: str, send: str = "latest") -> None:
        """``send="latest"`` asks about the new frame alone; ``"pair"`` asks
        about before+after.

        "Where is it now" wants only the new frame -- a coordinate is about a
        single image, and asking for one against a pair leaves it ambiguous
        which frame it refers to. The pair framing also draws repeated false
        negatives ("it has not moved") on small changes, where a plain
        locate-it question on one frame does not. "What changed" is the
        exception that needs both frames -- explicit /watch <q> passes
        ``send="pair"`` for that.
        """
        with self._lock:
            self._prompt = self._last_prompt = prompt
            self._send = send
            self._state = "quiet"
            self._events = self._skipped = 0

    def disarm(self) -> None:
        # _last_prompt outlives the disarm so a bare /watch can resume the same
        # question rather than silently reverting to the generic default.
        with self._lock:
            self._prompt = None
            self._state = "off"

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._prompt is not None

    @property
    def last_question(self) -> str:
        with self._lock:
            return self._last_prompt

    @property
    def send(self) -> str:
        with self._lock:
            return self._send

    def snapshot(self) -> WatchState:
        with self._lock:
            return WatchState(
                state=self._state, score=self._score, events=self._events,
                skipped=self._skipped, boxes=list(self._boxes), send=self._send,
            )

    def _fire(self, before: np.ndarray, after: np.ndarray, peak: float) -> None:
        prompt = self._prompt
        if prompt is None:
            return
        boxes = changed_boxes(before, after)
        with self._lock:
            self._boxes = boxes
        # Dropping an event is honest; queueing them all would build a backlog
        # of stale before/after pairs the user would read as current.
        if self._worker.busy():
            with self._lock:
                self._skipped += 1
            self._ui.emit(f"  [motion {peak:.5f}] skipped -- still answering the last one")
            return
        with self._lock:
            self._events += 1
            n = self._events
            send = self._send
        self._ui.emit("")
        self._ui.emit(f"  [motion #{n}, score {peak:.5f}] {len(boxes)} changed region(s)")
        self._worker.ask(
            prompt,
            frames=[after] if send == "latest" else [before, after],
            mode="images",   # video makes a coordinate's frame ambiguous
            note=f"motion {peak:.5f}",
            boxes=boxes,
            quiet=True,
        )

    def _loop(self) -> None:
        before: np.ndarray | None = None
        prev: np.ndarray | None = None
        quiet_ticks = 0
        started = 0.0
        peak = 0.0
        while not self._stop.is_set():
            time.sleep(MOTION_TICK_S)
            if not self.armed:
                prev = before = None
                continue
            try:
                frame = self._stream.fresh(min_advance=1)
            except CameraError:
                continue
            cur = _prep(frame)
            if prev is None:
                prev, before = cur, frame
                continue
            score = motion_score(prev, cur)
            prev = cur
            with self._lock:
                self._score = score
                state = self._state

            if state == "quiet":
                if score > self.threshold:
                    with self._lock:
                        self._state = "moving"
                    quiet_ticks, started, peak = 0, time.monotonic(), score
                else:
                    before = frame  # rolling last-known-quiet frame
            elif state == "moving":
                peak = max(peak, score)
                if score <= self.threshold:
                    quiet_ticks += 1
                else:
                    quiet_ticks = 0
                timed_out = time.monotonic() - started > MOTION_MAX_S
                if quiet_ticks >= MOTION_SETTLE_TICKS or timed_out:
                    with self._lock:
                        self._state = "quiet"
                    if before is not None:
                        self._fire(before, frame, peak)
                    before = frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
