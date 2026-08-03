"""The serial ask queue behind ``ask_qwen.py``.

Generation is serialized on the service side, so firing a second question while
one is in flight only makes both slower -- hence one at a time. The backlog is a
real queue rather than a single slot because a dropped question is
indistinguishable, in the transcript, from a question the model failed to
answer, and this harness exists to tell those apart.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from mt4_vision.camera import CameraError, FrameStream
from mt4_vision.console import BottomUI
from mt4_vision.preview import annotate_for_pointing, draw_outlined_text
from mt4_vision.qwen import QwenError, Region, Reply, ask, parse_regions
from mt4_vision.qwen_panel import Answer, Options, montage


@dataclass
class Job:
    """One queued question. ``frames`` pre-empts capturing.

    The watcher supplies its own before/after pair, which must not be replaced
    by a fresh capture taken seconds later -- by then the thing that moved has
    stopped being news.
    """

    prompt: str
    with_image: bool = True
    frames: list[np.ndarray] | None = None
    mode: str | None = None      # overrides Options.send_mode
    note: str = ""               # transcript prefix, e.g. a motion score
    # Change regions to outline on the VIEW only, in posted-frame coordinates.
    # Supplied by the watcher, which already computed them to decide to fire.
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


def format_region(r: Region) -> str:
    """One line: kind, label, compact bbox/point, description if any."""
    coords = ", ".join(f"{c:g}" for c in r.coords)
    suffix = f" -- {r.description}" if r.description else ""
    return f"{r.kind} {r.label or '?'}: [{coords}]{suffix}"


def _display_lines(text: str, regions: list[Region]) -> list[str]:
    """Split an answer into lines for the terminal transcript.

    BottomUI.emit() wraps a too-long line across rows rather than dropping
    its tail, so a JSON array on one line -- the schema /identify and the
    default both reply with -- keeps all its text on screen.
    But ``json.dumps(indent=2)`` explodes every ``bbox_2d`` into 6 lines of
    its own, which is technically readable and practically a scroll-fest for
    a handful of objects. ``regions`` is already the exact fields worth
    reading (label, box, description) with everything else -- schema
    boilerplate, brackets, per-number lines -- stripped, so a box/point reply
    prints as one line each instead. Prose and non-grounding JSON (a bare
    count, an object with no bbox/point key) still get the general
    fallbacks. The JSON kept for /save is the untouched original text --
    this only reshapes what gets echoed live.
    """
    if not text:
        return ["(empty response)"]
    stripped = text.strip()
    looks_json = stripped[:1] in "[{"
    whole: object = None
    if looks_json:
        try:
            whole = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            whole = None  # not JSON, or cut short by max_new_tokens

    # Unlabelled regions mean parse_regions fell back to bare coordinates, so
    # the summary would carry strictly less than the text it replaces -- show
    # the text instead. Never let the summary be the only thing printed when
    # it is not actually a summary of what came back.
    if regions and any(r.label for r in regions):
        lines = [format_region(r) for r in regions]
        if looks_json and whole is None:
            lines.append("!! reply cut off mid-JSON -- raise /tokens; complete entries only")
        return lines
    if whole is not None:
        return json.dumps(whole, indent=2).split("\n")
    return text.split("\n")


class QwenWorker:
    """Serial ask queue: one request on the GPU at a time, FIFO behind it.

    Generation is serialized on the service side, so firing a second question
    while one is in flight only makes both slower -- hence one at a time. The
    backlog is a real queue rather than a single slot because a dropped
    question is indistinguishable, in the transcript, from a question the
    model failed to answer, and this harness exists to tell those apart.
    ``/cancel`` discards what has not started yet.
    """

    def __init__(
        self,
        *,
        stream: FrameStream,
        url: str,
        opts: Options,
        ui: BottomUI,
    ) -> None:
        self._stream = stream
        self._url = url
        self._ui = ui
        self._lock = threading.Lock()
        self.opts = opts
        self._queue: list[Job] = []
        self._submitted: tuple[str, bool] | None = None  # most recent ask()
        self._pending: str | None = None
        self._started = 0.0
        self._answer: Answer | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qwen-ask", daemon=True)
        self._thread.start()

    # -- called from the main thread ---------------------------------------

    def ask(
        self,
        prompt: str,
        *,
        with_image: bool = True,
        frames: list[np.ndarray] | None = None,
        mode: str | None = None,
        note: str = "",
        boxes: list[tuple[int, int, int, int]] | None = None,
        quiet: bool = False,
    ) -> None:
        job = Job(prompt=prompt, with_image=with_image, frames=frames,
                  mode=mode, note=note, boxes=list(boxes or []))
        with self._lock:
            self._queue.append(job)
            self._submitted = (prompt, with_image)
            waiting = len(self._queue) + (1 if self._pending else 0)
        self._wake.set()
        if not quiet:
            self._ui.set_status(
                "asking..." if waiting == 1 else f"queued ({waiting} waiting)"
            )

    def busy(self) -> bool:
        """True if anything is in flight or waiting -- the watcher's back-off."""
        with self._lock:
            return self._pending is not None or bool(self._queue)

    def cancel_queued(self) -> int:
        """Drop everything not yet started. Returns how many were dropped."""
        with self._lock:
            dropped = len(self._queue)
            self._queue.clear()
        return dropped

    def clear(self) -> None:
        with self._lock:
            self._answer = None

    def last(self) -> Answer | None:
        with self._lock:
            return self._answer

    def last_submitted(self) -> tuple[str, bool] | None:
        """The most recently *asked* question -- what ``/again`` repeats.

        Deliberately not the most recently *answered* one: asking again while
        the first attempt is still running is the normal way to check whether
        an answer is stable, and it must not silently repeat something older.
        """
        with self._lock:
            return self._submitted

    def snapshot(self) -> tuple[Answer | None, str | None, float]:
        with self._lock:
            elapsed = (time.monotonic() - self._started) if self._pending else 0.0
            return self._answer, self._pending, elapsed

    def drain(self, timeout: float = 300.0) -> None:
        """Block until nothing is queued or in flight.

        Called on a clean exit so a piped batch of questions still gets every
        answer into the transcript -- without this, EOF on stdin arrives long
        before the model has finished the question it just took.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._pending is None and not self._queue:
                    return
            time.sleep(0.1)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)

    def grab(self) -> np.ndarray:
        """One fresh frame off the shared, continuously-drained stream.

        Never a one-shot open/release: on Windows DSHOW the device is not free
        again by the time the next open lands, so back-to-back captures (a
        ``/montage``, or simply two quick questions) fail with "could not open
        camera". The stream is held for the whole session instead.
        """
        return self._stream.fresh(min_advance=1)

    # -- worker thread ----------------------------------------------------

    def _capture(self, opts: Options) -> list[np.ndarray]:
        """The frames to send: the pinned one, or a fresh timed sequence."""
        if opts.pinned is not None:
            return [opts.pinned.copy()]
        want = 1 if opts.send_mode == "single" else max(1, opts.frames)
        frames = [self.grab()]
        for _ in range(want - 1):
            time.sleep(opts.frame_gap_s)
            frames.append(self.grab())
        return frames

    def _build_payload(
        self, prompt: str, opts: Options, frames: list[np.ndarray] | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, str, str]:
        """Return (frames_to_post, image_to_display, sent_prompt, send_mode).

        The displayed image and the posted frames diverge for the multi-frame
        modes: the model gets N frames, the human gets them tiled so all N are
        visible at once. ``Answer.sent_label`` names what actually went, so the
        view can never be mistaken for the payload.
        """
        if frames is None:
            frames = self._capture(opts)
        mode = opts.send_mode if len(frames) > 1 else "single"

        preamble = ""
        if mode == "montage":
            preamble = (
                f"This image is a {len(frames)}-frame time sequence from a fixed "
                f"camera, captured {opts.frame_gap_s:g} seconds apart and tiled "
                f"left-to-right then top-to-bottom, numbered 1 to {len(frames)}. "
            )
        elif mode == "images":
            preamble = (
                f"These {len(frames)} images are consecutive frames from one fixed "
                f"camera, in time order, {opts.frame_gap_s:g} seconds apart. "
            )
        # Video needs no preamble: the service labels each temporal group with
        # a real timestamp derived from fps, which is stronger than prose.

        if mode == "montage":
            posted = [montage(frames)]
        else:
            posted = frames
        if opts.grid:
            posted = [annotate_for_pointing(f) for f in posted]

        # Change boxes go on COPIES used only for the view. Drawing them on the
        # posted frames would alter what the model sees, and marking up the very
        # difference we are asking it to find would make the answer worthless.
        # Drawing them per-frame before tiling also keeps them in each frame's
        # own coordinate space, so montage scales them along with the pixels --
        # boxes drawn onto the finished composite land in the wrong place.
        shown = posted
        if boxes:
            shown = [f.copy() for f in posted]
            for frame in shown:
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        display = shown[0] if len(shown) == 1 else montage(shown)
        return posted, display, preamble + prompt, mode

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            with self._lock:
                job = self._queue.pop(0) if self._queue else None
                if job is not None:
                    self._pending = job.prompt
                    self._started = time.monotonic()
                opts = replace(self.opts)
            if job is None:
                continue

            prompt, with_image = job.prompt, job.with_image
            if job.mode is not None:
                opts = replace(opts, send_mode=job.mode)
            started = time.monotonic()
            sent_prompt, error, text = prompt, None, ""
            reply: Reply | None = None
            send_mode = "single"
            frame_hw: tuple[int, int] | None = None
            img = np.full((360, 640, 3), 40, dtype=np.uint8)
            draw_outlined_text(
                img, "text-only prompt (no image sent)", (24, 190),
                scale=0.7, color=(200, 200, 200),
            )
            try:
                posted: list[np.ndarray] = []
                if with_image:
                    posted, img, sent_prompt, send_mode = self._build_payload(
                        prompt, opts, frames=job.frames, boxes=job.boxes,
                    )
                    if posted:
                        frame_hw = (posted[0].shape[0], posted[0].shape[1])
                reply = ask(
                    sent_prompt,
                    posted if with_image else None,
                    mode="video" if send_mode == "video" else "images",
                    fps=opts.fps,
                    url=self._url,
                    max_new_tokens=opts.tokens,
                    do_sample=not opts.greedy,
                )
                text = reply.text
            except CameraError as exc:
                error = f"camera: {exc}"
            except QwenError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001 -- keep the harness alive
                error = f"{type(exc).__name__}: {exc}"

            answer = Answer(
                prompt=prompt,
                sent_prompt=sent_prompt,
                sent=img,
                text=text,
                regions=parse_regions(text) if text else [],
                latency_s=time.monotonic() - started,
                error=error,
                tokens=opts.tokens,
                had_image=with_image,
                send_mode=send_mode,
                reply=reply,
                frame_hw=frame_hw,
            )
            with self._lock:
                self._answer = answer
                self._pending = None

            self._ui.emit("")
            self._ui.emit(f"> {prompt}   [{answer.sent_label()}]")
            if error:
                self._ui.emit(f"  !! {error}")
                self._ui.set_status(f"failed after {answer.latency_s:.1f}s")
            else:
                warning = reply.frame_warning() if reply is not None else None
                if warning:
                    self._ui.emit(f"  !! {warning}")
                for line in _display_lines(text, answer.regions):
                    self._ui.emit(f"  {line}")
                note = f"{answer.latency_s:.1f}s"
                if reply is not None and reply.prompt_tokens:
                    note += f", {reply.prompt_tokens} prompt tok"
                if answer.regions:
                    note += f", {len(answer.regions)} region(s)"
                self._ui.set_status(f"answered in {note}")
