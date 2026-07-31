#!/usr/bin/env python3
"""Interactive Qwen3-VL harness: ask questions about what the camera sees.

Same shape as ``move_object_to_marker.py`` -- a pinned prompt line, a
background worker, a live annotated preview -- but the worker asks a
vision-language model instead of moving the arm, and the answer is text
rather than a box. Nothing here touches the arm unless you pass ``--park``.

The window is split: the submitted frame on the left (with any boxes/points
Qwen named drawn on it), an answer panel on the right, and a small live inset
so you can still aim the camera while an answer is on screen. What you see on
the left is always *the exact image that was POSTed* -- which is the point.
A wrong answer about a frame the arm was blocking is a different problem from
a wrong answer about a clean frame, and no amount of reading the text alone
tells you which happened.

Coordinates come back in whichever space this model build uses (see
``parse_regions``); ``/coords`` flips the interpretation so you can settle it
by eye in two keystrokes rather than guessing.

Prereqs:
  * The service running, and ``.\\scripts\\start_qwen_tunnel.ps1`` if remote
  * A camera. No calibration, no serial, no arm (unless ``--park``)

Example::

    python ask_qwen.py --camera 1
    python ask_qwen.py --prompt "how many cubes are on the desk?" --save
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from mt4_vision.camera import DEFAULT_CAMERA_INDEX, CameraError, FrameStream, capture_frame
from mt4_vision.console import BottomUI
from mt4_vision.preview import (
    LivePreview,
    PreviewStopped,
    annotate_for_pointing,
    draw_outlined_text,
)
from mt4_vision.qwen import DEFAULT_URL, QwenError, Region, generate, health, parse_regions

PANEL_W = 440
PANEL_PAD = 14
BODY_SCALE = 0.45
BODY_LINE_PX = 18
HEAD_SCALE = 0.52
INSET_W = 208

BG = (28, 28, 30)
RULE = (72, 72, 78)
DIM = (150, 150, 156)
TEXT = (238, 238, 240)
Q_BGR = (0, 200, 255)      # amber
A_BGR = (130, 230, 150)    # green
ERR_BGR = (90, 90, 250)    # red
WARN_BGR = (0, 200, 255)

# Distinguishable per-region colors, cycled.
REGION_BGR = [
    (0, 255, 255), (255, 128, 0), (255, 0, 255),
    (0, 255, 128), (128, 128, 255), (255, 255, 0),
]

# Capability probes worth running against any new VLM build before trusting
# it for anything on the desk. Ordered easiest-to-hardest: description and
# counting usually pass, grounding and fine spatial relations are where a
# small quantized model starts inventing.
PRESETS: list[tuple[str, str]] = [
    ("describe", "Describe what you see on the desk, briefly."),
    ("inventory", "List every distinct object you can see. One per line, no commentary."),
    ("count", "How many cubes are in this image? Reply with a single number and nothing else."),
    ("colors", "List each cube and its color, one per line, as 'color: cube'."),
    ("ground", 'Locate every object on the desk. Reply ONLY with JSON: '
               '[{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}]'),
    ("point", 'Point at the object nearest the centre of the desk. Reply ONLY with JSON: '
              '[{"point_2d": [x, y], "label": "<name>"}]'),
    ("ocr", "Read any text, numbers or codes visible in the image, verbatim."),
    ("tags", "There are small printed square fiducial tags on the desk. "
             "How many can you see, and roughly where is each one?"),
    ("spatial", "Which object is closest to the robot arm's gripper, and which is furthest? "
                "Answer in one sentence."),
    ("graspable", "A small two-finger parallel gripper is going to pick one object. "
                  "Which is the easiest and which is the hardest, and why?"),
    ("arm", "Is a robot arm visible in this image? If so, describe where it is and "
            "whether it is blocking your view of the desk."),
]

HELP = """commands (anything else is asked verbatim):
  <text>            ask about the current frame
  <Enter>           back to the live feed
  /help             this list
  /freeze  /thaw    pin the current frame, so repeat asks use one identical image
  /again            re-ask the last question (on the pinned frame if frozen)
  /cancel           drop queued questions that have not started yet
  /grid             toggle the labelled pixel grid drawn on the sent image
  /coords abs|norm  reinterpret returned coords as pixels or 0-1000 normalized
  /tokens N         max_new_tokens (default 256)
  /montage N [S]    ask about N frames S seconds apart, tiled as one image
  /noimage          ask the next question text-only (no frame)
  /preset [N]       list capability probes, or run one
  /save [name]      write the sent image, the view and a JSON record
  /health           re-query the service
  Ctrl+C / Ctrl+Z   quit"""


@dataclass
class Options:
    tokens: int = 256
    grid: bool = False
    # This build answers in 0-1000 normalized coords, verified against the
    # desk camera -- see mt4_vision.qwen.parse_regions. /coords flips it.
    coord_mode: str = "norm"  # "norm" | "abs"
    montage: int = 1
    montage_gap_s: float = 1.0
    pinned: np.ndarray | None = None


@dataclass
class Answer:
    prompt: str            # what the user typed
    sent_prompt: str       # what was actually POSTed (montage preamble etc.)
    sent: np.ndarray       # the exact image POSTed (None-image asks: a placeholder)
    text: str = ""
    regions: list[Region] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None
    tokens: int = 256
    had_image: bool = True


# --------------------------------------------------------------------------- #
# drawing


def _wrap(text: str, max_px: int, scale: float) -> list[str]:
    """Word-wrap to a pixel width, hard-breaking words that never fit."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    def width(s: str) -> int:
        (w, _h), _b = cv2.getTextSize(s, font, scale, 1)
        return w

    lines: list[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if width(trial) <= max_px:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            while width(word) > max_px and len(word) > 1:
                cut = len(word)
                while cut > 1 and width(word[:cut]) > max_px:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            cur = word
        if cur:
            lines.append(cur)
    return lines


def draw_regions(
    img: np.ndarray, regions: list[Region], coord_mode: str,
) -> tuple[int, int]:
    """Draw boxes/points on ``img``. Returns (drawn, in_bounds) counts."""
    h, w = img.shape[:2]
    inside = 0
    for i, raw in enumerate(regions):
        r = raw.scaled(w / 1000.0, h / 1000.0) if coord_mode == "norm" else raw
        color = REGION_BGR[i % len(REGION_BGR)]
        if r.in_bounds(w, h):
            inside += 1
        label = r.label or f"#{i + 1}"
        if r.kind == "box":
            x1, y1, x2, y2 = (int(v) for v in r.coords)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            draw_outlined_text(
                img, label, (x1 + 3, max(14, y1 - 6)), scale=0.5, color=color,
            )
        else:
            cx, cy = int(r.cx), int(r.cy)
            cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 22, 2)
            cv2.circle(img, (cx, cy), 13, color, 2)
            draw_outlined_text(
                img, label, (cx + 16, cy + 4), scale=0.5, color=color,
            )
    return len(regions), inside


def _inset(canvas: np.ndarray, frame: np.ndarray, tag: str) -> None:
    """Drop a small labelled thumbnail into the canvas' bottom-right."""
    h, w = frame.shape[:2]
    tw = INSET_W
    th = max(1, int(round(h * tw / w)))
    thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    ch, cw = canvas.shape[:2]
    x0, y0 = cw - tw - 12, ch - th - 12
    if x0 < 0 or y0 < 0:
        return
    canvas[y0 : y0 + th, x0 : x0 + tw] = thumb
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), (220, 220, 220), 1)
    draw_outlined_text(canvas, tag, (x0 + 6, y0 + 18), scale=0.5, color=(220, 220, 220))


def render_panel(
    height: int,
    *,
    answer: Answer | None,
    pending: str | None,
    elapsed: float,
    opts: Options,
    svc: str,
    frame_shape: tuple[int, int],
    region_counts: tuple[int, int] | None,
) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    x = PANEL_PAD
    body_px = PANEL_W - 2 * PANEL_PAD
    y = 26

    def rule(yy: int) -> int:
        cv2.line(panel, (x, yy), (PANEL_W - PANEL_PAD, yy), RULE, 1)
        return yy + 16

    def block(label: str, text: str, color: tuple[int, int, int], yy: int,
              *, limit: int | None = None) -> int:
        draw_outlined_text(panel, label, (x, yy), scale=0.42, color=DIM)
        yy += BODY_LINE_PX
        lines = _wrap(text, body_px, BODY_SCALE)
        clipped = 0
        if limit is not None and len(lines) > limit:
            clipped = len(lines) - limit
            lines = lines[:limit]
        for line in lines:
            draw_outlined_text(panel, line, (x, yy), scale=BODY_SCALE, color=color)
            yy += BODY_LINE_PX
        if clipped:
            draw_outlined_text(
                panel, f"... +{clipped} more lines (full text in terminal)",
                (x, yy), scale=0.4, color=DIM,
            )
            yy += BODY_LINE_PX
        return yy + 6

    h, w = frame_shape
    draw_outlined_text(panel, "QWEN3-VL PROBE", (x, y), scale=HEAD_SCALE, color=TEXT)
    y += BODY_LINE_PX
    draw_outlined_text(panel, svc, (x, y), scale=0.4, color=DIM)
    y += BODY_LINE_PX
    flags = f"{w}x{h}  {opts.tokens} tok  coords:{opts.coord_mode}"
    if opts.grid:
        flags += "  grid"
    if opts.pinned is not None:
        flags += "  FROZEN"
    if opts.montage > 1:
        flags += f"  montage:{opts.montage}"
    draw_outlined_text(panel, flags, (x, y), scale=0.4, color=DIM)
    y = rule(y + 10)

    if pending is not None:
        y = block("ASKING", pending, Q_BGR, y, limit=4)
        draw_outlined_text(
            panel, f"thinking... {elapsed:4.1f}s", (x, y),
            scale=BODY_SCALE, color=WARN_BGR,
        )
        y = rule(y + 12)

    if answer is None:
        if pending is None:
            for line in _wrap(
                "Type a question to ask about the current frame. "
                "/help lists the commands, /preset lists capability probes.",
                body_px, BODY_SCALE,
            ):
                draw_outlined_text(panel, line, (x, y), scale=BODY_SCALE, color=DIM)
                y += BODY_LINE_PX
        return panel

    y = block("Q", answer.prompt, Q_BGR, y, limit=5)
    y = rule(y)
    if answer.error:
        y = block("ERROR", answer.error, ERR_BGR, y, limit=8)
    else:
        remaining = max(2, (height - y - 70) // BODY_LINE_PX)
        y = block("A", answer.text or "(empty response)", A_BGR, y, limit=remaining)

    footer = height - 42
    cv2.line(panel, (x, footer), (PANEL_W - PANEL_PAD, footer), RULE, 1)
    stats = f"{answer.latency_s:.1f}s"
    if not answer.had_image:
        stats += "  text-only"
    if region_counts is not None and region_counts[0]:
        drawn, inside = region_counts
        stats += f"  {drawn} region{'s' if drawn != 1 else ''}"
        if inside < drawn:
            stats += f"  ({drawn - inside} off-frame)"
    draw_outlined_text(panel, stats, (x, footer + 20), scale=0.42, color=DIM)
    if region_counts is not None and region_counts[0] and region_counts[1] == 0:
        other = "norm" if opts.coord_mode == "abs" else "abs"
        draw_outlined_text(
            panel, f"! all coords off-frame -- try /coords {other}",
            (x, footer + 36), scale=0.42, color=WARN_BGR,
        )
    return panel


def compose(
    live: np.ndarray,
    *,
    answer: Answer | None,
    pending: str | None,
    elapsed: float,
    opts: Options,
    svc: str,
) -> np.ndarray:
    """Main pane (submitted frame, or live when nothing is being shown) + panel."""
    counts: tuple[int, int] | None = None
    if answer is not None:
        main = answer.sent.copy()
        if answer.regions:
            counts = draw_regions(main, answer.regions, opts.coord_mode)
        _inset(main, live, "LIVE")
    elif pending is not None:
        main = live.copy()
    else:
        main = live.copy()
        draw_outlined_text(main, "LIVE", (14, 28), scale=0.55, color=(220, 220, 220))

    panel = render_panel(
        main.shape[0],
        answer=answer, pending=pending, elapsed=elapsed, opts=opts, svc=svc,
        frame_shape=(main.shape[0], main.shape[1]), region_counts=counts,
    )
    return np.hstack([main, panel])


def montage(frames: list[np.ndarray]) -> np.ndarray:
    """Tile frames into one numbered image, left-to-right, top-to-bottom.

    The service takes a single image per request, so a time sequence has to
    become one picture to be askable at all. Numbering each tile is what
    makes an answer checkable -- without it there is no way to tell whether
    the model understood the ordering or guessed.
    """
    n = len(frames)
    if n == 1:
        return frames[0]
    cols = 1 if n == 1 else (2 if n <= 4 else 3)
    rows = (n + cols - 1) // cols
    h, w = frames[0].shape[:2]
    tw, th = w // cols, h // cols
    canvas = np.full((th * rows, tw * cols, 3), 20, dtype=np.uint8)
    for i, f in enumerate(frames):
        tile = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
        draw_outlined_text(
            canvas, str(i + 1), (c * tw + 10, r * th + 34), scale=1.0,
            color=(0, 220, 255),
        )
        cv2.rectangle(
            canvas, (c * tw, r * th), (c * tw + tw - 1, r * th + th - 1),
            (60, 60, 60), 1,
        )
    return canvas


# --------------------------------------------------------------------------- #
# worker


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
        self._queue: list[tuple[str, bool]] = []  # (prompt, with_image)
        self._submitted: tuple[str, bool] | None = None  # most recent ask()
        self._pending: str | None = None
        self._started = 0.0
        self._answer: Answer | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qwen-ask", daemon=True)
        self._thread.start()

    # -- called from the main thread ---------------------------------------

    def ask(self, prompt: str, *, with_image: bool = True) -> None:
        with self._lock:
            self._queue.append((prompt, with_image))
            self._submitted = (prompt, with_image)
            waiting = len(self._queue) + (1 if self._pending else 0)
        self._wake.set()
        self._ui.set_status(
            "asking..." if waiting == 1 else f"queued ({waiting} waiting)"
        )

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

    def _build_image(self, opts: Options) -> tuple[np.ndarray, str]:
        """The image to POST, plus any prompt preamble it needs."""
        if opts.pinned is not None:
            base = opts.pinned.copy()
            frames = [base]
        else:
            frames = [self.grab()]
            for _ in range(max(0, opts.montage - 1)):
                time.sleep(opts.montage_gap_s)
                frames.append(self.grab())

        preamble = ""
        if len(frames) > 1:
            preamble = (
                f"This image is a {len(frames)}-frame time sequence from a fixed "
                f"camera, captured {opts.montage_gap_s:g} seconds apart and tiled "
                f"left-to-right then top-to-bottom, numbered 1 to {len(frames)}. "
            )
        img = montage(frames)
        if opts.grid:
            img = annotate_for_pointing(img)
        return img, preamble

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            with self._lock:
                job = self._queue.pop(0) if self._queue else None
                if job is not None:
                    self._pending = job[0]
                    self._started = time.monotonic()
                opts = replace(self.opts)
            if job is None:
                continue

            prompt, with_image = job
            started = time.monotonic()
            sent_prompt, error, text = prompt, None, ""
            img = np.full((360, 640, 3), 40, dtype=np.uint8)
            draw_outlined_text(
                img, "text-only prompt (no image sent)", (24, 190),
                scale=0.7, color=(200, 200, 200),
            )
            try:
                if with_image:
                    img, preamble = self._build_image(opts)
                    sent_prompt = preamble + prompt
                text = generate(
                    sent_prompt,
                    img if with_image else None,
                    url=self._url,
                    max_new_tokens=opts.tokens,
                )
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
            )
            with self._lock:
                self._answer = answer
                self._pending = None

            self._ui.emit("")
            self._ui.emit(f"> {prompt}")
            if error:
                self._ui.emit(f"  !! {error}")
                self._ui.set_status(f"failed after {answer.latency_s:.1f}s")
            else:
                for line in (text or "(empty response)").split("\n"):
                    self._ui.emit(f"  {line}")
                note = f"{answer.latency_s:.1f}s"
                if answer.regions:
                    note += f", {len(answer.regions)} region(s)"
                self._ui.set_status(f"answered in {note}")


class HarnessPreview:
    """Composites the live feed, the submitted frame and the answer panel."""

    def __init__(
        self, stream: FrameStream, worker: QwenWorker, *, svc: str,
    ) -> None:
        self._stream = stream
        self._worker = worker
        self._svc = svc
        self._preview = LivePreview("qwen probe (q or Esc to stop)")
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._user_quit = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qwen-view", daemon=True)
        self._thread.start()

    def stopped_by_user(self) -> bool:
        return self._user_quit.is_set()

    def latest_canvas(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                live = self._stream.fresh(min_advance=1)
            except CameraError:
                time.sleep(0.2)
                continue
            answer, pending, elapsed = self._worker.snapshot()
            try:
                canvas = compose(
                    live, answer=answer, pending=pending, elapsed=elapsed,
                    opts=self._worker.opts, svc=self._svc,
                )
            except Exception:  # noqa: BLE001 -- a draw bug must not kill the feed
                time.sleep(0.2)
                continue
            with self._lock:
                self._latest = canvas
            try:
                self._preview.show(canvas)
            except PreviewStopped:
                self._user_quit.set()
                break

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._preview.close()


# --------------------------------------------------------------------------- #
# commands


def save_probe(
    outdir: Path, answer: Answer, canvas: np.ndarray | None, name: str = "",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S")
    if name:
        stem = f"{stem}_{name}"
    cv2.imwrite(str(outdir / f"{stem}_sent.jpg"), answer.sent)
    if canvas is not None:
        cv2.imwrite(str(outdir / f"{stem}_view.jpg"), canvas)
    record = {
        "prompt": answer.prompt,
        "sent_prompt": answer.sent_prompt,
        "response": answer.text,
        "error": answer.error,
        "latency_s": round(answer.latency_s, 3),
        "max_new_tokens": answer.tokens,
        "had_image": answer.had_image,
        "sent_size": [answer.sent.shape[1], answer.sent.shape[0]],
        "regions": [
            {"label": r.label, "kind": r.kind, "coords": list(r.coords)}
            for r in answer.regions
        ],
    }
    path = outdir / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def handle_command(
    line: str, *, worker: QwenWorker, ui: BottomUI, view: HarnessPreview | None,
    url: str, outdir: Path,
) -> bool:
    """Run a ``/command``. Returns False if the line was not a command."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    rest = parts[1:]
    opts = worker.opts

    if cmd in ("help", "h", "?"):
        ui.emit("")
        for row in HELP.split("\n"):
            ui.emit(row)
        ui.set_status("ready")
    elif cmd == "freeze":
        if opts.montage > 1:
            ui.set_status("/montage is on -- /montage 1 first (a pinned frame is not a sequence)")
            return True
        try:
            frame = worker.grab()
        except CameraError as exc:
            ui.set_status(f"camera: {exc}")
            return True
        opts.pinned = frame
        ui.set_status("frame frozen -- asks now reuse this exact image (/thaw to release)")
    elif cmd == "thaw":
        opts.pinned = None
        ui.set_status("thawed -- asks capture a fresh frame")
    elif cmd == "again":
        prev = worker.last_submitted()
        if prev is None:
            ui.set_status("nothing asked yet")
        else:
            worker.ask(prev[0], with_image=prev[1])
    elif cmd == "cancel":
        dropped = worker.cancel_queued()
        ui.set_status(
            f"dropped {dropped} queued question(s)" if dropped
            else "nothing queued (the one in flight cannot be recalled)"
        )
    elif cmd == "grid":
        opts.grid = not opts.grid
        ui.set_status(f"grid {'on' if opts.grid else 'off'} (redrawn on the next ask)")
    elif cmd == "coords":
        mode = (rest[0].lower() if rest else "")
        if mode.startswith("n"):
            opts.coord_mode = "norm"
        elif mode.startswith("a"):
            opts.coord_mode = "abs"
        else:
            opts.coord_mode = "norm" if opts.coord_mode == "abs" else "abs"
        ui.set_status(f"coords read as {opts.coord_mode}")
    elif cmd == "tokens":
        try:
            opts.tokens = max(1, int(rest[0]))
        except (IndexError, ValueError):
            ui.set_status("usage: /tokens N")
        else:
            ui.set_status(f"max_new_tokens = {opts.tokens}")
    elif cmd == "montage":
        try:
            want = max(1, min(9, int(rest[0])))
            gap = max(0.0, float(rest[1])) if len(rest) > 1 else opts.montage_gap_s
        except (IndexError, ValueError):
            ui.set_status("usage: /montage N [seconds-between]")
            return True
        # A frozen frame would be tiled with copies of itself and answered as
        # if it were a sequence -- a wrong answer with no visible cause.
        if want > 1 and opts.pinned is not None:
            ui.set_status("frame is frozen -- /thaw first, a montage needs live frames")
            return True
        opts.montage, opts.montage_gap_s = want, gap
        ui.set_status(
            f"montage {want} frames, {gap:g}s apart" if want > 1
            else "montage off (single frame)"
        )
    elif cmd == "noimage":
        prev = worker.last_submitted()
        prompt = " ".join(rest) or (prev[0] if prev else "")
        if not prompt:
            ui.set_status("usage: /noimage <question>")
        else:
            worker.ask(prompt, with_image=False)
    elif cmd in ("preset", "p"):
        if not rest:
            ui.emit("")
            ui.emit("presets:")
            for i, (name, prompt) in enumerate(PRESETS, start=1):
                ui.emit(f"  {i:2d}. {name:<10} {prompt[:70]}")
            ui.set_status("run one with /preset N")
        else:
            try:
                name, prompt = PRESETS[int(rest[0]) - 1]
            except (ValueError, IndexError):
                ui.set_status(f"no such preset (1-{len(PRESETS)})")
            else:
                ui.set_status(f"preset {name}")
                worker.ask(prompt)
    elif cmd == "save":
        last = worker.last()
        if last is None:
            ui.set_status("nothing to save yet")
        else:
            canvas = view.latest_canvas() if view is not None else None
            path = save_probe(outdir, last, canvas, " ".join(rest))
            ui.set_status(f"saved {path}")
    elif cmd == "health":
        try:
            info = health(url)
        except QwenError as exc:
            ui.set_status(str(exc))
        else:
            ui.emit(f"  health: {json.dumps(info)}")
            ui.set_status("service ok")
    elif cmd in ("live", "clear"):
        worker.clear()
        ui.set_status("live")
    else:
        ui.set_status(f"unknown command /{cmd} -- try /help")
    return True


# --------------------------------------------------------------------------- #


def camera_hint() -> str:
    """Which indices can be opened at all, for when auto-detect gives up.

    ``--camera`` defaults to the shared ArUco auto-detect, which answers "which
    camera sees the calibrated desk" -- but this harness needs no calibration
    and no markers, and the tags are routinely occluded by the very objects
    being probed. So a failure here is usually "pick an index", not "fix the
    desk", and the useful reply is the list of indices that exist.
    """
    openable = []
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        try:
            if cap.isOpened() and cap.read()[0]:
                openable.append(i)
        finally:
            cap.release()
    if not openable:
        return "no camera index 0-5 could be opened at all"
    return f"openable camera indices: {openable} -- pass one with --camera"


def run_once(args: argparse.Namespace, svc: str) -> int:
    """Non-interactive single question, for scripting."""
    frame = None
    if not args.no_image:
        try:
            frame = capture_frame(args.camera)
        except CameraError as exc:
            print(f"camera: {exc}\n{camera_hint()}", file=sys.stderr)
            return 1
        if args.grid:
            frame = annotate_for_pointing(frame)
    started = time.monotonic()
    try:
        text = generate(
            args.prompt, frame, url=args.url, max_new_tokens=args.tokens,
        )
    except QwenError as exc:
        print(f"qwen: {exc}", file=sys.stderr)
        return 1
    latency = time.monotonic() - started
    regions = parse_regions(text)

    print(f"# {svc}  ({latency:.1f}s)")
    print(text)
    for r in regions:
        coords = ", ".join(f"{c:g}" for c in r.coords)
        print(f"# {r.kind} {r.label or '?'}: [{coords}]")
    if args.save:
        answer = Answer(
            prompt=args.prompt, sent_prompt=args.prompt,
            sent=frame if frame is not None else np.zeros((8, 8, 3), np.uint8),
            text=text, regions=regions, latency_s=latency, tokens=args.tokens,
            had_image=frame is not None,
        )
        annotated = answer.sent.copy()
        draw_regions(annotated, regions, args.coords)
        print(f"# saved {save_probe(Path(args.outdir), answer, annotated)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Interactive Qwen3-VL harness: ask questions about the camera feed "
            "and see the answer (and any coordinates it returns) drawn on the "
            "exact frame that was sent"
        ),
    )
    p.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    p.add_argument("--url", default=DEFAULT_URL, help="service base URL")
    p.add_argument("--tokens", type=int, default=256, help="max_new_tokens")
    p.add_argument(
        "--grid", action="store_true",
        help="draw a labelled pixel grid on the sent image (helps pointing)",
    )
    p.add_argument(
        "--coords", choices=("abs", "norm"), default="norm",
        help=(
            "read returned coords as 0-1000 normalized (norm, measured default "
            "for this build) or raw pixels (abs)"
        ),
    )
    p.add_argument(
        "--prompt", default="",
        help="ask one question, print the answer, exit (no window)",
    )
    p.add_argument(
        "--no-image", action="store_true",
        help="with --prompt: text-only, do not send a frame",
    )
    p.add_argument("--save", action="store_true", help="with --prompt: save the record")
    p.add_argument("--outdir", default="qwen_probes", help="where /save writes")
    p.add_argument("--no-preview", action="store_true", help="no window; terminal only")
    p.add_argument(
        "--park", action="store_true",
        help="home and retreat the arm to camera park first (needs calibration)",
    )
    args = p.parse_args(argv)

    try:
        info = health(args.url)
    except QwenError as exc:
        print(f"qwen unavailable: {exc}", file=sys.stderr)
        return 1
    svc = f"{info.get('model', '?')}  {info.get('device', '?')}/{info.get('quantization', '?')}"
    if not info.get("loaded", False):
        print("note: model still loading; the first answer will be slow", file=sys.stderr)

    if args.prompt:
        return run_once(args, svc)

    client = None
    if args.park:
        # Optional and best-effort: the harness is about the camera and the
        # model, but an arm parked over the desk is a frame problem that
        # looks exactly like a model problem.
        try:
            from mt4_jog.client import Mt4Client, Mt4ClientError
            from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration
            from mt4_vision.pickplace import ensure_homed, retreat_for_camera

            calib = load_calibration(DEFAULT_CALIB_PATH)
            client = Mt4Client()
            ensure_homed(client)
            retreat_for_camera(client, calib)
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001 -- arm is not required here
            print(f"warning: could not park the arm ({exc}); continuing", file=sys.stderr)
            if client is not None:
                client.close()
                client = None

    opts = Options(tokens=args.tokens, grid=args.grid, coord_mode=args.coords)
    stream: FrameStream | None = None
    view: HarnessPreview | None = None
    ui = BottomUI("ask")
    worker: QwenWorker | None = None
    try:
        # Held open even with --no-preview: see QwenWorker.grab.
        stream = FrameStream(args.camera)
        worker = QwenWorker(stream=stream, url=args.url, opts=opts, ui=ui)
        if not args.no_preview:
            view = HarnessPreview(stream, worker, svc=svc)

        ui.emit(f"qwen3-vl probe -- {svc}")
        ui.emit("/help for commands, /preset for capability probes")
        ui.set_status("ready")
        while True:
            if view is not None and view.stopped_by_user():
                break
            line = ui.read_line()
            if line is None:
                break
            if not line:
                worker.clear()
                ui.set_status("live")
                continue
            if handle_command(
                line, worker=worker, ui=ui, view=view, url=args.url,
                outdir=Path(args.outdir),
            ):
                continue
            worker.ask(line)
        # Clean exit (EOF or window closed): let a question already with the
        # model finish, so a piped batch reports every answer.
        worker.drain()
        return 0
    except KeyboardInterrupt:
        return 0
    except CameraError as exc:
        ui.close()
        print(f"camera: {exc}\n{camera_hint()}", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            worker.close()
        ui.close()
        if view is not None:
            view.close()
        if stream is not None:
            stream.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
