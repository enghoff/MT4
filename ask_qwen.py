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
import os
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
from mt4_vision.qwen import (
    DEFAULT_URL,
    QwenError,
    Region,
    Reply,
    ask,
    generate,
    health,
    parse_regions,
)

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
  /frames N [gap]   capture N frames, gap seconds apart (N=1 for one still)
  /mode M           how they reach the model: single|montage|images|video
                      images = N separate images; direction right, ~2x tokens
                      video  = one sequence; change right, direction not, ~1x
                      montage= tiled into one picture; needs no server support
  /sample           toggle greedy (default) vs sampling at temperature 0.7
  /watch <question> ask it automatically whenever the desk moves and settles,
                      handing over the before/after pair. /watch off to stop
  /sens X           motion trigger threshold (default 0.0005; noise floor is
                      0.00016, a 25px object moving is ~0.0014)
  /noimage          ask the next question text-only (no frame)
  /preset [N]       list capability probes, or run one
  /save [name]      write the sent image, the view and a JSON record
  /health           re-query the service
  Ctrl+C / Ctrl+Z   quit"""


# How a multi-frame capture is presented to the model. These are not
# interchangeable -- measured against a moving-square stimulus, "video"
# detects that something moved at half the token cost but answers direction
# invariantly, while "images" gets direction right and flips its answer when
# the frames are reversed. "montage" tiles them into one picture and needs no
# multi-frame support at all, which is its only advantage. See docs/QWEN3-VL.md.
SEND_MODES = ("single", "montage", "images", "video")


@dataclass
class Options:
    tokens: int = 256
    grid: bool = False
    # This build answers in 0-1000 normalized coords, verified against the
    # desk camera -- see mt4_vision.qwen.parse_regions. /coords flips it.
    coord_mode: str = "norm"  # "norm" | "abs"
    send_mode: str = "single"
    frames: int = 1
    frame_gap_s: float = 0.5
    greedy: bool = True
    pinned: np.ndarray | None = None

    @property
    def fps(self) -> float:
        """Capture rate implied by the gap -- what video timestamps are built from."""
        return 1.0 / self.frame_gap_s if self.frame_gap_s > 0 else 1.0


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


@dataclass
class Answer:
    prompt: str            # what the user typed
    sent_prompt: str       # what was actually POSTed (montage preamble etc.)
    sent: np.ndarray       # what to DISPLAY: the posted image, or frames tiled
    text: str = ""
    regions: list[Region] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None
    tokens: int = 256
    had_image: bool = True
    send_mode: str = "single"
    reply: Reply | None = None   # service metadata: groups, timestamps, tokens

    def sent_label(self) -> str:
        """How the frames actually reached the model -- never guessed from the view."""
        if not self.had_image:
            return "text only"
        r = self.reply
        n = r.frames_sent if r is not None else 1
        if self.send_mode == "video" and r is not None:
            return f"video, {n} frames, ts {list(r.timestamps_s)}"
        if self.send_mode == "images" and n > 1:
            return f"{n} separate images"
        if self.send_mode == "montage":
            return "1 tiled image (montage)"
        return "1 image"


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
    watch: WatchState | None = None,
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
    if not opts.greedy:
        flags += "  SAMPLING"
    draw_outlined_text(panel, flags, (x, y), scale=0.4, color=DIM)
    y += BODY_LINE_PX
    send = opts.send_mode
    if send != "single":
        send += f" x{opts.frames} @{opts.frame_gap_s:g}s"
    draw_outlined_text(panel, f"send: {send}", (x, y), scale=0.4, color=DIM)
    if watch is not None and watch[0] != "off":
        y += BODY_LINE_PX
        state, score, events, skipped, _boxes = watch
        line = f"watch: {state}  {score:.5f}  {events} event(s)"
        if skipped:
            line += f"  {skipped} skipped"
        draw_outlined_text(
            panel, line, (x, y), scale=0.4,
            color=WARN_BGR if state == "moving" else A_BGR,
        )
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

    footer = height - 58
    cv2.line(panel, (x, footer), (PANEL_W - PANEL_PAD, footer), RULE, 1)
    # What actually reached the model, never inferred from the picture shown.
    draw_outlined_text(
        panel, f"sent: {answer.sent_label()}"[: 58], (x, footer + 18),
        scale=0.4, color=DIM,
    )
    warning = answer.reply.frame_warning() if answer.reply is not None else None
    if warning:
        draw_outlined_text(
            panel, "! frames were dropped", (x, footer + 52), scale=0.42, color=ERR_BGR,
        )
    footer += 16
    stats = f"{answer.latency_s:.1f}s"
    if answer.reply is not None and answer.reply.prompt_tokens:
        stats += f"  {answer.reply.prompt_tokens} tok"
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


WatchState = tuple[str, float, int, int, list]


def compose(
    live: np.ndarray,
    *,
    answer: Answer | None,
    pending: str | None,
    elapsed: float,
    opts: Options,
    svc: str,
    watch: WatchState | None = None,
) -> np.ndarray:
    """Main pane (submitted frame, or live when nothing is being shown) + panel."""
    counts: tuple[int, int] | None = None
    if answer is not None:
        main = answer.sent.copy()
        if answer.regions:
            counts = draw_regions(main, answer.regions, opts.coord_mode)
        # For images/video the view is a tiling of several posted frames, so
        # say so -- otherwise it reads as a single montage image having been
        # sent, which is a different thing to the model.
        if answer.send_mode in ("images", "video"):
            draw_outlined_text(
                main, f"VIEW: tiled for display - sent as {answer.send_mode}",
                (14, 28), scale=0.55, color=(0, 200, 255),
            )
        # Where the trigger saw change, so a human can tell whether it fired on
        # the object of interest or on a shadow.
        for bx in (watch[4] if watch else []):
            cv2.rectangle(main, (bx[0], bx[1]), (bx[2], bx[3]), (0, 200, 255), 1)
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
        watch=watch,
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
        quiet: bool = False,
    ) -> None:
        job = Job(prompt=prompt, with_image=with_image, frames=frames,
                  mode=mode, note=note)
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

        display = posted[0] if len(posted) == 1 else montage(posted)
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
            img = np.full((360, 640, 3), 40, dtype=np.uint8)
            draw_outlined_text(
                img, "text-only prompt (no image sent)", (24, 190),
                scale=0.7, color=(200, 200, 200),
            )
            try:
                posted: list[np.ndarray] = []
                if with_image:
                    posted, img, sent_prompt, send_mode = self._build_payload(
                        prompt, opts, frames=job.frames,
                    )
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
                for line in (text or "(empty response)").split("\n"):
                    self._ui.emit(f"  {line}")
                note = f"{answer.latency_s:.1f}s"
                if reply is not None and reply.prompt_tokens:
                    note += f", {reply.prompt_tokens} prompt tok"
                if answer.regions:
                    note += f", {len(answer.regions)} region(s)"
                self._ui.set_status(f"answered in {note}")


# Fraction of pixels that must change (after blur, above MOTION_DELTA) to count
# as motion. Measured noise floor on the desk camera over 60 static frames:
# mean 0.00008, max 0.00016. A 25x25px object moving produces ~0.0014, so this
# sits ~3x above the worst observed noise and ~3x below the smallest real event.
MOTION_THRESHOLD = 0.0005
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
        self._state = "off"
        self._score = 0.0
        self._events = 0
        self._skipped = 0
        self._boxes: list[tuple[int, int, int, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="motion", daemon=True)
        self._thread.start()

    def arm(self, prompt: str) -> None:
        with self._lock:
            self._prompt = prompt
            self._state = "quiet"
            self._events = self._skipped = 0

    def disarm(self) -> None:
        with self._lock:
            self._prompt = None
            self._state = "off"

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._prompt is not None

    def snapshot(self) -> tuple[str, float, int, int, list[tuple[int, int, int, int]]]:
        with self._lock:
            return self._state, self._score, self._events, self._skipped, list(self._boxes)

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
        self._ui.emit("")
        self._ui.emit(f"  [motion #{n}, score {peak:.5f}] {len(boxes)} changed region(s)")
        self._worker.ask(
            prompt,
            frames=[before, after],
            mode="images",   # video makes a coordinate's frame ambiguous
            note=f"motion {peak:.5f}",
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


class HarnessPreview:
    """Composites the live feed, the submitted frame and the answer panel."""

    def __init__(
        self, stream: FrameStream, worker: QwenWorker, *, svc: str,
        watcher: MotionWatcher | None = None,
    ) -> None:
        self._stream = stream
        self._worker = worker
        self._svc = svc
        self._watcher = watcher
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
            watch = self._watcher.snapshot() if self._watcher is not None else None
            try:
                canvas = compose(
                    live, answer=answer, pending=pending, elapsed=elapsed,
                    opts=self._worker.opts, svc=self._svc, watch=watch,
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
        "send_mode": answer.send_mode,
        "sent_label": answer.sent_label(),
        "service": answer.reply.raw if answer.reply is not None else None,
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
    url: str, outdir: Path, watcher: MotionWatcher | None = None,
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
        if opts.frames > 1:
            ui.set_status("/frames is >1 -- /frames 1 first (a pinned frame is not a sequence)")
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
    elif cmd == "frames":
        try:
            want = max(1, min(16, int(rest[0])))
            gap = max(0.05, float(rest[1])) if len(rest) > 1 else opts.frame_gap_s
        except (IndexError, ValueError):
            ui.set_status("usage: /frames N [seconds-between]")
            return True
        # A frozen frame would be repeated and answered as if it were a
        # sequence -- a wrong answer with no visible cause.
        if want > 1 and opts.pinned is not None:
            ui.set_status("frame is frozen -- /thaw first, a sequence needs live frames")
            return True
        opts.frames, opts.frame_gap_s = want, gap
        if want > 1 and opts.send_mode == "single":
            opts.send_mode = "images"
        ui.set_status(
            f"{want} frames {gap:g}s apart, sent as {opts.send_mode} "
            f"({opts.fps:g} fps)" if want > 1 else "single frame"
        )
    elif cmd == "mode":
        want = (rest[0].lower() if rest else "")
        match = [m for m in SEND_MODES if m.startswith(want)] if want else []
        if len(match) != 1:
            ui.set_status(f"usage: /mode {'|'.join(SEND_MODES)}")
            return True
        opts.send_mode = match[0]
        if opts.send_mode != "single" and opts.frames < 2:
            opts.frames = 4
        ui.set_status(
            f"mode {opts.send_mode}"
            + (f", {opts.frames} frames {opts.frame_gap_s:g}s apart"
               if opts.send_mode != "single" else "")
        )
    elif cmd == "watch":
        if watcher is None:
            ui.set_status("no watcher (needs the camera stream)")
        elif not rest or rest[0].lower() in ("off", "stop", "no"):
            was = watcher.armed
            watcher.disarm()
            _st, _sc, events, skipped, _b = watcher.snapshot()
            ui.set_status(
                f"watch off after {events} event(s)"
                + (f", {skipped} skipped while busy" if skipped else "")
                if was else "watch was not on"
            )
        elif opts.pinned is not None:
            ui.set_status("frame is frozen -- /thaw first, the watcher needs live frames")
        else:
            question = " ".join(rest)
            watcher.arm(question)
            ui.emit(f"  watching: {question}")
            ui.set_status(
                f"armed -- will ask on motion above {watcher.threshold:.5f} (/watch off)"
            )
    elif cmd == "sens":
        if watcher is None:
            ui.set_status("no watcher (needs the camera stream)")
            return True
        try:
            watcher.threshold = max(0.0, float(rest[0]))
        except (IndexError, ValueError):
            _st, score, _e, _s, _b = watcher.snapshot()
            ui.set_status(
                f"usage: /sens X (now {watcher.threshold:.5f}, live score {score:.5f})"
            )
        else:
            ui.set_status(f"motion threshold {watcher.threshold:.5f}")
    elif cmd == "sample":
        opts.greedy = not opts.greedy
        ui.set_status(
            "greedy (reproducible)" if opts.greedy
            else "sampling at the model's temperature 0.7 (answers will vary)"
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
    frames: list[np.ndarray] = []
    send_mode = "single"
    stream: FrameStream | None = None
    if not args.no_image:
        want = 1 if args.mode == "single" else max(1, args.frames)
        try:
            if want == 1:
                frames = [capture_frame(args.camera)]
            else:
                # One held stream, never repeated opens -- see QwenWorker.grab.
                stream = FrameStream(args.camera)
                frames = [stream.fresh(min_advance=1)]
                for _ in range(want - 1):
                    time.sleep(args.gap)
                    frames.append(stream.fresh(min_advance=1))
        except CameraError as exc:
            print(f"camera: {exc}\n{camera_hint()}", file=sys.stderr)
            return 1
        finally:
            if stream is not None:
                stream.close()
        send_mode = args.mode if len(frames) > 1 else "single"
        if send_mode == "montage":
            frames = [montage(frames)]
        if args.grid:
            frames = [annotate_for_pointing(f) for f in frames]

    started = time.monotonic()
    try:
        reply = ask(
            args.prompt, frames or None,
            mode="video" if send_mode == "video" else "images",
            fps=1.0 / args.gap if args.gap > 0 else 1.0,
            url=args.url, max_new_tokens=args.tokens,
            do_sample=args.sample,
        )
    except QwenError as exc:
        print(f"qwen: {exc}", file=sys.stderr)
        return 1
    latency = time.monotonic() - started
    regions = parse_regions(reply.text)

    print(f"# {svc}  ({latency:.1f}s, {reply.mode}, {reply.prompt_tokens} prompt tok)")
    warning = reply.frame_warning()
    if warning:
        print(f"# WARNING: {warning}", file=sys.stderr)
    print(reply.text)
    for r in regions:
        coords = ", ".join(f"{c:g}" for c in r.coords)
        print(f"# {r.kind} {r.label or '?'}: [{coords}]")
    if args.save:
        display = frames[0] if len(frames) == 1 else (
            montage(frames) if frames else np.zeros((8, 8, 3), np.uint8)
        )
        answer = Answer(
            prompt=args.prompt, sent_prompt=args.prompt, sent=display,
            text=reply.text, regions=regions, latency_s=latency, tokens=args.tokens,
            had_image=bool(frames), send_mode=send_mode, reply=reply,
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
        "--mode", choices=SEND_MODES, default="single",
        help=(
            "how multiple frames reach the model: images (direction right, ~2x "
            "tokens), video (change right, direction not, ~1x), montage (tiled "
            "into one picture)"
        ),
    )
    p.add_argument(
        "--watch", default="",
        help=(
            "arm the motion watcher with this question: it is asked, with the "
            "before/after pair, whenever the desk moves and settles"
        ),
    )
    p.add_argument(
        "--sens", type=float, default=MOTION_THRESHOLD,
        help=f"motion trigger threshold (default {MOTION_THRESHOLD})",
    )
    p.add_argument("--frames", type=int, default=1, help="frames to capture")
    p.add_argument("--gap", type=float, default=0.5, help="seconds between frames")
    p.add_argument(
        "--sample", action="store_true",
        help="sample at the model's temperature 0.7 instead of greedy (default greedy)",
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

    opts = Options(
        tokens=args.tokens, grid=args.grid, coord_mode=args.coords,
        send_mode=args.mode, frames=max(1, args.frames), frame_gap_s=max(0.05, args.gap),
        greedy=not args.sample,
    )
    stream: FrameStream | None = None
    view: HarnessPreview | None = None
    ui = BottomUI("ask")
    worker: QwenWorker | None = None
    watcher: MotionWatcher | None = None
    try:
        # Held open even with --no-preview: see QwenWorker.grab.
        stream = FrameStream(args.camera)
        worker = QwenWorker(stream=stream, url=args.url, opts=opts, ui=ui)
        watcher = MotionWatcher(stream, worker, ui, threshold=args.sens)
        if not args.no_preview:
            view = HarnessPreview(stream, worker, svc=svc, watcher=watcher)
        if args.watch:
            watcher.arm(args.watch)
            ui.emit(f"  watching: {args.watch}")

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
                outdir=Path(args.outdir), watcher=watcher,
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
        if watcher is not None:
            watcher.close()
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
    code = main()
    # Hard exit after main()'s cleanup has already released the camera and the
    # window. OpenCV highgui can keep a Windows process alive past the end of
    # main, and a lingering process holds the capture device open -- which
    # strands the camera for every other vision script in the repo until it is
    # killed by hand. Observed once; not worth risking again for a CLI whose
    # work is finished by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
