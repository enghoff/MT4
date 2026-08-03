"""What ``ask_qwen.py`` draws: the answer panel and the submitted frame.

The main pane is always *the exact image that was POSTed*, never the live feed.
A wrong answer about a frame the arm was blocking is a different problem from a
wrong answer about a clean frame, and no amount of reading the text tells you
which happened. :func:`compose` puts a live inset in the corner so both are on
screen at once.

:class:`Options` is the session's request settings, :class:`Answer` one reply
plus what the service really encoded -- ``Answer.sent_label`` reports how the
frames actually reached the model rather than inferring it from the picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from mt4_vision.preview import draw_inset, draw_outlined_text, wrap_text
from mt4_vision.qwen import Region, Reply


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


# How a multi-frame capture is presented to the model. These are not
# interchangeable -- measured against a moving-square stimulus, "video"
# detects that something moved at half the token cost but answers direction
# invariantly, while "images" gets direction right and flips its answer when
# the frames are reversed. "montage" tiles them into one picture and needs no
# multi-frame support at all, which is its only advantage. See docs/QWEN3-VL.md.
SEND_MODES = ("single", "montage", "images", "video")


# The service's own default is 256, which suits a sentence of prose but cuts
# the default watch question's JSON off after ~5 boxed objects (~35 tokens
# each, plus a description). Truncation there is not graceful: the array
# never closes, and only the entries that completed can be recovered.
DEFAULT_TOKENS = 700


@dataclass
class Options:
    tokens: int = DEFAULT_TOKENS
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
class Answer:
    prompt: str            # what the user typed
    sent_prompt: str       # what was actually POSTed (montage preamble etc.)
    sent: np.ndarray       # what to DISPLAY: the posted image, or frames tiled
    text: str = ""
    regions: list[Region] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None
    tokens: int = DEFAULT_TOKENS
    had_image: bool = True
    send_mode: str = "single"
    reply: Reply | None = None   # service metadata: groups, timestamps, tokens
    # Size of ONE frame as POSTed. Not sent.shape: for the multi-frame modes
    # sent is a display tiling at a fraction of the real size, and reporting
    # that as the frame size understates what the model actually got.
    frame_hw: tuple[int, int] | None = None

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


class WatchState(NamedTuple):
    """What ``MotionWatcher`` currently sees, for the panel to draw.

    A NamedTuple rather than a plain tuple because appending a field to a
    positionally-unpacked one silently broke the whole preview once: the panel
    unpacked 5 values, ``send`` became a 6th, every ``compose`` call raised
    ValueError, and ``HarnessPreview``'s draw-error guard swallowed it -- so the
    window simply never opened, with nothing on screen or in the log saying why.
    Attribute access cannot drift that way.
    """

    state: str
    score: float
    events: int
    skipped: int
    boxes: list[tuple[int, int, int, int]]
    send: str




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
        lines = wrap_text(text, max_px=body_px, scale=BODY_SCALE, hard_break=True)
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
    # Labelled "typed" because a watch event ignores this and sends its
    # before/after pair as 2 images -- the footer's "sent:" is what actually
    # went for the answer on screen, and the two disagreeing is normal.
    watching = watch is not None and watch.state != "off"
    label = "typed ask" if watching else "send"
    draw_outlined_text(panel, f"{label}: {send}", (x, y), scale=0.4, color=DIM)
    if watch is not None and watching:
        y += BODY_LINE_PX
        kind = "track" if watch.send == "latest" else "watch"
        line = f"{kind}: {watch.state}  {watch.score:.5f}  {watch.events} event(s)"
        if watch.skipped:
            line += f"  {watch.skipped} skipped"
        draw_outlined_text(
            panel, line, (x, y), scale=0.4,
            color=WARN_BGR if watch.state == "moving" else A_BGR,
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
            for line in wrap_text(
                "Type a question to ask about the current frame. "
                "/help lists the commands, /preset lists capability probes.",
                max_px=body_px, scale=BODY_SCALE, hard_break=True,
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
        # Change boxes are drawn per-frame before tiling, in _build_payload --
        # not here, where main may be a scaled composite and the boxes would
        # land in the wrong coordinate space.
        draw_inset(main, live, "LIVE", width_px=INSET_W)
    elif pending is not None:
        main = live.copy()
    else:
        main = live.copy()
        draw_outlined_text(main, "LIVE", (14, 28), scale=0.55, color=(220, 220, 220))

    # The size the MODEL was given, which for a tiled view is not the size of
    # what is on screen.
    shape = (
        answer.frame_hw if answer is not None and answer.frame_hw is not None
        else (live.shape[0], live.shape[1])
    )
    panel = render_panel(
        main.shape[0],
        answer=answer, pending=pending, elapsed=elapsed, opts=opts, svc=svc,
        frame_shape=shape, region_counts=counts, watch=watch,
    )
    return np.hstack([main, panel])
