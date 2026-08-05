"""Annotated live preview + video recording for pick/place scripts.

Draws the same cube/marker overlay as ``python -m mt4_vision scene`` on top
of every frame, so a human (or a saved recording) can watch the desk while
the planner works without touching the pick/place logic itself.
"""

from __future__ import annotations

import math
import sys
import threading
import time

import cv2
import numpy as np

from mt4_vision.calib import Calibration
from mt4_vision.camera import CameraError, FrameStream
from mt4_vision.detect import CubeDetection, MarkerDetection, detect_markers
from mt4_vision.scene import Scene, capture_scene
from mt4_vision.workspace import MARKER_DICT, dist_mm

# BGR overlay colors.
CUBE_BGR = (255, 255, 255)
PHANTOM_BGR = (90, 90, 90)
MARKER_FREE_BGR = (0, 200, 0)
MARKER_OCCUPIED_BGR = (0, 0, 255)
MARKER_UNKNOWN_BGR = (0, 200, 255)
TARGET_BGR = (255, 0, 255)
# move_to (a single target point) vs move_path/mq (a queued multi-waypoint
# path) -- distinct colors so a queued path is visible at a glance, decided
# by the command that was sent, NOT by how many legs it happens to bend into
# (a direct 1-waypoint move_path is still an mq, so still cyan).
TRAJECTORY_SINGLE_BGR = (0, 140, 255)  # orange -- move_to
TRAJECTORY_MULTI_BGR = (255, 255, 0)  # cyan -- move_path / mq


_OUTLINE_OFFSETS = [
    (-2, 0), (2, 0), (0, -2), (0, 2),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def draw_outlined_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    outline_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text with a solid outline that can't drift from the fill.

    cv2.putText's Hershey font spaces characters differently per
    ``thickness`` argument (confirmed via cv2.getTextSize: the same string
    measures ~10px wider at thickness 3 than at thickness 1 for a ~200px
    line) -- a thick outline pass and a thin fill pass at the same origin
    silently drift apart character by character, worse the further into the
    string, rather than actually overlapping. Keeping every pass at
    thickness 1 and faking the stroke with small offset copies avoids that
    entirely.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dx, dy in _OUTLINE_OFFSETS:
        cv2.putText(
            img, text, (org[0] + dx, org[1] + dy), font, scale,
            outline_color, 1, cv2.LINE_AA,
        )
    cv2.putText(img, text, org, font, scale, color, 1, cv2.LINE_AA)


def draw_cube_marker(
    img: np.ndarray,
    px: float,
    py: float,
    color_bgr: tuple[int, int, int],
    label: str,
) -> None:
    """Circle + outlined label at a cube's pixel position.

    The one shared per-cube-blob style every preview in this repo draws,
    whether the color encodes "this cube's actual color" (track_cube.py,
    verifying color detection) or "pick-quality vs phantom" (annotate_scene,
    verifying planner state).
    """
    cv2.circle(img, (int(px), int(py)), 8, color_bgr, 2)
    draw_outlined_text(img, label, (int(px) + 10, int(py)), scale=0.6, color=color_bgr)


def draw_lock_ring(img: np.ndarray, px: float, py: float, color_bgr: tuple[int, int, int]) -> None:
    """Circle + cross marking whichever cube is currently the focus.

    A pick/clear target in annotate_scene, a tracking lock in track_cube.py.
    """
    cx, cy = int(px), int(py)
    cv2.circle(img, (cx, cy), 16, color_bgr, 2)
    cv2.drawMarker(img, (cx, cy), color_bgr, cv2.MARKER_CROSS, 16, 2)


# cv2.arrowedLine's tipLength is a fraction of that leg's OWN length, applied
# to the shortest leg to get an absolute arrowhead pixel size, then reused
# (as a length-scaled fraction) on every other leg -- otherwise a long
# diagonal leg and a short leg in the same path get visibly different-sized
# arrowheads under one shared fraction.
_ARROWHEAD_TIP_FRACTION = 0.075
_ARROWHEAD_UNIFORM_SCALE = 0.8


def draw_trajectory(
    img: np.ndarray,
    points_px: list[tuple[float, float]],
    multi: bool = False,
) -> None:
    """Draw an in-flight move_to/move_path request: arrowed legs between
    waypoints, a dot at each intermediate one, and a cross at the end (the
    single-point move_to case is just that end cross, no legs). Color follows
    the command, not the leg count: ``multi`` False (move_to) draws orange,
    True (any move_path/mq) draws cyan even when it bends into a single leg.
    Every arrowhead is the same absolute size regardless of leg length (see
    _ARROWHEAD_TIP_FRACTION).
    """
    if not points_px:
        return
    pts = [(int(px), int(py)) for px, py in points_px]
    color_bgr = TRAJECTORY_MULTI_BGR if multi else TRAJECTORY_SINGLE_BGR
    legs = list(zip(pts, pts[1:]))
    leg_lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in legs]
    positive_lengths = [length for length in leg_lengths if length > 1e-6]
    tip_px = (
        _ARROWHEAD_UNIFORM_SCALE * _ARROWHEAD_TIP_FRACTION * min(positive_lengths)
        if positive_lengths
        else 0.0
    )
    for (a, b), length in zip(legs, leg_lengths):
        if length <= 1e-6:
            continue
        cv2.arrowedLine(
            img, a, b, color_bgr, 2, cv2.LINE_AA, tipLength=min(tip_px / length, 1.0),
        )
    for p in pts[:-1]:
        cv2.circle(img, p, 4, color_bgr, -1)
    cv2.drawMarker(img, pts[-1], color_bgr, cv2.MARKER_TILTED_CROSS, 14, 2)


def annotate_scene(
    frame: np.ndarray,
    scene: Scene,
    markers_px: list[MarkerDetection],
    *,
    target: CubeDetection | None = None,
    status_lines: list[str] | None = None,
    trajectory_px: list[tuple[float, float]] | None = None,
    trajectory_multi: bool = False,
) -> np.ndarray:
    """Return a copy of ``frame`` with cubes, markers, and a status header."""
    out = frame.copy()
    pick_ids = {id(c) for c in scene.cubes}
    for c in scene.raw_cubes if scene.raw_cubes is not None else scene.cubes:
        is_pick = id(c) in pick_ids
        color = CUBE_BGR if is_pick else PHANTOM_BGR
        label = c.color if is_pick else f"{c.color}?"
        draw_cube_marker(out, c.px, c.py, color, label)

    occupied_ids = {m.marker_id for m, _c in scene.occupied}
    free_ids = {m.marker_id for m in scene.free_markers}
    for m in markers_px:
        if m.marker_id in occupied_ids:
            color = MARKER_OCCUPIED_BGR
        elif m.marker_id in free_ids:
            color = MARKER_FREE_BGR
        else:
            color = MARKER_UNKNOWN_BGR
        cv2.drawMarker(out, (int(m.px), int(m.py)), color, cv2.MARKER_CROSS, 18, 2)
        draw_outlined_text(
            out, str(m.marker_id), (int(m.px) + 10, int(m.py) - 10), scale=0.6, color=color,
        )

    if trajectory_px:
        draw_trajectory(out, trajectory_px, trajectory_multi)

    if target is not None:
        draw_lock_ring(out, target.px, target.py, TARGET_BGR)

    for i, line in enumerate(status_lines or []):
        y = 24 + i * 22
        draw_outlined_text(out, line, (10, y), scale=0.6, color=(255, 255, 255))

    return out


class VideoRecorder:
    """Appends annotated frames to a video file at a fixed rate."""

    def __init__(self, *, video_path: str, fps: float = 10.0) -> None:
        self._video_path = video_path
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None

    def _open_writer(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(
            *("mp4v" if str(self._video_path).lower().endswith(".mp4") else "MJPG")
        )
        writer = cv2.VideoWriter(str(self._video_path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self._video_path}")
        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._open_writer(frame)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class PreviewStopped(Exception):
    """Raised when the user closes a ``LivePreview`` window (q or Esc)."""


_gui_available: bool | None = None
_warned_no_gui = False


def gui_available() -> bool:
    """Whether a caller should open a preview window at all.

    Public form of the ``imshow`` probe, for callers that show a preview *by
    default* rather than on request. ``LivePreview`` falls back to
    ``PIL.Image.show`` when OpenCV is headless, which is right for someone who
    explicitly asked for a window and wrong for a default -- it spawns a fresh
    system image viewer on every single update.
    """
    return _opencv_gui_available()


def _opencv_gui_available() -> bool:
    """True when ``cv2.imshow`` works (false for opencv-python-headless)."""
    global _gui_available
    if _gui_available is not None:
        return _gui_available
    try:
        blank = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imshow("__mt4_gui_probe__", blank)
        cv2.waitKey(1)
        cv2.destroyWindow("__mt4_gui_probe__")
        _gui_available = True
    except cv2.error:
        _gui_available = False
    return _gui_available


class LivePreview:
    """Pop-up window showing whatever frame it's handed.

    OpenCV highgui only stays responsive while something calls ``waitKey``.
    Callers often block for seconds on arm motion or ``input()``, which makes
    Windows mark the window "(Not Responding)". So all ``imshow`` / ``waitKey``
    work runs on a dedicated daemon thread; ``show`` just publishes the latest
    frame. q/Esc on that thread raises ``PreviewStopped`` on the next ``show``.

    When OpenCV was installed headless -- common if ``opencv-python-headless``
    shadows ``opencv-python`` -- falls back to ``PIL.Image.show``.
    """

    def __init__(self, window_name: str = "mt4 preview (q or Esc to stop)") -> None:
        self._window = window_name
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._user_quit = threading.Event()
        self._thread: threading.Thread | None = None

    def show(self, frame: np.ndarray) -> None:
        """Publish one frame; raise ``PreviewStopped`` if the user hit q/Esc."""
        global _warned_no_gui
        if self._user_quit.is_set():
            raise PreviewStopped()

        if not _opencv_gui_available():
            if not _warned_no_gui:
                _warned_no_gui = True
                print(
                    "WARNING: OpenCV has no GUI (is opencv-python-headless installed?). "
                    "Preview opens via the system image viewer instead.",
                    file=sys.stderr,
                )
            from PIL import Image

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).show(title=self._window)
            return

        with self._lock:
            self._frame = frame.copy()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._pump, name="mt4-preview", daemon=True,
            )
            self._thread.start()
            # Let the first paint land before the caller blocks on motion.
            time.sleep(0.05)
        if self._user_quit.is_set():
            raise PreviewStopped()

    def _pump(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is None:
                time.sleep(0.03)
                continue
            cv2.imshow(self._window, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                self._user_quit.set()
                break
        try:
            cv2.destroyWindow(self._window)
        except cv2.error:
            pass

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None
        if _opencv_gui_available():
            try:
                cv2.destroyWindow(self._window)
            except cv2.error:
                pass


# Gate for matching the cube a caller is currently acting on (``set_target``)
# against this tick's fresh detections -- generous enough to survive a frame
# or two of drift, tight enough not to ring the wrong same-color cube.
TARGET_MATCH_GATE_MM = 60.0


def _nearest_same_color(
    cubes: list[CubeDetection], color: str, x: float, y: float,
) -> CubeDetection | None:
    best: CubeDetection | None = None
    best_d = TARGET_MATCH_GATE_MM
    for c in cubes:
        if c.color != color or c.x is None or c.y is None:
            continue
        d = dist_mm(float(c.x), float(c.y), x, y)
        if d <= best_d:
            best = c
            best_d = d
    return best


class LiveFeed:
    """Continuously annotated preview/recording, decoupled from arm motion.

    Runs its own capture/detect/draw loop on a background thread against a
    caller-owned ``FrameStream``, so the feed keeps updating at ``fps`` while
    the main thread blocks on multi-second arm moves (pick/place). The arm is
    not retreated for this feed -- unlike the discrete captures a planner
    acts on, this is a monitor and shows whatever the camera currently sees,
    arm included.

    The caller opens and closes the ``FrameStream`` itself (typically shared
    with the planner's own "look now" captures, since only one consumer can
    hold the camera device); ``close()`` here only stops this feed's thread
    and output sinks, it never touches the stream.
    """

    def __init__(
        self,
        *,
        calib: Calibration,
        stream: FrameStream,
        fps: float = 10.0,
        video_path: str | None = None,
        show_preview: bool = False,
    ) -> None:
        self._calib = calib
        self._stream = stream
        self._period = 1.0 / fps if fps > 0 else 0.0
        self._recorder = (
            VideoRecorder(video_path=video_path, fps=fps) if video_path else None
        )
        self._live_preview = LivePreview() if show_preview else None
        self._status_lines: list[str] = []
        self._target: tuple[str, float, float] | None = None
        self._trajectory: list[tuple[float, float, float]] | None = None
        self._trajectory_multi = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.stopped_by_user = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_status(self, lines: list[str]) -> None:
        with self._lock:
            self._status_lines = list(lines)

    def set_target(self, color: str, x: float, y: float) -> None:
        with self._lock:
            self._target = (color, x, y)

    def clear_target(self) -> None:
        with self._lock:
            self._target = None

    def set_trajectory(
        self, waypoints: list[tuple[float, float, float]], multi: bool = False,
    ) -> None:
        """Show a move_to/move_path request as an overlay -- robot-frame
        (x, y, z) waypoints, projected to pixels every tick via
        Calibration.robot_to_pixel (height-corrected, not flattened to the
        table plane) (Mt4Client sets this when it sends the move, via its
        ``trajectory_sink`` hook, and clears it once the call returns,
        success or failure). ``multi`` colors the overlay by command:
        False for move_to, True for any move_path/mq (see draw_trajectory).
        """
        with self._lock:
            self._trajectory = [(float(x), float(y), float(z)) for x, y, z in waypoints]
            self._trajectory_multi = multi

    def clear_trajectory(self) -> None:
        with self._lock:
            self._trajectory = None
            self._trajectory_multi = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            tick_start = time.monotonic()
            try:
                frame = self._stream.fresh(min_advance=1)
            except CameraError:
                continue
            scene = capture_scene(self._calib, frame)
            markers_px = detect_markers(frame, MARKER_DICT)
            with self._lock:
                status_lines = list(self._status_lines)
                target_spec = self._target
                trajectory = self._trajectory
                trajectory_multi = self._trajectory_multi
            target = None
            if target_spec is not None:
                color, tx, ty = target_spec
                target = _nearest_same_color(
                    scene.raw_cubes if scene.raw_cubes is not None else scene.cubes,
                    color, tx, ty,
                )
            trajectory_px = None
            if trajectory:
                trajectory_px = [self._calib.robot_to_pixel(x, y, z) for x, y, z in trajectory]
            annotated = annotate_scene(
                frame, scene, markers_px, target=target, status_lines=status_lines,
                trajectory_px=trajectory_px, trajectory_multi=trajectory_multi,
            )
            if self._recorder is not None:
                self._recorder.write(annotated)
            if self._live_preview is not None:
                try:
                    self._live_preview.show(annotated)
                except PreviewStopped:
                    self.stopped_by_user.set()
                    break
            elapsed = time.monotonic() - tick_start
            remaining = self._period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._recorder is not None:
            self._recorder.close()
        if self._live_preview is not None:
            self._live_preview.close()


# Pointing overlay palette: a grid that reads clearly without competing with
# the desk, and entity ids in the same white as CUBE_BGR.
GRID_BGR = (90, 200, 255)
ENTITY_BGR = (255, 255, 255)
ENTITY_BLOCKED_BGR = (120, 120, 120)
POINTING_GRID_PX = 100


# Spacing of the normalized grid, in units of the 0-1000 scale.
POINTING_GRID_NORM = 100


def annotate_for_pointing(
    frame: np.ndarray,
    entities: list | None = None,
    *,
    grid_px: int = POINTING_GRID_PX,
    grid: str = "pixel",
) -> np.ndarray:
    """Frame with any known entity ids circled, over a numbered grid.

    Built for handing a frame to a viewer that has to name a position. Circling
    the ids is what ties a printed ArUco number, which cannot be read off an
    image, to the tag the viewer can see.

    ``grid`` picks which space the gridlines are labelled in, and the two
    readers of this frame do not want the same one:

    * ``"pixel"`` -- lines every ``grid_px`` pixels, labelled with the pixel
      column/row. ``mt4_camera_view``'s documented workflow is to read a
      coordinate off these and pass it to ``mt4_locate_at_pixel``.
    * ``"norm"`` -- lines every ``POINTING_GRID_NORM`` units of the 0-1000
      scale, labelled with that unit. This is what Qwen gets: it answers 0-1000
      per axis whatever the prompt asks, which is the documented convention for
      the model family, so a grid labelled in pixels names a space its reply is
      not in.
    * ``"none"`` -- circles only.

    **"norm" is chosen for coherence, not for a measured win.** Across five
    distinct cube layouts on the live desk, 9 action choices and 2 box targets
    per layout per overlay, the three are indistinguishable:

    ====================  ==================  ==================  ============
    overlay               correct action      mean box error      wrong cube
    ====================  ==================  ==================  ============
    pixel grid            34/45               6.6 / 11.6 px       0
    no grid               36/45               4.0 / 11.0 px       0
    0-1000 grid           34/45               5.0 / 11.7 px       0
    ====================  ==================  ==================  ============

    Nothing there separates them, so the tie is broken on the one argument that
    does not depend on a sample size: a grid labelled 100-1200 tells the model
    the frame is 1280 wide while the prompt tells it the right edge is 1000, and
    only one of those can be the space the reply is read in.

    Do not read a performance claim into this either way. An earlier round that
    photographed one static layout three times appeared to show large effects in
    both directions; three captures of the same desk are one trial repeated, and
    none of it survived five real layouts.

    ``entities`` is any sequence of objects with ``id``, ``pixel`` and
    ``pickable`` (i.e. ``mt4_vision.entities.Entity``); entities without a pixel
    are skipped. Kept duck-typed so preview does not depend on the entity layer.

    Markers and slots carry no pixel of their own;
    ``entities._desk_pixel_projector`` fills one in, so the overlay circles
    every id the prompt promises rather than showing cubes and nothing to put
    them on. See that projector's docstring for what it costs.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Grid first, so labels and ids sit on top of it.
    if grid == "pixel":
        for x in range(grid_px, w, grid_px):
            cv2.line(out, (x, 0), (x, h), GRID_BGR, 1)
            draw_outlined_text(out, str(x), (x + 3, 14), scale=0.4, color=GRID_BGR)
        for y in range(grid_px, h, grid_px):
            cv2.line(out, (0, y), (w, y), GRID_BGR, 1)
            draw_outlined_text(out, str(y), (3, y - 3), scale=0.4, color=GRID_BGR)
        draw_outlined_text(
            out, f"{w}x{h}  grid {grid_px}px", (3, h - 8), scale=0.45, color=GRID_BGR
        )
    elif grid == "norm":
        for k in range(POINTING_GRID_NORM, 1000, POINTING_GRID_NORM):
            x = int(k * w / 1000)
            cv2.line(out, (x, 0), (x, h), GRID_BGR, 1)
            draw_outlined_text(out, str(k), (x + 3, 14), scale=0.4, color=GRID_BGR)
            y = int(k * h / 1000)
            cv2.line(out, (0, y), (w, y), GRID_BGR, 1)
            draw_outlined_text(out, str(k), (3, y - 3), scale=0.4, color=GRID_BGR)
        draw_outlined_text(
            out, f"grid = 0-1000 scale, {POINTING_GRID_NORM} units",
            (3, h - 8), scale=0.45, color=GRID_BGR,
        )
    elif grid != "none":
        raise ValueError(f"grid must be 'pixel', 'norm' or 'none', not {grid!r}")

    for ent in entities or []:
        pixel = getattr(ent, "pixel", None)
        if pixel is None:
            continue
        color = ENTITY_BGR if getattr(ent, "pickable", False) else ENTITY_BLOCKED_BGR
        px, py = int(pixel[0]), int(pixel[1])
        cv2.circle(out, (px, py), 9, color, 2)
        draw_outlined_text(
            out, str(getattr(ent, "id", "?")), (px + 12, py + 4), scale=0.5, color=color
        )
    return out


# ------------------------------------------------- what the model answered

# Drawn on the exact frame the model looked at, never a fresh capture. Every
# question worth asking about a reply -- did it point at the thing it named, is
# that box the whole image, did it answer in the other coordinate space -- is a
# question about *that* frame, and a re-capture silently changes the subject.
#
# The palette separates the three things that are easy to conflate by eye:
QWEN_BOX_BGR = (0, 255, 255)      # yellow -- the reply, read the normal way
QWEN_ALT_BGR = (0, 140, 255)      # orange -- the same numbers, other space
QWEN_POINT_BGR = (255, 0, 255)    # magenta -- the point the decision carried
QWEN_BOUND_BGR = (0, 220, 0)      # green  -- what the stack resolved it to
QWEN_REFUSED_BGR = (0, 0, 255)    # red    -- resolved, then refused
QWEN_MASK_BGR = (128, 255, 0)     # spring green -- the segmented silhouette
# The mask is a fill, not an outline, because the questions about it are about
# area: did it take the whole object, did it leak into the shadow, is it only
# the one bright part. Light enough to read the desk through.
MASK_FILL_ALPHA = 0.4
# A mask filling this much of its own search window did not find an edge -- it
# took everything it was offered. Called out in the overlay for the same reason
# BIG_BOX_SHARE is: the resulting millimetres look perfectly ordinary.
WHOLE_WINDOW_SHARE = 0.9
# How dark the caption band goes. High enough to read white text over a lit
# desk, low enough that the pixels underneath are still legible.
PANEL_ALPHA = 0.6
# Bottom strip the caption stays out of, so it does not sit on the pointing
# overlay's own size/grid footer or run its descenders off the frame.
CAPTION_BOTTOM_RESERVE_PX = 20
# A box at least this share of the frame gets called out in the overlay: at
# that size the model has declined to answer, and a whole-frame box is
# otherwise indistinguishable from a real one. instruct.MAX_BOX_FRAME_SHARE
# reads this rather than restating it -- the picture and the refusal have to
# agree, and a comment asserting two numbers are equal does not make them so.
BIG_BOX_SHARE = 0.55


def draw_dashed_rect(
    img: np.ndarray,
    x0: float, y0: float, x1: float, y1: float,
    color: tuple[int, int, int],
    *,
    dash: int = 10,
    thickness: int = 1,
) -> None:
    """Rectangle in dashes, for a reading that is shown but not believed."""
    ix0, iy0, ix1, iy1 = int(x0), int(y0), int(x1), int(y1)
    for x in range(ix0, ix1, dash * 2):
        cv2.line(img, (x, iy0), (min(x + dash, ix1), iy0), color, thickness)
        cv2.line(img, (x, iy1), (min(x + dash, ix1), iy1), color, thickness)
    for y in range(iy0, iy1, dash * 2):
        cv2.line(img, (ix0, y), (ix0, min(y + dash, iy1)), color, thickness)
        cv2.line(img, (ix1, y), (ix1, min(y + dash, iy1)), color, thickness)


def draw_mask(
    img: np.ndarray,
    mask: np.ndarray | None,
    origin_px: tuple[float, float],
    colour: tuple[int, int, int],
    *,
    alpha: float = MASK_FILL_ALPHA,
) -> float:
    """Tint and outline a segmentation mask. Returns its fill share, 0.0 if none.

    ``mask`` is the sub-image ``locate`` cut out of the frame, not a full-frame
    mask, so ``origin_px`` says where its top-left corner sits. Clipped rather
    than assumed to fit: a detector box near the edge of the frame gets padded
    past it, and the mask window goes with it.

    The returned share is the mask's own area over its window's area. That
    number is a check on the segmenter rather than decoration -- near 1.0 means
    it claimed the whole search window and almost certainly did not find an
    edge, and very small means it found a fragment.
    """
    if mask is None or getattr(mask, "size", 0) == 0:
        return 0.0
    h, w = img.shape[:2]
    mh, mw = mask.shape[:2]
    ox, oy = int(round(origin_px[0])), int(round(origin_px[1]))
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + mw), min(h, oy + mh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = mask[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    on = sub.astype(bool)
    if not on.any():
        return 0.0

    region = img[y0:y1, x0:x1]
    tint = np.empty_like(region)
    tint[:] = colour
    region[on] = (
        region[on].astype(np.float32) * (1.0 - alpha)
        + tint[on].astype(np.float32) * alpha
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        on.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(region, contours, -1, colour, 1)
    return float(on.sum()) / float(on.size)


def wrap_text(
    text: str, *, max_px: int, scale: float, hard_break: bool = False,
) -> list[str]:
    """Greedy word wrap, measured in the font that will actually draw it.

    Wrapping on a character count is wrong here: the reasons this displays are
    model prose of no fixed width, and ``draw_outlined_text``'s Hershey font is
    not monospaced. ``getTextSize`` is the only thing that knows.

    ``hard_break`` additionally splits a single word too long to fit on any
    line, and keeps blank lines between paragraphs. Off by default because a
    caption over a camera frame has the whole frame width to work with and
    never hits either case; a narrow side panel showing ids, file paths and
    raw model output hits both, and without it an over-long token is drawn
    straight off the edge of the panel rather than wrapped.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    def width(s: str) -> int:
        return cv2.getTextSize(s, font, scale, 1)[0][0]

    lines: list[str] = []
    for para in (text.split("\n") if hard_break else [text]):
        words = para.split()
        if not words and hard_break:
            lines.append("")
            continue
        current = ""
        for word in words:
            if hard_break:
                # Chop anything that will never fit on a line of its own,
                # rather than letting cv2 clip it at the edge of the image.
                # This runs BEFORE the fits-on-this-line test, because the
                # greedy rule below accepts any word onto an empty line
                # unconditionally -- so an over-long word arriving first would
                # otherwise sail straight past the split and off the panel.
                while width(word) > max_px and len(word) > 1:
                    if current:
                        lines.append(current)
                        current = ""
                    cut = len(word)
                    while cut > 1 and width(word[:cut]) > max_px:
                        cut -= 1
                    lines.append(word[:cut])
                    word = word[cut:]
            trial = word if not current else f"{current} {word}"
            if not current or width(trial) <= max_px:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def clipped_text(text: str, *, max_px: int, scale: float) -> str:
    """``text`` cut down to what draws inside ``max_px``, ellipsised when cut.

    For a single line that has to end somewhere, where :func:`wrap_text` would
    wrap: a label beside a box has the room between that box and the edge of
    the frame and no more. Measured in the font that will draw it, because
    ``draw_outlined_text``'s Hershey font is not monospaced and a character
    count is not a width.

    The ellipsis is the point. Whatever this shortens is a summary of something
    written out in full elsewhere -- a gate's whole sentence in a transcript row
    -- and a silently truncated reason reads as the whole reason.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    def width(s: str) -> int:
        return cv2.getTextSize(s, font, scale, 1)[0][0]

    if width(text) <= max_px:
        return text
    cut = len(text)
    while cut > 1 and width(f"{text[:cut]}...") > max_px:
        cut -= 1
    return f"{text[:cut]}..."


def draw_inset(
    canvas: np.ndarray,
    frame: np.ndarray,
    tag: str,
    *,
    width_px: int = 208,
    margin: int = 12,
) -> None:
    """Drop a small labelled thumbnail into ``canvas``' bottom-right corner.

    What it is for: the main pane of an interactive harness shows a *held*
    frame -- the one a model answered about, seconds ago -- and the question
    "is the camera still seeing what I think it is" cannot be answered from
    that picture at all. The inset is the live feed, so the stale main pane
    and the current scene are on screen together and can never be confused.

    A no-op when the canvas is too small to take the thumbnail, rather than an
    exception: this draws on a preview, and a preview that raises is worse
    than one missing a corner.
    """
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return
    tw = width_px
    th = max(1, int(round(h * tw / w)))
    ch, cw = canvas.shape[:2]
    x0, y0 = cw - tw - margin, ch - th - margin
    if x0 < 0 or y0 < 0:
        return
    canvas[y0 : y0 + th, x0 : x0 + tw] = cv2.resize(
        frame, (tw, th), interpolation=cv2.INTER_AREA
    )
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + tw, y0 + th), (220, 220, 220), 1)
    draw_outlined_text(canvas, tag, (x0 + 6, y0 + 18), scale=0.5, color=(220, 220, 220))


def draw_caption(
    img: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    *,
    scale: float = 0.5,
    pad: int = 8,
) -> None:
    """Wrapped ``(text, colour)`` rows over a darkened band along the bottom.

    Darkened rather than filled: the desk under the caption stays visible, and
    the band is the one part of the overlay that has to be readable over
    whatever the camera happened to see.
    """
    h, w = img.shape[:2]
    line_h = max(12, int(round(22 * scale / 0.5)))
    wrapped = [
        (piece, colour)
        for text, colour in lines
        for piece in wrap_text(text, max_px=w - 2 * pad, scale=scale)
    ]
    if not wrapped:
        return
    # Leave the very bottom clear. annotate_for_pointing writes its own
    # "1280x720 grid 100px" footer there, and without this the last caption
    # line lands on top of it with its descenders off the edge of the frame.
    band = min(h, len(wrapped) * line_h + 2 * pad + CAPTION_BOTTOM_RESERVE_PX)
    y0 = h - band
    region = img[y0:h]
    cv2.addWeighted(
        np.zeros_like(region), PANEL_ALPHA, region, 1.0 - PANEL_ALPHA, 0, region
    )
    y = y0 + pad + line_h - 6
    for text, colour in wrapped:
        draw_outlined_text(img, text, (pad, y), scale=scale, color=colour)
        y += line_h


def _draw_alt_reading(
    img: np.ndarray, point: tuple[float, float], note: str
) -> None:
    """The same numbers read in the other coordinate convention.

    Drawn identically wherever it appears -- for the grounding reply and for
    the decision's point -- because the whole diagnostic is "these two marks
    are the same answer, and they are 200-280px apart".
    """
    cv2.drawMarker(
        img, (int(point[0]), int(point[1])), QWEN_ALT_BGR,
        cv2.MARKER_TILTED_CROSS, 16, 1,
    )
    draw_outlined_text(
        img, note, (int(point[0]) + 12, max(12, int(point[1]) - 10)),
        scale=0.45, color=QWEN_ALT_BGR,
    )


def draw_report(img: np.ndarray, findings) -> None:
    """Every object a REPORT boxed, coloured by whether the arm can take it.

    Duck-typed like the rest of this module: an entry needs ``index``,
    ``label``, ``box_px``, ``pickable`` and ``note`` (i.e.
    ``instruct.Finding``).

    Two colours rather than one, because the split inside the list is the
    answer. A report says which of the things on the desk the arm can pick up,
    and that verdict comes from reach, the keep-out, the desk edge and the jaw
    width -- none of which is visible in the photograph, so the picture is the
    only place the two can be compared. Each box carries the gate's own words
    when it is red, so "why not that one" is answered where it is asked.

    The number on a box is the number of its row in the terminal, so a
    rectangle and a line of text can be matched without counting from the top.
    Both lines are clipped to what fits between the box and the right edge --
    a gate's reason is a whole sentence, and the full one is in the row that
    carries the same number.
    """
    width = img.shape[1]
    for f in findings or ():
        box = getattr(f, "box_px", None)
        if box is None:
            continue
        colour = QWEN_BOUND_BGR if getattr(f, "pickable", False) else QWEN_REFUSED_BGR
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2)
        room = max(1, width - x0 - 4)
        draw_outlined_text(
            img,
            clipped_text(
                f"{getattr(f, 'index', '?')}. {getattr(f, 'label', '?')}",
                max_px=room, scale=0.5,
            ),
            (x0, max(12, y0 - 20)), scale=0.5, color=colour,
        )
        draw_outlined_text(
            img,
            clipped_text(str(getattr(f, "note", "")), max_px=room, scale=0.42),
            (x0, max(24, y0 - 6)), scale=0.42, color=colour,
        )


def annotate_qwen(
    base: np.ndarray,
    *,
    grounding=None,
    obj=None,
    action=None,
    report=None,
    bound_px: tuple[float, float] | None = None,
    dest_bound_px: tuple[float, float] | None = None,
    accepted: bool | None = None,
    caption: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """The model's own answer drawn over the frame it answered about.

    Duck-typed on purpose, like the rest of this module: ``grounding`` needs
    ``label`` / ``point_px`` / ``box_px`` / ``alt_point_px`` / ``alt_box_px``
    (i.e. ``instruct.Grounding``) and ``action`` needs ``point_px``, so preview
    keeps its one-way dependency on the layers above it.

    What each thing on the picture is for, since every one of them corresponds
    to a failure this loop actually had:

    * **Solid yellow box** -- the grounding reply. Its size and frame share are
      printed on it, because a box covering the whole image is the model
      declining to answer and it is otherwise indistinguishable from a real one.
    * **Dashed orange box/cross** -- the identical numbers read in the other
      coordinate convention. This build answers 0-1000 normalized whatever the
      prompt says, and the two readings land 200-280px apart, so a coordinate
      space error is a glance rather than an investigation.
    * **Spring-green fill** -- the GrabCut silhouette the box was seeded with
      (``obj``, a ``locate.LocatedObject``), with the measured size and how much
      of the search window it filled. This is the layer everything above depends
      on and the only one you cannot check from numbers: a mask that leaked into
      the object's shadow and one that fits it exactly report the same *kind* of
      answer, differing only in a millimetre count that looks plausible either
      way. Seeing the filled pixels is the check.
    * **Magenta ring** -- the point the decision carried. A magenta line
      through the destination ring is the axis the reply asked the object to be
      laid along; no line means it goes down square to the desk.
    * **Green or red ring, joined to the magenta one** -- the entity the stack
      resolved that point to, and the gap between them in pixels. Green when the
      action was accepted, red when it was refused. A model that names one thing
      and points at another draws a long line, which is the whole diagnosis.
    * **Arrow between two rings** -- a TRANSFER, drawn from the thing to the
      place it is going. Both ends are resolved before the arm moves, and a
      transfer draws no other picture, so an arrow pointing at the wrong marker
      is the only warning there will be: nothing after this looks at the desk
      again.
    * **Numbered green and red boxes** -- a REPORT's whole answer (``report``,
      a sequence of ``instruct.Finding``), green for the objects the arm can
      pick up and red for the rest, each with the gate that stopped it. See
      :func:`draw_report`.
    """
    out = base.copy()
    h, w = out.shape[:2]

    # Under the boxes, so a box edge is never hidden by the fill it produced.
    if obj is not None:
        mask = getattr(obj, "mask", None)
        origin = getattr(obj, "mask_origin_px", (0, 0))
        share = draw_mask(out, mask, origin, QWEN_MASK_BGR)
        cx, cy = getattr(obj, "px", None), getattr(obj, "py", None)
        if cx is not None and cy is not None:
            cv2.drawMarker(
                out, (int(cx), int(cy)), QWEN_MASK_BGR, cv2.MARKER_CROSS, 14, 2
            )
            long_mm = float(getattr(obj, "long_mm", 0.0) or 0.0)
            short_mm = float(getattr(obj, "short_mm", 0.0) or 0.0)
            note = f"mask {long_mm:.0f}x{short_mm:.0f}mm"
            if share:
                note += f"  fills {share:.0%} of its window"
            if share >= WHOLE_WINDOW_SHARE:
                note += "  <-- took the whole window, found no edge"
            # Below the mask window, not at the centroid: a label written across
            # the middle of the silhouette hides the thing it is describing, and
            # the shape is the reason this is on screen at all.
            ly = int(origin[1]) + (0 if mask is None else mask.shape[0]) + 16
            if ly > h - 12:
                ly = max(12, int(origin[1]) - 8)
            draw_outlined_text(
                out, note, (max(2, int(origin[0])), ly),
                scale=0.5, color=QWEN_MASK_BGR,
            )

    draw_report(out, report)

    if grounding is not None:
        alt_box = getattr(grounding, "alt_box_px", None)
        alt_pt = getattr(grounding, "alt_point_px", None)
        # The unbelieved reading underneath, so the primary always wins a tie.
        if alt_box is not None:
            draw_dashed_rect(out, *alt_box, QWEN_ALT_BGR)
        anchor = alt_pt or (None if alt_box is None else alt_box[:2])
        if alt_pt is not None:
            _draw_alt_reading(out, alt_pt, "same numbers, other coord space")
        elif anchor is not None:
            draw_outlined_text(
                out, "same numbers, other coord space",
                (int(anchor[0]) + 12, max(12, int(anchor[1]) - 10)),
                scale=0.45, color=QWEN_ALT_BGR,
            )

        box = getattr(grounding, "box_px", None)
        label = str(getattr(grounding, "label", "?"))
        if box is not None:
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(out, (x0, y0), (x1, y1), QWEN_BOX_BGR, 2)
            share = ((x1 - x0) * (y1 - y0)) / float(max(1, w * h))
            note = "  <-- the whole frame, not an object" if share >= BIG_BOX_SHARE else ""
            draw_outlined_text(
                out, f'"{label}"  {x1 - x0}x{y1 - y0}px  {share:.0%} of frame{note}',
                (x0, max(12, y0 - 6)), scale=0.5, color=QWEN_BOX_BGR,
            )
        point = getattr(grounding, "point_px", None)
        if point is not None:
            draw_lock_ring(out, point[0], point[1], QWEN_BOX_BGR)
            if box is None:
                draw_outlined_text(
                    out, f'"{label}" (point only, no box)',
                    (int(point[0]) + 18, int(point[1]) - 12),
                    scale=0.5, color=QWEN_BOX_BGR,
                )

    alt = None if action is None else getattr(action, "alt_point_px", None)
    if alt is not None:
        _draw_alt_reading(out, alt, "point, other coord space")

    point = None if action is None else getattr(action, "point_px", None)
    if point is not None:
        draw_lock_ring(out, point[0], point[1], QWEN_POINT_BGR)
        draw_outlined_text(
            out, "model point", (int(point[0]) + 18, int(point[1]) + 20),
            scale=0.5, color=QWEN_POINT_BGR,
        )
    colour = QWEN_REFUSED_BGR if accepted is False else QWEN_BOUND_BGR
    if bound_px is not None:
        bx, by = int(bound_px[0]), int(bound_px[1])
        cv2.circle(out, (bx, by), 20, colour, 2)
        if point is not None:
            cv2.line(
                out, (int(point[0]), int(point[1])), (bx, by), colour, 1, cv2.LINE_AA
            )
            gap = math.hypot(float(point[0]) - bx, float(point[1]) - by)
            draw_outlined_text(
                out, f"{gap:.0f}px",
                (int((point[0] + bx) / 2) + 6, int((point[1] + by) / 2) - 6),
                scale=0.45, color=colour,
            )

    # The destination half of a transfer. Its own point ring first, so a model
    # that names the right marker while pointing somewhere else shows the same
    # split here as it does for the thing being picked.
    dest_point = None if action is None else getattr(action, "dest_point_px", None)
    if dest_point is not None:
        draw_lock_ring(out, dest_point[0], dest_point[1], QWEN_POINT_BGR)
    # The release orientation, drawn for the same reason every coordinate is:
    # it is a thing the reply decided and the arm obeys, so it has to be
    # checkable at a glance rather than only in the JSON. Both ways along the
    # axis, because the jaws close across it and the reverse means the same.
    dest_axis = None if action is None else getattr(action, "dest_axis_px", None)
    if dest_point is not None and dest_axis is not None:
        ax, ay = dest_axis[0] - dest_point[0], dest_axis[1] - dest_point[1]
        span = (ax * ax + ay * ay) ** 0.5
        if span > 1.0:
            ux, uy = ax / span, ay / span
            reach = max(28.0, min(span, 90.0))
            cv2.line(
                out,
                (int(dest_point[0] - ux * reach), int(dest_point[1] - uy * reach)),
                (int(dest_point[0] + ux * reach), int(dest_point[1] + uy * reach)),
                QWEN_POINT_BGR, 2, cv2.LINE_AA,
            )
            draw_outlined_text(
                out, "lay along",
                (int(dest_point[0] + ux * reach) + 6,
                 int(dest_point[1] + uy * reach) + 4),
                scale=0.45, color=QWEN_POINT_BGR,
            )
    if dest_bound_px is not None:
        dx, dy = int(dest_bound_px[0]), int(dest_bound_px[1])
        cv2.circle(out, (dx, dy), 20, colour, 2)
        if bound_px is not None:
            cv2.arrowedLine(
                out, (int(bound_px[0]), int(bound_px[1])), (dx, dy),
                colour, 2, cv2.LINE_AA, tipLength=0.04,
            )
            draw_outlined_text(
                out, "moves to",
                (int((bound_px[0] + dx) / 2) + 6, int((bound_px[1] + dy) / 2) + 18),
                scale=0.5, color=colour,
            )

    if caption:
        draw_caption(out, list(caption))
    return out
