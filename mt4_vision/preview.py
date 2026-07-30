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


def annotate_for_pointing(
    frame: np.ndarray,
    entities: list | None = None,
    *,
    grid_px: int = POINTING_GRID_PX,
) -> np.ndarray:
    """Frame with a labelled pixel grid and any known entity ids drawn on it.

    Built for handing a frame to a vision-capable model and asking "which pixel
    is the pen". The grid is the point: a model reading coordinates off drawn,
    numbered gridlines is self-correcting about the coordinate space, whereas one
    estimating them from the image alone can be confidently off by a wide margin
    and there is no way to tell from the answer. Drawing the ids of what is
    already detected does the other half -- it stops the model pointing at a cube
    the entity table already covers.

    ``entities`` is any sequence of objects with ``id``, ``pixel`` and
    ``pickable`` (i.e. ``mt4_vision.entities.Entity``); entities without a pixel
    (open slots) are skipped. Kept duck-typed so preview does not depend on the
    entity layer.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Grid first, so labels and ids sit on top of it.
    for x in range(grid_px, w, grid_px):
        cv2.line(out, (x, 0), (x, h), GRID_BGR, 1)
        draw_outlined_text(out, str(x), (x + 3, 14), scale=0.4, color=GRID_BGR)
    for y in range(grid_px, h, grid_px):
        cv2.line(out, (0, y), (w, y), GRID_BGR, 1)
        draw_outlined_text(out, str(y), (3, y - 3), scale=0.4, color=GRID_BGR)
    draw_outlined_text(
        out, f"{w}x{h}  grid {grid_px}px", (3, h - 8), scale=0.45, color=GRID_BGR
    )

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
