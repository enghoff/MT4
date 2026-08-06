"""USB camera capture for the scene camera watching the work surface."""

from __future__ import annotations

import os
import sys
import threading

import cv2
import numpy as np

# Index used when no --camera is given. MT4_CAMERA_INDEX is the per-machine
# setting (which USB index the scene camera landed on -- 1 on a laptop
# whose built-in camera takes 0); 0 is the fallback when it is unset. Set it to
# -1 for auto-detect: scan indices for the camera that sees ArUco markers,
# which costs seconds per candidate but needs no per-machine configuration.
# Read at import from the real environment, not .env (only mt4_mcp loads that):
# a shell started before the variable was set inherits a copy without it.
DEFAULT_CAMERA_INDEX = int(os.environ.get("MT4_CAMERA_INDEX", "0"))
AUTO_SCAN_MAX_INDEX = 5
# The driver's default UVC mode is 640x480, where each ArUco marker (already
# viewed at a steep angle from an oblique mount, and small relative to a
# frame that has to cover the whole desk) is only ~20-35px per side -- a few
# pixels per code cell, right at the edge of reliable decoding. Requesting
# 720p roughly doubles that and made all 5 markers decode reliably in
# testing; the driver clamps to its nearest supported mode if unsupported.
CAPTURE_WIDTH = int(os.environ.get("MT4_CAMERA_WIDTH", "1280"))
CAPTURE_HEIGHT = int(os.environ.get("MT4_CAMERA_HEIGHT", "720"))
# The camera driver buffers several frames, so the first read() after a period
# of inactivity returns a stale image of the scene as it was seconds ago --
# fatal for pick-and-place, where we detect right before moving. Discard this
# many frames before keeping one.
FLUSH_FRAMES = 5
# Right after opening (and especially after the resolution switch above),
# auto-exposure hasn't converged yet -- frames come back badly overexposed,
# which washes out cube color saturation enough to break HSV detection.
# cap.grab() alone doesn't drive convergence (only decoded reads do), so this
# warm-up does full read()s, not grab()s. ~20 reads (~2-3s) was enough for
# brightness to stabilize in testing; cheap relative to a whole session.
WARMUP_READS = 20


class CameraError(Exception):
    """Raised when the camera cannot be opened or a frame cannot be read."""


# Auto-detect result, cached because opening each candidate camera costs
# seconds. Reset by unplugging/replugging only across process restarts.
_detected_index: int | None = None


def _open_raw(index: int) -> cv2.VideoCapture:
    # CAP_DSHOW: the default MSMF backend on Windows takes several seconds to
    # open and sometimes refuses resolution changes on this Lenovo camera.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        for _ in range(WARMUP_READS):
            cap.read()
    return cap


def _autodetect_index() -> int:
    """Find the work camera: the one that sees ArUco markers on the desk."""
    global _detected_index
    if _detected_index is not None:
        return _detected_index
    from mt4_vision.detect import scan_marker_dicts  # deferred: detect imports cv2 extras

    for index in range(AUTO_SCAN_MAX_INDEX + 1):
        cap = _open_raw(index)
        if not cap.isOpened():
            cap.release()
            continue
        try:
            frame = grab_frame(cap)
        except CameraError:
            cap.release()
            continue
        cap.release()
        if scan_marker_dicts(frame):
            _detected_index = index
            return index
    raise CameraError(
        f"no camera with visible ArUco markers found in indices 0-{AUTO_SCAN_MAX_INDEX}; "
        "set MT4_CAMERA_INDEX or pass --camera explicitly"
    )


def open_camera(index: int = DEFAULT_CAMERA_INDEX) -> cv2.VideoCapture:
    if index < 0:
        index = _autodetect_index()
    cap = _open_raw(index)
    if not cap.isOpened():
        raise CameraError(f"could not open camera index {index}")
    return cap


def grab_frame(cap: cv2.VideoCapture, flush: int = FLUSH_FRAMES) -> np.ndarray:
    """Read one fresh BGR frame, discarding buffered stale frames first."""
    for _ in range(flush):
        cap.grab()
    ok, frame = cap.read()
    if not ok or frame is None:
        raise CameraError("camera read failed")
    return frame


def capture_frame(index: int = DEFAULT_CAMERA_INDEX) -> np.ndarray:
    """One-shot open/grab/release for callers without a long-lived capture."""
    cap = open_camera(index)
    try:
        return grab_frame(cap)
    finally:
        cap.release()


class FrameStream:
    """Continuously drained camera for long sessions needing FRESH frames.

    A long-lived VideoCapture that is read only occasionally serves frames
    from the driver's buffer -- scenes many seconds old (arm mid-motion,
    cubes at previous positions), and a fixed flush count cannot promise
    reaching the present. A one-shot reopen per capture is fresh but costs
    2-3s of open + exposure warmup, and rapid reopen cycles are what cause
    the unconverged-exposure cold starts. This reader thread drains the
    stream at camera rate; ``fresh()`` blocks until a frame whose capture
    STARTED after the call completes (min_advance=2: the frame being
    delivered at call time plus one full frame period).
    """

    def __init__(self, index: int = DEFAULT_CAMERA_INDEX) -> None:
        self._cap = open_camera(index)
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stopped:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def fresh(self, min_advance: int = 2, timeout_s: float = 5.0) -> np.ndarray:
        """A frame captured entirely after this call."""
        with self._cond:
            target = self._seq + min_advance
            while self._seq < target and not self._stopped:
                if not self._cond.wait(timeout=timeout_s):
                    raise CameraError("frame stream stalled")
            if self._frame is None:
                raise CameraError("frame stream produced no frames")
            return self._frame.copy()

    def close(self) -> None:
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._cap.release()
