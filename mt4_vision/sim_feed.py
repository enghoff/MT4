"""Shared-memory scene-camera feed from MT4-sim.

``serve_firmware.py --camera`` publishes BGR frames into a named shared-memory
block. Set ``MT4_CAMERA_URL=shm://mt4_scene_cam`` (or pass the URL through
``open_camera``) and the control stack reads those frames instead of a USB
device -- no monkey-patch required.

Layout must stay in lockstep with ``mt4_sim.camera_feed``::

    offset  0  uint64  sequence number (monotone; written last)
    offset  8  uint32  width
    offset 12  uint32  height
    offset 16  uint32  channels (always 3 = BGR)
    offset 20  uint32  frame bytes
    offset 64  uint8[] BGR pixels
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

HEADER_FORMAT = "<QIIII"
HEADER_SIZE = 64


def parse_shm_url(url: str) -> str:
    if not url.startswith("shm://"):
        raise ValueError(f"camera feed URL must look like shm://name (got {url!r})")
    name = url[len("shm://") :].strip()
    if not name:
        raise ValueError(f"empty shared-memory name in {url!r}")
    return name


@dataclass(frozen=True)
class FrameMeta:
    seq: int
    width: int
    height: int
    channels: int
    nbytes: int


class SimFeedCapture:
    """Duck-typed ``cv2.VideoCapture`` over the sim's shared-memory feed."""

    def __init__(self, url: str, *, timeout_s: float = 30.0, warmup_reads: int = 2) -> None:
        self._url = url
        self._name = parse_shm_url(url)
        self._shm: shared_memory.SharedMemory | None = None
        self._opened = False
        self._frame: np.ndarray | None = None
        self._open(timeout_s=timeout_s)
        for _ in range(max(0, warmup_reads)):
            self.read()

    def _open(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._shm = shared_memory.SharedMemory(name=self._name)
                self._opened = True
                return
            except FileNotFoundError as exc:
                last_error = exc
                time.sleep(0.05)
        raise TimeoutError(
            f"no camera feed at {self._url} after {timeout_s:.0f}s "
            f"(is MT4-sim's serve_firmware.py --camera running?)"
        ) from last_error

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's spelling
        return self._opened and self._shm is not None

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 0.0

    def grab(self) -> bool:
        if not self.isOpened() or self._shm is None:
            return False
        try:
            _, frame = self._read_consistent()
        except RuntimeError:
            return False
        self._frame = frame
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.grab():
            return False, None
        return True, self._frame

    def release(self) -> None:
        self._opened = False
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def _read_meta(self) -> FrameMeta:
        assert self._shm is not None
        seq, width, height, channels, nbytes = struct.unpack_from(
            HEADER_FORMAT, self._shm.buf, 0
        )
        return FrameMeta(seq, width, height, channels, nbytes)

    def _read_consistent(self, *, retries: int = 8) -> tuple[FrameMeta, np.ndarray]:
        assert self._shm is not None
        last: Exception | None = None
        for _ in range(retries):
            try:
                meta = self._read_meta()
                if meta.seq == 0 or meta.nbytes <= 0:
                    time.sleep(0.01)
                    continue
                raw = bytes(self._shm.buf[HEADER_SIZE : HEADER_SIZE + meta.nbytes])
                again = self._read_meta()
                if again.seq != meta.seq or again.nbytes != meta.nbytes:
                    continue
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (meta.height, meta.width, meta.channels)
                )
                return again, frame.copy()
            except Exception as exc:  # noqa: BLE001 - reshape/size races
                last = exc
                time.sleep(0.0)
        raise RuntimeError(f"could not read a consistent camera frame from {self._url}") from last
