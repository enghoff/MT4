"""Client for the SAM 2.1 segmentation service.

An HTTP service reached at ``MT4_SAM_URL`` (default ``http://127.0.0.1:8767``).
It may run on this machine or on another host -- if the latter and you reach it
by SSH forward, see ``scripts/start_tunnel.ps1``. Setup: ``docs/SAM2.md``.

Where :mod:`mt4_vision.grounding` answers "where is the pen" with a box, this
answers "which pixels are it" with a mask, from a point or a box you already
have. The two compose, and :mod:`mt4_vision.locate` is where: every
measurement that starts from a box prompts this service and measures the
silhouette that comes back. Cube pick/place, calibration and stacking do not
touch it, and neither does measuring from a bare pixel, which segments by
distance from the local desk colour.

The service keeps the image encoder's output for the last few frames it has
seen, keyed by the bytes of the JPEG it was sent. Since :func:`segment`
encodes a given frame the same way every time, a second call about the same
frame skips the encoder -- about 20 ms of service time instead of 50.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_URL = os.environ.get("MT4_SAM_URL", "http://127.0.0.1:8767")

_UNREACHABLE_HINT = (
    "start the service (`systemctl start sam2` on the GPU host), or -- if it "
    "runs on another host reached by SSH forward -- "
    ".\\scripts\\start_tunnel.ps1"
)


class SamError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mask:
    """One candidate silhouette for one prompt, in submitted-frame pixels.

    ``object_index`` says which prompt it answers, and ``score`` is the model's
    own predicted IoU for it. A single point is genuinely ambiguous -- the cube,
    its top face, or the stack it sits on are all correct answers -- so the
    model returns three candidates per prompt and ranks them itself. They
    arrive best-first per object; :func:`best_per_object` takes the top one.
    """

    object_index: int
    score: float
    area: int
    # Pixel xyxy of the mask's extent, or None for an empty mask.
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray

    @property
    def cx(self) -> float:
        """Centroid x. The mask's own centre of area, not the box centre."""
        return float(np.nonzero(self.mask)[1].mean())

    @property
    def cy(self) -> float:
        return float(np.nonzero(self.mask)[0].mean())


def health(url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """GET /health. Raises SamError if the tunnel/service is down."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SamError(f"sam service unreachable at {url} ({exc}); {_UNREACHABLE_HINT}") from exc


def decode_rle(runs: Sequence[int], height: int, width: int) -> np.ndarray:
    """Run lengths back into a boolean mask.

    The service sends row-major runs that alternate background, foreground,
    background, ... always starting on background -- a leading zero-length run
    appears when the top-left pixel is inside the mask.
    """
    values = np.arange(len(runs)) % 2 == 1
    return np.repeat(values, runs).reshape(height, width)


def _jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise SamError("failed to JPEG-encode frame")
    return bytes(buf)


def _post(url: str, endpoint: str, fields: list[tuple[str, str]], image: bytes | None, timeout: float) -> dict:
    boundary = "----mt4sam"
    parts = [
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode("utf-8")
        for name, value in fields
    ]
    if image is not None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8")
            + image
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{url.rstrip('/')}/{endpoint}",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SamError(f"{endpoint} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SamError(f"sam service unreachable at {url} ({exc}); {_UNREACHABLE_HINT}") from exc


def embed(frame: np.ndarray, *, url: str = DEFAULT_URL, timeout: float = 60.0, jpeg_quality: int = 90) -> str:
    """Encode a frame now and return its ``image_id``, for prompting later.

    Worth calling on a frame the arm is about to be asked about while it is
    still moving: the encoder's 40 ms happens then rather than inside the call
    that needs the answer.
    """
    payload = _post(url, "embed", [], _jpeg(frame, jpeg_quality), timeout)
    return str(payload["image_id"])


def segment(
    frame: np.ndarray | None,
    *,
    points: Sequence[Sequence[float]] | None = None,
    labels: Sequence[int] | Sequence[Sequence[int]] | None = None,
    boxes: Sequence[Sequence[float]] | None = None,
    multimask: bool = True,
    image_id: str | None = None,
    url: str = DEFAULT_URL,
    timeout: float = 60.0,
    jpeg_quality: int = 90,
) -> list[Mask]:
    """Segment what the prompts point at in a BGR OpenCV frame.

    ``points`` is ``[[x, y], ...]``, one object per point, or
    ``[[[x, y], [x, y]], ...]`` to put several points on one object --  which
    is how a second point refines the first, with ``labels`` 1 for a point on
    the object and 0 for one off it. ``boxes`` is ``[[x1, y1, x2, y2], ...]``,
    the shape a Grounding DINO detection already has.

    Pass ``frame=None`` with an ``image_id`` from :func:`embed` to prompt a
    frame the service still holds. That raises :class:`SamError` (HTTP 404)
    once the frame has aged out of its cache, so keep the frame if you may
    need it again.

    Returns every candidate, best-first within each object.
    """
    if points is None and boxes is None:
        raise SamError("segment needs points, boxes, or both")

    fields: list[tuple[str, str]] = [("multimask", "true" if multimask else "false")]
    if points is not None:
        fields.append(("points", json.dumps([[float(v) for v in p] for p in points])))
    if labels is not None:
        # Flat [1, 0] (one per object) and nested [[1, 0]] (several on one
        # object) both go through; the service reads the shape it is given.
        flags = [list(v) if isinstance(v, (list, tuple)) else int(v) for v in labels]
        fields.append(("labels", json.dumps(flags)))
    if boxes is not None:
        fields.append(("boxes", json.dumps([[float(v) for v in b] for b in boxes])))

    if frame is None:
        if not image_id:
            raise SamError("segment needs a frame or an image_id")
        fields.append(("image_id", image_id))
        image = None
    else:
        image = _jpeg(frame, jpeg_quality)

    payload = _post(url, "segment", fields, image, timeout)
    height, width = int(payload["height"]), int(payload["width"])
    out: list[Mask] = []
    for entry in payload.get("masks") or []:
        bbox = entry.get("bbox")
        out.append(
            Mask(
                object_index=int(entry["object_index"]),
                score=float(entry["score"]),
                area=int(entry["area"]),
                bbox=tuple(int(v) for v in bbox) if bbox else None,
                mask=decode_rle(entry["rle"], height, width),
            )
        )
    return out


def best_per_object(masks: Sequence[Mask]) -> list[Mask]:
    """The highest-scoring candidate for each prompt, in prompt order."""
    best: dict[int, Mask] = {}
    for m in masks:
        if m.object_index not in best or m.score > best[m.object_index].score:
            best[m.object_index] = m
    return [best[i] for i in sorted(best)]
