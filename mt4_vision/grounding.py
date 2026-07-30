"""Client for the Grounding DINO service on media (via SSH tunnel).

The detector runs on the GPU host ``media`` and is reached at
``MT4_GROUNDING_URL`` (default ``http://127.0.0.1:8765``), which should be an
SSH local forward started with ``scripts/start_grounding_tunnel.ps1``.

Boxes feed the existing locate path: take a box centre (or best interior
point) and pass it to :func:`mt4_vision.locate.measure`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_URL = os.environ.get("MT4_GROUNDING_URL", "http://127.0.0.1:8765")


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    # Pixel xyxy in the submitted frame.
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "box": [self.x1, self.y1, self.x2, self.y2],
            "cx": self.cx,
            "cy": self.cy,
        }


class GroundingError(RuntimeError):
    pass


def health(url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """GET /health. Raises GroundingError if the tunnel/service is down."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise GroundingError(
            f"grounding service unreachable at {url} ({exc}); "
            "start the tunnel: .\\scripts\\start_grounding_tunnel.ps1"
        ) from exc


def detect(
    frame: np.ndarray,
    prompt: str,
    *,
    url: str = DEFAULT_URL,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    timeout: float = 60.0,
) -> list[Detection]:
    """Run open-vocab detection on a BGR OpenCV frame.

    ``prompt`` may be a single phrase (``"pen"``) or period-separated
    (``"pen. red cube."``). Returns detections sorted by score descending.
    """
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise GroundingError("failed to JPEG-encode frame")
    boundary = "----mt4grounding"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{prompt}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="box_threshold"\r\n\r\n'
        f"{box_threshold}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="text_threshold"\r\n\r\n'
        f"{text_threshold}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + bytes(buf) + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{url.rstrip('/')}/detect",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GroundingError(f"detect HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GroundingError(
            f"grounding service unreachable at {url} ({exc}); "
            "start the tunnel: .\\scripts\\start_grounding_tunnel.ps1"
        ) from exc

    if not payload.get("ok", True):
        raise GroundingError(payload.get("error", "detect failed"))

    out: list[Detection] = []
    for d in payload.get("detections") or []:
        box = d["box"]
        out.append(
            Detection(
                label=str(d.get("label", prompt)),
                score=float(d["score"]),
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )
        )
    out.sort(key=lambda d: d.score, reverse=True)
    return out
