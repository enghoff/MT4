"""Client for the Qwen3-VL vision-language service.

An HTTP service reached at ``MT4_QWEN_URL`` (default
``http://127.0.0.1:8766``). It may run on this machine or on another host --
if the latter and you reach it by SSH forward, see
``scripts/start_qwen_tunnel.ps1``. Setup: ``docs/QWEN3-VL.md``.

Unlike :mod:`mt4_vision.grounding`, which returns boxes, this returns free
text -- the model answers a question about a frame. When the prompt asks for
grounding, Qwen answers with JSON boxes/points embedded in that text, so
:func:`parse_regions` digs them back out for drawing.

Nothing in the arm stack depends on this module; it exists for the
interactive harness (``ask_qwen.py``) that checks what the VLM can and
cannot see.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, replace

import cv2
import numpy as np

DEFAULT_URL = os.environ.get("MT4_QWEN_URL", "http://127.0.0.1:8766")
# A 4B model in 4-bit on a mid-range card takes single-digit seconds for a
# few hundred tokens, and generation is serialized on the GPU -- so a queued
# request waits for the one ahead of it. Well clear of both.
DEFAULT_TIMEOUT_S = 180.0

_UNREACHABLE_HINT = (
    "start the service (`systemctl start qwen3-vl` on the GPU host), or -- if "
    "it runs on another host reached by SSH forward -- "
    ".\\scripts\\start_qwen_tunnel.ps1"
)


class QwenError(RuntimeError):
    pass


# Qwen's grounding replies key boxes and points under several names across
# prompt phrasings and model revisions; accept all of them rather than
# pinning one and silently drawing nothing.
_BOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box")
_POINT_KEYS = ("point_2d", "point", "coordinate")
_LABEL_KEYS = ("label", "name", "text", "object")


@dataclass(frozen=True)
class Region:
    """A box or point Qwen named in its answer, in submitted-frame pixels.

    ``kind`` is ``"box"`` (4 coords, xyxy) or ``"point"`` (2 coords). Whether
    those numbers are really pixels is the open question this class does not
    settle -- see :func:`parse_regions` and :meth:`in_bounds`.
    """

    label: str
    kind: str
    coords: tuple[float, ...]

    @property
    def cx(self) -> float:
        if self.kind == "point":
            return self.coords[0]
        return 0.5 * (self.coords[0] + self.coords[2])

    @property
    def cy(self) -> float:
        if self.kind == "point":
            return self.coords[1]
        return 0.5 * (self.coords[1] + self.coords[3])

    def scaled(self, sx: float, sy: float) -> Region:
        scale = (sx, sy) * (len(self.coords) // 2)
        return replace(
            self,
            coords=tuple(c * s for c, s in zip(self.coords, scale)),
        )

    def in_bounds(self, width: int, height: int, *, slack: float = 0.02) -> bool:
        """True if every coordinate lands inside the frame (with slack).

        A whole answer's worth of regions failing this is the signature of a
        coordinate-convention mismatch, not of a bad detection -- see the
        note in :func:`parse_regions`.
        """
        mx, my = width * (1 + slack), height * (1 + slack)
        xs = self.coords[0::2]
        ys = self.coords[1::2]
        return all(-width * slack <= x <= mx for x in xs) and all(
            -height * slack <= y <= my for y in ys
        )


def health(url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """GET /health. Raises QwenError if the tunnel/service is down."""
    try:
        with urllib.request.urlopen(
            f"{url.rstrip('/')}/health", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise QwenError(f"qwen service unreachable at {url} ({exc}); {_UNREACHABLE_HINT}") from exc


def generate(
    prompt: str,
    frame: np.ndarray | None = None,
    *,
    url: str = DEFAULT_URL,
    max_new_tokens: int = 256,
    timeout: float = DEFAULT_TIMEOUT_S,
    jpeg_quality: int = 90,
) -> str:
    """Ask the model ``prompt``, optionally about a BGR OpenCV ``frame``.

    Returns the answer text. Omit ``frame`` for a text-only prompt (useful
    for checking the service is really answering rather than echoing).
    """
    text = prompt.strip()
    if not text:
        raise QwenError("empty prompt")

    boundary = "----mt4qwen"
    parts = [
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{text}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="max_new_tokens"\r\n\r\n'
        f"{int(max_new_tokens)}\r\n".encode("utf-8")
    ]
    if frame is not None:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            raise QwenError("failed to JPEG-encode frame")
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; '
                f'filename="frame.jpg"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8")
            + bytes(buf)
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise QwenError(f"generate HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise QwenError(f"qwen service unreachable at {url} ({exc}); {_UNREACHABLE_HINT}") from exc

    if not payload.get("ok", True):
        raise QwenError(payload.get("error", "generate failed"))
    return str(payload.get("response", ""))


def _json_spans(text: str) -> Iterator[str]:
    """Candidate JSON substrings: fenced blocks first, then balanced spans."""
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        yield m.group(1)

    openers = {"[": "]", "{": "}"}
    i = 0
    while i < len(text):
        if text[i] not in openers:
            i += 1
            continue
        depth = 0
        end = -1
        for j in range(i, len(text)):
            if text[j] in openers:
                depth += 1
            elif text[j] in ("]", "}"):
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:
            return
        yield text[i : end + 1]
        i = end + 1


def _numbers(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        nums = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return nums


def _label_of(obj: dict) -> str:
    for key in _LABEL_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _regions_from_dict(obj: dict) -> list[Region]:
    label = _label_of(obj)
    out: list[Region] = []
    for keys, kind, arity in (
        (_BOX_KEYS, "box", 4),
        (_POINT_KEYS, "point", 2),
    ):
        for key in keys:
            if key not in obj:
                continue
            raw = obj[key]
            # One entry may hold a single coord list or a list of them.
            groups = raw if (isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple))) else [raw]
            for group in groups:
                nums = _numbers(group)
                if nums is not None and len(nums) == arity:
                    out.append(Region(label=label, kind=kind, coords=nums))
            break
    return out


def _walk(node: object, out: list[Region]) -> None:
    if isinstance(node, dict):
        out.extend(_regions_from_dict(node))
        for value in node.values():
            _walk(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk(item, out)


def parse_regions(text: str, *, fallback_bare: bool = True) -> list[Region]:
    """Pull boxes/points out of an answer, in the order the model wrote them.

    Qwen emits grounding as JSON (``[{"bbox_2d": [x1, y1, x2, y2], "label":
    "pen"}]``), sometimes fenced, sometimes wrapped in prose, sometimes with
    a different key -- all handled here. ``fallback_bare`` also accepts a
    bare ``[x1, y1, x2, y2]`` list when no keyed form is found.

    **The coordinates are returned exactly as written.** Which space they are
    in is not something a caller can assume: Qwen2-VL used a 0-1000
    normalized space, 2.5-VL onward is documented as absolute pixels, and
    "absolute" may still mean the processor's internally resized image rather
    than the one submitted.

    Measured on the reference deployment (Qwen3-VL-4B-Instruct, NF4, a
    1280x720 frame): output is **0-1000 normalized**, not pixels -- a box
    reported at x=807 sits at 1033px, and asking for the whole desk returns
    exactly ``[0, 433, 1000, 1000]``. So ``ask_qwen.py`` scales by default.
    Do not generalize that to another build; re-check it, which is what the
    harness and :meth:`Region.in_bounds` are for.
    """
    out: list[Region] = []
    for span in _json_spans(text):
        try:
            parsed = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            continue
        _walk(parsed, out)

    if not out and fallback_bare:
        for m in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*\]", text):
            nums = tuple(float(p) for p in m.group(1).split(","))
            if len(nums) == 4:
                out.append(Region(label="", kind="box", coords=nums))
            elif len(nums) == 2:
                out.append(Region(label="", kind="point", coords=nums))

    seen: set[tuple[str, str, tuple[float, ...]]] = set()
    unique: list[Region] = []
    for r in out:
        key = (r.label, r.kind, r.coords)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique
