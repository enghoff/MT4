"""Client for the Qwen3-VL vision-language service.

An HTTP service reached at ``MT4_QWEN_URL`` (default
``http://127.0.0.1:8766``). It may run on this machine or on another host --
if the latter and you reach it by SSH forward, see
``scripts/start_tunnel.ps1``. Setup: ``docs/QWEN3-VL.md``.

Unlike :mod:`mt4_vision.grounding`, which returns boxes, this returns free
text -- the model answers a question about a frame. When the prompt asks for
grounding, Qwen answers with JSON boxes/points embedded in that text, so
:func:`parse_regions` digs them back out for drawing.

:mod:`mt4_vision.instruct` reaches the model through :func:`ask`, one frame
and one task per call, and reads the action to take out of the answer.
``ask_qwen.py`` is the harness around that loop. Cube pick/place,
calibration, stacking and the MCP tools do not touch this module and work
with the service absent.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

DEFAULT_URL = os.environ.get("MT4_QWEN_URL", "http://127.0.0.1:8766")
# A 4B model in 4-bit on a mid-range card takes single-digit seconds for a
# few hundred tokens, and generation is serialized on the GPU -- so a queued
# request waits for the one ahead of it. Well clear of both.
DEFAULT_TIMEOUT_S = 180.0
# Frames per second the service should be told a frame sequence represents.
# Only used to label video timestamps; it must match the real capture spacing
# or "when did it happen" answers are wrong by the ratio of the two.
DEFAULT_FPS = 2.0
# The vision encoder pairs adjacent frames into one temporal patch, so video
# costs about half the tokens of the same frames sent as separate images
# (measured on the reference deployment: 695 vs 1341 prompt tokens for 6).
TEMPORAL_PATCH_SIZE = 2

_UNREACHABLE_HINT = (
    "start the service (`systemctl start qwen3-vl` on the GPU host), or -- if "
    "it runs on another host reached by SSH forward -- "
    ".\\scripts\\start_tunnel.ps1"
)


class QwenError(RuntimeError):
    pass


# Qwen's grounding replies key boxes and points under several names across
# prompt phrasings and model revisions; accept all of them rather than
# pinning one and silently drawing nothing.
_BOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box")
_POINT_KEYS = ("point_2d", "point", "coordinate")
_LABEL_KEYS = ("label", "name", "text", "object")
_DESC_KEYS = ("description", "desc", "caption")


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
    description: str = ""

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


@dataclass(frozen=True)
class Reply:
    """A service response, with enough metadata to tell whether it lied.

    ``text`` is the answer; everything else describes what the model was
    actually given. That matters because the video path's original failure was
    silent -- frames past the fourth were dropped and the answer still read
    fluently. :meth:`frame_warning` is the check worth making on every
    multi-frame call.
    """

    text: str
    mode: str = "text"
    frames_sent: int = 0
    images_encoded: int | None = None
    temporal_groups: int | None = None
    timestamps_s: tuple[float, ...] = ()
    prompt_tokens: int | None = None
    raw: dict = field(default_factory=dict)

    def frame_warning(self) -> str | None:
        """Non-None when the service encoded fewer frames than were sent."""
        if self.mode == "video" and self.temporal_groups is not None:
            # Frames are paired, so N frames should give ceil(N/2) groups.
            expected = (self.frames_sent + 1) // 2
            if self.temporal_groups < expected:
                return (
                    f"service encoded {self.temporal_groups} temporal groups for "
                    f"{self.frames_sent} frames (expected {expected}) -- frames "
                    "were dropped"
                )
        if self.mode in ("image", "images") and self.images_encoded is not None:
            if self.images_encoded < self.frames_sent:
                return (
                    f"service encoded {self.images_encoded} of {self.frames_sent} "
                    "images -- frames were dropped"
                )
        return None


def health(url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """GET /health. Raises QwenError if the tunnel/service is down."""
    try:
        with urllib.request.urlopen(
            f"{url.rstrip('/')}/health", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise QwenError(f"qwen service unreachable at {url} ({exc}); {_UNREACHABLE_HINT}") from exc
    except TimeoutError as exc:
        # See ask(): a read timeout is not a URLError, and an unwrapped one
        # here kills a caller's startup check with a traceback rather than the
        # message it was written to print. A /health this slow means the
        # service is alive but wholly occupied, which is worth saying.
        raise QwenError(
            f"qwen service at {url} did not answer /health within {timeout:g}s "
            "(alive, but busy generating or compiling)"
        ) from exc


def _jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise QwenError("failed to JPEG-encode frame")
    return bytes(buf)


def ask(
    prompt: str,
    frames: Sequence[np.ndarray] | np.ndarray | None = None,
    *,
    mode: str = "images",
    fps: float = DEFAULT_FPS,
    url: str = DEFAULT_URL,
    max_new_tokens: int = 256,
    do_sample: bool | None = None,
    temperature: float | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    jpeg_quality: int = 90,
) -> Reply:
    """Ask about zero or more BGR OpenCV frames. Returns a :class:`Reply`.

    ``mode`` decides how several frames are presented, and the two are not
    interchangeable -- measured against a moving-square stimulus on the
    reference deployment:

    * ``"images"`` -- independent images. Costs one full token budget each.
      The model compares them explicitly and gets *direction* right (and
      flips its answer when the frames are reversed, which video does not).
    * ``"video"`` -- one frame sequence with ``fps`` labelling the real
      capture rate. About half the tokens, and detects *that* something moved
      just as reliably, but answered direction invariantly in testing.

    So: ``video`` for "did anything change", ``images`` for "which way".
    A single frame is sent as one image regardless of ``mode``.

    ``do_sample=False`` forces greedy decoding. Worth passing whenever an
    answer is being compared against another: the model's own config samples
    at temperature 0.7, so two identical requests otherwise differ.
    """
    text = prompt.strip()
    if not text:
        raise QwenError("empty prompt")
    if mode not in ("images", "video"):
        raise QwenError(f"mode must be 'images' or 'video', got {mode!r}")

    if frames is None:
        seq: list[np.ndarray] = []
    elif isinstance(frames, np.ndarray) and frames.ndim == 3:
        seq = [frames]  # a single frame, passed bare
    else:
        seq = list(frames)

    # Video needs at least two frames to be a sequence at all; one frame is
    # an image whatever the caller asked for.
    field_name = "video" if (mode == "video" and len(seq) > 1) else "image"

    boundary = "----mt4qwen"
    fields: list[tuple[str, str]] = [
        ("prompt", text),
        ("max_new_tokens", str(int(max_new_tokens))),
    ]
    if field_name == "video":
        fields.append(("fps", repr(float(fps))))
    if do_sample is not None:
        fields.append(("do_sample", "true" if do_sample else "false"))
    if temperature is not None:
        fields.append(("temperature", repr(float(temperature))))

    parts = [
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode("utf-8")
        for name, value in fields
    ]
    for i, frame in enumerate(seq):
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="frame{i}.jpg"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n"
            ).encode("utf-8")
            + _jpeg(frame, jpeg_quality)
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
    except TimeoutError as exc:
        # Not a URLError: a socket read that times out AFTER the connection
        # was accepted raises this bare, so without its own clause it escapes
        # every `except QwenError` in the repo and surfaces as a traceback
        # from whatever thread was asking. Observed on a first image request
        # against a freshly started service.
        raise QwenError(
            f"qwen did not answer within {timeout:g}s. The service is up -- it "
            "is generating, or compiling for a frame size it has not seen "
            "(the reference build runs with torch.compile enabled, and the "
            "first request of a new shape pays for it). Retry, or raise the "
            "timeout."
        ) from exc

    if not payload.get("ok", True):
        raise QwenError(payload.get("error", "generate failed"))

    stamps = payload.get("timestamps_s") or ()
    return Reply(
        text=str(payload.get("response", "")),
        mode=str(payload.get("mode", "text")),
        frames_sent=int(payload.get("frames_sent", len(seq))),
        images_encoded=payload.get("images_encoded"),
        temporal_groups=payload.get("temporal_groups"),
        timestamps_s=tuple(float(s) for s in stamps),
        prompt_tokens=payload.get("prompt_tokens"),
        raw=payload,
    )


def generate(
    prompt: str,
    frame: np.ndarray | None = None,
    *,
    url: str = DEFAULT_URL,
    max_new_tokens: int = 256,
    timeout: float = DEFAULT_TIMEOUT_S,
    jpeg_quality: int = 90,
) -> str:
    """Ask about a single optional frame and return just the answer text.

    The simple case, kept as its own function because most callers want a
    string. Use :func:`ask` for several frames, video, greedy decoding, or
    the metadata that says what the service really encoded.
    """
    return ask(
        prompt,
        frame,
        url=url,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        jpeg_quality=jpeg_quality,
    ).text


def _json_spans(text: str) -> Iterator[str]:
    """Candidate JSON substrings: fenced blocks first, then balanced spans.

    An unbalanced opener is stepped over rather than ending the scan, so a
    reply cut off by ``max_new_tokens`` mid-array still yields the objects
    that did complete. Bailing there instead left the whole grounding reply
    to the bare-coordinate fallback, which drops every label and description
    and then dedups distinct objects that happen to share a box.
    """
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
            i += 1
            continue
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


def _desc_of(obj: dict) -> str:
    for key in _DESC_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _regions_from_dict(obj: dict) -> list[Region]:
    label = _label_of(obj)
    description = _desc_of(obj)
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
                    out.append(Region(label=label, kind=kind, coords=nums, description=description))
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
    exactly ``[0, 433, 1000, 1000]``. So callers scale: see
    ``instruct.to_frame_pixels``, which reads a reply as 0-1000 against the
    frame's own width and height. Do not generalize that to another build;
    re-check it, which is what :meth:`Region.in_bounds` is for.
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
