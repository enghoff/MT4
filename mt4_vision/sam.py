"""In-process SAM 2.1 prompted segmentation.

Loads ``facebook/sam2.1-hiera-small`` in this process on first use. Where
:mod:`mt4_vision.grounding` answers "where is the pen" with a box, this answers
"which pixels are it" with a mask, from a point or a box you already have.
:mod:`mt4_vision.locate` is where they compose: every measurement that starts
from a box prompts this module and measures the silhouette that comes back.
Cube pick/place, calibration and stacking do not touch it, and neither does
measuring from a bare pixel, which segments by distance from the local desk
colour.

The image encoder is the expensive half and does not depend on the prompt, so
a second call about a frame already seen reuses its embeddings. Setup:
``docs/SAM2.md``.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MODEL_ID = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-small")
CACHE_FRAMES = int(os.environ.get("SAM2_CACHE_FRAMES", "8"))
COMPILE_MODE = os.environ.get("SAM2_COMPILE_MODE", "max-autotune-no-cudagraphs")


class SamError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mask:
    """One candidate silhouette for one prompt, in frame pixels.

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


def decode_rle(runs: Sequence[int], height: int, width: int) -> np.ndarray:
    """Run lengths back into a boolean mask.

    Row-major runs that alternate background, foreground, background, ...
    always starting on background -- a leading zero-length run appears when
    the top-left pixel is inside the mask.
    """
    values = np.arange(len(runs)) % 2 == 1
    return np.repeat(values, runs).reshape(height, width)


def best_per_object(masks: Sequence[Mask]) -> list[Mask]:
    """The highest-scoring candidate for each prompt, in prompt order."""
    best: dict[int, Mask] = {}
    for m in masks:
        if m.object_index not in best or m.score > best[m.object_index].score:
            best[m.object_index] = m
    return [best[i] for i in sorted(best)]


def _frame_id(frame: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=12).hexdigest()


def _point_groups(raw: Sequence[Sequence[float]], field: str) -> list[list[list[float]]]:
    """Normalize ``points`` to one list of [x, y] per object.

    Accepts ``[[x, y], ...]`` -- each point its own object -- and
    ``[[[x, y], [x, y]], ...]``, which groups several points onto one object.
    """
    if not raw:
        raise SamError(f"{field} must be a non-empty list")
    if not all(isinstance(item, (list, tuple)) and item for item in raw):
        raise SamError(f"{field} must be [[x, y], ...] or [[[x, y], ...], ...]")
    groups = raw if isinstance(raw[0][0], (list, tuple)) else [[list(item)] for item in raw]
    groups = [[[float(v) for v in p] for p in g] for g in groups]
    sizes = {len(g) for g in groups}
    if len(sizes) > 1:
        raise SamError(
            f"every object in {field} needs the same number of points, got {sorted(sizes)}"
        )
    for g in groups:
        for p in g:
            if len(p) != 2:
                raise SamError(f"{field} entries must be [x, y], got {p}")
    return groups


def _label_groups(
    raw: Sequence[int] | Sequence[Sequence[int]] | None,
    groups: list[list[list[float]]],
) -> list[list[int]]:
    if raw is None:
        return [[1] * len(g) for g in groups]
    nested = bool(raw) and isinstance(raw[0], (list, tuple))
    labels = [[int(v) for v in g] for g in raw] if nested else [[int(v)] for v in raw]  # type: ignore[arg-type]
    if [len(g) for g in labels] != [len(g) for g in groups]:
        raise SamError("labels must have the same shape as points")
    return labels


class _SamEngine:
    """Lazy Sam2Model holder with an embedding cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = ""
        self._dtype: Any = None
        self._dtype_name = ""
        self._compile_status = "off"
        self._cache: OrderedDict[str, tuple[list[Any], int, int]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def health(self) -> dict[str, Any]:
        cuda = False
        if self._torch is not None:
            cuda = bool(self._torch.cuda.is_available())
        elif self._device == "cuda":
            cuda = True
        return {
            "ok": True,
            "model": MODEL_ID,
            "device": self._device or self._resolve_device(),
            "cuda": cuda,
            "loaded": self.loaded,
            "dtype": self._dtype_name or self._resolve_dtype_name(self._resolve_device()),
            "compile": self._compile_status,
            "cache": {
                "frames": len(self._cache),
                "capacity": CACHE_FRAMES,
                "hits": self._hits,
                "misses": self._misses,
            },
        }

    @staticmethod
    def _resolve_device() -> str:
        configured = os.environ.get("SAM2_DEVICE")
        if configured:
            return configured
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype_name(device: str) -> str:
        return os.environ.get(
            "SAM2_DTYPE", "fp16" if device == "cuda" else "fp32"
        )

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from PIL import Image
                from transformers import Sam2Model, Sam2Processor
            except ImportError as exc:
                raise SamError(
                    "SAM 2.1 needs torch, transformers>=4.57, torchvision, "
                    "accelerate and pillow; pip install -r requirements-sam.txt"
                ) from exc

            device = self._resolve_device()
            dtype_name = self._resolve_dtype_name(device)
            dtypes = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            if dtype_name not in dtypes:
                raise SamError(f"SAM2_DTYPE must be fp16, bf16 or fp32, got {dtype_name!r}")
            dtype = dtypes[dtype_name]

            # fp16 for weights and activations on CUDA. Measured on the
            # reference RTX 3070 against fp32: fp16 33 ms encoder / IoU 0.998;
            # bf16 39 ms and a 0.499 IoU outlier on a darkened frame when the
            # coarser mantissa flipped which multimask candidate won.
            processor = Sam2Processor.from_pretrained(MODEL_ID)
            started = time.perf_counter()
            model = Sam2Model.from_pretrained(MODEL_ID, dtype=dtype).to(device)
            model.eval()
            print(
                f"loaded {MODEL_ID} as {dtype_name} on {device} "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )

            compile_status = "off"
            compile_on = os.environ.get("SAM2_COMPILE", "1") not in ("0", "false", "False")
            if compile_on and device == "cuda":
                # Encoder input is always 1024x1024x3 after the processor, so
                # one static shape compiles once. Decoder stays eager: prompt
                # shapes vary and at ~7 ms it has little to give.
                # max-autotune-no-cudagraphs: CUDA graphs replay reuses output
                # buffers, which conflicts with caching encoder outputs.
                model.vision_encoder = torch.compile(
                    model.vision_encoder, mode=COMPILE_MODE, dynamic=False
                )
                compile_status = f"enabled ({COMPILE_MODE})"

            self._torch = torch
            self._processor = processor
            self._model = model
            self._device = device
            self._dtype = dtype
            self._dtype_name = dtype_name
            self._compile_status = compile_status

            started = time.perf_counter()
            self._warmup(Image)
            print(f"sam warmup took {time.perf_counter() - started:.1f}s", flush=True)

    def _warmup(self, Image: Any) -> None:
        """One synthetic request so cuDNN / allocations land before real work."""
        pil = Image.new("RGB", (1280, 720))
        embeddings = self._encode(pil)
        inputs = self._processor(
            input_points=[[[[640.0, 360.0]]]],
            input_labels=[[[1]]],
            original_sizes=[[720, 1280]],
            return_tensors="pt",
        )
        torch = self._torch
        with torch.inference_mode():
            out = self._model(
                image_embeddings=embeddings,
                input_points=inputs["input_points"].to(self._device),
                input_labels=inputs["input_labels"].to(self._device),
                multimask_output=True,
            )
            self._processor.post_process_masks(
                out.pred_masks.float(), [[720, 1280]]
            )[0].cpu()

    def _encode(self, pil: Any) -> list[Any]:
        torch = self._torch
        inputs = self._processor(images=pil, return_tensors="pt", device=self._device)
        pixel_values = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            embeddings = self._model.get_image_embeddings(pixel_values)
        if self._device == "cuda":
            # CUDA work is queued when the call returns; sync so encode time
            # does not land on the next wait.
            torch.cuda.synchronize()
        return embeddings

    def _cache_put(self, image_id: str, embeddings: list[Any], height: int, width: int) -> None:
        self._cache[image_id] = (embeddings, height, width)
        self._cache.move_to_end(image_id)
        while len(self._cache) > CACHE_FRAMES:
            self._cache.popitem(last=False)

    def _bgr_to_pil(self, frame: np.ndarray) -> Any:
        from PIL import Image

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise SamError(f"frame must be HxWx3 BGR, got shape {frame.shape}")
        rgb = frame[:, :, ::-1]
        return Image.fromarray(np.ascontiguousarray(rgb))

    def embed(self, frame: np.ndarray) -> str:
        self.ensure_loaded()
        image_id = _frame_id(frame)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        with self._lock:
            if image_id in self._cache:
                self._cache.move_to_end(image_id)
                self._hits += 1
                return image_id
            embeddings = self._encode(self._bgr_to_pil(frame))
            self._cache_put(image_id, embeddings, height, width)
            self._misses += 1
        return image_id

    def segment(
        self,
        frame: np.ndarray | None,
        *,
        points: Sequence[Sequence[float]] | None = None,
        labels: Sequence[int] | Sequence[Sequence[int]] | None = None,
        boxes: Sequence[Sequence[float]] | None = None,
        multimask: bool = True,
        image_id: str | None = None,
    ) -> list[Mask]:
        if points is None and boxes is None:
            raise SamError("segment needs points, boxes, or both")
        if frame is None and not image_id:
            raise SamError("segment needs a frame or an image_id")

        self.ensure_loaded()
        torch = self._torch

        groups = _point_groups(points, "points") if points is not None else None
        box_list = [[float(v) for v in b] for b in boxes] if boxes is not None else None

        with self._lock:
            if frame is not None:
                fid = _frame_id(frame)
                height, width = int(frame.shape[0]), int(frame.shape[1])
                if fid in self._cache:
                    embeddings, _, _ = self._cache[fid]
                    self._cache.move_to_end(fid)
                    self._hits += 1
                else:
                    embeddings = self._encode(self._bgr_to_pil(frame))
                    self._cache_put(fid, embeddings, height, width)
                    self._misses += 1
            else:
                assert image_id is not None
                if image_id not in self._cache:
                    raise SamError(
                        f"image_id {image_id} is not cached "
                        f"(capacity {CACHE_FRAMES} frames); pass the frame itself"
                    )
                embeddings, height, width = self._cache[image_id]
                self._cache.move_to_end(image_id)
                self._hits += 1

            prompt: dict[str, Any] = {}
            if groups is not None:
                label_groups = _label_groups(labels, groups)
                prompt_inputs = self._processor(
                    input_points=[groups],
                    input_labels=[label_groups],
                    original_sizes=[[height, width]],
                    return_tensors="pt",
                )
                prompt["input_points"] = prompt_inputs["input_points"].to(self._device)
                prompt["input_labels"] = prompt_inputs["input_labels"].to(self._device)
            if box_list is not None:
                box_inputs = self._processor(
                    input_boxes=[box_list],
                    original_sizes=[[height, width]],
                    return_tensors="pt",
                )
                prompt["input_boxes"] = box_inputs["input_boxes"].to(self._device)

            with torch.inference_mode():
                out = self._model(
                    image_embeddings=embeddings, multimask_output=multimask, **prompt
                )
                masks = self._processor.post_process_masks(
                    out.pred_masks.float(), [[height, width]]
                )[0]
                scores = out.iou_scores.float().cpu().numpy()
                masks_np = masks.cpu().numpy()

        n_candidates = scores.shape[2]
        masks_np = masks_np.reshape(-1, masks_np.shape[-2], masks_np.shape[-1])
        scores_flat = scores.reshape(-1)
        out_masks: list[Mask] = []
        for i, (m, s) in enumerate(zip(masks_np, scores_flat)):
            binary = m > 0.0 if m.dtype != bool else m
            rows = np.flatnonzero(binary.any(axis=1))
            cols = np.flatnonzero(binary.any(axis=0))
            bbox = (
                (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
                if rows.size
                else None
            )
            out_masks.append(
                Mask(
                    object_index=i // n_candidates,
                    score=float(s),
                    area=int(binary.sum()),
                    bbox=bbox,
                    mask=np.asarray(binary, dtype=bool),
                )
            )
        out_masks.sort(key=lambda m: (m.object_index, -m.score))
        return out_masks


_ENGINE = _SamEngine()


def health() -> dict:
    """Model / device / cache status. Loads nothing; ``loaded`` is false until first use."""
    return _ENGINE.health()


def embed(frame: np.ndarray) -> str:
    """Encode a frame now and return its ``image_id``, for prompting later.

    Worth calling on a frame the arm is about to be asked about while it is
    still moving: the encoder runs then rather than inside the call that needs
    the answer.
    """
    return _ENGINE.embed(frame)


def segment(
    frame: np.ndarray | None,
    *,
    points: Sequence[Sequence[float]] | None = None,
    labels: Sequence[int] | Sequence[Sequence[int]] | None = None,
    boxes: Sequence[Sequence[float]] | None = None,
    multimask: bool = True,
    image_id: str | None = None,
) -> list[Mask]:
    """Segment what the prompts point at in a BGR OpenCV frame.

    ``points`` is ``[[x, y], ...]``, one object per point, or
    ``[[[x, y], [x, y]], ...]`` to put several points on one object -- which
    is how a second point refines the first, with ``labels`` 1 for a point on
    the object and 0 for one off it. ``boxes`` is ``[[x1, y1, x2, y2], ...]``,
    the shape a Grounding DINO detection already has.

    Pass ``frame=None`` with an ``image_id`` from :func:`embed` to prompt a
    frame still held in the embedding cache. That raises :class:`SamError`
    once the frame has aged out, so keep the frame if you may need it again.

    Returns every candidate, best-first within each object.
    """
    return _ENGINE.segment(
        frame,
        points=points,
        labels=labels,
        boxes=boxes,
        multimask=multimask,
        image_id=image_id,
    )
