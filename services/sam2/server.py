"""SAM 2.1 (small) HTTP service for prompted segmentation.

Binds to 127.0.0.1 by default -- reach it from the MT4 host via SSH tunnel
(see scripts/start_tunnel.ps1), the same way as the Grounding DINO and
Qwen3-VL services.

A request is a frame plus point or box prompts; the answer is one binary mask
per prompt, run-length encoded at the frame's own resolution. The split
between :func:`_encode` and the prompt decoder is the shape of the whole
service: the image encoder is 27 ms and the decoder 7 ms, so a second prompt
on a frame already seen costs a quarter of the first one and the embedding
cache is what makes that true.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import Sam2Model, Sam2Processor

MODEL_ID = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-small")
DEVICE = os.environ.get("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOST = os.environ.get("SAM2_HOST", "127.0.0.1")
PORT = int(os.environ.get("SAM2_PORT", "8767"))

# fp16 for the weights and activations both. Measured on the reference RTX 3070
# against fp32 on the same frames, encoder time and worst-case mask agreement:
#
#   fp32  101 ms   --
#   bf16   39 ms   IoU 0.996 as shot, 0.499 on a frame darkened to 0.25x
#   fp16   33 ms   IoU 0.998 or better on every exposure tried
#
# The bf16 outlier is not a corrupt mask: its coarser mantissa flipped which of
# the three candidates won a near-tied predicted-IoU score, and the mask that
# won was a different, larger object part. That is exactly the failure a caller
# cannot see, since the score it reads still looks confident. fp16 is both the
# faster and the closer of the two here, so there is no trade to make.
DTYPE_NAME = os.environ.get("SAM2_DTYPE", "fp16" if DEVICE == "cuda" else "fp32")
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Compile the image encoder. Its input is always 1024x1024x3 -- the processor
# resizes every frame to the model's square input -- so one static shape is
# compiled once and replayed for every frame afterwards, with no recompilation.
# Measured 35 -> 27 ms. The prompt decoder stays eager on purpose: its shapes
# follow the prompt count, which varies per request, and at 7 ms it has little
# to give.
#
# "max-autotune-no-cudagraphs" rather than "reduce-overhead": CUDA graphs got
# 25 ms, 2.5 ms better, but replay reuses the output buffers, and this service
# holds encoder outputs in a cache across requests. Add the pinned memory pool
# on a card shared with qwen3-vl and it is the wrong 2.5 ms to buy.
#
# The first compile after an install costs ~100 s. With TORCHINDUCTOR_CACHE_DIR
# pointing somewhere persistent (the unit file sets it) a restart re-uses that
# work and costs 5 s. /health reports loaded:false until it finishes.
COMPILE = os.environ.get("SAM2_COMPILE", "1") not in ("0", "false", "False")
COMPILE_MODE = os.environ.get("SAM2_COMPILE_MODE", "max-autotune-no-cudagraphs")

# Encoder outputs kept keyed by image content, newest first. Three feature maps
# per frame, 8.4 MB together in fp16, so eight frames cost 67 MB of the card --
# worth it against 27 ms per repeat prompt on a frame already sent.
CACHE_FRAMES = int(os.environ.get("SAM2_CACHE_FRAMES", "8"))

_processor: Any = None
_model: Any = None
_dtype = _DTYPES[DTYPE_NAME]
_compile_status = "off"
# image_id -> (embeddings, height, width)
_cache: OrderedDict[str, tuple[list[torch.Tensor], int, int]] = OrderedDict()
_hits = 0
_misses = 0
# One request at a time on the GPU. The card also carries qwen3-vl (5.9 GB of
# 8 GB when loaded), and concurrent forwards would multiply activation memory
# against the 2.3 GB that leaves.
_gpu = asyncio.Lock()


def _load() -> None:
    global _processor, _model, _compile_status
    if _model is not None:
        return
    _processor = Sam2Processor.from_pretrained(MODEL_ID)
    started = time.perf_counter()
    _model = Sam2Model.from_pretrained(MODEL_ID, dtype=_dtype).to(DEVICE)
    _model.eval()
    print(f"loaded {MODEL_ID} as {DTYPE_NAME} in {time.perf_counter() - started:.1f}s", flush=True)

    if COMPILE and DEVICE == "cuda":
        _model.vision_encoder = torch.compile(
            _model.vision_encoder, mode=COMPILE_MODE, dynamic=False
        )
        _compile_status = f"enabled ({COMPILE_MODE})"

    started = time.perf_counter()
    _warmup()
    print(f"warmup took {time.perf_counter() - started:.1f}s", flush=True)


def _warmup() -> None:
    """Run one synthetic request end to end before serving.

    Compilation is only part of what the first request would otherwise pay for:
    cuDNN and cuBLAS pick kernels on first use, the processor's GPU resize path
    allocates, and the decoder allocates its own workspace. Measured together,
    a first real request cost 2.9 s where the ones after it cost 0.03 s. The
    frame size here matches the MT4 camera, so the mask upsample allocates for
    the shape it will actually see.
    """
    pil = Image.new("RGB", (1280, 720))
    embeddings = _encode(pil)
    inputs = _processor(
        input_points=[[[[640.0, 360.0]]]],
        input_labels=[[[1]]],
        original_sizes=[[720, 1280]],
        return_tensors="pt",
    )
    with torch.inference_mode():
        out = _model(
            image_embeddings=embeddings,
            input_points=inputs["input_points"].to(DEVICE),
            input_labels=inputs["input_labels"].to(DEVICE),
            multimask_output=True,
        )
        _processor.post_process_masks(out.pred_masks.float(), [[720, 1280]])[0].cpu()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(title="SAM 2.1", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "loaded": _model is not None,
        "dtype": DTYPE_NAME,
        "compile": _compile_status,
        "cache": {
            "frames": len(_cache),
            "capacity": CACHE_FRAMES,
            "hits": _hits,
            "misses": _misses,
        },
    }


def _image_id(raw: bytes) -> str:
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


def _decode_image(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(400, "empty image")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"cannot decode image: {exc}") from exc


def _encode(pil: Image.Image) -> list[torch.Tensor]:
    """Run the image encoder. Returns the three feature maps the decoder wants.

    ``device=DEVICE`` hands the resize and normalize to the GPU: measured
    15.6 ms on the CPU against 3.4 ms here, for a 1280x720 frame.
    """
    inputs = _processor(images=pil, return_tensors="pt", device=DEVICE)
    pixel_values = inputs["pixel_values"].to(_dtype)
    with torch.inference_mode():
        embeddings = _model.get_image_embeddings(pixel_values)
    if DEVICE == "cuda":
        # CUDA work is queued, not finished, when the call returns. Without
        # this the encoder's time lands on whichever later line first waits for
        # the GPU -- reported as 13 ms of encode and 99 ms of decode for a
        # frame whose real split is 27 and 7.
        torch.cuda.synchronize()
    return embeddings


def _cache_put(image_id: str, embeddings: list[torch.Tensor], height: int, width: int) -> None:
    _cache[image_id] = (embeddings, height, width)
    _cache.move_to_end(image_id)
    while len(_cache) > CACHE_FRAMES:
        _cache.popitem(last=False)


def _point_groups(raw: Any, field: str) -> list[list[list[float]]]:
    """Normalize ``points`` to one list of [x, y] per object.

    Accepts ``[[x, y], ...]`` -- each point its own object, which is what a
    caller pointing at N things sends -- and ``[[[x, y], [x, y]], ...]``, which
    groups several points onto one object so a second click can refine or
    subtract from the first.
    """
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, f"{field} must be a non-empty JSON array")
    if not all(isinstance(item, list) and item for item in raw):
        raise HTTPException(400, f"{field} must be [[x, y], ...] or [[[x, y], ...], ...]")
    groups = raw if isinstance(raw[0][0], list) else [[item] for item in raw]
    sizes = {len(g) for g in groups}
    if len(sizes) > 1:
        raise HTTPException(
            400, f"every object in {field} needs the same number of points, got {sorted(sizes)}"
        )
    for g in groups:
        for p in g:
            if len(p) != 2:
                raise HTTPException(400, f"{field} entries must be [x, y], got {p}")
    return groups


def _label_groups(raw: Any, groups: list[list[list[float]]]) -> list[list[int]]:
    """Foreground/background flags shaped like ``points``; all foreground by default."""
    if raw is None:
        return [[1] * len(g) for g in groups]
    if not isinstance(raw, list):
        raise HTTPException(400, "labels must be a JSON array")
    nested = bool(raw) and isinstance(raw[0], list)
    labels = [[int(v) for v in g] for g in raw] if nested else [[int(v)] for v in raw]
    if [len(g) for g in labels] != [len(g) for g in groups]:
        raise HTTPException(400, "labels must have the same shape as points")
    return labels


def _rle(mask: np.ndarray) -> list[int]:
    """Row-major run lengths, alternating, starting with a background run.

    A 1280x720 mask is 921,600 booleans; this turns the one measured here into
    3,619 runs in 0.2 ms. A leading 0 appears when the top-left pixel is inside
    the mask, so the alternation always starts on background.
    """
    flat = mask.reshape(-1)
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(bounds).astype(int).tolist()
    return runs if not flat[0] else [0] + runs


def _mask_entry(mask: np.ndarray, object_index: int, score: float) -> dict[str, Any]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    bbox = (
        [int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])] if rows.size else None
    )
    return {
        "object_index": object_index,
        "score": score,
        "area": int(mask.sum()),
        "bbox": bbox,
        "rle": _rle(mask),
    }


@app.post("/embed")
async def embed(image: UploadFile = File(...)) -> JSONResponse:
    """Encode a frame and keep it, so later /segment calls skip the encoder.

    Worth calling while the arm is still moving toward a frame's subject: the
    27 ms happens then rather than inside the request that needs the answer.
    """
    if _model is None:
        raise HTTPException(503, "model not loaded")
    raw = await image.read()
    pil = _decode_image(raw)
    image_id = _image_id(raw)

    global _hits, _misses
    started = time.perf_counter()
    async with _gpu:
        if image_id in _cache:
            _cache.move_to_end(image_id)
            _hits += 1
            cached = True
        else:
            _cache_put(image_id, _encode(pil), pil.height, pil.width)
            _misses += 1
            cached = False
    return JSONResponse(
        {
            "ok": True,
            "image_id": image_id,
            "width": pil.width,
            "height": pil.height,
            "cached": cached,
            "encode_s": round(time.perf_counter() - started, 4),
        }
    )


@app.post("/segment")
async def segment(
    image: UploadFile | None = File(None),
    image_id: str | None = Form(None),
    points: str | None = Form(None),
    labels: str | None = Form(None),
    boxes: str | None = Form(None),
    multimask: bool = Form(True),
) -> JSONResponse:
    """Segment what the prompts point at. Send an ``image``, or an ``image_id``.

    ``points`` is JSON: ``[[x, y], ...]`` for one object per point, or
    ``[[[x, y], [x, y]], ...]`` to put several points on one object. ``labels``
    matches its shape, 1 for a point on the object and 0 for one off it.
    ``boxes`` is ``[[x1, y1, x2, y2], ...]``. All coordinates are pixels in the
    frame as submitted; the processor rescales them to the model's 1024x1024
    input and the masks come back at the frame's own size.

    With ``multimask`` the model returns three candidates per object, which is
    how a single point says "the cube, the cube's face, or the whole stack" --
    they arrive sorted by the model's own predicted IoU, so a caller wanting
    one mask takes the first per ``object_index``.
    """
    if _model is None or _processor is None:
        raise HTTPException(503, "model not loaded")

    parsed: dict[str, Any] = {}
    for name, raw_json in (("points", points), ("labels", labels), ("boxes", boxes)):
        if raw_json is None or raw_json.strip() == "":
            parsed[name] = None
            continue
        try:
            parsed[name] = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"{name} is not valid JSON: {exc}") from exc
    if parsed["points"] is None and parsed["boxes"] is None:
        raise HTTPException(400, "send points, boxes, or both")

    global _hits, _misses
    started = time.perf_counter()
    encode_s = 0.0

    raw = await image.read() if image is not None else b""
    if raw:
        frame_id = _image_id(raw)
        if frame_id in _cache:
            # Decoding the JPEG costs 4 ms to learn a size the cache already
            # holds. The decode still happens under the lock below if the frame
            # is evicted between here and there.
            _, height, width = _cache[frame_id]
            pil = None
        else:
            pil = _decode_image(raw)
            height, width = pil.height, pil.width
    elif image_id:
        frame_id = image_id
        if frame_id not in _cache:
            raise HTTPException(
                404,
                f"image_id {frame_id} is not cached (capacity {CACHE_FRAMES} frames); "
                "send the image itself",
            )
        _, height, width = _cache[frame_id]
        pil = None
    else:
        raise HTTPException(400, "send an image or an image_id")

    groups = _point_groups(parsed["points"], "points") if parsed["points"] is not None else None
    prompt: dict[str, torch.Tensor] = {}
    if groups is not None:
        label_groups = _label_groups(parsed["labels"], groups)
        prompt_inputs = _processor(
            input_points=[groups],
            input_labels=[label_groups],
            original_sizes=[[height, width]],
            return_tensors="pt",
        )
        prompt["input_points"] = prompt_inputs["input_points"].to(DEVICE)
        prompt["input_labels"] = prompt_inputs["input_labels"].to(DEVICE)
    if parsed["boxes"] is not None:
        box_inputs = _processor(
            input_boxes=[parsed["boxes"]],
            original_sizes=[[height, width]],
            return_tensors="pt",
        )
        prompt["input_boxes"] = box_inputs["input_boxes"].to(DEVICE)

    async with _gpu:
        if frame_id in _cache:
            embeddings, _, _ = _cache[frame_id]
            _cache.move_to_end(frame_id)
            _hits += 1
        else:
            t0 = time.perf_counter()
            embeddings = _encode(pil if pil is not None else _decode_image(raw))
            encode_s = time.perf_counter() - t0
            _cache_put(frame_id, embeddings, height, width)
            _misses += 1

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = _model(
                image_embeddings=embeddings, multimask_output=multimask, **prompt
            )
            # Upsampling the 256x256 logits to the frame runs here rather than
            # after .cpu(): 0.1 ms on the GPU against 2-3 ms on the host.
            masks = _processor.post_process_masks(
                out.pred_masks.float(), [[height, width]]
            )[0]
            scores = out.iou_scores.float().cpu().numpy()
            masks_np = masks.cpu().numpy()
        decode_s = time.perf_counter() - t0

    # (objects, candidates, H, W) flattened object-major, which is the order
    # iou_scores comes in, so the two zip straight together.
    n_objects, n_candidates = scores.shape[1], scores.shape[2]
    masks_np = masks_np.reshape(-1, masks_np.shape[-2], masks_np.shape[-1])
    entries = [
        _mask_entry(m, i // n_candidates, float(s))
        for i, (m, s) in enumerate(zip(masks_np, scores.reshape(-1)))
    ]
    entries.sort(key=lambda e: (e["object_index"], -e["score"]))

    return JSONResponse(
        {
            "ok": True,
            "model": MODEL_ID,
            "image_id": frame_id,
            "width": width,
            "height": height,
            "objects": n_objects,
            "cached": encode_s == 0.0,
            "encode_s": round(encode_s, 4),
            "decode_s": round(decode_s, 4),
            "total_s": round(time.perf_counter() - started, 4),
            "masks": entries,
        }
    )


def main() -> None:
    import uvicorn

    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
