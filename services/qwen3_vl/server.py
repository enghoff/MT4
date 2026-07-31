"""Qwen3-VL-4B-Instruct HTTP service (4-bit quantized).

Binds to 127.0.0.1 by default -- reach it from another host via SSH tunnel,
the same way as the Grounding DINO service (see /opt/grounding_dino/server.py).

Accepts a single image, several images, or a frame sequence as video. Those
three are genuinely different to the model, not just packaging -- see
:func:`_build_video_inputs` for why video costs about half the vision tokens
of the same frames sent as images, and why it needs its own code path to
avoid silently discarding most of them.
"""

from __future__ import annotations

import io
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL_ID = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
DEVICE = os.environ.get("QWEN_VL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOST = os.environ.get("QWEN_VL_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_VL_PORT", "8766"))
LOAD_IN_4BIT = os.environ.get("QWEN_VL_LOAD_IN_4BIT", "1") not in ("0", "false", "False")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "256"))
DEFAULT_FPS = float(os.environ.get("QWEN_VL_DEFAULT_FPS", "2.0"))

# The vision encoder pairs adjacent frames into one temporal patch (config
# vision_config.temporal_patch_size), which is exactly why video costs about
# half per frame of the same frames sent as separate images.
TEMPORAL_PATCH_SIZE = 2

_TIMESTAMP_RE = re.compile(r"<([0-9.]+) seconds>")

_processor: Any = None
_model: Any = None


def _load() -> None:
    global _processor, _model
    if _model is not None:
        return
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    quant_config = None
    if LOAD_IN_4BIT and DEVICE == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto" if DEVICE == "cuda" else None,
        quantization_config=quant_config,
    )
    if DEVICE != "cuda":
        _model = _model.to(DEVICE)
    _model.eval()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(title="Qwen3-VL", version="2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "loaded": _model is not None,
        "quantization": "nf4-4bit" if (LOAD_IN_4BIT and DEVICE == "cuda") else "none",
        "modes": ["text", "image", "images", "video"],
        "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        "default_fps": DEFAULT_FPS,
    }


async def _decode(uploads: list[UploadFile], field: str) -> list[Image.Image]:
    out: list[Image.Image] = []
    for i, upload in enumerate(uploads):
        raw = await upload.read()
        if not raw:
            raise HTTPException(400, f"empty {field} at position {i}")
        try:
            out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"cannot decode {field} at position {i}: {exc}") from exc
    return out


def _build_image_inputs(prompt: str, images: list[Image.Image]):
    """Zero or more images as independent images -- the chat-template path.

    Each image gets its own vision block and its own full token budget, so
    cost grows linearly with count (measured: 880 vision tokens per 1280x720
    frame). Nothing temporal is encoded; the model has to infer any ordering
    from position and from whatever the prompt says.
    """
    content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt})
    return _processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def _build_video_inputs(prompt: str, frames: list[Image.Image], fps: float):
    """Frames as video, with every frame kept and honest timestamps.

    Two processor defaults have to be overridden or this path lies about its
    own input:

    * ``do_sample_frames`` defaults True, and the video processor then
      *resamples* what it was handed as though it were raw footage needing
      decimation to its default 2 fps. With no metadata it assumes the source
      was 24 fps, so a 6-frame request collapses to the ``min_frames=4``
      floor -- measured, 5/6/8/9 frames all produce an identical two-group
      tensor. Frames are dropped silently and the answer still reads fine.
    * Timestamps are baked into the prompt as ``<N seconds>`` markers, one per
      temporal group. Derived from that same assumed 24 fps they are wrong by
      the ratio of real to assumed rate (0.2 s reported for frames actually
      spanning 2.5 s), which quietly corrupts any "when did it happen" answer.

    ``video_metadata`` fixes the timestamps but is only honoured on a direct
    processor call -- through ``apply_chat_template`` it is ignored, or raises
    -- so the text and the tensors are built in two steps here.
    """
    text = _processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": prompt},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    arrays = [np.asarray(f) for f in frames]
    n = len(arrays)
    metadata = [{
        "fps": fps,
        "total_num_frames": n,
        "duration": n / fps if fps > 0 else float(n),
        "frames_indices": list(range(n)),
        "video_backend": "custom",
    }]
    return _processor(
        text=[text],
        videos=[arrays],
        return_tensors="pt",
        do_sample_frames=False,
        video_metadata=metadata,
    )


def _consumed(inputs) -> dict[str, Any]:
    """What the model actually got: groups, and the timestamps it was shown.

    Reported on every response because the failure this path had was silent
    -- a request whose frames were mostly discarded still returned a fluent
    answer. A caller that sent 6 frames can now check it.
    """
    info: dict[str, Any] = {}
    grid = inputs.get("video_grid_thw")
    if grid is not None:
        info["temporal_groups"] = int(grid[0][0])
    grid_i = inputs.get("image_grid_thw")
    if grid_i is not None:
        info["images_encoded"] = int(grid_i.shape[0])
    try:
        decoded = _processor.tokenizer.decode(inputs["input_ids"][0].tolist())
        stamps = [float(s) for s in _TIMESTAMP_RE.findall(decoded)]
        if stamps:
            info["timestamps_s"] = stamps
    except Exception:  # noqa: BLE001 -- diagnostics must never fail a request
        pass
    return info


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: list[UploadFile] | None = File(None),
    video: list[UploadFile] | None = File(None),
    fps: float = Form(DEFAULT_FPS),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    do_sample: bool | None = Form(None),
    temperature: float | None = Form(None),
) -> JSONResponse:
    """Chat with the model.

    Send ``image`` zero or more times for independent images, or ``video``
    two or more times for a frame sequence (with ``fps`` describing the real
    capture rate, which is what makes the timestamps meaningful). Omit both
    for a text-only prompt. ``image`` and ``video`` are mutually exclusive.

    ``do_sample=false`` gives greedy, reproducible output; the model's own
    generation config defaults to sampling at temperature 0.7, so two
    identical requests otherwise return different answers.
    """
    if _model is None or _processor is None:
        raise HTTPException(503, "model not loaded")

    text = prompt.strip()
    if not text:
        raise HTTPException(400, "empty prompt")

    images = [f for f in (image or []) if f is not None]
    frames = [f for f in (video or []) if f is not None]
    if images and frames:
        raise HTTPException(400, "send either image or video, not both")

    if frames:
        if len(frames) < 2:
            raise HTTPException(400, "video needs at least 2 frames; use image for one")
        if fps <= 0:
            raise HTTPException(400, "fps must be > 0")
        decoded = await _decode(frames, "video frame")
        sizes = {im.size for im in decoded}
        if len(sizes) > 1:
            raise HTTPException(
                400, f"video frames must all be the same size, got {sorted(sizes)}"
            )
        inputs = _build_video_inputs(text, decoded, fps)
        mode, n_frames = "video", len(decoded)
    elif images:
        decoded = await _decode(images, "image")
        inputs = _build_image_inputs(text, decoded)
        mode, n_frames = ("images" if len(decoded) > 1 else "image"), len(decoded)
    else:
        inputs = _build_image_inputs(text, [])
        mode, n_frames = "text", 0

    consumed = _consumed(inputs)
    inputs = inputs.to(_model.device)

    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if do_sample is not None:
        gen_kwargs["do_sample"] = do_sample
        if not do_sample:
            # Left set, the config's sampling params warn and mean nothing
            # under greedy decoding.
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
        gen_kwargs.setdefault("do_sample", True)

    with torch.no_grad():
        generated_ids = _model.generate(**inputs, **gen_kwargs)
    trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = _processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    payload: dict[str, Any] = {
        "ok": True,
        "model": MODEL_ID,
        "prompt": text,
        "mode": mode,
        "had_image": mode != "text",
        "frames_sent": n_frames,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "response": output_text[0] if output_text else "",
    }
    if mode == "video":
        payload["fps"] = fps
    payload.update(consumed)
    return JSONResponse(payload)


def main() -> None:
    import uvicorn

    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
