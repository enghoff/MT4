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

import asyncio
import io
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    CompileConfig,
    LogitsProcessor,
    Qwen3VLForConditionalGeneration,
    StaticCache,
    TorchAoConfig,
)

MODEL_ID = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
DEVICE = os.environ.get("QWEN_VL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOST = os.environ.get("QWEN_VL_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN_VL_PORT", "8766"))
LOAD_IN_4BIT = os.environ.get("QWEN_VL_LOAD_IN_4BIT", "1") not in ("0", "false", "False")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "256"))
DEFAULT_FPS = float(os.environ.get("QWEN_VL_DEFAULT_FPS", "2.0"))

# Which 4-bit backend. Both fit the card; they are not close on speed, because
# decode at batch 1 is bound by how fast weights can be turned back into
# something a matmul accepts:
#
#   nf4  -- bitsandbytes. Dequantizes into bf16 on every matmul. 34 tok/s
#           compiled, 6.1GB, ~25s to load.
#   int4 -- torchao weight-only, tinygemm. 43 tok/s compiled, 5.0GB, 4s to load.
#
# The gap only exists once the decode step is compiled: uncompiled, the two are
# 11.4 and 12.2 tok/s, because per-token dispatch overhead swamps any difference
# between the kernels. Which is the useful lesson -- the quantizer is worth
# about +26% on top of CUDA graphs and nothing at all without them.
#
# int4 is also the only one of the two transformers will compile without being
# forced (TorchAoHfQuantizer.is_compileable is True), so it retires the
# FORCE_COMPILE override rather than merely outrunning it.
QUANT = os.environ.get("QWEN_VL_QUANT", "int4" if LOAD_IN_4BIT else "none")
INT4_GROUP = int(os.environ.get("QWEN_VL_INT4_GROUP", "128"))
# torchao 0.17 defaults to the "plain" packing format, which targets newer
# architectures. "tile_packed_to_4d" is the TensorCore-tiled tinygemm layout
# and is the one that has a fast kernel on this sm86 card.
INT4_PACKING = os.environ.get("QWEN_VL_INT4_PACKING", "tile_packed_to_4d")
# Modules to leave in bf16, comma-separated. int4 is a decode optimization and a
# prefill pessimization -- weight-only int4 has to unpack for the wide GEMMs
# that prefill is made of, where bf16 just multiplies. Quantizing the vision
# tower therefore costs prefill (measured: 0.45s -> 0.90s for one frame) to
# speed up a token loop the vision tower takes no part in. Excluding it buys
# that back for ~0.6GB.
INT4_SKIP = [m for m in os.environ.get("QWEN_VL_INT4_SKIP", "visual").split(",") if m]

# Decode, not prefill, is what a request's time goes on: measured here, one
# 1280x720 frame prefills in 0.65s (912 prompt tokens) and then decodes at
# 82 ms/token. A 350-token /objects reply is therefore ~29s, essentially all
# of it in the token loop. The two knobs below target that loop.
#
# A reused StaticCache instead of the default DynamicCache: dynamic reallocates
# and torch.cat's the whole K/V of all 36 layers on every single token, static
# writes in place into buffers allocated once at load. It is also the
# precondition for compilation -- transformers refuses to compile unless the
# cache is a compileable type.
#
# Size it tightly. Attention runs over the WHOLE window every step regardless
# of how much of it is real, so an oversized cache is pure waste: measured on
# the same 912-token request, 4096 gave 23.6 tok/s, 2048 gave 34 and 1664 gave
# 43. 1664 is the smallest that still fits one image (912 prompt tokens) plus a
# full 700-token reply, which is the harness's normal shape.
# A request that would overrun it falls back to the dynamic cache and therefore
# to eager -- correct, just slow -- which is the case for `images` mode with
# several frames (6 measured at 5310 prompt tokens) and for the before/after
# pair that an explicit `/watch <question>` sends. Raise it if that is your
# normal workload, and expect the single-frame case to get slower for it.
CACHE_LEN = int(os.environ.get("QWEN_VL_CACHE_LEN", "1664"))
# Compile the decode step (transformers wraps only that; prefill stays eager,
# which is what keeps a variable-size vision input from recompiling per image).
COMPILE = os.environ.get("QWEN_VL_COMPILE", "0") not in ("0", "false", "False")
# transformers 5.14 gates compilation on `hf_quantizer.is_compileable`, and
# bitsandbytes reports False (as do GPTQ, HQQ and every other quantizer here
# except torchao and fp8), so `COMPILE=1` alone is a no-op that only logs a
# warning. Measured, that gate is conservative for this combination
# (transformers 5.14.1 / bnb 0.50 / torch 2.13 / sm86): 11.4 -> 31.5 tok/s,
# 48% -> 86% GPU utilization, coherent output, and byte-identical replies
# across repeated passes within one process.
#
# What it does NOT preserve is byte-identical replies against the *uncompiled*
# deployment -- but neither does the static cache on its own, which is the
# actual source of that drift (measured: static-cache-without-compile diverges
# from dynamic-cache-eager on 3 of 4 probe prompts). Both are the ordinary
# consequence of changed floating-point reduction order: one near-tied argmax
# goes the other way and the rest of the sequence follows. Greedy stays
# reproducible run-to-run, which is what /freeze and /again rely on; it is only
# comparisons against answers recorded before this change that are affected.
FORCE_COMPILE = os.environ.get("QWEN_VL_FORCE_COMPILE", "0") not in ("0", "false", "False")

# The vision encoder pairs adjacent frames into one temporal patch (config
# vision_config.temporal_patch_size), which is exactly why video costs about
# half per frame of the same frames sent as separate images.
TEMPORAL_PATCH_SIZE = 2

_TIMESTAMP_RE = re.compile(r"<([0-9.]+) seconds>")

_processor: Any = None
_model: Any = None
_cache: Any = None
_compile_status = "off"
_quantization = "unknown"
# One request at a time on the GPU. Already true by accident -- `generate` is
# blocking and never awaits, so it holds the event loop for its whole duration
# -- but the reused StaticCache makes it load-bearing rather than incidental:
# two overlapping generations would write into the same K/V buffers.
_gpu = asyncio.Lock()


def _enable_compile() -> str:
    """Turn on transformers' decode-step compilation. Returns a status string.

    ``CompileConfig(mode="reduce-overhead")`` is CUDA graphs: the ~36 layers'
    worth of tiny per-token kernel launches get captured once and replayed,
    instead of being dispatched from Python 700 times over.

    The quantizer gate is the catch. ``_valid_auto_compile_criteria`` silently
    declines to compile when ``hf_quantizer.is_compileable`` is False and only
    warns, so a config that looks enabled can do nothing at all. Bitsandbytes
    reports False -- deliberately, its 4-bit dequant path is not graph-safe as
    far as transformers is concerned. ``FORCE_COMPILE`` patches the property to
    find out whether that is a real limitation or a conservative default; if it
    is real, the failure mode to watch for is not a crash but *wrong output*,
    so compare a greedy answer against the uncompiled one before trusting it.
    """
    _model.generation_config.compile_config = CompileConfig(mode="reduce-overhead")
    quantizer = getattr(_model, "hf_quantizer", None)
    if quantizer is None or quantizer.is_compileable:
        return "enabled"
    name = type(quantizer).__name__
    if not FORCE_COMPILE:
        _model.generation_config.compile_config = None
        return f"skipped: {name}.is_compileable is False (set QWEN_VL_FORCE_COMPILE=1 to override)"
    type(quantizer).is_compileable = property(lambda self: True)
    return f"forced: overrode {name}.is_compileable=False"


def _quant_config() -> tuple[Any, str]:
    """The 4-bit backend to load with, plus the label /health reports.

    Weight-only int4 leaves activations in bf16 and only the weights packed,
    which is the right trade at batch 1: decode reads every weight once per
    token and does almost no arithmetic per byte read, so the format the weights
    are stored in is the whole ballgame.
    """
    if QUANT == "none" or DEVICE != "cuda":
        return None, "none"
    if QUANT == "nf4":
        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            "nf4-4bit",
        )
    if QUANT == "int4":
        from torchao.quantization import Int4WeightOnlyConfig

        label = f"torchao-int4 ({INT4_PACKING}, group {INT4_GROUP})"
        if INT4_SKIP:
            label += f", bf16: {'+'.join(INT4_SKIP)}"
        return (
            TorchAoConfig(
                quant_type=Int4WeightOnlyConfig(
                    group_size=INT4_GROUP, int4_packing_format=INT4_PACKING
                ),
                modules_to_not_convert=INT4_SKIP or None,
            ),
            label,
        )
    raise ValueError(f"QWEN_VL_QUANT must be nf4, int4 or none, got {QUANT!r}")


def _load() -> None:
    global _processor, _model, _cache, _compile_status, _quantization
    if _model is not None:
        return
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    quant_config, _quantization = _quant_config()
    started = time.perf_counter()
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto" if DEVICE == "cuda" else None,
        quantization_config=quant_config,
    )
    if DEVICE != "cuda":
        _model = _model.to(DEVICE)
    _model.eval()
    print(f"loaded {_quantization} in {time.perf_counter() - started:.1f}s", flush=True)

    if DEVICE == "cuda" and CACHE_LEN > 0:
        # Buffers are allocated lazily from the first key_states, so no device
        # or dtype is needed here; one object is built now and reset between
        # requests rather than reallocated per request. Requests that would
        # overrun CACHE_LEN fall back to the dynamic cache in `generate`.
        _cache = StaticCache(config=_model.config, max_cache_len=CACHE_LEN)
    if COMPILE:
        _compile_status = _enable_compile()
        # Capture the graph now. Without this the first real request pays the
        # whole compile (a minute or more) and reads as a hung service.
        if _model.generation_config.compile_config is not None:
            _warmup()


def _warmup(tokens: int = 8) -> None:
    """Run one tiny text-only generation to capture the decode graph at load.

    Text-only is enough: prefill stays eager and is the only part whose shape
    depends on the image, so the graph being captured here is the same one
    every subsequent request replays.
    """
    inputs = _build_image_inputs("warmup", []).to(_model.device)
    kwargs: dict[str, Any] = {"max_new_tokens": tokens, "do_sample": False}
    if _cache is not None:
        _cache.reset()
        kwargs["past_key_values"] = _cache
    started = time.perf_counter()
    with torch.no_grad():
        _model.generate(**inputs, **kwargs)
    print(f"warmup/compile took {time.perf_counter() - started:.1f}s", flush=True)


class _FirstTokenClock(LogitsProcessor):
    """Timestamps the first scored token, splitting prefill from decode.

    ``generate`` reports one wall time, and the two halves behind it want
    opposite fixes -- prefill is a single batched forward already close to the
    card's roofline, decode is N sequential ones bound by weight-dequant
    bandwidth. Measured here they are 0.65s and 82 ms/token, so a change that
    improves the total by 5% may have halved one and done nothing to the other.
    Reported per request so that stays visible instead of being re-derived by
    subtracting two runs at different token caps.
    """

    def __init__(self) -> None:
        self.first: float | None = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.first is None:
            self.first = time.perf_counter()
        return scores


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
        "quantization": _quantization,
        "modes": ["text", "image", "images", "video"],
        "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        "default_fps": DEFAULT_FPS,
        # Whether the two decode-loop knobs are actually in force, not merely
        # requested -- `compile` reports the gate's verdict, which is the whole
        # reason it is surfaced here.
        "cache": "static" if _cache is not None else "dynamic",
        "cache_len": CACHE_LEN if _cache is not None else None,
        "compile": _compile_status,
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

    prompt_len = int(inputs["input_ids"].shape[-1])
    # A static cache is a fixed allocation, so a request that would overrun it
    # falls back rather than failing. Reported in the payload because the fast
    # path silently not applying is exactly the kind of thing that turns a
    # perf measurement into a wrong conclusion.
    cache_used = "dynamic"
    if _cache is not None and prompt_len + max_new_tokens <= CACHE_LEN:
        _cache.reset()
        gen_kwargs["past_key_values"] = _cache
        cache_used = "static"

    clock = _FirstTokenClock()
    started = time.perf_counter()
    async with _gpu:
        with torch.no_grad():
            generated_ids = _model.generate(
                **inputs, logits_processor=[clock], **gen_kwargs
            )
    finished = time.perf_counter()
    trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = _processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    completion_tokens = int(generated_ids.shape[-1]) - prompt_len
    payload: dict[str, Any] = {
        "ok": True,
        "model": MODEL_ID,
        "prompt": text,
        "mode": mode,
        "had_image": mode != "text",
        "frames_sent": n_frames,
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_tokens,
        "generate_s": round(finished - started, 3),
        "cache": cache_used,
        "response": output_text[0] if output_text else "",
    }
    if clock.first is not None:
        # prefill_s covers the vision encoder and the one batched forward;
        # decode_s covers every token after the first.
        payload["prefill_s"] = round(clock.first - started, 3)
        payload["decode_s"] = round(finished - clock.first, 3)
        if completion_tokens > 1 and finished > clock.first:
            payload["decode_tok_s"] = round(
                (completion_tokens - 1) / (finished - clock.first), 1
            )
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
