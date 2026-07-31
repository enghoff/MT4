"""Qwen3-VL-4B-Instruct HTTP service (4-bit quantized).

Binds to 127.0.0.1 by default -- reach it from another host via SSH tunnel,
the same way as the Grounding DINO service (see /opt/grounding_dino/server.py).
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Any

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


app = FastAPI(title="Qwen3-VL", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "loaded": _model is not None,
        "quantization": "nf4-4bit" if (LOAD_IN_4BIT and DEVICE == "cuda") else "none",
    }


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile | None = File(None),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
) -> JSONResponse:
    """Chat with the model. ``image`` is optional -- omit it for text-only prompts."""
    if _model is None or _processor is None:
        raise HTTPException(503, "model not loaded")

    text = prompt.strip()
    if not text:
        raise HTTPException(400, "empty prompt")

    content: list[dict[str, Any]] = []
    if image is not None:
        raw = await image.read()
        if not raw:
            raise HTTPException(400, "empty image")
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"cannot decode image: {exc}") from exc
        content.append({"type": "image", "image": pil})
    content.append({"type": "text", "text": text})

    messages = [{"role": "user", "content": content}]
    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = _processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return JSONResponse(
        {
            "ok": True,
            "model": MODEL_ID,
            "prompt": text,
            "had_image": image is not None,
            "response": output_text[0] if output_text else "",
        }
    )


def main() -> None:
    import uvicorn

    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
