"""Grounding DINO HTTP service for open-vocabulary detection.

Binds to 127.0.0.1 by default -- reach it from the MT4 host via SSH tunnel
(see scripts/start_tunnel.ps1).
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
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = os.environ.get("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base")
DEVICE = os.environ.get("GROUNDING_DINO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOST = os.environ.get("GROUNDING_DINO_HOST", "127.0.0.1")
PORT = int(os.environ.get("GROUNDING_DINO_PORT", "8765"))

_processor: Any = None
_model: Any = None


def _load() -> None:
    global _processor, _model
    if _model is not None:
        return
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
    _model.eval()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(title="Grounding DINO", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "loaded": _model is not None,
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
) -> JSONResponse:
    """Open-vocab detect. ``prompt`` is period-separated phrases, e.g. ``pen. cube.``."""
    if _model is None or _processor is None:
        raise HTTPException(503, "model not loaded")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"cannot decode image: {exc}") from exc

    text = prompt.strip()
    if not text:
        raise HTTPException(400, "empty prompt")
    # Grounding DINO expects lowercase and trailing period on each phrase.
    if not text.endswith("."):
        text = text + "."
    text = text.lower()

    inputs = _processor(images=pil, text=text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)

    # transformers API shifted across versions; try the current grounded helper
    # then fall back to the older post_process_object_detection path.
    target_sizes = torch.tensor([pil.size[::-1]], device=DEVICE)
    results: list[dict[str, Any]]
    if hasattr(_processor, "post_process_grounded_object_detection"):
        try:
            processed = _processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            processed = _processor.post_process_grounded_object_detection(
                outputs,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        boxes = processed["boxes"].detach().cpu().tolist()
        scores = processed["scores"].detach().cpu().tolist()
        labels = processed.get("labels") or processed.get("text_labels") or []
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        results = []
        for box, score, label in zip(boxes, scores, labels):
            results.append(
                {
                    "label": str(label),
                    "score": float(score),
                    "box": [float(x) for x in box],  # xyxy
                }
            )
    else:
        processed = _processor.post_process_object_detection(
            outputs, threshold=box_threshold, target_sizes=target_sizes
        )[0]
        boxes = processed["boxes"].detach().cpu().tolist()
        scores = processed["scores"].detach().cpu().tolist()
        results = [
            {"label": prompt.rstrip("."), "score": float(s), "box": [float(x) for x in b]}
            for b, s in zip(boxes, scores)
        ]

    return JSONResponse(
        {
            "ok": True,
            "prompt": text,
            "width": pil.width,
            "height": pil.height,
            "detections": results,
        }
    )


def main() -> None:
    import uvicorn

    _load()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
