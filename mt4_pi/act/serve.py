"""Serve a trained ACT checkpoint over HTTP.

    python -m mt4_pi.act.serve --checkpoint /path/to/pretrained_model [--port 8100]

Runs wherever torch and `lerobot` are installed -- MEDIA, or the GPU box
itself for a quick check. Needs no MT4 code beyond `schema.py`.

Plain HTTP with a base64 JPEG rather than the WebSocket protocol the pi0.5
server speaks. That protocol exists to keep a 300 ms JAX inference alive
across keepalive timeouts; ACT infers in ~10-20 ms, and the MT4's serial link
costs ~204 ms per queued waypoint, so transport overhead is not the
bottleneck and is not worth a dependency. `http.server` from the stdlib is
sufficient and gives the client one less thing to install.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8100
SIDECAR_NAME = "mt4_act_meta.json"


def _load_policy(checkpoint: Path, device: str):
    """The policy plus its normalisation processors.

    lerobot 0.4.x moved normalisation OUT of the policy and into a pair of
    pipelines saved beside the weights (`policy_preprocessor*` /
    `policy_postprocessor*`). `ACTPolicy` owns no normalize modules and
    `predict_action_chunk` applies none, so calling it directly feeds raw
    tensors to a model trained on normalised ones and reads normalised
    outputs back as if they were radians.

    That is not a subtle degradation. Measured on a 20k-step checkpoint,
    skipping the processors put single-step error at 21 deg against a 0.65
    deg "don't move" baseline -- indistinguishable from a model that had
    learned nothing. Applying them gives 3.2 deg and beats the baseline 2.8x
    over the full chunk. Anything that loads an ACT checkpoint must load
    these too.
    """
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.eval()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    torch.set_grad_enabled(False)
    return policy, preprocessor, postprocessor


def _predict_chunk(policy, pre, post, batch: dict, n_steps: int) -> np.ndarray:
    """Full action chunk as (T, action_dim), across lerobot versions.

    `pre`/`post` are the normalisation pipelines -- see `_load_policy`. They
    are mandatory, not optional: without them the model is fed and read in
    the wrong units.
    """
    import torch

    if hasattr(policy, "predict_action_chunk"):
        with torch.inference_mode():
            chunk = post(policy.predict_action_chunk(pre(batch)))
        return chunk[0].float().cpu().numpy()

    # Older trees only expose select_action, which serves one row at a time
    # out of a queue refilled by a single forward pass -- so calling it
    # repeatedly after a reset yields that same chunk.
    policy.reset()
    rows = []
    with torch.inference_mode():
        for _ in range(n_steps):
            rows.append(post(policy.select_action(pre(batch)))[0].float().cpu().numpy())
    return np.stack(rows)


class _Handler(BaseHTTPRequestHandler):
    policy = None
    pre = None
    post = None
    meta: dict = {}
    device = "cpu"
    camera_key = "observation.images.overhead"
    n_steps = 8

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter than the stdlib default
        logger.debug(fmt, *args)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/meta":
            self._send(200, self.meta)
        else:
            self._send(404, {"error": f"no such path {self.path}"})

    def do_POST(self):
        if self.path.rstrip("/") != "/infer":
            self._send(404, {"error": f"no such path {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            t0 = time.perf_counter()
            actions = self._infer(request)
            self._send(
                200,
                {"actions": actions.tolist(), "latency_ms": (time.perf_counter() - t0) * 1000.0},
            )
        except Exception as exc:
            logger.exception("inference failed")
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _infer(self, request: dict) -> np.ndarray:
        import cv2
        import torch

        raw = base64.b64decode(request["image"])
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode the posted image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        expected = (self.meta["image_height"], self.meta["image_width"])
        if rgb.shape[:2] != expected:
            raise ValueError(f"image is {rgb.shape[:2]}, expected {expected}")

        state = np.asarray(request["state"], dtype=np.float32)
        n_cols = len(self.meta["state_columns"])
        if state.shape != (n_cols,):
            raise ValueError(f"state has shape {state.shape}, expected ({n_cols},)")

        # lerobot wants CHW float in [0, 1]. Dataset-statistics normalisation
        # is applied by the preprocessor inside _predict_chunk, NOT here and
        # NOT by the policy -- see _load_policy.
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        batch = {
            self.camera_key: image.to(self.device),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
        }
        return _predict_chunk(self.policy, self.pre, self.post, batch, self.n_steps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a lerobot pretrained_model directory")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--n-steps", type=int, default=8, help="chunk rows to return per request")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = Path(args.checkpoint)

    sidecar = checkpoint / SIDECAR_NAME
    if not sidecar.is_file():
        raise SystemExit(
            f"{SIDECAR_NAME} not found in {checkpoint}.\n"
            "It records which state columns survived pruning at pack time; without it the\n"
            "client cannot build the input vector the model was trained on. Copy it from the\n"
            "packed dataset root (runpod/push_weights.sh does this automatically)."
        )
    meta = json.loads(sidecar.read_text(encoding="utf-8"))

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("loading %s on %s", checkpoint, device)
    policy, preprocessor, postprocessor = _load_policy(checkpoint, device)

    _Handler.policy = policy
    _Handler.pre = preprocessor
    _Handler.post = postprocessor
    _Handler.meta = meta
    _Handler.device = device
    _Handler.camera_key = meta.get("camera_key", "observation.images.overhead")
    _Handler.n_steps = args.n_steps

    # Warm the kernels so the first real request is not an outlier.
    warm = {
        _Handler.camera_key: torch.zeros(
            1, 3, meta["image_height"], meta["image_width"], device=device
        ),
        "observation.state": torch.zeros(1, len(meta["state_columns"]), device=device),
    }
    t0 = time.perf_counter()
    chunk = _predict_chunk(policy, preprocessor, postprocessor, warm, args.n_steps)
    logger.info(
        "warmup ok: chunk %s in %.0f ms", chunk.shape, (time.perf_counter() - t0) * 1000.0
    )

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info("serving on http://%s:%d  (GET /meta, POST /infer)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
