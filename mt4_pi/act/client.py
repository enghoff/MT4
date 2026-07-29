"""Workstation-side client for the ACT policy server.

Stdlib plus numpy and cv2 -- no torch, no lerobot, no openpi. The arm-side
machine should not need the training stack installed to drive the arm.

The client owns building the state vector, because only it knows the live
joint angles and the requested task, and it must produce exactly the layout
the server was trained on. That layout lives in the server's `/meta`
(ultimately `mt4_act_meta.json`), not in `schema.py`, because constant
columns are pruned at pack time -- see `schema.prune_constant_columns`.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import urllib.error
import urllib.request

import numpy as np

from mt4_jog.kinematics import JointAnglesDeg
from mt4_pi.act.schema import build_state, select_columns

logger = logging.getLogger(__name__)

DEFAULT_HOST = "192.168.1.3"
DEFAULT_PORT = 8100


class ActPolicyClient:
    """Talks to `mt4_pi.act.serve`. `infer()` raises on transport failure --
    callers must stop the arm before retrying.
    """

    def __init__(
        self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, timeout: float = 5.0
    ) -> None:
        self._base = f"http://{host}:{port}"
        self._timeout = timeout
        self.meta = self._get("/meta")
        logger.info(
            "connected to %s: %d state columns, %dx%d, fps %s",
            self._base,
            len(self.meta["state_columns"]),
            self.meta["image_width"],
            self.meta["image_height"],
            self.meta["fps"],
        )

    # -- transport ---------------------------------------------------------

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self._base + path, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base + path, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"policy server error: {exc.read().decode('utf-8', 'replace')}") from exc

    # -- observation -------------------------------------------------------

    def prepare_image(self, frame_bgr: np.ndarray) -> str:
        """Resize a camera frame to the trained resolution and JPEG-encode it.

        Plain resize, not `resize_with_pad`: the export downscaled 1280x720 to
        640x360, which is the same 16:9 aspect, so there is nothing to pad and
        padding here would put the model off-distribution.
        """
        import cv2

        h, w = self.meta["image_height"], self.meta["image_width"]
        if frame_bgr.shape[:2] != (h, w):
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError("cv2.imencode failed on the camera frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def build_state(
        self,
        q_deg: JointAnglesDeg,
        gripper_fraction: float,
        color: str,
        place_kind: str,
        goal_xy_mm: tuple[float, float],
    ) -> np.ndarray:
        """Live arm state + task -> the exact vector this checkpoint expects.

        `q_deg.j4` must be the WORLD-FRAME wrist angle (`status.tcp.j4`), which
        is what `observation.joint_state_from_status` returns and what the
        labels were built from.
        """
        full = build_state(
            np.array(
                [math.radians(v) for v in (q_deg.j1, q_deg.j2, q_deg.j3, q_deg.j4)], dtype=np.float32
            ),
            gripper_fraction,
            color,
            place_kind,
            goal_xy_mm,
        )
        return select_columns(full, self.meta["state_columns"])

    # -- inference ---------------------------------------------------------

    def infer(self, frame_bgr: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Returns the action chunk as (T, 5): absolute joint targets (rad)
        plus a gripper fraction."""
        reply = self._post(
            "/infer", {"image": self.prepare_image(frame_bgr), "state": [float(v) for v in state]}
        )
        return np.asarray(reply["actions"], dtype=np.float32)


def _smoke_test() -> None:
    """`python -m mt4_pi.act.client [--host H]` -- checks the server responds
    on synthetic data. No camera or arm required."""
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = ActPolicyClient(host=args.host, port=args.port)
    print("meta:", json.dumps(client.meta, indent=2))

    rng = np.random.default_rng(0)
    frame = rng.integers(
        0, 256, (client.meta["image_height"], client.meta["image_width"], 3), dtype=np.uint8
    )
    state = client.build_state(
        JointAnglesDeg(0.0, 90.0, 0.0, 0.0), 0.2, "red", "to_marker", (150.0, 100.0)
    )
    for i in range(args.iters):
        t0 = time.perf_counter()
        chunk = client.infer(frame, state)
        dt = (time.perf_counter() - t0) * 1000.0
        print(
            f"iter {i}: {dt:7.1f} ms  chunk {chunk.shape}  "
            f"finite={bool(np.isfinite(chunk).all())}  first={np.round(chunk[0], 3)}"
        )


if __name__ == "__main__":
    _smoke_test()
