"""Thin wrapper around openpi_client.WebsocketClientPolicy: defensive
connection settings plus a CLI smoke test that needs no MT4 hardware.

See https://github.com/enghoff/pi0.5-server/blob/master/docs/mt4-client-integration.md
for the wire contract and the keepalive gotcha `ping_interval` works around
(OpenPI runs inference synchronously inside its asyncio event loop, so a
slow call -- e.g. the ~33s JAX compile on the first call after server
start -- can trip the default 20s WebSocket keepalive).
"""

from __future__ import annotations

import logging

from openpi_client import websocket_client_policy

logger = logging.getLogger(__name__)

DEFAULT_HOST = "192.168.1.3"
DEFAULT_PORT = 8000


class PolicyClient:
    """Reconnecting wrapper. `infer()` raises on a broken connection --
    callers are expected to stop the arm before calling reconnect().
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        ping_interval: float | None = -1.0,
    ) -> None:
        self._host = host
        self._port = port
        self._ping_interval = ping_interval
        self._policy = self._connect()

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        return websocket_client_policy.WebsocketClientPolicy(
            host=self._host, port=self._port, ping_interval=self._ping_interval
        )

    def get_server_metadata(self) -> dict:
        return self._policy.get_server_metadata()

    def infer(self, observation: dict) -> dict:
        return self._policy.infer(observation)

    def reconnect(self) -> None:
        logger.warning("reconnecting to %s:%s", self._host, self._port)
        self._policy = self._connect()


def _smoke_test() -> None:
    """`python -m mt4_pi.policy_client [--host H] [--port P]` -- verifies
    the server responds using synthetic data. No camera or arm required.
    """
    import argparse
    import time

    import numpy as np

    from mt4_pi.observation import DROID_NUM_JOINTS, IMAGE_SIZE

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = PolicyClient(host=args.host, port=args.port)
    print("server metadata:", client.get_server_metadata())

    rng = np.random.default_rng(0)
    obs = {
        "observation/exterior_image_1_left": rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(DROID_NUM_JOINTS, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "put the red cube in the tray",
    }

    for i in range(args.iters):
        t0 = time.perf_counter()
        actions = client.infer(obs)["actions"]
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"iter {i}: {dt:8.1f} ms  actions {actions.shape} finite={bool(np.isfinite(actions).all())}")


if __name__ == "__main__":
    _smoke_test()
