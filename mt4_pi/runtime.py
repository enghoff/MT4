"""Control loop: camera + MT4 state -> pi0.5 inference -> safety-checked
`mq` waypoints. See
https://github.com/enghoff/pi0.5-server/blob/master/docs/mt4-client-integration.md,
"Reference loop".

MEDIA now serves pi05_mt4_lora_merged (LoRA fine-tuned on real MT4
cube-shuffle demonstrations, see docs/PI05_FINETUNING_PIPELINE.md) rather
than the stock pi05_droid Franka policy, so client.queue_move() is enabled.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from mt4_jog.client import Mt4Client
from mt4_jog.kinematics import JointAnglesDeg
from mt4_vision.camera import capture_frame

from mt4_pi import adapter, safety
from mt4_pi.observation import build_observation
from mt4_pi.policy_client import PolicyClient

logger = logging.getLogger(__name__)

EXECUTION_HORIZON = 8  # of 15 -- matches the DROID reference default
# A warm call is ~250ms; treat a stall as a stop condition rather than a
# hang, per the doc's "Robustness" watchdog guidance.
INFERENCE_TIMEOUT_S = 5.0


def run(client: Mt4Client, policy: PolicyClient, prompt: str, *, max_ticks: int | None = None) -> None:
    """One control-loop pass per action chunk. Logs and executes every
    waypoint that passes safety.validate().
    """
    tick = 0
    while max_ticks is None or tick < max_ticks:
        status = client.get_status()
        if status.tcp is None or not status.joints:
            logger.warning("incomplete status, skipping tick: %s", status)
            time.sleep(0.1)
            continue

        frame = capture_frame()
        obs = build_observation(frame, status, prompt)

        t0 = time.perf_counter()
        try:
            chunk = policy.infer(obs)["actions"]
        except Exception:
            logger.exception("inference failed, stopping")
            client.stop()
            raise
        if time.perf_counter() - t0 > INFERENCE_TIMEOUT_S:
            logger.error("inference exceeded watchdog timeout, stopping")
            client.stop()
            return
        if not np.isfinite(chunk).all():
            logger.error("non-finite action chunk, stopping")
            client.stop()
            return

        q = JointAnglesDeg.from_steps(
            (status.joints["j1"], status.joints["j2"], status.joints["j3"], status.joints["j4"])
        )
        for action in chunk[:EXECUTION_HORIZON]:
            target = adapter.integrate_joint_target(q, action)
            grip = adapter.gripper_target(action)
            try:
                safety.validate(q, target, grip)
            except safety.SafetyViolation as exc:
                logger.warning("rejected by safety layer: %s", exc)
                break

            waypoint = adapter.to_waypoint(target, grip)
            logger.info("executing waypoint: %s", waypoint)
            client.queue_move(waypoint.x, waypoint.y, waypoint.z, waypoint.j4, waypoint.grip)
            q = target

        tick += 1
