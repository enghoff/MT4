"""Control loop: camera + MT4 state -> ACT inference -> safety-checked `mq` waypoints.

The ACT counterpart of `mt4_pi/runtime.py`. Structurally similar, but the
action semantics differ in one way that matters throughout: chunk rows are
absolute joint targets, so there is no integration against a live estimate
and therefore no drift to accumulate. A bad row is bad once; the next row
still says where the arm should actually be.

Nothing here runs until a checkpoint exists and `mt4_pi.act.serve` is up. See
docs/ACT_PIPELINE.md.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from mt4_jog.client import Mt4Client
from mt4_jog.kinematics import JointAnglesDeg
from mt4_vision.camera import capture_frame

from mt4_pi import safety
from mt4_pi.act import adapter
from mt4_pi.act.client import ActPolicyClient
from mt4_pi.jointstate import joint_state_from_status

logger = logging.getLogger(__name__)

#: Rows executed per inference. The server returns a chunk; executing all of
#: it would be open-loop for its whole span. 8 rows is 0.8 s of trained time
#: but ~1.6 s of wall clock, because each queued waypoint costs ~204 ms on
#: the serial link -- the arm sets this rate, not the GPU.
EXECUTION_HORIZON = 8

#: ACT inference is ~10-20 ms warm; anything near this is a stall, not work.
INFERENCE_TIMEOUT_S = 2.0

#: If the very first row of a chunk is further than this from where the arm
#: actually is, the policy has lost the plot -- an absolute target that far
#: out is not a step, it is a different pose. Distinct from the per-step caps
#: in `safety`, which would reject it too; catching it here lets the loop say
#: why rather than logging eight identical rejections.
MAX_FIRST_TARGET_DEV_DEG = 20.0


def _deviation_deg(a: JointAnglesDeg, b: JointAnglesDeg) -> float:
    return max(abs(a.j1 - b.j1), abs(a.j2 - b.j2), abs(a.j3 - b.j3), abs(a.j4 - b.j4))


def run(
    client: Mt4Client,
    policy: ActPolicyClient,
    *,
    color: str,
    place_kind: str,
    goal_xy_mm: tuple[float, float],
    max_ticks: int | None = None,
) -> None:
    """One control-loop pass per action chunk.

    The task is given as structured arguments rather than a prompt string
    because ACT has no language input -- conditioning rides in the state
    vector as one-hot columns. See `mt4_pi.act.schema`.
    """
    tick = 0
    while max_ticks is None or tick < max_ticks:
        status = client.get_status()
        if status.tcp is None or not status.joints:
            logger.warning("incomplete status, skipping tick: %s", status)
            time.sleep(0.1)
            continue

        frame = capture_frame()
        # j4 is the WORLD-FRAME wrist angle, matching the labels and what
        # queue_move() consumes. See mt4_pi.jointstate.
        q = joint_state_from_status(status)
        state = policy.build_state(
            q, adapter.gripper_fraction(status.tcp.grip), color, place_kind, goal_xy_mm
        )

        t0 = time.perf_counter()
        try:
            chunk = policy.infer(frame, state)
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

        first = adapter.joint_target(chunk[0])
        deviation = _deviation_deg(q, first)
        if deviation > MAX_FIRST_TARGET_DEV_DEG:
            logger.error(
                "first target deviates %.1f deg from the current pose (cap %.1f); "
                "the policy is not tracking the arm -- stopping",
                deviation, MAX_FIRST_TARGET_DEV_DEG,
            )
            client.stop()
            return

        # `current` for the safety check walks forward with the commanded
        # targets: each queued waypoint is where the arm will be when the next
        # one is consumed, so validating against the live pose would compare
        # against a pose already several waypoints stale.
        current = q
        for action in chunk[:EXECUTION_HORIZON]:
            target = adapter.joint_target(action)
            grip = adapter.gripper_target(action)
            try:
                safety.validate(current, target, grip)
            except safety.SafetyViolation as exc:
                logger.warning("rejected by safety layer: %s", exc)
                break

            waypoint = adapter.to_waypoint(target, grip)
            logger.info("executing waypoint: %s", waypoint)
            client.queue_move(waypoint.x, waypoint.y, waypoint.z, waypoint.j4, waypoint.grip)
            current = target

        tick += 1
