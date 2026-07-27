"""Validates a proposed joint target before it becomes an `mq` waypoint.

`mt4-jog` and the firmware remain authoritative on whether a pose is
physically valid -- this is an additional filter, not a replacement. See
https://github.com/enghoff/pi0.5-server/blob/master/docs/mt4-client-integration.md,
"Safety layer".
"""

from __future__ import annotations

import math

from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN
from mt4_jog.kinematics import JointAnglesDeg, fk_tcp, steps_from_angles
from mt4_vision.workspace import is_within_envelope, joints_within_soft_limits

# Per-step caps, independent of the firmware's own clamping -- the point is
# to reject a policy proposing a large jump before it is ever sent, not to
# rely on the firmware as the only backstop.
MAX_JOINT_STEP_DEG = 5.0  # q1..q3, per control-loop tick
MAX_J4_STEP_DEG = 10.0
MAX_TCP_STEP_MM = 15.0


class SafetyViolation(Exception):
    """Raised with a human-readable reason when validate() rejects a step."""


def validate(current: JointAnglesDeg, target: JointAnglesDeg, grip: int) -> None:
    if not all(math.isfinite(v) for v in (target.j1, target.j2, target.j3, target.j4)):
        raise SafetyViolation(f"non-finite joint target: {target}")

    for name, cur, tgt, cap in (
        ("q1", current.j1, target.j1, MAX_JOINT_STEP_DEG),
        ("q2", current.j2, target.j2, MAX_JOINT_STEP_DEG),
        ("q3", current.j3, target.j3, MAX_JOINT_STEP_DEG),
        ("j4", current.j4, target.j4, MAX_J4_STEP_DEG),
    ):
        step = abs(tgt - cur)
        if step > cap:
            raise SafetyViolation(f"{name} step {step:.2f} deg exceeds {cap} deg cap")

    if not GRIPPER_S_OPEN <= grip <= GRIPPER_S_CLOSED:
        raise SafetyViolation(f"gripper target {grip} outside servo range")

    if not joints_within_soft_limits(steps_from_angles(target)):
        raise SafetyViolation(f"joint target {target} outside soft limits / coupled J2+J3 range")

    tcp = fk_tcp(target)
    if not is_within_envelope(tcp.x, tcp.y, tcp.z):
        raise SafetyViolation(f"TCP ({tcp.x:.1f}, {tcp.y:.1f}, {tcp.z:.1f}) outside workspace envelope")

    cur_tcp = fk_tcp(current)
    tcp_step = math.dist((cur_tcp.x, cur_tcp.y, cur_tcp.z), (tcp.x, tcp.y, tcp.z))
    if tcp_step > MAX_TCP_STEP_MM:
        raise SafetyViolation(f"TCP step {tcp_step:.1f}mm exceeds {MAX_TCP_STEP_MM}mm cap")
