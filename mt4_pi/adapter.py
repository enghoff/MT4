"""pi0.5 action -> MT4 `mq` waypoint.

pi05_droid emits (15, 8) action chunks for a 7-DoF Franka: columns 0:7 are
joint VELOCITIES, column 7 is an absolute gripper POSITION (not a velocity).
Only columns 0:4 have an MT4 counterpart; 4:6 are Franka-only and dropped.
See https://github.com/enghoff/pi0.5-server/blob/master/docs/mt4-client-integration.md,
"The wire contract".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN
from mt4_jog.kinematics import JointAnglesDeg, fk_tcp

from mt4_pi.observation import MT4_NUM_JOINTS

DT_S = 1 / 15  # pi05_droid's training rate


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    j4: float
    grip: int


def integrate_joint_target(q_deg: JointAnglesDeg, action: np.ndarray, *, dt: float = DT_S) -> JointAnglesDeg:
    """Clip a DROID action to [-1, 1] and integrate columns 0:4 (q1..q4
    velocities) against the current joint state.
    """
    clipped = np.clip(action, -1.0, 1.0)
    dq = clipped[:MT4_NUM_JOINTS] * dt
    return JointAnglesDeg(
        q_deg.j1 + float(dq[0]),
        q_deg.j2 + float(dq[1]),
        q_deg.j3 + float(dq[2]),
        q_deg.j4 + float(dq[3]),
    )


def gripper_target(action: np.ndarray) -> int:
    """Column 7 is an absolute position, binarised per the DROID reference loop."""
    return GRIPPER_S_CLOSED if action[7] > 0.5 else GRIPPER_S_OPEN


def to_waypoint(target: JointAnglesDeg, grip: int) -> Waypoint:
    """Forward-kinematics the integrated joint target into the Cartesian
    pose queue_move() takes -- no IK round trip needed on this leg.
    """
    tcp = fk_tcp(target)
    return Waypoint(x=tcp.x, y=tcp.y, z=tcp.z, j4=target.j4, grip=grip)
