"""ACT action -> MT4 `mq` waypoint.

The ACT counterpart of `mt4_pi/adapter.py`, and deliberately much smaller.
The pi0.5 adapter had to clip a Franka velocity vector to a plausible rad/s
range and integrate it against a live state estimate; here the model emits
the pose it wants, in the units the arm already uses, so there is nothing to
integrate and nothing to clip. See `schema.py` for why the labels are
absolute.

Kept free of any dependency on the pi0.5 stack -- `mt4_pi.adapter` reaches
`openpi_client` through `mt4_pi.observation`, and this path should not need
openpi installed to drive the arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN
from mt4_jog.kinematics import JointAnglesDeg, fk_tcp

from mt4_pi.act.schema import ACTION_COLUMNS, MT4_NUM_JOINTS


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    j4: float
    grip: int


def joint_target(action: np.ndarray) -> JointAnglesDeg:
    """Columns 0:4 are absolute joint angles in radians -> degrees.

    j4 is the WORLD-FRAME wrist angle, the same quantity `queue_move` takes
    and the same one the labels were built from -- not the raw joint, which
    under the firmware's ORIENT=hold sits at `tcp_j4 - j1`. Conflating the two
    is what wrecked pi0.5 round 1; see `mt4_pi/observation.joint_state_from_status`.
    """
    if action.shape[-1] != len(ACTION_COLUMNS):
        raise ValueError(f"expected {len(ACTION_COLUMNS)} action columns, got {action.shape[-1]}")
    j = [math.degrees(float(v)) for v in action[:MT4_NUM_JOINTS]]
    return JointAnglesDeg(j[0], j[1], j[2], j[3])


def gripper_target(action: np.ndarray) -> int:
    """Column 4 is a gripper opening fraction -> firmware servo units.

    An exact inverse of `mt4_pi.observation._gripper_fraction`, with no
    threshold anywhere. The pi0.5 path binarised this against 0.5 and snapped
    closed to `GRIPPER_S_CLOSED`, both of which were wrong for our data: the
    labels are bimodal at 0.121 and 0.818, so 0.5 is not the midpoint, and
    0.818 corresponds to servo 255, not 285. Snapping to 285 commanded a
    harder squeeze than any demonstration ever used.

    Reproducing the demonstrated value continuously sidesteps the question --
    the model was trained on grips that actually worked, so the right thing to
    do is replay them, not re-derive a threshold.
    """
    span = GRIPPER_S_CLOSED - GRIPPER_S_OPEN
    servo = GRIPPER_S_OPEN + float(action[len(ACTION_COLUMNS) - 1]) * span
    return int(round(max(GRIPPER_S_OPEN, min(GRIPPER_S_CLOSED, servo))))


def gripper_fraction(servo_position: float) -> float:
    """Firmware gripper servo units -> the [0, 1] the model consumes.

    Same formula as `mt4_pi.observation._gripper_fraction` and
    `mt4_pi.collect.recorder._gripper_fraction`; duplicated for the same
    reason recorder.py duplicates it, to keep this path free of the pi0.5
    stack. `gripper_target` is its inverse.
    """
    span = GRIPPER_S_CLOSED - GRIPPER_S_OPEN
    return max(0.0, min(1.0, (servo_position - GRIPPER_S_OPEN) / span))


def to_waypoint(target: JointAnglesDeg, grip: int) -> Waypoint:
    """Forward-kinematics the joint target into the Cartesian pose
    `queue_move()` takes -- no IK round trip needed on this leg."""
    tcp = fk_tcp(target)
    return Waypoint(x=tcp.x, y=tcp.y, z=tcp.z, j4=target.j4, grip=grip)
