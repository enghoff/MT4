"""ACT action -> MT4 waypoint tests (no hardware).

These cover the two places the ACT path deliberately diverges from the pi0.5
adapter, both of which are corrections to bugs that reached the arm:
absolute joint targets instead of integrated velocities, and a continuous
gripper inverse instead of a 0.5 threshold onto GRIPPER_S_CLOSED.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN
from mt4_jog.kinematics import JointAnglesDeg, fk_tcp
from mt4_pi.act import adapter

# The extremes actually present in the 225-episode corpus. Both training
# paths normalise the gripper the same way, so these are the only two values
# a well-fit policy should be commanding.
DEMO_GRIP_OPEN = 0.121
DEMO_GRIP_CLOSED = 0.818


def _action(j1=0.0, j2=0.0, j3=0.0, j4=0.0, grip=0.0):
    return np.array([j1, j2, j3, j4, grip], dtype=np.float32)


def test_joint_target_is_absolute_and_converts_radians_to_degrees():
    target = adapter.joint_target(_action(j1=0.5, j2=1.8, j3=-0.3, j4=0.25))
    assert target.j1 == pytest.approx(math.degrees(0.5))
    assert target.j2 == pytest.approx(math.degrees(1.8))
    assert target.j3 == pytest.approx(math.degrees(-0.3))
    assert target.j4 == pytest.approx(math.degrees(0.25))


def test_joint_target_ignores_the_current_pose_entirely():
    """The defining property of an absolute target: the same action yields
    the same pose no matter where the arm is. The pi0.5 adapter could not
    satisfy this -- it integrated onto the live state."""
    action = _action(j1=0.2, j2=1.5, j3=0.1, j4=0.0)
    assert adapter.joint_target(action) == adapter.joint_target(action)


def test_joint_target_rejects_a_wrongly_sized_action():
    with pytest.raises(ValueError, match="action columns"):
        adapter.joint_target(np.zeros(8, dtype=np.float32))


def test_gripper_target_inverts_gripper_fraction():
    for servo in (GRIPPER_S_OPEN, 140, 180, 235, 255, GRIPPER_S_CLOSED):
        frac = adapter.gripper_fraction(servo)
        assert adapter.gripper_target(_action(grip=frac)) == servo


def test_gripper_target_reproduces_the_demonstrated_grip_not_a_hard_squeeze():
    """The pi0.5 adapter binarised at 0.5 and snapped closed to
    GRIPPER_S_CLOSED (285). The demonstrations never went past 255, so that
    commanded a harder squeeze than anything the model was trained on."""
    assert adapter.gripper_target(_action(grip=DEMO_GRIP_CLOSED)) == 255
    assert adapter.gripper_target(_action(grip=DEMO_GRIP_OPEN)) == 140
    assert adapter.gripper_target(_action(grip=DEMO_GRIP_CLOSED)) < GRIPPER_S_CLOSED


def test_gripper_target_has_no_threshold_and_stays_monotonic():
    """A mid-range prediction must land mid-range, not get snapped to an
    endpoint by a threshold that the bimodal data does not justify."""
    servos = [adapter.gripper_target(_action(grip=f)) for f in np.linspace(0.0, 1.0, 11)]
    assert servos == sorted(servos)
    assert adapter.gripper_target(_action(grip=0.5)) == pytest.approx(
        GRIPPER_S_OPEN + 0.5 * (GRIPPER_S_CLOSED - GRIPPER_S_OPEN), abs=1
    )
    assert len(set(servos)) > 2, "output collapsed to two levels -- a threshold crept back in"


@pytest.mark.parametrize("frac,expected", [(-3.0, GRIPPER_S_OPEN), (7.5, GRIPPER_S_CLOSED)])
def test_gripper_target_clamps_out_of_range_predictions(frac, expected):
    assert adapter.gripper_target(_action(grip=frac)) == expected


def test_to_waypoint_matches_forward_kinematics():
    target = JointAnglesDeg(10.0, 95.0, 5.0, 12.0)
    waypoint = adapter.to_waypoint(target, 200)
    tcp = fk_tcp(target)
    assert (waypoint.x, waypoint.y, waypoint.z) == pytest.approx((tcp.x, tcp.y, tcp.z))
    # j4 passes through as the world-frame wrist angle queue_move() expects,
    # rather than being re-derived from the raw joint.
    assert waypoint.j4 == pytest.approx(target.j4)
    assert waypoint.grip == 200


def test_gripper_fraction_spans_the_firmware_range():
    assert adapter.gripper_fraction(GRIPPER_S_OPEN) == pytest.approx(0.0)
    assert adapter.gripper_fraction(GRIPPER_S_CLOSED) == pytest.approx(1.0)
    assert adapter.gripper_fraction(GRIPPER_S_OPEN - 50) == pytest.approx(0.0)
    assert adapter.gripper_fraction(GRIPPER_S_CLOSED + 50) == pytest.approx(1.0)
