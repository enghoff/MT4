"""The ground / keep-out guard for joint-space motion, checked against the
same kinematics the firmware runs.

This mirrors ``refresh_envelope_guard_if_due()`` in
``firmware/mt4_jog/src/motion.cpp``. It cannot call that code, so it restates
the rule and pins the two properties the rule exists for:

  * motion that pushes the TCP further below the desk, or further into the
    base keep-out cylinder, is stopped
  * motion that escapes an existing violation is never stopped

The second is not a nicety. At the J2 limit switch the TCP sits at r=137.2mm,
already inside the 140mm cylinder, so a guard that simply refused every
violating pose would freeze the arm at the pose homing always ends near.

The gap this closes: ``motion_step_allowed()`` only ever knew about joint step
counters and the J2+J3 coupling, so `j` (joint jog) and `m` (relative move)
reached the desk with nothing objecting, while `mp`/`mq` and Cartesian jog were
both guarded. See the block comment on the firmware function.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_jog.joints import (
    GROUND_Z_MM,
    J2_J3_SUM_MAX_STEPS,
    J2_J3_SUM_MIN_STEPS,
    JOINT_SOFT_MAX_STEPS,
    JOINT_SOFT_MIN_STEPS,
)
from mt4_jog.kinematics import JointAnglesDeg, fk_tcp

KEEPOUT_RADIUS_MM = 140.0
LO, HI = JOINT_SOFT_MIN_STEPS, JOINT_SOFT_MAX_STEPS


def tcp_of(s1: int, s2: int, s3: int) -> tuple[float, float]:
    """(z, r) at these joint step counters."""
    t = fk_tcp(JointAnglesDeg.from_steps((s1, s2, s3, 0)))
    return t.z, math.hypot(t.x, t.y)


def guard_stops(here: tuple[int, int, int], ahead: tuple[int, int, int]) -> bool:
    """The firmware rule: stop when the predicted pose both violates and is
    worse than the current one."""
    z_now, r_now = tcp_of(*here)
    z_next, r_next = tcp_of(*ahead)
    into_ground = z_next < GROUND_Z_MM and z_next < z_now
    into_column = r_next < KEEPOUT_RADIUS_MM and r_next < r_now
    return into_ground or into_column


def within_joint_box(s2: int, s3: int) -> bool:
    return (
        LO[1] <= s2 <= HI[1]
        and LO[2] <= s3 <= HI[2]
        and J2_J3_SUM_MIN_STEPS <= s2 + s3 <= J2_J3_SUM_MAX_STEPS
    )


def test_joint_limits_alone_do_not_keep_the_tcp_above_the_desk():
    """The regression this guard exists for.

    If this ever fails, the joint soft limits have been tightened enough to
    imply the desk plane and the guard is redundant -- verify that before
    deleting anything.
    """
    worst_z = None
    worst_r = None
    for s2 in range(LO[1], HI[1] + 1, 25):
        for s3 in range(LO[2], HI[2] + 1, 25):
            if not within_joint_box(s2, s3):
                continue
            z, r = tcp_of(0, s2, s3)
            if worst_z is None or z < worst_z:
                worst_z = z
            if worst_r is None or r < worst_r:
                worst_r = r
    assert worst_z is not None and worst_r is not None
    # Measured 2026-08-02: 78mm below the desk, 22mm inside the column.
    assert worst_z < GROUND_Z_MM - 50.0, f"deepest legal joint pose z={worst_z:.1f}"
    assert worst_r < KEEPOUT_RADIUS_MM - 15.0, f"closest legal joint pose r={worst_r:.1f}"


def test_guard_stops_a_jog_descending_into_the_desk():
    # J2=3000, J3=-800 sits at z=116.2mm -- 1.2mm of desk clearance. Stepping
    # J3 down from there is exactly the jog that used to go through the table.
    here = (0, 3000, -800)
    z_here, _ = tcp_of(*here)
    assert z_here > GROUND_Z_MM, "start above the desk or the test proves nothing"
    # z is monotonic in J3, so stepping J3 negative lowers the TCP.
    descending = (0, 3000, -800 - 200)
    assert tcp_of(*descending)[0] < GROUND_Z_MM
    assert guard_stops(here, descending)


def test_guard_allows_a_jog_climbing_away_from_the_desk():
    # Start already under the desk: the escape must not be blocked.
    here = (0, 3000, -1400)
    assert tcp_of(*here)[0] < GROUND_Z_MM
    climbing = (0, 3000, -1400 + 200)
    assert tcp_of(*climbing)[0] > tcp_of(*here)[0]
    assert not guard_stops(here, climbing)


def test_guard_never_traps_the_arm_inside_the_keepout():
    """At the J2 limit switch the TCP is inside the cylinder by construction.

    Homing ends near there, so escape has to stay available or the arm is
    bricked until someone power-cycles and hand-moves it.
    """
    at_limit = (0, 0, -200)
    _z, r = tcp_of(*at_limit)
    assert r < KEEPOUT_RADIUS_MM, f"expected an in-cylinder pose, got r={r:.1f}"
    # J2 stepping positive always increases radius -- the escape direction.
    escaping = (0, 200, -200)
    assert tcp_of(*escaping)[1] > r
    assert not guard_stops(at_limit, escaping)


def test_guard_stops_a_jog_driving_deeper_into_the_column():
    here = (0, 400, -200)
    _z, r_here = tcp_of(*here)
    deeper = (0, 200, -200)
    assert tcp_of(*deeper)[1] < r_here
    if r_here < KEEPOUT_RADIUS_MM or tcp_of(*deeper)[1] < KEEPOUT_RADIUS_MM:
        assert guard_stops(here, deeper)


def test_ground_violation_is_monotonic_in_j3():
    """The guard's one-step-ahead comparison relies on this.

    If z stopped being monotonic in J3, a lookahead could straddle a local
    minimum and read "improving" while passing through the desk.
    """
    for s2 in range(LO[1], HI[1] + 1, 250):
        j3_lo = max(LO[2], J2_J3_SUM_MIN_STEPS - s2)
        j3_hi = min(HI[2], J2_J3_SUM_MAX_STEPS - s2)
        if j3_lo >= j3_hi:
            continue
        zs = [tcp_of(0, s2, s3)[0] for s3 in range(j3_lo, j3_hi + 1, 50)]
        assert all(zs[i] <= zs[i + 1] + 1e-6 for i in range(len(zs) - 1)), (
            f"z not monotonic in J3 at J2={s2}"
        )
