"""Wrist-angle geometry: what world-frame J4 a grasp or a transit wants.

Pure math -- no serial I/O, no calibration. Extracted from pickplace so the
motion primitives can sit on top of it without a circular import, and so the
one place that reasons about J4 soft limits is the one place that gets tested
(tests/test_face_align.py).

Two conventions live here and they are easy to confuse:

* **World-frame J4** is what ``mp``/``mq`` take: gripper yaw in the robot
  frame. Firmware drives ``joint_j4 = world_j4 - j1``, so a legal world angle
  can still be an illegal *joint* angle once J1 swings -- which is why every
  lattice search filters on the joint value at the target bearing.
* **Grasp periodicity** differs by object. A square cube face repeats every
  90°; an elongated object's cross-grasp repeats every 180°. That single
  number is the whole difference between the two lattice searches, and it is
  why one of them can fail to find any feasible angle at all.

Assumes firmware ``j4zero`` (``calibrate_j4.py``): jaws along the arm means
world J4 = 0.
"""

from __future__ import annotations

import math

from mt4_jog.joints import JOINT_SOFT_MAX_STEPS, JOINT_SOFT_MIN_STEPS
from mt4_jog.kinematics import STEPS_PER_DEG

# Steps of headroom kept inside each J4 soft limit when picking a candidate.
DEFAULT_J4_MARGIN_STEPS = 200


def fold_yaw_deg(yaw_deg: float, period_deg: float = 90.0) -> float:
    """Map any angle into (-period/2, period/2] -- one representative of a
    ``period_deg``-symmetric grasp direction."""
    half = period_deg / 2.0
    return (yaw_deg + half) % period_deg - half


def fold_square_yaw_deg(yaw_deg: float) -> float:
    """Map any angle into (-45, 45] -- one face of a square (90° period)."""
    return fold_yaw_deg(yaw_deg, 90.0)


def _j4_on_lattice(
    base_deg: float,
    period_deg: float,
    *,
    current_j4_deg: float | None = None,
    x: float | None = None,
    y: float | None = None,
    j4_margin_steps: int = DEFAULT_J4_MARGIN_STEPS,
    require_feasible: bool = False,
) -> float | None:
    """World-frame J4 (deg) on the ``period_deg`` lattice through ``base_deg``.

    Shared core of face-align (90°, square cube faces) and long-axis grasps
    (180°, an elongated object). Candidates cover the full circle either way;
    when ``current_j4_deg`` is given, the one minimizing wrist travel wins --
    but only among candidates whose *joint* J4 (world − j1) stays inside soft
    limits at (x, y). Preferring nearest world yaw alone can pin joint J4 past
    ±8100 on far −Y picks after the wrist has drifted near 90° (stack level-4:
    world 109° at j1≈−72° → joint 181° / 8130 steps → ``err mp joints``).

    ``require_feasible`` decides what happens when *nothing* fits: return None
    (abstain) rather than falling back to the unfiltered set and commanding an
    out-of-limit wrist.

    At this arm's stock soft limits that never triggers, for either period --
    the joint-J4 window is ~351° wide (±8100 steps less the margin, at 45
    steps/deg), so even the 180° lattice's two candidates per turn always leave
    one inside. Measured, not assumed. The filter above is still load-bearing:
    it decides *which* candidate is used, which is what keeps far −Y picks off
    ``err mp joints``. ``require_feasible`` is the guard for a rig whose limits
    are narrower than the period -- ``MT4_JOINT_SOFT_*`` can shrink that window
    below 180° -- where a silent fallback would be a wrong grasp rather than a
    suboptimal one.
    """
    base = fold_yaw_deg(base_deg, period_deg)
    k_max = int(math.ceil(360.0 / period_deg))
    candidates = [base + period_deg * k for k in range(-k_max, k_max + 1)]
    if x is not None and y is not None:
        j1 = math.degrees(math.atan2(y, x))
        lo = JOINT_SOFT_MIN_STEPS[3] / STEPS_PER_DEG[3] + j4_margin_steps / STEPS_PER_DEG[3]
        hi = JOINT_SOFT_MAX_STEPS[3] / STEPS_PER_DEG[3] - j4_margin_steps / STEPS_PER_DEG[3]
        feasible = [w for w in candidates if lo <= (w - j1) <= hi]
        if feasible:
            candidates = feasible
        elif require_feasible:
            return None
    if current_j4_deg is None:
        # Prefer the folded representative (or the feasible one closest to it).
        return min(candidates, key=lambda w: abs(w - base))
    return min(candidates, key=lambda w: abs(w - current_j4_deg))


def j4_for_face_align(
    cube_yaw_deg: float,
    *,
    current_j4_deg: float | None = None,
    x: float | None = None,
    y: float | None = None,
    j4_margin_steps: int = DEFAULT_J4_MARGIN_STEPS,
) -> float:
    """World-frame J4 (deg) so the jaws meet a cube face, not a corner.

    ``cube_yaw_deg`` is a robot-frame edge angle from detection; squares are
    90°-periodic. See ``_j4_on_lattice`` for the soft-limit reasoning. Always
    returns an angle -- with four candidates per turn something always fits.
    """
    j4 = _j4_on_lattice(
        cube_yaw_deg,
        90.0,
        current_j4_deg=current_j4_deg,
        x=x,
        y=y,
        j4_margin_steps=j4_margin_steps,
    )
    assert j4 is not None  # require_feasible is False on the 90° lattice
    return j4


def j4_for_long_axis(
    axis_yaw_deg: float,
    *,
    current_j4_deg: float | None = None,
    x: float | None = None,
    y: float | None = None,
    j4_margin_steps: int = DEFAULT_J4_MARGIN_STEPS,
) -> float | None:
    """World-frame J4 (deg) that closes the jaws ACROSS a long axis.

    For an elongated object (a pen, a screwdriver) the grasp direction is the
    perpendicular of its long axis, and it is 180°-periodic rather than 90°:
    the two jaws are interchangeable, but a quarter turn is a different grasp
    entirely -- it would try to close along the shaft. ``axis_yaw_deg`` is the
    robot-frame long axis (see ``mt4_vision.locate``).

    Returns None when no candidate keeps joint J4 inside its soft limits at
    (x, y). At stock limits that cannot happen (the joint-J4 window is wider
    than 180°); it is the guard for a rig whose ``MT4_JOINT_SOFT_*`` window is
    narrower than the grasp period, where the caller must abstain rather than
    grasp along the shaft instead of across it.
    """
    return _j4_on_lattice(
        axis_yaw_deg + 90.0,
        180.0,
        current_j4_deg=current_j4_deg,
        x=x,
        y=y,
        j4_margin_steps=j4_margin_steps,
        require_feasible=True,
    )


def j4_preserve_wrist(
    x: float,
    y: float,
    *,
    from_x: float,
    from_y: float,
    from_j4: float,
) -> float:
    """World-frame J4 that keeps joint J4 fixed across a J1 swing.

    ``Mt4Client.move_to(j4=None)`` holds *world* yaw, which commands
    ``joint_j4 = world_j4 - j1``. Large base swings (e.g. to marker 0 at
    j1≈−80°) then drive J4 past soft limits → ``err mp joints``. Holding
    the wrist joint instead yields ``world_j4 = j1_tgt + (from_j4 - j1_from)``.

    Superseded on the wire by the firmware ``w`` sentinel (which resolves this
    per leg, on-device, from wherever the previous leg actually ended); kept
    for reference and tests.
    """
    j1_from = math.degrees(math.atan2(from_y, from_x))
    j1_to = math.degrees(math.atan2(y, x))
    return j1_to + (from_j4 - j1_from)
