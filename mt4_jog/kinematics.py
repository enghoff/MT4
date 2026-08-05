"""MT4 forward kinematics and Cartesian jog mixing (matches jog firmware geometry)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Factory EEPROM geometry (mm). The MT4 is a parallel-link (palletizing) arm:
# J2 sets the upper-arm absolute angle, J3 sets the forearm absolute angle via
# the link rods (independent of J2), and the head platform stays level.
LINKAGE1 = 130.0  # shoulder -> elbow
LINKAGE2 = 150.0  # elbow -> wrist pivot
CENCER_OFFSET = 45.0  # J1 axis -> shoulder, horizontal
CENCER_HEIGHT = 140.0  # shoulder pivot height
HEAD_OFFSET = 35.0  # wrist pivot -> TCP, horizontal (head stays level)
HEAD_HEIGHT = 14.43  # TCP below wrist pivot

# Model angles at step-counter zero. J1/J4 zero at the post-home park
# (J1 centered; J4 jaws-along-arm after j4zero). J2/J3 zero at the
# limit/interference reference so pull-off changes do not invalidate the
# angle fit: after home, counters sit at +(J2_PULL, J3_PULL).
#
# J2/J3 derived from the 2026-07-21 tape-fit park pose (107.0°, −9.3°) under
# the pull-offs then in effect (1000 / 500): ref = park - SIGN * pull / SPD.
# FK at park still reports TCP ≈ (190.0, 0, 225.6). The 1000/500 literals are
# frozen fit provenance — do not retie to J*_HOME_PULLOFF_STEPS.
HOME_J1_DEG = 0.0
HOME_J2_DEG = 107.0 + 1000.0 / 35.0  # ≈135.571° at J2 limit switch
HOME_J3_DEG = -9.3 - 500.0 / 35.0  # ≈−23.586° at J3 interference
HOME_J4_DEG = 0.0

# All four measured 2026-07-06 (J2-J4 with a phone clinometer against the
# link; J1 by direct measurement of its yaw rotation), not taken from the
# factory EEPROM dump -- J1/J2/J3 share a physical motor/gearbox design
# (~35 steps/deg each). The dump has no J3 setting at all (its 35.556 belongs
# to unrelated extra axes), and its 852 for J4 comes from reading the axis
# letter "d" as J4, which it is not.
STEPS_PER_DEG: tuple[float, float, float, float] = (35.0, 35.0, 35.0, 45.0)

# +1 if positive step count increases firmware joint angle, -1 if driver is inverted.
# J3 confirmed inverted 2026-07-06: a +299-step probe raised the forearm tip
# rather than lowering it, which is what -1.0 predicts (photo-confirmed).
J_STEP_SIGN: tuple[float, float, float, float] = (1.0, -1.0, 1.0, 1.0)

DLS_LAMBDA = 0.05

# Positive joint angle => DIR pin low (matches jog.py "q" direction).
DIR_POS_HIGH: tuple[bool, bool, bool, bool] = (False, False, False, False)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def normalized(self) -> Vec3:
        n = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if n < 1e-9:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / n, self.y / n, self.z / n)


@dataclass(frozen=True)
class JointAnglesDeg:
    j1: float
    j2: float
    j3: float
    j4: float = 0.0

    @classmethod
    def from_steps(cls, steps: tuple[int, int, int, int]) -> JointAnglesDeg:
        spd = STEPS_PER_DEG
        s = J_STEP_SIGN
        return cls(
            HOME_J1_DEG + s[0] * steps[0] / spd[0],
            HOME_J2_DEG + s[1] * steps[1] / spd[1],
            HOME_J3_DEG + s[2] * steps[2] / spd[2],
            HOME_J4_DEG + s[3] * steps[3] / spd[3],
        )


def fk_tcp(q: JointAnglesDeg) -> Vec3:
    q1, q2, q3 = map(math.radians, (q.j1, q.j2, q.j3))
    radial = CENCER_OFFSET + LINKAGE1 * math.cos(q2) + LINKAGE2 * math.cos(q3) + HEAD_OFFSET
    return Vec3(
        radial * math.cos(q1),
        radial * math.sin(q1),
        CENCER_HEIGHT + LINKAGE1 * math.sin(q2) + LINKAGE2 * math.sin(q3) - HEAD_HEIGHT,
    )


def ws_j4_deg(q: JointAnglesDeg) -> float:
    """Gripper yaw in world frame (deg): j4_joint + j1. Invariant under orient hold."""
    return q.j4 + q.j1


def jacobian_mm_per_deg(q: JointAnglesDeg) -> np.ndarray:
    """3x4 matrix: d(TCP mm)/d(joint deg)."""
    j = np.zeros((3, 4))
    base = np.array([q.j1, q.j2, q.j3, q.j4], dtype=float)
    tcp0 = fk_tcp(q)
    p0 = np.array([tcp0.x, tcp0.y, tcp0.z])
    for i in range(4):
        trial = base.copy()
        trial[i] += 0.1
        q_t = JointAnglesDeg(*trial)
        p1 = np.array([fk_tcp(q_t).x, fk_tcp(q_t).y, fk_tcp(q_t).z])
        j[:, i] = (p1 - p0) / 0.1
    return j


def cartesian_joint_rates_deg(
    q: JointAnglesDeg,
    direction: Vec3,
    *,
    hold_orientation: bool = True,
) -> tuple[float, float, float, float] | None:
    """Model-space joint deg rates for unit world velocity along direction.

    When hold_orientation, J4 counters base yaw 1:1 (dq4 = -dq1), matching the
    firmware's `orient on` behavior. Driver step signs are applied in
    cartesian_step_rates().
    """
    v = direction.normalized()
    if abs(v.x) + abs(v.y) + abs(v.z) < 1e-9:
        return None

    j_pos = jacobian_mm_per_deg(q)[:, :3]
    jjt = j_pos @ j_pos.T + (DLS_LAMBDA**2) * np.eye(3)
    try:
        y = np.linalg.solve(jjt, np.array([v.x, v.y, v.z]))
        dq123 = j_pos.T @ y
    except np.linalg.LinAlgError:
        return None

    dq4 = 0.0
    if hold_orientation and abs(dq123[0]) > 1e-6:
        dq4 = -dq123[0]

    return float(dq123[0]), float(dq123[1]), float(dq123[2]), dq4


def cartesian_step_rates(
    q: JointAnglesDeg,
    direction: Vec3,
    *,
    hold_orientation: bool = True,
    master_scale: int = 10_000,
) -> tuple[int, int, int, int, int] | None:
    """Bresenham integer rates (j1..j4, master) for firmware cj command."""
    rates = cartesian_joint_rates_deg(q, direction, hold_orientation=hold_orientation)
    if rates is None:
        return None

    steps = [J_STEP_SIGN[i] * rates[i] * STEPS_PER_DEG[i] for i in range(4)]

    # Peak/master-scale spans all four joints, which is affordable only
    # because J4's steps/deg (~45) is close to J1's (~44): at the 852 the
    # EEPROM dump implies (an axis-letter misassignment) a modest
    # orientation-hold correction would dominate the DDA timing budget.
    # Including J4 costs at most a few percent of speed and gives exact
    # wrist-unwind fidelity instead of clamping J4 short.
    peak = max(abs(s) for s in steps)
    if peak < 1e-9:
        return None

    scale = master_scale / peak
    ints = [int(round(s * scale)) for s in steps]
    return ints[0], ints[1], ints[2], ints[3], master_scale


def steps_to_dir_high(joint_index: int, rate: int) -> bool:
    positive = rate > 0
    return DIR_POS_HIGH[joint_index] if positive else not DIR_POS_HIGH[joint_index]


# ---------------------------------------------------------------------------
# Position-level IK (closed form) -- for bounded point-to-point moves via the
# firmware `m` command, as opposed to the resolved-rate jog IK above.
# ---------------------------------------------------------------------------


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def ik_q2_q3(
    radial: float, z: float, near_q2: float, near_q3: float
) -> tuple[float, float] | None:
    """Closed-form two-link solve in the arm's vertical plane:

        LINKAGE1*(cos q2, sin q2) + LINKAGE2*(cos q3, sin q3) = target

    (absolute joint angles; circle-circle intersection). Returns the branch
    nearest (near_q2, near_q3), or None if the target is out of reach.
    """
    tx = radial - CENCER_OFFSET - HEAD_OFFSET
    ty = z - CENCER_HEIGHT + HEAD_HEIGHT
    d = math.hypot(tx, ty)
    if (
        d < 1e-6
        or d > LINKAGE1 + LINKAGE2
        or d < abs(LINKAGE1 - LINKAGE2)
    ):
        return None

    cos_alpha = (LINKAGE1**2 + d * d - LINKAGE2**2) / (2 * LINKAGE1 * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)
    beta = math.atan2(ty, tx)

    best: tuple[float, float, float] | None = None
    for sign in (1.0, -1.0):
        q2 = math.degrees(beta + sign * alpha)
        p1x = LINKAGE1 * math.cos(math.radians(q2))
        p1y = LINKAGE1 * math.sin(math.radians(q2))
        q3 = math.degrees(math.atan2(ty - p1y, tx - p1x))
        dist = abs(_wrap_deg(q2 - near_q2)) + abs(_wrap_deg(q3 - near_q3))
        if best is None or dist < best[0]:
            best = (dist, q2, q3)
    assert best is not None
    return best[1], best[2]


def ik_position(
    x: float,
    y: float,
    z: float,
    *,
    near: JointAnglesDeg,
    hold_orientation: bool = True,
) -> JointAnglesDeg | None:
    """Full position IK: TCP (x, y, z) mm -> joint angles, elbow branch and
    J1 wrap chosen nearest `near`. When hold_orientation, J4 counters the J1
    change 1:1 (same convention as the jog IK); pass False to leave J4
    untouched.
    """
    q1 = math.degrees(math.atan2(y, x))
    q1 = near.j1 + _wrap_deg(q1 - near.j1)
    sol = ik_q2_q3(math.hypot(x, y), z, near.j2, near.j3)
    if sol is None:
        return None
    q4 = near.j4 - (q1 - near.j1) if hold_orientation else near.j4
    return JointAnglesDeg(q1, sol[0], sol[1], q4)


def steps_from_angles(q: JointAnglesDeg) -> tuple[int, int, int, int]:
    """Inverse of JointAnglesDeg.from_steps(): absolute step counters."""
    home = (HOME_J1_DEG, HOME_J2_DEG, HOME_J3_DEG, HOME_J4_DEG)
    vals = (q.j1, q.j2, q.j3, q.j4)
    return tuple(
        int(round((vals[i] - home[i]) * STEPS_PER_DEG[i] * J_STEP_SIGN[i]))
        for i in range(4)
    )  # type: ignore[return-value]
