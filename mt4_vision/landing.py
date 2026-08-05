"""Shared reachable-table XY search for stack clear/park and unstack scatter.

Both scripts need the same underlying gate -- a point the arm can reach, the
camera can still see, clear of markers and cubes -- and one copy of it is the
point of this module. Two hand-tuned copies drift apart, and the drift surfaces
as ``CLAUDE.md``'s "no reachable clear spot". The search *strategies* still
differ (directional push vs random annulus); only the predicate and the shared
floors live here.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence

from mt4_vision.calib import Calibration
from mt4_vision.pickplace import near_camera_park
from mt4_vision.workspace import (
    MARKER_PAPER_CLEARANCE_MM,
    MAX_REACH_MM,
    MarkerSlot,
    dist_mm,
    in_work_region,
    is_mp_reachable_xy,
)

# Floor shared by stack clear and unstack scatter. Field case 2026-07-24: a
# cube parked close to the J1 keep-out was occluded by the arm's own
# camera-park silhouette and never seen again by later scans.
LANDING_MIN_RADIUS_MM = 170.0
# Outer bound for a landing, and a reach bound only. Camera coverage on this
# oblique mount is not a circle about the base, so ``in_work_region``'s frame
# projection is what decides how far out a cube may be put down. This constant
# only stops the search wasting candidates past any possible reach.
LANDING_MAX_RADIUS_MM = MAX_REACH_MM
# Finger clearance from other cubes when parking.
LANDING_SEP_MM = 45.0


def landing_ok(
    x: float,
    y: float,
    calib: Calibration,
    *,
    occupied: Sequence[tuple[float, float]] = (),
    sep_mm: float = LANDING_SEP_MM,
    markers: Sequence[MarkerSlot] | None = None,
    marker_clearance_mm: float | None = None,
    site_xy: tuple[float, float] | None = None,
    site_avoid_mm: float = 0.0,
    min_radius_mm: float = LANDING_MIN_RADIUS_MM,
    max_radius_mm: float = MAX_REACH_MM,
    avoid_camera_park: bool = False,
    blocked: Callable[[float, float], bool] | None = None,
) -> bool:
    """True when (x, y) is a workable, unconflicted table landing.

    ``calib`` is positional and required on purpose: the work region is not
    optional for a landing. Putting a cube where it cannot be picked up again
    is the exact failure this module exists to prevent, and a gate a caller can
    switch off is one that will be switched off.

    ``blocked`` is an optional extra veto (stack's camera-LOS shadow corridor).
    ``marker_clearance_mm`` defaults to off so stack can keep relying on the
    work-region gate alone; unstack passes ``MARKER_PAPER_CLEARANCE_MM``.
    """
    if not is_mp_reachable_xy(x, y):
        return False
    r = math.hypot(x, y)
    if r > max_radius_mm or r < min_radius_mm:
        return False
    if not in_work_region(x, y, calib):
        return False
    if avoid_camera_park and near_camera_park(x, y):
        return False
    if site_xy is not None and site_avoid_mm > 0.0:
        if dist_mm(x, y, site_xy[0], site_xy[1]) < site_avoid_mm:
            return False
    if any(dist_mm(x, y, ox, oy) < sep_mm for ox, oy in occupied):
        return False
    if markers is not None and marker_clearance_mm is not None:
        if any(dist_mm(x, y, m.x, m.y) < marker_clearance_mm for m in markers):
            return False
    if blocked is not None and blocked(x, y):
        return False
    return True


def expanding_angles_deg(step: float = 20.0) -> tuple[float, ...]:
    """0, then +-step, +-2*step, ... out to +-180 (straightest push first)."""
    angles = [0.0]
    a = step
    while a < 180.0:
        angles.append(a)
        angles.append(-a)
        a += step
    angles.append(180.0)
    return tuple(angles)


def annulus_grid(
    *,
    min_radius_mm: float = LANDING_MIN_RADIUS_MM,
    max_radius_mm: float = MAX_REACH_MM,
    step_mm: float = 40.0,
) -> list[tuple[float, float]]:
    """Rings of points spanning the reachable annulus."""
    pts: list[tuple[float, float]] = []
    r = min_radius_mm
    while r <= max_radius_mm:
        steps = max(1, round(2 * math.pi * r / step_mm))
        for i in range(steps):
            ang = 2 * math.pi * i / steps
            pts.append((r * math.cos(ang), r * math.sin(ang)))
        r += step_mm
    return pts


def push_aside_xy(
    sx: float,
    sy: float,
    cx: float,
    cy: float,
    occupied: Sequence[tuple[float, float]],
    calib: Calibration,
    *,
    park_mm: float,
    park_radii_extra_mm: Sequence[float] = (25.0, 45.0),
    angles_deg: Sequence[float] | None = None,
    markers: Sequence[MarkerSlot] | None = None,
    blocked: Callable[[float, float], bool] | None = None,
    sep_mm: float = LANDING_SEP_MM,
    min_radius_mm: float = LANDING_MIN_RADIUS_MM,
) -> tuple[float, float] | None:
    """Push away from the site along site→cube, with angle/radius fallbacks."""
    dx, dy = cx - sx, cy - sy
    r = math.hypot(dx, dy)
    if r < 1.0:
        dx, dy = sx, sy
        r = math.hypot(dx, dy) or 1.0
    ux, uy = dx / r, dy / r
    radii = (park_mm, *(park_mm + extra for extra in park_radii_extra_mm))
    angles = angles_deg if angles_deg is not None else expanding_angles_deg()
    for dist in radii:
        for angle_deg in angles:
            ang = math.radians(angle_deg)
            ca, sa = math.cos(ang), math.sin(ang)
            vx, vy = ux * ca - uy * sa, ux * sa + uy * ca
            tx, ty = sx + vx * dist, sy + vy * dist
            if landing_ok(
                tx, ty, calib,
                occupied=occupied, sep_mm=sep_mm, markers=markers,
                min_radius_mm=min_radius_mm, blocked=blocked,
            ):
                return (tx, ty)
    return None


def nearest_landing(
    sx: float,
    sy: float,
    calib: Calibration,
    *,
    preferred: Iterable[tuple[float, float]],
    fallback: Iterable[tuple[float, float]] = (),
    site_clear_mm: float,
    occupied: Sequence[tuple[float, float]] = (),
    markers: Sequence[MarkerSlot] | None = None,
    blocked: Callable[[float, float], bool] | None = None,
    sep_mm: float = LANDING_SEP_MM,
    min_radius_mm: float = LANDING_MIN_RADIUS_MM,
) -> tuple[float, float] | None:
    """Nearest candidate to the site that clears ``site_clear_mm`` and gates."""

    def _valid(x: float, y: float) -> bool:
        return dist_mm(x, y, sx, sy) >= site_clear_mm and landing_ok(
            x, y, calib,
            occupied=occupied, sep_mm=sep_mm, markers=markers,
            min_radius_mm=min_radius_mm, blocked=blocked,
        )

    candidates = [(x, y) for x, y in preferred if _valid(x, y)]
    if not candidates:
        candidates = [(x, y) for x, y in fallback if _valid(x, y)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: dist_mm(p[0], p[1], sx, sy))


def random_landing(
    rng: random.Random,
    calib: Calibration,
    *,
    sx: float,
    sy: float,
    markers: Sequence[MarkerSlot],
    avoid: Sequence[tuple[float, float]],
    spacing_mm: float,
    site_avoid_mm: float,
    attempts: int = 300,
    min_radius_mm: float = LANDING_MIN_RADIUS_MM,
    max_radius_mm: float = LANDING_MAX_RADIUS_MM,
    marker_clearance_mm: float = MARKER_PAPER_CLEARANCE_MM,
    blocked: Callable[[float, float], bool] | None = None,
) -> tuple[float, float] | None:
    """Random reachable table XY clear of site/markers/avoid, or None.

    ``blocked`` is the same extra veto ``landing_ok`` takes -- unstack passes
    the standing column's forearm shadow through it.
    """
    for _ in range(attempts):
        r = rng.uniform(min_radius_mm, max_radius_mm)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        x, y = r * math.cos(theta), r * math.sin(theta)
        if landing_ok(
            x, y, calib,
            occupied=avoid, sep_mm=spacing_mm, markers=markers,
            marker_clearance_mm=marker_clearance_mm,
            site_xy=(sx, sy), site_avoid_mm=site_avoid_mm,
            min_radius_mm=min_radius_mm, max_radius_mm=max_radius_mm,
            avoid_camera_park=True, blocked=blocked,
        ):
            return (x, y)
    return None


# The spacing-degradation loop that wraps random_landing deliberately does NOT
# live here: unstack_cubes owns it so its tests can monkeypatch that module's
# random_landing. A copy here would be dead code drifting from the live one --
# which is the failure this module exists to prevent.
