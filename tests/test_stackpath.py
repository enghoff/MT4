"""Pure-geometry tests for mt4_vision.stackpath (no hardware)."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_vision.stackpath import (
    COLUMN_AVOID_MM,
    HOVER_ABOVE_MIN_MM,
    STAGE_OFFSET_MM,
    StackPlanner,
    joint_reachable,
    max_z_at,
)

# Live calibration values (vision_calibration.json, 2026-07): 20mm cubes,
# pick at 127.2. Marker 3 is a real stack site (~220mm from the base).
CALIB = SimpleNamespace(table_z=127.2, cube_height_mm=20.0, safe_z=155.0)
MARKER3 = (153.6, 156.9)


def planner() -> StackPlanner:
    return StackPlanner(CALIB, *MARKER3)


def test_joint_ceiling_band_at_marker_radius():
    # J3 soft max caps TCP height ~315mm at the marker radii; plain
    # link-geometry reach would claim far more.
    top = max_z_at(*MARKER3)
    assert top is not None
    assert 300.0 < top < 330.0


def test_level9_buildable_level10_not():
    p = planner()
    hz9 = p.hover_z(9)
    assert hz9 is not None
    assert hz9 >= p.release_z(9) + HOVER_ABOVE_MIN_MM
    assert p.hover_z(10) is None


def test_retreat_mode_free_until_level8_slide_at_9():
    p = planner()
    assert p.free_retreat_z(8) is not None
    assert p.free_retreat_z(9) is None  # needs ~324mm TCP -- over the ceiling
    sz = p.slide_z(9)
    assert p.release_z(9) < sz <= p.site_max_z


def test_hover_clearance_every_buildable_level():
    p = planner()
    for level in range(1, 10):
        hz = p.hover_z(level)
        assert hz is not None, f"level {level} should be buildable"
        assert hz >= p.release_z(level) + HOVER_ABOVE_MIN_MM
        assert joint_reachable(p.sx, p.sy, hz)


def test_pose_safe_blocks_forearm_over_column():
    p = planner()
    sx, sy = MARKER3
    rs = math.hypot(sx, sy)
    # 50mm beyond the stack along its bearing, at safe_z: the forearm
    # sweeps through an 8-cube column.
    bx, by = sx * (1 + 50.0 / rs), sy * (1 + 50.0 / rs)
    assert joint_reachable(bx, by, CALIB.safe_z)
    assert not p.pose_safe(bx, by, CALIB.safe_z, levels=8)
    # High enough (and still under the ceiling), the forearm clears.
    assert p.pose_safe(bx, by, 300.0, levels=8)
    # On the base side of the stack the forearm never reaches the column.
    nx, ny = sx * (1 - 70.0 / rs), sy * (1 - 70.0 / rs)
    assert p.pose_safe(nx, ny, CALIB.safe_z, levels=8)


def test_pose_safe_blocks_low_pass_through_column():
    p = planner()
    sx, sy = MARKER3
    assert not p.pose_safe(sx + 20.0, sy, CALIB.safe_z, levels=5)
    # Same XY above the stack top with carry clearance is fine.
    assert p.pose_safe(sx + 20.0, sy, p.grip_top_z(5) + 10.0, levels=5)
    # No stack, no restriction.
    assert p.pose_safe(sx + 20.0, sy, CALIB.safe_z, levels=0)


def test_route_direct_when_clear_of_column():
    p = planner()
    a = (200.0, 0.0, 260.0)
    b = (240.0, -150.0, 155.0)
    assert p.route(a, b, levels=8) == [b]


def test_route_detours_when_direct_crosses_column():
    p = planner()
    sx, sy = MARKER3
    # Both endpoints are wedge- and column-safe, but the straight chord
    # passes through the column: the route must ring-walk the near side.
    a = (240.0, 60.0, 155.0)
    b = (110.0, 220.0, 155.0)
    ab = (b[0] - a[0], b[1] - a[1])
    cross = ab[0] * (sy - a[1]) - ab[1] * (sx - a[0])
    assert abs(cross) / math.hypot(*ab) < COLUMN_AVOID_MM
    wps = p.route(a, b, levels=8)
    assert wps is not None
    assert wps[-1] == b
    assert len(wps) >= 2
    for x, y, _z in wps[:-1]:
        assert math.hypot(x - sx, y - sy) >= COLUMN_AVOID_MM - 0.1


def test_route_detours_around_keepout_field_case():
    """Field case 2026-07-24 (marker 2, level 5): the straight chord from
    pick (59,179) to stage (160,-90) bottoms out at r~121 inside the J1
    keep-out; the router must detour via the base ring instead of failing.
    Every returned hop must keep chords clear of the keep-out so the
    firmware executes exactly the checked path (no hidden arcs)."""
    from mt4_vision.workspace import KEEPOUT_RADIUS_MM

    p = StackPlanner(CALIB, 161.9, -149.6)  # marker 2
    hz = p.hover_z(5)
    assert hz is not None
    a = (58.7, 179.3, CALIB.safe_z)
    b = (160.0, -90.0, hz)
    wps = p.route(a, b, levels=4)
    assert wps is not None
    assert wps[-1] == b
    prev = a
    for wp in wps:
        seg = (wp[0] - prev[0], wp[1] - prev[1])
        for i in range(21):
            t = i / 20
            x = prev[0] + seg[0] * t
            y = prev[1] + seg[1] * t
            assert math.hypot(x, y) >= KEEPOUT_RADIUS_MM - 0.6, (
                f"hop {prev}->{wp} dips into keep-out at ({x:.0f},{y:.0f})"
            )
        prev = wp


def test_stage_point_avoids_low_ceiling_bearings_at_level9():
    p = planner()
    hz = p.hover_z(9)
    assert hz is not None
    st = p.stage_point(hz, 8, prefer_xy=(280.0, 0.0))
    assert st is not None
    # The chosen stage must genuinely reach hover height.
    assert joint_reachable(st[0], st[1], hz)
    assert math.isclose(
        math.hypot(st[0] - p.sx, st[1] - p.sy), STAGE_OFFSET_MM, abs_tol=0.1
    )


def test_slide_exits_perpendicular_to_jaws_prefer_park_side():
    p = planner()
    exits = p.slide_exits(0.0, 9, prefer_xy=(200.0, 0.0))
    assert exits
    for ex in exits:
        # Perpendicular to a world-x jaw axis: pure +/-y offset.
        assert math.isclose(ex[0], p.sx, abs_tol=1e-6)
        assert math.isclose(abs(ex[1] - p.sy), STAGE_OFFSET_MM, abs_tol=1e-6)
    # Park side (-y of marker 3) ranks first.
    assert exits[0][1] < p.sy
