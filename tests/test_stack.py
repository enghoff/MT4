"""Pure-logic tests for stack_cubes helpers (no hardware)."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The frozen rig, never the live calibration -- including for the camera-shadow
# field cases below, which are the ones that most look like they want real
# numbers. Two things they need are exactly what a live file may not have:
# ``cube_top_homography``, without which ``stack_shadow_behind_unit`` returns
# None and there is no corridor to assert about at all, and a camera nadir
# matching the pose the case was recorded at, since the corridor is drawn from
# it. Read live, these tests report the state of somebody's desk. See ``rig``.
from rig import CALIB
from stack_cubes import (
    CLEAR_PARK_MM,
    SITE_CLEAR_MM,
    choose_park_slot,
    clear_aside_xy,
    cubes_near_site,
    in_stack_camera_shadow,
    release_z_for_level,
    stack_candidates,
    stack_shadow_behind_unit,
)


def test_cubes_near_site_filters_by_radius():
    near = SimpleNamespace(x=200.0, y=60.0, color="red")
    far = SimpleNamespace(x=280.0, y=0.0, color="blue")
    scene = SimpleNamespace(raw_cubes=[near, far])
    found = cubes_near_site(scene, 200.0, 55.0, radius_mm=70.0)
    assert found == [near]


def test_clear_aside_pushes_past_keep_clear():
    # Marker 4-ish; cube slightly off-center toward +y.
    dest = clear_aside_xy(243.0, 5.0, 251.0, 27.0, [], CALIB)
    assert dest is not None
    assert math.hypot(dest[0] - 243.0, dest[1] - 5.0) >= CLEAR_PARK_MM - 0.1
    # Must not land in the barely-outside free-slot ring that re-triggered clear.
    assert math.hypot(dest[0] - 243.0, dest[1] - 5.0) > SITE_CLEAR_MM + 20.0


def test_clear_aside_avoids_occupied():
    sx, sy = 243.0, 5.0
    cx, cy = 251.0, 27.0
    primary = clear_aside_xy(sx, sy, cx, cy, [], CALIB)
    assert primary is not None
    alt = clear_aside_xy(sx, sy, cx, cy, [primary], CALIB)
    assert alt is not None
    assert math.hypot(alt[0] - primary[0], alt[1] - primary[1]) >= 40.0


def test_clear_aside_stays_in_pick_hull():
    from mt4_vision.workspace import MarkerSlot, in_work_region

    # Tight triangle around the site; a long push along +y exits the hull.
    markers = [
        MarkerSlot(1, 200.0, 0.0),
        MarkerSlot(2, 280.0, 0.0),
        MarkerSlot(3, 240.0, 80.0),
    ]
    dest = clear_aside_xy(
        240.0, 20.0, 240.0, 40.0, [], CALIB, markers=markers,
    )
    assert dest is not None
    # Markers do not bound the work area, so what a clear must guarantee is
    # that the cube lands somewhere the arm can pick it up again.
    assert in_work_region(dest[0], dest[1], CALIB)


def test_clear_aside_skips_stack_shadow_corridor():
    calib = CALIB
    sx, sy = 178.7, 179.8
    behind = stack_shadow_behind_unit(calib, sx, sy)
    assert behind is not None
    # Cube already in the behind corridor; landing must not stay there.
    cx = sx + behind[0] * 40.0
    cy = sy + behind[1] * 40.0
    dest = clear_aside_xy(
        sx, sy, cx, cy, [], calib, behind_u=behind, shadow_levels=8,
    )
    assert dest is not None
    assert not in_stack_camera_shadow(
        dest[0], dest[1], sx, sy, behind, stack_levels=8,
    )


def test_clear_aside_stays_out_of_arm_occlusion_strip():
    """Clears must not land near the keep-out where the parked arm hides
    them from the camera (field case: (134,49) vanished from scans)."""
    from stack_cubes import CLEAR_MIN_RADIUS_MM

    # Cube on the base side of marker 3: the straight push aims at the
    # occlusion strip, so an alternative landing must be chosen.
    dest = clear_aside_xy(153.6, 156.9, 120.0, 120.0, [], CALIB)
    assert dest is not None
    assert math.hypot(dest[0], dest[1]) >= CLEAR_MIN_RADIUS_MM


def test_choose_park_slot_requires_clear_margin():
    # (200, 60) is ~70mm from marker 4 -- inside CLEAR_PARK_MM, must reject.
    scene = SimpleNamespace(
        free_slots=[(200.0, 60.0), (200.0, -60.0), (150.0, -250.0)],
    )
    spot = choose_park_slot(scene, 243.0, 5.0, CALIB)
    assert spot == (150.0, -250.0)


def test_stack_candidates_exclude_site_and_use_pickable():
    near = SimpleNamespace(x=205.0, y=60.0, color="red", yaw_deg=0.0)
    far = SimpleNamespace(x=280.0, y=0.0, color="green", yaw_deg=10.0)
    scene = SimpleNamespace(
        cubes=[near, far],
        pickable=lambda cubes: [far],
    )
    assert stack_candidates(scene, 200.0, 60.0) == [far]


def test_stack_candidates_hold_back_cubes_behind_the_standing_column():
    """A real cube radially beyond the column on its own bearing cannot be
    approached -- the forearm would cross over the stack. Taking it as the
    next pick fails the approach transit with no route and ends the run, so
    it must drop out of the candidate list while a neighbour stays in."""
    from mt4_vision.stackpath import StackPlanner

    calib = CALIB
    sx, sy = 153.6, 156.9
    planner = StackPlanner(calib, sx, sy)
    behind = SimpleNamespace(x=202.0, y=239.0, color="red", yaw_deg=0.0)
    beside = SimpleNamespace(x=250.0, y=-134.0, color="blue", yaw_deg=0.0)
    scene = SimpleNamespace(
        cubes=[behind, beside],
        pickable=lambda cubes: list(cubes),
    )
    cands = stack_candidates(
        scene, sx, sy, calib=calib, stack_levels=3, planner=planner,
    )
    assert behind not in cands
    assert beside in cands
    # Nothing standing yet: the same cube is a fine pick.
    assert behind in stack_candidates(
        scene, sx, sy, calib=calib, stack_levels=0, planner=planner,
    )


def test_stack_shadow_rejects_marker3_phantom():
    """Field case 2026-07-21: stack (179,180) → phantom ~(115,227)."""
    calib = CALIB
    sx, sy = 178.7, 179.8
    behind = stack_shadow_behind_unit(calib, sx, sy)
    assert behind is not None
    assert in_stack_camera_shadow(
        115.0, 227.0, sx, sy, behind, stack_levels=4,
    )
    # A cube off to the side of the corridor must still be pickable.
    assert not in_stack_camera_shadow(
        280.0, 0.0, sx, sy, behind, stack_levels=4,
    )
    phantom = SimpleNamespace(x=115.0, y=227.0, color="green", yaw_deg=0.0)
    real = SimpleNamespace(x=250.0, y=96.0, color="red", yaw_deg=0.0)
    scene = SimpleNamespace(
        cubes=[phantom, real],
        pickable=lambda cubes: list(cubes),
    )
    cands = stack_candidates(
        scene, sx, sy, calib=calib, stack_levels=4,
    )
    assert phantom not in cands
    assert real in cands


def test_stack_shadow_lateral_widens_with_level():
    """Field case 2026-07-24, marker 2 level 6: true site (161.9,-149.6),
    phantom read at (4.4,-203.7) -- 49mm lateral, past a fixed 45mm corridor
    width (calibrated from an 8mm lateral offset at level 4). The
    tolerance must widen with stack height on both axes, not just along."""
    calib = CALIB
    sx, sy = 161.9, -149.6
    behind = stack_shadow_behind_unit(calib, sx, sy)
    assert behind is not None
    assert in_stack_camera_shadow(4.4, -203.7, sx, sy, behind, stack_levels=6)
    # At low levels the corridor stays narrow -- a cube this far laterally off
    # the LOS at level 1 is a real, pickable cube.
    assert not in_stack_camera_shadow(4.4, -203.7, sx, sy, behind, stack_levels=1)


def test_pick_missed_detects_shoved_cube():
    """Field case 2026-07-24: pick at (55.8,195) shoved the red to (68,203)
    instead of gripping -- the same-color detection near the pick spot is
    the miss signature. A same-color cube 45mm+ away (another cube's legal
    clearance) must not trigger it."""
    from stack_cubes import pick_missed

    shoved = SimpleNamespace(x=68.1, y=202.7, color="red")
    other = SimpleNamespace(x=110.0, y=195.0, color="red")
    scene = SimpleNamespace(raw_cubes=[shoved, other])
    assert pick_missed(scene, ("red", 55.8, 195.0)) == (68.1, 202.7)
    assert pick_missed(scene, ("blue", 55.8, 195.0)) is None
    assert pick_missed(SimpleNamespace(raw_cubes=[other]), ("red", 55.8, 195.0)) is None
    assert pick_missed(scene, None) is None


def test_release_z_steps_by_cube_height():
    calib = SimpleNamespace(table_z=150.0, safe_z=185.0, cube_height_mm=20.0)
    # 4mm above stack top: empty / 1-cube / 2-cube
    assert release_z_for_level(calib, 1) == 154.0
    assert release_z_for_level(calib, 2) == 174.0
    assert release_z_for_level(calib, 3) == 194.0


def test_stack_clear_xy_prefers_approach_ray():
    from mt4_vision.pickplace import STACK_AXIS_CLEAR_MM, stack_clear_xy

    sx, sy = 211.0, 7.0
    # Approached from +Y (cube side); clear point should stay near that ray.
    clear = stack_clear_xy(sx, sy, 211.0, 80.0, STACK_AXIS_CLEAR_MM)
    assert clear is not None
    assert abs(math.hypot(clear[0] - sx, clear[1] - sy) - STACK_AXIS_CLEAR_MM) < 0.1
    assert clear[1] > sy  # same half-plane as the approach


def test_place_on_stack_queues_release_as_station():
    """Carry+seat ends on an open station in the same mq -- no blocking
    client.gripper() round trip between descend and retreat."""
    from mt4_vision.stackpath import StackPlanner
    from stack_cubes import place_on_stack

    calib = SimpleNamespace(
        table_z=127.2,
        cube_height_mm=20.0,
        safe_z=155.0,
        travel_speed_us=700,
        approach_speed_us=2400,
        grip_open_s=140,
        grip_close_s=240,
    )
    sx, sy = 153.6, 156.9
    planner = StackPlanner(calib, sx, sy)

    class _Tcp:
        def __init__(self):
            self.x, self.y, self.z, self.j4 = 200.0, 0.0, 155.0, 0.0

    class _Client:
        def __init__(self):
            self._tcp = _Tcp()
            self.calls = []
            self.gripper_calls = []

        def get_tcp(self):
            return self._tcp

        def gripper(self, action):
            self.gripper_calls.append(action)
            return {"ok": True}

        def move_path(self, waypoints, j4=None, grip=0, speed_us=0, dwell_ms=0, **kw):
            n = len(waypoints)

            def spread(v):
                return list(v) if isinstance(v, (list, tuple)) else [v] * n

            self.calls.append(
                {
                    "wps": list(waypoints),
                    "grips": list(grip) if isinstance(grip, (list, tuple))
                    else [grip] + [0] * (n - 1),
                    "dwells": spread(dwell_ms),
                }
            )
            return {"ok": True}

    client = _Client()
    place_on_stack(client, calib, planner, 1, park_xy=(200.0, 0.0))
    assert client.gripper_calls == []
    assert client.calls, "expected at least the carry+seat queue"
    seat = client.calls[0]
    assert seat["wps"][-1] == (sx, sy, planner.release_z(1))
    assert seat["dwells"][-1] > 0
    assert seat["grips"][-1] == calib.grip_open_s
