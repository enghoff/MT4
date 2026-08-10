"""Unit tests for detection-as-state scene + shuffle planner (no hardware).

Run: python tests/test_scene.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.detect import CubeDetection
from mt4_vision.policy import plan_shuffle
from mt4_vision.scene import Scene, verify_pick_place
from mt4_vision.workspace import MarkerSlot, rebuild_workspace_state
from rig import CALIB, cube_area_at


MARKERS = [
    MarkerSlot(0, 52.0, -258.6),
    # Inside MAX_REACH_MM (320); live marker 1 was ~354mm and failed at safe_z.
    MarkerSlot(1, 40.0, 300.0),
    MarkerSlot(2, 188.4, -161.3),
    MarkerSlot(3, 177.2, 181.5),
]


def cube(
    color: str, x: float, y: float, area: float | None = None
) -> CubeDetection:
    """A detection of a real cube at (x, y), sized the way one there images.

    Pass ``area`` only when the test is about size; see rig.cube_area_at.
    """
    if area is None:
        area = cube_area_at(x, y)
    return CubeDetection(color=color, px=0.0, py=0.0, area=area, x=x, y=y)


def scene(
    cubes: list[CubeDetection],
    visible: set[int] | None = None,
) -> Scene:
    if visible is None:
        visible = {m.marker_id for m in MARKERS}
    state = rebuild_workspace_state(
        CALIB, MARKERS, cubes, visible_marker_ids=visible
    )
    return Scene.from_workspace(state, calib=CALIB)


def test_blocker_to_free_marker():
    s = scene([cube("red", 240.0, 100.0)], visible={0, 1, 2, 3})
    action = plan_shuffle(s)
    assert action.kind == "pick"
    assert action.place_kind == "to_marker"
    assert action.place_marker_id is not None
    assert action.cube is not None
    assert action.cube.color == "red"


def test_cube_near_marker_blocks_place_even_if_tag_decodes():
    # 35mm from marker 3: outside occupy radius, inside place clearance.
    s = scene([cube("red", 177.2, 181.5 + 35.0)], visible={0, 1, 2, 3})
    assert 3 not in {m.marker_id for m in s.placeable_markers()}
    action = plan_shuffle(s)
    assert action.place_marker_id != 3


def test_occupied_marker_not_placeable():
    s = scene(
        [cube("green", 177.2, 181.5)],
        visible={0, 1, 2},  # marker 3 tag hidden
    )
    assert 3 not in {m.marker_id for m in s.placeable_markers()}
    assert any(m.marker_id == 3 for m, _ in s.occupied)


def test_beyond_max_reach_marker_not_placeable():
    """Markers past MAX_REACH_MM must not be place targets.

    Keep-out alone is insufficient: live marker 1 (~354mm) is reachable at
    table_z but not at safe_z, so every transit failed with mp unreachable.
    """
    from mt4_vision.workspace import MAX_REACH_MM

    far = MarkerSlot(9, 58.0, 349.0)
    assert math.hypot(far.x, far.y) > MAX_REACH_MM
    markers = list(MARKERS) + [far]
    state = rebuild_workspace_state(CALIB, markers, [], visible_marker_ids={0, 1, 2, 3, 9}
    )
    s = Scene.from_workspace(state)
    placeable_ids = {m.marker_id for m in s.placeable_markers()}
    assert 9 not in placeable_ids
    assert 9 in {m.marker_id for m in s.free_markers}
    action = plan_shuffle(s)
    # No cubes to move -- but if something planned a place, never onto 9.
    if action.kind == "pick":
        assert action.place_marker_id != 9


def test_prefer_blocker_over_marker_cube():
    s = scene(
        [cube("red", 240.0, 100.0), cube("blue", 177.2, 181.5)],
        visible={0, 1, 2},  # 3 occupied; others free
    )
    action = plan_shuffle(s)
    assert action.kind == "pick"
    assert action.place_kind == "to_marker"
    assert action.cube is not None
    assert action.cube.color == "red"


def test_marker_to_slot_when_full():
    cubes = [
        cube("red", 52.0, -258.6),
        cube("blue", 188.4, -161.3),
        cube("green", 177.2, 181.5),
        cube("yellow", 39.0, 300.0),
    ]
    s = scene(cubes, visible=set())  # no tags => occupied / unknown, not free
    action = plan_shuffle(s)
    assert action.kind == "pick", action
    assert action.place_kind == "to_slot", action
    assert action.place_marker_id is None


def test_blocker_to_slot_when_no_free_markers():
    """Open-table cubes must still park onto free slots when every marker is
    unknown (typical: arm over the pad, ArUco tags occluded)."""
    s = scene([cube("red", 240.0, 100.0)], visible=set())
    assert s.free_markers == []
    assert s.blockers()
    assert s.free_slots
    action = plan_shuffle(s)
    assert action.kind == "pick", action
    assert action.place_kind == "to_slot", action
    assert action.cube is not None
    assert action.cube.color == "red"


def test_marker_cube_to_free_marker_when_no_blocker():
    # Free markers 0,1,2; cube only on marker 3 -- relocate onto a free marker.
    s = scene(
        [cube("green", 177.2, 181.5)],
        visible={0, 1, 2},
    )
    action = plan_shuffle(s)
    assert action.kind == "pick", action
    assert action.place_kind == "to_marker", action
    assert action.place_marker_id in {0, 1, 2}
    assert action.cube is not None
    assert action.cube.color == "green"


def test_camera_park_adjacent_cube_is_pickable():
    """A cube near the camera-park pose (200,0) is a normal pick target.

    An exclusion there would hide the arm's own silhouette when it retreats to
    (200,0) between captures via retreat_for_camera(), but the live loop
    (shuffle.py) never calls that, so the position is not special and must not
    veto a real cube -- it would permanently exclude marker 4, ~42mm away, and
    two PLACEMENT_SLOTS entries."""
    from mt4_vision.scene import is_phantom_detection

    near_park = cube("green", 193.0, -51.0)
    assert not is_phantom_detection(near_park, CALIB)


def test_raw_near_marker_blocks_place_even_if_filtered_from_picks():
    """Occupancy uses raw blobs; a too-small blob still blocks place."""
    small = cube("green", 177.2, 181.5, area=150.0)
    state = rebuild_workspace_state(
        CALIB, MARKERS, [small], visible_marker_ids={0, 1, 2, 3}
    )
    assert any(m.marker_id == 3 for m, _ in state.occupied)
    assert 3 not in {m.marker_id for m in state.free_markers}
    s = Scene.from_workspace(state, pick_cubes=[], raw_cubes=[small])
    assert 3 not in {m.marker_id for m in s.placeable_markers()}
    assert plan_shuffle(s).kind == "wait"


def test_vacated_pose_not_planned_after_fresh_scene():
    """Detection-as-state: a gone cube simply isn't in the next scene."""
    before = scene([cube("red", 240.0, 100.0)], visible={0, 1, 2, 3})
    assert plan_shuffle(before).kind == "pick"
    after = scene([], visible={0, 1, 2, 3})
    assert plan_shuffle(after).kind == "wait"
    assert after.cubes == []


def test_off_desk_blob_filtered_as_phantom():
    from mt4_vision.scene import filter_phantoms, is_phantom_detection

    # Behind the desk's back edge: the arm's own body images up there on this
    # oblique mount, which is what this gate is for. (272,-188) is not such a
    # case: it sits outside the marker hull, but it is a perfectly good desk
    # location and a pick target.
    phantom = cube("red", -100.0, 250.0)
    real = cube("green", 177.2, 181.5)
    assert is_phantom_detection(phantom, CALIB)
    assert not is_phantom_detection(real, CALIB)
    kept = filter_phantoms([phantom, real], CALIB)
    assert [c.color for c in kept] == ["green"]


def test_far_desk_blob_is_a_pick_target_now():
    """What the work region exists to prevent.

    Measured on a live frame 2026-08-02: three cubes plainly on the desk and
    plainly in reach, all of them outside the marker-centre hull -- which is
    enough to drop them from the snapshot entirely. These are their measured
    positions.
    """
    from mt4_vision.scene import is_phantom_detection

    for x, y in ((176.1, -213.6), (266.5, -52.7), (132.8, 272.0)):
        assert not is_phantom_detection(cube("blue", x, y), CALIB), (x, y)


def test_a_place_target_must_survive_being_read_a_little_off():
    """Coverage at the slot centre is not coverage of the cube.

    (270,0) passes camera_covers exactly; a cube placed there was read at
    (272,-7) by the live detector, which does not, so it was in frame by the
    slot's reckoning and out of frame by the cube's -- and nothing picked it up
    again. A place target therefore has to hold coverage over the disc the cube
    will really be read in, while a cube already detected is judged where it
    actually reads (else visible cubes near the edge get refused).
    """
    from mt4_vision.workspace import (
        CUBE_READ_TOLERANCE_MM,
        free_placement_slots,
        in_work_region,
    )

    assert in_work_region(270.0, 0.0, CALIB), "the centre itself is covered"
    assert not in_work_region(
        270.0, 0.0, CALIB, read_tolerance_mm=CUBE_READ_TOLERANCE_MM
    ), "but not the disc a cube placed there would be read in"

    offered = free_placement_slots(CALIB, [], [])
    assert (270.0, 0.0) not in offered
    assert (150.0, 100.0) in offered, "a mid-table slot is unaffected"


def test_keepout_blob_filtered():
    from mt4_vision.scene import is_phantom_detection

    # The arm base's own hardware reads as small blue blobs near the column.
    # (-19, 161) is not a case for this test: at r=162mm it is comfortably
    # outside the keep-out and a legitimate pick target, so the point below is
    # one actually inside the cylinder.
    assert is_phantom_detection(cube("blue", 60.0, 60.0), CALIB)


def test_verify_pick_place_outcomes():
    placed = scene([cube("blue", 177.0, 181.0)], visible={0, 1, 2})
    assert (
        verify_pick_place(
            placed,
            pick_x=240.0,
            pick_y=0.0,
            pick_color="blue",
            place_x=177.2,
            place_y=181.5,
        )
        == "placed"
    )
    failed = scene([cube("blue", 241.0, 101.0)], visible={0, 1, 2, 3})
    assert (
        verify_pick_place(
            failed,
            pick_x=240.0,
            pick_y=100.0,
            pick_color="blue",
            place_x=177.2,
            place_y=181.5,
        )
        == "grasp_failed"
    )
    lost = scene([cube("blue", 100.0, 100.0)], visible={0, 1, 2, 3})
    assert (
        verify_pick_place(
            lost,
            pick_x=240.0,
            pick_y=100.0,
            pick_color="blue",
            place_x=177.2,
            place_y=181.5,
        )
        == "lost"
    )


def test_avoid_last_cube_when_alternative_exists():
    s = scene(
        [cube("red", 240.0, 100.0), cube("blue", 240.0, -150.0)],
        visible={0, 1, 2, 3},
    )
    action = plan_shuffle(s, avoid_xy=(240.0, 100.0))
    assert action.kind == "pick"
    assert action.cube is not None
    assert action.cube.color == "blue"


def test_avoid_last_cube_falls_back_when_sole_option():
    s = scene([cube("red", 240.0, 100.0)], visible={0, 1, 2, 3})
    action = plan_shuffle(s, avoid_xy=(240.0, 100.0))
    assert action.kind == "pick"
    assert action.cube is not None
    assert action.cube.color == "red"


def test_lookahead_second_move_visible_in_same_capture():
    """Two blockers, two free markers: a second independent move should be
    plannable from the same scene by excluding the first action's cube and
    destination marker -- this is the basis for shuffle.py's capture skip."""
    from mt4_vision.shuffle import _lookahead_action

    s = scene(
        [cube("red", 240.0, 100.0), cube("blue", 240.0, -150.0)],
        visible={0, 1, 2, 3},
    )
    first = plan_shuffle(s)
    assert first.kind == "pick"
    second = _lookahead_action(s, first, avoid_xy=None)
    assert second is not None
    assert second.kind == "pick"
    assert second.cube is not first.cube
    assert second.place_marker_id != first.place_marker_id


def test_blocker_choice_is_randomized_not_always_first():
    """Picking candidates must not collapse to always-the-first-in-list,
    which shows up as the same move repeating cycle after cycle."""
    s = scene(
        [
            cube("red", 240.0, 100.0),
            cube("blue", 240.0, -150.0),
            cube("yellow", 150.0, 250.0),
        ],
        visible={0, 1, 2, 3},
    )
    seen = set()
    for _ in range(200):
        action = plan_shuffle(s)
        if action.kind == "pick" and action.cube is not None:
            seen.add(action.cube.color)
    assert len(seen) > 1, f"expected varied picks across trials, got {seen}"


def test_lookahead_none_when_only_one_move_available():
    from mt4_vision.shuffle import _lookahead_action

    s = scene([cube("red", 240.0, 100.0)], visible={0, 1, 2, 3})
    first = plan_shuffle(s)
    assert first.kind == "pick"
    assert _lookahead_action(s, first, avoid_xy=None) is None


def run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())


def test_marker_at_touched_reach_limit_is_placeable():
    """Marker 1 lives at 322.5mm -- physically touched and reachable; the
    reach filter must not veto it (MAX_REACH_MM is 350; a 320 cap vetoes
    it)."""
    from mt4_vision.workspace import MarkerSlot, rebuild_workspace_state

    m1 = MarkerSlot(1, 45.0, 319.3)
    state = rebuild_workspace_state(CALIB, [m1], [], visible_marker_ids={1})
    s = Scene.from_workspace(state, pick_cubes=[], raw_cubes=[])
    assert 1 in {m.marker_id for m in s.placeable_markers()}
