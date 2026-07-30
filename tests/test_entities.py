"""Tests for the addressable entity layer (no hardware, no camera).

Two properties matter most and both are pinned here:

* ``pickable`` must agree exactly with the pick path (``Scene.pickable`` over
  the phantom-filtered set). A snapshot that advertises a target the pick path
  would refuse is worse than one that says nothing.
* every non-pickable detection must carry a ``reason``, because the defect this
  layer replaces was silence -- a rejected blob simply not being reported, so
  "that is the arm's own paint, 78mm outside the marker hull" came back as
  "no red cube in view".

Run: python tests/test_entities.py  (or pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.calib import GRIP_SQUEEZE_MM
from mt4_vision.detect import CubeDetection
from mt4_vision.entities import (
    KIND_CUBE,
    KIND_MARKER,
    KIND_OBJECT,
    KIND_SLOT,
    build_snapshot,
    pick_block_reason,
)
from mt4_vision.motion import YAW_PERIOD_LONG_AXIS, YAW_PERIOD_SQUARE
from mt4_vision.scene import PICK_MAX_AREA, PICK_MIN_AREA, Scene, filter_phantoms
from mt4_vision.workspace import (
    MAX_REACH_MM,
    MarkerSlot,
    PICK_CLEARANCE_MM,
    rebuild_workspace_state,
)

MARKERS = [
    MarkerSlot(0, 52.0, -258.6),
    MarkerSlot(1, 40.0, 300.0),
    MarkerSlot(2, 188.4, -161.3),
    MarkerSlot(3, 177.2, 181.5),
]
GOOD_AREA = 2500.0


def cube(color: str, x: float, y: float, area: float = GOOD_AREA) -> CubeDetection:
    return CubeDetection(
        color=color, px=100.0, py=200.0, area=area, x=x, y=y, yaw_deg=12.0
    )


def scene(cubes, visible=None) -> Scene:
    """A Scene built the way capture_scene builds one: raw list plus the
    phantom-filtered pick subset."""
    if visible is None:
        visible = {m.marker_id for m in MARKERS}
    state = rebuild_workspace_state(None, MARKERS, cubes, visible_marker_ids=visible)
    return Scene.from_workspace(
        state, pick_cubes=filter_phantoms(cubes, MARKERS), raw_cubes=cubes
    )


def snap(cubes, visible=None, objects=()):
    return build_snapshot(scene(cubes, visible), token="s1", objects=objects)


class FakeObject:
    """Duck-typed stand-in for locate.LocatedObject."""

    def __init__(self, x, y, *, label="pen", axis=37.0, long_mm=138.0, short_mm=9.0):
        self.label = label
        self.px, self.py = 742.0, 388.0
        self.x, self.y = x, y
        self.axis_yaw_deg = axis
        self.long_mm, self.short_mm = long_mm, short_mm
        self.confidence = 0.81


# -- the core invariant ---------------------------------------------------


def test_pickable_agrees_with_the_pick_path() -> None:
    cubes = [
        cube("blue", 200.0, -60.0),       # clean, isolated
        cube("red", 140.0, 240.0),        # clean, isolated (306mm away)
        cube("green", 100.0, 0.0),        # inside keep-out
        cube("yellow", 330.0, 180.0),     # beyond max reach
        cube("red", 240.0, 100.0),        # inside reach, 60mm outside the hull
        cube("blue", 200.0, -40.0),       # 20mm from the blue above
        cube("green", 160.0, 60.0, area=100.0),   # under the pick floor
        cube("red", 160.0, 0.0, area=9000.0),     # over the pick ceiling
    ]
    s = scene(cubes)
    expected = {id(c) for c in s.pickable(s.cubes)}
    sn = build_snapshot(s, token="s1")
    by_pixel_order = sn.of_kind(KIND_CUBE)
    assert len(by_pixel_order) == len(cubes), "every real detection is reported"
    for ent, src in zip(
        sorted(by_pixel_order, key=lambda e: (e.x, e.y)),
        sorted(cubes, key=lambda c: (c.x, c.y)),
    ):
        assert ent.pickable == (id(src) in expected), (ent.id, ent.reason)


def test_every_unpickable_cube_explains_itself() -> None:
    cubes = [
        cube("green", 100.0, 0.0),
        cube("yellow", 330.0, 180.0),
        cube("red", 240.0, 100.0),
        cube("green", 160.0, 60.0, area=100.0),
        cube("red", 160.0, 0.0, area=9000.0),
    ]
    sn = snap(cubes)
    for ent in sn.of_kind(KIND_CUBE):
        assert not ent.pickable
        assert ent.reason and len(ent.reason) > 10, ent


def test_pickable_cube_has_no_reason() -> None:
    sn = snap([cube("red", 200.0, -60.0)])
    ent = sn.of_kind(KIND_CUBE)[0]
    assert ent.pickable and ent.reason is None


# -- one case per gate ----------------------------------------------------


def test_reason_keepout() -> None:
    s = scene([cube("green", 100.0, 0.0)])
    r = pick_block_reason(s.raw_cubes[0], s)
    assert r and "keep-out" in r and "140" in r


def test_reason_max_reach() -> None:
    s = scene([cube("yellow", 330.0, 180.0)])
    r = pick_block_reason(s.raw_cubes[0], s)
    assert r and "max reach" in r and f"{MAX_REACH_MM:.0f}" in r


def test_reason_outside_hull_reports_the_distance() -> None:
    # r=260mm, so inside reach: the hull is genuinely the first gate to
    # fire, not a consequence of being unreachable.
    s = scene([cube("red", 240.0, 100.0)])
    r = pick_block_reason(s.raw_cubes[0], s)
    assert r and "outside the marker hull" in r
    assert "mm outside" in r  # the how-far, not just the fact


def test_reason_area_floor_and_ceiling() -> None:
    s = scene([cube("green", 200.0, -60.0, area=PICK_MIN_AREA - 1)])
    r = pick_block_reason(s.raw_cubes[0], s)
    assert r and "pick floor" in r
    s = scene([cube("green", 200.0, -60.0, area=PICK_MAX_AREA + 1)])
    r = pick_block_reason(s.raw_cubes[0], s)
    assert r and "pick ceiling" in r


def test_reason_finger_clearance() -> None:
    a, b = cube("red", 200.0, -60.0), cube("blue", 200.0, -60.0 + 20.0)
    s = scene([a, b])
    r = pick_block_reason(a, s)
    assert r and "neighbour cube" in r and f"{PICK_CLEARANCE_MM:.0f}mm" in r
    assert "20mm away" in r


def test_reason_ordering_prefers_the_fundamental_gate() -> None:
    """A blob both inside the keep-out and under the area floor should be
    described by the area problem -- it is not a cube at all."""
    s = scene([cube("green", 100.0, 0.0, area=50.0)])
    r = pick_block_reason(s.raw_cubes[0], s)
    assert r and "pick floor" in r


# -- markers --------------------------------------------------------------


def test_marker_ids_are_the_printed_tag_numbers() -> None:
    """'put it on marker 1' must resolve without a lookup table."""
    sn = snap([])
    ids = {e.id for e in sn.of_kind(KIND_MARKER)}
    assert ids == {"marker_0", "marker_1", "marker_2", "marker_3"}


def test_free_marker_is_placeable() -> None:
    sn = snap([])
    m3 = sn.get("marker_3")
    assert m3 is not None and m3.placeable and m3.reason is None
    assert "free" in m3.label


def test_occupied_marker_names_its_occupant() -> None:
    sn = snap([cube("red", 177.2, 181.5)])
    m3 = sn.get("marker_3")
    assert m3 is not None and not m3.placeable
    assert m3.holds is not None
    occupant = sn.get(m3.holds)
    assert occupant is not None and occupant.kind == KIND_CUBE
    assert occupant.on == "marker_3"
    assert m3.reason and "occupied by" in m3.reason


def test_undecoded_marker_is_never_a_place_target() -> None:
    sn = snap([], visible={0, 1, 2})  # marker 3's tag did not decode
    m3 = sn.get("marker_3")
    assert m3 is not None and not m3.placeable
    assert m3.reason and "did not decode" in m3.reason
    assert "unknown" in m3.label


def test_marker_beyond_reach_says_so() -> None:
    """Marker 1 sits at r~303mm here, inside reach; push a marker out instead."""
    far = [MarkerSlot(9, 340.0, 180.0)]
    state = rebuild_workspace_state(None, far, [], visible_marker_ids={9})
    sn = build_snapshot(
        Scene.from_workspace(state, pick_cubes=[], raw_cubes=[]), token="s1"
    )
    m9 = sn.get("marker_9")
    assert m9 is not None and not m9.placeable
    assert m9.reason and "max reach" in m9.reason


# -- slots and lookup ----------------------------------------------------


def test_free_slots_are_placeable() -> None:
    sn = snap([])
    slots = sn.of_kind(KIND_SLOT)
    assert slots and all(s.placeable and s.reason is None for s in slots)


def test_get_returns_none_for_an_unknown_id() -> None:
    sn = snap([cube("red", 200.0, -60.0)])
    assert sn.get("cube_99") is None
    assert sn.get("pen") is None
    assert sn.get("") is None


def test_snapshot_dict_shape() -> None:
    sn = snap([cube("red", 200.0, -60.0)])
    d = sn.as_dict()
    assert d["snapshot"] == "s1"
    assert isinstance(d["entities"], list) and d["entities"]
    assert "cubes=" in d["summary"]
    first = d["entities"][0]
    assert {"id", "kind", "label", "x", "y", "pickable", "placeable"} <= set(first)


# -- id stability --------------------------------------------------------


def test_ids_are_spatially_stable_under_detection_jitter() -> None:
    """A millimetre of wobble between frames must not swap two cubes' ids, or
    an id read from one snapshot means a different cube in the next."""
    a = snap([cube("red", 140.0, 240.0), cube("blue", 200.0, -60.0)])
    b = snap([cube("red", 140.7, 239.4), cube("blue", 199.3, -60.6)])
    pos = {e.id: (round(e.x), round(e.y)) for e in a.of_kind(KIND_CUBE)}
    for e in b.of_kind(KIND_CUBE):
        px, py = pos[e.id]
        assert abs(e.x - px) < 2 and abs(e.y - py) < 2


def test_ids_are_unique() -> None:
    sn = snap(
        [cube("red", 140.0, 240.0), cube("blue", 200.0, -60.0), cube("green", 100.0, 0.0)],
        objects=[FakeObject(213.4, -58.1), FakeObject(230.0, 40.0, label="key")],
    )
    ids = [e.id for e in sn.entities]
    assert len(ids) == len(set(ids))


# -- located objects -----------------------------------------------------


def test_object_entity_is_long_axis_periodic() -> None:
    sn = snap([], objects=[FakeObject(213.4, -58.1)])
    obj = sn.get("obj_1")
    assert obj is not None and obj.kind == KIND_OBJECT
    assert obj.label == "pen"
    assert obj.yaw_period_deg == YAW_PERIOD_LONG_AXIS
    assert obj.yaw_deg == 37.0
    assert obj.extent_mm == (138.0, 9.0)
    assert obj.pickable and obj.reason is None


def test_cube_entity_is_square_periodic() -> None:
    sn = snap([cube("red", 200.0, -60.0)])
    assert sn.of_kind(KIND_CUBE)[0].yaw_period_deg == YAW_PERIOD_SQUARE


def test_object_clearance_uses_its_short_axis_not_the_cube_rule() -> None:
    """A 138mm pen has neighbours inside PICK_CLEARANCE_MM constantly; applying
    a 20mm cube's finger allowance to it would refuse nearly every real grasp.
    A cube 30mm from the grasp point must not block a 9mm-wide pen."""
    near = cube("red", 200.0, -60.0)
    sn = build_snapshot(
        scene([near]),
        token="s1",
        objects=[FakeObject(200.0, -30.0)],  # 30mm from that cube
    )
    obj = sn.get("obj_1")
    assert obj is not None and obj.pickable, obj.reason
    # ...but something genuinely on top of the grasp point does block it.
    sn2 = build_snapshot(
        scene([near]), token="s1", objects=[FakeObject(200.0, -55.0)]
    )
    blocked = sn2.get("obj_1")
    assert blocked is not None and not blocked.pickable
    assert blocked.reason and "grasp point" in blocked.reason


def test_object_outside_reach_abstains() -> None:
    sn = snap([], objects=[FakeObject(400.0, 0.0)])
    obj = sn.get("obj_1")
    assert obj is not None and not obj.pickable
    assert obj.reason and "max reach" in obj.reason


# -- as_grasp bridge ----------------------------------------------------


def _calib(**kw):
    base = dict(
        table_z=127.2, cube_height_mm=20.0, safe_z=155.0,
        travel_speed_us=700, approach_speed_us=2400,
        grip_open_s=140, grip_close_s=240,
        grip_span_s_at_zero_mm=None, grip_span_s_per_mm=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_as_grasp_carries_yaw_and_period() -> None:
    sn = snap([], objects=[FakeObject(213.4, -58.1)])
    g = sn.get("obj_1").as_grasp()
    assert (g.x, g.y) == (213.4, -58.1)
    assert g.yaw_deg == 37.0 and g.yaw_period_deg == YAW_PERIOD_LONG_AXIS
    assert g.grip_close_s is None  # no calib -> inherit the calibration value


def test_as_grasp_sizes_the_close_value_to_the_object() -> None:
    sn = snap([], objects=[FakeObject(213.4, -58.1)])
    calib = _calib(grip_span_s_at_zero_mm=285.0, grip_span_s_per_mm=2.25)
    g = sn.get("obj_1").as_grasp(calib)
    # 9mm pen: much more closed than the 20mm cube's 240.
    assert g.grip_close_s is not None and g.grip_close_s > calib.grip_close_s
    assert abs(g.grip_close_s - (285.0 - 2.25 * (9.0 - GRIP_SQUEEZE_MM))) <= 1


def test_as_grasp_closes_past_the_measured_width() -> None:
    """Vision reports a silhouette, which reads wide for anything with height --
    the one error direction that grips nothing and still reports success."""
    sn = snap([], objects=[FakeObject(213.4, -58.1)])
    calib = _calib(grip_span_s_at_zero_mm=285.0, grip_span_s_per_mm=2.25)
    just_touching = 285.0 - 2.25 * 9.0
    assert sn.get("obj_1").as_grasp(calib).grip_close_s > just_touching


def test_as_grasp_falls_back_when_the_span_is_uncalibrated() -> None:
    sn = snap([], objects=[FakeObject(213.4, -58.1)])
    g = sn.get("obj_1").as_grasp(_calib())
    assert g.grip_close_s == 240  # the cube value, with a warning emitted


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
