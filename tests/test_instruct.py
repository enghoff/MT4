"""Tests for the policy layer's coordinate handling (no hardware, no service).

Every bug this file pins was found by driving the arm, which is an expensive
way to discover an arithmetic error. The one that motivated the file:

    python run_instruction.py "pick up the stapler and place it on marker 0"
    -> found a stapler but could not measure it: could not segment an object
       at (700, 687)

Qwen had located the stapler to within 4px. The reply ``[700, 700]`` was in a
0-1000 normalized space and meant pixel (896, 504); the code read it as pixel
(700, 700), 280px away on bare desk. It failed loudly there only by luck -- the
same misreading lands on a *different object* elsewhere in the frame and
measures it confidently, which nothing downstream can detect.

Measured 2026-08-02 on the live rig, four objects x two prompt forms x two
frame variants: 8 of 8 replies normalized, 1-10px from truth as normalized,
206-285px from truth as pixels. So normalized is the default and a coordinate
above 1000 is the only thing that can rule it out.

Run: python tests/test_instruct.py  (or pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mt4_vision.locate as locate_mod
from mt4_vision.calib import Calibration
from mt4_vision.entities import Snapshot
from mt4_vision.instruct import (
    COORD_SCALE,
    Grounding,
    Observation,
    _grounding,
    alternate_reading,
    excluded_destination_kind,
    instruction_attributes,
    point_readings,
    is_question,
    measure_grounding,
    named_destination_block_reason,
    noun_phrase,
    register_object,
    wrong_kind_block_reason,
    to_frame_pixels,
    unmatched_nouns,
)

FRAME_W, FRAME_H = 1280, 720
SIZE = (FRAME_W, FRAME_H)

# The live replies, verbatim, from the 2026-08-02 grounding run. Truth pixels
# for the cubes come from the HSV detector on the same frame; the stapler was
# read off the drawn grid by hand.
LIVE = {
    #  label        normalized box            truth pixel
    "stapler": ((630, 650, 782, 827), (900, 530)),
    "red cube": ((557, 606, 605, 693), (749, 464)),
    "green cube": ((300, 653, 343, 731), (411, 497)),
    "blue cube": ((792, 506, 833, 565), (1043, 386)),
}
# The point reply that broke the live run.
STAPLER_POINT = (700, 700)

MM_PER_PX = 0.5
CALIB = Calibration(
    homography=[[MM_PER_PX, 0.0, 0.0], [0.0, MM_PER_PX, 0.0], [0.0, 0.0, 1.0]],
    table_z=127.2,
    safe_z=155.0,
)


def _obs() -> Observation:
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    return Observation(
        frame=frame, annotated=frame, snapshot=Snapshot(token="t"), calib=CALIB
    )


# --------------------------------------------------------------- conversion


def test_normalized_point_is_scaled():
    assert to_frame_pixels((700, 700), SIZE) == (896.0, 504.0)


def test_normalized_box_is_scaled_and_keeps_its_shape():
    got = to_frame_pixels((630, 650, 782, 827), SIZE)
    assert len(got) == 4
    assert got[0] == 630 * FRAME_W / COORD_SCALE
    assert got[1] == 650 * FRAME_H / COORD_SCALE
    # y2 = 827 normalized is 595px -- inside a 720px frame, but 827 read as a
    # pixel is not. That overflow is the only self-evidence in the reply, and
    # it is why grounding asks for a box rather than a point.
    assert got[3] < FRAME_H < 827


def test_a_coordinate_above_the_scale_rules_normalized_out():
    """The one escape hatch: no normalized coordinate can exceed 1000."""
    assert to_frame_pixels((1014, 364, 1066, 407), SIZE) == (
        1014.0, 364.0, 1066.0, 407.0,
    )


def test_every_live_reply_lands_on_its_object():
    """The regression, on all four objects measured on the rig."""
    for label, (nbox, truth) in LIVE.items():
        x1, y1, x2, y2 = to_frame_pixels(nbox, SIZE)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        err = ((cx - truth[0]) ** 2 + (cy - truth[1]) ** 2) ** 0.5
        assert err < 15.0, f"{label}: box centre {err:.0f}px from truth"


def test_the_old_in_frame_rule_would_still_be_wrong():
    """Why the previous heuristic could not work, stated as a test.

    "Use the numbers as pixels when they land in frame, rescale otherwise"
    fires the wrong way on a normalized reply, because a normalized reply
    almost always lands in frame at this resolution: x <= 1000 < 1280 always.
    """
    for label, (nbox, truth) in LIVE.items():
        cx = (nbox[0] + nbox[2]) / 2
        cy = (nbox[1] + nbox[3]) / 2
        in_frame_as_pixels = 0 <= cx <= FRAME_W and 0 <= cy <= FRAME_H
        err = ((cx - truth[0]) ** 2 + (cy - truth[1]) ** 2) ** 0.5
        if in_frame_as_pixels:
            assert err > 100.0, f"{label}: in-frame but only {err:.0f}px off"


def test_stapler_point_reply_is_not_taken_literally():
    got = to_frame_pixels(STAPLER_POINT, SIZE)
    assert got != (700.0, 700.0)
    truth = LIVE["stapler"][1]
    err = ((got[0] - truth[0]) ** 2 + (got[1] - truth[1]) ** 2) ** 0.5
    assert err < 30.0, f"{err:.0f}px from the stapler"


# ---------------------------------------------------------------- alternate


def test_alternate_is_the_raw_numbers_when_they_fit_the_frame():
    assert alternate_reading(STAPLER_POINT, SIZE) == (700.0, 700.0)


def test_alternate_is_none_when_the_raw_numbers_leave_the_frame():
    # y2 = 827 > 720, so raw-as-pixels is not a possible reading at all.
    assert alternate_reading((630, 650, 782, 827), SIZE) is None


def test_alternate_is_none_when_pixels_were_already_forced():
    assert alternate_reading((1014, 364, 1066, 407), SIZE) is None


# ------------------------------------------------------------ point_readings


def test_a_decision_point_is_read_as_pixels_first():
    """Measured 2026-08-03: the decision reply is pixels, 5/5, to 0.4px.

    The entity list prints "at image point (x, y)" for every entity, so the
    prompt shows the model the space it wants. Read as normalized instead, the
    same five replies land 160px away -- which is the "model point is
    consistently way off" the preview made visible.
    """
    got = point_readings((320, 473), SIZE)
    assert got[0] == (320.0, 473.0)
    assert got[1] == (320 * FRAME_W / 1000.0, 473 * FRAME_H / 1000.0)


def test_a_coordinate_over_the_scale_has_only_one_reading():
    """Nothing normalized exceeds COORD_SCALE, so there is no second reading."""
    assert point_readings((1255, 402), SIZE) == ((1255.0, 402.0),)


def test_a_reading_that_falls_outside_the_frame_is_dropped():
    # 900 normalized is in frame vertically; 900 as a pixel row is not.
    got = point_readings((500, 900), SIZE)
    assert got == ((500 * FRAME_W / 1000.0, 900 * FRAME_H / 1000.0),)


def test_point_readings_rejects_junk():
    assert point_readings((), SIZE) == ()
    assert point_readings(("a", "b"), SIZE) == ()
    assert point_readings((None, 5), SIZE) == ()


# --------------------------------------------------------------- Grounding


def test_grounding_from_a_box_carries_both_the_box_and_its_centre():
    prim = to_frame_pixels(LIVE["stapler"][0], SIZE)
    g = _grounding("stapler", "box", prim, None, SIZE)
    assert g is not None
    assert g.box_px == tuple(prim)
    assert g.point_px == ((prim[0] + prim[2]) / 2, (prim[1] + prim[3]) / 2)
    assert g.alt_point_px is None


def test_grounding_from_a_point_has_no_box():
    g = _grounding("stapler", "point", (896.0, 504.0), (700.0, 700.0), SIZE)
    assert g is not None
    assert g.box_px is None
    assert g.alt_box_px is None
    assert g.alt_point_px == (700.0, 700.0)


def test_grounding_orders_box_corners():
    g = _grounding("thing", "box", (900.0, 500.0, 800.0, 400.0), None, SIZE)
    assert g is not None
    assert g.box_px == (800.0, 400.0, 900.0, 500.0)


def test_grounding_refuses_a_centre_outside_the_frame():
    assert _grounding("thing", "point", (-5.0, 400.0), None, SIZE) is None
    assert _grounding("thing", "point", (400.0, 900.0), None, SIZE) is None


def test_grounding_refuses_a_whole_frame_box():
    """"I don't know", written as a box.

    Live, asked to locate a "location", the model returned [0, 0, 1000, 1000] --
    the entire image. Its centre is in frame, so the only thing that says this
    is not an object is its size. Left to the projection it came out at
    315037x312990mm, a number about the camera's horizon rather than the reply.
    """
    whole = to_frame_pixels((0, 0, 1000, 1000), SIZE)
    assert _grounding("location", "box", whole, None, SIZE) is None
    # Just under the threshold still passes: the guard is a backstop for
    # non-answers, not a second opinion on plausible sizes.
    big = (0.0, 0.0, FRAME_W * 0.7, FRAME_H * 0.7)
    assert _grounding("thing", "box", big, None, SIZE) is not None


# --------------------------------------------------------- measure_grounding


class _FakeObject:
    def __init__(self, tag):
        self.tag = tag


def _patch_measure(monkey_calls, results):
    """Replace measure_with_box_fallback; record calls, replay ``results``."""
    def fake(frame, px, py, calib, label, *, box=None, **kw):
        monkey_calls.append({"point": (px, py), "box": box, "label": label})
        outcome = results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return fake


def test_measure_grounding_passes_the_box_through(monkeypatch=None):
    calls: list[dict] = []
    orig = locate_mod.measure_with_box_fallback
    locate_mod.measure_with_box_fallback = _patch_measure(
        calls, [_FakeObject("ok")]
    )
    try:
        g = Grounding("stapler", (900.0, 530.0), (806.0, 468.0, 1001.0, 595.0))
        obj, why = measure_grounding(_obs(), g)
    finally:
        locate_mod.measure_with_box_fallback = orig
    assert why == ""
    assert obj.tag == "ok"
    assert len(calls) == 1
    assert calls[0]["box"] == (806.0, 468.0, 1001.0, 595.0)
    assert calls[0]["point"] == (900.0, 530.0)


def test_measure_grounding_retries_the_alternate_reading():
    calls: list[dict] = []
    orig = locate_mod.measure_with_box_fallback
    locate_mod.measure_with_box_fallback = _patch_measure(
        calls, [locate_mod.LocateError("no"), _FakeObject("second")]
    )
    try:
        g = Grounding(
            "stapler", (896.0, 504.0), None, alt_point_px=(700.0, 700.0)
        )
        obj, why = measure_grounding(_obs(), g)
    finally:
        locate_mod.measure_with_box_fallback = orig
    assert why == ""
    assert obj.tag == "second"
    assert [c["point"] for c in calls] == [(896.0, 504.0), (700.0, 700.0)]


def test_measure_grounding_reports_the_first_failure_when_both_fail():
    calls: list[dict] = []
    orig = locate_mod.measure_with_box_fallback
    locate_mod.measure_with_box_fallback = _patch_measure(
        calls,
        [locate_mod.LocateError("primary said no"),
         locate_mod.LocateError("alternate said no")],
    )
    try:
        g = Grounding(
            "stapler", (896.0, 504.0), None, alt_point_px=(700.0, 700.0)
        )
        obj, why = measure_grounding(_obs(), g)
    finally:
        locate_mod.measure_with_box_fallback = orig
    assert obj is None
    assert "primary said no" in why
    assert len(calls) == 2


def test_measure_grounding_does_not_retry_without_an_alternate():
    calls: list[dict] = []
    orig = locate_mod.measure_with_box_fallback
    locate_mod.measure_with_box_fallback = _patch_measure(
        calls, [locate_mod.LocateError("nope")]
    )
    try:
        g = Grounding("stapler", (896.0, 504.0), None)
        obj, why = measure_grounding(_obs(), g)
    finally:
        locate_mod.measure_with_box_fallback = orig
    assert obj is None
    assert len(calls) == 1


# ------------------------------------------------------- unmatched_nouns


def _snap_with(*labels: str) -> Snapshot:
    from mt4_vision.entities import Entity

    return Snapshot(
        token="t",
        entities=[
            Entity(id=f"e{i}", kind="cube", label=lab, x=0.0, y=0.0)
            for i, lab in enumerate(labels)
        ],
    )


def test_an_unknown_noun_is_reported():
    got = unmatched_nouns("pick up the stapler", _snap_with("red cube"))
    assert got == {"stapler"}


def test_a_held_object_is_no_longer_unmatched():
    """The step-2 regression.

    Live: "pick up the stapler and place it on marker 0" picked the stapler at
    step 1, then step 2 searched for a stapler again -- it was in the gripper,
    so the model pointed at the gripper and the loop registered a phantom at
    robot (-50, -97), inside the J1 keep-out.
    """
    snap = _snap_with("red cube")
    assert unmatched_nouns("pick up the stapler", snap) == {"stapler"}
    assert unmatched_nouns("pick up the stapler", snap, held="stapler") == set()


def test_held_matching_is_per_word():
    snap = _snap_with("red cube")
    got = unmatched_nouns(
        "put the desk stapler on marker 0", snap, held="desk stapler"
    )
    assert got == set()


def test_held_does_not_mask_a_different_noun():
    snap = _snap_with("red cube")
    got = unmatched_nouns(
        "pick up the stapler then the screwdriver", snap, held="stapler"
    )
    assert got == {"screwdriver"}


def test_abstract_words_never_reach_the_grounder():
    """The 28-prompt sweep's biggest finding.

    Qwen answers a grounding request for an abstract noun instead of declining
    it, so anything that gets through here becomes a model call and possibly a
    registered entity. Measured: "clear" was registered at robot (24, -335)
    83x50mm and "there" at (-293, 6) 55x33mm, both as pickable objects.
    """
    snap = _snap_with("red cube", "blue cube")
    leaked = [
        "move the blue cube somewhere clear",
        "tidy up the desk",
        "pick up everything",
        "is there anything on the desk that is not a cube",
        "put it down over there",
        "clean up all of this",
    ]
    for text in leaked:
        assert unmatched_nouns(text, snap) == set(), f"{text!r} leaked"


def test_place_holder_nouns_for_a_position_never_reach_the_grounder():
    """A destination described rather than named.

    Measured live 2026-08-02: "pick up the green cube and place it on a
    non-marker location" grounded the word "location", got a whole-frame box,
    projected it to 315037x312990mm, and refused the entire task -- while the
    same snapshot was offering 14 free slots as destinations.
    """
    snap = _snap_with("green cube", "blue cube")
    leaked = [
        "pick up the green cube and place it on a non-marker location",
        "put the blue cube in a free spot",
        "move the green cube to an empty position",
        "place it on any open space",
        "put it down on a clear patch of desk",
        "set the blue cube on an unoccupied square",
    ]
    for text in leaked:
        assert unmatched_nouns(text, snap) == set(), f"{text!r} leaked"


def test_a_real_object_noun_still_gets_through():
    snap = _snap_with("red cube")
    assert unmatched_nouns("pick up the stapler", snap) == {"stapler"}
    assert unmatched_nouns("move the screwdriver here", snap) == {"screwdriver"}


# ------------------------------------------------- excluded_destination_kind


def test_a_task_that_rules_out_markers_is_recognised():
    for text in [
        "pick up the green cube and place it on a non-marker location",
        "put the blue cube somewhere that is not a marker",
        "move the red cube off the markers",
        "place it down with no marker under it",
        "put the cube other than a marker",
        "set it down clear of the markers",
    ]:
        assert excluded_destination_kind(text) == "marker", text


def test_an_ordinary_task_excludes_nothing():
    for text in [
        "put the blue cube on marker 3",
        "pick up the stapler",
        "move the green cube to slot 4",
        "tidy the desk",
    ]:
        assert excluded_destination_kind(text) is None, text


# ------------------------------------------------------------ noun_phrase


def test_noun_phrase_stops_at_a_preposition():
    got = noun_phrase("pick up the grey rock next to the blue cube", "rock")
    assert got == "the grey rock"


def test_noun_phrase_keeps_a_multi_word_modifier():
    assert noun_phrase("pick up the dark red cube", "cube") == "the dark red cube"


def test_noun_phrase_abstains_when_the_head_is_absent():
    """Not a fallback to the whole sentence -- that manufactured contradictions.

    The model named obj_2, labelled "stone", for "the grey rock". Harvesting
    the sentence anyway refused it as ``the task says ['blue', 'grey', 'rock']
    but obj_2 is 'stone'``: three words the instruction never claimed about a
    stone. With no phrase to read, the honest answer is no evidence.
    """
    snap = _snap_with("grey rock")
    assert noun_phrase("pick up the grey rock", "stone") == ""
    assert instruction_attributes(
        "pick up the grey rock next to the blue cube", snap, head="stone"
    ) == set()


def test_attributes_ignore_a_landmark_outside_the_phrase():
    """The wrong-refusal this fixes.

    Live: "pick up the grey rock next to the blue cube" was refused as
    ``the task says ['blue', 'grey', 'rock'] but obj_5 is 'rock'``. The blue
    cube was a landmark, never a requirement.
    """
    snap = _snap_with("blue cube", "grey rock")
    text = "pick up the grey rock next to the blue cube"
    assert "blue" in instruction_attributes(text, snap)
    assert "blue" not in instruction_attributes(text, snap, head="rock")
    assert "grey" in instruction_attributes(text, snap, head="rock")


def test_marker_numbers_are_read_from_the_whole_instruction():
    """The destination is named outside the target's phrase by construction."""
    from mt4_vision.entities import Entity

    snap = Snapshot(
        token="t",
        entities=[
            Entity(id="cube_1", kind="cube", label="blue cube", x=0.0, y=0.0),
            Entity(id="marker_3", kind="marker", label="marker 3 (free)", x=1.0, y=1.0),
        ],
    )
    got = instruction_attributes(
        "put the blue cube on marker 3", snap, head="cube"
    )
    assert "blue" in got and "marker3" in got


# -------------------------------------------------- register / guards


class _Obj:
    def __init__(self, x, y, label="thing"):
        self.x, self.y, self.label = x, y, label


def test_register_object_merges_two_hits_on_one_thing():
    objects: dict = {}
    eid, new = register_object(objects, _Obj(109.0, 179.0, "grey"), seq=1)
    assert new and eid == "obj_1"
    eid2, new2 = register_object(objects, _Obj(108.0, 179.0, "rock"), seq=2)
    assert not new2 and eid2 == "obj_1"
    assert len(objects) == 1


def test_register_object_keeps_two_distinct_things():
    objects: dict = {}
    register_object(objects, _Obj(109.0, 179.0, "rock"), seq=1)
    eid, new = register_object(objects, _Obj(109.0, 220.0, "pen"), seq=2)
    assert new and eid == "obj_2" and len(objects) == 2


def test_register_object_folds_the_new_word_into_the_label():
    """Both words described the same thing, so the entity answers to both.

    Without this the second word stays "unmatched" and the task is refused for
    naming something that is in fact registered -- measured on "the grey rock",
    where "grey" registered obj_1 and "rock" then had nothing to match.
    """
    from mt4_vision.locate import LocatedObject

    def obj(x, y, label):
        return LocatedObject(
            label=label, px=1.0, py=1.0, x=x, y=y,
            axis_yaw_deg=0.0, long_mm=60.0, short_mm=40.0, confidence=0.9,
        )

    objects: dict = {}
    register_object(objects, obj(109.0, 179.0, "grey"), seq=1)
    eid, new = register_object(objects, obj(108.0, 179.0, "rock"), seq=2)
    assert not new and eid == "obj_1"
    assert objects["obj_1"].label == "grey rock"


def test_a_destination_held_by_the_thing_being_moved_is_not_blocked():
    """"put the green cube on marker 2" with it already there is done, not
    refused for being occupied by itself."""
    from mt4_vision.entities import Entity

    snap = Snapshot(
        token="t",
        entities=[
            Entity(
                id="marker_2", kind="marker", label="marker 2 (occupied)",
                x=1.0, y=1.0, placeable=False, reason="occupied by cube_4",
                holds="cube_4",
            ),
        ],
    )
    text = "put the green cube on marker 2"
    assert named_destination_block_reason(text, snap) is not None
    assert named_destination_block_reason(text, snap, moving="cube_4") is None
    assert named_destination_block_reason(text, snap, moving="cube_9") is not None


def test_picking_a_cube_when_the_task_asked_for_a_slot_is_refused():
    """The silent substitution: "pick up slot 5" returned PICK_ENTITY cube_2."""
    from mt4_vision.entities import Entity

    cube = Entity(id="cube_2", kind="cube", label="blue cube", x=0.0, y=0.0)
    snap = Snapshot(token="t", entities=[cube])
    assert wrong_kind_block_reason("pick up slot 5", cube, snap) is not None
    assert wrong_kind_block_reason("pick up marker 3", cube, snap) is not None
    # The target IS named, so a two-part instruction still passes.
    assert wrong_kind_block_reason(
        "put the blue cube on marker 3", cube, snap
    ) is None
    # No never-pickable kind named at all: nothing to object to.
    assert wrong_kind_block_reason("tidy up the desk", cube, snap) is None
    assert wrong_kind_block_reason("pick up slot 5", None, snap) is None


def test_the_kind_guard_does_not_block_a_named_object_bound_for_a_marker():
    """The regression this guard caused, and the reason it tests labels now.

    A registered object's KIND is the word "object", which no real instruction
    contains, so comparing kinds refused every
    "pick up the <thing> and put it on marker N" -- the main non-cube use case.
    Live: "pick up the stapler and place it on marker 4" was refused as
    ``obj_1 is a object the task never mentions``. Cubes hid it, because "cube"
    is both their kind and a word people type.
    """
    from mt4_vision.entities import Entity

    obj = Entity(id="obj_1", kind="object", label="stapler", x=0.0, y=0.0)
    snap = Snapshot(token="t", entities=[obj])
    assert wrong_kind_block_reason(
        "pick up the stapler and place it on marker 4", obj, snap
    ) is None
    assert wrong_kind_block_reason(
        "put the stapler on slot 3", obj, snap
    ) is None
    # A merged multi-word label matches on any of its words.
    rock = Entity(id="obj_2", kind="object", label="grey rock", x=0.0, y=0.0)
    snap2 = Snapshot(token="t", entities=[rock])
    assert wrong_kind_block_reason(
        "put the rock on marker 1", rock, snap2
    ) is None
    # But an object the task never names is still refused.
    assert wrong_kind_block_reason("pick up marker 1", rock, snap2) is not None


def test_questions_are_recognised():
    for text in (
        "is there anything on the desk that is not a cube",
        "what is on marker 3",
        "how many cubes are there?",
        "Where is the stapler?",
    ):
        assert is_question(text), text
    for text in ("pick up the blue cube", "put it on marker 3", "tidy up"):
        assert not is_question(text), text


def test_a_blocked_named_destination_is_reported():
    from mt4_vision.entities import Entity

    snap = Snapshot(
        token="t",
        entities=[
            Entity(
                id="marker_2", kind="marker", label="marker 2 (occupied)",
                x=1.0, y=1.0, placeable=False, reason="occupied by cube_5",
            ),
            Entity(
                id="marker_3", kind="marker", label="marker 3 (free)",
                x=2.0, y=2.0, placeable=True,
            ),
        ],
    )
    why = named_destination_block_reason("put the blue cube on marker 2", snap)
    assert why is not None and "occupied by cube_5" in why
    assert named_destination_block_reason(
        "put the blue cube on marker 3", snap
    ) is None
    assert named_destination_block_reason("pick up the blue cube", snap) is None


# ------------------------------------------------------------------ TRANSFER
#
# One decision names both ends of the move and the arm carries it out without
# stopping, so this is the last place either end can be caught. Nothing after
# the decision looks at the desk again -- the after-the-fact vision check was
# removed 2026-08-03 for reporting a completed pick as a failure, having
# template-matched the ArUco tag lying under the stapler rather than the
# stapler. These tests are what stands in for it.


def _transfer_snapshot(**over):
    """A stapler, a free marker, and an occupied one, spaced so points bind.

    x pixels are above ``COORD_SCALE`` on purpose: that is the one thing that
    rules the normalized reading out, so each point has a single unambiguous
    interpretation and these tests are about the transfer logic rather than
    about coordinate spaces (which the top half of this file already covers).
    """
    from mt4_vision.entities import Entity

    ents = [
        Entity(
            id="obj_1", kind="object", label="stapler", x=100.0, y=50.0,
            pixel=(1100.0, 150.0), pickable=True, source="vlm",
            yaw_deg=30.0, yaw_period_deg=180.0,
        ),
        Entity(
            id="marker_3", kind="marker", label="marker 3 (free)",
            x=150.0, y=250.0, pixel=(1100.0, 620.0), placeable=True,
            source="aruco",
        ),
        Entity(
            id="marker_2", kind="marker", label="marker 2 (occupied)",
            x=40.0, y=250.0, pixel=(1030.0, 400.0), placeable=False,
            reason="occupied by cube_5", source="aruco",
        ),
    ]
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    return Observation(
        frame=frame, annotated=frame, calib=CALIB,
        snapshot=Snapshot(token="t", entities=ents), **over,
    )


def _decide_on(reply_json: str, obs, instruction: str):
    """Run the real ``decide`` against a canned model reply."""
    import mt4_vision.instruct as instruct_mod

    class _Reply:
        text = reply_json

    saved = instruct_mod.ask
    instruct_mod.ask = lambda *a, **k: _Reply()
    try:
        return instruct_mod.decide(obs, instruction)
    finally:
        instruct_mod.ask = saved


def test_a_transfer_resolves_both_ends_against_one_snapshot():
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_3", '
        '"dest_2d": [1100, 620], "reason": "the stapler goes on marker 3"}',
        _transfer_snapshot(),
        "put the stapler on marker 3",
    )
    assert act.ok, act.reason
    assert act.kind == "TRANSFER"
    assert (act.entity_id, act.label) == ("obj_1", "stapler")
    assert (act.dest_entity_id, act.dest_label) == ("marker_3", "marker 3 (free)")
    # Both ends survive into as_dict, which is what the MCP layer and the
    # transcripts read.
    assert act.as_dict()["dest_entity_id"] == "marker_3"
    assert act.as_dict()["dest_agreed"] is True


def test_a_transfer_onto_an_occupied_marker_is_refused_before_the_grasp():
    """Refused while the stapler is still on the desk, not after it is held."""
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_2", '
        '"dest_2d": [1030, 400], "reason": "put it there"}',
        _transfer_snapshot(),
        "put the stapler on marker 2",
    )
    assert not act.ok
    assert "occupied by cube_5" in act.reason


def test_a_transfer_refusal_says_which_end_failed():
    """Both halves word their refusals identically, so the prefix is the only
    thing telling a reader it is the destination that will not do."""
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_9", '
        '"dest_2d": [1100, 620], "reason": "put it there"}',
        _transfer_snapshot(),
        "put the stapler somewhere",
    )
    assert not act.ok
    assert act.reason.startswith("the destination will not do:")
    assert "marker_9" in act.reason


def test_a_transfer_needs_an_empty_gripper():
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_3", '
        '"dest_2d": [1100, 620], "reason": "move it"}',
        _transfer_snapshot(held="red cube"),
        "put the stapler on marker 3",
    )
    assert not act.ok
    assert "red cube" in act.reason and "PLACE_ENTITY" in act.reason


def test_a_transfer_cannot_name_a_cube_as_its_destination():
    """The source half's rules and the destination half's rules are the ones
    the two separate actions always used, so this refuses for the same reason
    ``PLACE_ENTITY cube_2`` always did."""
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "obj_1", '
        '"dest_2d": [1100, 150], "reason": "move it"}',
        _transfer_snapshot(),
        "put the stapler somewhere",
    )
    assert not act.ok
    assert "the destination will not do:" in act.reason


def test_the_single_target_actions_still_resolve_after_the_split():
    """``_resolve_target`` is the old tail of ``decide`` lifted out whole; these
    two pin that lifting it out changed nothing for the callers that existed."""
    pick = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "reason": "pick it"}',
        _transfer_snapshot(),
        "pick up the stapler",
    )
    assert pick.ok and pick.kind == "PICK_ENTITY" and pick.entity_id == "obj_1"
    assert pick.dest_entity_id is None

    place = _decide_on(
        '{"action": "PLACE_ENTITY", "entity_id": "marker_3", '
        '"point_2d": [1100, 620], "reason": "put it down"}',
        _transfer_snapshot(held="stapler"),
        "put it on marker 3",
    )
    assert place.ok and place.entity_id == "marker_3"

    # And the gripper-state rules survived being moved out of the resolver.
    assert not _decide_on(
        '{"action": "PLACE_ENTITY", "entity_id": "marker_3", '
        '"point_2d": [1100, 620], "reason": "put it down"}',
        _transfer_snapshot(),
        "put it on marker 3",
    ).ok


def test_grasp_for_squares_a_destination_and_keeps_an_object_angle():
    """What the arm is actually handed, now that nothing re-measures first."""
    from mt4_vision.instruct import grasp_for

    snap = _transfer_snapshot().snapshot
    obj = grasp_for(snap.get("obj_1"), CALIB)
    assert (obj.x, obj.y) == (100.0, 50.0)
    # The object's own angle, on the 180 lattice: the jaws close ACROSS a
    # stapler's long axis, and the pick is the only thing that knows which way
    # that runs.
    assert obj.yaw_deg == 30.0 and obj.yaw_period_deg == 180.0

    dest = grasp_for(snap.get("marker_3"), CALIB)
    assert (dest.x, dest.y) == (150.0, 250.0)
    # A destination has no angle of its own, so the held thing lands squared to
    # the world axes rather than at whatever angle the pick left the wrist.
    assert dest.yaw_deg == 0.0 and dest.yaw_period_deg == 90.0


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception:
            bad += 1
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)


# --------------------------------------------------- attributes vs. identity
#
# "pick up the green statue and place it on marker 0", 2026-08-03, with a green
# statue plainly on the desk:
#
#   registered obj_1: statue at (180, -77) 136x15mm
#   STOP  reason: the task says ['green', 'statue'] but obj_1 is 'statue'
#
# Not bad luck -- a closed loop. `instruction_attributes` pools its vocabulary
# from every entity's LABEL, so "green" was required because a green *cube* was
# on the desk. `unmatched_nouns` grounds only words no label contains, so
# "green" was never grounded and could never enter the statue's label. The two
# rules are exact complements: a shared adjective is required by construction
# and absent by construction. Every "green statue" fails, forever.


def _labelled(colour_of_obj, *, obj_label="statue"):
    from mt4_vision.entities import Entity

    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    label = f"{colour_of_obj} {obj_label}" if colour_of_obj else obj_label
    ents = [
        Entity(
            id="obj_1", kind="object", label=label, color=colour_of_obj,
            x=100.0, y=50.0, pixel=(1100.0, 150.0), pickable=True, source="vlm",
        ),
        Entity(
            id="cube_1", kind="cube", label="green cube", color="green",
            x=-20.0, y=-90.0, pixel=(1100.0, 650.0), pickable=True,
        ),
    ]
    return Observation(
        frame=frame, annotated=frame, calib=CALIB,
        snapshot=Snapshot(token="t", entities=ents),
    )


def test_a_measured_colour_on_an_object_satisfies_the_attribute_check():
    """The reported bug. A green statue and a green cube in one snapshot."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "reason": "the statue"}',
        _labelled("green"),
        "pick up the green statue",
    )
    assert act.ok, act.reason
    assert act.entity_id == "obj_1"


def test_an_unmeasured_colour_abstains_rather_than_refusing():
    """A grey stapler matches no HSV band, so classify_color returns None.

    That is "nobody established a colour", not "it is not green", and the
    difference decides whether a correct answer is refused.
    """
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "reason": "the statue"}',
        _labelled(None),
        "pick up the green statue",
    )
    assert act.ok, act.reason


def test_a_measured_colour_that_contradicts_the_task_still_refuses():
    """The protection this check exists for has to survive the fix."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "reason": "the statue"}',
        _labelled("red"),
        "pick up the green statue",
    )
    assert not act.ok
    assert "green" in act.reason


def test_a_noun_that_names_a_different_object_still_refuses():
    """Nouns keep the label rule: there the label IS all that is known, so
    absence really is contradiction."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "reason": "close enough"}',
        _labelled("green", obj_label="stone"),
        "pick up the green rock",
    )
    assert not act.ok
    assert "rock" in act.reason or "stone" in act.reason


def test_a_wrong_coloured_cube_is_still_refused():
    """The original reason this check exists, unchanged."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "cube_1", '
        '"point_2d": [1100, 650], "reason": "a cube"}',
        _labelled("green"),
        "pick up the red cube",
    )
    assert not act.ok


def test_the_colour_vocabulary_comes_from_the_detector():
    """Written out by hand it would drift; the bands are the source of truth."""
    from mt4_vision.detect import COLOR_RANGES
    from mt4_vision.instruct import _color_vocabulary

    assert _color_vocabulary() == frozenset(COLOR_RANGES)
    assert "green" in _color_vocabulary()
