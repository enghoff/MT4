"""Tests for the policy layer (no hardware, no service).

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
206-285px from truth as pixels. So both readings are kept, and neither is
discarded on the strength of the prompt alone.

The protocol these cover: the model is given the ArUco tags and nothing else,
answers with a box rather than an id, and is not second-guessed about what it
boxed. So the three things worth pinning are that nothing pickable reaches the
prompt, that the reply is read faithfully, and that every refusal is structural
or physical.

Run: python tests/test_instruct.py  (or pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mt4_vision.instruct as instruct
from mt4_vision.calib import Calibration
from mt4_vision.entities import Entity, Snapshot
from mt4_vision.instruct import Action, Observation

FRAME_W, FRAME_H = 1280, 720
SIZE = (FRAME_W, FRAME_H)

MM_PER_PX = 0.5
CALIB = Calibration(
    homography=[[MM_PER_PX, 0.0, 0.0], [0.0, MM_PER_PX, 0.0], [0.0, 0.0, 1.0]],
    table_z=127.2,
    safe_z=155.0,
)


def _obs(*, entities=(), **over) -> Observation:
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    return Observation(
        frame=frame, annotated=frame, calib=CALIB,
        snapshot=Snapshot(token="t", entities=list(entities), summary="cubes=2"),
        **over,
    )


def _marker(n: int, x: float, y: float, px: float, py: float, **over) -> Entity:
    return Entity(
        id=f"marker_{n}", kind="marker", label=f"marker {n} (free)",
        x=x, y=y, pixel=(px, py), placeable=True, source="aruco", **over,
    )


def _cube(n: int, x: float, y: float, px: float, py: float) -> Entity:
    return Entity(
        id=f"cube_{n}", kind="cube", label="blue cube", x=x, y=y,
        pixel=(px, py), pickable=True, source="hsv",
    )


def _desk(**over) -> Observation:
    """Two cubes and two decoded tags. The cubes must never reach the model.

    Pixels and robot millimetres agree under CALIB (pixel = robot / 0.5), so a
    destination pixel in these tests projects where the fixture says it does
    and the work-region gate sees a reachable point.
    """
    return _obs(
        entities=[
            _cube(1, 100.0, 50.0, 200.0, 100.0),
            _cube(2, 40.0, 90.0, 80.0, 180.0),
            _marker(3, 150.0, 250.0, 300.0, 500.0),
            _marker(2, 40.0, 250.0, 80.0, 500.0),
        ],
        **over,
    )


def _decide_on(reply_json: str, obs, instruction: str) -> Action:
    """Run the real ``decide`` against a canned model reply."""

    class _Reply:
        text = reply_json

    saved = instruct.ask
    instruct.ask = lambda *a, **k: _Reply()
    try:
        return instruct.decide(obs, instruction)
    finally:
        instruct.ask = saved


# -- nothing pickable reaches the model ------------------------------------ #
#
# The whole point of the 2026-08-03 rewrite. Qwen is the visual grounding, so it
# is told what it cannot see (a tag's printed number) and nothing else. A
# regression here is silent and expensive: the model starts answering from a
# list again, and the list is cube-complete and object-sparse.


def test_the_prompt_names_the_tags_and_nothing_else():
    # An instruction that names no object, so anything cube-shaped in the
    # prompt got there from the detector rather than from the user.
    prompt = instruct.build_prompt(_desk(), "pick up the thing on the left")
    assert "marker_3" in prompt
    assert "marker_2" in prompt
    for banned in ("cube_1", "cube_2", "blue cube", "cube", "slot_", "obj_"):
        assert banned not in prompt, f"{banned!r} leaked into the prompt"


def test_the_prompt_advertises_no_capabilities():
    """A capability flag invites the model to treat the list as the set of
    legal answers, which is the opposite of what the list is for."""
    prompt = instruct.build_prompt(_desk(), "tidy the desk")
    for banned in ("can be picked up", "can be placed on", "not available"):
        assert banned not in prompt


def test_the_prompt_says_the_tag_list_is_not_the_desk():
    """Without this the model reads a short list as "these are the only things
    here" and forces the task onto one of them."""
    prompt = instruct.build_prompt(_desk(), "pick up the pen")
    assert "NOT a list of what is on the desk" in prompt


def test_the_instruction_is_passed_through_verbatim():
    """No noun extraction, no stopword filter, no rewriting.

    A word list that decides which parts of a sentence name objects fails on
    ordinary English: matching words exactly, ``holding`` is a thing to go and
    find, and grounding it registers a 169x19mm object at robot (-456, -178).
    """
    text = "leave the object you are holding on the table, please comply"
    prompt = instruct.build_prompt(_desk(held="blue cube"), text)
    assert f'Task: "{text}"' in prompt


def test_markers_are_the_only_model_facing_entities():
    obs = _desk()
    assert [e.id for e in obs.markers] == ["marker_3", "marker_2"]
    # The full snapshot still has the cubes -- the internal gates need them.
    assert len(obs.snapshot.entities) == 4
    assert obs.marker("cube_1") is None
    assert obs.marker("marker_3") is not None


def test_the_overlay_circles_only_tags(monkeypatch):
    """A circled cube is the picture-shaped version of a cube list."""
    drawn: list[list[str]] = []

    def _fake_annotate(frame, entities=None, **kw):
        drawn.append([e.id for e in (entities or ())])
        return frame

    snap = _desk().snapshot
    monkeypatch.setattr(instruct, "annotate_for_pointing", _fake_annotate)
    monkeypatch.setattr(instruct, "load_calibration", lambda: CALIB)
    monkeypatch.setattr(instruct, "capture_scene", lambda calib, frame: object())
    monkeypatch.setattr(instruct, "build_snapshot", lambda sc, token: snap)
    instruct.observe(frame=np.zeros((FRAME_H, FRAME_W, 3), np.uint8))
    assert drawn == [["marker_3", "marker_2"]]


# -- the reply is read faithfully ------------------------------------------ #


def test_a_transfer_carries_a_box_and_a_tag():
    act = _decide_on(
        '{"action": "TRANSFER", "box_2d": [180, 80, 220, 120], '
        '"label": "blue cube", "dest_marker": "marker_3", '
        '"reason": "the cube goes on marker 3"}',
        _desk(),
        "put the blue cube on marker 3",
    )
    assert act.ok, act.reason
    assert act.kind == "TRANSFER"
    assert act.label == "blue cube"
    assert act.source is not None
    assert act.source.box_px == (180.0, 80.0, 220.0, 120.0)
    assert act.point_px == (200.0, 100.0)
    assert act.dest_entity_id == "marker_3"


def test_a_pick_needs_only_a_box():
    act = _decide_on(
        '{"action": "PICK", "box_2d": [180, 80, 220, 120], '
        '"label": "pen", "dest_marker": null, "dest_2d": null, '
        '"reason": "no destination named"}',
        _desk(),
        "pick up the pen",
    )
    assert act.ok, act.reason
    assert (act.kind, act.label) == ("PICK", "pen")
    assert act.dest_entity_id is None and act.dest_point_px is None


def test_a_place_takes_a_bare_pixel_destination():
    """"Somewhere clear" is a pixel, not a slot id. There are no slot ids."""
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": [80, 500], '
        '"reason": "an empty patch of desk"}',
        _desk(held="pen"),
        "put it down somewhere clear",
    )
    assert act.ok, act.reason
    assert act.kind == "PLACE"
    assert act.dest_point_px == (80.0, 500.0)
    assert act.dest_entity_id is None


def test_a_tag_destination_wins_when_both_are_given():
    act = _decide_on(
        '{"action": "PLACE", "dest_marker": "marker_2", "dest_2d": [10, 10], '
        '"reason": "on the tag"}',
        _desk(held="pen"),
        "put it on marker 2",
    )
    assert act.ok, act.reason
    assert act.dest_entity_id == "marker_2"
    assert act.dest_point_px is None


def test_pixels_lead_when_both_readings_are_possible():
    """Both survive, pixels first, because the prompt draws the grid in that
    space. The other is carried for ``measure_grounding`` to retry with."""
    act = _decide_on(
        '{"action": "PICK", "box_2d": [200, 300, 260, 360], "label": "pen", '
        '"reason": "the pen"}',
        _desk(),
        "pick up the pen",
    )
    assert act.ok, act.reason
    assert act.source.box_px == (200.0, 300.0, 260.0, 360.0)
    assert act.source.alt_box_px is not None
    ax0, ay0, _ax1, _ay1 = act.source.alt_box_px
    assert abs(ax0 - 200 * FRAME_W / 1000.0) < 0.5
    assert abs(ay0 - 300 * FRAME_H / 1000.0) < 0.5


def test_the_live_stapler_box_reads_as_normalized_only():
    """The failure this file was opened for. Read as pixels, that box centres
    at y=738 on a 720px frame -- off the bottom, so not a possible reading at
    all, and the normalized one (which was right to 4px) is what is left."""
    act = _decide_on(
        '{"action": "PICK", "box_2d": [630, 650, 782, 827], "label": "stapler", '
        '"reason": "the stapler"}',
        _desk(),
        "pick up the stapler",
    )
    assert act.ok, act.reason
    x0, y0, _x1, _y1 = act.source.box_px
    assert abs(x0 - 630 * FRAME_W / 1000.0) < 0.5
    assert abs(y0 - 650 * FRAME_H / 1000.0) < 0.5
    assert act.source.alt_box_px is None


def test_both_readings_of_a_box_can_hit_different_real_objects():
    """The known hole, pinned so a future fix has a failing case to aim at.

    On a 1280x720 frame the box `[777, 538, 920, 666]` centres at (848, 602)
    read as pixels and (1086, 433) read as 0-1000. On the live desk those are a
    binder clip and a stapler. Both segment, so the measure-then-retry chain
    never rejects the first, and the loop grips whichever reading leads.

    Three ways of asking the reply which space it meant were measured and none
    works -- see the module docstring in ``mt4_vision.instruct``. Until
    something outside the reply supplies that signal, pixels lead and the other
    reading is only a retry.
    """
    act = _decide_on(
        '{"action": "PICK", "box_2d": [777, 538, 920, 666], '
        '"label": "stapler", "reason": "the stapler"}',
        _desk(),
        "pick up the stapler",
    )
    assert act.ok, act.reason
    assert act.source.point_px == (848.5, 602.0)
    assert act.source.alt_point_px is not None
    ax, ay = act.source.alt_point_px
    assert abs(ax - 1086) < 2 and abs(ay - 433) < 2, (ax, ay)


def test_done_is_taken_at_face_value():
    act = _decide_on(
        '{"action": "DONE", "reason": "the cube is on marker 3"}',
        _desk(),
        "put the blue cube on marker 3",
    )
    assert act.kind == "DONE" and act.ok


def test_a_question_now_moves_the_arm():
    """The text is not preprocessed, and that includes not deciding on the
    operator's behalf that a question was not an order. Deliberate: a phrasing
    rule cannot tell a survey question from a politely-worded instruction."""
    act = _decide_on(
        '{"action": "PICK", "box_2d": [180, 80, 220, 120], '
        '"label": "cube", "reason": "the only thing not a marker"}',
        _desk(),
        "is there anything on the desk that is not a cube?",
    )
    assert act.ok and act.kind == "PICK"


# -- what still refuses, and all of it is structural or physical ----------- #


def test_a_missing_box_is_refused():
    act = _decide_on(
        '{"action": "PICK", "box_2d": null, "reason": "the cube"}',
        _desk(),
        "pick up the cube",
    )
    assert not act.ok
    assert "box_2d" in act.reason


def test_a_whole_frame_box_is_refused_as_a_non_answer():
    """Asked to locate something absent, this build returns the whole image
    rather than declining. Measured on "location": box (0, 0, 1000, 1000)."""
    act = _decide_on(
        f'{{"action": "PICK", "box_2d": [0, 0, {FRAME_W}, {FRAME_H}], '
        '"label": "location", "reason": "everything"}',
        _desk(),
        "pick up the location",
    )
    assert not act.ok
    assert "% of the frame" in act.reason


def test_a_tag_that_did_not_decode_is_named_in_the_refusal():
    """The one text-derived rule, and it concerns the datum we supply rather
    than the model's grounding. Measured on a desk of markers 1-4: "place it on
    marker 0" answers marker_3, then marker_1, then "task already completed",
    never once that the number is not there."""
    act = _decide_on(
        '{"action": "PLACE", "dest_marker": "marker_0", "reason": "marker 0"}',
        _desk(held="pen"),
        "put it on marker 0",
    )
    assert not act.ok
    assert "marker_0" in act.reason
    assert "marker_3" in act.reason and "marker_2" in act.reason


def test_a_place_with_no_destination_is_refused():
    act = _decide_on(
        '{"action": "PLACE", "dest_marker": null, "dest_2d": null, '
        '"reason": "put it down"}',
        _desk(held="pen"),
        "put it down",
    )
    assert not act.ok
    assert "destination" in act.reason


def test_picking_while_holding_is_refused_and_names_the_correction():
    act = _decide_on(
        '{"action": "PICK", "box_2d": [180, 80, 220, 120], '
        '"label": "cube", "reason": "the cube"}',
        _desk(held="stapler"),
        "pick up the cube",
    )
    assert not act.ok
    assert "stapler" in act.reason and "/held" in act.reason


def test_placing_with_an_empty_gripper_is_refused_and_names_the_correction():
    """Nothing on this rig can sense a grip, so ``held`` is session state that
    only the operator can correct. A refusal that does not name ``/held`` leaves
    them re-wording the task instead of fixing the state it was judged against.
    """
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": [80, 500], "reason": "on the desk"}',
        _desk(),
        "leave the object you are holding on the table",
    )
    assert not act.ok
    assert "/held" in act.reason


def test_an_unknown_action_is_refused():
    act = _decide_on('{"action": "LOCATE_AT_PIXEL", "point_2d": [1, 2]}', _desk(), "x")
    assert not act.ok
    assert "LOCATE_AT_PIXEL" in act.reason


def test_a_reply_that_is_not_json_is_refused():
    act = _decide_on("I would pick up the blue cube.", _desk(), "pick a cube")
    assert not act.ok
    assert "no JSON object" in act.reason


# -- destinations resolve to poses, and the pixel is not snapped ----------- #


def test_a_tag_destination_uses_its_calibrated_position():
    obs = _desk(held="pen")
    act = Action("PLACE", True, "on the tag", dest_entity_id="marker_3")
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert (grasp.x, grasp.y) == (150.0, 250.0)


def test_a_destination_pixel_is_projected_not_snapped_to_a_tag():
    """A pixel 4px from marker_2 still means that pixel. Snapping it would be
    the loop overriding the one thing it asked the model to decide."""
    obs = _desk(held="pen")
    act = Action("PLACE", True, "just beside it", dest_point_px=(304.0, 500.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert (grasp.x, grasp.y) == (152.0, 250.0)
    assert (grasp.x, grasp.y) != (150.0, 250.0)


def test_an_unreachable_destination_pixel_is_refused():
    obs = _desk(held="pen")
    act = Action("PLACE", True, "over there", dest_point_px=(5.0, 5.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is None
    assert "destination pixel" in why


def test_a_destination_naming_an_undecoded_tag_is_refused():
    obs = _desk(held="pen")
    act = Action("PLACE", True, "on it", dest_entity_id="marker_9")
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is None
    assert "marker_9" in why


# -- grip geometry --------------------------------------------------------- #


def test_a_pick_measures_at_zero_assumed_height():
    """Grip as low as possible, at the point the model identified, oriented by
    the GrabCut mask. That is exactly ``object_height_mm=0`` -- no height
    inference and no parallax de-inflation anywhere in the pick path."""
    seen: dict[str, object] = {}

    def _fake_measure(obs, g, *, label=None, object_height_mm=None):
        seen["height"] = object_height_mm
        seen["box"] = g.box_px
        return object(), ""

    act = _decide_on(
        '{"action": "PICK", "box_2d": [180, 80, 220, 120], '
        '"label": "cube", "reason": "the cube"}',
        _desk(),
        "pick up the cube",
    )
    saved = instruct.measure_grounding
    instruct.measure_grounding = _fake_measure
    try:
        obj, why = instruct.measure_source(_desk(), act)
    finally:
        instruct.measure_grounding = saved

    assert obj is not None, why
    assert seen["height"] == 0.0 == instruct.PICK_AT_TABLE_HEIGHT_MM
    assert seen["box"] == (180.0, 80.0, 220.0, 120.0)


def test_a_grasp_defaults_to_table_z():
    """``Grasp.z is None`` means ``calib.table_z``, which is the low grip."""
    from mt4_vision.motion import square_place

    assert square_place(100.0, 50.0).grasp_z(CALIB) == CALIB.table_z


def test_grasp_for_squares_a_tag_destination():
    grasp = instruct.grasp_for(_marker(3, 150.0, 250.0, 1100.0, 620.0), CALIB)
    assert (grasp.x, grasp.y) == (150.0, 250.0)
    assert grasp.yaw_deg == 0.0
    assert grasp.yaw_period_deg == 90.0


# -- the transcript record ------------------------------------------------- #


def test_as_dict_carries_the_box_and_the_destination():
    act = _decide_on(
        '{"action": "TRANSFER", "box_2d": [180, 80, 220, 120], '
        '"label": "cube", "dest_marker": "marker_3", "reason": "move it"}',
        _desk(),
        "put the cube on marker 3",
    )
    d = act.as_dict()
    assert d["action"] == "TRANSFER"
    assert d["box_px"] == [180.0, 80.0, 220.0, 120.0]
    assert d["point_px"] == [200.0, 100.0]
    assert d["dest_entity_id"] == "marker_3"
    # No entity_id and no agreement flags: there is no id to agree about.
    assert "entity_id" not in d and "agreed" not in d


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
