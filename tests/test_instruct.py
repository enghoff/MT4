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

import mt4_vision.instruct as instruct
from mt4_vision.calib import Calibration
from mt4_vision.entities import Snapshot
from mt4_vision.instruct import (
    Observation,
    is_question,
    register_object,
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


# ---------------------------------------------------------------- alternate


# ------------------------------------------------------------ point_readings


# --------------------------------------------------------------- Grounding


# --------------------------------------------------------- measure_grounding


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


# ------------------------------------------------------------ noun_phrase


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


# ------------------------------------------------- a number the desk lacks
#
# "pick up the clamp and place it on marker 0", 2026-08-03, on a desk whose
# calibration carries markers 1, 2, 3 and 4 and no marker 0:
#
#   [1] TRANSFER obj_1 (clamp) -> marker_3   reason: clamp is on marker_3
#       -> moved the clamp onto marker 3 (free)
#   [2] TRANSFER obj_2 (clamp) -> marker_1   reason: clamp is on marker_1 as per task
#       -> moved the clamp onto marker 1 (free)
#   [3] DONE                                 reason: task already completed
#
# Three ok=True decisions and two real arm moves, none of them touching
# anything the instruction named. The cause was a direction: every number check
# derived its requirement by walking the SNAPSHOT's markers and asking whether
# the text mentioned each one. A number the desk does not have therefore
# produced no requirement at all, so the destination attribute set was empty and
# every marker satisfied it vacuously -- the guard was a no-op in exactly the
# case where the model has no correct answer available to give.


def _clamp_desk(**over):
    """The live desk shape: one graspable object and two free markers, 1 and 3.

    x pixels are above ``COORD_SCALE`` so each point has a single unambiguous
    reading, as in ``_transfer_snapshot``.
    """
    from mt4_vision.entities import Entity

    ents = [
        Entity(
            id="obj_1", kind="object", label="clamp", x=100.0, y=50.0,
            pixel=(1100.0, 150.0), pickable=True, source="vlm",
            yaw_deg=30.0, yaw_period_deg=180.0,
        ),
        Entity(
            id="marker_3", kind="marker", label="marker 3 (free)",
            x=150.0, y=250.0, pixel=(1100.0, 620.0), placeable=True,
            source="aruco",
        ),
        Entity(
            id="marker_1", kind="marker", label="marker 1 (free)",
            x=40.0, y=250.0, pixel=(1030.0, 400.0), placeable=True,
            source="aruco",
        ),
    ]
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    return Observation(
        frame=frame, annotated=frame, calib=CALIB,
        snapshot=Snapshot(token="t", entities=ents), **over,
    )


def test_the_correct_transfer_is_still_carried_out():
    """The whole point of refusing the rest: this one has to go through."""
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_3", '
        '"dest_2d": [1100, 620], "reason": "onto marker 3"}',
        _clamp_desk(),
        "pick up the clamp and place it on marker 3",
    )
    assert act.ok, act.reason
    assert (act.entity_id, act.dest_entity_id) == ("obj_1", "marker_3")


def test_an_unnumbered_destination_is_still_the_models_to_choose():
    """The guard reads numbers, so a task that names none constrains nothing."""
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_1", '
        '"dest_2d": [1030, 400], "reason": "a free marker"}',
        _clamp_desk(),
        "put the clamp on a free marker",
    )
    assert act.ok, act.reason
    assert act.dest_entity_id == "marker_1"


# --------------------------------------------- pointing alone was unchecked
#
# The module docstring says neither naming nor pointing is trusted alone. It was
# not holding. ``_matches_attributes`` ran against the id the model WROTE, so
# every branch that settles on the point instead -- a reply with
# ``entity_id: null``, or one whose name and point the task's own wording
# resolves -- reached the arm with no attribute check at all. Reproduced offline
# on the live snapshot shape, at both ends of a move.


def test_a_point_only_reply_that_lands_on_the_right_thing_still_works():
    """The gate is about contradiction, not about demanding an id."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": null, '
        '"point_2d": [1100, 150], "reason": "the statue"}',
        _labelled("green"),
        "pick up the green statue",
    )
    assert act.ok, act.reason
    assert act.entity_id == "obj_1"


# -- a destination id that is not an id ------------------------------------- #
#
# Measured live 2026-08-03 on "move stapler to center aruco marker": step 1
# picked the stapler, step 2 answered PLACE_ENTITY with entity_id "stapler" --
# the label of the thing in the jaws, which the prompt says is never a
# destination -- and the run was abandoned with the stapler still held. The
# distinction these pin: a destination that is not in the snapshot cannot mean
# "the desk does not have it" the way a pick target can, because markers and
# slots are enumerated exhaustively. So free text is discarded and the point
# decides, while an id-SHAPED string is still a claim about the desk and is
# still refused.


def test_a_place_whose_id_is_a_label_falls_back_to_the_point():
    act = _decide_on(
        '{"action": "PLACE_ENTITY", "entity_id": "stapler", '
        '"point_2d": [1100, 620], "reason": "put it down there"}',
        _transfer_snapshot(held="stapler"),
        "move the stapler to the marker",
    )
    assert act.ok, act.reason
    assert act.entity_id == "marker_3"
    # The junk the model wrote is still reported, so the transcript's
    # "[model said stapler]" makes the fallback visible rather than silent.
    assert act.model_entity_id == "stapler"


# -- words that name a kind already in the snapshot ------------------------- #


def test_a_kind_synonym_is_only_credited_when_that_kind_is_present():
    """With no marker there is nothing for "the aruco tag" to already be."""
    from mt4_vision.entities import Entity

    obs = _transfer_snapshot()
    only_object = Snapshot(
        token="t",
        entities=[
            Entity(id="obj_1", kind="object", label="stapler", x=100.0, y=50.0,
                   pixel=(1100.0, 150.0), pickable=True, source="vlm"),
        ],
    )
    assert "aruco" in instruct.unmatched_nouns("put it on the aruco", only_object)
    assert "aruco" not in instruct.unmatched_nouns("put it on the aruco", obs.snapshot)


def test_block_is_not_credited_as_a_cube():
    """A wooden block the HSV detector never saw must still reach the grounder."""
    snap = _transfer_snapshot().snapshot
    assert "block" in instruct.unmatched_nouns("pick up the block", snap)


# =========================================================================== #
# Gates removed 2026-08-03 -- owner's decision
#
# "let's get the code working and introduce checks if things fail, rather than
# have the implementation fail because we don't trust what qwen is telling us."
#
# Thirteen tests used to live here asserting that each of these replies was
# REFUSED. They are now asserted to proceed, which is the honest record of what
# was traded away: every one of these gates could turn a reply that identified
# the right thing into an abandoned task, and between them they abandoned three
# consecutive real runs (docs/qwen3_vl_policy_status.md §2u).
#
# What still refuses is tested further down: the physical envelope, a reply that
# identifies nothing at all, a question, and the gripper-state rules.


def test_an_unknown_destination_id_now_resolves_by_the_point():
    """Was: "the destination will not do: 'marker_9' is not in this snapshot"."""
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_9", '
        '"dest_2d": [1100, 620], "reason": "put it there"}',
        _transfer_snapshot(),
        "put the stapler somewhere",
    )
    assert act.ok, act.reason
    assert act.dest_entity_id == "marker_3"
    # The string the model wrote is still carried, so the transcript prints
    # "[model said dest marker_9]" and the substitution is never silent.
    assert act.model_dest_entity_id == "marker_9"


def test_an_unknown_pick_id_now_resolves_by_the_point():
    """Was: "'stapler' is not in this snapshot ... never a nearby substitute"."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "stapler", '
        '"point_2d": [1100, 150], "reason": "the stapler"}',
        _transfer_snapshot(),
        "pick up the stapler",
    )
    assert act.ok, act.reason
    assert act.entity_id == "obj_1"
    assert act.model_entity_id == "stapler"


def test_a_colour_that_disagrees_no_longer_refuses_anything():
    """Was two separate refusals, for cubes and for grounded objects alike.

    The cost, stated: a green cube is now picked for "the red cube" when the
    reply names it. Colour was the only thing telling two cubes apart.
    """
    for reply_id, px in (("cube_1", 650), ("obj_1", 150)):
        act = _decide_on(
            '{"action": "PICK_ENTITY", "entity_id": "%s", '
            '"point_2d": [1100, %d], "reason": "that one"}' % (reply_id, px),
            _labelled("green"),
            "pick up the red cube",
        )
        assert act.ok, act.reason
        assert act.entity_id == reply_id


def test_a_noun_that_names_a_different_object_no_longer_refuses():
    """Was: "obj_1 is 'green stone'" for a task naming a rock."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "reason": "close enough"}',
        _labelled("green", obj_label="stone"),
        "pick up the green rock",
    )
    assert act.ok, act.reason
    assert act.entity_id == "obj_1"


def test_a_destination_the_instruction_never_had_no_longer_refuses():
    """Was the clamp run: "marker 0 is not on this desk" (§2s).

    The arm now goes to whichever marker the reply names. Deliberate: the
    refusal was measured blocking tasks whose destination the snapshot listed.
    """
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_3", '
        '"dest_2d": [1100, 620], "reason": "clamp is on marker_3"}',
        _clamp_desk(),
        "pick up the clamp and place it on marker 0",
    )
    assert act.ok, act.reason
    assert act.dest_entity_id == "marker_3"


def test_done_is_taken_at_face_value():
    """Was: "reported the task complete, but marker 0 is not on this desk"."""
    act = _decide_on(
        '{"action": "DONE", "reason": "task already completed"}',
        _clamp_desk(),
        "pick up the clamp and place it on marker 0",
    )
    assert act.kind == "DONE" and act.ok


def test_a_named_marker_no_longer_has_to_be_the_one_acted_on():
    """Was: the task said marker 3, the reply said marker_1, refused."""
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_1", '
        '"dest_2d": [1030, 400], "reason": "over here"}',
        _clamp_desk(),
        "pick up the clamp and place it on marker 3",
    )
    assert act.ok, act.reason
    assert act.dest_entity_id == "marker_1"


def test_a_point_only_reply_is_no_longer_held_to_the_tasks_words():
    """Was: pointed at cube_1 for "the green statue", refused."""
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": null, '
        '"point_2d": [1100, 650], "reason": "that one"}',
        _labelled("green"),          # obj_1 green statue, cube_1 green cube
        "pick up the green statue",
    )
    assert act.ok, act.reason
    assert act.entity_id == "cube_1"


def test_the_id_wins_over_a_point_that_disagrees():
    """Was: "named obj_1 but pointed at cube_1, with nothing to settle which".

    The id is the model's explicit answer, so it leads and the point is only a
    fallback. No cross-examination either way.
    """
    act = _decide_on(
        '{"action": "PICK_ENTITY", "entity_id": "obj_1", '
        '"point_2d": [1100, 650], "reason": "that one"}',
        _labelled("green"),
        "pick up something",
    )
    assert act.ok, act.reason
    assert act.entity_id == "obj_1"


# -- what still refuses ----------------------------------------------------- #


def test_the_physical_envelope_still_refuses():
    """The one gate left, and the reason the rest are safe to drop.

    Entity.placeable / pickable carry reach, the J1 keep-out, ground Z, finger
    clearance and the desk polygon. Nothing above can command a pose the
    envelope would have rejected anyway.
    """
    act = _decide_on(
        '{"action": "TRANSFER", "entity_id": "obj_1", '
        '"point_2d": [1100, 150], "dest_entity_id": "marker_2", '
        '"dest_2d": [1030, 400], "reason": "put it there"}',
        _transfer_snapshot(),
        "put the stapler on marker 2",
    )
    assert not act.ok
    assert "occupied by cube_5" in act.reason


def test_a_reply_identifying_nothing_still_refuses():
    """Not a judgement about the reply -- the absence of one."""
    act = _decide_on(
        '{"action": "PLACE_ENTITY", "entity_id": "stapler", '
        '"point_2d": [5, 5], "reason": "there"}',
        _transfer_snapshot(held="stapler"),
        "move the stapler to the marker",
    )
    assert not act.ok
    assert "identifies a target" in act.reason


def test_aruco_and_tag_name_the_marker_kind_rather_than_a_new_object():
    """Grounding "aruco" registered obj_1 19.6mm from the marker it meant."""
    snap = _transfer_snapshot().snapshot          # holds a marker, labelled "stapler"
    for word in ("aruco", "tag", "fiducial", "markers"):
        # Nothing unmatched: "stapler" is a label in this snapshot and the
        # synonym now resolves to the marker kind that is in it.
        assert instruct.unmatched_nouns(f"move stapler to the {word}", snap) == set(), word
        # ...while a genuinely unknown noun in the same sentence still surfaces,
        # so this is a narrower vocabulary, not a disabled check.
        assert instruct.unmatched_nouns(f"move the pliers to the {word}", snap) == {
            "pliers"
        }, word
