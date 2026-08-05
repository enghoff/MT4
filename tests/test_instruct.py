"""Tests for the policy layer (no hardware, no service).

Every bug this file pins was found by driving the arm, which is an expensive
way to discover an arithmetic error. The one that motivated the file:

    python ask_qwen.py "pick up the stapler and place it on marker 0"
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


# Model coordinates are 0-1000 across the frame, so a fixture that wants a box
# round a known pixel has to say so in that space.
def _norm(px: float, py: float) -> tuple[float, float]:
    return px * 1000.0 / FRAME_W, py * 1000.0 / FRAME_H


def _norm_box(x1: float, y1: float, x2: float, y2: float) -> str:
    a, b = _norm(x1, y1)
    c, d = _norm(x2, y2)
    return f"[{a:.4f}, {b:.4f}, {c:.4f}, {d:.4f}]"


def _close(got, want, tol: float = 0.5) -> bool:
    return all(abs(g - w) <= tol for g, w in zip(got, want))

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
# Qwen is the visual grounding, so it is told what it cannot see (a tag's
# printed number) and nothing else. A failure here is silent and expensive: the
# model answers from a list instead, and the list is cube-complete and
# object-sparse.


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


def test_the_prompt_offers_no_way_to_name_a_destination():
    """An id field is an invitation to answer "on the red thing" with a tag
    number, which reads as exact and cannot be checked against anything."""
    prompt = instruct.build_prompt(_desk(held="pen"), "put it on marker 3")
    assert "dest_marker" not in prompt
    assert "dest_2d" in prompt


def test_the_prompt_says_an_object_is_a_legal_destination():
    """The gap that forced the substitution: with dest_2d described as bare
    desk only, "on top of that" had no expressible answer but a tag id."""
    prompt = instruct.build_prompt(_desk(held="pen"), "put it on the red one")
    assert "on top of something else" in prompt


def test_the_prompt_lists_each_tag_position_for_echoing_back():
    """The listed numbers are the answer for a tag destination, not a hint
    about one, so they have to be in the 0-1000 space the reply is read in."""
    prompt = instruct.build_prompt(_desk(), "tidy up")
    # marker_3 sits at pixel (300, 500) of a 1280x720 frame.
    assert "marker_3 at (234, 694)" in prompt


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
    assert [e.kind for e in obs.markers] == ["marker", "marker"]


def test_the_overlay_circles_only_tags_over_a_0_1000_grid(monkeypatch):
    """A circled cube is the picture-shaped version of a cube list.

    The grid mode is pinned because the drawn numbers have to name the space the
    reply is read in: 0-1000 per axis, not pixels. Measured across five live
    layouts the three overlays score the same, so nothing here rests on a
    performance claim -- it rests on the labels not contradicting the prompt.
    ``mt4_camera_view`` still wants "pixel"; this caller does not.
    """
    drawn: list[list[str]] = []
    modes: list[str] = []

    def _fake_annotate(frame, entities=None, **kw):
        drawn.append([e.id for e in (entities or ())])
        modes.append(kw.get("grid"))
        return frame

    snap = _desk().snapshot
    monkeypatch.setattr(instruct, "annotate_for_pointing", _fake_annotate)
    monkeypatch.setattr(instruct, "load_calibration", lambda: CALIB)
    monkeypatch.setattr(instruct, "capture_scene", lambda calib, frame: object())
    monkeypatch.setattr(instruct, "build_snapshot", lambda sc, token: snap)
    instruct.observe(frame=np.zeros((FRAME_H, FRAME_W, 3), np.uint8))
    assert drawn == [["marker_3", "marker_2"]]
    assert modes == ["norm"]


def test_the_prompt_says_the_gridlines_are_the_0_1000_scale():
    """The drawn numbers and the requested space have to be the same space."""
    prompt = instruct.build_prompt(_desk(), "pick up the pen")
    assert "gridlines drawn on the image are that same scale" in prompt


# -- the reply is read faithfully ------------------------------------------ #


def test_a_transfer_carries_a_box_and_an_echoed_tag_position():
    """A tag destination is that tag's listed coordinates, copied back.

    marker_3 is printed at the 0-1000 form of pixel (300, 500), so a reply that
    means "on marker 3" says those two numbers and resolves to the tag's own
    robot position.
    """
    a, b = _norm(300.0, 500.0)
    act = _decide_on(
        '{"action": "TRANSFER", "box_2d": ' + _norm_box(180, 80, 220, 120) + ', '
        f'"label": "blue cube", "dest_2d": [{a:.4f}, {b:.4f}], '
        '"reason": "the cube goes on marker 3"}',
        _desk(),
        "put the blue cube on marker 3",
    )
    assert act.ok, act.reason
    assert act.kind == "TRANSFER"
    assert act.label == "blue cube"
    assert act.source is not None
    assert _close(act.source.box_px, (180.0, 80.0, 220.0, 120.0))
    assert _close(act.point_px, (200.0, 100.0))
    assert _close(act.dest_point_px, (300.0, 500.0))


def test_a_pick_needs_only_a_box():
    act = _decide_on(
        '{"action": "PICK", "box_2d": ' + _norm_box(180, 80, 220, 120) + ', '
        '"dest_2d": null, "label": "pen", "reason": "no destination named"}',
        _desk(),
        "pick up the pen",
    )
    assert act.ok, act.reason
    assert (act.kind, act.label) == ("PICK", "pen")
    assert act.dest_point_px is None


def test_a_place_takes_a_bare_pixel_destination():
    """"Somewhere clear" is a pixel, and so is everywhere else."""
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": [62.5, 694.4444], '
        '"reason": "an empty patch of desk"}',
        _desk(held="pen"),
        "put it down somewhere clear",
    )
    assert act.ok, act.reason
    assert act.kind == "PLACE"
    assert _close(act.dest_point_px, (80.0, 500.0))


def test_naming_a_tag_instead_of_pointing_at_it_is_refused():
    """The destination channel is coordinates only, so an id is not a
    destination -- and the refusal has to say what to write instead, because
    the numbers were already in the prompt."""
    act = _decide_on(
        '{"action": "PLACE", "dest_marker": "marker_2", "dest_2d": null, '
        '"reason": "on the tag"}',
        _desk(held="pen"),
        "put it on marker 2",
    )
    assert not act.ok
    assert "marker_2" in act.reason and "dest_2d" in act.reason


def test_an_object_destination_is_the_object_not_a_tag_beside_it():
    """The failure this protocol exists to stop: told to place on an object,
    the model answers with a tag id, which is exact, agrees with nothing and
    can be 46mm from the thing the task named. With coordinates as the only
    channel the reply lands where it pointed, and the transcript shows it."""
    obs = _desk(held="green cube")
    a, b = _norm(240.0, 460.0)   # on an object, 36mm clear of marker_3
    act = _decide_on(
        '{"action": "PLACE", '
        f'"dest_2d": [{a:.4f}, {b:.4f}], "reason": "onto the other cube"}}',
        obs,
        "place it on the red cube",
    )
    assert act.ok, act.reason
    assert _close(act.dest_point_px, (240.0, 460.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert _close((grasp.x, grasp.y), (120.0, 230.0))
    assert instruct.tag_at(obs, grasp.x, grasp.y) is None


def test_the_normalized_reading_leads_when_both_are_possible():
    """0-1000 is the space this model answers in, whatever the prompt asks.

    Measured over 3 targets x 2 prompt styles on one 1280x720 frame: box
    centres land 2-13px from truth read as 0-1000, and 264-363px away read as
    raw pixels, 6 of 6. Asking for pixels does not change what comes back.
    """
    act = _decide_on(
        '{"action": "PICK", "box_2d": [200, 300, 260, 360], "label": "pen", '
        '"reason": "the pen"}',
        _desk(),
        "pick up the pen",
    )
    assert act.ok, act.reason
    assert _close(
        act.source.box_px,
        (200 * FRAME_W / 1000, 300 * FRAME_H / 1000,
         260 * FRAME_W / 1000, 360 * FRAME_H / 1000),
    )
    # The raw-pixel reading stays as the retry, never discarded.
    assert act.source.alt_box_px == (200.0, 300.0, 260.0, 360.0)


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
    """The soft spot that survives reading 0-1000 first.

    A coordinate under 1000 is only *probably* normalized, so the raw-pixel
    reading is kept and retried -- and the retry fires only when the leading
    reading fails to segment. Where both readings land on real, separately
    measurable objects, a wrong leading reading is not caught by anything.

    On a 1280x720 frame `[777, 538, 920, 666]` centres at (1086, 433) read as
    0-1000 and (848, 602) read as pixels. On the live desk those were a stapler
    and a binder clip.
    """
    act = _decide_on(
        '{"action": "PICK", "box_2d": [777, 538, 920, 666], '
        '"label": "stapler", "reason": "the stapler"}',
        _desk(),
        "pick up the stapler",
    )
    assert act.ok, act.reason
    assert _close(act.source.point_px, (1086.4, 433.4), tol=1.0)
    assert act.source.alt_point_px == (848.5, 602.0)


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


def test_a_tag_the_desk_does_not_have_is_refused_by_name():
    """A tag with no listed position has no coordinates to echo, so the model
    can only name it. Measured on a desk of markers 1-4: "place it on marker 0"
    answers marker_3, then marker_1, then "task already completed", never once
    that the number is not there."""
    act = _decide_on(
        '{"action": "PLACE", "dest_marker": "marker_0", "reason": "marker 0"}',
        _desk(held="pen"),
        "put it on marker 0",
    )
    assert not act.ok
    assert "marker_0" in act.reason


def test_a_place_with_no_destination_is_refused():
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": null, "reason": "put it down"}',
        _desk(held="pen"),
        "put it down",
    )
    assert not act.ok
    assert "dest_2d" in act.reason


def test_a_full_gripper_is_not_told_to_prefer_transfer():
    """With the jaws full, TRANSFER is not the same errand in one motion -- it
    is picking up a second object and dropping the first. A standing "prefer
    TRANSFER" against "place the held object on marker 2" was measured live
    sending the arm to box a lookalike on the desk, twice in a row."""
    full = instruct.build_prompt(
        _desk(held="green cube"), "place the held object on marker 2"
    )
    empty = instruct.build_prompt(_desk(), "put the green cube on marker 2")
    assert "Prefer it over PICK then PLACE" in empty
    assert "Prefer it over PICK then PLACE" not in full
    # And it must say what boxing something would mean instead.
    assert "DIFFERENT object" in full
    assert "NOT lying on the desk" in full


def test_a_repeated_refusal_signature_ignores_remeasured_numbers():
    """A re-measured object jitters, so two refusals of one request are two
    strings: live, ``r=367mm`` then ``r=373mm`` for the same out-of-reach blob.
    Comparing the prose with numbers stripped is what makes them one."""
    from mt4_vision.instruct_worker import _refusal_signature

    a = "cannot pick it up: r=367mm is beyond the 350mm max reach"
    b = "cannot pick it up: r=373mm is beyond the 350mm max reach"
    assert _refusal_signature(a) == _refusal_signature(b)
    other = "the destination will not do: outside the camera frame"
    assert _refusal_signature(a) != _refusal_signature(other)


def test_the_same_refusal_twice_stops_instead_of_spending_the_budget():
    """One retry is for a bad exposure; a second is for nothing. Nothing moves
    between tries, so a reason that recurs is about the request. Live, this loop
    spent every step re-deciding an out-of-reach blob and reporting a max-reach
    error that had nothing to do with the actual task."""
    from mt4_vision.instruct_worker import TaskWorker

    class _UI:
        def __init__(self):
            self.lines = []

        def emit(self, text=""):
            self.lines.append(text)

        def set_status(self, msg):
            pass

    class _Fake:
        def __init__(self):
            self._ui = _UI()
            self._last_refusal = None

    w = _Fake()
    first = TaskWorker._refused(
        w, "cannot pick it up: r=367mm is beyond the 350mm max reach",
        "green cube", 1, 6,
    )
    second = TaskWorker._refused(
        w, "cannot pick it up: r=373mm is beyond the 350mm max reach",
        "green cube", 2, 6,
    )
    assert first is True
    assert second is False
    assert any("same way twice" in ln for ln in w._ui.lines)
    # Stopping with a full gripper has to say the object is still in the jaws.
    assert any("Still holding the green cube" in ln for ln in w._ui.lines)


def test_a_different_refusal_still_gets_its_retry():
    """The bad-exposure case the retry exists for: a tag that did not decode on
    one frame decodes on the next, and the refusal reads differently."""
    from mt4_vision.instruct_worker import TaskWorker

    class _UI:
        def emit(self, text=""):
            pass

        def set_status(self, msg):
            pass

    class _Fake:
        def __init__(self):
            self._ui = _UI()
            self._last_refusal = None

    w = _Fake()
    assert TaskWorker._refused(w, "marker_2 did not decode", None, 1, 6) is True
    assert TaskWorker._refused(
        w, "the destination will not do: r=367mm is beyond reach", None, 2, 6
    ) is True


def test_the_gripper_belief_does_not_gate_a_pick():
    """``held`` is typed in by hand, not sensed, so it cannot refuse an action.

    A stale belief is the common case, not the rare one: an abort mid-carry
    leaves the session sure it is holding something it dropped. Gating on it
    blocks a physically fine pick until someone types ``/held``.
    """
    act = _decide_on(
        '{"action": "PICK", "box_2d": ' + _norm_box(180, 80, 220, 120) + ', '
        '"label": "cube", "reason": "the cube"}',
        _desk(held="stapler"),
        "pick up the cube",
    )
    assert act.ok
    assert act.kind == "PICK"


def test_the_gripper_belief_does_not_gate_a_place():
    """Placing with jaws the session thinks are empty is a move and an open."""
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": [80, 500], "reason": "on the desk"}',
        _desk(),
        "leave the object you are holding on the table",
    )
    assert act.ok
    assert act.kind == "PLACE"


def test_a_model_chosen_stop_is_marked_declined():
    """STOP from the model is its answer, and `declined` is what says so.

    The loop needs to tell it from a STOP that parsing forced, because the two
    deserve opposite handling: one is a judgement about the request, the other
    is a bad reply worth another look.
    """
    act = _decide_on(
        '{"action": "STOP", "reason": "there is no red cube on this desk"}',
        _desk(),
        "open the gripper",
    )
    assert act.kind == "STOP"
    assert not act.ok
    assert act.declined
    assert "no red cube" in act.reason


def test_a_forced_stop_is_not_declined():
    """A malformed reply says nothing about the request, so it stays retryable."""
    for reply in (
        "I would pick up the blue cube.",
        '{"action": "PICK", "box_2d": null, "reason": "the cube"}',
        '{"action": "LOCATE_AT_PIXEL", "point_2d": [1, 2]}',
    ):
        act = _decide_on(reply, _desk(), "pick up the cube")
        assert act.kind == "STOP", reply
        assert not act.ok, reply
        assert not act.declined, reply


def test_a_declined_stop_stops_on_the_first_step_holding_or_not():
    """Six identical STOPs is what this replaces.

    Live, ``open the gripper`` was answered STOP six times running, reworded
    each time so the same-signature guard never fired, because a full gripper
    was taken as a reason to look again. The object stays in the jaws either
    way, so the loop reports that once instead of six times.
    """
    from mt4_vision.instruct_worker import TaskWorker

    class _UI:
        def __init__(self):
            self.lines = []

        def emit(self, text=""):
            self.lines.append(text)

        def set_status(self, msg):
            pass

    class _Fake:
        def __init__(self):
            self._ui = _UI()
            self._last_refusal = None

    holding = _Fake()
    TaskWorker._declined(holding, "there is no red cube on this desk", "red cube")
    assert any("STOP is the answer" in ln for ln in holding._ui.lines)
    assert any("Still holding the red cube" in ln for ln in holding._ui.lines)

    empty = _Fake()
    TaskWorker._declined(empty, "there is no red cube on this desk", None)
    assert any("STOP is the answer" in ln for ln in empty._ui.lines)
    assert not any("Still holding" in ln for ln in empty._ui.lines)


def test_an_unknown_action_is_refused():
    act = _decide_on('{"action": "LOCATE_AT_PIXEL", "point_2d": [1, 2]}', _desk(), "x")
    assert not act.ok
    assert "LOCATE_AT_PIXEL" in act.reason


def test_a_reply_that_is_not_json_is_refused():
    act = _decide_on("I would pick up the blue cube.", _desk(), "pick a cube")
    assert not act.ok
    assert "no JSON object" in act.reason


# -- destinations resolve to poses, and the pixel is not snapped ----------- #


def test_an_echoed_tag_position_lands_on_that_tag_squared():
    """marker_3's listed pixel, echoed back, resolves to its robot position.

    The round trip is the tag's robot XY through ``robot_to_pixel``, printed as
    whole 0-1000 units, and back through ``pixel_to_robot`` -- the same
    homography inverted, so only the rounding survives. Measured across the five
    tags on the live desk: 0.05-0.48mm.
    """
    obs = _desk(held="pen")
    act = Action("PLACE", True, "on the tag", dest_point_px=(300.0, 500.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert (grasp.x, grasp.y) == (150.0, 250.0)
    # A destination has no orientation of its own, so it squares to the axes.
    assert grasp.yaw_deg == 0.0
    assert grasp.yaw_period_deg == 90.0


def test_a_destination_pixel_is_projected_not_snapped_to_a_tag():
    """A pixel 4px from marker_3 still means that pixel, even though ``tag_at``
    reports it as standing on the tag. Snapping it would be the loop overriding
    the one thing it asked the model to decide."""
    obs = _desk(held="pen")
    act = Action("PLACE", True, "just beside it", dest_point_px=(304.0, 500.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert (grasp.x, grasp.y) == (152.0, 250.0)
    assert (grasp.x, grasp.y) != (150.0, 250.0)
    assert instruct.tag_at(obs, grasp.x, grasp.y).id == "marker_3"


def test_an_unreachable_destination_pixel_is_refused():
    obs = _desk(held="pen")
    act = Action("PLACE", True, "over there", dest_point_px=(5.0, 5.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is None
    assert "destination pixel" in why


def test_a_destination_the_camera_cannot_see_is_still_allowed():
    """Coverage says the arm would not see the result again, not that it cannot
    do the thing. That is the operator's call about a spot they pointed at, and
    it is worth 316 cm2 of the 1541 cm2 the arm can hold a grasp pose over."""
    from mt4_vision.workspace import work_region_block_reason

    obs = _desk(held="pen")
    # (320, 40) px is robot (160, 20): reachable, on the desk, and 40px from the
    # top border, inside the 50px margin the coverage test wants.
    assert "camera frame" in work_region_block_reason(160.0, 20.0, CALIB)
    act = Action("PLACE", True, "off to the side", dest_point_px=(320.0, 40.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert (grasp.x, grasp.y) == (160.0, 20.0)


def test_dropping_coverage_does_not_drop_reach():
    """The envelope gates are a different kind of thing and all still apply."""
    obs = _desk(held="pen")
    act = Action("PLACE", True, "way out there", dest_point_px=(960.0, 1360.0))
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is None
    assert "max reach" in why


def test_a_destination_axis_sets_the_release_orientation():
    """Two points, not an angle: the model has no access to the robot frame, so
    it points at a second place and the direction is read off in millimetres."""
    obs = _desk(held="pen")
    act = Action(
        "PLACE", True, "along the edge",
        dest_point_px=(300.0, 500.0), dest_axis_px=(400.0, 500.0),
    )
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert (grasp.x, grasp.y) == (150.0, 250.0)
    assert grasp.yaw_deg == 0.0
    # 180, not 90: an axis to lay the object along reads the same reversed.
    assert grasp.yaw_period_deg == 180.0

    act = Action(
        "PLACE", True, "across it",
        dest_point_px=(300.0, 500.0), dest_axis_px=(300.0, 600.0),
    )
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert grasp.yaw_deg == 90.0


def test_a_destination_axis_too_short_to_mean_anything_squares_instead():
    """Two points 5mm apart are one point and rounding noise, not a direction."""
    obs = _desk(held="pen")
    act = Action(
        "PLACE", True, "here",
        dest_point_px=(300.0, 500.0), dest_axis_px=(310.0, 500.0),
    )
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert grasp.yaw_deg == 0.0
    assert grasp.yaw_period_deg == 90.0


def test_a_zero_axis_is_read_as_no_axis():
    """This build fills an unused optional field with ``[0, 0]`` rather than
    leaving it null -- observed live beside a real dest_2d. Read literally that
    is "lay it along the line to the top-left corner", a wrist angle nothing
    asked for."""
    obs = _desk(held="pen")
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": [234, 694], "dest_axis_2d": [0, 0], '
        '"reason": "on the tag"}',
        obs,
        "put it on marker 3",
    )
    assert act.ok, act.reason
    assert act.dest_axis_px is None
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert grasp.yaw_period_deg == 90.0


def test_a_null_axis_is_read_as_no_axis():
    obs = _desk(held="pen")
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": [234, 694], '
        '"dest_axis_2d": [null, null], "reason": "on the tag"}',
        obs,
        "put it on marker 3",
    )
    assert act.ok, act.reason
    assert act.dest_axis_px is None


def test_a_reply_with_no_axis_squares_to_the_world_axes():
    obs = _desk(held="pen")
    act = _decide_on(
        '{"action": "PLACE", "dest_2d": [234, 694], "reason": "on the tag"}',
        obs,
        "put it on marker 3",
    )
    assert act.ok, act.reason
    assert act.dest_axis_px is None
    grasp, why = instruct.destination_grasp(obs, act)
    assert grasp is not None, why
    assert grasp.yaw_period_deg == 90.0


def test_tag_at_names_a_tag_underfoot_and_nothing_further_off():
    """Reporting only -- it labels a coordinate, it never moves one. The
    threshold is half a printed tag, which measures 41.2-44.1mm on a side."""
    obs = _desk()
    assert instruct.tag_at(obs, 150.0, 250.0).id == "marker_3"
    assert instruct.tag_at(obs, 150.0, 250.0 + instruct.ON_TAG_MM - 1) is not None
    assert instruct.tag_at(obs, 150.0, 250.0 + instruct.ON_TAG_MM + 1) is None


# -- REPORT: an answer about several things at once ------------------------ #
#
# "Find all the pickable objects" has no answer in prose. Which things the arm
# can take depends on reach, the J1 keep-out, ground Z, the desk polygon and the
# jaw width, and a model looking at a photograph can see none of them. So the
# answer is one box per object and the stack measures each and runs the real
# gate.
#
# The list comes from a SECOND call, and that is the load-bearing fact here.
# Measured on a nine-cube desk (2026-08-04): the decision prompt carrying an
# `objects` field returned 1 of 9, the same prompt worded harder returned 0, and
# a prompt that only enumerates returned 9 of 9. What these pin is the split
# that follows from it, and that the list survives intact once it arrives --
# boxed, counted, and never quietly shortened.


def _listing(*boxes: tuple[str, tuple[float, float, float, float]]) -> str:
    """An enumeration reply in the shape this model actually writes one.

    Fenced markdown around a JSON array of ``bbox_2d`` -- not the ``box_2d`` the
    decision schema names, and not bare. ``qwen.parse_regions`` is what reads
    it, which is why the enumeration call does not impose a schema of its own.
    """
    rows = ",\n".join(
        f'\t{{"bbox_2d": {_norm_box(*box)}, "label": "{label}"}}'
        for label, box in boxes
    )
    return f"```json\n[\n{rows}\n]\n```"


def _decide_two_call(decision: str, listing: str, obs, instruction: str) -> Action:
    """The real ``decide`` against the two canned replies a REPORT draws.

    First the decision, then the enumeration. Anything after that gets an empty
    string, so a third call would show up as a report of nothing rather than as
    a silently reused reply.
    """
    replies = [decision, listing]

    class _Reply:
        def __init__(self, text: str) -> None:
            self.text = text

    saved = instruct.ask
    instruct.ask = lambda *a, **k: _Reply(replies.pop(0) if replies else "")
    try:
        return instruct.decide(obs, instruction)
    finally:
        instruct.ask = saved


REPORT_REPLY = '{"action": "REPORT", "reason": "there are two things here"}'


def test_a_report_takes_its_list_from_the_enumeration_call():
    act = _decide_two_call(
        REPORT_REPLY,
        _listing(
            ("red cube", (180, 80, 220, 120)),
            ("stapler", (600, 400, 700, 460)),
        ),
        _desk(),
        "find all the pickable objects",
    )
    assert act.ok, act.reason
    assert act.kind == "REPORT"
    assert [g.label for g in act.report] == ["red cube", "stapler"]
    assert _close(act.report[0].box_px, (180.0, 80.0, 220.0, 120.0))
    assert _close(act.report[1].box_px, (600.0, 400.0, 700.0, 460.0))
    assert act.report_notes == ()
    # A report moves nothing, so it carries neither end of a motion.
    assert act.source is None and act.dest_point_px is None
    # And the prose answer is still the decision reply's, not the listing's.
    assert "two things here" in act.reason


def test_a_box_in_the_decision_reply_is_not_a_report():
    """The 1-of-9 failure, pinned. Given an `objects` field the model filled
    `box_2d` with one object and copied it into the list; whatever it writes
    there now, the report is what the enumeration call returned."""
    act = _decide_two_call(
        '{"action": "REPORT", "box_2d": ' + _norm_box(10, 10, 30, 30) + ', '
        '"label": "red cube", "objects": [{"box_2d": '
        + _norm_box(10, 10, 30, 30) + ', "label": "red cube"}], '
        '"reason": "only the red cube"}',
        _listing(
            ("red cube", (180, 80, 220, 120)),
            ("green cube", (300, 200, 340, 240)),
            ("blue cube", (600, 400, 640, 440)),
        ),
        _desk(),
        "find all the pickable objects",
    )
    assert act.ok, act.reason
    assert len(act.report) == 3
    assert not _close(act.report[0].box_px, (10.0, 10.0, 30.0, 30.0))


def test_a_report_box_is_read_in_the_same_space_a_pick_box_is():
    """One convention for every box in the protocol. A reader should not have
    to remember which field follows which rule -- see ``box_readings``."""
    act = _decide_two_call(
        REPORT_REPLY,
        '[{"bbox_2d": [200, 300, 260, 360], "label": "pen"}]',
        _desk(),
        "what is on the desk?",
    )
    assert act.ok, act.reason
    assert _close(
        act.report[0].box_px,
        (200 * FRAME_W / 1000, 300 * FRAME_H / 1000,
         260 * FRAME_W / 1000, 360 * FRAME_H / 1000),
    )
    assert act.report[0].alt_box_px == (200.0, 300.0, 260.0, 360.0)


def test_an_empty_report_is_a_real_answer_not_a_refusal():
    """"There are none" is what the question asked for, so it is not a failure
    and there is nothing to look at a second time."""
    act = _decide_two_call(
        '{"action": "REPORT", "reason": "the desk is bare"}',
        "[]",
        _desk(),
        "find all the pickable objects",
    )
    assert act.ok
    assert act.report == ()
    assert not act.declined
    assert "bare" in act.reason


def test_a_listing_the_service_could_not_answer_is_a_note_not_a_crash():
    """The decision is already made and reported by the time the list is asked
    for. Losing the list to a service error must not lose the reply that came
    with it, and must not be silent either."""
    from mt4_vision.qwen import QwenError

    def _boom(*a, **k):
        raise QwenError("qwen service unreachable at http://127.0.0.1:8766")

    obs = _desk()
    saved = instruct.ask
    instruct.ask = _boom
    try:
        found, notes = instruct.enumerate_objects(obs, "find all the objects")
    finally:
        instruct.ask = saved
    assert found == ()
    assert len(notes) == 1 and "unreachable" in notes[0]


def test_an_unreadable_listing_entry_is_named_and_the_rest_survive():
    """A report claims completeness, so a dropped entry is a wrong answer, not
    a smaller one. The readable objects stand and the gap is printed."""
    act = _decide_two_call(
        REPORT_REPLY,
        '[{"bbox_2d": ' + _norm_box(180, 80, 220, 120) + ', "label": "red cube"}, '
        '{"bbox_2d": [4000, 4000, 4100, 4100], "label": "thing at the back"}]',
        _desk(),
        "find all the pickable objects",
    )
    assert act.ok, act.reason
    assert [g.label for g in act.report] == ["red cube"]
    assert len(act.report_notes) == 1
    assert "thing at the back" in act.report_notes[0]


def test_a_whole_frame_entry_is_rejected_inside_a_report_too():
    """Asked to locate something absent this build returns the whole image
    rather than an empty list, and inside a report that would read as one more
    object. The same threshold a PICK's box faces applies to every entry."""
    act = _decide_two_call(
        REPORT_REPLY,
        '[{"bbox_2d": [0, 0, 1000, 1000], "label": "desk"}, '
        '{"bbox_2d": ' + _norm_box(180, 80, 220, 120) + ', "label": "red cube"}]',
        _desk(),
        "find all the pickable objects",
    )
    assert act.ok, act.reason
    assert [g.label for g in act.report] == ["red cube"]
    assert "% of the frame" in act.report_notes[0]


def test_the_decision_schema_names_no_object_list():
    """The regression this design exists to prevent. An ``objects`` field in
    the decision schema collapsed a report to 1 of 9 objects; moving it to the
    front of the schema fixed the count and broke TRANSFER instead -- asked to
    "put a blue cube on marker 2", the reply put marker_2's box in ``box_2d``,
    which is the field the PICK TARGET is read from. A silent wrong target is
    worse than a short list, so the decision prompt has no list in it at all."""
    for task in ("find all the pickable objects", "put a blue cube on marker 2"):
        prompt = instruct.build_prompt(_desk(), task)
        assert '"objects"' not in prompt, task
        assert "box_2d" in prompt, task
    # REPORT is still offered, and says it needs no box.
    prompt = instruct.build_prompt(_desk(), "find all the pickable objects")
    assert "REPORT" in prompt
    assert "no box is needed here" in prompt


def test_the_enumeration_prompt_carries_the_task_and_forbids_pre_filtering():
    """The one thing the model must not do here. Reach, the keep-out, the desk
    edge and the jaw width decide what can be picked up and none is in the
    photograph -- and the decision prompt's own attempt at a list came back
    reasoning that "the red cube is the only object ... that can be picked up",
    which is the model ruling on geometry it cannot see."""
    prompt = instruct.build_report_prompt("find all the pickable objects")
    assert 'Task: "find all the pickable objects"' in prompt
    assert "bbox_2d" in prompt
    assert "not only the ones you think" in prompt
    assert "leaving one out" in prompt


def test_measure_report_runs_the_pick_gate_on_every_object():
    """Pickability is answered by ``source_entity`` -- the same predicate a PICK
    must satisfy -- so a report cannot advertise a target a pick would refuse,
    nor omit one it would accept. A box that will not segment keeps its row."""
    act = _decide_two_call(
        REPORT_REPLY,
        _listing(
            ("red cube", (180, 80, 220, 120)),
            ("stapler", (600, 400, 700, 460)),
            ("smudge", (900, 600, 940, 640)),
        ),
        _desk(),
        "find all the pickable objects",
    )
    obs = _desk()

    class _Obj:
        long_mm, short_mm = 20.0, 20.0

    def _fake_measure(_obs, g, *, label=None, object_height_mm=None):
        if g.label == "smudge":
            return None, "GrabCut found no foreground in that box"
        return _Obj(), ""

    def _fake_entity(_obs, _obj, *, eid="obj_1"):
        if eid == "obj_2":
            return Entity(
                id=eid, kind="object", label="stapler", x=300.0, y=190.0,
                pickable=False, reason="r=356mm is beyond the 350mm max reach",
            )
        return Entity(
            id=eid, kind="object", label="red cube", x=200.0, y=-40.0,
            pickable=True,
        )

    saved = (instruct.measure_grounding, instruct.source_entity)
    instruct.measure_grounding, instruct.source_entity = _fake_measure, _fake_entity
    try:
        found = instruct.measure_report(obs, act)
    finally:
        instruct.measure_grounding, instruct.source_entity = saved

    assert [f.index for f in found] == [1, 2, 3]
    assert [f.pickable for f in found] == [True, False, False]
    assert found[0].note == "pickable"
    assert "max reach" in found[1].note
    assert "no foreground" in found[2].note
    # The unmeasurable one keeps its row rather than shrinking the answer.
    assert "could not be measured" in found[2].line()
    assert "beyond the 350mm max reach" in found[1].line()


def test_a_report_is_saved_with_its_boxes():
    """``/save`` writes the decision; for a report the boxes are the question
    and the measured rows (added by the worker) are the answer."""
    act = _decide_two_call(
        REPORT_REPLY,
        _listing(("red cube", (180, 80, 220, 120))),
        _desk(),
        "find all the pickable objects",
    )
    d = act.as_dict()
    assert d["action"] == "REPORT" and d["ok"] is True
    assert d["objects"][0]["label"] == "red cube"
    assert _close(d["objects"][0]["box_px"], (180.0, 80.0, 220.0, 120.0))
    assert _close(d["objects"][0]["point_px"], (200.0, 100.0))


# -- grip geometry --------------------------------------------------------- #


def test_a_pick_assumes_the_configured_cube_height():
    """The pick path pins ``object_height_mm`` to ``calib.cube_height_mm``.

    Height cannot be measured from one view, and every millimetre of error in
    it moves the aim point by 1.4-1.9mm on this mount. Leaving it unset infers
    it from the silhouette, which read 0.4 to 16.8mm for 20mm cubes across nine
    readings of one frame (2026-08-04) because the silhouette takes in shadow
    and its size swings with the box the model drew. Pinning it routes the
    centroid through the calibrated cube-top map instead, which lands 1.0-2.6mm
    from the cube detector's reading where inferring landed 8.1-22.5mm. A fixed
    zero -- no height at all -- aims 18.1-22.4mm outward of a 20mm cube.
    """
    seen: dict[str, object] = {}

    sentinel = object()

    def _fake_measure(obs, g, *, label=None, object_height_mm=sentinel):
        seen["height"] = object_height_mm
        seen["box"] = g.box_px
        return object(), ""

    act = _decide_on(
        '{"action": "PICK", "box_2d": ' + _norm_box(180, 80, 220, 120) + ', '
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
    assert seen["height"] == _desk().calib.cube_height_mm
    assert _close(seen["box"], (180.0, 80.0, 220.0, 120.0))


def test_a_grasp_defaults_to_table_z():
    """``Grasp.z is None`` means ``calib.table_z``, which is the low grip."""
    from mt4_vision.motion import square_place

    assert square_place(100.0, 50.0).grasp_z(CALIB) == CALIB.table_z


# -- the transcript record ------------------------------------------------- #


def test_as_dict_carries_the_box_and_the_destination():
    a, b = _norm(300.0, 500.0)
    act = _decide_on(
        '{"action": "TRANSFER", "box_2d": ' + _norm_box(180, 80, 220, 120) + ', '
        f'"label": "cube", "dest_2d": [{a:.4f}, {b:.4f}], "reason": "move it"}}',
        _desk(),
        "put the cube on marker 3",
    )
    d = act.as_dict()
    assert d["action"] == "TRANSFER"
    assert _close(d["box_px"], (180.0, 80.0, 220.0, 120.0))
    assert _close(d["point_px"], (200.0, 100.0))
    assert _close(d["dest_point_px"], (300.0, 500.0))
    # No ids at either end: neither a pick target nor a destination is a name.
    assert "entity_id" not in d and "dest_entity_id" not in d


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
