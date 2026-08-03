"""Coordinate conventions and grounding -- mt4_vision.instruct_reply.

Every bug pinned here was found by driving the arm, which is an expensive way to
discover an arithmetic error. The one that motivated the file:

    python run_instruction.py "pick up the stapler and place it on marker 0"
    -> found a stapler but could not measure it: could not segment an object
       at (700, 687)

Qwen had located the stapler to within 4px. The reply ``[700, 700]`` was in a
0-1000 normalized space and meant pixel (896, 504); read as pixel (700, 700) it
landed 280px away on bare desk. It failed loudly there only by luck -- the same
misreading lands on a *different object* elsewhere in the frame and measures it
confidently, which nothing downstream can detect.

Measured 2026-08-02 on the live rig, four objects x two prompt forms x two frame
variants: 8 of 8 replies normalized, 1-10px from truth as normalized, 206-285px
from truth as pixels. So normalized is the default for a grounding reply, and a
coordinate above 1000 is the only thing that can rule it out.
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
    box_readings,
    measure_grounding,
    point_readings,
    to_frame_pixels,
)
from mt4_vision.instruct_reply import _grounding

FRAME_W, FRAME_H = 1280, 720
SIZE = (FRAME_W, FRAME_H)

# The live replies, verbatim, from the 2026-08-02 grounding run. Truth pixels for
# the cubes come from the HSV detector on the same frame; the stapler was read
# off the drawn grid by hand.
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


def test_box_readings_lead_with_the_normalized_reading():
    """0-1000 is the space this model answers in, and reading it that way lands
    2-13px from truth against 264-363px as raw pixels, 6 of 6 on one frame.

    Ordered like point_readings, deliberately: one prompt asks for both a box
    and a point, so one convention answers for both fields.
    """
    got = box_readings((100, 200, 160, 260), SIZE)
    assert got[0] == to_frame_pixels((100, 200, 160, 260), SIZE)
    assert got[1] == (100.0, 200.0, 160.0, 260.0)


def test_box_readings_drop_a_pixel_reading_that_centres_off_frame():
    """The live stapler box. Centre y as pixels is 738 on a 720px frame, so
    only the normalized reading survives -- which was the correct one."""
    got = box_readings((630, 650, 782, 827), SIZE)
    assert len(got) == 1
    assert got[0] == to_frame_pixels((630, 650, 782, 827), SIZE)


def test_box_readings_leave_one_reading_when_a_coordinate_rules_normalized_out():
    got = box_readings((1014, 364, 1066, 407), SIZE)
    assert got == ((1014.0, 364.0, 1066.0, 407.0),)


def test_a_decision_point_is_read_as_normalized_first():
    """The reply is 0-1000, whatever the prompt asks for.

    Measured over 3 targets x 2 prompt styles on one 1280x720 frame: read as
    0-1000 the box centres land 2-13px from truth, 6 of 6; read as raw pixels,
    264-363px away. The raw reading is kept second as a retry, because a
    coordinate under 1000 is only probably normalized.
    """
    got = point_readings((320, 473), SIZE)
    assert got[0] == (320 * FRAME_W / 1000.0, 473 * FRAME_H / 1000.0)
    assert got[1] == (320.0, 473.0)


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

