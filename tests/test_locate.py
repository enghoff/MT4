"""Tests for pixel-hint -> measured object (no hardware, no camera).

Synthetic frames through a synthetic homography, so the expected millimetres and
degrees are known exactly rather than eyeballed. The properties that matter:

* an off-centre hint converges to the same centroid -- this is why a VLM only
  has to land *on* the object, and the whole reason the caller-side grounding
  design works at all;
* extents and axis yaw come out in true table millimetres/degrees;
* implausible measurements and lost objects REFUSE rather than returning a
  confident wrong number, because the arm acts on whatever comes back.

Run: python tests/test_locate.py  (or pytest)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.calib import Calibration
from mt4_vision.locate import (
    LocateError,
    grasp_feasibility,
    measure,
    refine_at_hint,
    relocate,
)

# 0.5 mm/px, robot origin at pixel (0, 0), +y down the image. A pure scale so
# expected values are arithmetic, not a fit.
MM_PER_PX = 0.5
CALIB = Calibration(
    homography=[[MM_PER_PX, 0.0, 0.0], [0.0, MM_PER_PX, 0.0], [0.0, 0.0, 1.0]],
    table_z=127.2,
    safe_z=155.0,
    grip_open_s=140,
    grip_close_s=240,
)
from rig import CALIB as RIG_CALIB, calibration as rig_calibration

FRAME_W, FRAME_H = 1280, 720
DESK = (140, 150, 165)  # a light wooden desk, in BGR
PEN = (35, 30, 30)  # a dark pen


def frame_with_bar(
    cx: float, cy: float, long_px: float, short_px: float, angle_deg: float
) -> np.ndarray:
    """A dark rotated bar on a flat desk, with mild noise so the segmenter is
    not handed a perfectly uniform (unrealistically easy) image."""
    img = np.full((FRAME_H, FRAME_W, 3), DESK, dtype=np.uint8)
    rng = np.random.default_rng(7)
    img = np.clip(
        img.astype(np.int16) + rng.integers(-4, 5, img.shape, dtype=np.int16), 0, 255
    ).astype(np.uint8)
    box = cv2.boxPoints(((cx, cy), (long_px, short_px), angle_deg))
    cv2.fillPoly(img, [np.int32(box)], PEN)
    return img


# A 138mm x 9mm pen at 0.5mm/px = 276px x 18px, at 37 deg in the image. Image +y
# is down and robot +y follows it under this homography, so the image angle and
# the robot angle coincide -- which keeps the expected value obvious.
PEN_CX, PEN_CY = 700.0, 380.0
PEN_LONG_PX, PEN_SHORT_PX = 276.0, 18.0
PEN_ANGLE = 37.0
PEN_WIN = 420  # the bar is 276px long; the window must contain it


def pen_frame() -> np.ndarray:
    return frame_with_bar(PEN_CX, PEN_CY, PEN_LONG_PX, PEN_SHORT_PX, PEN_ANGLE)


def fold180(a: float) -> float:
    """Distance to the nearest 180-deg-equivalent angle."""
    return abs((a + 90.0) % 180.0 - 90.0)


# -- measurement ---------------------------------------------------------


def test_measures_extents_in_table_millimetres() -> None:
    obj = measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    assert abs(obj.long_mm - PEN_LONG_PX * MM_PER_PX) < 3.0, obj.long_mm
    assert abs(obj.short_mm - PEN_SHORT_PX * MM_PER_PX) < 2.0, obj.short_mm


def test_measures_axis_yaw_in_robot_frame() -> None:
    obj = measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    assert fold180(obj.axis_yaw_deg - PEN_ANGLE) < 2.0, obj.axis_yaw_deg


def test_projects_the_centroid_through_the_table_plane_map() -> None:
    obj = measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 3.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 3.0


def test_uses_the_flat_map_not_the_cube_top_map() -> None:
    """A cube-top homography must not be consulted for something lying flat: it
    exists to correct 20mm of parallax and would introduce that error here."""
    calib = Calibration(
        homography=CALIB.homography,
        table_z=127.2,
        safe_z=155.0,
        # A deliberately wrong cube-top map: if it were used, x would double.
        cube_top_homography=[[2 * MM_PER_PX, 0, 0], [0, 2 * MM_PER_PX, 0], [0, 0, 1]],
    )
    obj = measure(pen_frame(), PEN_CX, PEN_CY, calib, "pen", win=PEN_WIN)
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 3.0


def test_offcentre_hint_converges_to_the_same_centroid() -> None:
    """The load-bearing property of caller-side grounding: the model only has
    to land on the object, not on its centre."""
    base = measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    rad = math.radians(PEN_ANGLE)
    for along in (-90.0, -40.0, 40.0, 90.0):  # along the shaft, still on it
        hx = PEN_CX + math.cos(rad) * along
        hy = PEN_CY + math.sin(rad) * along
        obj = measure(pen_frame(), hx, hy, CALIB, "pen", win=PEN_WIN)
        assert math.hypot(obj.x - base.x, obj.y - base.y) < 6.0, (along, obj.x, obj.y)
        assert fold180(obj.axis_yaw_deg - base.axis_yaw_deg) < 3.0


def test_refine_mask_is_roughly_the_bar_area() -> None:
    mask, _bounds = refine_at_hint(pen_frame(), PEN_CX, PEN_CY, win=PEN_WIN)
    expected = PEN_LONG_PX * PEN_SHORT_PX
    assert 0.5 * expected < mask.sum() < 2.0 * expected, mask.sum()


def test_illumination_gradient_does_not_flood() -> None:
    """The failure that a single desk colour plus a tolerance cannot survive.

    Across a window the wood's own brightness ramp is larger than the difference
    between a pen and the desk, so a global reference leaks across the gradient
    and the mask swallows the desk (observed on the real desk: a Sharpie's mask
    covering half the table). The background must be estimated locally.
    """
    img = pen_frame().astype(np.int16)
    ramp = np.linspace(-45, 45, FRAME_W, dtype=np.int16)[None, :, None]
    img = np.clip(img + ramp, 0, 255).astype(np.uint8)
    obj = measure(img, PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    expected = PEN_LONG_PX * PEN_SHORT_PX * MM_PER_PX**2
    assert obj.long_mm * obj.short_mm < 3.0 * expected, (obj.long_mm, obj.short_mm)
    assert abs(obj.short_mm - PEN_SHORT_PX * MM_PER_PX) < 3.0, obj.short_mm


def test_soft_shadow_beside_an_object_is_not_part_of_it() -> None:
    """A shadow is desk with less light: it belongs to the background."""
    img = pen_frame()
    shadow = np.zeros((FRAME_H, FRAME_W), np.uint8)
    rad = math.radians(PEN_ANGLE)
    off = 26.0  # just clear of the bar, across its axis
    cv2.line(
        shadow,
        (int(PEN_CX - math.cos(rad) * 130 - math.sin(rad) * -off),
         int(PEN_CY - math.sin(rad) * 130 + math.cos(rad) * -off)),
        (int(PEN_CX + math.cos(rad) * 130 - math.sin(rad) * -off),
         int(PEN_CY + math.sin(rad) * 130 + math.cos(rad) * -off)),
        255, 30,
    )
    shadow = cv2.GaussianBlur(shadow, (61, 61), 0)
    darkened = np.clip(
        img.astype(np.int16) - (shadow[:, :, None] * 0.22).astype(np.int16), 0, 255
    ).astype(np.uint8)
    obj = measure(darkened, PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    assert abs(obj.short_mm - PEN_SHORT_PX * MM_PER_PX) < 3.0, obj.short_mm


def test_width_is_the_typical_cross_section_not_the_widest_point() -> None:
    """A pen's cap must not set the jaw target for the whole shaft."""
    img = pen_frame()
    rad = math.radians(PEN_ANGLE)
    cap_cx = PEN_CX + math.cos(rad) * 110
    cap_cy = PEN_CY + math.sin(rad) * 110
    box = cv2.boxPoints(((cap_cx, cap_cy), (55.0, 44.0), PEN_ANGLE))
    cv2.fillPoly(img, [np.int32(box)], PEN)
    obj = measure(img, PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    assert abs(obj.short_mm - PEN_SHORT_PX * MM_PER_PX) < 3.0, obj.short_mm


def test_marker_paper_is_excluded_from_segmentation() -> None:
    """Tags and their paper are scene furniture; on the real desk every
    remaining fusion was an object merging with the paper beside it."""
    # robot (100, 100) -> pixel (200, 200) under this homography.
    img = frame_with_bar(200.0, 200.0, 90.0, 20.0, 0.0)
    ok = measure(img, 200.0, 200.0, CALIB, "bar", win=260)
    assert ok.long_mm > 20.0
    try:
        measure(img, 200.0, 200.0, CALIB, "bar", win=260, marker_xy=[(100.0, 100.0)])
    except LocateError:
        return
    raise AssertionError("expected a refusal for an object inside a marker paper")


def test_disagreeing_windows_refuse() -> None:
    """The arm acts on whatever comes back, so two scales that disagree across
    the grasp axis must abstain instead of one of them winning.

    Patched by hand rather than via the monkeypatch fixture so this file still
    runs as a plain script.
    """
    from mt4_vision import locate as mod

    real = mod._measure_one
    calls = {"n": 0}

    def shifted(frame, px, py, calib, win, exclude, height):
        got = real(frame, px, py, calib, win, exclude, height)
        calls["n"] += 1
        if got is None or calls["n"] == 1:
            return got
        obj, mask = got
        rad = math.radians(obj.axis_yaw_deg)
        return (
            mod.replace(
                obj,
                x=obj.x - math.sin(rad) * 20.0,
                y=obj.y + math.cos(rad) * 20.0,
            ),
            mask,
        )

    mod._measure_one = shifted
    try:
        measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    except LocateError as exc:
        assert "across" in str(exc)
        return
    finally:
        mod._measure_one = real
    raise AssertionError("expected a refusal when the two windows disagree")


def test_square_object_measures_as_square() -> None:
    """The module is not pen-specific: a cube-sized square reads as ~equal
    axes, which is how yaw_period would be chosen wrongly if extents lied."""
    img = frame_with_bar(600.0, 300.0, 40.0, 40.0, 0.0)
    obj = measure(img, 600.0, 300.0, CALIB, "block", win=200)
    assert abs(obj.long_mm - obj.short_mm) < 3.0
    assert abs(obj.long_mm - 40.0 * MM_PER_PX) < 3.0


def test_compact_object_tolerates_long_edge_swap_between_windows() -> None:
    """Near-square masks flip which edge is 'long' (~90° yaw) across scales;
    that must not refuse -- grip orientation is not critical without a shaft."""
    from mt4_vision import locate as mod

    img = frame_with_bar(600.0, 300.0, 40.0, 40.0, 12.0)
    real = mod._measure_one

    def flipped(frame, px, py, calib, win, exclude, height):
        got = real(frame, px, py, calib, win, exclude, height)
        if got is None or win >= 200:
            return got
        obj, mask = got
        alt = mod.replace(
            obj,
            axis_yaw_deg=obj.axis_yaw_deg + 90.0,
            long_mm=obj.short_mm,
            short_mm=obj.long_mm,
        )
        return alt, mask

    mod._measure_one = flipped
    try:
        obj = measure(img, 600.0, 300.0, CALIB, "block", win=200)
    finally:
        mod._measure_one = real
    assert abs(obj.long_mm - obj.short_mm) < 3.0


def test_measure_box_from_axis_aligned_detector() -> None:
    """DINO-style AABB: long/short follow the box sides in robot mm."""
    from mt4_vision.locate import measure_box

    img = np.full((FRAME_H, FRAME_W, 3), DESK, dtype=np.uint8)
    # 80x20 px box at 0.5mm/px -> 40x10 mm
    obj = measure_box(img, 100, 200, 180, 220, CALIB, "eraser", confidence=0.4)
    assert abs(obj.long_mm - 40.0) < 1.0
    assert abs(obj.short_mm - 10.0) < 1.0
    assert abs(obj.x - 140.0 * MM_PER_PX) < 1.0
    assert abs(obj.y - 210.0 * MM_PER_PX) < 1.0
    assert obj.confidence == 0.4


def test_measure_falls_back_to_box_when_mask_is_unstable() -> None:
    from mt4_vision import locate as mod
    from mt4_vision.locate import measure_with_box_fallback

    real = mod._measure_one
    calls = {"n": 0}

    def shifted(frame, px, py, calib, win, exclude, height):
        got = real(frame, px, py, calib, win, exclude, height)
        calls["n"] += 1
        if got is None or calls["n"] == 1:
            return got
        obj, mask = got
        rad = math.radians(obj.axis_yaw_deg)
        return (
            mod.replace(
                obj,
                x=obj.x - math.sin(rad) * 20.0,
                y=obj.y + math.cos(rad) * 20.0,
            ),
            mask,
        )

    mod._measure_one = shifted
    # Force GrabCut to fail so we exercise desk then final AABB fallback.
    real_gc = mod._segment_grabcut
    mod._segment_grabcut = lambda *a, **k: None
    try:
        # Mask path would refuse; box around the pen still yields a pose.
        half_l, half_s = PEN_LONG_PX / 2.0, PEN_SHORT_PX / 2.0
        obj = measure_with_box_fallback(
            pen_frame(), PEN_CX, PEN_CY, CALIB, "pen",
            box=(
                PEN_CX - half_l, PEN_CY - half_s,
                PEN_CX + half_l, PEN_CY + half_s,
            ),
            win=PEN_WIN,
        )
    finally:
        mod._measure_one = real
        mod._segment_grabcut = real_gc
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 6.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 6.0


def test_grabcut_measures_pen_from_loose_box() -> None:
    """GrabCut turns a DINO-style AABB (of the rotated silhouette) into a mask."""
    from mt4_vision.locate import measure_grabcut, refine_at_box

    pts = cv2.boxPoints(((PEN_CX, PEN_CY), (PEN_LONG_PX, PEN_SHORT_PX), PEN_ANGLE))
    pad = 12.0
    box = (
        float(pts[:, 0].min()) - pad,
        float(pts[:, 1].min()) - pad,
        float(pts[:, 0].max()) + pad,
        float(pts[:, 1].max()) + pad,
    )
    mask, _origin = refine_at_box(pen_frame(), *box)
    assert mask.sum() > 2000
    # Mask should be tighter than the padded box area.
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    assert mask.sum() < 0.7 * box_area

    obj = measure_grabcut(pen_frame(), *box, CALIB, "pen", confidence=0.55)
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 8.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 8.0
    assert obj.short_mm < 25.0
    assert obj.confidence == 0.55


def test_measure_prefers_grabcut_when_box_given() -> None:
    """With a detector box, GrabCut runs first (desk measure is not needed)."""
    from mt4_vision import locate as mod
    from mt4_vision.locate import measure_with_box_fallback

    real = mod.measure

    def boom(*_a, **_k):
        raise LocateError("desk path should not run when GrabCut succeeds")

    pts = cv2.boxPoints(((PEN_CX, PEN_CY), (PEN_LONG_PX, PEN_SHORT_PX), PEN_ANGLE))
    pad = 12.0
    box = (
        float(pts[:, 0].min()) - pad,
        float(pts[:, 1].min()) - pad,
        float(pts[:, 0].max()) + pad,
        float(pts[:, 1].max()) + pad,
    )
    mod.measure = boom
    try:
        obj = measure_with_box_fallback(
            pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", box=box,
        )
    finally:
        mod.measure = real
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 8.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 8.0
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    assert obj.mask_area_px < 0.7 * box_area


def test_measure_falls_back_to_desk_when_grabcut_fails() -> None:
    from mt4_vision import locate as mod
    from mt4_vision.locate import measure_with_box_fallback

    real_gc = mod._segment_grabcut
    mod._segment_grabcut = lambda *a, **k: None
    try:
        pts = cv2.boxPoints(((PEN_CX, PEN_CY), (PEN_LONG_PX, PEN_SHORT_PX), PEN_ANGLE))
        pad = 12.0
        obj = measure_with_box_fallback(
            pen_frame(), PEN_CX, PEN_CY, CALIB, "pen",
            box=(
                float(pts[:, 0].min()) - pad,
                float(pts[:, 1].min()) - pad,
                float(pts[:, 0].max()) + pad,
                float(pts[:, 1].max()) + pad,
            ),
            win=PEN_WIN,
        )
    finally:
        mod._segment_grabcut = real_gc
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 6.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 6.0


# -- refusals ------------------------------------------------------------


def test_hint_on_bare_desk_refuses() -> None:
    """A hint beside a thin object segments the desk; that must not become a
    confident measurement of a 200mm 'object'."""
    img = pen_frame()
    try:
        obj = measure(img, 200.0, 120.0, CALIB, "pen", win=PEN_WIN)
    except LocateError:
        return
    # If something was segmented anyway it must at least not be the pen.
    assert math.hypot(obj.x - PEN_CX * MM_PER_PX, obj.y - PEN_CY * MM_PER_PX) > 20.0


def test_implausibly_large_measurement_refuses() -> None:
    """A bar longer than anything the gripper could handle means the mask has
    flooded the desk, so its centroid is meaningless."""
    img = frame_with_bar(640.0, 360.0, 900.0, 500.0, 0.0)
    try:
        measure(img, 640.0, 360.0, CALIB, "slab", win=600)
    except LocateError as exc:
        assert "plausible" in str(exc) or "segment" in str(exc)
        return
    raise AssertionError("expected a refusal for an implausible measurement")


def test_hint_off_frame_edge_refuses() -> None:
    for px, py in ((0.0, 0.0), (FRAME_W - 1.0, FRAME_H - 1.0)):
        try:
            refine_at_hint(pen_frame(), px, py, win=8)
        except LocateError:
            continue
        raise AssertionError("expected a refusal at the frame edge")


# -- re-acquire ----------------------------------------------------------


def test_relocate_finds_a_shifted_object() -> None:
    obj = measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    moved = frame_with_bar(
        PEN_CX + 30.0, PEN_CY + 20.0, PEN_LONG_PX, PEN_SHORT_PX, PEN_ANGLE
    )
    again = relocate(moved, obj, CALIB, win=PEN_WIN)
    assert again is not None
    assert abs(again.x - (PEN_CX + 30.0) * MM_PER_PX) < 6.0
    assert abs(again.y - (PEN_CY + 20.0) * MM_PER_PX) < 6.0
    assert again.confidence > 0.6


def test_relocate_returns_none_when_the_object_is_gone() -> None:
    """The failure mode must be 're-scan', never a low-score match: the arm
    acts on whatever position comes back."""
    obj = measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    bare = np.full((FRAME_H, FRAME_W, 3), DESK, dtype=np.uint8)
    assert relocate(bare, obj, CALIB, win=PEN_WIN) is None


def test_relocate_respects_its_score_floor() -> None:
    obj = measure(pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", win=PEN_WIN)
    same = pen_frame()
    assert relocate(same, obj, CALIB, win=PEN_WIN, min_score=0.99) is not None
    assert relocate(same, obj, CALIB, win=PEN_WIN, min_score=1.01) is None


# -- feasibility ---------------------------------------------------------


def _obj(x, y, *, axis=37.0, short=9.0, long_mm=138.0):
    from mt4_vision.locate import LocatedObject

    return LocatedObject(
        label="pen", px=700.0, py=380.0, x=x, y=y,
        axis_yaw_deg=axis, long_mm=long_mm, short_mm=short, confidence=0.9,
    )


def _span_calib(**kw) -> Calibration:
    """The real rig plus the measured jaw-span model (jaws open to ~64mm).

    Not module-level CALIB: that one is a deliberately arithmetic 0.5mm/px
    scale for testing the measurement maths, and grasp_feasibility now asks
    the work region whether the camera can see the grasp point -- a question
    only a real camera model can answer.
    """
    kw.setdefault("grip_span_s_at_zero_mm", 285.0)
    kw.setdefault("grip_span_s_per_mm", 2.25)
    return rig_calibration(grip_close_s=240, **kw)


def test_feasible_object_in_the_workspace() -> None:
    ok, reason = grasp_feasibility(_obj(200.0, -60.0), _span_calib())
    assert ok and reason is None


def test_infeasible_inside_keepout() -> None:
    ok, reason = grasp_feasibility(_obj(100.0, 0.0), RIG_CALIB)
    assert not ok and "keep-out" in reason


def test_infeasible_beyond_reach() -> None:
    ok, reason = grasp_feasibility(_obj(400.0, 0.0), RIG_CALIB)
    assert not ok and "max reach" in reason


def test_infeasible_past_the_cameras_coverage() -> None:
    """Placing where the camera cannot see loses the object until it is moved
    by hand, so it is refused even though the arm can physically reach it.

    The gate is the actual frame projection rather than a radius, so the
    refusal is bearing-dependent: straight out along +x the camera's near
    edge cuts in at ~284mm, while the same radius to the side is fine.
    """
    ok, reason = grasp_feasibility(_obj(320.0, 0.0), RIG_CALIB)
    assert not ok and "camera frame" in reason
    ok_side, _ = grasp_feasibility(_obj(0.0, 320.0), RIG_CALIB)
    assert ok_side, "a 240mm radius cap refuses this; the camera sees it fine"


def test_infeasible_wider_than_the_jaws_open() -> None:
    calib = _span_calib()
    # Jaws open to (285-140)/2.25 ~ 64mm.
    ok, _ = grasp_feasibility(_obj(200.0, -60.0, short=40.0), calib)
    assert ok
    ok, reason = grasp_feasibility(_obj(200.0, -60.0, short=80.0), calib)
    assert not ok
    # The wording comes from locate.jaw_span_block_reason, the one place the
    # width test lives now -- entities.object_entity shares it, so the policy
    # loop and the MCP server cannot disagree about what the jaws can hold.
    assert "80mm" in reason and "64mm" in reason


def test_too_open_check_is_skipped_when_uncalibrated() -> None:
    """Without the measured jaw model there is no jaws-open limit to compare
    against, and inventing one would refuse valid grasps. Closing too FAR on a
    wide object only squeezes, which the jaws tolerate."""
    ok, _ = grasp_feasibility(_obj(200.0, -60.0, short=80.0), RIG_CALIB)
    assert ok
    ok, _ = grasp_feasibility(_obj(200.0, -60.0, short=17.0), RIG_CALIB)
    assert ok


def test_narrow_object_allowed_when_the_jaw_model_is_uncalibrated() -> None:
    """Object picks open fully then command a full close; the servo stops on
    resistance, so a missing jaw-span model is not a reason to refuse."""
    ok, reason = grasp_feasibility(_obj(200.0, -60.0, short=9.0), RIG_CALIB)
    assert ok and reason is None
    ok, reason = grasp_feasibility(_obj(200.0, -60.0, short=9.0), _span_calib())
    assert ok and reason is None


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
