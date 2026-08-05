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
import pytest

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
    # Force the mask path to fail so we exercise desk then final AABB fallback.
    real_seg = mod._segment_sam
    mod._segment_sam = lambda *a, **k: None
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
        mod._segment_sam = real_seg
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 6.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 6.0


def pen_box(pad: float = 12.0) -> tuple[float, float, float, float]:
    """The AABB a detector would draw around the rotated pen, loosened by ``pad``."""
    pts = cv2.boxPoints(((PEN_CX, PEN_CY), (PEN_LONG_PX, PEN_SHORT_PX), PEN_ANGLE))
    return (
        float(pts[:, 0].min()) - pad,
        float(pts[:, 1].min()) - pad,
        float(pts[:, 0].max()) + pad,
        float(pts[:, 1].max()) + pad,
    )


def fake_sam(mask: np.ndarray):
    """Stand in for the SAM service, answering every box with ``mask``.

    The service is a network call, so the box path is exercised here against a
    known silhouette. What is under test is the plumbing either side of it --
    box in, crop bounds and millimetres out -- not the model.
    """
    from mt4_vision.sam import Mask

    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def segment(_frame, **_kwargs):
        return [Mask(object_index=0, score=0.97, area=int(mask.sum()), bbox=bbox, mask=mask)]

    return segment


def pen_mask() -> np.ndarray:
    """The pen's true silhouette, full frame."""
    m = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    pts = cv2.boxPoints(((PEN_CX, PEN_CY), (PEN_LONG_PX, PEN_SHORT_PX), PEN_ANGLE))
    filled = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cv2.fillPoly(filled, [np.int32(pts)], 1)
    m[filled.astype(bool)] = True
    return m


def test_sam_measures_pen_from_loose_box() -> None:
    """A detector AABB becomes a mask, and the mask becomes millimetres."""
    from mt4_vision import locate as mod
    from mt4_vision.locate import measure_sam, refine_at_box

    box = pen_box()
    real = mod.sam_segment
    mod.sam_segment = fake_sam(pen_mask())
    try:
        mask, origin = refine_at_box(pen_frame(), *box)
        assert mask.sum() > 2000
        # The mask is tighter than the padded box it was prompted with.
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        assert mask.sum() < 0.7 * box_area
        # The crop covers the box with its margin, so the stored template has
        # desk context around the object.
        assert origin[0] <= box[0] and origin[1] <= box[1]

        obj = measure_sam(pen_frame(), *box, CALIB, "pen", confidence=0.55)
    finally:
        mod.sam_segment = real
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 8.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 8.0
    assert obj.short_mm < 25.0
    assert obj.confidence == 0.55


def test_measure_prefers_the_mask_when_box_given() -> None:
    """With a detector box, the mask path runs first (desk measure is not needed)."""
    from mt4_vision import locate as mod
    from mt4_vision.locate import measure_with_box_fallback

    real = mod.measure
    real_sam = mod.sam_segment

    def boom(*_a, **_k):
        raise LocateError("desk path should not run when the mask succeeds")

    box = pen_box()
    mod.measure = boom
    mod.sam_segment = fake_sam(pen_mask())
    try:
        obj = measure_with_box_fallback(
            pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", box=box,
        )
    finally:
        mod.measure = real
        mod.sam_segment = real_sam
    assert abs(obj.x - PEN_CX * MM_PER_PX) < 8.0
    assert abs(obj.y - PEN_CY * MM_PER_PX) < 8.0
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    assert obj.mask_area_px < 0.7 * box_area


def test_unreachable_service_refuses_rather_than_measuring_the_box() -> None:
    """A service outage stops the ladder; the weaker rungs would answer wrongly."""
    from mt4_vision import locate as mod
    from mt4_vision.locate import measure_with_box_fallback
    from mt4_vision.sam import SamError

    def down(*_a, **_k):
        raise SamError("sam service unreachable at http://127.0.0.1:8767")

    real = mod.sam_segment
    mod.sam_segment = down
    try:
        with pytest.raises(LocateError, match="unreachable"):
            measure_with_box_fallback(
                pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", box=pen_box(),
            )
    finally:
        mod.sam_segment = real


def test_measure_falls_back_to_desk_when_the_mask_fails() -> None:
    from mt4_vision import locate as mod
    from mt4_vision.locate import measure_with_box_fallback

    real_seg = mod._segment_sam
    mod._segment_sam = lambda *a, **k: None
    try:
        obj = measure_with_box_fallback(
            pen_frame(), PEN_CX, PEN_CY, CALIB, "pen", box=pen_box(), win=PEN_WIN,
        )
    finally:
        mod._segment_sam = real_seg
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


# -- height, and the table-plane projection it buys ----------------------- #
#
# On the real oblique mount an object's top face images displaced outward from
# the point the camera sits above, so the middle of a silhouette is not above
# the object's footprint. These tests build the silhouette a prism of KNOWN
# size and height would cast through the rig's measured camera geometry, and
# ask where the measurement puts its footprint. The truth is exact by
# construction, which is what makes them worth more than the live cube
# comparison: a cube gives one height, this gives any.


def prism_mask(
    calib,
    cx: float,
    cy: float,
    long_mm: float,
    short_mm: float,
    height_mm: float,
    yaw_deg: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """The silhouette a box on the table casts, as (mask, frame origin).

    Footprint and top outline are both projected through
    ``robot_to_pixel(x, y, z)`` -- the measured pinhole parallax -- and the
    silhouette is their convex hull, which is what a camera sees of a convex
    prism.
    """
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    corners = [
        (cx + lx * c - ly * s, cy + lx * s + ly * c)
        for lx, ly in (
            (-long_mm / 2, -short_mm / 2), (long_mm / 2, -short_mm / 2),
            (long_mm / 2, short_mm / 2), (-long_mm / 2, short_mm / 2),
        )
    ]
    pix = [calib.robot_to_pixel(x, y, calib.table_z) for x, y in corners]
    pix += [calib.robot_to_pixel(x, y, calib.table_z + height_mm) for x, y in corners]
    hull = cv2.convexHull(np.array(pix, dtype=np.float32))
    x0 = int(hull[:, 0, 0].min()) - 4
    y0 = int(hull[:, 0, 1].min()) - 4
    mask = np.zeros(
        (int(hull[:, 0, 1].max()) + 5 - y0, int(hull[:, 0, 0].max()) + 5 - x0), np.uint8
    )
    cv2.fillConvexPoly(mask, (hull - [[x0, y0]]).astype(np.int32), 1)
    return mask, (x0, y0)


def footprint_gap(calib, site, long_mm, short_mm, height_mm, yaw, *, assume=None):
    """(distance, across-axis distance) from the true footprint centre, in mm.

    The across-axis component is the one that decides a grasp: the jaws close
    across the long axis, so error along it still lands on the object.
    """
    from mt4_vision.locate import _object_from_mask

    mask, origin = prism_mask(calib, site[0], site[1], long_mm, short_mm, height_mm, yaw)
    frame = np.zeros((origin[1] + mask.shape[0] + 8, origin[0] + mask.shape[1] + 8, 3), np.uint8)
    obj = _object_from_mask(frame, mask, origin, calib, assume)
    assert obj is not None, "synthetic silhouette failed to measure"
    dx, dy = obj.x - site[0], obj.y - site[1]
    nx, ny = -math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    return math.hypot(dx, dy), abs(dx * nx + dy * ny), obj


# Three places spanning the desk. The parallax gain runs 1.2-1.8 across them,
# so a rule that only works near the camera nadir fails here.
CUBE_SITES = ((160.0, -150.0), (250.0, 60.0), (120.0, 200.0))


def test_a_tall_compact_object_projects_onto_its_footprint() -> None:
    """The case the flat projection gets worst, and the one this exists for.

    Two sizes so the answer is not tuned to the 20mm cube, at three places and
    four orientations. Read flat, the aim point sits up to 19.4mm outward
    across the grasp axis for the cube and 28.5mm for the rock -- past what the
    jaws tolerate, in the direction the parallax predicts. Measured live on the
    arm against cube_top_homography: 18.0-22.1mm out flat, 4.0-9.9mm inferred.
    """
    for long_mm, short_mm, height_mm in ((20, 20, 20), (45, 40, 30)):
        worst_flat = worst_inferred = 0.0
        for site in CUBE_SITES:
            for yaw in (0.0, 45.0, 90.0, 135.0):
                _d, flat, _o = footprint_gap(
                    RIG_CALIB, site, long_mm, short_mm, height_mm, yaw, assume=0.0
                )
                _d, across, obj = footprint_gap(
                    RIG_CALIB, site, long_mm, short_mm, height_mm, yaw
                )
                worst_flat = max(worst_flat, flat)
                worst_inferred = max(worst_inferred, across)
                # The height itself, within ~4mm of the truth at every pose --
                # this is what makes the projection land.
                assert abs(obj.height_mm - height_mm) < 4.0, obj.height_mm
        assert worst_flat > 15.0, worst_flat
        assert worst_inferred < 2.0, worst_inferred


def test_a_flat_object_is_left_where_it_lies() -> None:
    """Nothing is invented for a sheet of paper.

    The width cue alone claims 25.6-78.7mm of height for this sheet and drags
    the aim point 8.5-25.1mm off. The radial stretch that would justify such a
    height is not in the silhouette, so the minimum of the two cues holds the
    point within 5.3mm across the grasp axis -- against 3.2mm for the flat
    projection, which is the right answer here. Confirmed live on ArUco paper,
    whose position is known independently: the inferred height comes out 0.0
    and the point moves under 3mm.
    """
    for site in CUBE_SITES:
        for yaw in (0.0, 45.0, 90.0, 135.0):
            _d, across, obj = footprint_gap(RIG_CALIB, site, 90, 60, 0.5, yaw)
            _d, flat, _o = footprint_gap(RIG_CALIB, site, 90, 60, 0.5, yaw, assume=0.0)
            assert obj.height_mm < 16.0, obj.height_mm
            assert across < 6.0, (site, yaw, across)
            assert across - flat < 4.0, (site, yaw, across, flat)


def test_a_long_object_pointing_at_the_camera_keeps_its_grasp_axis() -> None:
    """A 140x12x12mm pen laid along the camera azimuth is the case the radial
    stretch cue cannot read: the footprint's own length looks like stretch, and
    it claims 42-108mm of height. The width cue is tight for exactly this
    shape, so the minimum takes it -- 10.7-11.2mm against a true 12 -- and the
    jaws land within 0.5mm of the shaft.
    """
    from mt4_vision.locate import (
        _height_from_sweep, _parallax_gain, _radial_basis, _table_extents_mm,
    )

    for site in CUBE_SITES:
        u, v = _radial_basis(RIG_CALIB, site[0], site[1])
        # Lay the pen along the outward radial direction, the worst case.
        yaw = math.degrees(math.atan2(u[1], u[0]))
        mask, origin = prism_mask(RIG_CALIB, site[0], site[1], 140, 12, 12, yaw)
        ys, xs = np.nonzero(mask)
        outline = np.stack([xs + origin[0], ys + origin[1]], axis=1).astype(float)
        radial, cross = _table_extents_mm(RIG_CALIB, outline, u, v)
        sweep_only = _height_from_sweep(radial, cross, _parallax_gain(RIG_CALIB, *site))
        assert sweep_only > 40.0, sweep_only

        _d, across, obj = footprint_gap(RIG_CALIB, site, 140, 12, 12, yaw)
        assert obj.height_mm < sweep_only, "the minimum must reject the loose cue"
        assert abs(obj.height_mm - 12.0) < 2.0, obj.height_mm
        assert across < 1.0, (site, across)


def test_a_long_object_lying_across_the_camera_gets_no_correction() -> None:
    """The residual, pinned so it is not mistaken for a regression.

    A pen laid ACROSS the camera azimuth is genuinely ambiguous from one view:
    its silhouette is the same one a flat strip three times as wide would cast,
    and their centres are ~10mm apart. The stretch cue reads no radial
    elongation, takes the low side, and the point stays where the flat
    projection puts it -- 6.1-12.1mm outward across the shaft. Resolving it
    would mean over-correcting every flat object instead, which is the trade
    this direction of error buys.
    """
    from mt4_vision.locate import _radial_basis

    worst = 0.0
    for site in CUBE_SITES:
        u, _v = _radial_basis(RIG_CALIB, site[0], site[1])
        yaw = math.degrees(math.atan2(u[1], u[0])) + 90.0
        _d, across, obj = footprint_gap(RIG_CALIB, site, 140, 12, 12, yaw)
        _d, flat, _o = footprint_gap(RIG_CALIB, site, 140, 12, 12, yaw, assume=0.0)
        assert obj.height_mm == 0.0, obj.height_mm
        assert abs(across - flat) < 0.01, (across, flat)
        worst = max(worst, across)
    assert 6.0 < worst < 13.0, worst


def test_height_from_sweep_reads_the_radial_stretch_only() -> None:
    """The arithmetic, on its own: stretch is the radial extent minus the
    across-radial one, and height is that divided by the parallax gain."""
    from mt4_vision.locate import _height_from_sweep

    assert _height_from_sweep(55.0, 20.0, 1.75) == (55.0 - 20.0) / 1.75
    # No stretch, or less radial extent than across it: claim nothing rather
    # than a negative height.
    assert _height_from_sweep(40.0, 40.0, 1.75) == 0.0
    assert _height_from_sweep(20.0, 140.0, 1.75) == 0.0
    # No measured camera geometry means no parallax to undo.
    assert _height_from_sweep(55.0, 20.0, 0.0) == 0.0


def test_an_explicit_height_overrides_both_cues() -> None:
    """``object_height_mm`` is for a caller that knows the object, and it wins:
    passing 0 restores the plain flat projection."""
    d_zero, _a, zero = footprint_gap(RIG_CALIB, (160.0, -150.0), 20, 20, 20, 0.0, assume=0.0)
    d_true, _a, told = footprint_gap(RIG_CALIB, (160.0, -150.0), 20, 20, 20, 0.0, assume=20.0)
    assert zero.height_mm == 0.0
    assert told.height_mm == 20.0
    assert d_true < d_zero


def test_the_cube_height_routes_through_the_calibrated_map() -> None:
    """At exactly ``cube_height_mm`` the measured cube-top map aims, not the
    analytic contraction.

    The map is fitted against cubes the arm placed at known coordinates, so it
    carries the detector's centroid bias as well as the parallax. Measured
    2026-08-04 against the cube detector's own reading, three objects with a
    snug box: 3.2mm mean through the map, 14.3mm through the analytic form at
    the same height. Any other height has no such map and must not borrow it.
    """
    from mt4_vision.locate import _is_cube_height, _object_from_mask, _unproject

    calib = RIG_CALIB
    assert _is_cube_height(calib, calib.cube_height_mm)
    assert not _is_cube_height(calib, calib.cube_height_mm + 5.0)
    assert not _is_cube_height(rig_calibration(cube_top_homography=None), calib.cube_height_mm)

    mask, origin = prism_mask(calib, 160.0, -150.0, 20, 20, 20, 0.0)
    frame = np.zeros((origin[1] + mask.shape[0] + 8, origin[0] + mask.shape[1] + 8, 3), np.uint8)
    obj = _object_from_mask(frame, mask, origin, calib, calib.cube_height_mm)
    assert obj is not None
    mx, my = calib.pixel_to_robot(obj.px, obj.py, on_cube_top=True)
    assert math.hypot(obj.x - mx, obj.y - my) < 1e-6
    # Not the analytic contraction, which is a different answer here.
    ax, ay = _unproject(calib, obj.px, obj.py, calib.cube_height_mm / 2.0)
    assert math.hypot(obj.x - ax, obj.y - ay) > 1.0


def test_an_assumed_cube_height_does_not_shrink_the_measured_width() -> None:
    """The assumption moves the aim point; it must not de-inflate the width.

    Subtracting a height nobody measured takes more off than the silhouette was
    inflated by -- on a live frame 2026-08-04 it drove a 20mm cube's 20mm
    reading to the 1mm clamp, retiring the only test for an object too wide for
    the jaws. A silhouette reads wide, so leaving it errs toward refusing.
    """
    calib = RIG_CALIB
    mask, origin = prism_mask(calib, 160.0, -150.0, 20, 20, 20, 0.0)
    frame = np.zeros((origin[1] + mask.shape[0] + 8, origin[0] + mask.shape[1] + 8, 3), np.uint8)
    from mt4_vision.locate import _object_from_mask

    assumed = _object_from_mask(frame, mask, origin, calib, calib.cube_height_mm)
    raw = _object_from_mask(frame, mask, origin, calib, 0.0)
    assert abs(assumed.short_mm - raw.short_mm) < 1e-6
    assert assumed.short_mm > 10.0


def test_no_camera_geometry_means_no_height_correction() -> None:
    """Without a measured nadir and lens height there is no parallax model, so
    the measurement claims no height rather than guessing at one."""
    from mt4_vision.locate import _object_from_mask

    flat_cam = rig_calibration(cam_xy_robot=None, cam_height_mm=None)
    mask, origin = prism_mask(RIG_CALIB, 160.0, -150.0, 20, 20, 20, 0.0)
    frame = np.zeros((origin[1] + mask.shape[0] + 8, origin[0] + mask.shape[1] + 8, 3), np.uint8)
    obj = _object_from_mask(frame, mask, origin, flat_cam, None)
    assert obj is not None and obj.height_mm == 0.0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
