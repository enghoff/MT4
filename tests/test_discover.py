"""Tests for hint-free object discovery (no hardware, no camera).

``discover`` exists so that a pen is in the entity list for the same reason a
cube is -- because it is on the desk -- rather than because somebody pointed at
it first. The properties worth pinning:

* it finds several objects at once, and puts a hint **on** each rather than
  merely near it, since ``locate._segment`` refuses a hint 10px off its blob;
* **it does not require a uniform desk.** The background is estimated locally,
  so an illumination gradient larger than the object-to-desk contrast, plus
  grain, must change nothing. This is the property most likely to be broken by
  a well-meaning "simplification" to a single reference colour, and the reason
  that simplification was abandoned in the first place;
* it is limited by *scale*, not colour: anything wider than the background
  kernel vanishes into its own background estimate. Asserted here so the limit
  is a documented property rather than a surprise in the field;
* off-desk pixels are masked, and un-measurable proposals are reported rather
  than silently dropped.

Run: python tests/test_discover.py  (or pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.calib import Calibration
from mt4_vision.discover import (
    MIN_BLOB_PX,
    Proposal,
    _interior_point,
    desk_pixel_mask,
    discover,
    propose,
)
from mt4_vision.locate import BG_MEDIAN_KERNEL_PX

FRAME_W, FRAME_H = 1280, 720
DESK = (140, 150, 165)  # a light wooden desk, in BGR
DARK = (35, 30, 30)  # anything sitting on it

# 0.5 mm/px, robot origin at pixel (0, 0) -- the same pure-scale map
# tests/test_locate.py uses, so expected millimetres stay arithmetic. The whole
# frame is "desk" unless a test says otherwise.
MM_PER_PX = 0.5
CALIB = Calibration(
    homography=[[MM_PER_PX, 0.0, 0.0], [0.0, MM_PER_PX, 0.0], [0.0, 0.0, 1.0]],
    table_z=127.2,
    safe_z=155.0,
    grip_open_s=140,
    grip_close_s=240,
)

# Well separated so their 280px measurement windows do not overlap, and clear
# of the frame edge so no window gets clipped below the background kernel.
OBJECTS = [(300.0, 250.0), (700.0, 400.0), (1000.0, 220.0)]
OBJ_LONG_PX, OBJ_SHORT_PX = 40.0, 14.0


def desk(gradient: bool = False, grain: bool = False) -> np.ndarray:
    """A desk that is deliberately not a flat colour."""
    img = np.full((FRAME_H, FRAME_W, 3), DESK, dtype=np.uint8)
    rng = np.random.default_rng(7)
    img = img.astype(np.int16)
    img += rng.integers(-4, 5, img.shape, dtype=np.int16)
    if gradient:
        # 90 grey levels corner to corner -- larger than the object's own
        # contrast against the desk, which is what breaks a global threshold.
        gx = np.linspace(-45, 45, FRAME_W, dtype=np.float32)
        gy = np.linspace(-45, 45, FRAME_H, dtype=np.float32)
        img += (gx[None, :, None] + gy[:, None, None]).astype(np.int16) // 2
    if grain:
        # Wood grain: fine, low-amplitude, horizontally streaked.
        streak = rng.integers(-9, 10, (FRAME_H, 1, 1), dtype=np.int16)
        img += np.repeat(np.repeat(streak, FRAME_W, axis=1), 3, axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def with_objects(img: np.ndarray, centres=OBJECTS, angle: float = 20.0) -> np.ndarray:
    out = img.copy()
    for cx, cy in centres:
        box = cv2.boxPoints(((cx, cy), (OBJ_LONG_PX, OBJ_SHORT_PX), angle))
        cv2.fillPoly(out, [np.int32(box)], DARK)
    return out


def found_near(props: list[Proposal], cx: float, cy: float, tol: float = 25.0) -> bool:
    return any(abs(p.px - cx) < tol and abs(p.py - cy) < tol for p in props)


# -- the basic job -------------------------------------------------------


def test_finds_every_object_without_a_hint() -> None:
    props = propose(with_objects(desk()), CALIB)
    for cx, cy in OBJECTS:
        assert found_near(props, cx, cy), (cx, cy, [p.as_dict() for p in props])


def test_proposes_a_point_on_the_object_not_merely_near_it() -> None:
    """``_segment`` gives up when the hint is more than 10px off its component,
    so a hint that lands beside the object is the same as no hint at all."""
    frame = with_objects(desk())
    for p in propose(frame, CALIB):
        b, g, r = frame[int(p.py), int(p.px)]
        assert (int(b), int(g), int(r)) == DARK, (p.as_dict(), (b, g, r))


def test_orders_proposals_largest_first() -> None:
    areas = [p.area_px for p in propose(with_objects(desk()), CALIB)]
    assert areas == sorted(areas, reverse=True), areas


def test_measures_discovered_objects_in_table_millimetres() -> None:
    found, _rej = discover(
        with_objects(desk()), CALIB, require_work_region=False
    )
    assert len(found) == len(OBJECTS), [o.as_dict() for o in found]
    for obj in found:
        assert abs(obj.long_mm - OBJ_LONG_PX * MM_PER_PX) < 4.0, obj.long_mm
        assert abs(obj.short_mm - OBJ_SHORT_PX * MM_PER_PX) < 3.0, obj.short_mm


# -- the desk is not required to be uniform ------------------------------


def test_survives_an_illumination_gradient_larger_than_the_object_contrast() -> None:
    props = propose(with_objects(desk(gradient=True)), CALIB)
    for cx, cy in OBJECTS:
        assert found_near(props, cx, cy), (cx, cy, [p.as_dict() for p in props])


def test_survives_wood_grain() -> None:
    props = propose(with_objects(desk(grain=True)), CALIB)
    for cx, cy in OBJECTS:
        assert found_near(props, cx, cy), (cx, cy, [p.as_dict() for p in props])


def test_a_textured_gradient_desk_invents_nothing() -> None:
    """The false-positive floor. Bare desk, every nuisance turned on, nothing
    on it -- anything proposed here would be proposed on every real frame."""
    props = propose(desk(gradient=True, grain=True), CALIB)
    assert props == [], [p.as_dict() for p in props]


def test_gradient_does_not_move_the_measured_position() -> None:
    """Not just 'still found' -- found in the same place. A background estimate
    that leaked the gradient would bias the centroid down the slope."""
    plain, _ = discover(with_objects(desk()), CALIB, require_work_region=False)
    lit, _ = discover(
        with_objects(desk(gradient=True, grain=True)), CALIB,
        require_work_region=False,
    )
    assert len(plain) == len(lit) == len(OBJECTS)
    for a in plain:
        assert any(
            abs(a.x - b.x) < 2.0 and abs(a.y - b.y) < 2.0 for b in lit
        ), (a.as_dict(), [b.as_dict() for b in lit])


# -- the real limit is scale --------------------------------------------


def oversized_frame() -> np.ndarray:
    big = 3 * BG_MEDIAN_KERNEL_PX
    frame = desk()
    cv2.rectangle(
        frame, (500 - big // 2, 300 - big // 2), (500 + big // 2, 300 + big // 2),
        DARK, -1,
    )
    return frame


def test_offers_no_grasp_point_on_something_too_wide_to_measure() -> None:
    """A book or a phone is wider than the background kernel, so the fine pass
    sees only its four CORNERS -- and each corner cut out on its own measures
    as a small, plausible, stable object. Measured before the coarse veto
    existed: four phantom 12x6mm objects, all four surviving the two-window
    stability check. The arm would have been offered four fake grasp points on
    a book, which is worse than not seeing the book at all."""
    big = 3 * BG_MEDIAN_KERNEL_PX
    props = propose(oversized_frame(), CALIB)
    assert not found_near(props, 500, 300, tol=big), [p.as_dict() for p in props]
    found, _rej = discover(oversized_frame(), CALIB, require_work_region=False)
    assert found == [], [o.as_dict() for o in found]


def test_the_oversize_veto_spares_ordinary_objects() -> None:
    """The veto must not be a blunt instrument: a real object next to an
    oversized one is still a real object."""
    frame = oversized_frame()
    cv2.circle(frame, (1000, 220), 12, DARK, -1)
    props = propose(frame, CALIB)
    assert found_near(props, 1000, 220), [p.as_dict() for p in props]


def test_drops_blobs_under_the_area_floor() -> None:
    frame = desk()
    cv2.circle(frame, (600, 300), 2, DARK, -1)  # ~13px2, well under the floor
    props = propose(frame, CALIB)
    assert all(p.area_px >= MIN_BLOB_PX for p in props), [
        p.as_dict() for p in props
    ]


# -- masking and reporting ----------------------------------------------


def test_desk_polygon_masks_off_table_pixels() -> None:
    """Pixels imaging the wall behind the desk must not propose anything."""
    # Robot y = 0.5 * pixel y, so y_robot < 200 is the top 400 rows.
    calib = Calibration(
        homography=CALIB.homography,
        table_z=127.2,
        safe_z=155.0,
        table_polygon_robot=[[0.0, 200.0], [640.0, 200.0], [640.0, 360.0], [0.0, 360.0]],
    )
    mask = desk_pixel_mask(np.zeros((FRAME_H, FRAME_W, 3), np.uint8), calib)
    assert not mask[100, 640] and mask[600, 640]
    props = propose(with_objects(desk()), calib)
    # (300, 250) and (1000, 220) are above the edge; only (700, 400) survives.
    assert found_near(props, 700, 400)
    assert not found_near(props, 300, 250)
    assert not found_near(props, 1000, 220)


def test_no_polygon_accepts_the_whole_frame() -> None:
    mask = desk_pixel_mask(np.zeros((FRAME_H, FRAME_W, 3), np.uint8), CALIB)
    assert mask.all()


def test_reports_rejections_rather_than_swallowing_them() -> None:
    """A caller tuning this needs to know why an expected thing is missing.

    The keep-out is the realistic case: on the live desk the arm's own base
    proposes a perfectly good-looking blob every frame, and "r=94mm is inside
    the keep-out" is the useful answer rather than silence.
    """
    # Robot xy = 0.5 * pixel, so pixel (140, 140) is r=99mm -- inside the
    # 140mm J1 keep-out, and therefore never a pick target.
    frame = with_objects(desk(), centres=[(140.0, 140.0)])
    found, rejected = discover(frame, CALIB, require_work_region=True)
    assert found == [], [o.as_dict() for o in found]
    assert any("work region" in r for r in rejected), rejected


def test_work_region_gate_can_be_turned_off_for_inspection() -> None:
    frame = with_objects(desk(), centres=[(140.0, 140.0)])
    found, _rej = discover(frame, CALIB, require_work_region=False)
    assert len(found) == 1, [o.as_dict() for o in found]


def test_interior_point_lands_inside_a_non_convex_mask() -> None:
    """A centroid would fall in the notch of an L; the distance peak cannot."""
    mask = np.zeros((80, 80), np.uint8)
    mask[10:70, 10:26] = 1
    mask[54:70, 10:70] = 1
    x, y = _interior_point(mask)
    assert mask[int(y), int(x)] == 1, (x, y)


if __name__ == "__main__":
    import traceback

    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)
