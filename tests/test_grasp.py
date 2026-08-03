"""Tests for antipodal grasp planning (no hardware, no camera).

The behaviour under test is the one the stapler exposed on 2026-08-02: the
narrowest part of the *whole outline* is not the width at the point being
gripped, and the centroid is often the one place the jaws cannot close.

Shapes here are built in robot millimetres directly, so the expected answers
are arithmetic rather than empirical.

Run: python tests/test_grasp.py  (or pytest)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.grasp import (
    FINGER_WIDTH_MM,
    MIN_GRASP_MM,
    SPAN_MARGIN_MM,
    plan_grasp,
)

JAWS_MM = 36.0


def _rect(cx, cy, long_mm, short_mm, deg=0.0, step=1.0):
    """Filled rectangle as robot-mm samples, rotated by ``deg``."""
    us = np.arange(-long_mm / 2, long_mm / 2 + step, step)
    vs = np.arange(-short_mm / 2, short_mm / 2 + step, step)
    u, v = np.meshgrid(us, vs)
    th = math.radians(deg)
    x = cx + u * math.cos(th) - v * math.sin(th)
    y = cy + u * math.sin(th) + v * math.cos(th)
    return np.column_stack([x.ravel(), y.ravel()])


def _yaw_close(got, want, period=180.0, tol=8.0):
    d = abs((got - want) % period)
    return min(d, period - d) <= tol


# ------------------------------------------------------------- simple cases


def test_a_cube_is_grasped_at_its_centre():
    plan, why = plan_grasp(_rect(200.0, -60.0, 25.0, 25.0), max_span_mm=JAWS_MM)
    assert plan is not None, why
    assert abs(plan.x - 200.0) < 3.0 and abs(plan.y + 60.0) < 3.0
    assert plan.width_mm <= 26.0
    assert plan.offset_mm < 3.0


def test_a_pen_is_grasped_across_its_short_axis():
    """A 140x9mm pen along the x axis: the jaws must close across y."""
    plan, why = plan_grasp(_rect(200.0, 0.0, 140.0, 9.0), max_span_mm=JAWS_MM)
    assert plan is not None, why
    assert plan.width_mm < 12.0
    # yaw_deg is the axis the jaws close ACROSS, i.e. the pen's long axis.
    assert _yaw_close(plan.yaw_deg, 0.0)
    assert plan.offset_mm < 8.0


def test_a_rotated_pen_is_found_at_its_own_angle():
    plan, why = plan_grasp(
        _rect(200.0, 0.0, 140.0, 9.0, deg=35.0), max_span_mm=JAWS_MM
    )
    assert plan is not None, why
    assert plan.width_mm < 13.0
    assert _yaw_close(plan.yaw_deg, 35.0)


# ------------------------------------------- the shape that motivated this


def _stapler(cx=140.0, cy=-200.0, deg=0.0):
    """A wide base with a narrow rail on top of it, like the real one.

    150mm long. The base is 70mm across -- wider than the jaws. The rail runs
    the same length but is only 18mm across. There is no angle at which the
    whole outline fits between the jaws, and the centroid sits in the base.
    """
    base = _rect(cx, cy, 150.0, 70.0, deg=deg)
    rail = _rect(cx, cy - 0.0, 150.0, 18.0, deg=deg)
    return np.vstack([base, rail])


def test_the_whole_stapler_outline_does_not_fit_the_jaws():
    """Sanity: the short axis of the outline is 70mm, so the old rule refuses."""
    pts = _stapler()
    short = pts[:, 1].max() - pts[:, 1].min()
    assert short > JAWS_MM


def test_a_narrow_handle_is_found_when_the_body_is_too_wide():
    """A saucepan in plan view: a 60mm body with a 14mm handle off one side.

    The handle is the only grippable part and it is nowhere near the centroid.
    This is the property the stapler exposed, reduced to something a top-down
    silhouette can actually express.
    """
    body = _rect(140.0, -215.0, 60.0, 60.0)          # y -245 .. -185
    handle = _rect(140.0, -150.0, 14.0, 70.0)        # y -185 .. -115, 14mm wide
    plan, why = plan_grasp(np.vstack([body, handle]), max_span_mm=JAWS_MM)
    assert plan is not None, why
    assert plan.width_mm <= 18.0, f"gripped {plan.width_mm:.0f}mm"
    # The grip has to be on the handle, not the body.
    assert plan.y > -185.0, f"grasp at y={plan.y:.0f} is in the body"
    assert plan.offset_mm > 5.0, "expected an off-centre grasp"


def test_a_corner_is_never_returned_as_a_grasp():
    """Why CONTACT_FLATNESS_MM exists.

    A diagonal slice near the corner of a 25mm cube is 22mm across -- narrower
    than any face -- so "minimise the width" alone chooses it. Closing there
    squeezes a wedge the jaws push out. The grasp must be two facing surfaces.
    """
    plan, why = plan_grasp(_rect(200.0, -60.0, 25.0, 25.0), max_span_mm=JAWS_MM)
    assert plan is not None, why
    assert _yaw_close(plan.yaw_deg, 0.0, period=90.0), (
        f"yaw {plan.yaw_deg} is a diagonal, not a face"
    )
    assert plan.width_mm > 23.0, f"{plan.width_mm:.1f}mm is a corner, not a face"


def test_a_uniformly_too_wide_object_is_refused_with_its_narrowest_width():
    plan, why = plan_grasp(_rect(200.0, 0.0, 120.0, 70.0), max_span_mm=JAWS_MM)
    assert plan is None
    assert "70mm" in why or "69mm" in why or "71mm" in why
    assert "no angle" in why


# ------------------------------------------------------------ finger width


def test_a_thin_neck_narrower_than_the_finger_is_not_reported_as_the_grip():
    """A dumbbell: two 60mm blobs joined by a 6mm neck 4mm long.

    The neck is narrow, but the fingers are 12mm wide, so they straddle it and
    meet the blobs. The plan must report what the fingers ACTUALLY close on,
    not the narrowest cross-section anywhere.
    """
    # Blobs meet the neck at y = +-2, so a 12mm finger centred on the neck
    # covers 4mm of each blob as well.
    left = _rect(200.0, -32.0, 40.0, 60.0)   # y -62 .. -2
    right = _rect(200.0, 32.0, 40.0, 60.0)   # y   2 ..  62
    neck = _rect(200.0, 0.0, 6.0, 4.0)       # y  -2 ..   2, 6mm across
    plan, why = plan_grasp(
        np.vstack([left, neck, right]), max_span_mm=200.0,
        finger_width_mm=FINGER_WIDTH_MM,
    )
    assert plan is not None, why
    # A 12mm finger centred on a 4mm neck also covers 4mm of each blob, so the
    # honest width is the blob width, not 6mm.
    assert plan.width_mm > 20.0, f"reported {plan.width_mm:.1f}mm across a neck"


# ----------------------------------------------------------------- refusals


def test_mask_noise_is_not_a_grasp():
    speck = _rect(200.0, 0.0, 2.0, 1.0, step=0.25)
    plan, why = plan_grasp(speck, max_span_mm=JAWS_MM)
    assert plan is None
    assert "noise" in why or "too small" in why or "not enough" in why


def test_too_few_points_is_refused():
    plan, why = plan_grasp(np.zeros((2, 2)), max_span_mm=JAWS_MM)
    assert plan is None and "not enough" in why


def test_a_useless_jaw_range_is_refused():
    plan, why = plan_grasp(
        _rect(200.0, 0.0, 25.0, 25.0), max_span_mm=MIN_GRASP_MM + SPAN_MARGIN_MM - 1
    )
    assert plan is None and "approach margin" in why


def test_the_margin_is_respected():
    """An object exactly at the jaw limit must be refused, not planned."""
    plan, _ = plan_grasp(_rect(200.0, 0.0, 60.0, JAWS_MM - 1.0), max_span_mm=JAWS_MM)
    assert plan is None or plan.width_mm <= JAWS_MM - SPAN_MARGIN_MM


# ------------------------------------------------------------- consistency


def test_the_plan_lands_inside_the_object():
    for shape in (
        _rect(200.0, 0.0, 25.0, 25.0),
        _rect(200.0, 0.0, 140.0, 9.0, deg=20.0),
        np.vstack([_rect(140.0, -215.0, 60.0, 60.0), _rect(140.0, -150.0, 14.0, 70.0)]),
    ):
        plan, why = plan_grasp(shape, max_span_mm=JAWS_MM)
        assert plan is not None, why
        d = np.hypot(shape[:, 0] - plan.x, shape[:, 1] - plan.y).min()
        assert d < 4.0, f"grasp point is {d:.1f}mm from any part of the object"


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
