"""Tests for grasp planning (no hardware, no camera).

The behaviour under test: the jaws go to the object's own centre and close
across the narrow axis of its silhouette, whatever that axis measures. Width is
reported and never refused on.

Shapes here are built in robot millimetres directly, so the expected answers are
arithmetic rather than empirical.

Run: python tests/test_grasp.py  (or pytest)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.grasp import plan_grasp


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


def _plan(shape, *, x=None, y=None):
    """Plan at the shape's own centroid unless a point is given."""
    if x is None:
        x, y = shape.mean(axis=0)
    return plan_grasp(shape, x=float(x), y=float(y))


# ----------------------------------------------------------- the grasp point


def test_the_grasp_point_is_the_point_it_was_given():
    """Not the mean of the samples. The caller's centre is height-corrected and
    the mean of a projected cloud is not the projection of its mean."""
    shape = _rect(200.0, 0.0, 40.0, 25.0)
    plan, why = _plan(shape, x=213.5, y=-58.25)
    assert plan is not None, why
    assert (plan.x, plan.y) == (213.5, -58.25)


# ----------------------------------------------------------------- the angle


def test_a_pen_yaw_is_its_long_axis():
    """A 140x9mm pen along the x axis. ``yaw_deg`` is the axis the jaws close
    ACROSS, so it is the pen's long axis and the jaws travel along y."""
    plan, why = _plan(_rect(200.0, 0.0, 140.0, 9.0))
    assert plan is not None, why
    assert _yaw_close(plan.yaw_deg, 0.0)
    assert abs(plan.width_mm - 9.0) < 1.5
    assert abs(plan.length_mm - 140.0) < 1.5


def test_a_rotated_pen_is_found_at_its_own_angle():
    for deg in (35.0, 90.0, 125.0, 170.0):
        plan, why = _plan(_rect(200.0, 0.0, 140.0, 9.0, deg=deg))
        assert plan is not None, why
        assert _yaw_close(plan.yaw_deg, deg), f"{deg}deg read as {plan.yaw_deg}"


def test_the_axis_does_not_depend_on_where_the_object_sits():
    """The PCA is run on the shape, so a translation of the samples -- which is
    all a height correction is -- moves no axis and no extent."""
    a, why_a = _plan(_rect(200.0, 0.0, 90.0, 20.0, deg=55.0))
    shifted = _rect(200.0, 0.0, 90.0, 20.0, deg=55.0) + np.array([27.0, -13.0])
    b, why_b = _plan(shifted, x=200.0, y=0.0)
    assert a is not None and b is not None, (why_a, why_b)
    assert abs(a.yaw_deg - b.yaw_deg) < 1e-6
    assert abs(a.width_mm - b.width_mm) < 1e-6
    assert abs(a.length_mm - b.length_mm) < 1e-6


def test_a_square_is_planned_at_its_centre_whatever_the_axis():
    """A square has no meaningful major axis, and it does not need one: every
    angle is the same grasp, so only the point and the width have to be right."""
    plan, why = _plan(_rect(200.0, -60.0, 25.0, 25.0))
    assert plan is not None, why
    assert abs(plan.x - 200.0) < 1e-9 and abs(plan.y + 60.0) < 1e-9
    assert abs(plan.width_mm - 25.0) < 1.5


# ----------------------------------------------------------------- refusals


def test_a_wide_object_is_planned_not_refused():
    """No width gate. A 70mm narrow axis is reported and the plan stands: the
    servo stops on resistance, and a silhouette width reads wide for anything
    tall (the live stapler measured 50mm across on 2026-08-05)."""
    plan, why = _plan(_rect(200.0, 0.0, 120.0, 70.0))
    assert plan is not None, why
    assert abs(plan.width_mm - 70.0) < 1.5
    assert _yaw_close(plan.yaw_deg, 0.0)


def test_a_stapler_is_gripped_at_its_middle_across_its_body():
    """A wide base with a narrow rail along it, like the real one.

    The grasp is the centre and the body's cross axis, not the tapered end. A
    search for the narrowest band instead settles on the handle tip, whose local
    direction is 39 deg off the body's cross axis -- measured 2026-08-05, that
    grips the object by one end and lets it swing.
    """
    base = _rect(140.0, -200.0, 150.0, 70.0)
    rail = _rect(140.0, -200.0, 150.0, 18.0)
    shape = np.vstack([base, rail])
    plan, why = _plan(shape)
    assert plan is not None, why
    assert abs(plan.x - 140.0) < 1.0 and abs(plan.y + 200.0) < 1.0
    assert _yaw_close(plan.yaw_deg, 0.0), "the jaws must close across the body"


def test_a_saucepan_handle_is_not_hunted_for():
    """A 60mm body with a 14mm handle off one side. The grasp is the centre of
    the whole mask -- this deliberately does not go looking for the handle."""
    body = _rect(140.0, -215.0, 60.0, 60.0)
    handle = _rect(140.0, -150.0, 14.0, 70.0)
    shape = np.vstack([body, handle])
    plan, why = _plan(shape)
    assert plan is not None, why
    cx, cy = shape.mean(axis=0)
    assert abs(plan.x - cx) < 1e-9 and abs(plan.y - cy) < 1e-9


def test_mask_noise_is_not_a_grasp():
    plan, why = _plan(_rect(200.0, 0.0, 2.0, 1.0, step=0.25))
    assert plan is None
    assert "noise" in why or "not enough" in why


def test_too_few_points_is_refused():
    plan, why = plan_grasp(np.zeros((2, 2)), x=0.0, y=0.0)
    assert plan is None and "not enough" in why


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
