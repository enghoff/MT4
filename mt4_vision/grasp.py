"""Where on an object to close the jaws, and at what angle.

Until now every non-cube grasp was the silhouette centroid, closing across the
object's short axis. For a cube those are the same thing and the answer is
right. For anything else the centroid is often the one place the jaws cannot
close, and the short axis of the *whole* outline says nothing about the width
at the point being gripped.

Measured on this rig 2026-08-02, the stapler makes the point twice over. One
run segmented only its metal rail, aimed there, and the pick worked -- across
16mm. The next run segmented the whole stapler, aimed at the centroid, and the
jaws closed on air across 73mm. Nothing about the arm changed. The difference
was which part of the object the grasp happened to land on, and nothing chose
it deliberately.

What this module does
---------------------
An antipodal-grasp search over the object's footprint, in robot millimetres:

* rotate the footprint through every candidate jaw angle;
* slide a **finger-width window** along the jaw's own axis;
* in each window, the jaws must span the object's full extent across the
  closing direction -- that extent, not the object's short axis, is what has to
  fit between the fingers;
* keep the narrowest window that fits, preferring one near the centre of mass
  so the object does not pivot out of the jaws as they close.

Working in millimetres rather than pixels is not a nicety. The camera is
steeply oblique, so a fixed pixel window is a different physical size at each
end of the desk, and the whole question here is physical: does this fit between
two fingers that open a fixed number of millimetres.

What it deliberately does not do
--------------------------------
No force reasoning, no friction cone, no reasoning about what the object *is*.
It answers one geometric question -- can the fingers close here, and is there
room for them -- and it answers it the same way for a cube, a pen and a
stapler. Anything about grip retention is unmeasurable on this rig: there is no
sensor in the jaws, so a plan that looks good and slips is indistinguishable
from one that works. Nothing downstream closes that gap either -- the policy
loop's after-the-fact vision check was removed 2026-08-03 for answering wrongly
in both directions (see ``instruct.grasp_for``), so this geometric test, run
BEFORE the move, is the last thing that can refuse a bad grasp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# How much of the object the finger face covers along the jaw axis. The plan
# must fit the object between the jaws across this whole band, not just on one
# infinitely thin line -- a pen resting against a mug has a perfect zero-width
# grasp line 1mm from a place the fingers cannot go.
FINGER_WIDTH_MM = 12.0
# Narrower than this is mask noise, not an object. A grasp planned across 1mm
# of speckle closes on nothing and reports success.
MIN_GRASP_MM = 3.0
# Keep this much of the jaw range in hand. The measured width is a silhouette
# and this camera reads wide for tall objects, but the error is not always in
# that direction, and a plan at the exact limit has no room for the approach
# being a couple of millimetres off.
SPAN_MARGIN_MM = 3.0
# Jaw angles tried. 5 degrees is finer than the wrist's own repeatability and
# costs nothing -- the search is a few thousand numpy operations.
ANGLE_STEP_DEG = 5
# Sliding-window step along the jaw axis.
STEP_MM = 2.0
# Weighs "narrow" against "near the centre of mass". At 0.6 a grasp 5mm further
# out has to be 3mm narrower to win. Off-centre grasps are not wrong -- a
# saucepan is picked up by its handle -- they just let the object rotate as the
# jaws close, so they have to earn it.
#
# This was 0.25 and that was too cheap. Measured on the live stone: the
# silhouette included a spill of shadow, and a 14mm-wide sliver of shadow 22mm
# off centre beat the stone's own 28mm body, so the arm reached for bare desk.
CENTRE_PENALTY = 0.6
# The finger band must hold at least this share of the material in the fullest
# band at the same angle. A wisp of shadow, an antenna, a cable tail: all are
# narrow, all are graspable on paper, and none of them is the object. Density
# is what separates "a thin part of this thing" from "a thin thing next to it"
# without needing to know what either is.
MIN_BAND_SHARE = 0.15
# How much the two contact surfaces may wander across the finger's width
# before this stops being an antipodal grasp.
#
# Without this the search returns **corners**. A diagonal slice near the corner
# of a 25mm cube is only 22mm across, narrower than any face, so "minimise the
# width" picks it every time -- and closing there squeezes a wedge that the
# jaws push out rather than hold. Measured on the synthetic cube: yaw 140
# degrees, width 22mm, 12mm from the centre, on an object whose correct grasp
# is 25mm dead centre.
#
# Requiring both contact lines to be flat across the finger is what makes the
# grasp antipodal rather than merely narrow: two facing surfaces, not the
# converging sides of a point.
CONTACT_FLATNESS_MM = 4.0


@dataclass(frozen=True)
class GraspPlan:
    """Where to put the TCP and how to turn the wrist.

    ``yaw_deg`` follows the same convention as ``LocatedObject.axis_yaw_deg``
    and ``motion.Grasp(yaw_period_deg=180)``: it is the axis the jaws close
    *across*, so the finger faces end up parallel to it.
    """

    x: float
    y: float
    yaw_deg: float
    width_mm: float
    # Object extent along the jaw axis at the grasp -- how much of the object
    # the fingers actually have hold of.
    length_mm: float
    # Distance from the object's centre of mass, in the plane.
    offset_mm: float


def plan_grasp(
    points_mm: np.ndarray,
    *,
    max_span_mm: float,
    finger_width_mm: float = FINGER_WIDTH_MM,
    min_grasp_mm: float = MIN_GRASP_MM,
    margin_mm: float = SPAN_MARGIN_MM,
    angle_step_deg: int = ANGLE_STEP_DEG,
    step_mm: float = STEP_MM,
    flatness_mm: float = CONTACT_FLATNESS_MM,
    band_share: float = MIN_BAND_SHARE,
) -> tuple[GraspPlan | None, str]:
    """Best (point, angle) for closing the jaws on this footprint.

    ``points_mm`` is an (N, 2) array of robot-frame samples of the object's
    footprint -- the mask, not the outline, so interior evidence counts.

    Returns ``(plan, "")`` or ``(None, why not)``. The refusal names the
    narrowest grasp that *was* found, because "no grasp under 36mm; the
    narrowest is 71mm" is a fact about the object, while "cannot be picked up"
    is a fact about nothing.
    """
    pts = np.asarray(points_mm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        return None, "not enough of the object was measured to plan a grasp"

    usable = max_span_mm - margin_mm
    if usable < min_grasp_mm:
        return None, (
            f"the jaws open to {max_span_mm:.0f}mm, which leaves nothing once "
            f"the {margin_mm:.0f}mm approach margin is taken out"
        )

    centre = pts.mean(axis=0)
    local = pts - centre

    best: tuple[float, GraspPlan] | None = None
    narrowest = math.inf

    for deg in range(0, 180, angle_step_deg):
        th = math.radians(deg)
        # u runs along the jaw axis (the faces are parallel to it), v is the
        # direction the jaws travel.
        cos_t, sin_t = math.cos(th), math.sin(th)
        u = local[:, 0] * cos_t + local[:, 1] * sin_t
        v = -local[:, 0] * sin_t + local[:, 1] * cos_t

        lo, hi = float(u.min()), float(u.max())
        if hi - lo < 1e-6:
            continue
        half = finger_width_mm / 2.0
        # Window centres: every step along the axis, plus the two ends so a
        # short object still gets one window.
        starts = np.arange(lo + half, hi - half + step_mm, step_mm)
        if len(starts) == 0:
            starts = np.array([(lo + hi) / 2.0])

        counts = np.array([
            int(((u >= c - half) & (u <= c + half)).sum()) for c in starts
        ])
        floor = max(6, int(counts.max() * band_share))

        for c, n in zip(starts, counts):
            if n < floor:
                continue
            band = (u >= c - half) & (u <= c + half)
            vb, ub = v[band], u[band]
            # Both contact lines must be flat across the finger, or this is a
            # corner rather than two facing surfaces. See CONTACT_FLATNESS_MM.
            order = np.argsort(ub)
            thirds = [t for t in np.array_split(order, 3) if len(t)]
            if len(thirds) < 3:
                continue
            lows = [float(vb[t].min()) for t in thirds]
            highs = [float(vb[t].max()) for t in thirds]
            if (max(lows) - min(lows) > flatness_mm) or (
                max(highs) - min(highs) > flatness_mm
            ):
                continue
            width = float(vb.max() - vb.min())
            narrowest = min(narrowest, width)
            if width < min_grasp_mm or width > usable:
                continue
            # Centre the fingers on the material they are closing on, not on
            # the window's nominal axis position.
            v_mid = float((vb.max() + vb.min()) / 2.0)
            length = float(ub.max() - ub.min())
            gx = centre[0] + c * cos_t - v_mid * sin_t
            gy = centre[1] + c * sin_t + v_mid * cos_t
            offset = math.hypot(gx - centre[0], gy - centre[1])
            score = width + CENTRE_PENALTY * offset
            if best is None or score < best[0]:
                best = (
                    score,
                    GraspPlan(
                        x=gx, y=gy, yaw_deg=float(deg),
                        width_mm=width, length_mm=length, offset_mm=offset,
                    ),
                )

    if best is None:
        if narrowest is math.inf:
            return None, "the footprint is too small to plan a grasp across"
        if narrowest > usable:
            return None, (
                f"the narrowest grip anywhere on it is {narrowest:.0f}mm and "
                f"the jaws give {usable:.0f}mm of usable opening -- no angle "
                "or position on this object fits between them"
            )
        return None, (
            f"every candidate grip is under {min_grasp_mm:.0f}mm wide, which "
            "is mask noise rather than something to hold"
        )
    return best[1], ""


def footprint_mm(
    mask: np.ndarray,
    origin_px: tuple[int, int],
    calib,
    *,
    max_points: int = 3000,
    offset_mm: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Mask pixels as robot-frame millimetres, subsampled for speed.

    ``offset_mm`` is added to every point. Callers pass the difference between
    the object's height-corrected centre and its raw table-plane centre, so the
    plan lands where the rest of the stack believes the object is rather than
    where its shadow projects. Applying one offset to the whole footprint is an
    approximation -- the parallax correction varies across the object -- but it
    is the same approximation the existing measurement already makes when it
    corrects a single centroid.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2), float)
    if len(xs) > max_points:
        take = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[take], ys[take]
    ox, oy = origin_px
    # One batched projection, not 3000 scalar ones. calib.pixel_to_robot builds
    # a 3x3 from a nested list on every call, and object_entity re-plans a grasp
    # for every registered object in every snapshot -- measured 6.4ms for 2821
    # points against ~0.1ms for this.
    h = np.asarray(calib.homography, dtype=np.float64)
    pts = np.column_stack(
        [xs + ox, ys + oy, np.ones(len(xs))]
    ).astype(np.float64)
    proj = pts @ h.T
    return proj[:, :2] / proj[:, 2:3] + np.asarray(offset_mm, dtype=np.float64)
