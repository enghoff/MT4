"""Where on an object to close the jaws, and at what angle.

The grasp is the object's own centre and its own narrow axis:

* **the point** is the mask's centre of mass, unprojected from
  ``calib.cube_height_mm`` onto the table by ``locate._height_corrected``. That
  is the coordinate the rest of the stack already uses for the object, so the
  plan and the measurement cannot disagree about where the thing is.
* **the angle** is the *major* axis of the mask's principal components. The jaws
  travel across it, so the finger faces lie along the object's longest direction
  and close over its narrowest.

Working in robot millimetres rather than pixels is not a nicety, and neither is
running the PCA there rather than in the image. The camera is steeply oblique
and the table homography rotates and shears, so the major axis of a silhouette
in pixels is not the major axis of the same silhouette on the desk.
:func:`footprint_mm` projects every mask pixel through the homography first, and
the axes come out of that.

What it deliberately does not do
--------------------------------
**No search over grasp positions.** The jaws go to the middle of the mask, which
is the answer an operator can predict from the picture. There is no hunting for
some narrower part of the object to close on instead: a grasp at the tapered end
of a stapler is 39° off its body's cross axis, and it grips the object by one
end and lets it swing (measured 2026-08-05).

**No width gate.** ``width_mm`` is reported and never refused on. The gripper
servo stops on resistance, so closing on something too wide costs nothing, and
the only width available here is a *silhouette* width: on this oblique mount a
silhouette spans an object's top and its near side face together, which reads
far too wide for anything tall. Measured on the live stapler 2026-08-05, 50mm
across a body the 35mm of usable jaw could plausibly have taken. The one refusal
left is the mask-noise floor, which is about whether there is an object at all.

**No force reasoning, no friction cone, no reasoning about what the object
*is*.** Anything about grip retention is unmeasurable on this rig: there is no
sensor in the jaws, so a plan that looks good and slips is indistinguishable
from one that works. Nothing downstream closes that gap either -- the policy
loop has no after-the-fact vision check, because such a check answers wrongly in
both directions (see ``instruct.decide``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Narrower than this is mask noise, not an object. A grasp planned across 1mm
# of speckle closes on nothing and reports success.
MIN_GRASP_MM = 3.0
# Fewer samples than this describe no shape, so there are no axes to find.
MIN_POINTS = 4


@dataclass(frozen=True)
class GraspPlan:
    """Where to put the TCP and how to turn the wrist.

    ``yaw_deg`` is the object's major axis in the robot frame, the convention
    ``LocatedObject.axis_yaw_deg`` and ``motion.Grasp(yaw_period_deg=180)``
    share: the jaws travel across it, so the finger faces end up parallel to it.
    """

    x: float
    y: float
    yaw_deg: float
    # Extent across the minor axis -- what has to fit between the fingers.
    width_mm: float
    # Extent along the major axis -- how much object the fingers span.
    length_mm: float


def plan_grasp(
    points_mm: np.ndarray,
    *,
    x: float,
    y: float,
    min_grasp_mm: float = MIN_GRASP_MM,
) -> tuple[GraspPlan | None, str]:
    """Grasp at ``(x, y)``, closing across the narrow axis of ``points_mm``.

    ``points_mm`` is an (N, 2) array of robot-frame samples of the object's
    footprint -- the mask, not the outline, so interior evidence counts. Only
    its *shape* is read: the axes and both extents come from the principal
    components and are invariant to where the samples sit. That is why ``(x, y)``
    is a separate argument. The caller's own height-corrected centre is the grasp
    point, not the mean of a flat projection, and the two are not the same
    number -- the homography is projective, so the mean of the projected pixels
    is not the projection of their mean.

    Returns ``(plan, "")`` or ``(None, why not)``. It refuses only when there is
    no shape to read: too few samples, or a narrow axis thin enough to be mask
    speckle. Width never refuses -- see the module docstring.
    """
    pts = np.asarray(points_mm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < MIN_POINTS:
        return None, "not enough of the object was measured to plan a grasp"

    local = pts - pts.mean(axis=0)
    # eigh on the 2x2 scatter matrix returns ascending eigenvalues, so the
    # second column is the direction of greatest spread. The extents are the
    # full spans along each, not the eigenvalues: what must fit between two
    # fingers is a distance across the object, not a variance.
    _, axes = np.linalg.eigh(local.T @ local)
    minor_v, major_v = axes[:, 0], axes[:, 1]
    width_mm = float(np.ptp(local @ minor_v))
    length_mm = float(np.ptp(local @ major_v))

    if width_mm < min_grasp_mm:
        return None, (
            f"its narrow axis measures {width_mm:.1f}mm, which is mask noise "
            "rather than something to hold"
        )
    return (
        GraspPlan(
            x=float(x),
            y=float(y),
            yaw_deg=math.degrees(math.atan2(float(major_v[1]), float(major_v[0]))),
            width_mm=width_mm,
            length_mm=length_mm,
        ),
        "",
    )


def footprint_mm(
    mask: np.ndarray,
    origin_px: tuple[int, int],
    calib,
    *,
    max_points: int = 3000,
) -> np.ndarray:
    """Mask pixels as robot-frame millimetres, subsampled for speed.

    Projected through the flat table homography, with no height correction:
    :func:`plan_grasp` reads only the shape of this cloud, and a height
    correction is one translation applied to every point, which moves no axis
    and no extent.
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
    return proj[:, :2] / proj[:, 2:3]
