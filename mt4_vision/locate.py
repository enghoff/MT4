"""Pixel hint -> measured object: the grounding half of "pick up the pen".

``detect.py`` sees four HSV colours in cube-shaped blobs (area 800-6000px2,
aspect <= 2.0). A pen is none of those, so no amount of language ability makes
it referable -- the vocabulary, not the parser, is the wall.

The split here is deliberate: **the caller supplies semantics, this module
supplies geometry.** A VLM is asked only "which pixel", never "how far, at what
angle, how wide" -- every number that reaches the arm comes from OpenCV plus the
existing calibration. That matters because pointing accuracy and metric accuracy
fail in completely different ways: a hint 20px off still lands *on* a pen and
the mask recovers its true centroid, whereas a guessed millimetre offset is
simply wrong and nothing downstream can tell.

Two consequences worth knowing:

* Yaw is the object's **long axis**, and an elongated grasp is 180°-periodic:
  the jaws must close across the shaft, not along it. See
  ``mt4_vision.wrist.j4_for_long_axis``.
* An object's own **height** cannot be ignored, even for "thin" things. The base
  projection is the flat table-plane homography, exact for a point *on* the
  table -- but a silhouette's centroid is not on the table, and this camera is
  steeply oblique (nadir ~360-490mm away, 244mm up), so the flat inverse is off
  by ~1.5x the height. A 12mm pen reads ~9mm from where it is, past the ~10mm
  the jaws tolerate: measured on hardware, the arm shoved the pen aside instead
  of gripping it, in exactly the direction this predicts. So the centroid is
  unprojected at half the object's height and the width is de-inflated by the
  whole of it. One view cannot separate height from width, so the height is
  *assumed* equal to the width unless ``object_height_mm`` says otherwise --
  right for pens, markers and tools, wrong for a flat sheet. It is reported on
  ``LocatedObject.height_mm`` because it is an assumption, not a measurement.

Extents also clip where an object's own contrast against the desk runs out: the
white end of a white pen on light wood is not separable, so ``long_mm`` reads
short and the centroid slides along the axis. Along a long axis that is harmless
for grasping -- it still lands on the shaft -- which is why the axis and the width
are the numbers that get checked for stability, not the length.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from mt4_vision.calib import Calibration
from mt4_vision.wrist import j4_for_long_axis
from mt4_vision.workspace import (
    MAX_REACH_MM,
    MAX_VERIFIABLE_RADIUS_MM,
    is_mp_reachable_xy,
)

# Crop side (px) segmented around a hint. At this mount (~0.5mm/px) 280px spans
# ~150mm, enough for a pen end to end.
DEFAULT_WINDOW_PX = 280
# Side (px) of the median filter that estimates the desk under the object. Must
# exceed the widest object in pixels, or the object survives its own background
# estimate and vanishes; must stay well under the window, or the illumination
# gradient survives instead.
BG_MEDIAN_KERNEL_PX = 61
# Weight on Lab's L axis when measuring how far a pixel is from the local desk.
# Below 1.0 on purpose: a shadow is desk with less light, so it moves L a lot and
# a/b little, and full weight makes every shadow an object.
L_WEIGHT = 0.35
# Deviation floor under which nothing is an object. Otsu always finds a split,
# including "desk vs slightly darker desk", so bare desk needs an absolute bar.
MIN_DEVIATION = 10.0
# Half-side (mm) of the region around each marker treated as scene furniture.
# The tags sit on ~67mm paper held by white tape; all of it is background. This
# is why a pen beside a tag no longer fuses with it -- but it also means an
# object resting ON a marker cannot be measured. Move it off the tag.
PAPER_HALF_MM = 42.0
# A measurement is only returned if two window sizes agree within these bounds.
# Segmentation is the least trustworthy step in the chain, and the arm acts on
# whatever comes back, so disagreement must abstain rather than pick a winner.
STABILITY_WINDOW_RATIO = 0.72
STABILITY_CENTROID_MM = 6.0
STABILITY_SHORT_MM = 5.0
STABILITY_YAW_DEG = 12.0
# long/short at or below this: no obvious shaft. minAreaRect freely swaps which
# edge is "long" (≈90° yaw flips) between window sizes, and the grasp does not
# care -- jaws can meet any face. Above this, yaw disagreement is load-bearing.
COMPACT_ASPECT = 1.35
# Normalized-cross-correlation floor for re-acquiring a registered object.
# Below this, report nothing rather than a low-confidence guess -- a wrong
# position is worse than "re-scan", because the arm will act on it.
RELOCATE_MIN_SCORE = 0.6
RELOCATE_SEARCH_PX = 80
# Refuse implausible measurements rather than picking at them: a mask that has
# flooded the desk yields a huge "object" whose centroid is meaningless.
MAX_PLAUSIBLE_LONG_MM = 200.0
MIN_PLAUSIBLE_LONG_MM = 4.0


@dataclass(frozen=True)
class LocatedObject:
    """A non-cube object measured from one frame."""

    label: str
    px: float  # refined centroid, pixels
    py: float
    x: float  # robot frame, mm, on the table plane
    y: float
    axis_yaw_deg: float  # robot-frame long axis
    long_mm: float
    short_mm: float
    confidence: float
    # Height above the table assumed when unprojecting the centroid. Reported
    # because it is an assumption, not a measurement, and it moves the grasp
    # point by ~1.5x itself at this mount -- see _assumed_height_mm.
    height_mm: float = 0.0
    # BGR crop around the object, for re-acquiring it in a later frame.
    template: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1, 3), np.uint8))
    mask_area_px: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "pixel": [round(self.px, 1), round(self.py, 1)],
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "axis_yaw_deg": round(self.axis_yaw_deg, 1),
            "long_mm": round(self.long_mm, 1),
            "short_mm": round(self.short_mm, 1),
            "confidence": round(self.confidence, 2),
            "assumed_height_mm": round(self.height_mm, 1),
        }


class LocateError(Exception):
    """Raised when a hint cannot be turned into a measurement."""


def is_compact(long_mm: float, short_mm: float) -> bool:
    """True when there is no obvious long axis (cube-like, not pen-like)."""
    return float(long_mm) <= COMPACT_ASPECT * max(float(short_mm), 1e-6)


def _window(frame: np.ndarray, px: float, py: float, win: int) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    half = win // 2
    x0 = max(0, min(int(round(px)) - half, w - 1))
    y0 = max(0, min(int(round(py)) - half, h - 1))
    x1 = min(w, x0 + win)
    y1 = min(h, y0 + win)
    return x0, y0, x1, y1


def marker_paper_mask(
    frame: np.ndarray, calib: Calibration, marker_xy: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Boolean mask of the marker papers -- scene furniture, never objects.

    Measured on the real desk: every segmentation failure that survived a decent
    background model was an object fusing with the white paper or black glyph it
    sat next to, not with the desk. The tags are the one piece of clutter whose
    position is already known, so excluding them is free.

    Takes marker positions rather than a ``Scene`` so this module stays
    hardware-free and testable, and so undecoded ("unknown") markers -- whose
    paper is just as much background -- are covered too.
    """
    h, w = frame.shape[:2]
    out = np.zeros((h, w), np.uint8)
    d = PAPER_HALF_MM
    for mx, my in marker_xy:
        quad = np.array(
            [
                calib.robot_to_pixel(mx + sx * d, my + sy * d)
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(out, quad, 1)
    return out.astype(bool)


def _segment(
    frame: np.ndarray,
    px: float,
    py: float,
    win: int,
    exclude: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Foreground component under the hint, by deviation from the local desk.

    The desk is estimated *locally*, with a large median filter, rather than as
    one colour with a tolerance. That matters more than any threshold choice:
    across a 280px window the wood's own illumination gradient is larger than
    the difference between a pen and the desk, so a single reference colour
    leaks across the gradient and floods (measured: a Sharpie's mask swallowing
    half the desk). A local estimate absorbs the gradient and soft shadows into
    the background, leaving only what is small and high-frequency -- an object.

    The split then comes from Otsu on the deviation map, so there is no tuned
    colour tolerance to go stale when the lighting changes.
    """
    x0, y0, x1, y1 = _window(frame, px, py, win)
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    if crop.size == 0 or min(crop.shape[:2]) < BG_MEDIAN_KERNEL_PX:
        return None
    hx = max(0, min(int(round(px)) - x0, crop.shape[1] - 1))
    hy = max(0, min(int(round(py)) - y0, crop.shape[0] - 1))

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    desk = cv2.medianBlur(lab, BG_MEDIAN_KERNEL_PX)
    scale = np.array([L_WEIGHT, 1.0, 1.0], np.float32)
    dev = np.linalg.norm(
        (lab.astype(np.float32) - desk.astype(np.float32)) * scale, axis=2
    )

    ex = (
        exclude[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]]
        if exclude is not None
        else np.zeros(crop.shape[:2], bool)
    )
    if ex.all():
        return None
    d8 = np.clip(dev, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8[~ex], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = ((dev > max(MIN_DEVIATION, float(otsu))) & ~ex).astype(np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(fg)
    if n <= 1:
        return None
    want = int(labels[hy, hx])
    if want == 0:
        # Hint landed just off the object (or in a hole); accept the nearest
        # component within a few pixels, but never a distant one -- that would
        # silently measure something the caller did not point at.
        ys, xs = np.nonzero(labels)
        if not len(ys):
            return None
        d2 = (ys - hy) ** 2 + (xs - hx) ** 2
        i = int(d2.argmin())
        if d2[i] > 100:
            return None
        want = int(labels[ys[i], xs[i]])
    mask = (labels == want).astype(np.uint8)
    return (mask, (x0, y0, x1, y1)) if mask.sum() else None


def refine_at_hint(
    frame: np.ndarray,
    px: float,
    py: float,
    *,
    win: int = DEFAULT_WINDOW_PX,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Segment the object under (px, py); return its mask and the crop bounds.

    This is what makes moderate pointing accuracy sufficient: the caller only
    has to land on the object, and the mask supplies the centroid.
    """
    got = _segment(frame, px, py, win, exclude)
    if got is None:
        raise LocateError(
            f"could not segment an object at ({px:.0f}, {py:.0f}) -- the hint "
            "may be on the desk, on a marker paper, or on something larger than "
            "the segmentation window"
        )
    return got


# GrabCut: detector AABB is probable FG; a small centre disc is sure FG so the
# GMM has a colour sample. Marking the box border as sure BG looks tempting for
# loose boxes but eats thin rotated objects whose tips sit in that border.
GRABCUT_ITERS = 5
GRABCUT_MARGIN_PX = 32  # desk context around the box for the GMM
GRABCUT_SURE_FG_RADIUS_PX = 10
GRABCUT_MIN_AREA_FRAC = 0.02  # refuse if FG is tiny vs the box


def _clip_box(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> tuple[int, int, int, int] | None:
    h, w = frame.shape[:2]
    xa, xb = sorted((int(math.floor(x1)), int(math.ceil(x2))))
    ya, yb = sorted((int(math.floor(y1)), int(math.ceil(y2))))
    xa = max(0, min(xa, w - 1))
    xb = max(0, min(xb, w))
    ya = max(0, min(ya, h - 1))
    yb = max(0, min(yb, h))
    if xb - xa < 4 or yb - ya < 4:
        return None
    return xa, ya, xb, yb


def _segment_grabcut(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    exclude: np.ndarray | None = None,
    *,
    iters: int = GRABCUT_ITERS,
    margin_px: int = GRABCUT_MARGIN_PX,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Foreground mask from a detector AABB via GrabCut.

    The box is a strong prior from Grounding DINO; GrabCut turns it into a
    silhouette that desk-deviation segmentation often cannot find (low contrast,
    textured objects, soft shadows). Marker paper is forced to sure background.
    """
    clipped = _clip_box(frame, x1, y1, x2, y2)
    if clipped is None:
        return None
    bx0, by0, bx1, by1 = clipped
    h, w = frame.shape[:2]
    # Crop with margin so GrabCut sees real desk outside the box.
    cx0 = max(0, bx0 - margin_px)
    cy0 = max(0, by0 - margin_px)
    cx1 = min(w, bx1 + margin_px)
    cy1 = min(h, by1 + margin_px)
    crop = np.ascontiguousarray(frame[cy0:cy1, cx0:cx1])
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None

    lx0, ly0 = bx0 - cx0, by0 - cy0
    lx1, ly1 = bx1 - cx0, by1 - cy0
    bw, bh = lx1 - lx0, ly1 - ly0

    gc = np.full(crop.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    gc[ly0:ly1, lx0:lx1] = cv2.GC_PR_FGD
    # Seed sure FG at the box centre so GrabCut has an object colour sample
    # even when most of a loose AABB is desk.
    hx = max(0, min((lx0 + lx1) // 2, crop.shape[1] - 1))
    hy = max(0, min((ly0 + ly1) // 2, crop.shape[0] - 1))
    r = max(3, min(GRABCUT_SURE_FG_RADIUS_PX, bw // 4, bh // 4))
    cv2.circle(gc, (hx, hy), r, int(cv2.GC_FGD), thickness=-1)

    if exclude is not None:
        ex = exclude[cy0:cy1, cx0:cx1]
        if ex.shape[:2] == gc.shape:
            gc[ex.astype(bool)] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Keep only the component under the box centre (or nearest inside the box).
    n, labels = cv2.connectedComponents(fg)
    if n <= 1:
        return None
    want = int(labels[hy, hx])
    if want == 0:
        ys, xs = np.nonzero(labels)
        if not len(ys):
            return None
        # Prefer components that overlap the detector box.
        in_box = (
            (xs >= lx0) & (xs < lx1) & (ys >= ly0) & (ys < ly1) & (labels[ys, xs] > 0)
        )
        if in_box.any():
            ys, xs = ys[in_box], xs[in_box]
        d2 = (ys - hy) ** 2 + (xs - hx) ** 2
        i = int(d2.argmin())
        want = int(labels[ys[i], xs[i]])
    mask = (labels == want).astype(np.uint8)
    area = int(mask.sum())
    box_area = max(1, bw * bh)
    if area < max(16, int(box_area * GRABCUT_MIN_AREA_FRAC)):
        return None
    return mask, (cx0, cy0, cx1, cy1)


def refine_at_box(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """GrabCut mask for a detector AABB (preview / measure path)."""
    got = _segment_grabcut(frame, x1, y1, x2, y2, exclude)
    if got is None:
        raise LocateError(
            f"GrabCut found no foreground in box "
            f"({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
        )
    return got


def _rect_from_mask(
    mask: np.ndarray, origin: tuple[int, int]
) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """(centroid, 4 corners) in FRAME pixels from a crop-local mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise LocateError("mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
    ox, oy = origin
    corners = [(float(cx) + ox, float(cy) + oy) for cx, cy in box]
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()) + ox, float(ys.mean()) + oy)
    return centroid, corners


def _parallax_gain(calib: Calibration, x: float, y: float) -> float:
    """Millimetres of table-plane error per millimetre of height at (x, y).

    The flat homography puts a point at height h where its camera ray *crosses
    the table*, which is radially outward from the nadir. On this steeply oblique
    mount the nadir sits ~360-490mm away, so the gain is 1.4-2.0: a 12mm-tall pen
    is reported ~9mm from where it is, past the ~10mm the jaws tolerate. Measured
    on hardware -- the arm shoved the pen instead of gripping it, in exactly the
    direction this predicts.
    """
    cam, height = calib.cam_xy_robot, calib.cam_height_mm
    if not cam or not height:
        return 0.0
    return math.hypot(x - cam[0], y - cam[1]) / float(height)


def _unproject(
    calib: Calibration, px: float, py: float, height_mm: float
) -> tuple[float, float]:
    """Pixel -> robot XY for a point ``height_mm`` above the table.

    Exact inverse of the forward parallax in ``Calibration.robot_to_pixel``,
    which scales (x, y) about the nadir by ``H / (H - h)``: undoing it is a
    contraction toward the nadir by ``(H - h) / H``.
    """
    fx, fy = calib.pixel_to_robot(px, py, on_cube_top=False)
    cam, height = calib.cam_xy_robot, calib.cam_height_mm
    if not cam or not height or height_mm <= 0.0 or height - height_mm <= 1e-6:
        return fx, fy
    k = (float(height) - height_mm) / float(height)
    return cam[0] + (fx - cam[0]) * k, cam[1] + (fy - cam[1]) * k


def _assumed_height_mm(
    silhouette_mm: float, gain: float, cos_radial: float
) -> float:
    """Object height inferred from how much its own height inflated its width.

    A silhouette is the union of the footprint and the top outline, the latter
    thrown outward from the nadir by ``h * gain``, so across the grasp axis it
    measures ``w + h * gain * cos_radial``. One view cannot separate w from h, so
    this assumes a cross-section **as tall as it is wide** -- true of pens,
    markers and tools lying on a table, which is what this module is for, and
    false of a flat sheet, which would be reported as thicker and nearer than it
    is. Pass ``object_height_mm=0`` for something genuinely flat.
    """
    if gain <= 0.0:
        # No measured camera geometry, so there is no correction to make and
        # nothing to infer a height from. Claim nothing.
        return 0.0
    denom = 1.0 + gain * max(0.0, min(1.0, cos_radial))
    return max(0.0, silhouette_mm / denom) if denom > 1e-6 else 0.0


def _median_width_px(mask: np.ndarray, long_axis_deg_px: float) -> float:
    """Median cross-section width (px) across the object's long axis.

    Not ``minAreaRect``'s short side, which is the *widest* point anywhere along
    the object -- a pen's cap, or a stray shadow lobe, sets it for the whole
    shaft. The gripper closes at one place, so the typical width is the honest
    number to hand it.
    """
    h, w = mask.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), long_axis_deg_px, 1.0)
    diag = int(math.hypot(h, w)) + 2
    rot[0, 2] += (diag - w) / 2.0
    rot[1, 2] += (diag - h) / 2.0
    up = cv2.warpAffine(mask, rot, (diag, diag), flags=cv2.INTER_NEAREST)
    cols = up.sum(axis=0)
    nz = cols[cols > 0]
    return float(np.median(nz)) if nz.size else 0.0


def _object_from_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    calib: Calibration,
    object_height_mm: float | None,
) -> LocatedObject | None:
    """Geometry from a crop-local mask; None if the silhouette is degenerate."""
    x0, y0 = origin
    try:
        centroid, corners = _rect_from_mask(mask, (x0, y0))
    except LocateError:
        return None
    robot = [calib.pixel_to_robot(cx, cy, on_cube_top=False) for cx, cy in corners]
    cx, cy = calib.pixel_to_robot(centroid[0], centroid[1], on_cube_top=False)

    # minAreaRect corners are ordered around the rect, so consecutive pairs are
    # its two side directions.
    e0 = (robot[1][0] - robot[0][0], robot[1][1] - robot[0][1])
    e1 = (robot[2][0] - robot[1][0], robot[2][1] - robot[1][1])
    p0 = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    p1 = math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
    len0, len1 = math.hypot(*e0), math.hypot(*e1)
    if len0 >= len1:
        long_v, long_mm, short_mm, long_px, short_px = e0, len0, len1, p0, p1
    else:
        long_v, long_mm, short_mm, long_px, short_px = e1, len1, len0, p1, p0
    if long_mm < 1e-6 or long_px < 1e-6 or short_px < 1e-6:
        return None
    axis_yaw = math.degrees(math.atan2(long_v[1], long_v[0]))

    # The long axis in *pixel* space, to de-rotate the mask. Robot-frame yaw is
    # not usable here: the homography rotates and shears.
    ix = corners[1][0] - corners[0][0] if len0 >= len1 else corners[2][0] - corners[1][0]
    iy = corners[1][1] - corners[0][1] if len0 >= len1 else corners[2][1] - corners[1][1]
    med_px = _median_width_px(mask, math.degrees(math.atan2(iy, ix)))
    if med_px > 0:
        short_mm *= med_px / short_px

    # Undo the object's own height. Both the grasp point and the width need it:
    # the centroid of a silhouette sits half the object's height up, and the
    # width is inflated by the whole of it.
    gain = _parallax_gain(calib, cx, cy)
    cam = calib.cam_xy_robot
    cos_radial = 1.0
    if cam:
        d = math.hypot(cam[0] - cx, cam[1] - cy)
        if d > 1e-6:
            ax, ay = -math.sin(math.radians(axis_yaw)), math.cos(math.radians(axis_yaw))
            cos_radial = abs(ax * (cam[0] - cx) + ay * (cam[1] - cy)) / d
    height = (
        _assumed_height_mm(short_mm, gain, cos_radial)
        if object_height_mm is None
        else max(0.0, float(object_height_mm))
    )
    if height > 0.0:
        short_mm = max(1.0, short_mm - height * gain * cos_radial)
        # The silhouette's centroid lies midway between the footprint and the
        # top outline, so it unprojects at half the object's height.
        cx, cy = _unproject(calib, centroid[0], centroid[1], height / 2.0)

    th, tw = mask.shape[:2]
    return LocatedObject(
        label="",
        px=centroid[0],
        py=centroid[1],
        x=cx,
        y=cy,
        axis_yaw_deg=axis_yaw,
        long_mm=long_mm,
        short_mm=short_mm,
        confidence=1.0,
        height_mm=height,
        template=np.ascontiguousarray(frame[y0 : y0 + th, x0 : x0 + tw]),
        mask_area_px=float(mask.sum()),
    )


def _measure_one(
    frame: np.ndarray,
    px: float,
    py: float,
    calib: Calibration,
    win: int,
    exclude: np.ndarray | None,
    object_height_mm: float | None,
) -> tuple[LocatedObject, np.ndarray] | None:
    """One window's measurement, or None if nothing could be segmented."""
    got = _segment(frame, px, py, win, exclude)
    if got is None:
        return None
    mask, (x0, y0, _x1, _y1) = got
    obj = _object_from_mask(frame, mask, (x0, y0), calib, object_height_mm)
    if obj is None:
        return None
    return obj, mask


def measure(
    frame: np.ndarray,
    px: float,
    py: float,
    calib: Calibration,
    label: str,
    *,
    win: int = DEFAULT_WINDOW_PX,
    confidence: float = 1.0,
    marker_xy: Sequence[tuple[float, float]] = (),
    object_height_mm: float | None = None,
) -> LocatedObject:
    """Measure the object at a pixel hint in robot-frame millimetres.

    The rect corners are projected through the flat table-plane homography, so
    ``long_mm``/``short_mm`` are true table distances and ``axis_yaw_deg`` is a
    robot-frame angle -- computed the same way ``detect._robot_edge_yaw_deg``
    derives a cube's edge yaw, but at table height rather than cube-top height.

    Measured at two window sizes, and **refuses unless they agree**. Everything
    downstream -- where the arm goes, how far the jaws close -- is derived from
    the mask, and segmentation is by far the least reliable step in the chain, so
    a disagreement between scales has to abstain rather than pick a winner.

    ``short_mm`` is a silhouette width, so for anything with height it reads
    *wide*: this camera is steeply oblique, and an object's side face projects
    onto the table plane next to it (a 12mm round pen measures ~24mm). Treat it
    as an upper bound, and verify the grasp rather than trusting it.
    """
    exclude = marker_paper_mask(frame, calib, marker_xy) if marker_xy else None
    primary = _measure_one(frame, px, py, calib, win, exclude, object_height_mm)
    if primary is None:
        raise LocateError(
            f"could not segment an object at ({px:.0f}, {py:.0f}) -- the hint "
            "may be on the desk, on a marker paper, or on something larger than "
            "the segmentation window"
        )
    obj, _mask = primary

    if not MIN_PLAUSIBLE_LONG_MM <= obj.long_mm <= MAX_PLAUSIBLE_LONG_MM:
        raise LocateError(
            f"measured {obj.long_mm:.0f}x{obj.short_mm:.0f}mm at "
            f"({px:.0f}, {py:.0f}), outside the plausible "
            f"{MIN_PLAUSIBLE_LONG_MM:.0f}-{MAX_PLAUSIBLE_LONG_MM:.0f}mm range -- "
            "the mask has most likely flooded the desk, so its centroid means "
            "nothing"
        )

    second_win = max(BG_MEDIAN_KERNEL_PX, int(win * STABILITY_WINDOW_RATIO))
    if second_win != win:
        other = _measure_one(
            frame, px, py, calib, second_win, exclude, object_height_mm
        )
        if other is None:
            raise LocateError(
                f"segmentation at ({px:.0f}, {py:.0f}) did not survive a "
                f"{second_win}px window -- too unstable to act on"
            )
        alt = other[0]
        dx, dy = alt.x - obj.x, alt.y - obj.y
        if is_compact(obj.long_mm, obj.short_mm):
            # Near-square: which edge is "long" flips ~90° between scales and
            # is not load-bearing for the grasp. Only centroid and overall size.
            drift = math.hypot(dx, dy)
            dsize = abs(
                (alt.long_mm + alt.short_mm) - (obj.long_mm + obj.short_mm)
            )
            if drift > STABILITY_CENTROID_MM or dsize > 2.0 * STABILITY_SHORT_MM:
                raise LocateError(
                    f"unstable segmentation at ({px:.0f}, {py:.0f}): "
                    f"compact object drifted {drift:.0f}mm / {dsize:.0f}mm size "
                    f"across window scales"
                )
        else:
            # Split disagreement along and across the axis. Along-axis drift is
            # expected (faded ends clip differently per scale) and harmless for
            # a shaft grasp; across-axis drift makes the jaws miss.
            rad = math.radians(obj.axis_yaw_deg)
            along = abs(dx * math.cos(rad) + dy * math.sin(rad))
            across = abs(-dx * math.sin(rad) + dy * math.cos(rad))
            dshort = abs(alt.short_mm - obj.short_mm)
            dyaw = abs((alt.axis_yaw_deg - obj.axis_yaw_deg + 90.0) % 180.0 - 90.0)
            if (
                across > STABILITY_CENTROID_MM
                or along > max(STABILITY_CENTROID_MM, 0.5 * obj.long_mm)
                or dshort > STABILITY_SHORT_MM
                or dyaw > STABILITY_YAW_DEG
            ):
                raise LocateError(
                    f"unstable segmentation at ({px:.0f}, {py:.0f}): {win}px and "
                    f"{second_win}px windows disagree by {across:.0f}mm across the "
                    f"axis, {along:.0f}mm along it, {dshort:.0f}mm in width and "
                    f"{dyaw:.0f}deg in angle -- refusing rather than guessing which "
                    "is right"
                )

    return replace(obj, label=label, confidence=confidence)


def measure_box(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    calib: Calibration,
    label: str,
    *,
    confidence: float = 1.0,
    object_height_mm: float | None = None,
) -> LocatedObject:
    """Geometry from an axis-aligned detector box (e.g. Grounding DINO).

    Fallback when desk-deviation segmentation is unstable or missing: the box
    is coarser (image-aligned AABB, loose on rotated objects) but consistent.
    Good enough for a full-close grip that does not need a precise jaw span.
    """
    h, w = frame.shape[:2]
    xa, xb = sorted((float(x1), float(x2)))
    ya, yb = sorted((float(y1), float(y2)))
    xa = max(0.0, min(xa, w - 1.0))
    xb = max(0.0, min(xb, w - 1.0))
    ya = max(0.0, min(ya, h - 1.0))
    yb = max(0.0, min(yb, h - 1.0))
    if xb - xa < 2.0 or yb - ya < 2.0:
        raise LocateError(
            f"detector box too small ({xb - xa:.0f}x{yb - ya:.0f}px) to measure"
        )

    # Corners ordered around the AABB -- consecutive pairs are the two sides.
    corners = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
    centroid = (0.5 * (xa + xb), 0.5 * (ya + yb))
    robot = [calib.pixel_to_robot(cx, cy, on_cube_top=False) for cx, cy in corners]
    cx, cy = calib.pixel_to_robot(centroid[0], centroid[1], on_cube_top=False)

    e0 = (robot[1][0] - robot[0][0], robot[1][1] - robot[0][1])
    e1 = (robot[2][0] - robot[1][0], robot[2][1] - robot[1][1])
    len0, len1 = math.hypot(*e0), math.hypot(*e1)
    if len0 >= len1:
        long_v, long_mm, short_mm = e0, len0, len1
    else:
        long_v, long_mm, short_mm = e1, len1, len0
    if long_mm < 1e-6:
        raise LocateError("detector box projects to a degenerate rect")
    if not MIN_PLAUSIBLE_LONG_MM <= long_mm <= MAX_PLAUSIBLE_LONG_MM:
        raise LocateError(
            f"box measures {long_mm:.0f}x{short_mm:.0f}mm, outside the plausible "
            f"{MIN_PLAUSIBLE_LONG_MM:.0f}-{MAX_PLAUSIBLE_LONG_MM:.0f}mm range"
        )
    axis_yaw = math.degrees(math.atan2(long_v[1], long_v[0]))

    gain = _parallax_gain(calib, cx, cy)
    cam = calib.cam_xy_robot
    cos_radial = 1.0
    if cam:
        d = math.hypot(cam[0] - cx, cam[1] - cy)
        if d > 1e-6:
            ax, ay = -math.sin(math.radians(axis_yaw)), math.cos(math.radians(axis_yaw))
            cos_radial = abs(ax * (cam[0] - cx) + ay * (cam[1] - cy)) / d
    height = (
        _assumed_height_mm(short_mm, gain, cos_radial)
        if object_height_mm is None
        else max(0.0, float(object_height_mm))
    )
    if height > 0.0:
        short_mm = max(1.0, short_mm - height * gain * cos_radial)
        cx, cy = _unproject(calib, centroid[0], centroid[1], height / 2.0)

    ix0, iy0 = int(math.floor(xa)), int(math.floor(ya))
    ix1, iy1 = int(math.ceil(xb)), int(math.ceil(yb))
    return LocatedObject(
        label=label,
        px=centroid[0],
        py=centroid[1],
        x=cx,
        y=cy,
        axis_yaw_deg=axis_yaw,
        long_mm=long_mm,
        short_mm=short_mm,
        confidence=float(confidence),
        height_mm=height,
        template=np.ascontiguousarray(frame[iy0:iy1, ix0:ix1]),
        mask_area_px=float((xb - xa) * (yb - ya)),
    )


def measure_grabcut(
    frame: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    calib: Calibration,
    label: str,
    *,
    confidence: float = 1.0,
    marker_xy: Sequence[tuple[float, float]] = (),
    object_height_mm: float | None = None,
) -> LocatedObject:
    """Measure via GrabCut seeded by a detector AABB.

    Tighter than ``measure_box`` (axis-aligned AABB) when the object contrasts
    with the desk; still refuses on empty / flooded masks.
    """
    exclude = marker_paper_mask(frame, calib, marker_xy) if marker_xy else None
    got = _segment_grabcut(frame, x1, y1, x2, y2, exclude)
    if got is None:
        raise LocateError(
            f"GrabCut found no foreground in box "
            f"({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
        )
    mask, (ox, oy, _x1, _y1) = got
    obj = _object_from_mask(frame, mask, (ox, oy), calib, object_height_mm)
    if obj is None:
        raise LocateError("GrabCut mask produced a degenerate silhouette")
    if not MIN_PLAUSIBLE_LONG_MM <= obj.long_mm <= MAX_PLAUSIBLE_LONG_MM:
        raise LocateError(
            f"GrabCut measured {obj.long_mm:.0f}x{obj.short_mm:.0f}mm, outside "
            f"the plausible {MIN_PLAUSIBLE_LONG_MM:.0f}-{MAX_PLAUSIBLE_LONG_MM:.0f}mm "
            "range -- the mask has most likely flooded the desk"
        )
    return replace(obj, label=label, confidence=float(confidence))


def measure_with_box_fallback(
    frame: np.ndarray,
    px: float,
    py: float,
    calib: Calibration,
    label: str,
    *,
    box: tuple[float, float, float, float] | None = None,
    win: int = DEFAULT_WINDOW_PX,
    confidence: float = 1.0,
    marker_xy: Sequence[tuple[float, float]] = (),
    object_height_mm: float | None = None,
) -> LocatedObject:
    """Prefer GrabCut from ``box``; else desk ``measure``; else raw ``measure_box``."""
    if box is not None:
        try:
            return measure_grabcut(
                frame, box[0], box[1], box[2], box[3], calib, label,
                confidence=confidence, marker_xy=marker_xy,
                object_height_mm=object_height_mm,
            )
        except LocateError:
            pass
    try:
        return measure(
            frame, px, py, calib, label,
            win=win, confidence=confidence, marker_xy=marker_xy,
            object_height_mm=object_height_mm,
        )
    except LocateError:
        if box is None:
            raise
        return measure_box(
            frame, box[0], box[1], box[2], box[3], calib, label,
            confidence=confidence, object_height_mm=object_height_mm,
        )


def relocate(
    frame: np.ndarray,
    obj: LocatedObject,
    calib: Calibration,
    *,
    search_px: int = RELOCATE_SEARCH_PX,
    min_score: float = RELOCATE_MIN_SCORE,
    win: int = DEFAULT_WINDOW_PX,
    marker_xy: Sequence[tuple[float, float]] = (),
) -> LocatedObject | None:
    """Find ``obj`` again in a fresh frame, or None.

    A registered object cannot be re-detected from scratch the way an HSV cube
    can -- there is no colour rule for "pen" -- so template matching near its
    last pixel is the only re-detector available. Note the tension with
    ``scene.py``'s no-persistent-tracks rule: the threshold is strict and
    failure returns None precisely so a vacated pose cannot linger, which is
    what that rule protects.

    Rotation defeats ``matchTemplate``, which is acceptable here because a
    re-acquire only ever spans motion the arm itself performed -- but it is why
    the failure mode must be None (re-scan) and never a low-score match.

    For the same reason, **None is not evidence that the object left the table**.
    Do not use this to check whether a grasp lifted something: a failed grasp
    that merely rotated the object returns None too, which reads as success. Ask
    instead whether the place it used to occupy still looks unlike bare desk.
    """
    th, tw = obj.template.shape[:2]
    if th < 2 or tw < 2:
        return None
    h, w = frame.shape[:2]
    x0 = max(0, int(obj.px) - tw // 2 - search_px)
    y0 = max(0, int(obj.py) - th // 2 - search_px)
    x1 = min(w, int(obj.px) + tw // 2 + search_px)
    y1 = min(h, int(obj.py) + th // 2 + search_px)
    region = frame[y0:y1, x0:x1]
    if region.shape[0] < th or region.shape[1] < tw:
        return None
    scores = cv2.matchTemplate(region, obj.template, cv2.TM_CCOEFF_NORMED)
    _minv, maxv, _minl, maxl = cv2.minMaxLoc(scores)
    if maxv < min_score:
        return None
    hint_x = x0 + maxl[0] + tw / 2.0
    hint_y = y0 + maxl[1] + th / 2.0
    try:
        return measure(
            frame, hint_x, hint_y, calib, obj.label,
            win=win, confidence=float(maxv), marker_xy=marker_xy,
        )
    except LocateError:
        return None


def grasp_feasibility(
    obj: LocatedObject, calib: Calibration
) -> tuple[bool, str | None]:
    """Can the arm actually take this object? Returns (ok, reason-if-not).

    Reuses the workspace predicates rather than restating them, and adds the
    two checks specific to a non-cube: the jaws must be able to span its short
    axis, and a wrist angle must exist that closes across its long axis.
    """
    r = math.hypot(obj.x, obj.y)
    if not is_mp_reachable_xy(obj.x, obj.y):
        return False, f"keep-out (r={r:.0f}mm)"
    if r > MAX_REACH_MM:
        return False, f"beyond max reach (r={r:.0f}mm)"
    if r > MAX_VERIFIABLE_RADIUS_MM:
        return False, f"beyond camera (r={r:.0f}mm)"
    span_open = _span_mm(calib, int(calib.grip_open_s))
    if span_open is not None and obj.short_mm > span_open:
        return False, (
            f"wider than jaws ({obj.short_mm:.0f}>{span_open:.0f}mm)"
        )
    if is_compact(obj.long_mm, obj.short_mm):
        # Near-square: jaw orientation is not load-bearing (90° period).
        return True, None
    if j4_for_long_axis(obj.axis_yaw_deg, x=obj.x, y=obj.y) is None:
        return False, "no J4 angle in soft limits"
    return True, None


def _span_mm(calib: Calibration, s: int) -> float | None:
    """Jaw span (mm) at gripper value ``s``, or None when uncalibrated."""
    at_zero = calib.grip_span_s_at_zero_mm
    per_mm = calib.grip_span_s_per_mm
    if at_zero is None or per_mm is None or per_mm == 0:
        return None
    return (float(at_zero) - float(s)) / float(per_mm)
