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
  steeply oblique (nadir ~360-490mm away, 242mm up), so the flat inverse is off
  by ~1.5x the height. A 12mm pen reads ~9mm from where it is, past the ~10mm
  the jaws tolerate: measured on hardware, the arm shoved the pen aside instead
  of gripping it, in exactly the direction this predicts. So the centroid is
  unprojected at half the object's height and the width is de-inflated by the
  whole of it.

  One view cannot measure height, so it is inferred from two independent cues
  and the **smaller** of the two is taken -- see :func:`_height_corrected`.
  Each cue is an over-estimate under a different shape, and each is tight
  where the other is loose, so the minimum is right for both a 20mm cube and a
  sheet of paper. It is reported on ``LocatedObject.height_mm`` because it is
  an inference, not a measurement, and ``object_height_mm`` overrides it for a
  caller that knows better.

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
from mt4_vision.detect import classify_color
from mt4_vision.wrist import j4_for_long_axis
from mt4_vision.workspace import work_region_block_reason

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
# is why a pen beside a tag does not fuse with it -- but it also means an
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
    # Height above the table inferred when unprojecting the centroid. Reported
    # because it is an inference, not a measurement, and it moves the grasp
    # point by ~1.5x itself at this mount -- see _height_corrected.
    height_mm: float = 0.0
    # BGR crop around the object, for re-acquiring it in a later frame.
    template: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((1, 1, 3), np.uint8))
    mask_area_px: float = 0.0
    # The silhouette itself, and where its top-left sits in the full frame.
    # Kept because ``long_mm``/``short_mm`` describe the whole outline and say
    # nothing about the width at any particular point on it -- which is the
    # only question that decides whether the jaws can close. See
    # ``mt4_vision.grasp``. Empty when the measurement predates this field.
    mask: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros((0, 0), np.uint8)
    )
    mask_origin_px: tuple[int, int] = (0, 0)
    # Measured colour, from the same named HSV bands the cube detector uses, or
    # None when the object is not clearly one of them. Distinct from anything in
    # ``label``: the label is the noun a language model supplied, this is a
    # measurement. Keeping them apart is what lets ``instruct`` tell "the task
    # says green and this thing is red" (a real contradiction, refuse) from
    # "the task says green and nobody has established what colour this is"
    # (unknowable, abstain). Conflating those refused "the green statue" on a
    # desk that had a green statue on it -- see entities.object_entity.
    color: str | None = None

    def footprint_offset_mm(self, calib) -> tuple[float, float]:
        """Height correction applied to the centroid, as a plane offset.

        ``x``/``y`` are unprojected at the object's assumed height; the mask is
        in raw pixels. Grasp planning maps mask pixels through the table plane
        and then shifts them by this, so a planned point lands in the same
        frame of reference the rest of the stack uses for this object.
        """
        rx, ry = calib.pixel_to_robot(self.px, self.py)
        return self.x - float(rx), self.y - float(ry)

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


def desk_deviation(crop: np.ndarray, kernel: int = BG_MEDIAN_KERNEL_PX) -> np.ndarray:
    """Per-pixel distance from the locally-estimated desk, in weighted Lab units.

    The desk is estimated *locally*, with a large median filter, rather than as
    one colour with a tolerance. That matters more than any threshold choice:
    across a 280px window the wood's own illumination gradient is larger than
    the difference between a pen and the desk, so a single reference colour
    leaks across the gradient and floods (measured: a Sharpie's mask swallowing
    half the desk). A local estimate absorbs the gradient and soft shadows into
    the background, leaving only what is small and high-frequency -- an object.

    Consequently the method is indifferent to what colour the desk is, and to
    wood grain, gradients and soft shadows. What it *is* sensitive to is scale:
    anything wider than ``BG_MEDIAN_KERNEL_PX`` survives its own background
    estimate and disappears. That is the real scope limit, not desk uniformity.

    Shared by :func:`_segment` (one window around a hint) and
    :mod:`mt4_vision.discover` (the whole desk, no hint), so the two cannot
    drift into different definitions of "unlike the desk". ``kernel`` is
    exposed for discover's coarse second pass, which deliberately probes a
    much larger scale to recognise structures too big for the default one;
    every other caller should leave it alone.
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    desk = cv2.medianBlur(lab, int(kernel) | 1)
    scale = np.array([L_WEIGHT, 1.0, 1.0], np.float32)
    return np.linalg.norm(
        (lab.astype(np.float32) - desk.astype(np.float32)) * scale, axis=2
    )


def _segment(
    frame: np.ndarray,
    px: float,
    py: float,
    win: int,
    exclude: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Foreground component under the hint, by deviation from the local desk.

    The split comes from Otsu on :func:`desk_deviation`, so there is no tuned
    colour tolerance to go stale when the lighting changes.
    """
    x0, y0, x1, y1 = _window(frame, px, py, win)
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    if crop.size == 0 or min(crop.shape[:2]) < BG_MEDIAN_KERNEL_PX:
        return None
    hx = max(0, min(int(round(px)) - x0, crop.shape[1] - 1))
    hy = max(0, min(int(round(py)) - y0, crop.shape[0] - 1))

    dev = desk_deviation(crop)

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
            paper = ex.astype(bool)
            inside = np.zeros_like(paper)
            inside[ly0:ly1, lx0:lx1] = True
            # Outside the detector box, marker paper is background and that is
            # the end of it.
            gc[paper & ~inside] = cv2.GC_BGD
            # Inside the box it is only a PRIOR. The mask is geometric -- a
            # quad around each marker's calibrated position -- so it cannot
            # tell paper from something lying on top of paper, and the box is
            # a strong statement that the object is right here.
            #
            # Measured 2026-08-02: the stapler was resting ON marker 0's paper
            # (the arm had placed it there). The exclusion covered 53.8% of
            # Qwen's box, including the stapler itself, so GrabCut found no
            # foreground at all -- 8 grounding calls out of 8. Every
            # measurement then fell through to the raw box, reporting the
            # AABB's dimensions as the object's: 150x74mm for a stapler, with
            # no silhouette for grasp planning to use.
            #
            # GC_PR_BGD lets GrabCut reclaim pixels its colour model says are
            # not paper, which is exactly the object on top of it, while still
            # steering it away from the high-contrast black-and-white tag that
            # made this exclusion necessary in the first place.
            gc[paper & inside] = cv2.GC_PR_BGD
            # The sure-FG seed has to survive: it is the only thing telling
            # GrabCut what the object's colour actually is.
            cv2.circle(gc, (hx, hy), r, int(cv2.GC_FGD), thickness=-1)

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
) -> tuple[tuple[float, float], list[tuple[float, float]], np.ndarray]:
    """(centroid, 4 corners, outline) in FRAME pixels from a crop-local mask.

    The outline is the silhouette contour itself, not the rectangle's corners.
    It is what :func:`_table_extents_mm` measures the radial sweep from, and a
    minAreaRect around a swept shape is looser than the shape.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise LocateError("mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
    ox, oy = origin
    corners = [(float(cx) + ox, float(cy) + oy) for cx, cy in box]
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()) + ox, float(ys.mean()) + oy)
    outline = contour.reshape(-1, 2).astype(np.float64) + np.array([ox, oy], np.float64)
    return centroid, corners, outline


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


def _radial_basis(
    calib: Calibration, x: float, y: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Unit vectors (outward from the nadir, across it) at (x, y), or None.

    Height displaces a silhouette purely along the outward vector, so this is
    the frame every height cue is measured in. None when the camera geometry is
    unmeasured, or when the point sits on the nadir itself and "outward" has no
    direction.
    """
    cam = calib.cam_xy_robot
    if not cam:
        return None
    dx, dy = x - float(cam[0]), y - float(cam[1])
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    return (dx / n, dy / n), (-dy / n, dx / n)


def _table_extents_mm(
    calib: Calibration,
    outline_px: np.ndarray,
    u: tuple[float, float],
    v: tuple[float, float],
) -> tuple[float, float]:
    """Outline width (mm) along the outward radial axis, and across it.

    Both measured on the flat table plane, which is the plane the sweep happens
    in -- the outline is projected through ``pixel_to_robot`` before either
    extent is taken, so lens and perspective foreshortening are already undone.
    """
    pts = np.array(
        [
            calib.pixel_to_robot(float(px), float(py), on_cube_top=False)
            for px, py in outline_px
        ],
        dtype=np.float64,
    )
    if not len(pts):
        return 0.0, 0.0
    pu = pts @ np.array(u, dtype=np.float64)
    pv = pts @ np.array(v, dtype=np.float64)
    return float(pu.max() - pu.min()), float(pv.max() - pv.min())


def _assumed_height_mm(
    silhouette_mm: float, gain: float, cos_radial: float
) -> float:
    """Height inferred from how much the object's own height inflated its width.

    A silhouette is the union of the footprint and the top outline, the latter
    thrown outward from the nadir by ``h * gain``, so across the grasp axis it
    measures ``w + h * gain * cos_radial``. One view cannot separate w from h, so
    this assumes a cross-section **as tall as it is wide** -- true of pens,
    markers and tools lying on a table, false of a flat sheet, which it reports
    as thicker and nearer than it is. Synthetic prisms through this
    calibration: 0.1-3.9mm across the grasp axis for a 12mm pen, 5.2-25.1mm for
    a 90x60x0.5mm sheet. Live on ArUco paper, which is flat and whose position
    is known: it invents 15-22mm of height and moves the point 13.0-14.6mm.

    Tight where the object really is round in section, loose for anything flat,
    which is the half :func:`_height_from_sweep` answers.
    """
    if gain <= 0.0:
        # No measured camera geometry, so there is no correction to make and
        # nothing to infer a height from. Claim nothing.
        return 0.0
    denom = 1.0 + gain * max(0.0, min(1.0, cos_radial))
    return max(0.0, silhouette_mm / denom) if denom > 1e-6 else 0.0


def _height_from_sweep(radial_mm: float, cross_mm: float, gain: float) -> float:
    """Height inferred from how far the silhouette is stretched radially.

    An object standing on the table images as its footprint unioned with its
    top outline, and the top is thrown outward from the nadir by exactly
    ``h * gain``. That stretch lands **entirely** on the radial axis: the
    across-radial extent of the silhouette is the footprint's own, untouched.
    So the difference between the two extents is the stretch, and the height
    follows -- given a footprint about as deep as it is wide, which is what
    ``cross_mm`` stands in for.

    Tight for anything compact, whatever its height, and tight for anything
    flat, whatever its outline: a sheet has no stretch to find and this returns
    zero. Loose for a long object pointing at the camera, where the footprint's
    own length is counted as stretch -- 42-108mm of invented height for a
    140mm pen. That is the half :func:`_assumed_height_mm` answers.

    Synthetic prisms through this calibration, error across the grasp axis:
    0.0-1.2mm for a 20mm cube (the flat projection gives 2.6-17.6mm), 0.1-3.0mm
    for a 60x40x1mm card, 0.1-1.8mm for a 45x40x30mm rock.
    """
    if gain <= 0.0:
        return 0.0
    return max(0.0, radial_mm - cross_mm) / gain


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


def _check_plausible(long_mm: float, short_mm: float, what: str) -> None:
    """Refuse an implausible extent rather than picking at it.

    A mask that has flooded the desk yields a huge "object" whose centroid is
    meaningless, and the arm acts on whatever comes back. ``what`` names the
    path that produced the measurement, because which segmenter flooded is the
    first thing worth knowing.
    """
    if MIN_PLAUSIBLE_LONG_MM <= long_mm <= MAX_PLAUSIBLE_LONG_MM:
        return
    raise LocateError(
        f"{what} measured {long_mm:.0f}x{short_mm:.0f}mm, outside the plausible "
        f"{MIN_PLAUSIBLE_LONG_MM:.0f}-{MAX_PLAUSIBLE_LONG_MM:.0f}mm range -- the "
        "mask has most likely flooded the desk, so its centroid means nothing"
    )


def _height_corrected(
    calib: Calibration,
    centroid_px: tuple[float, float],
    cx: float,
    cy: float,
    axis_yaw: float,
    short_mm: float,
    outline_px: np.ndarray,
    object_height_mm: float | None,
) -> tuple[float, float, float, float]:
    """Apply the object's own height to its centre and width. (cx, cy, short, h).

    One copy, because both segmentation paths need it and they had it written
    out identically. See the module docstring for why height cannot be ignored:
    this camera is steeply oblique, so the flat table inverse puts a silhouette
    centroid ~1.5x its height away from the thing itself -- past what the jaws
    tolerate for a 12mm pen.

    **The height is the smaller of two cues**, :func:`_assumed_height_mm` from
    the width inflation and :func:`_height_from_sweep` from the radial stretch.
    Both are upper bounds that go loose under opposite shapes -- the first
    invents height for anything flat, the second for anything long pointing at
    the camera -- and neither is ever tight on the shape that defeats the
    other, so the minimum is the one number that is close for a cube, a card
    and a pen alike. Measured across the grasp axis over 84 synthetic prisms
    through this calibration: mean 3.9mm, and 9 of 84 outside the ~10mm the
    jaws tolerate, against 7.2mm / 23 of 84 for the flat projection and
    11.0mm / 32 of 84 for the width cue alone.

    Taking the minimum means erring toward **under**-correction, and one shape
    gets no correction at all: something long lying across the camera azimuth,
    where the stretch cue reads no radial elongation and wins the minimum. A
    140x12x12mm pen there stays 6.1-12.1mm outward, exactly where the flat
    projection puts it. That is a genuine one-view ambiguity -- the same
    silhouette is cast by a flat strip three times as wide, whose centre is
    ~10mm away -- and resolving it in the pen's favour would over-correct every
    flat object instead.

    What defeats both cues is a mask that is not one object: two red cubes 44mm
    apart segment as a 121mm blob, and every rule then aims at the gap. That is
    the segmentation's job to refuse -- see :func:`_check_plausible` and the
    two-window stability check in :func:`measure`.

    ``cos_radial`` is how much of the object's short axis points along the
    camera azimuth. Only that component is inflated by height, so a pen lying
    across the view direction needs no de-inflation and one lying along it
    needs all of it.
    """
    gain = _parallax_gain(calib, cx, cy)
    basis = _radial_basis(calib, cx, cy)
    cos_radial = 1.0
    if basis is not None:
        u, v = basis
        ax, ay = -math.sin(math.radians(axis_yaw)), math.cos(math.radians(axis_yaw))
        cos_radial = abs(ax * u[0] + ay * u[1])
    if object_height_mm is not None:
        height = max(0.0, float(object_height_mm))
    else:
        height = _assumed_height_mm(short_mm, gain, cos_radial)
        if basis is not None and len(outline_px):
            radial_mm, cross_mm = _table_extents_mm(calib, outline_px, u, v)
            height = min(height, _height_from_sweep(radial_mm, cross_mm, gain))
    if height > 0.0:
        short_mm = max(1.0, short_mm - height * gain * cos_radial)
        # The silhouette's centroid lies midway between the footprint and the
        # top outline, so it unprojects at half the object's height.
        cx, cy = _unproject(calib, centroid_px[0], centroid_px[1], height / 2.0)
    return cx, cy, short_mm, height


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
        centroid, corners, outline = _rect_from_mask(mask, (x0, y0))
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
    cx, cy, short_mm, height = _height_corrected(
        calib, centroid, cx, cy, axis_yaw, short_mm, outline, object_height_mm
    )

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
        # Carried so grasp planning can ask about the width at a *point* on the
        # object. long_mm/short_mm describe the whole outline and cannot answer
        # that -- see mt4_vision.grasp.
        mask=np.ascontiguousarray(mask.astype(np.uint8)),
        mask_origin_px=(int(x0), int(y0)),
        # Through the silhouette, not the bounding box: the box around anything
        # elongated is mostly desk, and desk matches no band, which would drag
        # every honest colour under COLOR_MIN_SHARE.
        color=classify_color(
            frame, mask.astype(np.uint8), (int(x0), int(y0)), calib
        ),
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

    ``short_mm`` starts as a silhouette width, which reads *wide* for anything
    with height: this camera is steeply oblique, and an object's side face
    projects onto the table plane next to it (a 12mm round pen measures ~24mm).
    It is de-inflated by the inferred height, so it is only as good as that
    inference. Treat it as an upper bound, and verify the grasp rather than
    trusting it.
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

    _check_plausible(obj.long_mm, obj.short_mm, f"the window at ({px:.0f}, {py:.0f})")

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
    _check_plausible(long_mm, short_mm, "the box")
    axis_yaw = math.degrees(math.atan2(long_v[1], long_v[0]))

    # The box corners stand in for the outline there is no mask to supply. An
    # image-aligned AABB is looser than a silhouette in both directions at
    # once, so the radial-stretch cue reads high here and the width cue is
    # usually what the minimum picks -- which is this path's historical
    # behaviour, and appropriate for a last-resort measurement.
    cx, cy, short_mm, height = _height_corrected(
        calib, centroid, cx, cy, axis_yaw, short_mm,
        np.array(corners, dtype=np.float64), object_height_mm,
    )

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
        # No silhouette on this path, so the box interior stands in -- see
        # classify_color. Desk showing at the box edges only makes the answer
        # None, never a wrong colour.
        color=classify_color(frame[iy0:iy1, ix0:ix1], calibration=calib),
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
    _check_plausible(obj.long_mm, obj.short_mm, "GrabCut")
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
    again, _why = relocate_detail(
        frame, obj, calib, search_px=search_px, min_score=min_score,
        win=win, marker_xy=marker_xy,
    )
    return again


def relocate_detail(
    frame: np.ndarray,
    obj: LocatedObject,
    calib: Calibration,
    *,
    search_px: int = RELOCATE_SEARCH_PX,
    min_score: float = RELOCATE_MIN_SCORE,
    win: int = DEFAULT_WINDOW_PX,
    marker_xy: Sequence[tuple[float, float]] = (),
) -> tuple[LocatedObject | None, str]:
    """:func:`relocate`, plus which of its four failures happened.

    The reason matters because they call for opposite responses. "It moved" is
    a scene problem the caller should report; "found it, could not re-measure
    it" is ours, and telling a user their stationary stapler moved sends them
    looking at the desk instead of at the code.

    **The re-measure runs through the matched template's own bounds.** That box
    is a tight, true outline of the object in this frame -- it is the crop the
    object was registered from, found again -- so it seeds GrabCut exactly the
    way a detector box does. The plain desk-deviation :func:`measure` cannot
    segment a stapler at all (measured 2026-08-02: one that had not moved
    template-matched at **0.993** on the identical pixel), so without the
    template bounds any object that only GrabCut could find in the first place
    is permanently un-re-acquirable: registration succeeds, and every attempt
    to act on it is refused.
    """
    th, tw = obj.template.shape[:2]
    if th < 2 or tw < 2:
        return None, "no usable template was stored for this object"
    h, w = frame.shape[:2]
    x0 = max(0, int(obj.px) - tw // 2 - search_px)
    y0 = max(0, int(obj.py) - th // 2 - search_px)
    x1 = min(w, int(obj.px) + tw // 2 + search_px)
    y1 = min(h, int(obj.py) + th // 2 + search_px)
    region = frame[y0:y1, x0:x1]
    if region.shape[0] < th or region.shape[1] < tw:
        return None, (
            f"the {tw}x{th}px template does not fit the search area at the "
            "frame edge"
        )
    scores = cv2.matchTemplate(region, obj.template, cv2.TM_CCOEFF_NORMED)
    _minv, maxv, _minl, maxl = cv2.minMaxLoc(scores)
    if maxv < min_score:
        return None, (
            f"best template match {maxv:.2f} is under {min_score:.2f} -- it "
            "moved, rotated, or something is over it"
        )
    bx0, by0 = x0 + maxl[0], y0 + maxl[1]
    box = (float(bx0), float(by0), float(bx0 + tw), float(by0 + th))
    hint_x, hint_y = bx0 + tw / 2.0, by0 + th / 2.0
    try:
        return measure_with_box_fallback(
            frame, hint_x, hint_y, calib, obj.label,
            box=box, win=win, confidence=float(maxv), marker_xy=marker_xy,
        ), ""
    except LocateError as exc:
        return None, (
            f"matched it at {maxv:.2f} but could not re-measure it: {exc}"
        )


def grasp_feasibility(
    obj: LocatedObject, calib: Calibration
) -> tuple[bool, str | None]:
    """Can the arm actually take this object? Returns (ok, reason-if-not).

    Reuses the workspace predicates rather than restating them, and adds the
    two checks specific to a non-cube: the jaws must be able to span its short
    axis, and a wrist angle must exist that closes across its long axis.
    """
    region = work_region_block_reason(obj.x, obj.y, calib)
    if region is not None:
        return False, region
    span = jaw_span_block_reason(obj.short_mm, calib)
    if span is not None:
        return False, span
    if is_compact(obj.long_mm, obj.short_mm):
        # Near-square: jaw orientation is not load-bearing (90° period).
        return True, None
    if j4_for_long_axis(obj.axis_yaw_deg, x=obj.x, y=obj.y) is None:
        return False, "no J4 angle in soft limits"
    return True, None


def plan_object_grasp(obj: LocatedObject, calib: Calibration):
    """Best jaw placement on ``obj``, or (None, why not).

    Falls back to the outline when no silhouette was stored -- ``measure_box``
    has no mask, and neither do objects measured before the mask was carried --
    so the answer degrades to "close across the short axis at the centroid"
    rather than to a refusal.
    """
    from mt4_vision.grasp import GraspPlan, footprint_mm, plan_grasp

    span = _span_mm(calib, int(calib.grip_open_s))
    if span is None:
        # Nothing to plan against. Keep the historical behaviour rather than
        # inventing a jaw width -- see jaw_span_block_reason.
        return GraspPlan(
            x=obj.x, y=obj.y, yaw_deg=obj.axis_yaw_deg,
            width_mm=obj.short_mm, length_mm=obj.long_mm, offset_mm=0.0,
        ), ""
    mask = getattr(obj, "mask", None)
    if mask is None or getattr(mask, "size", 0) == 0:
        reason = jaw_span_block_reason(obj.short_mm, calib)
        if reason is not None:
            return None, reason
        return GraspPlan(
            x=obj.x, y=obj.y, yaw_deg=obj.axis_yaw_deg,
            width_mm=obj.short_mm, length_mm=obj.long_mm, offset_mm=0.0,
        ), ""
    pts = footprint_mm(
        obj.mask, obj.mask_origin_px, calib,
        offset_mm=obj.footprint_offset_mm(calib),
    )
    return plan_grasp(pts, max_span_mm=span)


def jaw_span_block_reason(short_mm: float, calib: Calibration) -> str | None:
    """"Too wide for the jaws" in prose, or None. The single width test.

    Lives here so ``grasp_feasibility`` (CLI, MCP server) and
    ``entities.object_entity`` (the policy loop) cannot drift into two
    different answers about whether the arm can hold something -- which they
    had, because ``object_entity`` had no width test at all.

    Silent when ``grip_span_s_*`` are unmeasured, which is a deliberate
    fail-open: inventing a jaw width would refuse real objects on a rig whose
    gripper nobody has measured. It is not a safe default, only the least
    wrong one, and it is why measuring the span matters. Measured on this rig
    2026-08-02 by photographing the jaws at table height across S 140..230:
    ``span_mm = (205.0 - S) / 1.797``, so 36mm at ``grip_open_s`` = 140.

    Note ``short_mm`` is a *silhouette* width and reads wide for anything with
    height on this oblique mount, so the comparison already errs toward
    refusing. That is the right direction: closing on air is the failure with
    no detector.
    """
    span_open = _span_mm(calib, int(calib.grip_open_s))
    if span_open is None or short_mm <= span_open:
        return None
    return (
        f"it measures {short_mm:.0f}mm across the grasp and the jaws open to "
        f"{span_open:.0f}mm -- they would close beside it, not on it"
    )


def _span_mm(calib: Calibration, s: int) -> float | None:
    """Jaw span (mm) at gripper value ``s``, or None when uncalibrated."""
    at_zero = calib.grip_span_s_at_zero_mm
    per_mm = calib.grip_span_s_per_mm
    if at_zero is None or per_mm is None or per_mm == 0:
        return None
    return (float(at_zero) - float(s)) / float(per_mm)
