"""Find every object on the desk, without knowing what any of them are.

``scene.py`` enumerates cubes because HSV can name a colour. Everything else --
a pen, a key, a stapler -- only ever reached the entity list because a human or
a VLM pointed at it first (``locate.measure``, ``mt4_locate_at_pixel``). That
asymmetry is not a detail: an object nobody registered is not merely hard to
select, it is absent from the world model, so "pick up the stapler" answers
"no such entity" even with the thing plainly in frame.

This module removes the hint. It proposes one grasp-point candidate per
desk-deviating blob, and hands each to the ordinary
:func:`mt4_vision.locate.measure` path -- so from measurement onward a
discovered object and a pointed-at object are the same code, with the same
two-window stability check, the same plausibility band and the same
work-region gate. Nothing here measures anything itself.

**What it assumes, and what it does not.** It does not assume a uniform desk:
:func:`~mt4_vision.locate.desk_deviation` estimates the surface locally with a
median filter, so wood grain, illumination gradients and soft shadows are
absorbed into the background rather than flagged (that was measured -- a single
reference colour let a Sharpie's mask swallow half the desk). What it does
assume is **scale**: the estimator's ``BG_MEDIAN_KERNEL_PX`` window means
anything wider than about 30mm at this mount survives its own background
estimate and vanishes. A pen, key or cube is found; a book, a phone or a
mousepad is not, on any desk. That is the real limit of the method, and
``MAX_BLOB_PX`` below is where it is enforced honestly instead of silently.

Expect over-detection rather than under-detection: cables, tape, and the arm's
own body all deviate from the desk. That is the safer direction, because a
spurious proposal meets ``measure``'s stability check and then
``in_work_region``, whereas a missed object is invisible to every later stage.
:func:`discover` returns what survived; :func:`propose` exposes the raw blobs
for tuning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from mt4_vision.calib import Calibration
from mt4_vision.locate import (
    BG_MEDIAN_KERNEL_PX,
    MIN_DEVIATION,
    LocateError,
    LocatedObject,
    desk_deviation,
    marker_paper_mask,
    measure,
)
from mt4_vision.workspace import dist_mm, in_work_region

# Smallest blob worth measuring. At this mount (~0.5mm/px) a 12mm pen is ~24px
# long and ~16px of silhouette wide, so ~200px2; 100 keeps that with margin
# while dropping single-speckle noise the 3x3 opening missed.
MIN_BLOB_PX = 100.0
# Largest blob worth measuring. Beyond the background estimator's own window a
# blob is not an object being detected, it is the estimator failing -- the arm's
# body, a merged cable run, or the desk edge. MAX_PLAUSIBLE_LONG_MM (200mm) at
# ~0.5mm/px is a 400px extent, so this is that extent squared, loosely: past it
# `measure` would refuse anyway, and refusing here saves the segmentation.
MAX_BLOB_PX = 40000.0
# Two measurements this close describe one physical object. Sized from the
# stability tolerance (STABILITY_CENTROID_MM = 6mm): two hints on one blob can
# legitimately land this far apart, further apart means two things.
MERGE_MM = 12.0
# Dilate the marker-paper exclusion by this much before proposing. The paper's
# own white tape edges are high-contrast and would otherwise each propose an
# object; `measure` excludes the paper but only after a hint already exists.
EXCLUDE_DILATE_PX = 9
# Coarse second pass, for recognising structures too wide for the default
# background estimate. An object wider than BG_MEDIAN_KERNEL_PX does not vanish
# cleanly -- its interior is absorbed but its CORNERS still deviate, and each
# corner then measures as a small, plausible, perfectly stable object. Measured
# on a synthetic 183px square: four phantom 12x6mm objects, all four surviving
# the two-window stability check. On a real desk that is a book or a phone
# offering the arm four fake grasp points, so it has to be vetoed, not ranked.
#
# The veto is the same deviation test at ~8x the scale, where such a structure
# shows up whole instead of only at its corners. Done on a downscaled copy
# because cv2.medianBlur refuses ksize>=16 histogram bins above ~301px and
# costs 56ms there anyway; /8 with a 61px kernel is the same effective 488px
# window for 2ms.
OVERSIZE_SCALE = 8
OVERSIZE_KERNEL_PX = 61
# How many background kernels wide a structure must be before it is vetoed.
# Not 1.0, even though 1.0 is where the estimator's guarantee actually ends.
# Measured on the live desk (2026-08-02, 9 cubes + arm + cables + a figurine),
# coarse component widths in full-res pixels:
#
#     cubes                     24, 40, 40, 48, 57, 64
#     arm base / remote / arm   94, 101, 242
#
# A 20mm cube is only ~40px of footprint but this mount is steeply oblique, so
# its side face projects out beside it and the silhouette reaches 64px -- just
# past the 61px kernel. Vetoing at exactly one kernel therefore deletes cubes,
# which is the one thing the benchmark object must survive. 1.25 sits in the
# 64-94px gap with ~20% margin either side. Between 1.0 and 1.25 kernels the
# fine pass can still emit corner slivers; the plausibility band and the
# two-window stability check are what catch those.
OVERSIZE_WIDTH_KERNELS = 1.25


@dataclass(frozen=True)
class Proposal:
    """One candidate grasp point, before anything has been measured."""

    px: float
    py: float
    area_px: float

    def as_dict(self) -> dict[str, object]:
        return {
            "pixel": [round(self.px, 1), round(self.py, 1)],
            "area_px": round(self.area_px),
        }


def _pixel_grid_robot(
    frame: np.ndarray, calib: Calibration
) -> tuple[np.ndarray, np.ndarray]:
    """Robot-frame X and Y for every pixel, on the table plane.

    The same flat homography ``Calibration.pixel_to_robot`` applies, done once
    for the whole grid instead of per point -- a per-pixel Python call over
    1280x720 is not viable and this is exact, not an approximation of it.
    """
    h, w = frame.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=-1)  # h, w, 3
    hm = np.array(calib.homography, dtype=np.float64)
    out = pts @ hm.T
    wz = out[..., 2]
    # A pixel on the horizon divides by ~0; it is not desk, so mark it far away
    # rather than letting an inf reach the polygon test.
    safe = np.where(np.abs(wz) < 1e-9, np.nan, wz)
    return out[..., 0] / safe, out[..., 1] / safe


def _points_in_polygon(
    x: np.ndarray, y: np.ndarray, poly: np.ndarray
) -> np.ndarray:
    """Vectorized even-odd containment test, matching ``workspace.on_table``."""
    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        # Does the upward ray from (x, y) cross this edge?
        straddles = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x1 - x0) * (y - y0) / (y1 - y0) + x0
        inside ^= straddles & (x < xint)
    return inside


def desk_pixel_mask(frame: np.ndarray, calib: Calibration) -> np.ndarray:
    """True where a pixel images desk surface, per the calibrated polygon.

    Only a coarse pre-filter: it keeps the wall, the monitor and the floor from
    generating hundreds of proposals, and costs one vectorized homography. The
    binding gate is still ``in_work_region`` on each measured candidate, which
    also knows about reach, keep-out, lift and camera coverage.
    """
    poly = getattr(calib, "table_polygon_robot", None)
    if not poly:
        return np.ones(frame.shape[:2], dtype=bool)
    rx, ry = _pixel_grid_robot(frame, calib)
    inside = _points_in_polygon(rx, ry, np.asarray(poly, dtype=np.float64))
    return inside & np.isfinite(rx) & np.isfinite(ry)


def oversize_mask(
    frame: np.ndarray,
    keep: np.ndarray,
    *,
    max_width_px: float = OVERSIZE_WIDTH_KERNELS * BG_MEDIAN_KERNEL_PX,
) -> np.ndarray:
    """Pixels belonging to a structure too wide to measure. See OVERSIZE_SCALE.

    True over the *whole* structure, not just where it deviated at the fine
    scale, which is the point: the fine pass only ever sees such a thing's
    corners, and a corner is indistinguishable from a small object once it has
    been cut out and measured.

    The criterion is **width**, not area, because that is precisely what the
    background estimator's window bounds -- a 183px square is only 33k px2,
    under ``MAX_BLOB_PX``, yet three kernels wide and therefore unmeasurable.

    Width is tested *per pixel*, by opening with a disk, rather than per
    connected component. Per-component is wrong: at the coarse scale a cube
    100px from the arm's base merges into the base's blob, and judging the
    merged blob by its widest part deletes two real cubes on the live frame. Opening asks "does a disk this wide fit here", which the
    cube fails and the base passes, whatever they are connected to.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(
        frame, (max(1, w // OVERSIZE_SCALE), max(1, h // OVERSIZE_SCALE)),
        interpolation=cv2.INTER_AREA,
    )
    if min(small.shape[:2]) < OVERSIZE_KERNEL_PX:
        return np.zeros((h, w), dtype=bool)
    keep_small = (
        cv2.resize(
            keep.astype(np.uint8), (small.shape[1], small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    )
    if not keep_small.any():
        return np.zeros((h, w), dtype=bool)

    dev = desk_deviation(small, OVERSIZE_KERNEL_PX)
    d8 = np.clip(dev, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(
        d8[keep_small], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    fg = ((dev > max(MIN_DEVIATION, float(otsu))) & keep_small).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Radius, in coarse pixels, of the disk a region must contain to count as
    # too wide. Opening keeps only pixels where it fits.
    r = max(1, int(round(max_width_px / OVERSIZE_SCALE / 2.0)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.morphologyEx(fg, cv2.MORPH_OPEN, disk)
    if not core.any():
        return np.zeros((h, w), dtype=bool)
    # Opening leaves only the core; grow it back so the structure's own edges
    # and corners -- the parts the fine pass actually proposed -- are covered
    # too. r + 1 restores the eroded rim without reaching far past it.
    grow = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * (r + 1) + 1, 2 * (r + 1) + 1)
    )
    big = cv2.dilate(core, grow)
    return cv2.resize(
        big, (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def _interior_point(mask: np.ndarray) -> tuple[float, float]:
    """The point furthest inside ``mask`` -- guaranteed on the object.

    A centroid is not: an L-shaped or curved silhouette puts its centroid in
    the notch, and ``_segment`` only tolerates a hint 10px off its component
    before refusing. The distance transform's peak is always interior.
    """
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    _mn, _mx, _mnloc, mxloc = cv2.minMaxLoc(dt)
    return float(mxloc[0]), float(mxloc[1])


def propose(
    frame: np.ndarray,
    calib: Calibration,
    *,
    marker_xy: Sequence[tuple[float, float]] = (),
    min_blob_px: float = MIN_BLOB_PX,
    max_blob_px: float = MAX_BLOB_PX,
) -> list[Proposal]:
    """Candidate grasp points for everything on the desk that is not the desk.

    Ordered largest blob first, so a caller that truncates keeps the most
    substantial objects. No measurement, no calibration beyond the desk mask --
    :func:`discover` is what turns these into positions.
    """
    if frame.size == 0 or min(frame.shape[:2]) < BG_MEDIAN_KERNEL_PX:
        return []

    dev = desk_deviation(frame)
    keep = desk_pixel_mask(frame, calib)
    if marker_xy:
        paper = marker_paper_mask(frame, calib, marker_xy)
        if paper is not None:
            paper = cv2.dilate(
                paper.astype(np.uint8),
                np.ones((EXCLUDE_DILATE_PX, EXCLUDE_DILATE_PX), np.uint8),
            ).astype(bool)
            keep &= ~paper
    if not keep.any():
        return []

    # Otsu over desk pixels only. Including the wall would put the split
    # between "wall" and "desk" and call every object background.
    d8 = np.clip(dev, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8[keep], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = ((dev > max(MIN_DEVIATION, float(otsu))) & keep).astype(np.uint8)

    # Same morphology as _segment: close pinholes, then drop speckle.
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    too_big = oversize_mask(frame, keep)

    n, labels, stats, _cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out: list[Proposal] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not min_blob_px <= area <= max_blob_px:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        sub = labels[y0 : y0 + h, x0 : x0 + w] == i
        hx, hy = _interior_point(sub)
        px, py = x0 + hx, y0 + hy
        if too_big[int(py), int(px)]:
            # A corner of something the background estimate swallowed.
            continue
        out.append(Proposal(px=px, py=py, area_px=area))
    out.sort(key=lambda p: -p.area_px)
    return out


def discover(
    frame: np.ndarray,
    calib: Calibration,
    *,
    marker_xy: Sequence[tuple[float, float]] = (),
    label: str = "object",
    require_work_region: bool = True,
    max_objects: int = 24,
) -> tuple[list[LocatedObject], list[str]]:
    """Measure every proposal that survives. Returns (objects, rejections).

    Rejections are kept and returned rather than swallowed: on this desk most
    of them are the honest answer (a cable is not graspable, the arm's own body
    is not an object), and a caller tuning the enumerator needs to see why a
    thing it expected is missing. They are prose, in the same style as an
    entity's ``reason``.

    ``label`` is deliberately generic. Naming is not this module's job -- a VLM
    or Grounding DINO can label an ``obj_N`` after the fact, and the geometry
    does not depend on getting the noun right.
    """
    proposals = propose(frame, calib, marker_xy=marker_xy)
    found: list[LocatedObject] = []
    rejected: list[str] = []

    for p in proposals[:max_objects]:
        try:
            obj = measure(frame, p.px, p.py, calib, label, marker_xy=marker_xy)
        except LocateError as exc:
            rejected.append(f"({p.px:.0f},{p.py:.0f}) {p.area_px:.0f}px2: {exc}")
            continue
        if require_work_region and not in_work_region(obj.x, obj.y, calib):
            rejected.append(
                f"({p.px:.0f},{p.py:.0f}): measured ({obj.x:.0f},{obj.y:.0f}) "
                "is outside the work region"
            )
            continue
        near = next(
            (o for o in found if dist_mm(o.x, o.y, obj.x, obj.y) < MERGE_MM), None
        )
        if near is not None:
            # Two blobs of one object (a shaft split by a highlight, say).
            # Keep the larger measurement; it saw more of the silhouette.
            if obj.long_mm > near.long_mm:
                found[found.index(near)] = obj
            continue
        found.append(obj)

    return found, rejected
