#!/usr/bin/env python3
"""Measure the desk surface and store it as a robot-frame polygon.

Why this exists
---------------
Pick and place need to know where the desk is. Until 2026-08-02 they inferred
it from the convex hull of the calibrated ArUco marker centres, which is not a
statement about the desk at all -- it is a statement about where somebody
happened to tape five pieces of paper. Measured over the whole table, that
proxy admitted 828cm^2 of a surface where the arm can safely work 2278cm^2,
and it rejected cubes that were plainly on the desk and plainly in reach.

What is actually measurable
---------------------------
Only the BACK edge, where the desk meets the wall. The other three sides run
off the camera frame and lie past the arm's 342mm reach in every direction, so
there is nothing out there to measure and nothing that could bind. The stored
polygon therefore has one measured side and three nominal ones placed well
outside the envelope; ``on_table`` is a genuine point-in-polygon test, but on
this rig it is a half-plane in disguise. If the desk is ever replaced with a
smaller one, re-run this and the extra sides become real.

How the edge is found
---------------------
**No surface colour is named anywhere in this file, and none may be added.**
A colour written into the source is a claim about one desk under one light,
and it does not survive either changing. Measured 2026-08-04, a window of hue
8..30 with saturation >= 40 -- fitted to this very desk -- admitted 0.0% of
five clear patches of the same desk under brighter light, which read hue 4
saturation 27. The scan then found only sensor speckle, and seven captures of
the static desk on a fully converged camera (identical hue 4 / saturation 27 /
value 212 from read 21 through read 300) returned 1, 4, 2, 7, 0, 15 and 7
"clean" columns. A polygon fitted to speckle is far worse than no polygon,
because it is silent and it moves the desk toward the arm.

So the desk's appearance is LEARNED from the frame being measured, at pixels
the calibration already proves are desk. The arm can only reach the desk, so a
grid of robot points inside the reach envelope at ``table_z``, mapped through
the table homography, lands on desk by construction. Their robust median and
spread in Lab is the reference, and every later comparison is expressed in
units of that spread, so nothing in the method knows or cares what the surface
is made of.

Comparison is on CHROMA (Lab a and b) wherever chroma will carry it, and the
scan tries that first. Illumination moves L and barely touches a and b, and the
back strip of this desk sits in shade 60 L below its lit middle with its chroma
right to within 1 spread. Judging on L there rejects the strip and reports the
edge 160mm off, at +88mm instead of -72mm -- with 138 agreeing columns over
1000px of frame width at 7mm residual, so nothing downstream would question it.

Lightness is admitted only when chroma is measured to carry nothing: a white
bench under a black curtain has one chroma throughout, and L is the only signal
there is. Which pass was used is reported, and the decision is the ``separation``
number, not a setting.

The boundary is then the top outline of the desk's own CONNECTED REGION, read
off column by column. Nothing is assumed or learned about what stands behind
the desk; its colour, its brightness and whether it is uniform at all stop
mattering, and the only requirement left is that it does not look like the
desk. That requirement is measured rather than assumed -- ``separation`` in
the output is how far the material above the boundary actually sits from the
desk, and below MIN_SEPARATION_SIGMA the script refuses instead of guessing.

Taking the region, rather than testing each column on its own, is what makes
the clutter on a working desk harmless. A cable across the back edge, an ArUco
pad, a cube, the arm's base: each is a hole in the region or a notch out of its
top, and none of them disconnects the surface or moves its outline. Rules that
scan a column independently have to be told how much interruption to tolerate,
and the right answer differs per column -- two cables near this desk's back
edge push a 40px-run rule 100px down the desk across the whole left half of
the frame.

Why the fit is robust rather than least-squares
-----------------------------------------------
The region's top outline is the desk edge in most columns, but not all. Where
the arm and its controller box stand against the back of the desk the outline
follows their lower silhouette instead, and at the frame's left and right
borders it follows the desk's side edges, which are vertical rather than the
horizontal boundary being measured. Those are minority contaminants -- 164 of
320 columns survive on this rig -- so the fit trims against a constant model
before measuring anything.

Residuals of ~11mm are expected and are not slack in the trace, which is
smooth to 0.47px between adjacent columns. The boundary sits far outside the
region the table homography was fitted over, since every marker observation is
well inside the desk, and near the horizon the extrapolation is worth 1.7 to
2.2mm per pixel row. That is what EDGE_MARGIN_MM covers.

The edge is modelled as a CONSTANT x unless the surviving columns genuinely
span the desk (MIN_SLOPE_SPAN_MM). This is not laziness. A least-squares slope
fitted over a 66mm span of y extrapolates to nonsense 300mm away, and over such
a span two runs minutes apart produced slopes of +0.05 and -0.26 -- another way
of saying the slope is unmeasured. Over the +-342mm of reach a plausible true
tilt moves the edge by less than the safety margin, so the constant costs
little where the span is short.

With the surface traced across the frame the span is no longer short, and the
slope becomes a real measurement: five captures 2026-08-04 spanning 750mm of
robot y agree on -0.0586 to -0.0618, repeatable to the third decimal, with the
intercept at -71.7 to -72.6mm against a jog-touch edge of -72.2mm.

"Conservative" means the stored edge sits at the measured boundary point
CLOSEST to the arm (90th percentile of the mapped x, to shrug off outliers),
so a tilt in either direction is absorbed rather than cutting into the wall.
EDGE_MARGIN_MM is then added on top. That margin is doing real work: the
boundary sits close to the image horizon, where a one-pixel error in the row
estimate is worth several millimetres.

Usage
-----
    python calibrate_table_edge.py               # measure, show, write
    python calibrate_table_edge.py --dry-run     # measure and show only
    python calibrate_table_edge.py --save-overlay edge.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mt4_vision.calib import (
    DEFAULT_CALIB_PATH,
    Calibration,
    load_calibration,
    update_calibration,
)
from mt4_vision.camera import capture_frame
from mt4_vision.workspace import MAX_REACH_MM, joint_reachable

# Pulled off the fitted edge toward the arm. Covers the fit residual (~10mm
# worst case) plus the row-estimate sensitivity near the horizon.
EDGE_MARGIN_MM = 25.0
# The polygon's three unmeasured sides, well outside a 350mm reach.
NOMINAL_EXTENT_MM = 500.0
# Speckle smaller than this is removed before the desk region is traced, and
# occluders thinner than this are bridged. Sized from what actually lies on a
# working desk: the cables crossing the back of this one are 10-15px wide in a
# 1280x720 frame.
SPECKLE_KERNEL_PX = 3
OCCLUDER_KERNEL_PX = 21
# Rows above the boundary that the desk/background contrast is measured over.
CONTRAST_BAND_PX = 10
# Rows the desk edge could plausibly occupy. Outside this the mapping is
# either above the horizon or well onto the table.
SEARCH_ROWS = (200, 520)
# Fit a tilted line only when the inliers cover at least this much of the
# robot Y axis. Below it there is no leverage and the slope is noise.
MIN_SLOPE_SPAN_MM = 250.0
# Quantile of the mapped edge x used for the constant model. 0.9 rather than
# 1.0 so a single mis-detected column cannot drag the whole desk inward.
EDGE_QUANTILE = 0.9

# --- Learning the desk's appearance ---------------------------------------
#
# Robot-frame grid the desk reference is sampled from, at ``table_z``. Bounded
# by reach, so every point is somewhere the arm can put a gripper -- which on
# this rig means somewhere there is desk. The inner radius keeps samples out of
# the J1 keep-out, where the arm's own base and controller box stand in the
# image rather than desk.
REFERENCE_GRID_STEP_MM = 20.0
REFERENCE_INNER_RADIUS_MM = 150.0
# Half-width of the patch sampled at each grid pixel. Small enough to sit
# inside a marker's white border or between cables rather than straddling them.
REFERENCE_PATCH_PX = 3
# Samples this far from the running median are not desk -- a marker's black
# ink, a cube, a cable, the arm. Distance is in units of the sample spread
# (see ``_scaled_distance``), so the trim carries to any surface.
REFERENCE_TRIM_SIGMA = 2.5
REFERENCE_TRIM_PASSES = 3
MIN_REFERENCE_SAMPLES = 40
# A pixel counts as desk within this scaled distance of the reference.
DESK_MAX_SIGMA = 3.0
# Per-channel spread floor, in Lab units. A perfectly uniform surface measures
# a spread near zero, which would make every other material astronomically far
# away and every pixel of the surface itself a mismatch.
MIN_CHANNEL_SPREAD = 2.0

# Lab's chroma channels. The preferred pair to compare on, for the reason
# ``Reference`` documents: this desk reads L 161 shaded and 213 lit, one
# surface, and comparing on all three channels finds it in 180 of 320 columns
# where chroma finds it in all 320.
CHROMA_CHANNELS = (1, 2)

# --- Accepting or refusing the measurement --------------------------------
#
# All three must hold, because no one of them separates a measurement from
# noise on its own. A frame whose surface is recognised gives ~164 inlier
# columns spanning ~1000px of frame width at 11.1-11.3mm residual (five
# captures, 2026-08-04). Speckle on a frame whose surface is NOT recognised
# gave 0 to 15 columns clustered in a few hundred px on one side, so a count
# alone cannot decide; a fit contaminated by the arm's own outline measured
# 17.8mm residual over a comparable count and span, so residual is what
# catches that one.
MIN_EDGE_COLUMNS = 60
MIN_EDGE_SPAN_PX = 400
MAX_EDGE_RMS_MM = 15.0
# Below this the desk and whatever is behind it are the same material as far as
# the camera is concerned, and no boundary between them exists to be found.
# Reported either way, so a marginal rig says so instead of guessing.
MIN_SEPARATION_SIGMA = 4.0


@dataclass(frozen=True)
class Reference:
    """One material's measured appearance in this frame, in Lab.

    ``channels`` is which of L, a, b the comparison uses. Chroma alone is
    preferred and tried first, because illumination moves L and barely touches
    a and b: this desk's own back strip sits in shade 60 L below its lit middle
    with its chroma right to within 1 spread, and including L there rejects the
    strip and reports the edge 160mm off. Lightness is admitted only when
    chroma turns out to carry no information -- a white bench under a black
    curtain has one chroma throughout, and L is then the only signal there is.
    """

    centre: np.ndarray  # (3,) median Lab
    spread: np.ndarray  # (3,) robust per-channel spread, floored
    samples: int
    kept: int
    channels: tuple[int, ...] = CHROMA_CHANNELS

    def sigma(self, lab: np.ndarray) -> np.ndarray:
        """Scaled distance from this material, per pixel."""
        sub = list(self.channels)
        return _scaled_distance(lab[..., sub], self.centre[sub], self.spread[sub])

    def __str__(self) -> str:
        how = "chroma" if tuple(self.channels) == CHROMA_CHANNELS else "lightness+chroma"
        return (
            f"Lab ({self.centre[0]:.0f},{self.centre[1]:.0f},{self.centre[2]:.0f}) "
            f"spread ({self.spread[0]:.1f},{self.spread[1]:.1f},{self.spread[2]:.1f}) "
            f"from {self.kept}/{self.samples} samples, matched on {how}"
        )


def _scaled_distance(
    lab: np.ndarray, centre: np.ndarray, spread: np.ndarray
) -> np.ndarray:
    """Euclidean distance in units of each channel's own spread.

    Scale-free by construction: a pale near-neutral laminate and a saturated
    wood both come out as "a few units of their own variation", so no absolute
    colour distance has to be picked for a surface this code has not seen.
    """
    d = (lab.astype(np.float32) - centre) / spread
    return np.sqrt((d * d).sum(axis=-1))


def _spread(samples: np.ndarray, centre: np.ndarray) -> np.ndarray:
    return np.maximum(
        1.4826 * np.median(np.abs(samples - centre), axis=0), MIN_CHANNEL_SPREAD
    )


def _robust_reference(
    samples: np.ndarray, channels: tuple[int, ...] = CHROMA_CHANNELS
) -> Reference:
    """Median Lab and spread of ``samples``, trimming the minority that differ.

    Belonging is decided on ``channels``; the returned spread is measured on all
    three regardless, so a caller can report the lightness range a surface
    covers even when lightness took no part in the decision.
    """
    keep = np.ones(len(samples), dtype=bool)
    sub = list(channels)
    for _ in range(REFERENCE_TRIM_PASSES):
        centre = np.median(samples[keep], axis=0)
        spread = _spread(samples[keep], centre)
        nxt = (
            _scaled_distance(samples[:, sub], centre[sub], spread[sub])
            <= REFERENCE_TRIM_SIGMA
        )
        if nxt.sum() < MIN_REFERENCE_SAMPLES:
            break
        keep = nxt
    centre = np.median(samples[keep], axis=0)
    return Reference(
        centre=centre,
        spread=_spread(samples[keep], centre),
        samples=len(samples),
        kept=int(keep.sum()),
        channels=channels,
    )


def reference_sample_pixels(
    calib: Calibration, width: int, height: int
) -> list[tuple[int, int]]:
    """Pixels the calibration proves are desk: reachable robot points at table_z."""
    out: list[tuple[int, int]] = []
    reach = min(MAX_REACH_MM, NOMINAL_EXTENT_MM)
    steps = np.arange(-reach, reach + 0.1, REFERENCE_GRID_STEP_MM)
    for x in steps:
        for y in steps:
            r = math.hypot(float(x), float(y))
            if r < REFERENCE_INNER_RADIUS_MM or r > MAX_REACH_MM:
                continue
            if not joint_reachable(float(x), float(y), calib.table_z):
                continue
            px, py = calib.robot_to_pixel(float(x), float(y))
            ix, iy = int(round(px)), int(round(py))
            if (
                REFERENCE_PATCH_PX <= ix < width - REFERENCE_PATCH_PX
                and REFERENCE_PATCH_PX <= iy < height - REFERENCE_PATCH_PX
            ):
                out.append((ix, iy))
    return out


def learn_desk_reference(
    frame: np.ndarray, calib: Calibration, channels: tuple[int, ...] = CHROMA_CHANNELS
) -> Reference:
    """Measure the desk's Lab appearance from pixels known to be desk.

    Each sample is the median of a small patch, so a single speck of ink or
    dust cannot move it. The samples are then trimmed against their own
    running median, which is what removes the minority that landed on a
    marker, a cube, a cable or the arm.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    pixels = reference_sample_pixels(calib, width, height)
    if len(pixels) < MIN_REFERENCE_SAMPLES:
        raise ValueError(
            f"only {len(pixels)} desk reference pixels inside the frame "
            f"(need {MIN_REFERENCE_SAMPLES}) -- the table homography and the "
            "frame do not agree on where the desk is"
        )
    r = REFERENCE_PATCH_PX
    samples = np.array(
        [
            np.median(
                lab[iy - r:iy + r + 1, ix - r:ix + r + 1].reshape(-1, 3), axis=0
            )
            for ix, iy in pixels
        ],
        dtype=np.float32,
    )
    return _robust_reference(samples, channels)


@dataclass(frozen=True)
class EdgeScan:
    pixels: np.ndarray  # (n, 2) boundary pixels, one per clean column
    desk: Reference
    columns_scanned: int
    separation: float  # how far the material above the boundary is from the desk
    surface_px: int  # size of the desk region the boundary bounds


def desk_region(
    is_desk: np.ndarray, seeds: list[tuple[int, int]]
) -> np.ndarray:
    """The one connected desk region that the known-desk pixels sit in.

    Taking the region rather than testing columns independently is what makes
    the clutter on a working desk harmless. A cable lying across the back edge,
    an ArUco pad, a cube, the arm's own base: each is a hole in the region or a
    notch out of its top, and neither disconnects the surface or moves its
    outline. Column-at-a-time rules have to be told how much interruption to
    tolerate, and the answer differs per column -- measured 2026-08-04, two
    cables near this desk's back edge push a 40px-run rule 100px down the desk
    across the whole left half of the frame.

    Closing bridges those thin occluders before the region is traced; opening
    drops the speckle that would otherwise seed regions of its own.
    """
    clean = cv2.morphologyEx(
        is_desk.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPECKLE_KERNEL_PX,) * 2),
    )
    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OCCLUDER_KERNEL_PX,) * 2),
    )
    count, labels = cv2.connectedComponents(clean)
    hits = [int(labels[iy, ix]) for ix, iy in seeds if labels[iy, ix] != 0]
    if not hits:
        return np.zeros_like(clean, dtype=bool)
    winner = max(set(hits), key=hits.count)
    return labels == winner


def find_edge_pixels(
    frame: np.ndarray, calib: Calibration, *, step: int = 4
) -> EdgeScan:
    """Boundary pixels along the top of the desk, one per column.

    The desk's own region is traced and its upper outline read off column by
    column, so nothing has to be assumed or learned about whatever stands
    behind the desk. Its colour, its brightness and whether it is uniform at
    all stop mattering -- only that it does not look like the desk.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    seeds = reference_sample_pixels(calib, width, height)
    lo, hi = SEARCH_ROWS
    hi = min(hi, height - 1)
    columns = list(range(0, width, step))

    attempts = [
        _trace(frame, calib, lab, seeds, columns, lo, hi, channels)
        for channels in (CHROMA_CHANNELS, (0, 1, 2))
    ]
    for scan in attempts:
        if scan.separation >= MIN_SEPARATION_SIGMA:
            return scan
    return max(attempts, key=lambda s: s.separation)


def _trace(
    frame: np.ndarray,
    calib: Calibration,
    lab: np.ndarray,
    seeds: list[tuple[int, int]],
    columns: list[int],
    lo: int,
    hi: int,
    channels: tuple[int, ...],
) -> EdgeScan:
    """One pass at tracing the surface, comparing on ``channels``."""
    desk = learn_desk_reference(frame, calib, channels)
    region = desk_region(desk.sigma(lab) <= DESK_MAX_SIGMA, seeds)
    pixels: list[tuple[int, int]] = []
    above: list[np.ndarray] = []
    for px in columns:
        rows = np.nonzero(region[lo:hi, px])[0]
        if not len(rows):
            continue
        py = int(rows[0]) + lo
        if py == lo:  # region already open at the top: no boundary in range
            continue
        pixels.append((px, py))
        above.append(np.median(lab[max(0, py - CONTRAST_BAND_PX):py, px], axis=0))

    return EdgeScan(
        pixels=np.array(pixels, dtype=float).reshape(-1, 2),
        desk=desk,
        columns_scanned=len(columns),
        separation=(
            float(np.median(desk.sigma(np.array(above, dtype=np.float32))))
            if above
            else 0.0
        ),
        surface_px=int(region.sum()),
    )


def fit_edge(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, np.ndarray, str]:
    """Model the edge as ``x = a*y + b``; returns (a, b, inliers, model name).

    Trims outliers against a constant model first -- that is the model with
    enough support to reject on -- then upgrades to a tilted line only when
    the survivors span MIN_SLOPE_SPAN_MM of the robot Y axis. Otherwise a is
    zero and b is the conservative (arm-side) quantile of the survivors.
    """
    keep = np.ones(len(xs), dtype=bool)
    for _ in range(3):
        centre = float(np.median(xs[keep]))
        spread = float(np.quantile(np.abs(xs[keep] - centre), 0.80))
        keep = np.abs(xs - centre) <= max(spread, 5.0)
        if keep.sum() < 4:
            keep = np.ones(len(xs), dtype=bool)
            break

    span = float(ys[keep].max() - ys[keep].min())
    if span >= MIN_SLOPE_SPAN_MM:
        design = np.column_stack([ys[keep], np.ones(int(keep.sum()))])
        sol, *_ = np.linalg.lstsq(design, xs[keep], rcond=None)
        return float(sol[0]), float(sol[1]), keep, "tilted line"
    return (
        0.0,
        float(np.quantile(xs[keep], EDGE_QUANTILE)),
        keep,
        f"constant (y span {span:.0f}mm < {MIN_SLOPE_SPAN_MM:.0f}mm)",
    )


def acceptance_failure(
    scan: EdgeScan, keep: np.ndarray | None = None, rms_mm: float | None = None
) -> str | None:
    """The first acceptance criterion the measurement misses, if any.

    Separation is judged before anything that needs a fit, so a rig whose desk
    and background genuinely look alike is told that, rather than being told it
    found too few columns -- which is true but does not say why.
    """
    if scan.separation < MIN_SEPARATION_SIGMA:
        return (
            f"the desk and the surface behind it differ by only "
            f"{scan.separation:.1f} spreads (need {MIN_SEPARATION_SIGMA:.0f}) -- "
            "no boundary between them can be located by appearance"
        )
    if keep is None or rms_mm is None:
        return (
            f"only {len(scan.pixels)} of {scan.columns_scanned} columns found a "
            "boundary at all, too few to fit -- the boundary is most likely "
            "occluded; park the arm clear of the back of the desk"
        )
    n = int(keep.sum())
    if n < MIN_EDGE_COLUMNS:
        return (
            f"only {n} columns survived the fit (need {MIN_EDGE_COLUMNS}), from "
            f"{len(scan.pixels)}/{scan.columns_scanned} columns that found a "
            "boundary"
        )
    span = float(scan.pixels[keep, 0].max() - scan.pixels[keep, 0].min())
    if span < MIN_EDGE_SPAN_PX:
        return (
            f"the {n} surviving columns cover only {span:.0f}px of frame width "
            f"(need {MIN_EDGE_SPAN_PX}) -- a boundary that short is a local "
            "feature, not the desk edge"
        )
    if rms_mm > MAX_EDGE_RMS_MM:
        return (
            f"fit residual {rms_mm:.1f}mm exceeds {MAX_EDGE_RMS_MM:.0f}mm -- the "
            "surviving columns do not lie on one line"
        )
    return None


def polygon_from_edge(a: float, b: float, *, margin_mm: float) -> list[list[float]]:
    """Desk polygon: the measured back edge plus three nominal far sides."""
    b_safe = b + margin_mm
    e = NOMINAL_EXTENT_MM
    return [
        [a * -e + b_safe, -e],
        [e, -e],
        [e, e],
        [a * e + b_safe, e],
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    ap.add_argument("--margin-mm", type=float, default=EDGE_MARGIN_MM)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save-overlay", type=Path, default=None)
    ap.add_argument(
        "--frame",
        type=Path,
        default=None,
        help="measure a saved image instead of the camera (for diagnosis)",
    )
    args = ap.parse_args()

    calib = load_calibration(args.calib)
    if args.frame is not None:
        frame = cv2.imread(str(args.frame))
        if frame is None:
            print(f"could not read {args.frame}", file=sys.stderr)
            return 1
    else:
        frame = capture_frame(args.camera)
    height, width = frame.shape[:2]

    scan = find_edge_pixels(frame, calib)
    print(f"frame {width}x{height}")
    print(f"  desk {scan.desk}")
    print(f"  desk region traced: {scan.surface_px} px")
    print(
        f"  separation: what sits above the boundary is {scan.separation:.1f} "
        f"spreads from the desk (under {MIN_SEPARATION_SIGMA:.0f} and no "
        "boundary is findable)"
    )
    print(f"  {len(scan.pixels)}/{scan.columns_scanned} columns found a boundary")
    if len(scan.pixels) < 4:
        print(f"\nRefusing to write: {acceptance_failure(scan)}.", file=sys.stderr)
        return 1

    pix = scan.pixels
    robot = np.array(
        [calib.pixel_to_robot(px, py, on_cube_top=True) for px, py in pix]
    )
    a, b, keep, model = fit_edge(robot[:, 0], robot[:, 1])
    residual = robot[keep, 0] - (a * robot[keep, 1] + b)
    rms = float(residual.std())

    print(f"\nback edge (robot frame): x = {a:+.4f}*y {b:+.1f}   [{model}]")
    print(
        f"  inliers {int(keep.sum())}/{len(robot)}   "
        f"measured x range {robot[keep, 0].min():.1f} .. {robot[keep, 0].max():.1f}mm   "
        f"rms {rms:.1f}mm"
    )
    print(f"  y span of inliers: {robot[keep, 1].min():.0f} .. {robot[keep, 1].max():.0f}mm")
    print(f"  frame width covered: {pix[keep, 0].min():.0f} .. {pix[keep, 0].max():.0f}px")
    print(f"  margin applied: {args.margin_mm:.0f}mm toward the arm")

    if args.save_overlay is not None:
        vis = frame.copy()
        for (px, py), k in zip(pix, keep):
            cv2.circle(vis, (int(px), int(py)), 2, (0, 255, 0) if k else (0, 0, 255), -1)
        cv2.imwrite(str(args.save_overlay), vis)
        print(f"  overlay written to {args.save_overlay}")

    problem = acceptance_failure(scan, keep, rms)
    if problem is not None:
        print(f"\nRefusing to write: {problem}.", file=sys.stderr)
        print(
            "A desk polygon fitted to noise is worse than none -- it is silent, "
            "and it moves the desk toward the arm.",
            file=sys.stderr,
        )
        return 1

    for y in (-300, -150, 0, 150, 300):
        print(f"    at y={y:5d}: edge x={a * y + b:7.1f}  usable from x={a * y + b + args.margin_mm:7.1f}")

    polygon = polygon_from_edge(a, b, margin_mm=args.margin_mm)
    poly = np.array(polygon, dtype=np.float32)

    # What the arm loses to this edge: the outermost bearing whose full reach
    # still lands on the desk.
    print("\n  reach ring vs the stored edge:")
    for bearing in range(75, 121, 5):
        for sign in (1, -1):
            th = math.radians(sign * bearing)
            x, y = MAX_REACH_MM * math.cos(th), MAX_REACH_MM * math.sin(th)
            inside = cv2.pointPolygonTest(poly, (float(x), float(y)), True)
            state = "on desk " if inside >= 0 else "off desk"
            print(f"    bearing {sign * bearing:4d}: {state} by {abs(inside):5.0f}mm")

    reachable_lost = sum(
        1
        for x in np.arange(-360, 360.1, 5.0)
        for y in np.arange(-360, 360.1, 5.0)
        if joint_reachable(float(x), float(y), calib.table_z)
        and cv2.pointPolygonTest(poly, (float(x), float(y)), False) < 0
    )
    print(f"\n  reachable table area past the edge: {reachable_lost * 25 / 100:.0f}cm^2")

    if args.dry_run:
        print("\n--dry-run: calibration not written")
        return 0

    # Only the two fields this script measures, merged onto the current file
    # -- see calib.update_calibration. The probe above takes minutes.
    update_calibration(
        Path(args.calib), based_on=calib,
        table_polygon_robot=polygon,
        frame_size_px=[int(width), int(height)],
    )
    print(f"\nwrote table_polygon_robot and frame_size_px to {args.calib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
