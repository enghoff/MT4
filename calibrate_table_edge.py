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
Each image column is scanned downward for the first sustained transition into
desk-coloured pixels, and those boundary pixels are mapped through the table
homography into robot frame.

The edge is then modelled as a CONSTANT x, not as a tilted line, unless the
surviving columns genuinely span the desk (MIN_SLOPE_SPAN_MM). This is not
laziness. The arm's column, its controller box, the gripper assembly and the
door frame occlude most of the boundary, so on this rig the clean columns
cover a narrow band -- and a least-squares slope fitted over a 66mm span of y
extrapolates to nonsense 300mm away. Two runs minutes apart produced slopes of
+0.05 and -0.26, which is another way of saying the slope is unmeasured. Over
the +-342mm of reach, a plausible true tilt moves the edge by less than the
safety margin, so a conservative constant is both simpler and more defensible
than a fitted line nobody should trust.

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
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from mt4_vision.calib import (
    DEFAULT_CALIB_PATH,
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
# A column's transition counts only when desk runs this far below it and
# non-desk runs this far above it -- rejects the arm, cables and the box.
WOOD_RUN_PX = 40
WALL_RUN_PX = 25
# Rows the desk edge could plausibly occupy. Outside this the mapping is
# either above the horizon or well onto the table.
SEARCH_ROWS = (200, 520)
MIN_EDGE_COLUMNS = 15
# Fit a tilted line only when the inliers cover at least this much of the
# robot Y axis. Below it there is no leverage and the slope is noise.
MIN_SLOPE_SPAN_MM = 250.0
# Quantile of the mapped edge x used for the constant model. 0.9 rather than
# 1.0 so a single mis-detected column cannot drag the whole desk inward.
EDGE_QUANTILE = 0.9

# Desk surface HSV, and the wall above it.
#
# The wall test looks strict -- it costs every column where the door frame
# rather than the wall sits above the desk. It is load-bearing anyway.
# Relaxed to a plain "not desk-coloured above", the detector finds the lower
# edge of every dark object ON the desk instead: the controller box, the black
# ArUco pads, shadows. Those false edges outnumber the true boundary, so the
# robust trimming keeps them and throws the real edge away, moving the fitted
# desk 150mm toward the arm. Green above is what makes
# a detection mean "wall", and nothing on this desk is green.
WOOD_HSV = ((8, 40, 90), (30, 255, 255))
WALL_HSV = ((60, 0, 0), (105, 255, 150))


def find_edge_pixels(frame: np.ndarray, *, step: int = 4) -> list[tuple[int, int]]:
    """Boundary pixels where non-desk gives way to desk, one per clean column."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    h, s, v = (hsv[:, :, i].astype(int) for i in range(3))
    wood = (
        (h >= WOOD_HSV[0][0]) & (h <= WOOD_HSV[1][0])
        & (s >= WOOD_HSV[0][1]) & (v >= WOOD_HSV[0][2])
    )
    wall = (
        (h >= WALL_HSV[0][0]) & (h <= WALL_HSV[1][0]) & (v <= WALL_HSV[1][2])
    )
    lo, hi = SEARCH_ROWS
    hi = min(hi, height - WOOD_RUN_PX)
    out: list[tuple[int, int]] = []
    for px in range(0, width, step):
        for py in range(max(lo, WALL_RUN_PX), hi):
            # Desk below, sustained; wall above, sustained.
            if (
                wood[py:py + WOOD_RUN_PX, px].mean() > 0.85
                and wall[py - WALL_RUN_PX:py, px].mean() > 0.7
            ):
                out.append((px, py))
                break
    return out


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
    args = ap.parse_args()

    calib = load_calibration(args.calib)
    frame = capture_frame(args.camera)
    height, width = frame.shape[:2]

    pixels = find_edge_pixels(frame)
    print(f"frame {width}x{height}: {len(pixels)} columns with a clean edge")
    if len(pixels) < MIN_EDGE_COLUMNS:
        print(
            f"Only {len(pixels)} clean columns (need >={MIN_EDGE_COLUMNS}). The "
            "boundary is probably occluded -- park the arm clear of the back "
            "of the desk and retry.",
            file=sys.stderr,
        )
        return 1

    pix = np.array(pixels, dtype=float)
    robot = np.array(
        [calib.pixel_to_robot(px, py, on_cube_top=True) for px, py in pix]
    )
    a, b, keep, model = fit_edge(robot[:, 0], robot[:, 1])
    residual = robot[keep, 0] - (a * robot[keep, 1] + b)

    print(f"\nback edge (robot frame): x = {a:+.4f}*y {b:+.1f}   [{model}]")
    print(
        f"  inliers {int(keep.sum())}/{len(robot)}   "
        f"measured x range {robot[keep, 0].min():.1f} .. {robot[keep, 0].max():.1f}mm   "
        f"rms {residual.std():.1f}mm"
    )
    print(f"  y span of inliers: {robot[keep, 1].min():.0f} .. {robot[keep, 1].max():.0f}mm")
    print(f"  margin applied: {args.margin_mm:.0f}mm toward the arm")
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

    if args.save_overlay is not None:
        vis = frame.copy()
        for (px, py), k in zip(pix, keep):
            cv2.circle(vis, (int(px), int(py)), 2, (0, 255, 0) if k else (0, 0, 255), -1)
        cv2.imwrite(str(args.save_overlay), vis)
        print(f"  overlay written to {args.save_overlay}")

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
