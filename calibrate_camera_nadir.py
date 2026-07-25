#!/usr/bin/env python3
"""Measure the camera's parallax geometry (nadir + lens height) from the arm.

The trajectory overlay (``Calibration.robot_to_pixel``) needs to know where a
point at height ``z`` lands in the image. For a pinhole camera that is pure
geometry: a point at height ``h`` above the table projects, through the flat
table homography, onto the table-plane intersection of its camera ray, which
sits radially outward from the camera *nadir* (the robot XY directly under the
lens) by ``cam_height / (cam_height - h)``. So two numbers fully describe it:
the nadir ``cam_xy_robot`` and the lens height ``cam_height_mm``.

On this rig the overhead camera is steeply *oblique* -- the nadir is far off
to one side, well outside the workspace -- so raising the TCP shifts its image
a lot, and getting these two numbers right is what makes the overlay track the
arm instead of drawing the trajectory far too low.

Rather than have a human measure the lens position (the earlier approach: a
prompted ``cam_height_mm`` plus a guessed nadir), this derives BOTH from
vision + the arm, like every other calibration here: grip a cube, hover it at
a column of known heights over several XY, detect where it lands, and fit the
radial model to the apparent-vs-true offsets. The recovered lens height comes
out within a couple mm of a tape measure, and the fit tracks the arm to
~10px across the workspace.

Bias-free by construction: the constant grip/detection offset (blob centroid
vs TCP) is the SAME at every height in a column, so fitting the offset's
*growth with height* cancels it -- unlike the cube-top map, which bakes that
bias in and cannot serve the arm.

Run (arm + camera free, a cube or three on the desk):

    python calibrate_camera_nadir.py

No human interaction; the arm sweeps its own probe cube.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import (
    DEFAULT_CALIB_PATH,
    CalibrationError,
    load_calibration,
)
from mt4_vision.camera import capture_frame
from mt4_vision.detect import detect_cubes
from mt4_vision.pickplace import home_arm, pick, retreat_for_camera
from mt4_vision.workspace import MAX_REACH_MM, is_mp_reachable_xy

# Probe columns (robot XY). Spread in BOTH axes: the y-spread pins the nadir's
# y (a +y and a -y column give opposite-sign vertical parallax that triangulate
# it), and the shape of offset-vs-height pins the lens height (hence the nadir
# radius) largely independent of XY. Kept at a modest radius so every height in
# the sweep stays reachable, and away from the base so the forearm does not
# occlude the held cube from the oblique camera.
DEFAULT_COLUMNS = [
    (200.0, -60.0),
    (200.0, 60.0),
    (170.0, 0.0),
    (230.0, -30.0),
    (185.0, 90.0),
    (185.0, -90.0),
]
# Heights above the table to sample per column (mm). Start just off the table
# (parallax ~0, so the flat map places the anchor almost exactly -- a reliable
# lock) and rise in even steps. The top few often drop out to occlusion; the
# fit only needs a handful of clean rungs per column.
DEFAULT_HEIGHTS = [8, 28, 48, 68, 88, 108]
# Anchor: the lowest rung must detect a cube within this pixel radius of the
# flat prediction, and its back-projected XY must be within this many mm of the
# column XY -- else a nearby desk cube was grabbed, so skip the column.
ANCHOR_RADIUS_PX = 50.0
ANCHOR_XY_TOL_MM = 25.0
# Continuity: each higher rung is matched to the nearest same-color blob within
# this radius of the PREVIOUS rung's pixel, so the track cannot jump to a
# static desk cube. Stops the column on the first miss (occlusion).
TRACK_RADIUS_PX = 95.0
SETTLE_S = 0.35
# Refuse to write a fit worse than this (mm rms of the radial model against the
# measured apparent offsets) -- a bad sweep should not poison the overlay.
MAX_ACCEPT_RMS_MM = 20.0


def _reds(frame, calib, color: str):
    return [c for c in detect_cubes(frame, calib) if c.color == color]


def _nearest(frame, calib, color: str, px: float, py: float, radius: float):
    best = None
    best_d = radius
    for c in _reds(frame, calib, color):
        d = math.hypot(c.px - px, c.py - py)
        if d < best_d:
            best_d = d
            best = c
    return best


def _flat_pixel(calib, x: float, y: float) -> tuple[float, float]:
    m0 = np.linalg.inv(np.array(calib.homography, dtype=np.float64))
    w = m0 @ np.array([x, y, 1.0])
    return float(w[0] / w[2]), float(w[1] / w[2])


def fit_nadir_height(records: list[dict], calib) -> tuple[tuple[float, float], float, float]:
    """Fit ``offset = (P - nadir) * h/(H - h)`` to apparent-vs-true offsets.

    Grid-search the lens height ``H`` (the offset shape is convex in ``H``,
    largely independent of the nadir); at each ``H`` the nadir is a linear
    weighted least squares. Returns (nadir, H, robot_rms_mm).
    """
    P = np.array([[r["cx"], r["cy"]] for r in records], dtype=np.float64)
    off = np.array(
        [[r["A_x"] - r["cx"], r["A_y"] - r["cy"]] for r in records],
        dtype=np.float64,
    )
    hh = np.array([float(r["h"]) for r in records], dtype=np.float64)

    def at_H(H: float):
        g = hh / (H - hh)
        s2 = float(np.sum(g * g))
        if s2 < 1e-12:
            return (0.0, 0.0), 1e18
        nx = float(np.sum(g * (g * P[:, 0] - off[:, 0])) / s2)
        ny = float(np.sum(g * (g * P[:, 1] - off[:, 1])) / s2)
        pred = (P - np.array([nx, ny])) * g[:, None]
        rms = float(np.sqrt(np.mean(np.sum((pred - off) ** 2, axis=1))))
        return (nx, ny), rms

    best = None
    for H in np.arange(max(160.0, hh.max() + 6.0), 2000.0, 2.0):
        n, rms = at_H(H)
        if best is None or rms < best[2]:
            best = (H, n, rms)
    H, n, rms = best
    return n, float(H), rms


def sweep_column(client, calib, cx, cy, heights, color, *, speed) -> list[dict]:
    """Track a held cube up one vertical column; return per-rung records."""
    tz = calib.table_z
    z_top = max(heights[-1] + tz, calib.safe_z)
    try:
        client.move_to(cx, cy, z_top, speed_us=speed, j4="wrist")
        client.move_to(cx, cy, tz + heights[0], speed_us=speed, j4="hold")
    except Mt4ClientError as exc:
        print(f"  column ({cx:.0f},{cy:.0f}) unreachable: {exc}")
        return []
    time.sleep(SETTLE_S + 0.05)
    frame = capture_frame()
    ax, ay = _flat_pixel(calib, cx, cy)
    det = _nearest(frame, calib, color, ax, ay, ANCHOR_RADIUS_PX)
    if det is None:
        print(f"  column ({cx:.0f},{cy:.0f}): no anchor near ({ax:.0f},{ay:.0f}) -- skip")
        return []
    A = calib.pixel_to_robot(det.px, det.py)
    if math.hypot(A[0] - cx, A[1] - cy) > ANCHOR_XY_TOL_MM:
        print(f"  column ({cx:.0f},{cy:.0f}): anchor is a stray cube -- skip")
        return []
    out = [dict(cx=cx, cy=cy, h=float(heights[0]), px=det.px, py=det.py, A_x=A[0], A_y=A[1])]
    prev = (det.px, det.py)
    print(f"  column ({cx:.0f},{cy:.0f}) locked at h={heights[0]}mm")
    for h in heights[1:]:
        try:
            client.move_to(cx, cy, tz + h, speed_us=speed, j4="hold")
        except Mt4ClientError as exc:
            print(f"    h={h}: unreachable ({exc}); stop column")
            break
        time.sleep(SETTLE_S)
        frame = capture_frame()
        det = _nearest(frame, calib, color, prev[0], prev[1], TRACK_RADIUS_PX)
        if det is None:
            print(f"    h={h}: lost track (occlusion); stop column")
            break
        A = calib.pixel_to_robot(det.px, det.py)
        out.append(dict(cx=cx, cy=cy, h=float(h), px=det.px, py=det.py, A_x=A[0], A_y=A[1]))
        prev = (det.px, det.py)
    client.move_to(cx, cy, z_top, speed_us=speed, j4="hold")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate camera nadir + height from the arm")
    ap.add_argument("--port", default=None)
    ap.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    ap.add_argument("--color", default="red", help="probe cube color (default red)")
    ap.add_argument("--dry-run", action="store_true", help="fit and report, do not write the calibration")
    args = ap.parse_args()

    try:
        calib = load_calibration(Path(args.calib))
    except CalibrationError as exc:
        print(exc, file=sys.stderr)
        return 1

    speed = calib.travel_speed_us
    tz = calib.table_z
    client = Mt4Client() if not args.port else Mt4Client(port=args.port)
    records: list[dict] = []
    grip_xy: tuple[float, float] | None = None
    try:
        client.ensure_connected()
        if not client.get_status().homed:
            print("Homing first...")
            home_arm(client)
        st = client.get_status()
        client.move_to(st.tcp.x, st.tcp.y, max(st.tcp.z, calib.safe_z), speed_us=speed, j4="wrist")

        frame = capture_frame()
        cand = [
            c for c in _reds(frame, calib, args.color)
            if c.x is not None
            and is_mp_reachable_xy(float(c.x), float(c.y))
            and math.hypot(float(c.x), float(c.y)) <= MAX_REACH_MM
        ]
        if not cand:
            print(f"No reachable {args.color} cube to use as a probe", file=sys.stderr)
            return 1
        # Grip the one nearest a mid-desk anchor -- likely isolated, and the
        # sweep columns fan out from there.
        cand.sort(key=lambda c: math.hypot(float(c.x) - 200.0, float(c.y) + 120.0))
        g = cand[0]
        grip_xy = (float(g.x), float(g.y))
        print(f"Gripping {args.color} cube at ({grip_xy[0]:.1f},{grip_xy[1]:.1f})")
        pick(client, calib, grip_xy[0], grip_xy[1], yaw_deg=g.yaw_deg)

        columns = [
            (x, y) for (x, y) in DEFAULT_COLUMNS
            if is_mp_reachable_xy(x, y) and math.hypot(x, y) <= MAX_REACH_MM
        ]
        for cx, cy in columns:
            records += sweep_column(client, calib, cx, cy, DEFAULT_HEIGHTS, args.color, speed=speed)
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        # Always set the cube down and clear the camera before returning.
        try:
            if grip_xy is not None and client.connected:
                client.move_to(grip_xy[0], grip_xy[1], calib.safe_z, speed_us=speed, j4="wrist")
                client.move_to(grip_xy[0], grip_xy[1], tz, speed_us=calib.approach_speed_us, j4="hold")
                client.gripper(calib.grip_open_s)
                time.sleep(0.5)
                client.move_to(grip_xy[0], grip_xy[1], calib.safe_z, speed_us=speed, j4="hold")
                retreat_for_camera(client, calib)
        except Mt4ClientError as exc:
            print(f"(cleanup move failed: {exc})", file=sys.stderr)
        client.close()

    cols_used = sorted({(r["cx"], r["cy"]) for r in records})
    ys = {c[1] for c in cols_used}
    print(f"\nCollected {len(records)} rungs across {len(cols_used)} columns: {cols_used}")
    if len(cols_used) < 2 or len(records) < 6:
        print("Too few usable columns/rungs to fit (need >=2 columns) -- not writing", file=sys.stderr)
        return 1
    if max(ys) - min(ys) < 40.0:
        print("Columns lack y-spread; the nadir's y is unconstrained -- not writing", file=sys.stderr)
        return 1

    nadir, H, rms = fit_nadir_height(records, calib)
    # Pixel error against the measured detections.
    calib.cam_xy_robot = [round(nadir[0], 1), round(nadir[1], 1)]
    calib.cam_height_mm = round(H, 1)
    perr = [
        math.hypot(*(np.array(calib.robot_to_pixel(r["cx"], r["cy"], tz + r["h"])) - np.array([r["px"], r["py"]])))
        for r in records
    ]
    print(f"\nnadir = ({nadir[0]:.1f}, {nadir[1]:.1f})   cam_height = {H:.1f} mm")
    print(f"robot-space rms = {rms:.2f} mm   overlay pixel error mean={np.mean(perr):.1f} max={np.max(perr):.1f}px")

    if rms > MAX_ACCEPT_RMS_MM:
        print(f"Fit rms {rms:.1f}mm exceeds {MAX_ACCEPT_RMS_MM}mm -- refusing to write", file=sys.stderr)
        return 1
    if args.dry_run:
        print("(dry run -- not writing)")
        return 0
    calib.save(Path(args.calib))
    print(f"Saved cam_xy_robot + cam_height_mm to {args.calib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
