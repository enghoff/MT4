#!/usr/bin/env python3
"""Measure the camera's parallax geometry (nadir + lens height) from the arm.

The trajectory overlay (``Calibration.robot_to_pixel``) needs to know where a
point at height ``z`` lands in the image. For a pinhole camera that is pure
geometry: a point at height ``h`` above the table projects, through the flat
table homography, onto the table-plane intersection of its camera ray, which
sits radially outward from the camera *nadir* (the robot XY directly under the
lens) by ``cam_height / (cam_height - h)``. So two numbers fully describe it:
the nadir ``cam_xy_robot`` and the lens height ``cam_height_mm``.

The scene camera does not have to be overhead, and on an oblique mount the
nadir lands far off to one side, well outside the workspace. Raising the TCP
then shifts its image a lot, and getting these two numbers right is what makes
the overlay track the arm instead of drawing the trajectory far too low.

Rather than have a human measure the lens position (a prompted
``cam_height_mm`` plus a guessed nadir), this derives BOTH from
vision + the arm, like every other calibration here: grip a cube, hover it at
a column of known heights over several XY spread across the desk, detect where
it lands, and fit the radial model to the apparent-vs-true offsets. The
recovered lens height comes out within a couple mm of a tape measure, and the
fit tracks the arm to ~10px near the workspace center; the columns span wide
+/-Y so the fit stays honest toward the frame edges too (the one place the
residual grows, where lens barrel distortion the pinhole model can't express
starts to bite).

Bias-free by construction: the constant grip/detection offset (blob centroid
vs TCP) is the SAME at every height in a column, so fitting the offset's
*growth with height* cancels it -- unlike the cube-top map, which bakes that
bias in and cannot serve the arm.

Robust to this rig's two nuisances: the black gripper claw self-occludes a low
held cube from the oblique camera (so a column is tracked through whatever
heights ARE visible, skipping the occluded ones, rather than demanding the
lowest rung), and the held cube is told apart from desk cubes by elimination
(the desk cubes are photographed before the grip; the held one is the same
color that ISN'T sitting where a desk cube was).

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
    update_calibration,
)
from mt4_vision.camera import capture_frame
from mt4_vision.detect import detect_cubes
from mt4_vision.pickplace import home_arm, pick, retreat_for_camera
from mt4_vision.workspace import MAX_REACH_MM, is_mp_reachable_xy

# Probe columns (robot XY). Spread in BOTH axes: the y-spread pins the nadir's
# y (a +y and a -y column give opposite-sign vertical parallax that triangulate
# it), and the shape of offset-vs-height pins the lens height (hence the nadir
# radius) largely independent of XY. The wide +/-Y columns are what keep the
# fit honest toward the frame edges; the central ones anchor the middle. Kept
# at a modest radius so most heights stay reachable, and away from the base so
# the forearm does not occlude the held cube from the oblique camera.
DEFAULT_COLUMNS = [
    (200.0, -60.0),
    (200.0, 60.0),
    (170.0, 0.0),
    (230.0, -30.0),
    (170.0, 150.0),
    (170.0, -150.0),
    (150.0, 200.0),
    (150.0, -200.0),
    (200.0, 120.0),
    (200.0, -120.0),
]
# Heights above the table to sample per column (mm), low to high. Higher rungs
# probe the regime where parallax (and any residual) is largest, but also where
# reach/occlusion tend to drop out -- fine, missing rungs are simply skipped.
DEFAULT_HEIGHTS = [8, 28, 48, 68, 88, 108, 128, 148]
# A detected red is the same DESK cube (not the probe) if its mapped XY is
# within this of a pre-grip desk position.
STATIC_TOL_MM = 35.0
# First visible rung of a column locks onto the probe as the nearest non-desk
# red to the model's predicted pixel, within this radius (generous: the current
# model is what we are refining, off by tens of px toward the edges).
LOCK_RADIUS_PX = 150.0
# Later rungs follow the previous rung's pixel by continuity within this radius;
# if that misses (a skipped/occluded rung created a gap) re-acquire against the
# model prediction within the wider radius.
TRACK_RADIUS_PX = 120.0
REACQUIRE_RADIUS_PX = 170.0
# Keep a column only if it yielded at least this many rungs.
MIN_RUNGS_PER_COL = 3
SETTLE_S = 0.35
# Refuse to write a fit worse than this (mm rms of the radial model against the
# measured apparent offsets) -- a mis-tracked sweep should not poison the
# overlay. A wide-Y fit is inherently distortion-limited (the oblique lens
# barrel-distorts the frame edges, which no single pinhole radial model can
# express), so ~20-30mm rms is expected and fine here; only a truly broken
# sweep (latched a wrong blob, absurd nadir) lands well past this.
MAX_ACCEPT_RMS_MM = 35.0


def _reds(frame, calib, color: str):
    return [c for c in detect_cubes(frame, calib) if c.color == color]


def _is_desk_cube(c, desk_xy: list[tuple[float, float]]) -> bool:
    if c.x is None or c.y is None:
        return False
    return any(
        math.hypot(float(c.x) - sx, float(c.y) - sy) <= STATIC_TOL_MM
        for sx, sy in desk_xy
    )


def _nearest_to(cands, px: float, py: float, radius: float):
    best = None
    best_d = radius
    for c in cands:
        d = math.hypot(c.px - px, c.py - py)
        if d < best_d:
            best_d = d
            best = c
    return best


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


def sweep_column(
    client, calib, cx, cy, heights, color, desk_xy, *, speed
) -> list[dict]:
    """Track a held cube up one vertical column; return per-rung records.

    Skips rungs where the probe is occluded or unreachable; identifies the
    probe by elimination against ``desk_xy`` so it never latches a desk cube.
    """
    tz = calib.table_z
    z_top = max(heights[-1] + tz, calib.safe_z)
    try:
        client.move_to(cx, cy, z_top, speed_us=speed, j4="wrist")
    except Mt4ClientError as exc:
        print(f"  column ({cx:.0f},{cy:.0f}) unreachable: {exc}")
        return []
    out: list[dict] = []
    prev: tuple[float, float] | None = None
    for h in heights:
        try:
            client.move_to(cx, cy, tz + h, speed_us=speed, j4="hold")
        except Mt4ClientError:
            break  # higher rungs only get less reachable
        time.sleep(SETTLE_S)
        frame = capture_frame()
        cands = [c for c in _reds(frame, calib, color) if not _is_desk_cube(c, desk_xy)]
        if not cands:
            continue  # occluded at this height -- skip, keep the column
        pred = calib.robot_to_pixel(cx, cy, tz + h)
        if prev is None:
            det = _nearest_to(cands, pred[0], pred[1], LOCK_RADIUS_PX)
        else:
            det = _nearest_to(cands, prev[0], prev[1], TRACK_RADIUS_PX) or _nearest_to(
                cands, pred[0], pred[1], REACQUIRE_RADIUS_PX
            )
        if det is None:
            continue
        A = calib.pixel_to_robot(det.px, det.py)
        out.append(dict(cx=cx, cy=cy, h=float(h), px=det.px, py=det.py, A_x=A[0], A_y=A[1]))
        prev = (det.px, det.py)
    client.move_to(cx, cy, z_top, speed_us=speed, j4="hold")
    if len(out) < MIN_RUNGS_PER_COL:
        print(f"  column ({cx:.0f},{cy:.0f}): only {len(out)} rung(s) -- dropped")
        return []
    print(f"  column ({cx:.0f},{cy:.0f}): {len(out)} rungs")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate camera nadir + height from the arm")
    ap.add_argument("--port", default=None)
    ap.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    ap.add_argument("--color", default="red", help="probe cube color (default red)")
    ap.add_argument("--dry-run", action="store_true", help="fit and report, do not write the calibration")
    ap.add_argument("--dump", default=None, help="also write the raw per-rung records to this JSON (for offline analysis)")
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

        # Photograph the desk before gripping: every reachable probe-color cube
        # here is a DESK cube; the one we lift becomes "the red that moved".
        frame = capture_frame()
        desk = _reds(frame, calib, args.color)
        desk_xy_all = [(float(c.x), float(c.y)) for c in desk if c.x is not None]
        cand = [
            c for c in desk
            if c.x is not None
            and is_mp_reachable_xy(float(c.x), float(c.y))
            and math.hypot(float(c.x), float(c.y)) <= MAX_REACH_MM
        ]
        if not cand:
            print(f"No reachable {args.color} cube to use as a probe", file=sys.stderr)
            return 1
        cand.sort(key=lambda c: math.hypot(float(c.x) - 200.0, float(c.y) + 120.0))
        g = cand[0]
        grip_xy = (float(g.x), float(g.y))
        # The lifted cube's own former spot is no longer a desk cube.
        desk_xy = [
            s for s in desk_xy_all
            if math.hypot(s[0] - grip_xy[0], s[1] - grip_xy[1]) > STATIC_TOL_MM
        ]
        print(f"Gripping {args.color} cube at ({grip_xy[0]:.1f},{grip_xy[1]:.1f}); "
              f"{len(desk_xy)} desk cube(s) to exclude")
        pick(client, calib, grip_xy[0], grip_xy[1], yaw_deg=g.yaw_deg)

        columns = [
            (x, y) for (x, y) in DEFAULT_COLUMNS
            if is_mp_reachable_xy(x, y) and math.hypot(x, y) <= MAX_REACH_MM
        ]
        for cx, cy in columns:
            records += sweep_column(client, calib, cx, cy, DEFAULT_HEIGHTS, args.color, desk_xy, speed=speed)
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

    if args.dump:
        import json
        Path(args.dump).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} raw rungs to {args.dump}")

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
    cam_xy_robot = [round(nadir[0], 1), round(nadir[1], 1)]
    cam_height_mm = round(H, 1)
    calib.cam_xy_robot, calib.cam_height_mm = cam_xy_robot, cam_height_mm
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
    # Merge onto the file as it stands NOW, writing only the two numbers this
    # script measured. The sweep above takes minutes, and saving the whole
    # object would revert anything written during it -- which is how these two
    # were lost twice on 2026-08-03. See calib.update_calibration.
    update_calibration(
        Path(args.calib), based_on=calib,
        cam_xy_robot=cam_xy_robot, cam_height_mm=cam_height_mm,
    )
    print(f"Saved cam_xy_robot + cam_height_mm to {args.calib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
