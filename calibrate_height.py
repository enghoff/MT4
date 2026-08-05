#!/usr/bin/env python3
"""Calibrate the cube-top-height pixel->robot correction, fully autonomously.

The main calibration (calibrate_vision.py) fits a homography from markers
lying flat on the table -- accurate for anything at table height (confirmed
by goto-marker landing dead-on), but cubes are detected by their top face,
cube_height_mm above the table, and camera parallax shifts that face's pixel
position enough to matter: ~15-19mm of pick error measured directly against
known-accurate marker coordinates.

This script measures that shift directly by placing a cube at a grid of
robot-frame points spread across the reachable workspace and photographing
where it lands -- the ground truth here is the arm's own positioning (we
choose the target and command the arm there directly), not a marker at all,
so unlike marker-only calibration this doesn't need any correspondence to
already-calibrated points.

A first attempt fit only 3 points (the 3 reachable markers) exactly (zero
residual) and made pick accuracy *worse*, not better: unlike marker corner
detection (sub-pixel precise), a cube's color-blob centroid is noisy --
lighting, slight rotation, partial occlusion -- and a 3-point exact affine
has zero redundancy to average that noise out; it just bakes the noise into
the fit. Sampling more points across the workspace gives real least-squares
averaging (and enables a full homography once >=4 points are collected).

No human interaction needed -- the arm places its own calibration probe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from mt4_jog.joints import JOINT_SOFT_MAX_STEPS, JOINT_SOFT_MIN_STEPS
from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import (
    DEFAULT_CALIB_PATH,
    CalibrationError,
    fit_transform,
    load_calibration,
    update_calibration,
    reprojection_errors,
)
from mt4_vision.camera import capture_frame
from mt4_vision.detect import CubeDetection, detect_cubes
from mt4_vision.pickplace import (
    CAMERA_PARK_Z,
    center_placed_cube,
    home_arm,
    near_camera_park,
    pick,
    place,
    retreat_for_camera,
)
from mt4_vision.scene import filter_phantoms
from mt4_vision.workspace import (
    MAX_REACH_MM,
    PLACE_CLEARANCE_MM,
    PLACEMENT_SLOTS,
    is_mp_reachable_xy,
    marker_slots_from_calibration,
)

# Always include calibrated marker centers in the height-probe grid when they
# aren't already covered. Markers 0/1 (x~35-50) sit ~100mm outside the
# PLACEMENT_SLOTS hull; unprobed, cube-top picks there miss by ~26mm
# (measured 2026-07-21).
MARKER_PROBE_COVER_MM = 25.0


def _marker_probe_targets(
    calib, grid: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Reachable marker centers not already near a scheduled probe target."""
    extras: list[tuple[float, float]] = []
    for m in marker_slots_from_calibration(calib):
        if not is_mp_reachable_xy(m.x, m.y):
            continue
        if math.hypot(m.x, m.y) > MAX_REACH_MM:
            continue
        if any(math.hypot(m.x - gx, m.y - gy) < MARKER_PROBE_COVER_MM for gx, gy in grid):
            continue
        extras.append((m.x, m.y))
    return extras

# Grid of robot-frame (x, y) targets spread across the reachable workspace,
# one quadrant/radius at a time -- ground truth is the arm's own positioning,
# not vision, so these don't need to be markers or anywhere special. Radii
# chosen to stay comfortably under MAX_REACH_MM at any angle in the set.
# Ordered to start in the well-lit region near the base and finish with the
# far, dimmer area (a detection failure there then costs one point at the
# end of the chain rather than resetting it at the start).
# Note no point near (100, 0): the homed arm's gripper hangs over that
# region in the camera view, so a cube placed there can't be re-detected.
GRID_POINTS = PLACEMENT_SLOTS + [
    # Probe-only densification of the +y quadrant: the 6-point 2026-07-18
    # collection left 9-12mm similarity residuals there (slowly-varying
    # centroid bias a 4-DOF similarity cannot absorb); >=8 usable points
    # upgrade the fit to bundle+affine, which can. The last point doubles as
    # the holdout, so the validation number measures the marker-3 region
    # picks actually struggled in.
    (140.0, 160.0),
    (215.0, 205.0),
    # Near-base markers 0/1 sit at x~35-52 while every PLACEMENT_SLOTS probe
    # is at x>=140. Cube-top extrapolation there measured ~26mm (2026-07-21
    # place-on-m1: truth (35,266) read as (60,261)). These approach the
    # marker papers without landing on the tags.
    (100.0, -240.0),
    (100.0, 240.0),
]
# How far (px) the placed cube's centroid may land from where it's expected
# and still count as "the probe" -- generous given measured parallax shifts
# were under 15px, but tight enough to reject the arm's own body (which
# reads as a color blob elsewhere in the frame).
MATCH_RADIUS_PX = 120.0
# pick() is open-loop -- no force/current sensing, so a closed gripper isn't
# proof of a grasp. If the cube is still within this many px of where it sat
# before the pick, the grasp almost certainly failed (seen live: several
# "successful" placements were the untouched cube still sitting at its very
# first starting position).
GRASP_FAIL_RADIUS_PX = 30.0
# Vision-bootstrap picks (no arm-known position yet) go through the table
# homography, which reads a cube-top ~this much low in X (height parallax,
# measured 15-19mm with the camera at large +X beyond the desk edge). Once a
# probe has been arm-placed, its position is known exactly and no nudge is
# used -- that is precisely the error this whole calibration will remove.
BOOTSTRAP_NUDGE_X_MM = 17.0
# After placing, the *raw* table-plane pixel->robot estimate of the detected
# position (before any height correction) should land within this many mm of
# the intended target -- parallax is ~15-20mm and the interim table map can
# be off by a few more cm in places; this is a mis-grasp/slip detector, not
# a tight tolerance. Anything further off means the placement didn't land
# where intended and the point should be discarded rather than poisoning
# the fit.
PLACEMENT_SANITY_MM = 100.0
# Reset cruise speed before starting: whatever a prior session left it at
# (e.g. leftover from gripper-timing/APPROACH_SPEED_US testing) could be slow
# enough that a full cross-table pick/place leg exceeds the client's move
# timeout even though the arm completes it fine -- confirmed happening at a
# leftover 2400us during manual testing.
RESET_SPEED_US = 1524
HOME_SETTLE_S = 0.5
# center_placed_cube's ±90° J4 rotation can pin the wrist at a soft limit;
# any subsequent mp move then fails with err mp segment.
J4_SOFT_MARGIN_STEPS = 300


def find_probe(frame, color: str, near_px: tuple[float, float]) -> CubeDetection | None:
    candidates = [c for c in detect_cubes(frame, calibration=None) if c.color == color]
    if not candidates:
        return None
    nx, ny = near_px
    best = min(candidates, key=lambda c: math.hypot(c.px - nx, c.py - ny))
    if math.hypot(best.px - nx, best.py - ny) > MATCH_RADIUS_PX:
        return None
    return best


def _needs_pose_recovery(client: Mt4Client, calib) -> tuple[bool, str]:
    """True when the arm should home/retreat before the next pick/place leg."""
    status = client.get_status()
    j4_steps = status.joints.get("j4", 0)
    if (
        j4_steps >= JOINT_SOFT_MAX_STEPS[3] - J4_SOFT_MARGIN_STEPS
        or j4_steps <= JOINT_SOFT_MIN_STEPS[3] + J4_SOFT_MARGIN_STEPS
    ):
        return True, f"J4 near soft limit ({j4_steps} steps)"
    tcp = status.tcp
    if tcp is not None and (
        tcp.z < calib.safe_z - 5.0
        or tcp.z < CAMERA_PARK_Z - 10.0
        or not near_camera_park(tcp.x, tcp.y)
    ):
        return True, f"TCP stranded at ({tcp.x:.0f},{tcp.y:.0f},{tcp.z:.0f})"
    return False, ""


def _recover_pose(client: Mt4Client, calib, reason: str) -> None:
    print(f"{reason}, re-homing...")
    home_arm(client)
    retreat_for_camera(client, calib)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate cube-top-height parallax")
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument(
        "--holdout", type=int, default=1,
        help="reserve this many grid points to validate the fit instead of using them in it (default 1)",
    )
    parser.add_argument(
        "--probe-at", type=float, nargs=2, default=None, metavar=("X", "Y"),
        help="exact robot XY of a cube the arm previously placed -- skips the "
             "less-accurate vision bootstrap for the first pick",
    )
    parser.add_argument(
        "--avoid", type=float, nargs=2, action="append", default=[],
        metavar=("X", "Y"),
        help="skip grid targets within place clearance of this robot XY "
             "(e.g. a cube that must stay put); repeatable",
    )
    parser.add_argument(
        "--extra-target", type=float, nargs=2, action="append", default=[],
        metavar=("X", "Y"),
        help="additional probe target(s) beyond the built-in grid; repeatable",
    )
    parser.add_argument(
        "--skip-grid", action="store_true",
        help="probe only the --extra-target points (e.g. to densify a region "
             "and --merge with an earlier same-color session)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="include the calibration's stored probe_observations in the fit. "
             "Only sound for same-color, same-camera-pose observations -- "
             "cross-color merges are OFF by ~10px and are refused.",
    )
    parser.add_argument(
        "--reps", type=int, default=1,
        help="placements per grid target (default 1). Release drag moves the "
             "cube 2-6mm per placement in a semi-random direction; extra reps "
             "average it out. With reps > 1 use --holdout 0 (the holdout "
             "split is by observation, so a held-out target would leak into "
             "the fit via its other reps) and validate offline instead.",
    )
    args = parser.parse_args()

    try:
        calib = load_calibration(Path(args.calib))
    except CalibrationError as exc:
        print(exc, file=sys.stderr)
        return 1

    camera_kwargs = {} if args.camera is None else {"index": args.camera}
    client = Mt4Client() if not args.port else Mt4Client(port=args.port)

    try:
        client.ensure_connected()
        status = client.get_status()
        if status.homed:
            print("Already homed")
        else:
            print("Homing first...")
            home_arm(client)  # raises on failure instead of sailing on unhomed
        needs_recovery, reason = _needs_pose_recovery(client, calib)
        if needs_recovery:
            _recover_pose(client, calib, reason)
        status = client.get_status()
        # No-op move at the current pose, purely to reset cruise speed (see
        # RESET_SPEED_US) before the real moves below.
        client.move_to(status.tcp.x, status.tcp.y, status.tcp.z, speed_us=RESET_SPEED_US)

        base_grid = [] if args.skip_grid else list(GRID_POINTS)
        base_grid += [(float(x), float(y)) for x, y in args.extra_target]
        if not args.skip_grid:
            for mx, my in _marker_probe_targets(calib, base_grid):
                print(f"Adding marker center ({mx:.1f},{my:.1f}) to height-probe grid")
                base_grid.append((mx, my))
        grid = [(x, y) for x, y in base_grid if math.hypot(x, y) <= MAX_REACH_MM]
        for ax, ay in args.avoid:
            dropped_targets = [
                (x, y) for x, y in grid
                if math.hypot(x - ax, y - ay) < PLACE_CLEARANCE_MM
            ]
            if dropped_targets:
                print(f"Skipping target(s) {dropped_targets} -- within place "
                      f"clearance of avoided ({ax:.0f},{ay:.0f})")
            grid = [t for t in grid if t not in dropped_targets]
        min_targets = 1 if args.merge else 5
        if len(grid) < min_targets:
            print(f"Only {len(grid)} usable grid points; need >={min_targets}", file=sys.stderr)
            return 1

        markers = marker_slots_from_calibration(calib)
        frame = capture_frame(**camera_kwargs)
        # Phantom-filter the probe pool (keep-out cylinder, area, hull, reach):
        # the arm base's own hardware intermittently reads as small blue blobs
        # inside the keep-out zone, and the grasp-failure rotation below would
        # otherwise eventually send the gripper at one.
        all_blobs = detect_cubes(frame, calib)
        cubes = filter_phantoms(all_blobs, calib)
        dropped = [c for c in all_blobs if c not in cubes]
        if dropped:
            print("Ignoring phantom blob(s): "
                  + ", ".join(f"{c.color}({c.x:.0f},{c.y:.0f})" for c in dropped))
        if not cubes:
            print("No reachable cube in view to use as a height probe", file=sys.stderr)
            return 1
        # Rotate through every reachable cube on repeated grasp failure --
        # the bootstrap map's error varies across the table, so a different
        # cube in a better-mapped spot may grasp fine.
        probe_pool = [(c.color, (c.x, c.y)) for c in cubes]
        if args.probe_at:
            # The arm-known cube must be the probe -- order the pool by
            # distance to it.
            probe_pool.sort(
                key=lambda e: math.hypot(e[1][0] - args.probe_at[0], e[1][1] - args.probe_at[1])
            )
        probe_color, probe_xy = probe_pool[0]
        print(f"Probe candidates: {[(c, (round(x), round(y))) for c, (x, y) in probe_pool]}")
        print(f"Using the {probe_color} cube at ({probe_xy[0]:.1f},{probe_xy[1]:.1f}) first")
        print(f"Sampling {len(grid)} points, holding out last {args.holdout} for validation")

        pixel_pts: list[tuple[float, float]] = []
        robot_pts: list[tuple[float, float]] = []
        probe_colors: list[str] = []

        def keep_probe_observations(*, replace: bool = False) -> None:
            """Persist raw (pixel, robot, color) observations -- on every exit path that
            has any, not just success: a run aborted by a dropped cube already
            paid the arm time for its points, and offline refits can merge
            them with a later session's. Color matters: red- and blue-cube
            centroids of the same physical position measured ~10px apart
            (different HSV masks include different side faces), so
            cross-color merges must model that or stay single-color.

            Failure paths APPEND to the stored archive (replace=False): a
            2-point aborted run must not destroy a prior full collection
            (2026-07-20: an aborted run overwrote 10 good observations with
            2 phantom ones). Only a successful refit replaces the archive
            -- and when --merge was used, pixel_pts already contains the
            stored points, so nothing is lost there either."""
            if not pixel_pts:
                return
            new = [
                {"pixel": [round(p[0], 2), round(p[1], 2)], "robot": list(r), "color": c}
                for p, r, c in zip(pixel_pts, robot_pts, probe_colors)
            ]
            if not replace and calib.probe_observations:
                seen = {json.dumps(o, sort_keys=True) for o in calib.probe_observations}
                new = calib.probe_observations + [
                    o for o in new if json.dumps(o, sort_keys=True) not in seen
                ]
            calib.probe_observations = new
            # Only what this script measures, merged onto the file as it stands
            # now. This closure fires REPEATEDLY through a probe run that takes
            # minutes, and saving the whole object made every one of those
            # writes revert the file to its state when the run started --
            # silently undoing any other calibration finished in between. See
            # calib.update_calibration.
            measured = {"probe_observations": calib.probe_observations}
            if calib.cube_top_homography is not None:
                measured["cube_top_homography"] = calib.cube_top_homography
                measured["cube_top_residual"] = calib.cube_top_residual
            update_calibration(Path(args.calib), based_on=calib, **measured)

        def locate_probe(near_xy: tuple[float, float]) -> CubeDetection | None:
            """Re-detect the probe fresh (never trust a carried-over
            assumption -- avoids compounding drift after any failed
            grasp/placement earlier in the run). Returns the detection;
            find_probe() bypasses the hull filter via calibration=None, which
            leaves CubeDetection.x/.y unset, so robot XY is computed
            separately here via the raw (uncorrected) table homography --
            good enough to re-locate and pick up the probe by.
            """
            frame = capture_frame(**camera_kwargs)
            found = find_probe(frame, probe_color, _inverse_guess(calib, *near_xy))
            if found is not None:
                return found
            candidates = [
                c for c in filter_phantoms(detect_cubes(frame, calib), calib)
                if c.color == probe_color and math.hypot(c.x, c.y) <= MAX_REACH_MM
            ]
            if not candidates:
                return None
            return min(
                candidates,
                key=lambda c: math.hypot(c.x - near_xy[0], c.y - near_xy[1]),
            )

        grasp_fails = 0
        # Once the probe has been arm-placed somewhere, its position is known
        # EXACTLY (the arm put it there) -- vastly better than re-deriving it
        # through the still-uncorrected camera map. Vision only verifies the
        # cube is still where it was left; the pick targets the known coords.
        known_xy: tuple[float, float] | None = args.probe_at
        first_center = True
        for gx, gy in grid:
            needs_recovery, reason = _needs_pose_recovery(client, calib)
            if needs_recovery:
                _recover_pose(client, calib, reason)
            pre_pick = locate_probe(known_xy or probe_xy)
            if pre_pick is None:
                if known_xy is not None:
                    # Not visible but the arm knows where it left it -- e.g.
                    # the homed arm occludes that spot. Pick blind; the
                    # post-place check still validates the data point.
                    print(f"\nTarget ({gx:.1f},{gy:.1f}): probe not visible, picking blind "
                          f"at arm-known ({known_xy[0]:.1f},{known_xy[1]:.1f})")
                    pre_pick_px = None
                    pick_yaw = None
                else:
                    print(f"\nTarget ({gx:.1f},{gy:.1f}): lost track of the probe cube -- aborting", file=sys.stderr)
                    keep_probe_observations()
                    if pixel_pts:
                        print(f"Kept {len(pixel_pts)} collected probe observation(s) "
                              "in the calibration file (cube_top_homography unchanged)")
                    return 1
            else:
                pre_pick_px = (pre_pick.px, pre_pick.py)
                pick_yaw = pre_pick.yaw_deg
            seen_xy = calib.pixel_to_robot(*pre_pick_px) if pre_pick_px else None
            if (
                known_xy is not None
                and seen_xy is not None
                and math.hypot(seen_xy[0] - known_xy[0], seen_xy[1] - known_xy[1])
                > PLACEMENT_SANITY_MM
            ):
                print(f"  Probe not where it was left (seen ~({seen_xy[0]:.0f},{seen_xy[1]:.0f}), "
                      f"expected ({known_xy[0]:.0f},{known_xy[1]:.0f})) -- falling back to vision")
                known_xy = None
            if known_xy is not None:
                probe_xy = known_xy
                origin = "arm-known"
            elif calib.cube_top_homography:
                # A cube-top map (even the crude interim one) already contains
                # the parallax correction -- use it, no nudge.
                probe_xy = calib.pixel_to_robot(*pre_pick_px, on_cube_top=True)
                origin = "vision(cube-top)"
            else:
                probe_xy = (seen_xy[0] + BOOTSTRAP_NUDGE_X_MM, seen_xy[1])
                origin = "vision+nudge"
            print(f"\nTarget ({gx:.1f},{gy:.1f}): probe at "
                  f"({probe_xy[0]:.1f},{probe_xy[1]:.1f}) [{origin}], picking...")

            holding_probe = False
            try:
                pick(client, calib, probe_xy[0], probe_xy[1], yaw_deg=pick_yaw)
                holding_probe = True
                retreat_for_camera(client, calib)
            except Mt4ClientError as exc:
                print(f"  Pick failed ({exc}), skipping this point")
                known_xy = None
                continue

            # Grasp verification: pick() has no force/current sensing, so a
            # closed gripper isn't proof of a grasp -- check the cube is
            # actually gone from where it was. Blind picks (no pre-pick
            # sighting) skip this; the post-place check still gates the data.
            frame = capture_frame(**camera_kwargs)
            still_there = (
                find_probe(frame, probe_color, pre_pick_px) if pre_pick_px else None
            )
            if still_there is not None and math.hypot(
                still_there.px - pre_pick_px[0], still_there.py - pre_pick_px[1]
            ) < GRASP_FAIL_RADIUS_PX:
                grasp_fails += 1
                known_xy = None
                print("  Grasp likely failed (cube still at its start position), skipping")
                if grasp_fails >= 2 and len(probe_pool) > 1:
                    # The bootstrap map may be off where this cube sits --
                    # switch to a different cube in a better-mapped spot.
                    probe_pool.append(probe_pool.pop(0))
                    probe_color, probe_xy = probe_pool[0]
                    grasp_fails = 0
                    print(f"  Switching probe to the {probe_color} cube at "
                          f"({probe_xy[0]:.1f},{probe_xy[1]:.1f})")
                continue
            grasp_fails = 0

            try:
                # base avoidance is the firmware's job now: mp routes around
                # the keep-out cylinder on its own
                place(client, calib, gx, gy, lift_after=False)
                holding_probe = False
                center_placed_cube(
                    client, calib, gx, gy, lift_before_rotate=first_center,
                )
                first_center = False
                retreat_for_camera(client, calib)
            except Mt4ClientError as exc:
                print(f"  Place/center failed ({exc}), skipping this point")
                if holding_probe and known_xy is not None:
                    try:
                        print(f"  Recovering by re-homing and replacing probe at "
                              f"({known_xy[0]:.1f},{known_xy[1]:.1f})")
                        _recover_pose(client, calib, "  Placement failed while holding probe")
                        place(client, calib, known_xy[0], known_xy[1])
                        holding_probe = False
                        retreat_for_camera(client, calib)
                        probe_xy = known_xy
                        continue
                    except Mt4ClientError as recovery_exc:
                        print(f"  Recovery place failed ({recovery_exc})")
                known_xy = None
                continue
            time.sleep(HOME_SETTLE_S)

            frame = capture_frame(**camera_kwargs)
            # Expected pixel is unknown a priori (that's what we're
            # measuring) -- fall back to the table homography's own guess
            # for the search center, which is at worst ~20mm (a few px) off.
            guess_px = _inverse_guess(calib, gx, gy)
            found = find_probe(frame, probe_color, guess_px)
            if found is None:
                print(f"  Could not find {probe_color} probe near ({gx:.1f},{gy:.1f}), skipping")
                known_xy = None
                continue

            raw_x, raw_y = calib.pixel_to_robot(found.px, found.py)
            sanity_err = math.hypot(raw_x - gx, raw_y - gy)
            if sanity_err > PLACEMENT_SANITY_MM:
                print(f"  Detected position ({raw_x:.1f},{raw_y:.1f}) is {sanity_err:.0f}mm "
                      f"from the target -- likely a bad placement, discarding")
                known_xy = None
                continue

            print(f"  Probe detected at pixel ({found.px:.1f},{found.py:.1f}), "
                  f"raw estimate ({raw_x:.1f},{raw_y:.1f}) [{sanity_err:.0f}mm off]")
            pixel_pts.append((found.px, found.py))
            robot_pts.append((gx, gy))
            probe_colors.append(probe_color)
            probe_xy = (gx, gy)
            known_xy = (gx, gy)

            # Extra reps at the same target: re-grip (arm-known, exact) and
            # re-place, photographing each landing -- averages out the 2-6mm
            # release drag that a single placement bakes into the data.
            for _rep in range(max(0, args.reps - 1)):
                try:
                    pick(client, calib, gx, gy)
                    retreat_for_camera(client, calib)
                    place(client, calib, gx, gy, lift_after=False)
                    center_placed_cube(client, calib, gx, gy)
                    retreat_for_camera(client, calib)
                except Mt4ClientError as exc:
                    print(f"  rep failed ({exc}); target stays single-sampled")
                    break
                time.sleep(HOME_SETTLE_S)
                frame = capture_frame(**camera_kwargs)
                found = find_probe(frame, probe_color, _inverse_guess(calib, gx, gy))
                if found is None:
                    print("  rep: probe not redetected; skipping rep")
                    break
                raw_x, raw_y = calib.pixel_to_robot(found.px, found.py)
                if math.hypot(raw_x - gx, raw_y - gy) > PLACEMENT_SANITY_MM:
                    print("  rep: sanity check failed; skipping rep")
                    break
                print(f"  rep: probe at pixel ({found.px:.1f},{found.py:.1f})")
                pixel_pts.append((found.px, found.py))
                robot_pts.append((gx, gy))
                probe_colors.append(probe_color)

        if args.merge and calib.probe_observations:
            stored = calib.probe_observations
            run_colors = set(probe_colors) or {probe_color}
            stored_colors = {o.get("color") for o in stored}
            if stored_colors - run_colors:
                print(
                    f"--merge refused: stored observations are {sorted(stored_colors)} "
                    f"but this run probed {sorted(run_colors)} -- centroids of "
                    "different colors are offset ~10px and would poison the fit",
                    file=sys.stderr,
                )
                keep_probe_observations()
                return 1
            print(f"Merging {len(stored)} stored observation(s) into the fit")
            pixel_pts = [tuple(o["pixel"]) for o in stored] + pixel_pts
            robot_pts = [tuple(o["robot"]) for o in stored] + robot_pts
            probe_colors = [o.get("color", probe_color) for o in stored] + probe_colors

        holdout = args.holdout if args.holdout > 0 else 0
        if len(pixel_pts) - holdout < 4:
            print(
                f"\nonly {len(pixel_pts)} usable point(s) ({holdout} held out); "
                "need >=4 to fit with -- not refitting",
                file=sys.stderr,
            )
            keep_probe_observations()
            return 1

        fit_px = pixel_pts[: len(pixel_pts) - holdout] if holdout else pixel_pts
        fit_rb = robot_pts[: len(robot_pts) - holdout] if holdout else robot_pts

        if calib.bundle_homography:
            # Preferred: reuse the corner bundle's perspective and fit only a
            # low-DOF alignment on top -- a similarity (4 DOF) in the metric
            # plane frame, or an affine (6 DOF) with >=8 points to also absorb
            # the slowly-varying part of the centroid bias. An unconstrained
            # 8-DOF homography from a handful of noisy blob centroids is
            # exactly the overfit that broke the table map; never again.
            import numpy as np

            from mt4_vision.calib import fit_affine
            from mt4_vision.table_fit import _apply_h, fit_similarity_2d

            hb = np.array(calib.bundle_homography)
            src = _apply_h(hb, np.array(fit_px, dtype=np.float64))
            dst = np.array(fit_rb, dtype=np.float64)
            if len(fit_px) >= 8:
                align = np.array(fit_affine([tuple(p) for p in src], [tuple(r) for r in dst]))
                kind = "bundle+affine"
            else:
                align = fit_similarity_2d(src, dst)
                kind = "bundle+similarity"
            matrix = (align @ hb).tolist()
        else:
            matrix, kind = fit_transform(fit_px, fit_rb)

        errors = reprojection_errors(matrix, fit_px, fit_rb)
        print(f"\n{kind} fit from {len(fit_px)} point(s)")
        print(f"In-fit reprojection error (mm): {[round(e, 2) for e in errors]}")

        if holdout:
            held_px = pixel_pts[-holdout:]
            held_rb = robot_pts[-holdout:]
            held_errors = reprojection_errors(matrix, held_px, held_rb)
            print(f"Held-out validation error (mm): {[round(e, 2) for e in held_errors]}")
            print("(This is the number that matters -- in-fit error is expected to look good regardless)")

        calib.cube_top_homography = matrix

        # Residual layer: the per-location mapping error is stable to ~1mm
        # but nonlinear in position -- store the per-target mean residual for
        # smooth interpolation at read time (Calibration._residual_correction).
        import numpy as np

        hmat = np.array(matrix, dtype=np.float64)
        by_target: dict[tuple[float, float], list[tuple[float, float]]] = {}
        for (ppx, ppy), (rx, ry) in zip(fit_px, fit_rb):
            v = hmat @ np.array([ppx, ppy, 1.0])
            by_target.setdefault((rx, ry), []).append(
                (rx - float(v[0] / v[2]), ry - float(v[1] / v[2]))
            )
        calib.cube_top_residual = {
            "points": [[x, y] for (x, y) in by_target],
            "deltas": [
                [float(np.mean([d[0] for d in ds])), float(np.mean([d[1] for d in ds]))]
                for ds in by_target.values()
            ],
            "sigma_mm": 60.0,
            "reg": 0.25,
        }
        post = [
            math.hypot(
                calib.pixel_to_robot(ppx, ppy, on_cube_top=True)[0] - rx,
                calib.pixel_to_robot(ppx, ppy, on_cube_top=True)[1] - ry,
            )
            for (ppx, ppy), (rx, ry) in zip(fit_px, fit_rb)
        ]
        print(f"With residual layer (mm): {[round(e, 2) for e in post]}")

        keep_probe_observations(replace=True)
        print(f"\nSaved cube_top_homography + residual layer ({len(pixel_pts)} "
              f"raw probe observations kept) to {args.calib}")
        return 0
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()


def _inverse_guess(calib, x: float, y: float) -> tuple[float, float]:
    """Rough pixel-space search center for a robot XY: numerically invert
    the table homography (good to a few px; cube-top parallax is <15px)."""
    import numpy as np

    h = np.array(calib.homography, dtype=np.float64)
    h_inv = np.linalg.inv(h)
    v = h_inv @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


if __name__ == "__main__":
    raise SystemExit(main())
