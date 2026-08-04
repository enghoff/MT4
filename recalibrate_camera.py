#!/usr/bin/env python3
"""Refit the table-plane camera calibration after the camera moves --
*without* touching the arm, when the robot base and markers are known not
to have moved.

calibrate_vision.py's touch step exists to discover each marker's robot-
frame XY by jogging the TCP onto it -- but if the robot and markers are
unchanged, those XYs (already stored in vision_calibration.json's
raw_marker_observations) are still correct. All the camera move actually
invalidated is the *pixel* side of each correspondence. So this script:

  1. Loads the existing calibration's per-marker robot XYs (no arm motion).
  2. Captures a fresh frame and auto-detects each marker's *current* pixel
     center + corners.
  3. Refits the table-plane homography from (new pixel, old robot) pairs,
     via the same table_fit.fit_table_map() calibrate_vision.py uses.
  4. Backs up the old calibration, then saves the new one -- clearing
     cube_top_homography, since that correction was fit at the *old*
     camera pose and does not carry over. Run calibrate_height.py next
     (that step is already fully automatic -- no human input needed).

If the "robot and markers didn't move" assumption is wrong, this produces a
confidently wrong calibration with no independent way to catch it from
pixels alone -- the consistency check below (does one homography explain
every marker's old robot XY from its new pixel position) is internal, not
proof the assumption holds. An outlier usually *does* mean a marker moved
(or the wrong marker id matched, or its original touch was recorded off)
and is worth investigating before trusting the result. The check uses the
worst of the in-fit and leave-one-out errors: least squares smears a single
bad correspondence across all markers, so the culprit's own in-fit residual
can sit under the threshold while its leave-one-out error -- refit without
it, then measure it -- shows the disagreement at full size.

Usage:
  python recalibrate_camera.py --camera 1
  python recalibrate_camera.py --camera 1 --dry-run   # fit and report only
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from mt4_vision.calib import (
    CalibrationError,
    DEFAULT_CALIB_PATH,
    load_calibration,
)
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, grab_frame, open_camera
from mt4_vision.detect import detect_markers
from mt4_vision.table_fit import fit_table_map
from mt4_vision.workspace import MARKER_DICT

# A refit marker whose old robot XY disagrees with the new camera geometry
# by more than this is more likely a moved marker / id mismatch than noise
# -- same threshold calibrate_vision.py uses for the equivalent touch check.
SUSPECT_RESIDUAL_MM = 25.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument(
        "--output",
        default=None,
        help="where to save the refit calibration (default: overwrite --calib, after backing it up)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fit and print the report, but don't write anything",
    )
    args = parser.parse_args()

    calib_path = Path(args.calib)
    output_path = Path(args.output) if args.output else calib_path

    try:
        prev = load_calibration(calib_path)
    except CalibrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    old_obs = prev.raw_marker_observations or {}
    old_robot = {int(mid): tuple(data["robot"]) for mid, data in old_obs.items()}
    if len(old_robot) < 3:
        print(
            f"error: existing calibration only has {len(old_robot)} marker "
            "robot position(s) recorded -- need >=3. This script refits from "
            "the *existing* touch data; it can't help if there isn't any.",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded {calib_path}: {len(old_robot)} marker(s) with known robot XY: {sorted(old_robot)}")

    cap = open_camera(args.camera)
    try:
        frame = grab_frame(cap)
    finally:
        cap.release()

    detected = {m.marker_id: m for m in detect_markers(frame, MARKER_DICT)}
    matched_ids = sorted(set(old_robot) & set(detected))
    missing = sorted(set(old_robot) - set(detected))
    if missing:
        print(f"Note: marker(s) {missing} not visible in this frame (occluded/out of view?) -- skipped")
    if len(matched_ids) < 3:
        print(
            f"error: only {len(matched_ids)} marker(s) both visible now and in the "
            "old calibration -- need >=3 to refit. Clear whatever's occluding the "
            "others, or reposition the camera so more are visible.",
            file=sys.stderr,
        )
        return 1
    if len(matched_ids) < len(old_robot):
        print(
            f"warning: refitting from {len(matched_ids)} of {len(old_robot)} previously "
            "calibrated markers -- fewer correspondences means a weaker fit, "
            "especially for perspective."
        )

    # Markers flagged exclude_from_fit keep their stored arm-frame robot XY
    # as a *slot coordinate* but stay out of the map fit: marker 1's touch
    # was recorded at the arm's reach limit where arm-frame and camera
    # geometry disagree by ~30mm -- including it drags the whole similarity.
    excluded = sorted(
        mid for mid in matched_ids
        if old_obs.get(str(mid), {}).get("exclude_from_fit")
    )
    fit_ids = [mid for mid in matched_ids if mid not in excluded]
    if excluded:
        print(f"Excluding marker(s) {excluded} from the fit (exclude_from_fit "
              "set); their slots keep the stored arm-frame coordinates")
    if len(fit_ids) < 3:
        print(
            f"error: only {len(fit_ids)} fit-eligible marker(s) after "
            "exclusions -- need >=3",
            file=sys.stderr,
        )
        return 1

    touch_px = {mid: (detected[mid].px, detected[mid].py) for mid in fit_ids}
    touch_robot = {mid: old_robot[mid] for mid in fit_ids}
    # Corners from EVERY visible marker, not just the matched ones -- the
    # bundle's perspective doesn't need robot XYs, only equal-size squares
    # (same superset behavior as calibrate_vision.py's ref_markers).
    marker_corners = {
        mid: m.corners for mid, m in detected.items() if m.corners is not None
    }

    matrix, report = fit_table_map(marker_corners, touch_px, touch_robot)
    print(f"\n{report.kind} fit from markers {fit_ids}")
    if report.corner_rms_px is not None:
        print(
            f"corner-bundle RMS: {report.corner_rms_px}px (~{report.corner_rms_mm}mm; "
            ">1px suggests lens distortion)"
        )
    print(f"Per-marker residual vs. stored robot XY (mm): {report.touch_residuals_mm}")
    if report.touch_loo_mm:
        print(f"Per-marker leave-one-out error (mm): {report.touch_loo_mm}")
    for note in report.notes:
        print(f"NOTE: {note}")

    worst_err = dict(report.touch_residuals_mm)
    for mid, err in report.touch_loo_mm.items():
        worst_err[mid] = max(worst_err.get(mid, 0.0), err)
    suspects = sorted(m for m, e in worst_err.items() if e > SUSPECT_RESIDUAL_MM)
    if suspects:
        print(
            f"\nWARNING: marker(s) {suspects} disagree with the new camera geometry "
            f"by >{SUSPECT_RESIDUAL_MM}mm (worst of in-fit and leave-one-out error)"
        )
        print(
            "  This usually means that marker (or the robot) moved, or its stored "
            "touch was recorded off-center -- inspect before trusting this "
            "calibration. If it moved (or the touch is suspect), re-record that "
            "marker with calibrate_vision.py's touch process."
        )

    if args.dry_run:
        print("\n--dry-run: not writing anything")
        return 1 if suspects else 0

    if prev.cube_top_homography is not None:
        print(
            "\nCube_top_homography cleared -- it was fit at the old camera pose "
            "and does not carry over. Run calibrate_height.py next (fully "
            "automatic, no touching required) before picking cubes; until then, "
            "cube picks fall back to the less accurate table-plane map"
        )
    if prev.cam_xy_robot is not None or prev.cam_height_mm is not None:
        print(
            "Cam_xy_robot/cam_height_mm cleared -- they encode the OLD camera's "
            "position for the parallax fallback, which would misdirect cube "
            "picks at the new pose"
        )

    backup_dir = calib_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{calib_path.stem}_pre_recalibrate_{stamp}{calib_path.suffix}"
    if calib_path.exists():
        shutil.copy2(calib_path, backup_path)
        print(f"\nBacked up previous calibration to {backup_path}")

    # ``replace``, not a fresh ``Calibration``. Building one field by field
    # meant anything not named took its dataclass default, and on 2026-08-03
    # that silently reverted grip_span_s_at_zero_mm (212.3) and
    # grip_span_s_per_mm (1.881) to None -- the jaw-opening model, a property
    # of the gripper, which moving a camera cannot invalidate. Its refusal gate
    # fails open when unmeasured, so nothing complained; the arm simply stopped
    # declining objects too wide for it. Carrying everything and naming the
    # exceptions inverts that: forgetting a field now preserves it.
    new_calib = replace(
        prev,
        homography=matrix,
        cube_top_homography=None,
        # The residual table belongs to the cube-top map being cleared above.
        cube_top_residual=None,
        # Pixel<->robot pairs recorded through the OLD camera pose.
        probe_observations=None,
        bundle_homography=report.bundle_h,
        raw_marker_observations={
            str(mid): {
                # Carry provenance/flags (exclude_from_fit, robot_source, ...)
                # through the rebuild -- dropping them silently re-arms the
                # marker-1 trap on the next refit.
                **{
                    k: v
                    for k, v in old_obs.get(str(mid), {}).items()
                    if k not in ("pixel", "corners", "robot")
                },
                "pixel": [detected[mid].px, detected[mid].py],
                "corners": marker_corners.get(mid),
                "robot": list(old_robot[mid]),
            }
            for mid in matched_ids
        },
        # The parallax-fallback camera pose belongs to the OLD camera position
        # -- carrying it over would actively misdirect cube picks once
        # cube_top_homography is cleared (pixel_to_robot falls back to it).
        cam_xy_robot=None,
        cam_height_mm=None,
        # NOT carried over: color_xy_offset_mm. Those offsets compensate the
        # color-dependent centroid bias (which side faces each HSV mask
        # admits) -- a function of the viewing geometry, so a camera move
        # invalidates them just like the parallax fallback below. Re-measure
        # at the new pose if needed.
        color_xy_offset_mm={},
    )
    # Every loss this script intends, listed. Anything else that would go
    # missing makes save() raise rather than write -- see Calibration.save.
    new_calib.save(
        output_path,
        clearing=(
            "cube_top_homography", "cube_top_residual", "probe_observations",
            "cam_xy_robot", "cam_height_mm", "color_xy_offset_mm",
        ),
    )
    print(f"Saved refit calibration to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
