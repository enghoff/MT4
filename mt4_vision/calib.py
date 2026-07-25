"""Camera-to-robot calibration: table-plane homography plus pick heights.

The homography maps camera pixels to robot-frame XY *on the table plane* (the
plane the ArUco markers lie on). It is fit from pixel/robot correspondences
gathered by jogging the arm's TCP to touch each marker center -- this absorbs
the entire camera pose, so no camera intrinsics are needed.

Cubes are detected by their top face, which sits cube_height_mm above the
table plane, so the raw table-plane homography output is off for them by a
camera-parallax amount -- confirmed empirically (see calibrate_height.py) to
be roughly constant across the desk for this camera's mounting, not clearly
radial from a nadir point, so the primary correction is cube_top_homography:
a second homography/affine fit the same way as the table-plane one, but from
correspondences at cube-top height (an object of known height placed exactly
at already-calibrated robot XYs, photographed, its pixel position paired with
that XY). cam_xy_robot/cam_height_mm remain as a fallback radial-scaling
correction for setups without a cube_top_homography.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB_PATH = Path(
    os.environ.get("MT4_VISION_CALIB", Path(__file__).resolve().parent.parent / "vision_calibration.json")
)


class CalibrationError(Exception):
    """Raised when calibration data is missing, unloadable, or degenerate."""


# One-shot per process: a table-plane recalibration clears
# cube_top_homography (correctly -- it was fit at the old camera pose), and
# nothing downstream failed when the refit was skipped; picks just silently
# regained 15-30mm of parallax error. Warn at use time so every entry path
# (recalibrate script, manual JSON edits, backup restores) is covered.
_warned_no_cube_top_correction = False


def _warn_no_cube_top_correction() -> None:
    global _warned_no_cube_top_correction
    if _warned_no_cube_top_correction:
        return
    _warned_no_cube_top_correction = True
    print(
        "WARNING: cube_top_homography is not set (no parallax fallback "
        "either) -- cube positions come from the uncorrected table-plane "
        "map, ~15-30mm of error. Run: python calibrate_height.py "
        "(required again after every table-plane recalibration).",
        file=sys.stderr,
    )


@dataclass
class Calibration:
    # Row-major 3x3 pixel -> robot-XY homography (table plane).
    homography: list[list[float]]
    # Robot-frame Z of the table surface under the camera (mm). Also the TCP Z
    # for gripping a cube sitting on the table -- the gripper straddles the
    # cube's lower faces, so grip height and table height coincide here (the
    # former separate ``pick_z`` was always set equal to this and was folded
    # in).
    table_z: float
    # Travel height between moves (mm). Keep modest -- the arm should stay
    # low over the desk, well inside its envelope.
    safe_z: float
    # Step period (us) for safe_z transits. Must be below the firmware ramp's
    # MP_ACCEL_START_US (1800) to engage accel/decel; 700 is fastest.
    travel_speed_us: int = 700
    # Step period (us) for pick/place descent and table touch -- deliberately
    # slow (>= 1800) so the firmware ramp stays off near the work surface.
    approach_speed_us: int = 2400
    # Gripper S values for this cube size (firmware absolute, 120-285).
    grip_open_s: int = 140
    grip_close_s: int = 240
    # Cube edge length (mm), used for the parallax correction and place height.
    cube_height_mm: float = 20.0
    # Preferred cube-top correction: a second pixel -> robot-XY homography
    # fit directly at cube-top height (see calibrate_height.py) from
    # arm-placed probe cubes -- the arm's commanded place position is the
    # ground truth. None if not yet calibrated.
    cube_top_homography: list[list[float]] | None = None
    # Smooth residual layer over the cube-top map: the per-location mapping
    # error is stable to ~1mm (measured 2026-07-18) but nonlinear in position
    # -- beyond what the affine alignment can express -- so the probe
    # residuals themselves are interpolated: Gaussian-weighted mean of
    # `deltas` at `points` (robot frame, mm), with a regularizer that shrinks
    # the correction to zero away from the probes (no extrapolation
    # pathology). {"points": [[x,y],..], "deltas": [[dx,dy],..],
    # "sigma_mm": 60.0, "reg": 0.25}
    cube_top_residual: dict | None = None
    # Pixel -> metric-table-frame homography from the marker-corner bundle
    # (table_fit.py). Perspective comes from 20 subpixel corners, so this is
    # the trustworthy projective part; the maps above are (similarity .
    # bundle) compositions of it. Stored so cheap low-DOF refits (e.g. the
    # cube-top similarity from probes) can reuse the perspective without
    # re-solving it.
    bundle_homography: list[list[float]] | None = None
    # Raw calibration observations (marker id -> {pixel, corners, robot}),
    # kept so the map can be refit offline without redoing the physical
    # session -- the fitted matrix alone is not invertible back to its data,
    # which cost us dearly once.
    raw_marker_observations: dict | None = None
    # Raw cube-top probe observations from calibrate_height.py (list of
    # {pixel, robot}), kept for the same reason: without them the cube-top
    # fit can be neither refit offline nor outlier-checked.
    probe_observations: list | None = None
    # Fallback parallax correction, used only when cube_top_homography is
    # unset: robot XY directly under the camera and the camera lens height
    # above the table (mm). None disables the correction entirely.
    cam_xy_robot: list[float] | None = None
    cam_height_mm: float | None = None
    # Per-color HSV overrides merged over detect.COLOR_RANGES defaults.
    color_ranges: dict = field(default_factory=dict)
    # Per-color robot-frame XY correction (mm), added to cube-top-mapped
    # detections. The blob centroid is not the top-face center: each color's
    # HSV band admits a different mix of lit/shaded side faces, so the
    # centroid bias is color-dependent -- a map calibrated with one probe
    # color mis-locates the others by a constant few-to-15mm offset
    # (measured 2026-07-18 by arm-placing different colors at one spot).
    color_xy_offset_mm: dict = field(default_factory=dict)
    # When True, vision picks command J4 from CubeDetection.yaw_deg. On by
    # default; assumes firmware ``j4zero`` (``calibrate_j4.py``) so world
    # J4 = 0 means jaws along the arm.
    face_align_picks: bool = True
    # Pixel-space convex hull of the marker centers. Detections outside it
    # are rejected (the arm's orange body and off-desk clutter live there).
    workspace_hull_px: list[list[float]] | None = None

    def pixel_to_robot(self, px: float, py: float, *, on_cube_top: bool = False) -> tuple[float, float]:
        if on_cube_top and self.cube_top_homography:
            h = np.array(self.cube_top_homography, dtype=np.float64)
            v = h @ np.array([px, py, 1.0])
            x, y = float(v[0] / v[2]), float(v[1] / v[2])
            if self.cube_top_residual:
                dx, dy = self._residual_correction(x, y)
                x += dx
                y += dy
            return x, y

        h = np.array(self.homography, dtype=np.float64)
        v = h @ np.array([px, py, 1.0])
        x, y = float(v[0] / v[2]), float(v[1] / v[2])
        if on_cube_top and self.cam_xy_robot and self.cam_height_mm:
            scale = (self.cam_height_mm - self.cube_height_mm) / self.cam_height_mm
            cx, cy = self.cam_xy_robot
            x = cx + (x - cx) * scale
            y = cy + (y - cy) * scale
        elif on_cube_top:
            _warn_no_cube_top_correction()
        return x, y

    def robot_to_pixel(self, x: float, y: float, z: float | None = None) -> tuple[float, float]:
        """Robot XY(Z) -> pixel, height-corrected for camera parallax.

        With no z (or z at table_z) this is the flat table-plane inverse --
        exact for a point on the table, off by the parallax amount for
        anything above it.

        With z given, the primary correction is the true pinhole geometry of
        the *arm's TCP*: a point at height h projects, through the flat table
        homography, onto the table-plane intersection of its camera ray, which
        sits radially outward from the camera nadir by

            scale = cam_height / (cam_height - h)

        (isotropic scaling of (x, y) about ``cam_xy_robot``). This is exact
        given the arm's true height -- no cube involved -- and both parameters
        are measured directly against the arm (calibrate_camera_nadir.py:
        grip a cube, sweep it through known heights, fit nadir + cam_height to
        where it lands). On this rig the camera is steeply *oblique*, so the
        nadir is far off to one side (well outside the workspace) and raising
        the TCP shifts its image a lot along the camera azimuth -- which the
        earlier cube-top model badly under-corrected (drew the trajectory too
        low). The direct fit recovers cam_height within ~2mm of the physical
        measurement and tracks the arm to <~15px at any operating height.

        Falls back, only when ``cam_xy_robot`` is unset, to scaling the
        cube-top homography's parallax offset by ``h/cube_height`` -- the
        cube-detection map carries a direction-dependent centroid bias and a
        sub-cube effective height, so it is anisotropic and under-shifts with
        height; usable near the cube-top baseline, not at travel/hover. Then
        to no correction at all. Fine for overlay/debug drawing; do not use
        this for anything that drives a pick.
        """
        m0 = np.linalg.inv(np.array(self.homography, dtype=np.float64))

        def flat_pixel(rx: float, ry: float) -> tuple[float, float]:
            w = m0 @ np.array([rx, ry, 1.0])
            return float(w[0] / w[2]), float(w[1] / w[2])

        px, py = flat_pixel(x, y)
        if z is None:
            return px, py
        height_mm = z - self.table_z
        if abs(height_mm) < 1e-6:
            return px, py
        if (
            self.cam_xy_robot
            and self.cam_height_mm
            and self.cam_height_mm - height_mm > 1e-6
        ):
            # Primary: measured camera geometry (nadir + lens height). Exact
            # pinhole parallax of the TCP at its true height.
            cx, cy = self.cam_xy_robot
            scale = self.cam_height_mm / (self.cam_height_mm - height_mm)
            return flat_pixel(cx + (x - cx) * scale, cy + (y - cy) * scale)
        if self.cube_top_homography:
            # Fallback (no measured nadir): scale the cube-top parallax offset
            # linearly with height off its cube-top baseline. Anisotropic and
            # low above the baseline (see docstring) -- overlay-only stopgap.
            m_ct = np.linalg.inv(np.array(self.cube_top_homography, dtype=np.float64))
            pc = m_ct @ np.array([x, y, 1.0])
            qx, qy = self.pixel_to_robot(float(pc[0] / pc[2]), float(pc[1] / pc[2]))
            ratio = height_mm / self.cube_height_mm
            return flat_pixel(x + (qx - x) * ratio, y + (qy - y) * ratio)
        return px, py

    def _residual_correction(self, x: float, y: float) -> tuple[float, float]:
        r = self.cube_top_residual
        pts = np.array(r["points"], dtype=np.float64)
        deltas = np.array(r["deltas"], dtype=np.float64)
        two_s2 = 2.0 * float(r.get("sigma_mm", 60.0)) ** 2
        reg = float(r.get("reg", 0.25))
        w = np.exp(-((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2) / two_s2)
        denom = w.sum() + reg
        return (
            float((w * deltas[:, 0]).sum() / denom),
            float((w * deltas[:, 1]).sum() / denom),
        )

    def save(self, path: Path = DEFAULT_CALIB_PATH) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


def load_calibration(path: Path = DEFAULT_CALIB_PATH) -> Calibration:
    if not Path(path).exists():
        raise CalibrationError(
            f"no calibration at {path} -- run: python calibrate_vision.py"
        )
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Ignore unknown / retired keys (e.g. removed j4_face_offset_deg) so old
    # JSON files still load; the next save drops them.
    known = {f.name for f in fields(Calibration)}
    return Calibration(**{k: v for k, v in data.items() if k in known})


def fit_homography(
    pixel_pts: list[tuple[float, float]], robot_pts: list[tuple[float, float]]
) -> list[list[float]]:
    """Least-squares pixel->robot homography from >=4 correspondences."""
    if len(pixel_pts) < 4 or len(pixel_pts) != len(robot_pts):
        raise CalibrationError(
            f"need >=4 matched points, got {len(pixel_pts)} pixel / {len(robot_pts)} robot"
        )
    src = np.array(pixel_pts, dtype=np.float64)
    dst = np.array(robot_pts, dtype=np.float64)
    h, _mask = cv2.findHomography(src, dst, 0)
    if h is None:
        raise CalibrationError("homography fit failed (degenerate point set?)")
    return h.tolist()


def fit_affine(
    pixel_pts: list[tuple[float, float]], robot_pts: list[tuple[float, float]]
) -> list[list[float]]:
    """Affine pixel->robot fit from >=3 correspondences, embedded as a 3x3
    homography (bottom row [0, 0, 1]) so downstream code is unchanged.

    Only reach for this when just 3 markers are physically reachable: an
    affine map cannot model perspective foreshortening, so accuracy degrades
    away from the calibration triangle.
    """
    if len(pixel_pts) < 3 or len(pixel_pts) != len(robot_pts):
        raise CalibrationError(
            f"need >=3 matched points, got {len(pixel_pts)} pixel / {len(robot_pts)} robot"
        )
    src = np.hstack(
        [np.array(pixel_pts, dtype=np.float64), np.ones((len(pixel_pts), 1))]
    )
    dst = np.array(robot_pts, dtype=np.float64)
    coef, _res, rank, _sv = np.linalg.lstsq(src, dst, rcond=None)
    if rank < 3:
        raise CalibrationError("affine fit failed (collinear points?)")
    h = np.eye(3)
    h[0, :] = coef[:, 0]
    h[1, :] = coef[:, 1]
    return h.tolist()


def fit_transform(
    pixel_pts: list[tuple[float, float]], robot_pts: list[tuple[float, float]]
) -> tuple[list[list[float]], str]:
    """Fit the best available pixel->robot map: full homography with >=4
    points, affine with exactly 3. Returns (matrix, kind)."""
    if len(pixel_pts) >= 4:
        return fit_homography(pixel_pts, robot_pts), "homography"
    return fit_affine(pixel_pts, robot_pts), "affine"


def reprojection_errors(
    homography: list[list[float]],
    pixel_pts: list[tuple[float, float]],
    robot_pts: list[tuple[float, float]],
) -> list[float]:
    h = np.array(homography, dtype=np.float64)
    errors = []
    for (px, py), (rx, ry) in zip(pixel_pts, robot_pts):
        v = h @ np.array([px, py, 1.0])
        errors.append(float(np.hypot(v[0] / v[2] - rx, v[1] / v[2] - ry)))
    return errors
