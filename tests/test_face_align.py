"""Unit tests for cube face-align J4 helpers (no hardware)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.calib import Calibration
from mt4_vision.pickplace import (
    fold_square_yaw_deg,
    j4_for_face_align,
    j4_preserve_wrist,
    resolve_place_j4,
)


class _Tcp:
    def __init__(self, j4):
        self.j4 = j4


class _StubClient:
    def __init__(self, tcp=None):
        self._tcp = tcp

    def get_tcp(self):
        return self._tcp


def _calib(**kw) -> Calibration:
    return Calibration(
        homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        table_z=0.0, safe_z=80.0, **kw,
    )


def test_fold_square_yaw_period_90():
    assert abs(fold_square_yaw_deg(0.0)) < 1e-9
    assert abs(fold_square_yaw_deg(90.0)) < 1e-9
    assert abs(fold_square_yaw_deg(45.0) - (-45.0)) < 1e-9
    assert abs(fold_square_yaw_deg(30.0) - 30.0) < 1e-9
    assert abs(fold_square_yaw_deg(-30.0) - (-30.0)) < 1e-9


def test_j4_face_align_folds_without_current():
    assert abs(j4_for_face_align(10.0) - 10.0) < 1e-9
    assert abs(j4_for_face_align(100.0) - 10.0) < 1e-9


def test_j4_face_align_picks_nearest_90_to_current():
    # Cube edge at 10°, current wrist at 95° → prefer 100° over 10°.
    j4 = j4_for_face_align(10.0, current_j4_deg=95.0)
    assert abs(j4 - 100.0) < 1e-9


def test_j4_face_align_avoids_joint_soft_limit_at_far_neg_y():
    # Stack level-4 failure: wrist ~89°, cube yaw ~-161° at (73,-224).
    # Nearest world candidate is ~109°, but joint J4 = 109 - (-72) ≈ 181°
    # exceeds soft max -- must fall back to ~19°.
    j4 = j4_for_face_align(
        -161.24, current_j4_deg=88.8, x=73.1, y=-223.6,
    )
    assert abs(j4 - 18.76) < 0.5
    # joint J4 stays inside ±180° soft window
    j1 = __import__("math").degrees(__import__("math").atan2(-223.6, 73.1))
    assert abs(j4 - j1) < 180.0


def test_j4_preserve_wrist_holds_joint_across_j1_swing():
    # Park (200,0) world j4=77.8 → joint j4=77.8; at marker-0 bearing
    # j1≈−79.2, world j4 should be ≈−1.4 so joint stays 77.8.
    j4 = j4_preserve_wrist(
        49.1, -256.6, from_x=200.0, from_y=0.0, from_j4=77.8
    )
    assert abs(j4 - (-1.367)) < 0.05


def test_resolve_place_j4_off_skips_hardware():
    # Explicitly disabled: no squaring, and no hardware access either.
    calib = _calib()
    assert resolve_place_j4(None, calib, axis_align=False) is None


def test_resolve_place_j4_squares_to_zero_nearest_current():
    calib = _calib()
    # Default is on; no TCP reading: folded 0°.
    assert abs(resolve_place_j4(_StubClient(), calib) - 0.0) < 1e-9
    # Wrist at 95°: nearest 90°-equivalent of 0° is 90°.
    j4 = resolve_place_j4(_StubClient(_Tcp(95.0)), calib)
    assert abs(j4 - 90.0) < 1e-9


if __name__ == "__main__":
    test_fold_square_yaw_period_90()
    test_j4_face_align_folds_without_current()
    test_j4_face_align_picks_nearest_90_to_current()
    test_j4_face_align_avoids_joint_soft_limit_at_far_neg_y()
    test_j4_preserve_wrist_holds_joint_across_j1_swing()
    test_resolve_place_j4_off_skips_hardware()
    test_resolve_place_j4_squares_to_zero_nearest_current()
    print("ok")
