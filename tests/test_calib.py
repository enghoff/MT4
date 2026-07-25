"""Calibration mapping guards."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_vision import calib as calib_mod
from mt4_vision.calib import Calibration

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def make_calib(**overrides) -> Calibration:
    return Calibration(
        homography=IDENTITY, table_z=144.0, safe_z=185.0, **overrides
    )


def _reset_warning(monkeypatch):
    monkeypatch.setattr(calib_mod, "_warned_no_cube_top_correction", False)


def test_missing_cube_top_warns_once(capsys, monkeypatch):
    _reset_warning(monkeypatch)
    c = make_calib()
    c.pixel_to_robot(10.0, 10.0, on_cube_top=True)
    c.pixel_to_robot(20.0, 20.0, on_cube_top=True)
    err = capsys.readouterr().err
    assert err.count("cube_top_homography") == 1
    assert "calibrate_height.py" in err


def test_table_plane_mapping_never_warns(capsys, monkeypatch):
    _reset_warning(monkeypatch)
    make_calib().pixel_to_robot(10.0, 10.0)
    assert capsys.readouterr().err == ""


def test_cube_top_homography_set_no_warning(capsys, monkeypatch):
    _reset_warning(monkeypatch)
    c = make_calib(cube_top_homography=IDENTITY)
    c.pixel_to_robot(10.0, 10.0, on_cube_top=True)
    assert capsys.readouterr().err == ""


def test_radial_fallback_no_warning(capsys, monkeypatch):
    _reset_warning(monkeypatch)
    c = make_calib(cam_xy_robot=[400.0, 0.0], cam_height_mm=600.0)
    c.pixel_to_robot(10.0, 10.0, on_cube_top=True)
    assert capsys.readouterr().err == ""


def test_color_xy_offset_applied_to_cube_detections():
    import cv2  # noqa: F401 -- ensures OpenCV present for detect_cubes
    import numpy as np

    from mt4_vision.detect import detect_cubes

    frame = np.zeros((200, 200, 3), np.uint8)
    frame[80:130, 60:110] = (0, 0, 255)  # red square, ~2500px^2 (closer-mount scale)
    base = make_calib(cube_top_homography=IDENTITY)
    offset = make_calib(
        cube_top_homography=IDENTITY, color_xy_offset_mm={"red": [5.0, -3.0]}
    )
    a = detect_cubes(frame, base)[0]
    b = detect_cubes(frame, offset)[0]
    assert abs(b.x - a.x - 5.0) < 1e-6
    assert abs(b.y - a.y + 3.0) < 1e-6


def test_top_face_centroid_ignores_darker_side_face():
    """Red-like case: bright top face, darker side face below -- the
    detection centroid must land on the top face, not the blob middle."""
    import numpy as np

    from mt4_vision.detect import detect_cubes

    frame = np.zeros((200, 200, 3), np.uint8)
    frame[60:110, 80:130] = (0, 0, 230)   # top face: bright red, center (104.5, 84.5)
    frame[110:135, 80:130] = (0, 0, 120)  # side face below: darker red
    det = detect_cubes(frame)[0]
    # whole-blob centroid would sit at y ~= 97; top-face center is 84.5
    assert abs(det.px - 104.5) < 1.0
    assert abs(det.py - 84.5) < 2.0


def test_top_face_centroid_ignores_brighter_side_face():
    """Green-like case (observed live): the lit side face is BRIGHTER than
    the top face. Brightness ranking would pick the side; geometry must
    still pick the top."""
    import numpy as np

    from mt4_vision.detect import detect_cubes

    frame = np.zeros((200, 200, 3), np.uint8)
    frame[60:110, 80:130] = (0, 140, 0)   # top face: mid green, center (104.5, 84.5)
    frame[110:135, 80:130] = (0, 235, 0)  # side face below: brightly lit green
    det = detect_cubes(frame)[0]
    assert abs(det.px - 104.5) < 1.0
    assert abs(det.py - 84.5) < 2.0


def test_top_face_centroid_unbiased_when_blob_is_all_top_face():
    """Near the camera nadir only the top face is visible: the segmented
    centroid must equal the plain blob centroid (no upward bias)."""
    import numpy as np

    from mt4_vision.detect import detect_cubes

    frame = np.zeros((200, 200, 3), np.uint8)
    frame[60:110, 80:130] = (0, 0, 200)   # uniform square, center (104.5, 84.5)
    det = detect_cubes(frame)[0]
    assert abs(det.px - 104.5) < 0.6
    assert abs(det.py - 84.5) < 0.6


def test_cube_top_residual_layer():
    """Correction applies near a probe point and decays to ~zero far away."""
    c = make_calib(
        cube_top_homography=IDENTITY,
        cube_top_residual={
            "points": [[100.0, 100.0]],
            "deltas": [[8.0, -4.0]],
            "sigma_mm": 60.0,
            "reg": 0.25,
        },
    )
    # at the probe point: most of the delta (shrunk slightly by reg)
    x, y = c.pixel_to_robot(100.0, 100.0, on_cube_top=True)
    assert 105.0 < x < 108.0
    assert 96.0 < y < 97.5
    # far away: correction ~0
    x, y = c.pixel_to_robot(500.0, 500.0, on_cube_top=True)
    assert abs(x - 500.0) < 0.1
    assert abs(y - 500.0) < 0.1
