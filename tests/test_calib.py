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


# ------------------------------------------------- save() drop-protection
#
# 2026-08-03: recalibrate_camera.py rebuilt a fresh Calibration naming 16 of
# the 22 fields; the six it did not name took their dataclass defaults. Two of
# them -- grip_span_s_at_zero_mm = 212.3 and grip_span_s_per_mm = 1.881, the
# jaw-opening model -- are properties of the GRIPPER, which moving a camera
# cannot invalidate. Their refusal gate fails open when unmeasured, so nothing
# complained; the arm simply stopped declining objects too wide to hold. These
# tests are what makes losing a measurement an error instead of a default.


def test_cleared_fields_names_what_a_save_would_lose():
    from mt4_vision.calib import cleared_fields

    previous = {
        "grip_span_s_per_mm": 1.881,
        "cam_height_mm": 244.0,
        "color_xy_offset_mm": {"red": [1.0, 2.0]},
        "table_z": 127.2,
        "cube_top_homography": None,
    }
    incoming = {
        "grip_span_s_per_mm": None,      # nulled
        # cam_height_mm absent entirely
        "color_xy_offset_mm": {},        # emptied -- same loss as nulling
        "table_z": 127.2,                # unchanged
        "cube_top_homography": None,     # was already empty, not a loss
    }
    assert cleared_fields(previous, incoming) == [
        "cam_height_mm", "color_xy_offset_mm", "grip_span_s_per_mm",
    ]


def test_cleared_fields_ignores_a_field_the_old_file_never_had():
    """A field added to the dataclass after a file was written was never
    measured, so its absence from that file is not a loss."""
    from mt4_vision.calib import cleared_fields

    assert cleared_fields({"table_z": 127.2}, {"table_z": 127.2, "new_thing": None}) == []


def test_save_refuses_to_silently_drop_a_measured_value(tmp_path):
    from dataclasses import replace

    from mt4_vision.calib import CalibrationError

    path = tmp_path / "vision_calibration.json"
    make_calib(grip_span_s_at_zero_mm=212.3, grip_span_s_per_mm=1.881).save(path)

    # The exact shape of the bug: a fresh object built field-by-field, so the
    # jaw model reverts to its default.
    try:
        make_calib().save(path)
    except CalibrationError as exc:
        assert "grip_span_s_at_zero_mm=212.3" in str(exc)
        assert "grip_span_s_per_mm=1.881" in str(exc)
    else:
        raise AssertionError("save() accepted a silent drop")

    # The file on disk is untouched -- a refused save must not half-write.
    from mt4_vision.calib import load_calibration
    assert load_calibration(path).grip_span_s_per_mm == 1.881

    # replace() carries everything, so it just works.
    replace(load_calibration(path), table_z=130.0).save(path)
    reloaded = load_calibration(path)
    assert reloaded.table_z == 130.0 and reloaded.grip_span_s_per_mm == 1.881


def test_a_deliberate_clear_is_allowed_when_it_is_named(tmp_path):
    from mt4_vision.calib import load_calibration

    path = tmp_path / "vision_calibration.json"
    make_calib(cam_height_mm=244.0, cam_xy_robot=[518.1, -35.0]).save(path)

    # What recalibrate_camera.py does: the camera moved, so its pose is stale.
    make_calib().save(path, clearing=("cam_height_mm", "cam_xy_robot"))
    assert load_calibration(path).cam_height_mm is None


def test_naming_one_clear_does_not_excuse_another(tmp_path):
    from mt4_vision.calib import CalibrationError, load_calibration

    path = tmp_path / "vision_calibration.json"
    make_calib(cam_height_mm=244.0, grip_span_s_per_mm=1.881).save(path)
    try:
        make_calib().save(path, clearing=("cam_height_mm",))
    except CalibrationError as exc:
        assert "grip_span_s_per_mm" in str(exc)
        assert "cam_height_mm" not in str(exc)
    else:
        raise AssertionError("an unnamed drop rode along with a named one")
    assert load_calibration(path).grip_span_s_per_mm == 1.881


def test_the_first_save_to_a_new_path_has_nothing_to_protect(tmp_path):
    make_calib().save(tmp_path / "fresh.json")
    assert (tmp_path / "fresh.json").exists()


# ------------------------------------------- the load-early / save-late loss
#
# Every calibration script loads the file, drives the arm for minutes, then
# saves. Calibration is a mutable dataclass, so that save writes back the file
# as it was at LOAD time plus its own edits -- reverting anything written in
# between. calibrate_height saves repeatedly *during* its run, so it does this
# over and over. Measured 2026-08-03: cam_xy_robot and cam_height_mm were
# written by calibrate_camera_nadir.py and were null again twice the same day,
# with no error from anything.


def test_a_stale_writer_no_longer_reverts_another_scripts_work(tmp_path):
    """The exact sequence that lost the camera nadir, twice."""
    from mt4_vision.calib import load_calibration, update_calibration

    path = tmp_path / "vision_calibration.json"
    make_calib().save(path)

    # Script A loads the file and starts a long probe run.
    stale = load_calibration(path)

    # Script B finishes in the meantime and writes what it measured.
    update_calibration(path, cam_xy_robot=[518.1, -35.0], cam_height_mm=244.0)

    # Script A now saves what IT measured. A whole-object stale.save(path)
    # would write back a snapshot that predates B and silently drop both.
    stale.table_polygon_robot = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    update_calibration(
        path, table_polygon_robot=stale.table_polygon_robot
    )

    after = load_calibration(path)
    assert after.table_polygon_robot == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert after.cam_xy_robot == [518.1, -35.0]
    assert after.cam_height_mm == 244.0


def test_the_old_whole_object_save_is_what_save_now_refuses(tmp_path):
    """Belt to the braces: even if a script forgets to merge, it cannot lose
    a measurement silently -- save() reads the file and refuses."""
    from mt4_vision.calib import CalibrationError, load_calibration, update_calibration

    path = tmp_path / "vision_calibration.json"
    make_calib().save(path)
    stale = load_calibration(path)
    update_calibration(path, cam_height_mm=244.0)

    try:
        stale.save(path)
    except CalibrationError as exc:
        assert "cam_height_mm=244.0" in str(exc)
    else:
        raise AssertionError("a stale whole-object save was accepted")
    assert load_calibration(path).cam_height_mm == 244.0


def test_a_merge_warns_when_the_table_map_moved_underneath_it(tmp_path, capsys):
    """The values were measured against a map that is no longer current, so
    merging them is arithmetic the operator has to agree to."""
    from mt4_vision.calib import load_calibration, update_calibration

    path = tmp_path / "vision_calibration.json"
    make_calib().save(path)
    based_on = load_calibration(path)

    refit = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    update_calibration(path, homography=refit)
    update_calibration(path, based_on=based_on, cam_height_mm=244.0)

    assert "has been re-fit since this run loaded it" in capsys.readouterr().err
    # It still writes -- the caller measured something real, and refusing here
    # would strand a completed hardware run with nowhere to put its result.
    assert load_calibration(path).cam_height_mm == 244.0
