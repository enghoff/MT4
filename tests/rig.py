"""One synthetic-but-real rig for tests that need a work region.

Where the numbers come from
---------------------------
The homography, camera nadir/height, table heights and desk polygon are the
live values measured on the physical rig (``vision_calibration.json``, camera
refit 2026-07-21, desk edge 2026-08-02). They are frozen here rather than
loaded from that file on purpose: a test that reads the live calibration
changes its own expectations every time somebody recalibrates, which makes a
red test ambiguous between "the code broke" and "the desk moved".

Why tests need a calibration at all
-----------------------------------
The work region is defined by where the desk physically is and what the camera
can see, and neither of those can be expressed as a list of marker positions --
which is all a test could otherwise supply. So the rig comes with the tests.

``MARKERS`` are the live marker layout. Tests carrying their own
slightly-shifted set work just as well, since markers do not gate anything.
"""

from __future__ import annotations

from mt4_vision.calib import Calibration
from mt4_vision.workspace import MarkerSlot, expected_cube_area_px2

# Measured desk edge x = -72.2mm plus the 25mm safety margin, closed off with
# three nominal sides far outside a 350mm reach. See calibrate_table_edge.py.
TABLE_POLYGON = [
    [-47.2, -500.0],
    [500.0, -500.0],
    [500.0, 500.0],
    [-47.2, 500.0],
]

HOMOGRAPHY = [
    [-0.4113912840092131, 21.88840372175907, -7751.187453160993],
    [10.199101850740103, -0.6413458684138484, -6918.915082712266],
    [-0.0007562964205790095, 0.03748048081823857, 1.0],
]

CUBE_TOP_HOMOGRAPHY = [
    [-0.2640131345308154, 22.303695888926278, -7808.390919357973],
    [9.7373210428213, -0.549897750906098, -6647.578881547855],
    [-0.0007562964205790095, 0.03748048081823857, 1.0],
]


def calibration(**overrides) -> Calibration:
    """A Calibration good enough for every geometric predicate."""
    kwargs = dict(
        homography=HOMOGRAPHY,
        cube_top_homography=CUBE_TOP_HOMOGRAPHY,
        table_z=127.2,
        safe_z=155.0,
        cube_height_mm=20.0,
        grip_open_s=140,
        grip_close_s=255,
        travel_speed_us=700,
        approach_speed_us=2400,
        cam_xy_robot=[518.1, -35.0],
        cam_height_mm=244.0,
        table_polygon_robot=[list(p) for p in TABLE_POLYGON],
        frame_size_px=[1280, 720],
    )
    kwargs.update(overrides)
    return Calibration(**kwargs)


CALIB = calibration()

# Detector blob area as a fraction of the geometric silhouette
# expected_cube_area_px2 predicts: median 0.92 over 229 logged detections
# spanning the whole table (two shuffle runs, 2026-08-10; the ratio ran
# 0.78-1.04 with no drift across the 3x positional range).
CUBE_AREA_FILL = 0.92


def cube_area_at(x: float, y: float, calib: Calibration = CALIB) -> float:
    """The blob area a real cube resting at (x, y) is detected with.

    Test fixtures used to carry one hard-coded area for every position, which
    is the claim that a cube looks the same size everywhere -- and that claim is
    what the position-aware size gate exists to retire. A cube in a fixture
    should be the size a cube at that spot really is, so a test about the
    keep-out or the desk edge is not quietly also a test about area.
    """
    return expected_cube_area_px2(float(x), float(y), calib) * CUBE_AREA_FILL

# Live marker layout (vision_calibration.json, 2026-07-21 refit).
MARKERS = [
    MarkerSlot(0, 48.1, -228.5),
    MarkerSlot(1, 34.7, 233.7),
    MarkerSlot(2, 161.9, -149.6),
    MarkerSlot(3, 153.6, 156.9),
    MarkerSlot(4, 211.3, 7.3),
]
