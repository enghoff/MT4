"""Pure-logic tests for mt4_vision.landing (no hardware)."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_vision.landing import (
    LANDING_MIN_RADIUS_MM,
    LANDING_SEP_MM,
    annulus_grid,
    landing_ok,
    nearest_landing,
    push_aside_xy,
    random_landing,
)
from mt4_vision.workspace import dist_mm


def test_landing_ok_rejects_keepout_and_near_base():
    assert not landing_ok(50.0, 0.0)  # inside J1 keep-out
    assert not landing_ok(150.0, 0.0)  # below LANDING_MIN_RADIUS_MM
    assert landing_ok(200.0, 0.0)


def test_landing_ok_sep_and_site_avoid():
    assert not landing_ok(
        200.0, 0.0, occupied=[(200.0, 20.0)], sep_mm=LANDING_SEP_MM,
    )
    assert not landing_ok(
        200.0, 0.0, site_xy=(200.0, 50.0), site_avoid_mm=60.0,
    )
    assert landing_ok(
        200.0, 0.0, site_xy=(200.0, 80.0), site_avoid_mm=60.0,
    )


def test_push_aside_and_nearest_share_the_predicate():
    sx, sy = 243.0, 5.0
    dest = push_aside_xy(sx, sy, 251.0, 27.0, occupied=[], park_mm=110.0)
    assert dest is not None
    assert math.hypot(dest[0] - sx, dest[1] - sy) >= 110.0 - 0.1
    assert landing_ok(dest[0], dest[1], min_radius_mm=LANDING_MIN_RADIUS_MM)

    park = nearest_landing(
        sx, sy,
        preferred=[(280.0, 0.0), (150.0, 100.0)],
        fallback=annulus_grid(),
        site_clear_mm=110.0,
    )
    assert park is not None
    assert dist_mm(park[0], park[1], sx, sy) >= 110.0


def test_random_landing_stays_in_shared_band():
    from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration
    from mt4_vision.workspace import marker_slots_from_calibration

    calib = load_calibration(DEFAULT_CALIB_PATH)
    markers = marker_slots_from_calibration(calib)
    site = next(m for m in markers if m.marker_id == 4)
    xy = random_landing(
        random.Random(7),
        sx=site.x, sy=site.y, markers=markers, avoid=[],
        spacing_mm=75.0, site_avoid_mm=90.0,
    )
    assert xy is not None
    assert LANDING_MIN_RADIUS_MM <= math.hypot(*xy)
