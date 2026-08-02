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
from mt4_vision.workspace import dist_mm, in_work_region
from rig import CALIB, MARKERS


def test_landing_ok_rejects_keepout_and_near_base():
    assert not landing_ok(50.0, 0.0, CALIB)  # inside J1 keep-out
    assert not landing_ok(150.0, 0.0, CALIB)  # below LANDING_MIN_RADIUS_MM
    assert landing_ok(200.0, 0.0, CALIB)


def test_landing_ok_sep_and_site_avoid():
    assert not landing_ok(
        200.0, 0.0, CALIB, occupied=[(200.0, 20.0)], sep_mm=LANDING_SEP_MM,
    )
    assert not landing_ok(
        200.0, 0.0, CALIB, site_xy=(200.0, 50.0), site_avoid_mm=60.0,
    )
    assert landing_ok(
        200.0, 0.0, CALIB, site_xy=(200.0, 80.0), site_avoid_mm=60.0,
    )


def test_landing_ok_refuses_off_desk_and_off_camera():
    """The two gates that make a landing findable again.

    Both points are inside the arm's joint envelope, so nothing but the work
    region stops them -- which is the whole reason a landing is not just
    "somewhere the arm can reach".
    """
    assert not landing_ok(-100.0, 250.0, CALIB)  # past the desk's back edge
    assert not landing_ok(320.0, 0.0, CALIB)     # past the camera's near edge


def test_landing_ok_accepts_the_far_desk_the_marker_hull_rejected():
    # Measured live 2026-08-02: on the desk, in reach, in frame, and outside
    # the old marker-centre hull. A cube parked here used to be unfindable.
    assert landing_ok(266.5, -52.7, CALIB)
    assert landing_ok(176.1, -213.6, CALIB)


def test_push_aside_and_nearest_share_the_predicate():
    sx, sy = 243.0, 5.0
    dest = push_aside_xy(
        sx, sy, 251.0, 27.0, [], CALIB, park_mm=110.0,
    )
    assert dest is not None
    assert math.hypot(dest[0] - sx, dest[1] - sy) >= 110.0 - 0.1
    assert landing_ok(
        dest[0], dest[1], CALIB, min_radius_mm=LANDING_MIN_RADIUS_MM
    )

    park = nearest_landing(
        sx, sy, CALIB,
        preferred=[(280.0, 0.0), (150.0, 100.0)],
        fallback=annulus_grid(),
        site_clear_mm=110.0,
    )
    assert park is not None
    assert dist_mm(park[0], park[1], sx, sy) >= 110.0
    assert in_work_region(park[0], park[1], CALIB)


def test_random_landing_stays_in_shared_band():
    site = next(m for m in MARKERS if m.marker_id == 4)
    xy = random_landing(
        random.Random(7), CALIB,
        sx=site.x, sy=site.y, markers=MARKERS, avoid=[],
        spacing_mm=75.0, site_avoid_mm=90.0,
    )
    assert xy is not None
    assert LANDING_MIN_RADIUS_MM <= math.hypot(*xy)
    assert in_work_region(xy[0], xy[1], CALIB)


def test_random_landing_can_use_the_widened_region():
    """Scatter used to be capped at a 240mm circle.

    Over many draws it should now reach past that, since the cap was camera
    coverage stated as a radius and the camera reaches much further off-axis.
    """
    site = next(m for m in MARKERS if m.marker_id == 4)
    radii = []
    for seed in range(60):
        xy = random_landing(
            random.Random(seed), CALIB,
            sx=site.x, sy=site.y, markers=MARKERS, avoid=[],
            spacing_mm=45.0, site_avoid_mm=90.0,
        )
        if xy is not None:
            radii.append(math.hypot(*xy))
    assert radii, "scatter found no landing at all"
    assert max(radii) > 240.0, f"never left the old radius cap (max {max(radii):.0f}mm)"
