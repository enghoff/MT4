"""Pure-logic tests for unstack_cubes helpers (no hardware)."""

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration
from mt4_vision.workspace import (
    MARKER_PAPER_CLEARANCE_MM,
    dist_mm,
    is_mp_reachable_xy,
    marker_slots_from_calibration,
)
from unstack_cubes import (
    SCATTER_MAX_RADIUS_MM,
    SCATTER_MIN_RADIUS_MM,
    SITE_AVOID_MM,
    find_landing,
    random_landing,
    random_place_j4,
)


def _calib_and_markers():
    calib = load_calibration(DEFAULT_CALIB_PATH)
    markers = marker_slots_from_calibration(calib)
    site = next(m for m in markers if m.marker_id == 4)
    return calib, markers, site


def test_random_landing_respects_reach_and_spacing():
    _calib, markers, site = _calib_and_markers()
    avoid = [(200.0, -60.0), (150.0, 100.0)]
    rng = random.Random(1)
    xy = random_landing(
        rng, sx=site.x, sy=site.y, markers=markers, avoid=avoid, spacing_mm=75.0,
    )
    assert xy is not None
    x, y = xy
    r = math.hypot(x, y)
    assert SCATTER_MIN_RADIUS_MM <= r <= SCATTER_MAX_RADIUS_MM
    assert is_mp_reachable_xy(x, y)
    assert dist_mm(x, y, site.x, site.y) >= SITE_AVOID_MM
    for ox, oy in avoid:
        assert dist_mm(x, y, ox, oy) >= 75.0
    for m in markers:
        assert dist_mm(x, y, m.x, m.y) >= MARKER_PAPER_CLEARANCE_MM


def test_random_landing_avoids_marker_papers_directly_on_a_marker():
    """A dense avoid ring right around one non-site marker must still steer
    the draw off that marker's own paper (a fixed check, not just spacing
    from prior drops)."""
    _calib, markers, site = _calib_and_markers()
    other = next(m for m in markers if m.marker_id != site.marker_id)
    rng = random.Random(2)
    for _ in range(20):
        xy = random_landing(
            rng, sx=site.x, sy=site.y, markers=markers, avoid=[], spacing_mm=75.0,
        )
        assert xy is not None
        x, y = xy
        assert dist_mm(x, y, other.x, other.y) >= MARKER_PAPER_CLEARANCE_MM


def test_find_landing_degrades_spacing_when_crowded():
    """When the preferred spacing can't be satisfied, find_landing must fall
    back to a tighter one rather than raising -- exercised here by pinning
    random_landing to only succeed at the tightest fallback."""
    import unstack_cubes as uc

    calls = []

    def fake_random_landing(rng, *, sx, sy, markers, avoid, spacing_mm, attempts=0):
        calls.append(spacing_mm)
        if spacing_mm == uc.DROP_SPACING_FALLBACKS_MM[-1]:
            return (250.0, 0.0)
        return None

    orig = uc.random_landing
    uc.random_landing = fake_random_landing
    try:
        landing, spacing = find_landing(
            random.Random(3), sx=0.0, sy=0.0, markers=[], avoid=[],
        )
    finally:
        uc.random_landing = orig
    assert landing == (250.0, 0.0)
    assert spacing == uc.DROP_SPACING_FALLBACKS_MM[-1]
    assert calls == list(uc.DROP_SPACING_FALLBACKS_MM)


def test_find_landing_raises_when_desk_has_no_room():
    import unstack_cubes as uc
    from mt4_jog.client import Mt4ClientError

    orig = uc.random_landing
    uc.random_landing = lambda *a, **k: None
    try:
        try:
            find_landing(random.Random(4), sx=0.0, sy=0.0, markers=[], avoid=[])
            assert False, "expected Mt4ClientError"
        except Mt4ClientError:
            pass
    finally:
        uc.random_landing = orig


def test_random_place_j4_delegates_to_face_align_with_the_drawn_angle():
    from mt4_vision.pickplace import j4_for_face_align

    x, y = 240.0, -150.0
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    expected_angle = rng2.uniform(0.0, 360.0)
    expected = j4_for_face_align(expected_angle, current_j4_deg=None, x=x, y=y)
    assert random_place_j4(x, y, rng1) == expected


def test_pick_grip_height_matches_stack_release_line():
    """unstack_cubes.pick_from_stack grips level N at grip_top_z(N - 1);
    stack_cubes released it 4mm above that same line. If either script's
    height formula drifts, the two must disagree here first."""
    from mt4_vision.stackpath import StackPlanner
    from stack_cubes import release_z_for_level

    calib, _markers, site = _calib_and_markers()
    planner = StackPlanner(calib, site.x, site.y)
    for level in (1, 2, 3):
        grip_z = planner.grip_top_z(level - 1)
        assert abs(grip_z - (release_z_for_level(calib, level) - 4.0)) < 1e-9


def test_pick_from_stack_queues_open_barrier_and_close():
    """Approach opens on the first transit leg, barriers at hover, closes
    as a station at grip height -- no blocking client.gripper() calls."""
    from types import SimpleNamespace

    from mt4_vision.motion import BARRIER_DWELL_MS
    from mt4_vision.stackpath import StackPlanner
    from unstack_cubes import pick_from_stack

    calib, _markers, site = _calib_and_markers()
    # Live calib already has grip_open_s / grip_close_s; pin speeds the
    # FakeClient's move_path recording expects from send_legs.
    planner = StackPlanner(calib, site.x, site.y)
    grip_z = planner.grip_top_z(0)
    hz = planner.hover_z(1)
    assert hz is not None

    class _Tcp:
        def __init__(self):
            self.x, self.y, self.z, self.j4 = 200.0, 0.0, float(calib.safe_z), 0.0

    class _Client:
        def __init__(self):
            self._tcp = _Tcp()
            self.calls = []
            self.gripper_calls = []

        def get_tcp(self):
            return self._tcp

        def get_status(self):
            return SimpleNamespace(homed=True, joints={"j4": 0})

        def gripper(self, action):
            self.gripper_calls.append(action)
            return {"ok": True}

        def move_path(self, waypoints, j4=None, grip=0, speed_us=0, dwell_ms=0, **kw):
            n = len(waypoints)

            def spread(v):
                return list(v) if isinstance(v, (list, tuple)) else [v] * n

            self.calls.append(
                {
                    "wps": list(waypoints),
                    "grips": list(grip) if isinstance(grip, (list, tuple))
                    else [grip] + [0] * (n - 1),
                    "dwells": spread(dwell_ms),
                    "speeds": spread(speed_us),
                }
            )
            return {"ok": True}

    client = _Client()
    pick_from_stack(
        client, calib, planner, 1, approach_prefer_xy=(200.0, 0.0),
    )
    assert client.gripper_calls == []
    assert len(client.calls) == 1
    call = client.calls[0]
    # Open rides the first transit leg; a barrier sits at hover before the
    # slow descend; close station repeats the grip pose.
    assert call["grips"][0] == int(calib.grip_open_s)
    # Find barrier (dwell=1, grip=0) then close station at grip_z.
    barrier_i = next(
        i for i, (d, g) in enumerate(zip(call["dwells"], call["grips"]))
        if d == BARRIER_DWELL_MS and g == 0
    )
    assert call["wps"][barrier_i] == (site.x, site.y, hz)
    close_i = next(
        i for i, (wp, d, g) in enumerate(
            zip(call["wps"], call["dwells"], call["grips"])
        )
        if d > BARRIER_DWELL_MS and g == int(calib.grip_close_s)
    )
    assert call["wps"][close_i] == (site.x, site.y, grip_z)
    assert barrier_i < close_i
    # Station pose must match its predecessor (firmware "err mq station pose").
    assert call["wps"][close_i] == call["wps"][close_i - 1]
