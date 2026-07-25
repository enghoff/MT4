"""Leg-composition tests for pickplace.routed_travel (no hardware).

Verifies how the queued `mq` path is assembled -- which legs are appended,
which run slow, and which wrist each leg holds -- without touching a real
arm. Uses a recording fake client plus the real StackPlanner geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.pickplace import routed_travel
from mt4_vision.stackpath import StackPlanner

# Live calibration values (vision_calibration.json, 2026-07).
CALIB = SimpleNamespace(
    pick_z=127.2,
    cube_height_mm=20.0,
    safe_z=155.0,
    travel_speed_us=700,
    approach_speed_us=2400,
)
MARKER3 = (153.6, 156.9)


class _Tcp:
    def __init__(self, x, y, z, j4=0.0):
        self.x, self.y, self.z, self.j4 = x, y, z, j4


class _RecordingClient:
    """Captures each move_path() call's (waypoints, j4, speed_us)."""

    def __init__(self, tcp):
        self._tcp = tcp
        self.calls: list[tuple[list, object, object]] = []

    def get_tcp(self):
        return self._tcp

    def move_path(self, waypoints, j4=None, grip=0, speed_us=0, timeout=None):
        self.calls.append((list(waypoints), j4, speed_us))
        return {"ok": True}


def _planner() -> StackPlanner:
    return StackPlanner(CALIB, *MARKER3)


def test_descend_leg_appended_and_slow():
    client = _RecordingClient(_Tcp(200.0, 0.0, 260.0))
    routed_travel(
        client, CALIB, _planner(), 220.0, -60.0, CALIB.safe_z, 0,
        descend=(220.0, -60.0, CALIB.pick_z), step="t",
    )
    wps, _j4, speed = client.calls[-1]
    assert wps[-1] == (220.0, -60.0, CALIB.pick_z)  # descend appended last
    assert len(speed) == len(wps)
    assert speed[-1] == int(CALIB.approach_speed_us)  # only the drop is slow
    assert all(s == CALIB.travel_speed_us for s in speed[:-1])


def test_then_and_descend_both_appended():
    client = _RecordingClient(_Tcp(200.0, 0.0, 260.0))
    routed_travel(
        client, CALIB, _planner(), 220.0, -60.0, CALIB.safe_z, 0,
        then=[(230.0, -50.0, CALIB.safe_z)],
        descend=(230.0, -50.0, CALIB.pick_z), step="t",
    )
    wps, _j4, speed = client.calls[-1]
    assert wps[-2] == (230.0, -50.0, CALIB.safe_z)  # then hop
    assert wps[-1] == (230.0, -50.0, CALIB.pick_z)  # slow descend
    assert speed[-1] == int(CALIB.approach_speed_us)
    assert all(s == CALIB.travel_speed_us for s in speed[:-1])


def test_final_j4_holds_from_arrival_onward():
    # a->b detours around the level-6 column, so the route has a genuine
    # transit leg (wrist) distinct from the arrival leg (final_j4).
    planner = _planner()
    a = (60.0, 250.0, CALIB.safe_z)
    b = (250.0, 60.0, CALIB.safe_z)
    route = planner.route(a, b, 6)
    assert route is not None and len(route) >= 2  # premise for the test
    client = _RecordingClient(_Tcp(*a))
    routed_travel(
        client, CALIB, planner, b[0], b[1], b[2], 6,
        final_j4=33.0, then=[(b[0], b[1], b[2] + 5.0)], step="t",
    )
    _wps, j4, _speed = client.calls[-1]
    n_transit = len(route) - 1
    assert isinstance(j4, list)
    assert j4[:n_transit] == ["wrist"] * n_transit  # transit legs hold wrist
    assert all(v == 33.0 for v in j4[n_transit:])  # arrival + then hold final_j4


def test_plain_transit_collapses_to_scalars():
    # No then/descend/final_j4: uniform wrist + travel speed collapse to
    # scalars, matching the pre-fusion wire form exactly.
    client = _RecordingClient(_Tcp(200.0, 0.0, 260.0))
    routed_travel(
        client, CALIB, _planner(), 220.0, -60.0, CALIB.safe_z, 0, step="t"
    )
    _wps, j4, speed = client.calls[-1]
    assert j4 == "wrist"
    assert speed == CALIB.travel_speed_us
