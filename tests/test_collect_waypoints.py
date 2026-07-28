"""Interpolation tests for mt4_pi.collect.recorder.WaypointLog (no hardware).

The log's whole job is turning a handful of blocking pick()/place() calls
into a per-frame pose label, so the cases that matter are the ones where
"when did the arm actually move?" differs from "when did the call return?".
Time is driven by a fake monotonic clock -- these must not depend on how
long the test host takes to run them.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN
from mt4_jog.status import TcpPose
from mt4_pi.collect import recorder
from mt4_pi.collect.recorder import WaypointLog
from mt4_vision.pickplace import MoveEvent

# Homed FK TCP (CLAUDE.md): the pose an episode actually seeds from.
SEED = TcpPose(x=190.0, y=0.0, z=226.0, j4=0.0, grip=GRIPPER_S_OPEN, speed=800)


class FakeClock:
    """Stand-in for time.monotonic() with an explicit cursor."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(recorder.time, "monotonic", fake)
    return fake


def test_dead_time_before_a_leg_is_held_not_ramped(clock):
    """The pick->place handover stalls ~0.5s on serial round trips with the
    arm stationary. Ticks landing in it must read the OLD pose, not a
    fraction of the way to the new one."""
    log = WaypointLog(SEED)
    stall_end = clock.advance(0.5)          # ensure_homed + get_tcp + j4 resolve
    move_end = clock.advance(1.5)           # the leg itself
    log.on_waypoint(MoveEvent("move to safe height", 300.0, 0.0, 226.0, started_at=stall_end))

    # Anywhere inside the stall: still parked at the seed.
    for t in (1000.0, 1000.25, 1000.49):
        assert log.state_at(t)["x"] == pytest.approx(190.0)
    # Halfway through the leg proper: halfway along it.
    assert log.state_at(stall_end + 0.75)["x"] == pytest.approx(245.0)
    assert log.state_at(move_end)["x"] == pytest.approx(300.0)


def test_leg_interpolates_by_its_own_measured_duration(clock):
    """A slow _approach and a fast _travel must each ramp over their own
    span -- not a shared average, and not the whole episode."""
    log = WaypointLog(SEED)
    fast_start = clock.t
    clock.advance(0.4)
    log.on_waypoint(MoveEvent("travel", 250.0, 0.0, 226.0, started_at=fast_start))
    slow_start = clock.t
    clock.advance(3.0)
    log.on_waypoint(MoveEvent("descend", 250.0, 0.0, 130.0, started_at=slow_start))

    assert log.state_at(fast_start + 0.2)["x"] == pytest.approx(220.0)
    assert log.state_at(slow_start + 1.5)["z"] == pytest.approx(178.0)


def test_missing_fields_hold_previous_value(clock):
    """MoveEvent's None == unchanged convention: a grip-only event must not
    disturb the pose channels."""
    log = WaypointLog(SEED)
    start = clock.t
    clock.advance(0.8)                       # GRIPPER_SETTLE_S
    log.on_waypoint(MoveEvent("grip", grip=GRIPPER_S_CLOSED, started_at=start))

    end = log.state_at(clock.t)
    assert end["x"] == pytest.approx(190.0)
    assert end["z"] == pytest.approx(226.0)
    assert end["grip"] == pytest.approx(1.0)
    # Gripper ramps across the settle window, not before it.
    assert log.state_at(start)["grip"] == pytest.approx(0.0)
    assert log.state_at(start + 0.4)["grip"] == pytest.approx(0.5)


def test_state_at_returns_copies_not_internal_samples(clock):
    """Callers round and mutate what they get back; the log must not alias
    its own samples into meta.json / ticks.jsonl."""
    log = WaypointLog(SEED)
    clock.advance(1.0)
    log.on_waypoint(MoveEvent("travel", 250.0, 0.0, 226.0, started_at=1000.0))

    before = log.state_at(0.0)               # clamps to the seed sample
    after = log.state_at(9999.0)             # clamps to the last sample
    before["z"] = -999.0
    after["z"] = -999.0
    assert log.state_at(0.0)["z"] == pytest.approx(226.0)
    assert log.state_at(9999.0)["z"] == pytest.approx(226.0)


def test_waypoint_count_counts_events_not_samples(clock):
    """Each event adds a dead-time hold as well as its endpoint, so
    n_waypoints must come from a counter, not len(_log)."""
    log = WaypointLog(SEED)
    assert log.waypoint_count() == 0
    for _ in range(3):
        start = clock.t
        clock.advance(0.5)
        log.on_waypoint(MoveEvent("travel", 250.0, 0.0, 226.0, started_at=start))
    assert log.waypoint_count() == 3


def test_event_without_started_at_still_interpolates(clock):
    """travel_orthogonal and any future unstamped caller must degrade to the
    old whole-span ramp rather than blowing up."""
    log = WaypointLog(SEED)
    clock.advance(2.0)
    log.on_waypoint(MoveEvent("unstamped", 290.0, 0.0, 226.0))
    assert log.state_at(1001.0)["x"] == pytest.approx(240.0)


def test_zero_duration_leg_does_not_divide_by_zero(clock):
    """A leg that returns within one clock tick (already at target: firmware
    no-ops it) must clamp, not explode."""
    log = WaypointLog(SEED)
    log.on_waypoint(MoveEvent("no-op", 190.0, 0.0, 226.0, started_at=clock.t))
    log.on_waypoint(MoveEvent("real", 250.0, 0.0, 226.0, started_at=clock.t))
    assert log.state_at(clock.t)["x"] == pytest.approx(190.0)
    assert log.state_at(clock.t + 1.0)["x"] == pytest.approx(250.0)


def test_multi_leg_path_uses_ended_at_not_callback_time(clock):
    """_emit_path_waypoints reports several legs of one queued move_path()
    call back-to-back, all AFTER the whole call has already returned --
    their on_waypoint() calls land within the same instant of real time,
    but each carries its own started_at/ended_at spanning when that leg
    actually ran. Stamping with time.monotonic() instead of event.ended_at
    collapses every leg onto ~the same timestamp: exactly the "arm
    teleports through the whole route" failure MoveEvent's docstring warns
    about, and what left ep_000109's pick->place transit logged as an
    instantaneous 82mm/66deg jump while the camera still showed the old
    pose for the next ~1.2s."""
    log = WaypointLog(SEED)
    call_start = clock.advance(0.5)             # dead time before the call
    leg1_end = call_start + 1.0                  # lift leg: 1s of its own
    leg2_end = call_start + 3.0                  # travel leg: 2s of its own
    clock.advance(3.0)                            # move_path() finally returns
    log.on_waypoint(MoveEvent(
        "transit leg 1/2", 190.0, 0.0, 260.0,
        started_at=call_start, ended_at=leg1_end,
    ))
    log.on_waypoint(MoveEvent(
        "transit leg 2/2", 300.0, 100.0, 130.0,
        started_at=leg1_end, ended_at=leg2_end,
    ))

    # Leg 1's midpoint: still ramping the FIRST leg, not already at leg 2's
    # target and not frozen at the seed either.
    mid1 = log.state_at(call_start + 0.5)
    assert mid1["x"] == pytest.approx(190.0)
    assert mid1["z"] == pytest.approx(243.0)
    # Leg 2's midpoint: partway from leg 1's end to leg 2's target.
    mid2 = log.state_at(call_start + 2.0)
    assert mid2["x"] == pytest.approx(245.0)
    assert mid2["y"] == pytest.approx(50.0)
    assert mid2["z"] == pytest.approx(195.0)
    # Only at leg 2's real end does the log claim arrival.
    assert log.state_at(leg2_end)["x"] == pytest.approx(300.0)


def test_gripper_fraction_spans_the_firmware_range():
    assert recorder._gripper_fraction(GRIPPER_S_OPEN) == pytest.approx(0.0)
    assert recorder._gripper_fraction(GRIPPER_S_CLOSED) == pytest.approx(1.0)
    # Out-of-range servo reads clamp rather than extrapolating.
    assert recorder._gripper_fraction(GRIPPER_S_OPEN - 50) == pytest.approx(0.0)
    assert recorder._gripper_fraction(GRIPPER_S_CLOSED + 50) == pytest.approx(1.0)
