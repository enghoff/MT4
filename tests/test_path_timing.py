"""Per-leg timing of a queued (mq) path -- the stacking label defect.

One queued path is one blocking call, so the recorder has to split the single
measured duration back across its legs after the fact. Splitting it by
Cartesian distance put a 171-degree single-tick discontinuity into the
recorded stacking corpus: `routed_travel` applies the face-align wrist angle
on the leg *after* the wrist-held lift, and that leg is often a ~1mm height
adjustment. By distance it deserves no time; by motor steps an 80-degree
wrist sweep is the largest single piece of work in the whole carry.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_jog.kinematics import STEPS_PER_DEG
from mt4_vision.pickplace import MoveEvent, _emit_path_waypoints, _leg_step_counts

# Measured off a live level-1 carry: grab pose, lift to safe_z holding the
# wrist, then a 1.2mm nudge to hover height that also squares the wrist.
GRAB = (69.3, -207.3, 127.2)
LIFT = (69.3, -207.3, 155.0)
HOVER = (69.3, -207.3, 156.2)
START_J4 = -80.8


def durations(wps, j4_list, speeds, span=12.0, start_j4=START_J4):
    events: list[MoveEvent] = []
    _emit_path_waypoints(events.append, "carry", GRAB, wps, j4_list, speeds,
                         0.0, span, start_j4_world=start_j4)
    return [e.ended_at - e.started_at for e in events]


# ---------------------------------------------------------------------------
# step counting
# ---------------------------------------------------------------------------


def test_wrist_only_leg_is_not_free():
    """The regression itself: a leg that barely moves the TCP but swings the
    wrist must not be counted as ~zero work."""
    counts = _leg_step_counts(GRAB, [LIFT, HOVER], ["wrist", 0.0], START_J4)
    assert counts is not None
    assert counts[1] > counts[0], (
        f"the 1.2mm wrist leg ({counts[1]:.0f} steps) should outweigh the "
        f"27.8mm lift ({counts[0]:.0f} steps)"
    )
    # 80.8 deg of world yaw at j1 = atan2(-207.3, 69.3); the joint turns by
    # the world change minus the (zero) j1 change over this vertical leg.
    assert counts[1] > 3000


def test_wrist_sentinel_legs_hold_the_joint():
    """"wrist" holds the J4 *joint* across a J1 swing, so a pure lift with no
    base rotation costs no J4 steps at all."""
    counts = _leg_step_counts(GRAB, [LIFT], ["wrist"], START_J4)
    assert counts is not None
    lift_deg = counts[0] / STEPS_PER_DEG[1]
    assert 0.0 < lift_deg < 90.0  # J2/J3 only, a real but bounded amount


def test_unknown_start_wrist_does_not_invent_a_swing():
    """Callers without a TCP read pass None; the wrist must then contribute
    nothing rather than a fabricated delta from an assumed zero."""
    known = _leg_step_counts(GRAB, [LIFT, HOVER], ["wrist", 0.0], START_J4)
    unknown = _leg_step_counts(GRAB, [LIFT, HOVER], ["wrist", 0.0], None)
    assert known is not None and unknown is not None
    assert unknown[1] < known[1]


def test_unreachable_waypoint_reports_no_counts():
    """So the caller can fall back to distance rather than guess."""
    assert _leg_step_counts(GRAB, [(9000.0, 0.0, 100.0)], [None], START_J4) is None


# ---------------------------------------------------------------------------
# time allocation
# ---------------------------------------------------------------------------


def test_the_span_is_conserved_and_legs_are_contiguous():
    events: list[MoveEvent] = []
    _emit_path_waypoints(events.append, "carry", GRAB, [LIFT, HOVER],
                         ["wrist", 0.0], [700, 700], 100.0, 112.0,
                         start_j4_world=START_J4)
    assert events[0].started_at == 100.0
    assert events[-1].ended_at == 112.0
    for a, b in zip(events, events[1:]):
        assert a.ended_at == b.started_at


def test_slow_approach_leg_gets_time_proportional_to_its_speed():
    """The final descend runs at approach speed; identical step counts at a
    slower speed must take proportionally longer."""
    wps = [LIFT, (69.3, -207.3, 182.8)]          # two equal-length lifts
    fast = durations(wps, [None, None], [700, 700])
    slow = durations(wps, [None, None], [700, 2400])
    assert slow[1] / slow[0] > fast[1] / fast[0]


def test_distance_weighting_would_have_starved_the_wrist_leg():
    """Guards the fix by pinning what the old behaviour looked like: on
    distance alone the wrist leg is 1.2mm against 27.8mm, about 4% of the
    path, and the recorded jump followed from exactly that."""
    by_distance = math.dist(LIFT, HOVER) / (math.dist(GRAB, LIFT) + math.dist(LIFT, HOVER))
    assert by_distance < 0.05
    got = durations([LIFT, HOVER], ["wrist", 0.0], [700, 700])
    assert got[1] / sum(got) > 0.5


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
