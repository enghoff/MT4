"""Unit tests for the `mq` wire form and move_path's per-leg normalization.

Covers the grip-station contract added for queued grasps: per-waypoint grip
and dwell, the positional speed slot a dwell forces, and the timeout budget
that keeps a path from tripping on time the firmware legitimately spends on
the gripper. No serial, no hardware -- _move_path_locked is stubbed out.

Run: python tests/test_move_path.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_jog.client import (
    MOVE_TIMEOUT_S,
    Mt4Client,
    Mt4ClientError,
    _per_waypoint,
    mq_wire_cmd,
)
from mt4_jog.joints import (
    GRIPPER_S_CLOSED,
    GRIPPER_S_OPEN,
    MQ_STATION_DWELL_MAX_MS,
    gripper_sweep_ms,
)


class RecordingClient(Mt4Client):
    """Captures what move_path() would have sent, without a serial port."""

    def __init__(self) -> None:
        super().__init__(port="COM_TEST")
        self.calls: list[dict[str, object]] = []

    def _move_path_locked(
        self, waypoints, j4_tokens, grips, speeds, dwells, timeout
    ):
        self.calls.append(
            {
                "waypoints": list(waypoints),
                "j4_tokens": list(j4_tokens),
                "grips": list(grips),
                "speeds": list(speeds),
                "dwells": list(dwells),
                "timeout": timeout,
            }
        )
        return {"ok": True}

    def last(self) -> dict[str, object]:
        assert self.calls, "move_path did not reach _move_path_locked"
        return self.calls[-1]


WPS3 = [(200.0, 0.0, 200.0), (200.0, 0.0, 118.0), (200.0, 0.0, 118.0)]


def expect_error(fn, needle: str) -> None:
    try:
        fn()
    except Mt4ClientError as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc!r}"
        return
    raise AssertionError(f"expected Mt4ClientError containing {needle!r}")


# -- wire form ------------------------------------------------------------


def test_wire_plain_leg() -> None:
    """No speed, no dwell: five fields, firmware-precision coordinates."""
    assert mq_wire_cmd(213.4, -58.1, 118.0, "127.20") == (
        "mq 213.40 -58.10 118.00 127.20 0"
    )


def test_wire_speed_only() -> None:
    assert mq_wire_cmd(200.0, 0.0, 200.0, "w", 0, 700) == (
        "mq 200.00 0.00 200.00 w 0 700"
    )


def test_wire_dwell_forces_speed_slot() -> None:
    """The wire is positional: a station with no speed override still needs an
    explicit 0, or firmware reads the dwell as a step period."""
    cmd = mq_wire_cmd(213.4, -58.1, 118.0, "127.20", 262, 0, 150)
    assert cmd == "mq 213.40 -58.10 118.00 127.20 262 0 150"
    assert cmd.split()[-2:] == ["0", "150"]


def test_wire_dwell_with_speed() -> None:
    assert mq_wire_cmd(213.4, -58.1, 118.0, "127.20", 262, 2400, 150) == (
        "mq 213.40 -58.10 118.00 127.20 262 2400 150"
    )


def test_wire_respects_64_byte_line_buffer() -> None:
    """Firmware line_buf is 64 bytes; a truncated line fails to parse (with
    the usage string, not a clear error), so an over-long command is raised
    here rather than sent.

    At the firmware's 0.1mm precision this is a backstop against absurd input,
    not a live constraint -- real MT4 coordinates are 3 digits, so even the
    widest realistic station leaves ~10 bytes of headroom. Both directions are
    asserted so a future field added to the wire form trips the second half.
    """
    assert (
        len(mq_wire_cmd(-150.25, -250.75, 115.00, "-179.99", 285, 2400, 5000))
        <= 54
    )
    big = -1234567.89
    expect_error(
        lambda: mq_wire_cmd(big, big, big, f"{big:.2f}", 285, 2400, 5000),
        "64-byte",
    )


# -- per-waypoint normalization ------------------------------------------


def test_scalar_grip_is_first_leg_only() -> None:
    """Deliberately NOT the broadcast rule j4/speed_us follow: re-commanding
    the same grip on every leg would be meaningless."""
    c = RecordingClient()
    c.move_path(WPS3, grip=240)
    assert c.last()["grips"] == [240, 0, 0]


def test_scalar_grip_zero_stays_zero() -> None:
    c = RecordingClient()
    c.move_path(WPS3)
    assert c.last()["grips"] == [0, 0, 0]


def test_grip_list_is_per_leg() -> None:
    c = RecordingClient()
    c.move_path(WPS3, grip=[0, 0, 240], dwell_ms=[0, 0, 150])
    assert c.last()["grips"] == [0, 0, 240]
    assert c.last()["dwells"] == [0, 0, 150]


def test_dwell_scalar_broadcasts() -> None:
    """Unlike grip, dwell follows the ordinary broadcast rule."""
    c = RecordingClient()
    c.move_path(WPS3, dwell_ms=150)
    assert c.last()["dwells"] == [150, 150, 150]


def test_speed_and_j4_lists_still_work() -> None:
    """The pre-existing mixed-leg behaviour must be untouched."""
    c = RecordingClient()
    c.move_path(WPS3, j4=["wrist", 12.0, 12.0], speed_us=[700, 700, 2400])
    assert c.last()["j4_tokens"] == ["w", "12.00", "12.00"]
    assert c.last()["speeds"] == [700, 700, 2400]


def test_wrong_length_list_rejected() -> None:
    c = RecordingClient()
    expect_error(
        lambda: c.move_path(WPS3, dwell_ms=[0, 150]), "2 value(s) for 3"
    )
    expect_error(lambda: c.move_path(WPS3, grip=[240, 0]), "2 value(s) for 3")


def test_per_waypoint_broadcasts_sentinels_whole() -> None:
    """A bare 'wrist'/None is one value, not an iterable to spread."""
    assert _per_waypoint("wrist", 3, "j4") == ["wrist"] * 3
    assert _per_waypoint(None, 2, "j4") == [None, None]


# -- validation ----------------------------------------------------------


def test_out_of_range_grip_rejected_in_any_slot() -> None:
    c = RecordingClient()
    expect_error(
        lambda: c.move_path(WPS3, grip=[0, 0, GRIPPER_S_CLOSED + 1]), "grip must be"
    )
    expect_error(
        lambda: c.move_path(WPS3, grip=[GRIPPER_S_OPEN - 1, 0, 0]), "grip must be"
    )


def test_dwell_bounds() -> None:
    c = RecordingClient()
    expect_error(lambda: c.move_path(WPS3, dwell_ms=-1), "dwell_ms must be")
    expect_error(
        lambda: c.move_path(WPS3, dwell_ms=MQ_STATION_DWELL_MAX_MS + 1),
        "dwell_ms must be",
    )
    c.move_path(WPS3, dwell_ms=MQ_STATION_DWELL_MAX_MS)
    assert c.last()["dwells"][0] == MQ_STATION_DWELL_MAX_MS


# -- timeout budget ------------------------------------------------------


def test_no_station_leaves_timeout_alone() -> None:
    c = RecordingClient()
    c.move_path(WPS3)
    assert c.last()["timeout"] == MOVE_TIMEOUT_S


def test_each_station_adds_worst_case_sweep_plus_its_dwell() -> None:
    """Worst-case sweep, not the true S delta -- knowing the delta would need
    a `?` probe, which move_path exists to avoid."""
    c = RecordingClient()
    c.move_path(WPS3, grip=[0, 0, 240], dwell_ms=[0, 0, 150])
    per_station = gripper_sweep_ms(GRIPPER_S_OPEN, GRIPPER_S_CLOSED) + 150
    assert abs(c.last()["timeout"] - (MOVE_TIMEOUT_S + per_station / 1000.0)) < 1e-9

    c.move_path(WPS3, grip=[0, 240, 140], dwell_ms=[0, 150, 150])
    assert abs(c.last()["timeout"] - (MOVE_TIMEOUT_S + 2 * per_station / 1000.0)) < 1e-9


def test_sweep_ms_matches_firmware_rate() -> None:
    """GRIPPER_SWEEP_RATE is 120 S/s, so a full 120..285 travel is 1.375s and
    a 20mm-cube grip (140->240) is 833ms -- the numbers a flat 700ms
    assumption would have under-waited."""
    assert abs(gripper_sweep_ms(GRIPPER_S_OPEN, GRIPPER_S_CLOSED) - 1375.0) < 0.5
    assert abs(gripper_sweep_ms(140, 240) - 833.33) < 0.5
    assert gripper_sweep_ms(240, 140) == gripper_sweep_ms(140, 240)
    assert gripper_sweep_ms(200, 200) == 0.0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
