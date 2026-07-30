"""Joint and pin map for MT4 custom jog firmware."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PORT: str | None = None
DEFAULT_BAUD = 115200

J1_HOME_CENTER_STEPS = 4580
J2_HOME_PULLOFF_STEPS = 1000
J3_HOME_PULLOFF_STEPS = 500

# Soft joint step limits + desk plane. Mirror firmware config.h
# MT4_JOINT_SOFT_* / MT4_GROUND_Z_MM. J2/J3 counters are limit-referenced
# (steps=0 at J2 switch / J3 interference). J1 switch-side min is replaced
# at home with -J1_HOME_CENTER_STEPS; J2 min is forced to 0. Ground lowered
# 2026-07-21 with home-angle refit (desk contact ~127mm in new FK frame).
GROUND_Z_MM = 115.0
# Soft joint limits (limit-referenced for J2/J3). Mirror firmware
# firmware/mt4_jog/src/config.h MT4_JOINT_SOFT_*.
JOINT_SOFT_MIN_STEPS: tuple[int, int, int, int] = (-4800, 0, -1550, -8100)
JOINT_SOFT_MAX_STEPS: tuple[int, int, int, int] = (4580, 3950, 1650, 8100)
# Coupled extension limit: j2_steps + j3_steps (see firmware MT4_J2_J3_SUM_*).
# Equivalent to j2_deg - j3_deg >= ~15.2° at full stretch. Values shifted
# +1500 from the old park-zero frame (1000+500 pull-off reference move).
J2_J3_SUM_MIN_STEPS = -200
J2_J3_SUM_MAX_STEPS = 4410

# All four measured 2026-07-06 (J2-J4 with a phone clinometer against the
# link; J1 by direct measurement of its yaw rotation), replacing the
# factory-EEPROM-derived guesses -- J1/J2/J3 share a physical motor/gearbox
# design (~35 steps/deg each). J3's own EEPROM setting was missing from the
# dump entirely (the old 35.556 was borrowed from unrelated extra axes), and
# J4's old value (852) was a wrong axis-letter assumption ("d" = J4).
# Per README.md: duplicated in firmware/mt4_jog/src/kinematics.{h,cpp} and
# mt4_jog/kinematics.py -- no shared config file, edit all three together.
# Zero Python importers of this copy is expected, not dead code.
STEPS_PER_DEG: tuple[float, float, float, float] = (35.0, 35.0, 35.0, 45.0)

# Enforced in jog firmware (g o / g c sweep); client only starts/stops sweep.
GRIPPER_S_OPEN = 120
GRIPPER_S_CLOSED = 285
# Mirror firmware gripper.h GRIPPER_SWEEP_RATE: S units per second the
# firmware's sweep state machine advances the servo. Full open->closed is
# therefore (285-120)/120 = 1.375s, and a 20mm-cube grip (140->240) is 0.83s.
GRIPPER_SWEEP_RATE_S_PER_S = 120

# Shared jog / `mp` move step period (microseconds between DDA ticks).
JOG_SPEED_MIN_US = 700
JOG_SPEED_MAX_US = 4000

# Mirror firmware config.h MQ_QUEUE_CAPACITY: pending `mq` waypoints the
# firmware holds behind the leg currently executing.
MQ_QUEUE_CAPACITY = 8
# Mirror firmware motion.cpp MQ_STATION_DWELL_MAX_MS: a grip station holds the
# whole queue, so the firmware caps the settle it will honor.
MQ_STATION_DWELL_MAX_MS = 5000


def gripper_sweep_ms(from_s: int, to_s: int) -> float:
    """Milliseconds the firmware's sweep takes to travel ``from_s``->``to_s``.

    For host-side *timeout budgeting only* -- never as the authority on
    whether a grip finished. A queued grip station waits on the firmware's
    own sweep state machine (which knows exactly), and a standalone
    ``Mt4Client.gripper()`` sleeps its own fixed settle; both are correct
    without this. It exists so a queued path's timeout covers the time its
    stations will spend gripping instead of tripping on them.
    """
    return abs(int(to_s) - int(from_s)) * 1000.0 / GRIPPER_SWEEP_RATE_S_PER_S


@dataclass(frozen=True)
class Joint:
    number: int
    name: str
    gcode: str
    drive: int
    direction: int
    limit_pin: int | None = None

    @property
    def label(self) -> str:
        return f"J{self.number} {self.name}"


JOINTS: tuple[Joint, ...] = (
    Joint(1, "base", "X", 23, 22, limit_pin=21),
    Joint(2, "shoulder", "Y", 25, 24, limit_pin=20),
    Joint(3, "elbow", "Z", 27, 26),
    Joint(4, "wrist", "A", 35, 36),
)

LIMIT_JOINTS: dict[str, str] = {
    f"I{j.limit_pin}": j.label for j in JOINTS if j.limit_pin is not None
}
