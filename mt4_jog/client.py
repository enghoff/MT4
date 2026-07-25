"""Long-lived serial client for MT4 jog firmware."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from mt4_jog.joints import (
    DEFAULT_BAUD,
    DEFAULT_PORT,
    GRIPPER_S_CLOSED,
    GRIPPER_S_OPEN,
    GROUND_Z_MM,
    J1_HOME_CENTER_STEPS,
    J2_HOME_PULLOFF_STEPS,
    JOG_SPEED_MAX_US,
    JOG_SPEED_MIN_US,
    MQ_QUEUE_CAPACITY,
)
from mt4_jog.serial import (
    FirmwareNotReadyError,
    STATUS_WAIT_S,
    SerialGoneError,
    await_firmware_alive,
    close_quiet,
    open_serial,
    read_lines,
    send,
    write_raw,
)
from mt4_jog.status import Mt4Status, TcpPose, parse_status_lines, parse_tcp_line

if TYPE_CHECKING:
    import serial as serial_module

COMMAND_WAIT_S = 0.5
GRIP_WAIT_S = 1.0
# Firmware acks an absolute `g <value>` as soon as it parses the command, not
# once the servo physically arrives -- and read_lines() (see serial.py) exits
# on serial silence, which follows that immediate ack by ~100ms, long before
# the ~700ms the gripper actually takes to reach its target. Without this,
# gripper() returns while the servo is still mid-travel, so pick()/place()'s
# next move_to() starts while the grip hasn't closed (or opened) yet.
GRIPPER_SETTLE_S = 0.8
# Homing seeks limit switches with no position feedback beforehand, so it
# needs the same generous ceiling as jog.py's HOME_WAIT_S.
HOME_TIMEOUT_S = 180.0
# If firmware hasn't acked a `home` command (home start/fail/err) within
# this long, assume the line was swallowed (e.g. an MCU reset between
# connect and the write) and re-send once. Harmless if homing actually is
# in flight: do_home()'s serial_abort() reads and discards any line that
# isn't "!"/"stop".
HOME_ACK_RESEND_S = 5.0
# Timeout for coordinated m/mp moves to complete.
MOVE_TIMEOUT_S = 30.0
# Cap line collection so a chatty firmware stream cannot block forever.
READ_HARD_LIMIT_S = 1.0
# Ceiling on how long a single read_lines() poll waits while draining an
# async move/home completion line, so the overall timeout loop stays responsive.
POLL_WAIT_S = 0.3


class Mt4ClientError(Exception):
    """Raised when the arm is unreachable or returns an unexpected response."""


def _j4_wire_token(j4: float | str | None) -> str:
    """Wire form of a move's J4 field: a number, or a firmware sentinel.

    None / "hold" -> `h`: hold the world yaw the arm has when the leg is
    planned. "wrist" -> `w`: hold the J4 *joint* angle across the leg's J1
    swing (what big base swings need -- a world hold there drives joint
    J4 = world - j1 past its soft limits; firmware-native version of
    pickplace.j4_preserve_wrist). Both resolve firmware-side at leg-plan
    time, so no `?` status probe is needed and queued legs resolve against
    wherever the previous leg actually ended.
    """
    if j4 is None or j4 == "hold":
        return "h"
    if j4 == "wrist":
        return "w"
    if isinstance(j4, str):
        raise Mt4ClientError(
            f"j4 must be a number, None, 'hold', or 'wrist' (got {j4!r})"
        )
    return f"{float(j4):.2f}"


def _per_waypoint(value: object, n: int, name: str) -> list:
    """Normalize a move_path field to one value per waypoint (length ``n``).

    A ``list``/``tuple`` is taken as an explicit per-waypoint sequence (its
    length must match the waypoint count); anything else -- including a bare
    ``"wrist"``/``"hold"`` sentinel or ``None`` -- is a single value
    broadcast to every leg. The firmware carries speed and J4 mode on each
    queued ``mq`` leg independently (see MpWaypoint in motion.cpp), so this
    is what lets one queued path mix, e.g., fast transit legs with a slow
    final descent, or a per-leg wrist mode.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise Mt4ClientError(
                f"move_path: {name} has {len(value)} value(s) for {n} waypoint(s)"
            )
        return list(value)
    return [value] * n


class Mt4Client:
    """Thread-safe owner of the MT4 serial connection."""

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
    ) -> None:
        self.port = port
        self.baud = baud
        self._ser: serial_module.Serial | None = None
        # Reentrant so methods like move_to() can call _get_status_unlocked()
        # (to default J4 / check homed) while already holding the lock.
        self._lock = threading.RLock()
        self._interrupt = threading.Event()
        # Optional duck-typed sink (e.g. mt4_vision.preview.LiveFeed) a
        # caller can point at to get every move_to()/move_path() drawn as a
        # live overlay: set_trajectory(waypoints) when the move is sent,
        # clear_trajectory() once the call returns. No import here so
        # mt4_jog stays independent of mt4_vision.
        self.trajectory_sink = None
        # Last robot-frame TCP (x, y, z) this client has seen reported by
        # firmware (status probe or a move's own final "tcp ..." line).
        # Prepended to the trajectory_sink overlay so it shows the leg from
        # wherever the arm actually is, not just the requested waypoints.
        # None until the first status read or completed move this session.
        self._last_tcp: tuple[float, float, float] | None = None

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def ensure_connected(self) -> None:
        with self._lock:
            self._ensure_connected_unlocked()

    def _ensure_connected_unlocked(self) -> None:
        if self.connected:
            return
        try:
            self._ser = open_serial(self.port, self.baud)
        except Exception as exc:
            raise Mt4ClientError(
                f"Could not open {self.port} @ {self.baud} baud: {exc}"
            ) from exc
        try:
            await_firmware_alive(self._ser, port_label=self.port or "auto-detected port")
        except FirmwareNotReadyError as exc:
            close_quiet(self._ser)
            self._ser = None
            raise Mt4ClientError(str(exc)) from exc
        except SerialGoneError as exc:
            close_quiet(self._ser)
            self._ser = None
            raise Mt4ClientError(str(exc)) from exc
        except Exception:
            close_quiet(self._ser)
            self._ser = None
            raise

    def close(self) -> None:
        with self._lock:
            close_quiet(self._ser)
            self._ser = None

    def _require_serial(self) -> serial_module.Serial:
        if not self.connected:
            raise Mt4ClientError("Not connected to the arm")
        assert self._ser is not None
        return self._ser

    def _mark_gone_unlocked(self, exc: SerialGoneError) -> None:
        close_quiet(self._ser)
        self._ser = None
        raise Mt4ClientError(str(exc)) from exc

    def _send_and_collect(self, cmd: str, wait: float) -> list[str]:
        ser = self._require_serial()
        try:
            return send(
                ser,
                cmd,
                wait=wait,
                hard_limit=wait + READ_HARD_LIMIT_S,
            )
        except SerialGoneError as exc:
            self._mark_gone_unlocked(exc)
            raise  # unreachable; keeps type-checkers happy

    def _drain_serial_unlocked(self) -> list[str]:
        """Discard any unread firmware lines (stale status, prior home ok, etc.)."""
        ser = self._require_serial()
        drained: list[str] = []
        try:
            while True:
                batch = read_lines(ser, 0.05, hard_limit=0.12)
                if not batch:
                    break
                drained.extend(batch)
        except SerialGoneError as exc:
            self._mark_gone_unlocked(exc)
            raise
        return drained

    def _get_status_unlocked(self, *, retries: int = 1) -> Mt4Status:
        self._ensure_connected_unlocked()
        status = Mt4Status()
        for attempt in range(retries):
            if attempt > 0:
                time.sleep(0.15)
            lines = self._send_and_collect("?", wait=STATUS_WAIT_S)

            # `?` and `pos` both print the same joint-position + derived
            # `tcp x=...` line (see firmware print_joint_pos()), so retrying
            # with the lighter `pos` query is a real fallback if `?`'s fuller
            # reply got cut off. There's no separate firmware "tcp" command.
            if not lines:
                lines = self._send_and_collect("pos", wait=1.5)

            status = parse_status_lines(lines)
            status.parse_failed = status.tcp is None and not status.joints
            if not status.parse_failed:
                break
        if status.tcp is not None:
            self._last_tcp = (status.tcp.x, status.tcp.y, status.tcp.z)
        return status

    def get_status(self) -> Mt4Status:
        with self._lock:
            return self._get_status_unlocked(retries=3)

    def get_tcp(self) -> TcpPose:
        status = self.get_status()
        if status.tcp is None:
            raise Mt4ClientError("Could not read TCP pose")
        return status.tcp

    def stop(self) -> list[str]:
        with self._lock:
            self._ensure_connected_unlocked()
            return self._send_and_collect("stop", wait=COMMAND_WAIT_S)

    def j4_zero(self) -> dict[str, object]:
        """Rewrite firmware J4 origin so current pose reports world j4 = 0.

        No motion. After this, jaws aligned with the arm (at the calibration
        pose) read as world-frame J4 ≈ 0; face-align picks can use offset 0.
        Survives subsequent ``home``; lost on power cycle until re-run.
        """
        with self._lock:
            self._ensure_connected_unlocked()
            lines = self._send_and_collect("j4zero", wait=COMMAND_WAIT_S)
            for line in lines:
                if line.startswith("err"):
                    return {"ok": False, "error": line, "lines": lines}
                if line.startswith("ok j4zero") or (
                    line.startswith("ok ") and "pos J" in line
                ):
                    # Firmware prints "ok j4zero pos ..." then a tcp line.
                    status = self._get_status_unlocked(retries=2)
                    return {
                        "ok": True,
                        "lines": lines,
                        "tcp": status.tcp.as_dict() if status.tcp else None,
                        "joints": dict(status.joints),
                    }
            return {
                "ok": False,
                "error": "no j4zero ack (firmware may need flash)",
                "lines": lines,
            }

    def request_interrupt(self) -> None:
        """Ask an in-flight move/gripper settle to abort (e.g. shuffle home key)."""
        self._interrupt.set()

    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    def _sleep_interruptible(self, seconds: float) -> bool:
        """Sleep in short slices; return True if interrupted."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._interrupt.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
        return False

    def _abort_if_interrupted_unlocked(self, collected: list[str]) -> dict[str, object] | None:
        if not self._interrupt.is_set():
            return None
        self._send_and_collect("stop", wait=COMMAND_WAIT_S)
        return {"ok": False, "error": "interrupted", "lines": collected}

    def _await_completion(
        self,
        *,
        done_prefix: str,
        timeout: float,
        collected: list[str],
    ) -> dict[str, object]:
        """Poll for an async `<cmd> done ...` / `err ...` line after an `ok`
        ack, for `m`/`mp` moves. Returns a dict with "ok", "lines", and (on success) the
        parsed final "tcp" pose if one was printed.

        `collected` may already contain the terminal line -- a no-op move
        (e.g. all-zero relative deltas) gets "ok m" + "m done ..." back
        synchronously in the same initial read, with nothing further to
        poll for -- so every batch (including the one passed in) is scanned
        before waiting for more.
        """

        def scan(batch: list[str]) -> dict[str, object] | None:
            for line in batch:
                if line.startswith("err"):
                    return {"ok": False, "error": line, "lines": collected}
                if line.startswith(done_prefix):
                    tcp = next(
                        (parse_tcp_line(candidate) for candidate in batch),
                        None,
                    )
                    return {
                        "ok": True,
                        "lines": collected,
                        "tcp": tcp.as_dict() if tcp else None,
                    }
            return None

        result = scan(collected)
        if result is not None:
            return result

        ser = self._require_serial()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            aborted = self._abort_if_interrupted_unlocked(collected)
            if aborted is not None:
                return aborted
            try:
                lines = read_lines(
                    ser, POLL_WAIT_S, hard_limit=POLL_WAIT_S + READ_HARD_LIMIT_S
                )
            except SerialGoneError as exc:
                self._mark_gone_unlocked(exc)
                raise
            collected.extend(lines)
            result = scan(lines)
            if result is not None:
                return result
        return {"ok": False, "error": f"Timed out waiting for {done_prefix!r}", "lines": collected}

    def home(
        self,
        j1_center: int = J1_HOME_CENTER_STEPS,
        j2_pull: int = J2_HOME_PULLOFF_STEPS,
        timeout: float = HOME_TIMEOUT_S,
    ) -> dict[str, object]:
        """Run on-device homing (`home <j1> <j2>`) and wait for completion."""
        with self._lock:
            self.clear_interrupt()
            self._ensure_connected_unlocked()
            ser = self._require_serial()
            self._drain_serial_unlocked()
            cmd = f"home {j1_center} {j2_pull}\n".encode("ascii")
            try:
                write_raw(ser, cmd)
            except SerialGoneError as exc:
                self._mark_gone_unlocked(exc)
                raise
            collected: list[str] = []
            saw_home_start = False
            resent = False
            ack_deadline = time.monotonic() + HOME_ACK_RESEND_S
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if (
                    not saw_home_start
                    and not resent
                    and time.monotonic() > ack_deadline
                ):
                    # No ack yet -- the command was likely swallowed (MCU
                    # reset since connect). Re-send once; if homing actually
                    # is running, the firmware discards this line unread.
                    try:
                        write_raw(ser, cmd)
                    except SerialGoneError as exc:
                        self._mark_gone_unlocked(exc)
                        raise
                    resent = True
                try:
                    lines = read_lines(
                        ser, POLL_WAIT_S, hard_limit=POLL_WAIT_S + READ_HARD_LIMIT_S
                    )
                except SerialGoneError as exc:
                    self._mark_gone_unlocked(exc)
                    raise
                collected.extend(lines)
                for line in lines:
                    if line == "home start":
                        saw_home_start = True
                    if line == "home ok" and saw_home_start:
                        self._drain_serial_unlocked()
                        status = self._get_status_unlocked(retries=3)
                        return {
                            "ok": True,
                            "lines": collected,
                            "homed": status.homed,
                            "tcp": status.tcp.as_dict() if status.tcp else None,
                            "parse_failed": status.parse_failed,
                        }
                    if line.startswith("home fail") or line.startswith("err"):
                        return {"ok": False, "error": line, "lines": collected}
            return {"ok": False, "error": "Homing timed out", "lines": collected}

    def _notify_trajectory(
        self, waypoints: list[tuple[float, float, float]], multi: bool,
    ) -> None:
        sink = self.trajectory_sink
        if sink is None:
            return
        # Lead with the arm's last known position (if any) so the overlay
        # draws the whole path starting from wherever it actually is, not
        # just the requested waypoints.
        points = [self._last_tcp, *waypoints] if self._last_tcp is not None else list(waypoints)
        # `multi` is the command kind (move_to vs move_path/mq), not the leg
        # count -- so a 1-waypoint move_path still colors as a queued path,
        # and the prepended start point can't flip a move_to's color.
        sink.set_trajectory(points, multi)

    def _clear_trajectory(self) -> None:
        sink = self.trajectory_sink
        if sink is not None:
            sink.clear_trajectory()

    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        j4: float | str | None = None,
        grip: int = 0,
        speed_us: int = 0,
        timeout: float = MOVE_TIMEOUT_S,
    ) -> dict[str, object]:
        """Absolute Cartesian move (`mp`). Requires homing this session.

        `grip=0` leaves the gripper unchanged (firmware convention). `j4`
        is a world-frame yaw in degrees, or a firmware-resolved sentinel:
        None / "hold" keeps the current world yaw through the move (same
        as `orient on`); "wrist" keeps the J4 *joint* angle across the J1
        swing (use for large base swings, where a world hold would drive
        joint J4 past its soft limits). The sentinels resolve on the
        firmware at plan time, so no `?` status probe round trip is spent
        here -- homing is validated by the firmware itself ("err not
        homed" is re-raised as Mt4ClientError to keep this method's
        pre-flight contract).
        """
        if speed_us and not JOG_SPEED_MIN_US <= speed_us <= JOG_SPEED_MAX_US:
            raise Mt4ClientError(
                f"speed_us must be 0 or {JOG_SPEED_MIN_US}-{JOG_SPEED_MAX_US}"
            )
        if grip and not GRIPPER_S_OPEN <= grip <= GRIPPER_S_CLOSED:
            raise Mt4ClientError(
                f"grip must be 0 (unchanged) or {GRIPPER_S_OPEN}-{GRIPPER_S_CLOSED}"
            )
        if z < GROUND_Z_MM - 0.05:
            raise Mt4ClientError(
                f"z={z:.1f} is below ground plane ({GROUND_Z_MM:.1f}mm)"
            )
        j4_token = _j4_wire_token(j4)

        self._notify_trajectory([(x, y, z)], multi=False)
        try:
            with self._lock:
                self._ensure_connected_unlocked()
                # Firmware line_buf is 64 bytes; full Python float repr of
                # detection XY + j4 + speed easily overflows and truncates mid-
                # token (parse fails with the usage string). Match the firmware's
                # own 0.1mm / 0.1deg reporting precision.
                cmd = f"mp {x:.2f} {y:.2f} {z:.2f} {j4_token} {int(grip)}"
                if speed_us:
                    cmd += f" {int(speed_us)}"
                if len(cmd) >= 64:
                    raise Mt4ClientError(
                        f"mp command exceeds 64-byte firmware line buffer: {cmd!r}"
                    )
                collected = self._send_and_collect(cmd, wait=1.0)
                if any(line.startswith("err not homed") for line in collected):
                    raise Mt4ClientError(
                        "Arm has not homed this session -- call home() first"
                    )
                result = self._await_completion(
                    done_prefix="mp done", timeout=timeout, collected=collected
                )
                if result.get("ok"):
                    # The arm is now at the commanded target, so anchor the
                    # next overlay's origin there directly. _await_completion's
                    # "tcp" is unreliable for this -- the firmware prints the
                    # "tcp ..." pose a few ms AFTER "mp done", usually in a
                    # later read than the one that satisfies completion, so it
                    # is typically absent here; depending on it left _last_tcp
                    # frozen and every following move_to drew from the same
                    # stale point.
                    self._last_tcp = (x, y, z)
                return result
        finally:
            self._clear_trajectory()

    def move_path(
        self,
        waypoints: list[tuple[float, float, float]],
        j4: float | str | None | list[float | str | None] = None,
        grip: int = 0,
        speed_us: int | list[int] = 0,
        timeout: float = MOVE_TIMEOUT_S,
    ) -> dict[str, object]:
        """Move through a sequence of Cartesian waypoints as one continuous
        firmware-side path, instead of a blocking `move_to()` per point.

        EVERY waypoint (the last included) goes out as `mq`: the first
        cold-starts if the arm is idle, the rest queue behind it and splice
        in without the per-waypoint stop/settle/reaccel cycle (see the `mq`
        protocol doc in firmware/mt4_jog/src/main.cpp for what is and isn't
        smoothed -- each leg still ramps down near its own end). The last
        leg must NOT be an `mp`: firmware `mp` mid-flight is an immediate
        override that drops every still-queued leg and cuts straight from
        wherever the arm is to its own target -- exactly the corner the
        queued route exists to avoid.

        Completion: the firmware emits one "mp done pos ..." each time its
        queue drains -- normally once, at the very end, but a short leg can
        finish before the next `mq` lands, draining the queue early; that
        next `mq` then cold-starts a fresh path (acked "ok mq" instead of
        "ok mq queued N"). So this counts cold-start acks and waits for
        that many "mp done" lines; anything less risks returning while the
        arm is still moving (and poisoning the next camera capture).

        `j4` is a world-frame yaw in degrees applied to every leg, or a
        firmware-resolved sentinel: None / "hold" holds world yaw, "wrist"
        holds the J4 *joint* angle leg-by-leg across each J1 swing (the
        firmware resolves either at each leg's OWN plan time, from
        wherever the previous leg actually ended -- true per-leg wrist
        behavior, which no up-front host-side value could reproduce; use
        "wrist" for transit routes with big base swings). No `?` status
        probe is sent: the firmware validates homing on every `mq` itself
        ("err not homed" comes back in the failure dict), which also makes
        this safe to use as the awaited finisher after raw `queue_move()`
        calls.

        `j4` and `speed_us` may also be given as a per-waypoint list (same
        length as `waypoints`) instead of one value broadcast to all legs --
        each queued `mq` leg carries its own speed and J4 mode on the
        firmware (MpWaypoint in motion.cpp), so a single path can run fast
        transit legs and then a slow final descent, or vary the wrist mode
        per leg. A scalar (or bare "hold"/"wrist"/None) still applies to
        every leg.

        `speed_us` is the per-leg step period (0 leaves the current one
        unchanged); each list entry must be 0 or within
        [JOG_SPEED_MIN_US, JOG_SPEED_MAX_US].

        `grip` (if nonzero) fires on the FIRST leg only, matching a single
        `move_to()` call's semantics -- for a gripper action partway through
        a route, call `gripper()` directly between `move_path()` calls
        rather than threading it through here.
        """
        if not waypoints:
            raise Mt4ClientError("move_path: at least one waypoint is required")
        if len(waypoints) > MQ_QUEUE_CAPACITY + 1:
            # Worst case (no leg completes while the rest are being sent)
            # needs len-1 firmware queue slots behind the executing leg.
            raise Mt4ClientError(
                f"move_path: at most {MQ_QUEUE_CAPACITY + 1} waypoints per "
                f"call (firmware `mq` queue holds {MQ_QUEUE_CAPACITY})"
            )
        speeds = _per_waypoint(speed_us, len(waypoints), "speed_us")
        for s in speeds:
            if s and not JOG_SPEED_MIN_US <= s <= JOG_SPEED_MAX_US:
                raise Mt4ClientError(
                    f"speed_us must be 0 or {JOG_SPEED_MIN_US}-{JOG_SPEED_MAX_US} (got {s})"
                )
        if grip and not GRIPPER_S_OPEN <= grip <= GRIPPER_S_CLOSED:
            raise Mt4ClientError(
                f"grip must be 0 (unchanged) or {GRIPPER_S_OPEN}-{GRIPPER_S_CLOSED}"
            )
        for x, y, z in waypoints:
            if z < GROUND_Z_MM - 0.05:
                raise Mt4ClientError(
                    f"z={z:.1f} is below ground plane ({GROUND_Z_MM:.1f}mm)"
                )

        j4_tokens = [
            _j4_wire_token(v) for v in _per_waypoint(j4, len(waypoints), "j4")
        ]

        self._notify_trajectory(waypoints, multi=True)
        try:
            return self._move_path_locked(waypoints, j4_tokens, grip, speeds, timeout)
        finally:
            self._clear_trajectory()

    def _move_path_locked(
        self,
        waypoints: list[tuple[float, float, float]],
        j4_tokens: list[str],
        grip: int,
        speeds: list[int],
        timeout: float,
    ) -> dict[str, object]:
        with self._lock:
            self._ensure_connected_unlocked()

            def build_cmd(i: int, x: float, y: float, z: float, g: int) -> str:
                cmd = f"mq {x:.2f} {y:.2f} {z:.2f} {j4_tokens[i]} {int(g)}"
                if speeds[i]:
                    cmd += f" {int(speeds[i])}"
                if len(cmd) >= 64:
                    raise Mt4ClientError(
                        f"mq command exceeds 64-byte firmware line buffer: {cmd!r}"
                    )
                return cmd

            collected: list[str] = []
            cold_starts = 0
            dones = 0
            saw_ack = False
            joined_existing = False

            def scan(batch: list[str]) -> str | None:
                """Fold a batch into the ack/done counters; return an err line."""
                nonlocal cold_starts, dones, saw_ack, joined_existing
                for line in batch:
                    if line.startswith("err"):
                        return line
                    if line == "ok mq":
                        saw_ack = True
                        cold_starts += 1
                    elif line.startswith("ok mq queued"):
                        if not saw_ack:
                            # Our first waypoint queued behind a path that
                            # was already in flight before this call (raw
                            # queue_move() usage); that combined path ends
                            # with one extra "mp done" of its own.
                            joined_existing = True
                        saw_ack = True
                    elif line.startswith("mp done"):
                        # A done seen before any ack of ours belongs to a
                        # pre-existing path draining as we arrived -- our
                        # first waypoint then cold-starts and its ack (which
                        # firmware prints synchronously, always before any
                        # done that could include it) hasn't shown up yet.
                        if saw_ack:
                            dones += 1
                return None

            def expected_dones() -> int:
                n = cold_starts + (1 if joined_existing else 0)
                # All acks lost to serial noise: best effort, wait for one
                # drain rather than returning with the arm still moving.
                return n if n else 1

            def fail_stopped(error: str) -> dict[str, object]:
                # Deterministic failure state: `stop` cancels the in-flight
                # path AND the pending queue without emitting "mp done"
                # (commands.cpp: stop_jog + motion_cancel_move), so no stray
                # completion line is left to confuse the next command.
                collected.extend(self._send_and_collect("stop", wait=COMMAND_WAIT_S))
                return {"ok": False, "error": error, "lines": collected}

            for i, (x, y, z) in enumerate(waypoints):
                cmd = build_cmd(i, x, y, z, grip if i == 0 else 0)
                batch = self._send_and_collect(cmd, wait=0.5)
                collected.extend(batch)
                err = scan(batch)
                if err is not None:
                    return fail_stopped(err)

            ser = self._require_serial()
            deadline = time.monotonic() + timeout
            while dones < expected_dones():
                if time.monotonic() > deadline:
                    return fail_stopped(
                        f"Timed out waiting for {expected_dones()}x 'mp done'"
                    )
                aborted = self._abort_if_interrupted_unlocked(collected)
                if aborted is not None:
                    return aborted
                try:
                    lines = read_lines(
                        ser, POLL_WAIT_S, hard_limit=POLL_WAIT_S + READ_HARD_LIMIT_S
                    )
                except SerialGoneError as exc:
                    self._mark_gone_unlocked(exc)
                    raise
                collected.extend(lines)
                err = scan(lines)
                if err is not None:
                    return fail_stopped(err)

            # The final pose's "tcp x=..." line trails the last "mp done" by
            # a few ms and may not have been read yet; grab it so the last
            # parseable tcp line in `collected` is the final pose, not an
            # intermediate drain's.
            try:
                collected.extend(read_lines(ser, 0.25, hard_limit=0.5))
            except SerialGoneError as exc:
                self._mark_gone_unlocked(exc)
                raise
            tcp = None
            for line in reversed(collected):
                tcp = parse_tcp_line(line)
                if tcp is not None:
                    break
            # Anchor the next overlay's origin at the commanded endpoint (the
            # arm is now there), matching move_to and staying correct even if
            # the trailing "tcp ..." line above was dropped or garbled. The
            # parsed tcp is still returned below for callers that want the
            # firmware-reported pose.
            self._last_tcp = waypoints[-1]
            return {
                "ok": True,
                "lines": collected,
                "tcp": tcp.as_dict() if tcp else None,
            }

    def queue_move(
        self,
        x: float,
        y: float,
        z: float,
        j4: float | str | None = None,
        grip: int = 0,
        speed_us: int = 0,
    ) -> dict[str, object]:
        """Enqueue one Cartesian waypoint (`mq`) without waiting for it (or
        anything queued ahead of it) to finish -- the raw primitive
        `move_path()` is built on. Returns as soon as the firmware acks
        ("ok mq" / "ok mq queued N" / "err ...").

        `j4` accepts the same field as `move_to()`: a world yaw in degrees,
        or None/"hold"/"wrist" -- sentinels the firmware resolves when the
        leg is planned (for a queued leg: from wherever the leg ahead of it
        ended), so no status probe is sent and an unhomed arm surfaces as
        {"ok": False, "error": "err not homed"} rather than a raised
        Mt4ClientError.

        To finish and await a hand-built queue, end with a `move_path()`
        call (a single-waypoint one works: its leg just queues behind yours
        and it counts queue-drain completions). Do NOT end with
        `move_to()`: firmware `mp` sent mid-flight is an immediate override
        that drops every still-queued leg and cuts straight from wherever
        the arm is to its own target. An `mq` queue left dangling with
        nothing awaiting it just sits there, and its eventual "mp done"
        lands unread in the serial buffer.
        """
        if speed_us and not JOG_SPEED_MIN_US <= speed_us <= JOG_SPEED_MAX_US:
            raise Mt4ClientError(
                f"speed_us must be 0 or {JOG_SPEED_MIN_US}-{JOG_SPEED_MAX_US}"
            )
        if grip and not GRIPPER_S_OPEN <= grip <= GRIPPER_S_CLOSED:
            raise Mt4ClientError(
                f"grip must be 0 (unchanged) or {GRIPPER_S_OPEN}-{GRIPPER_S_CLOSED}"
            )
        if z < GROUND_Z_MM - 0.05:
            raise Mt4ClientError(
                f"z={z:.1f} is below ground plane ({GROUND_Z_MM:.1f}mm)"
            )
        j4_token = _j4_wire_token(j4)

        with self._lock:
            self._ensure_connected_unlocked()
            cmd = f"mq {x:.2f} {y:.2f} {z:.2f} {j4_token} {int(grip)}"
            if speed_us:
                cmd += f" {int(speed_us)}"
            if len(cmd) >= 64:
                raise Mt4ClientError(
                    f"mq command exceeds 64-byte firmware line buffer: {cmd!r}"
                )
            lines = self._send_and_collect(cmd, wait=0.5)
            for line in lines:
                if line.startswith("err"):
                    return {"ok": False, "error": line, "lines": lines}
            return {"ok": True, "lines": lines}

    def set_speed(self, speed_us: int) -> dict[str, object]:
        """Set the shared jog/`m`/`mp` step period (`speed <us>`)."""
        if not JOG_SPEED_MIN_US <= speed_us <= JOG_SPEED_MAX_US:
            raise Mt4ClientError(
                f"speed_us must be {JOG_SPEED_MIN_US}-{JOG_SPEED_MAX_US}"
            )
        with self._lock:
            self._ensure_connected_unlocked()
            lines = self._send_and_collect(
                f"speed {int(speed_us)}", wait=COMMAND_WAIT_S
            )
            for line in lines:
                if line.startswith("err"):
                    return {"ok": False, "error": line, "lines": lines}
            return {"ok": True, "lines": lines}

    def move_relative(
        self,
        dj1: int,
        dj2: int,
        dj3: int,
        dj4: int,
        dgrip: int = 0,
        timeout: float = MOVE_TIMEOUT_S,
    ) -> dict[str, object]:
        """Bounded relative joint-step move (`m`). Does not require homing --
        deltas are relative to the current step counters, whatever they are.
        """
        with self._lock:
            self._ensure_connected_unlocked()
            cmd = f"m {dj1} {dj2} {dj3} {dj4} {dgrip}"
            collected = self._send_and_collect(cmd, wait=1.0)
            return self._await_completion(
                done_prefix="m done", timeout=timeout, collected=collected
            )

    def gripper(self, action: str | int) -> dict[str, object]:
        """Gripper sweep/set (`g o|c|stop|<120-285>`).

        An absolute target (int) blocks until the servo has had time to
        physically arrive (see GRIPPER_SETTLE_S) -- the firmware only acks
        that it parsed the command, not that motion finished, so callers
        that move the arm right after (pick()/place()) would otherwise race
        a gripper that's still mid-travel. Sweep actions ('open'/'close'/
        'stop') are open-ended/continuous, so they return as soon as sent.
        """
        settle = False
        if isinstance(action, str):
            normalized = action.strip().lower()
            arg = {"open": "o", "close": "c", "stop": "stop"}.get(normalized)
            if arg is None:
                raise Mt4ClientError(
                    "action must be 'open', 'close', 'stop', or an int "
                    f"{GRIPPER_S_OPEN}-{GRIPPER_S_CLOSED}"
                )
        else:
            if not GRIPPER_S_OPEN <= action <= GRIPPER_S_CLOSED:
                raise Mt4ClientError(
                    f"action must be {GRIPPER_S_OPEN}-{GRIPPER_S_CLOSED}"
                )
            arg = str(int(action))
            settle = True

        with self._lock:
            self._ensure_connected_unlocked()
            lines = self._send_and_collect(f"g {arg}", wait=GRIP_WAIT_S)
        for line in lines:
            if line.startswith("err"):
                return {"ok": False, "error": line, "lines": lines}
        if settle:
            with self._lock:
                if self._sleep_interruptible(GRIPPER_SETTLE_S):
                    self._send_and_collect("stop", wait=COMMAND_WAIT_S)
                    return {"ok": False, "error": "interrupted", "lines": lines}
        return {"ok": True, "lines": lines}
