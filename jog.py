#!/usr/bin/env python3
"""Keyboard / Xbox gamepad jog for MT4 custom jog firmware — hold to move, H to home.

Cartesian-only jog (world-frame TCP motion via firmware `cj`, on-device
Jacobian/DLS resolved-rate) plus J4 wrist roll and the gripper. Direct
per-joint jog (J1-J3) has been dropped -- Cartesian jog is the only motion
mode now.
"""

from __future__ import annotations

import argparse
import sys
import time

from mt4_jog.gamepad import A, B, START, X, Y, XboxGamepad
from mt4_jog.joints import (
    DEFAULT_BAUD,
    GRIPPER_S_CLOSED,
    GRIPPER_S_OPEN,
    J1_HOME_CENTER_STEPS,
    J2_HOME_PULLOFF_STEPS,
    LIMIT_JOINTS,
)
from mt4_jog.console import format_firmware_line, print_status
from mt4_jog.ports import Mt4PortError, port_display, resolve_port
from mt4_jog.serial import (
    FirmwareNotReadyError,
    SerialGoneError,
    await_firmware_alive,
    drain_lines,
    open_serial,
    send,
    send_quick,
)
from mt4_jog.status import TcpPose, parse_status_lines

POLL_MS = 10
HOME_WAIT_S = 180.0
CJ_RESEND_S = 0.05
# Gripper commands are resent on a timer while the key is held, as Cartesian
# jog is above: sent once per key-transition, a single dropped serial line
# leaves the gripper stuck until the next transition.
GRIP_RESEND_S = 0.05
SPEED_STEP_US = 100
SPEED_MIN_US = 700
SPEED_MAX_US = 4000
DEFAULT_SPEED_US = 1524
# `speed <us>` is a plain idempotent set command (no start/stop state), so
# holding -/= just re-sends it on a repeat timer -- no protocol change needed.
SPEED_REPEAT_S = 0.08
Y_LONG_PRESS_S = 0.5

# World-frame TCP jog (mm/s direction; firmware normalizes).
CART_BINDINGS: dict[str, tuple[int, int, int]] = {
    "i": (0, 0, 1),
    "k": (0, 0, -1),
    "s": (0, 1, 0),
    "w": (0, -1, 0),
    "a": (1, 0, 0),
    "d": (-1, 0, 0),
}

VK = {
    "i": 0x49,
    "k": 0x4B,
    "s": 0x53,
    "w": 0x57,
    "a": 0x41,
    "d": 0x44,
    "j": 0x4A,
    "l": 0x4C,
    "q": 0x51,
    "e": 0x45,
    "esc": 0x1B,
    "space": 0x20,
    "0": 0x30,
    "h": 0x48,
    "grave": 0xC0,
    "minus": 0xBD,
    "plus": 0xBB,
}


def print_help(*, gamepad: bool) -> None:
    print("MT4 jog — hold keys or sticks to move, release to stop")
    print()
    print("Keyboard:")
    print("  I / K     world +Z / -Z")
    print("  S / W     world +Y / -Y")
    print("  A / D     world +X / -X")
    print("  J / L     J4 wrist roll (also while moving XYZ)")
    print("  -  / =    nudge keyboard jog speed slower / faster (live)")
    print("  H       home J1 + J2 (on-device)")
    print(
        f"  Q / E   gripper sweep open / close "
        f"(S{GRIPPER_S_OPEN}–S{GRIPPER_S_CLOSED} on MT4; release = stop)"
    )
    print("  SPACE   status")
    print("  `       toggle invert world X + Y")
    print("  0       stop, drivers off")
    print("  ESC     quit")
    if gamepad:
        print()
        print("Xbox controller (player 1):")
        print("  Left stick        world X / Y")
        print("  Right stick Y     world Z")
        print("  Right stick X     J4 wrist roll (also while moving XYZ)")
        print(
            "  Stick throw       jog speed (radial throw, max of sticks; "
            f"full = {SPEED_MIN_US} µs; not saved for keyboard)"
        )
        print("  LT / RT           gripper open / close")
        print("  Y short / long    goto / store TCP x,y,z + J4")
        print("  A                 home")
        print("  B                 stop, drivers off")
        print("  X                 status")
        print("  Start             toggle invert world X + Y")
        print("  Back              quit")
    print()
    limits = ", ".join(f"{io}={label}" for io, label in sorted(LIMIT_JOINTS.items()))
    print(f"Limits: {limits}")
    print()


def format_limit(line: str) -> str:
    parts = line.split()
    if len(parts) < 2 or not parts[0].startswith("I"):
        return line
    pin, _, raw = parts[0].partition("=")
    joint = LIMIT_JOINTS.get(pin, "")
    label = f"{pin} {joint}" if joint else pin
    if parts[1] == "TRIG":
        return f"{label} triggered (raw={raw})"
    return f"{label} released (raw={raw})"


def drain_async(ser, buf: list[str], verbose: bool) -> None:
    for line in drain_lines(ser, buf):
        if line.startswith("lim "):
            print(f"Limit: {format_limit(line[4:])}")
        elif line.startswith("home "):
            print(format_firmware_line(line))
        elif line.startswith("pos "):
            print(format_firmware_line(line))
        elif line.startswith("ok speed "):
            # Stick throw streams many speed updates; only echo under -v.
            # Keyboard -/= prints Speed itself when the setting changes.
            if verbose:
                print(format_firmware_line(line), file=sys.stderr)
        elif line.startswith("mp done") or line.startswith("ok mp"):
            print(line)
        elif line.startswith("err "):
            print(line)
        elif verbose:
            print(line, file=sys.stderr)


def key_down(key: str) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    vk = VK.get(key)
    return vk is not None and bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def flush_console_input() -> None:
    """Drop keystrokes typed during a jog session so they don't leak into the
    shell prompt after the script exits.

    key_down polls GetAsyncKeyState, which is global and never consumes the
    console's own input buffer -- every key pressed while jogging still lands
    there and would otherwise replay into the shell once the process exits.
    """
    if sys.platform != "win32":
        return
    import msvcrt

    while msvcrt.kbhit():
        msvcrt.getwch()


def _process_parents() -> dict[int, int]:
    """pid -> parent pid for every running process (Toolhelp32 snapshot)."""
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    TH32CS_SNAPPROCESS = 0x00000002
    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == -1:
        return {}
    parents: dict[int, int] = {}
    try:
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if kernel32.Process32First(snapshot, ctypes.byref(entry)):
            while True:
                parents[entry.th32ProcessID] = entry.th32ParentProcessID
                if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)
    return parents


def console_focused() -> bool:
    """True when the foreground window belongs to this process's own
    terminal/shell -- i.e. an ancestor of this process currently has OS
    input focus.

    GetAsyncKeyState (key_down above) is global -- it fires from any window,
    not just this one. Callers that want a key to act only while this
    terminal is the foreground window (e.g. a background loop's re-home
    hotkey) should gate on this too.

    A direct GetConsoleWindow() == GetForegroundWindow() comparison doesn't
    work here: under Windows Terminal / VS Code / Cursor, the visible,
    focusable window belongs to the terminal-emulator process, not the
    hidden conhost window GetConsoleWindow() returns a handle for -- that
    comparison is always False under those hosts, silently disabling
    whatever depends on it. Instead, walk up the process tree from the
    foreground window's owning PID and this process's own PID; if they
    share an ancestor, the terminal hosting us is the one in focus. This
    can't distinguish separate tabs/panes within the same terminal window
    (not exposed by these APIs), but correctly ignores focus in unrelated
    apps (browser, editor, etc).
    """
    if sys.platform != "win32":
        return True
    import ctypes
    import os
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    fg_hwnd = user32.GetForegroundWindow()
    if not fg_hwnd:
        return True

    fg_pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(fg_hwnd, ctypes.byref(fg_pid))
    target = fg_pid.value
    if not target:
        return True

    return _pid_shares_ancestry(_process_parents(), os.getpid(), target)


def _pid_shares_ancestry(parents: dict[int, int], pid: int, target: int) -> bool:
    """True when ``target`` is ``pid`` itself or one of its ancestors."""
    seen: set[int] = set()
    while pid and pid not in seen:
        if pid == target:
            return True
        seen.add(pid)
        pid = parents.get(pid, 0)
    return False


def gamepad_button_held(gamepad: XboxGamepad | None, mask: int) -> bool:
    """Poll fresh controller state before testing a held button."""
    if gamepad is None or not mask:
        return False
    gamepad.poll()
    return gamepad.is_pressed(mask)


def wait_until_released(
    ser,
    buf: list[str],
    verbose: bool,
    poll_s: float,
    *,
    gamepad: XboxGamepad | None,
    keyboard_keys: tuple[str, ...] = (),
    gamepad_mask: int = 0,
) -> None:
    """Block until keyboard keys and/or a gamepad button are released."""
    while any(key_down(key) for key in keyboard_keys) or gamepad_button_held(
        gamepad, gamepad_mask
    ):
        drain_async(ser, buf, verbose)
        time.sleep(poll_s)


def stop_jog(ser) -> None:
    send_quick(ser, "stop")
    time.sleep(0.02)


def clear_active_motion(ser) -> tuple[None, None]:
    """Stop jog/gripper and return cleared motion state."""
    stop_jog(ser)
    gripper_sweep_stop(ser)
    return None, None


def apply_xy_invert(
    vector: tuple[int, int, int] | None,
    invert_xy: bool,
) -> tuple[int, int, int] | None:
    if vector is None or not invert_xy:
        return vector
    x, y, z = vector
    return -x, -y, z


def pressed_cart_vector(
    gamepad_cart: tuple[int, int, int] | None = None,
    *,
    invert_xy: bool = False,
) -> tuple[int, int, int] | None:
    vec = [0, 0, 0]
    for key, delta in CART_BINDINGS.items():
        if key_down(key):
            for i in range(3):
                vec[i] += delta[i]
    if gamepad_cart is not None:
        for i in range(3):
            vec[i] += gamepad_cart[i]
    vec = [max(-1, min(1, v)) for v in vec]
    if vec == [0, 0, 0]:
        return None
    return apply_xy_invert((vec[0], vec[1], vec[2]), invert_xy)


def sync_cart_jog(
    ser, vector: tuple[int, int, int] | None, j4_roll: int = 0
) -> None:
    """Send the unified jog command: Cartesian direction and/or J4 roll.
    The firmware layers the roll onto the resolved-rate solution (`cj dx dy
    dz [j4]`), so the wrist can rotate while the TCP moves; a zero vector
    with nonzero roll is a pure wrist roll through the same path."""
    if vector is None and j4_roll == 0:
        stop_jog(ser)
        return
    x, y, z = vector if vector is not None else (0, 0, 0)
    send_quick(ser, f"cj {x} {y} {z} {j4_roll}")


def j4_roll_state(gamepad_j4: bool | None = None) -> int:
    """J4 roll direction from J/L keys or right stick X: +1 / -1 in joint
    step sign (J = positive), 0 when idle or both held."""
    j = key_down("j")
    l = key_down("l")
    dir_high: bool | None = None
    if j and not l:
        dir_high = False
    elif l and not j:
        dir_high = True
    elif gamepad_j4 is not None:
        dir_high = gamepad_j4
    if dir_high is None:
        return 0
    # DIR-high on J4 is the negative joint direction (J_DIR_POS_HIGH is
    # false), matching the retired single-axis jog's physical direction.
    return -1 if dir_high else 1


def run_home(ser, buf: list[str], j1: int, j2: int, verbose: bool) -> None:
    cmd = f"home {j1} {j2}"
    if verbose:
        print(f">>> {cmd}", file=sys.stderr)
    ser.write(f"{cmd}\n".encode("ascii"))
    ser.flush()
    deadline = time.monotonic() + HOME_WAIT_S
    while time.monotonic() < deadline:
        for line in drain_lines(ser, buf):
            if line.startswith("home ") or line.startswith("lim "):
                if line.startswith("lim "):
                    print(f"Limit: {format_limit(line[4:])}")
                else:
                    print(format_firmware_line(line))
            if line == "home ok" or line.startswith("home fail"):
                return
        time.sleep(0.02)
    print("Homing timed out", file=sys.stderr)


def gripper_sweep_open(ser) -> None:
    send_quick(ser, "g o")


def gripper_sweep_close(ser) -> None:
    send_quick(ser, "g c")


def gripper_sweep_stop(ser) -> None:
    send_quick(ser, "g stop")


def gripper_key_state(gamepad_grip: str | None = None) -> str | None:
    """Return 'open', 'close', or None when Q/E not held (or both held)."""
    q = key_down("q")
    e = key_down("e")
    if q and not e:
        return "open"
    if e and not q:
        return "close"
    return gamepad_grip


def speed_us_from_stick_factor(factor: float) -> int:
    """Map 0..1 stick factor to step period (1 = fastest / SPEED_MIN_US)."""
    factor = max(0.0, min(1.0, factor))
    return int(round(SPEED_MAX_US + factor * (SPEED_MIN_US - SPEED_MAX_US)))


def query_tcp(ser, buf: list[str], verbose: bool) -> TcpPose | None:
    """Stop is caller's job; read current TCP pose via `?`."""
    lines = send(ser, "?", wait=1.0)
    drain_async(ser, buf, verbose)
    return parse_status_lines(lines).tcp


def start_mp_restore(ser, pose: TcpPose, *, verbose: bool) -> None:
    """Fire an absolute restore move; do not wait — jog/stop can preempt it."""
    cmd = (
        f"mp {pose.x:.3f} {pose.y:.3f} {pose.z:.3f} "
        f"{pose.j4:.3f} 0 {SPEED_MIN_US}"
    )
    print(
        f"Goto: x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f} "
        f"j4={pose.j4:.1f}  speed={SPEED_MIN_US}"
    )
    if verbose:
        print(f">>> {cmd}", file=sys.stderr)
    send_quick(ser, cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="MT4 jog")
    parser.add_argument(
        "--port",
        default=None,
        help="serial port (auto-detect MT4 if omitted)",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--poll-ms", type=int, default=POLL_MS)
    parser.add_argument("--j1-center", type=int, default=J1_HOME_CENTER_STEPS)
    parser.add_argument("--j2-pull", type=int, default=J2_HOME_PULLOFF_STEPS)
    parser.add_argument(
        "--no-orient",
        action="store_true",
        help="disable J4 wrist unwind during Cartesian jog",
    )
    parser.add_argument(
        "--no-gamepad",
        action="store_true",
        help="disable Xbox controller input (keyboard only)",
    )
    parser.add_argument(
        "--gamepad-deadzone",
        type=int,
        default=9000,
        help="stick deadzone for Xbox controller (default: 9000)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("Requires Windows (GetAsyncKeyState / XInput)", file=sys.stderr)
        return 1

    gamepad = None
    if not args.no_gamepad:
        gamepad = XboxGamepad(deadzone=args.gamepad_deadzone)
        if not gamepad.available:
            print("XInput not available — keyboard only", file=sys.stderr)
            gamepad = None

    try:
        port = resolve_port(args.port, baud=args.baud)
    except Mt4PortError as exc:
        print(exc, file=sys.stderr)
        return 1

    print_help(gamepad=gamepad is not None)
    print(port_display(port, baud=args.baud, explicit=args.port is not None))
    if gamepad is not None:
        print("Xbox controller: plug in before start")

    poll_s = args.poll_ms / 1000.0
    buf: list[str] = [""]
    # Active jog command: (cart vector or None, j4 roll) once anything is
    # held, else None.
    active_cart: tuple[tuple[int, int, int] | None, int] | None = None
    grip_state: str | None = None
    last_cj_send = 0.0
    last_grip_send = 0.0
    # Keyboard -/= setting. Stick throw overrides firmware speed only while
    # sticks are active and never writes back into this value.
    keyboard_speed_us = DEFAULT_SPEED_US
    applied_speed_us = keyboard_speed_us
    last_speed_adjust = 0.0
    invert_xy = False
    grave_was_down = False
    stored_pose: TcpPose | None = None
    y_press_start: float | None = None
    y_long_fired = False

    with open_serial(port, args.baud) as ser:
        try:
            await_firmware_alive(ser, port_label=port)
            send(ser, "all f", wait=0.5)
            send(ser, "orient off" if args.no_orient else "orient on", wait=0.3)
            # Discard limit/status chatter from the handshake; SPACE still prints status.
            drain_lines(ser, buf)
            print("Ready")
        except FirmwareNotReadyError as exc:
            print(exc, file=sys.stderr)
            return 1
        except SerialGoneError as exc:
            print(exc, file=sys.stderr)
            return 1

        try:
            while True:
                drain_async(ser, buf, args.verbose)

                pad = gamepad.poll() if gamepad is not None else None
                pad_cart = pad.cart if pad and pad.connected else None
                pad_j4 = pad.j4 if pad and pad.connected else None
                pad_grip = pad.grip if pad and pad.connected else None
                now = time.monotonic()

                if key_down("esc") or (pad is not None and pad.quit):
                    break

                grave_down = key_down("grave")
                toggle_invert = (grave_down and not grave_was_down) or (
                    pad is not None and pad.toggle_invert_xy
                )
                grave_was_down = grave_down
                if toggle_invert:
                    invert_xy = not invert_xy
                    print(f"XY invert: {'on' if invert_xy else 'off'}")
                    if active_cart is not None:
                        stop_jog(ser)
                        active_cart = None
                    wait_until_released(
                        ser,
                        buf,
                        args.verbose,
                        poll_s,
                        gamepad=gamepad,
                        keyboard_keys=("grave",),
                        gamepad_mask=START,
                    )
                    grave_was_down = key_down("grave")
                    continue

                if key_down("space") or (pad is not None and pad.status):
                    active_cart, grip_state = clear_active_motion(ser)
                    print_status(send(ser, "?", wait=1.0))
                    wait_until_released(
                        ser, buf, args.verbose, poll_s, gamepad=gamepad,
                        keyboard_keys=("space",), gamepad_mask=X,
                    )
                    continue

                if key_down("0") or (pad is not None and pad.stop_all):
                    active_cart, grip_state = clear_active_motion(ser)
                    send(ser, "e0", wait=0.2)
                    send(ser, "all f", wait=0.3)
                    wait_until_released(
                        ser, buf, args.verbose, poll_s, gamepad=gamepad,
                        keyboard_keys=("0",), gamepad_mask=B,
                    )
                    continue

                if key_down("h") or (pad is not None and pad.home):
                    active_cart, grip_state = clear_active_motion(ser)
                    run_home(ser, buf, args.j1_center, args.j2_pull, args.verbose)
                    wait_until_released(
                        ser, buf, args.verbose, poll_s, gamepad=gamepad,
                        keyboard_keys=("h",), gamepad_mask=A,
                    )
                    continue

                # Y: long-press stores TCP x/y/z + J4; short-press recalls
                # at max speed (gripper unchanged).
                y_down = pad is not None and pad.connected and bool(pad.buttons & Y)
                if y_down:
                    if y_press_start is None:
                        y_press_start = now
                        y_long_fired = False
                    elif (
                        not y_long_fired
                        and now - y_press_start >= Y_LONG_PRESS_S
                    ):
                        y_long_fired = True
                        active_cart, grip_state = clear_active_motion(ser)
                        tcp = query_tcp(ser, buf, args.verbose)
                        if tcp is None:
                            print("Could not read TCP pose to store")
                        else:
                            stored_pose = tcp
                            print(
                                f"Stored: x={tcp.x:.1f} y={tcp.y:.1f} "
                                f"z={tcp.z:.1f} j4={tcp.j4:.1f}"
                            )
                        wait_until_released(
                            ser, buf, args.verbose, poll_s,
                            gamepad=gamepad, gamepad_mask=Y,
                        )
                        y_press_start = None
                        y_long_fired = False
                        continue
                elif y_press_start is not None:
                    was_long = y_long_fired
                    y_press_start = None
                    y_long_fired = False
                    if not was_long:
                        if stored_pose is None:
                            print("No stored pose (long-press Y to store)")
                        else:
                            active_cart, grip_state = clear_active_motion(ser)
                            start_mp_restore(
                                ser, stored_pose, verbose=args.verbose
                            )
                        continue

                minus = key_down("minus")
                plus = key_down("plus")
                if (minus or plus) and now - last_speed_adjust >= SPEED_REPEAT_S:
                    # Lower period = faster; "=" (plus) speeds up. Keyboard
                    # setting only — the gamepad shoulders do not touch speed.
                    keyboard_speed_us += -SPEED_STEP_US if plus else SPEED_STEP_US
                    keyboard_speed_us = max(
                        SPEED_MIN_US, min(SPEED_MAX_US, keyboard_speed_us)
                    )
                    last_speed_adjust = now
                    print(f"Speed: {keyboard_speed_us}")

                stick_factor = (
                    pad.speed_factor
                    if pad is not None and pad.connected
                    else None
                )
                if stick_factor is not None:
                    # Ephemeral stick speed; leave keyboard_speed_us alone.
                    stick_speed = speed_us_from_stick_factor(stick_factor)
                    if stick_speed != applied_speed_us:
                        send_quick(ser, f"speed {stick_speed}")
                        applied_speed_us = stick_speed
                elif applied_speed_us != keyboard_speed_us:
                    send_quick(ser, f"speed {keyboard_speed_us}")
                    applied_speed_us = keyboard_speed_us

                # Unified jog: Cartesian direction and J4 roll are one `cj`
                # command, so the wrist rotates while the TCP moves.
                desired_cart = pressed_cart_vector(pad_cart, invert_xy=invert_xy)
                desired_roll = j4_roll_state(pad_j4)
                if desired_cart is not None or desired_roll != 0:
                    desired = (desired_cart, desired_roll)
                    if desired != active_cart or now - last_cj_send >= CJ_RESEND_S:
                        sync_cart_jog(ser, desired_cart, desired_roll)
                        active_cart = desired
                        last_cj_send = now
                elif active_cart is not None:
                    stop_jog(ser)
                    active_cart = None

                desired_grip = gripper_key_state(pad_grip)
                if desired_grip is None:
                    if grip_state is not None:
                        gripper_sweep_stop(ser)
                        grip_state = None
                elif desired_grip != grip_state or now - last_grip_send >= GRIP_RESEND_S:
                    if desired_grip == "open":
                        gripper_sweep_open(ser)
                    else:
                        gripper_sweep_close(ser)
                    grip_state = desired_grip
                    last_grip_send = now

                time.sleep(poll_s)
        except SerialGoneError as exc:
            print(exc, file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print()
        finally:
            try:
                stop_jog(ser)
                gripper_sweep_stop(ser)
                send(ser, "e0", wait=0.2)
                send(ser, "all f", wait=0.3)
            except SerialGoneError:
                pass

    print("Bye")
    return 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    finally:
        flush_console_input()
    raise SystemExit(_exit_code)
