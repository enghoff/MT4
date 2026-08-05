#!/usr/bin/env python3
"""Manual J4 origin calibration via firmware ``j4zero``.

J4 has no home switch: its step counter starts at 0 wherever the wrist sat
at boot. Face-aligned picks
need a known relationship between "jaws along the arm" and world-frame J4.

This script parks the TCP on the arm axis (y=0), lets the operator jog until
the jaws look aligned with the arm, then sends firmware ``j4zero`` — which
rewrites J4's step counter so that pose reports **world J4 = 0** (no motion).
Pick/place then treat world J4 = 0 as jaws-along-arm.

The zero survives ``home`` (firmware preserves J4 across homing). It is lost
on power cycle / reflash until this script is run again.

Controls (same as jog.py for J4):
  - Left thumbstick horizontal: world X nudge
  - J / L (or right stick X): J4 wrist roll
  - H / A: home (on-device); arm ends at home pose, not the park pose
  - ENTER / Start: run j4zero and save
  - ESC: abort
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from jog import (
    CJ_RESEND_S,
    DEFAULT_SPEED_US,
    VK,
    drain_async,
    flush_console_input,
    j4_roll_state,
    key_down,
    run_home,
    speed_us_from_stick_factor,
    stop_jog,
    sync_cart_jog,
    wait_until_released,
)
from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.gamepad import A, START, XboxGamepad
from mt4_jog.joints import DEFAULT_BAUD, J1_HOME_CENTER_STEPS, J2_HOME_PULLOFF_STEPS
from mt4_jog.ports import resolve_port
from mt4_jog.serial import (
    await_firmware_alive,
    close_quiet,
    open_serial,
    send,
    send_quick,
)
from mt4_jog.status import parse_status_lines
from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration
from mt4_vision.pickplace import _travel, ensure_homed, retreat_for_camera

DEFAULT_POSE = (230.0, 0.0, 215.0)  # y must be 0 so axis is ~robot x-axis

VK.update({"enter": 0x0D})


def _query_tcp_j4(ser) -> float | None:
    lines = send(ser, "?", wait=1.0)
    status = parse_status_lines(lines)
    if status.tcp is None:
        return None
    return float(status.tcp.j4)


def _world_x_from_inputs(pad_cart: tuple[int, int, int] | None) -> int:
    if pad_cart is None:
        return 0
    return int(pad_cart[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual J4 origin (firmware j4zero)")
    parser.add_argument("--port", default="", help="serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument(
        "--pose",
        type=float,
        nargs=3,
        default=DEFAULT_POSE,
        metavar=("X", "Y", "Z"),
        help="TCP park pose; y must be 0 so the arm axis is the x-axis",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="report current world J4, but don't j4zero / save")
    parser.add_argument("--no-gamepad", action="store_true")
    parser.add_argument("--gamepad-deadzone", type=int, default=9000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if abs(args.pose[1]) > 1e-6:
        print(
            "error: pose y must be 0 (park needs j1≈0 so the arm axis is robot x)",
            file=sys.stderr,
        )
        return 1

    port = resolve_port(args.port or None, baud=args.baud, probe=False)
    calib = load_calibration(Path(args.calib))

    client = Mt4Client(port=port, baud=args.baud)
    try:
        client.ensure_connected()
        ensure_homed(client)
        _travel(client, calib, *args.pose, "park for J4 alignment", j4=0.0)
        time.sleep(1.0)
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()

    gamepad = None
    if not args.no_gamepad:
        gp = XboxGamepad(deadzone=args.gamepad_deadzone)
        if gp.available:
            gamepad = gp
        else:
            print("XInput not available — keyboard-only", file=sys.stderr)

    print(
        "\nAlign jaws with the arm axis, then confirm:\n"
        "  Left thumbstick horizontal  world X nudge\n"
        "  J / L (or right stick X)    J4 wrist roll\n"
        "  H / A                       home (on-device, leaves park pose)\n"
        "  ENTER / Start               firmware j4zero (world J4 → 0)\n"
        "  ESC                         abort\n"
    )

    poll_s = 0.01
    applied_speed_us = DEFAULT_SPEED_US
    buf: list[str] = [""]
    confirmed = False
    ser = open_serial(port, args.baud)
    try:
        await_firmware_alive(ser, port_label=port)
        send(ser, "all f", wait=0.5)
        send(ser, "orient on", wait=0.3)
        drain_async(ser, buf, args.verbose)

        active_cart: tuple[int, int, int] | None = None
        active_roll = 0
        last_cj_send = 0.0

        while True:
            drain_async(ser, buf, args.verbose)
            pad = gamepad.poll() if gamepad is not None else None
            now = time.monotonic()

            if key_down("esc") or (pad is not None and (pad.quit or pad.stop_all)):
                print("Aborted (no changes).")
                return 1

            confirm = key_down("enter") or (pad is not None and (pad.edges & START))
            if confirm:
                stop_jog(ser)
                time.sleep(0.05)
                before = _query_tcp_j4(ser)
                if before is None:
                    print("error: could not read world J4", file=sys.stderr)
                    return 1
                print(f"Aligned at world J4 = {before:+.1f} deg")
                confirmed = True
                break

            if key_down("h") or (pad is not None and pad.home):
                stop_jog(ser)
                active_cart = None
                active_roll = 0
                run_home(
                    ser, buf, J1_HOME_CENTER_STEPS, J2_HOME_PULLOFF_STEPS,
                    args.verbose,
                )
                print("Homed — re-park to the alignment pose before aligning J4.")
                wait_until_released(
                    ser, buf, args.verbose, poll_s, gamepad=gamepad,
                    keyboard_keys=("h",), gamepad_mask=A,
                )
                continue

            pad_cart = pad.cart if pad is not None and pad.connected else None
            pad_j4 = pad.j4 if pad is not None and pad.connected else None

            if pad is not None and pad.connected and pad.speed_factor is not None:
                stick_speed = speed_us_from_stick_factor(pad.speed_factor)
                if stick_speed != applied_speed_us:
                    send_quick(ser, f"speed {stick_speed}")
                    applied_speed_us = stick_speed

            dx = _world_x_from_inputs(pad_cart)
            desired_cart = (dx, 0, 0) if dx != 0 else None
            desired_roll = j4_roll_state(pad_j4)

            if desired_cart is not None or desired_roll != 0:
                if (
                    desired_cart != active_cart
                    or desired_roll != active_roll
                    or now - last_cj_send >= CJ_RESEND_S
                ):
                    sync_cart_jog(ser, desired_cart, desired_roll)
                    active_cart = desired_cart
                    active_roll = desired_roll
                    last_cj_send = now
            elif active_cart is not None or active_roll != 0:
                stop_jog(ser)
                active_cart = None
                active_roll = 0

            time.sleep(poll_s)
    finally:
        try:
            stop_jog(ser)
        except Exception:
            pass
        close_quiet(ser)

    if not confirmed:
        return 1

    client = Mt4Client(port=port, baud=args.baud)
    try:
        client.ensure_connected()
        if args.dry_run:
            tcp = client.get_tcp()
            print(f"--dry-run: world J4 still {tcp.j4:+.1f}; not zeroed / not saved")
            return 0

        result = client.j4_zero()
        if not result.get("ok"):
            print(f"j4zero failed: {result.get('error', result)}", file=sys.stderr)
            return 1
        tcp = result.get("tcp") or client.get_tcp().as_dict()
        print(f"j4zero ok — world J4 now {float(tcp['j4']):+.2f} deg "
              f"(expect ~0)")

        retreat_for_camera(client, calib)
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    finally:
        flush_console_input()
    raise SystemExit(_exit_code)
