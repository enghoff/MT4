#!/usr/bin/env python3
"""Interactive camera->robot calibration: jog the TCP onto each reachable
ArUco marker and record it. Writes vision_calibration.json for mt4_vision.

Flow: home the arm, capture a reference frame of the markers, then jog with
the same controls as jog.py. Touch the TCP to a marker center and
record it -- markers can be visited in any order, and unreachable ones are
simply skipped. 3 recorded markers give an affine fit (accurate inside the
marker triangle, no perspective correction); 4+ give a full homography.

Remaps vs jog.py: digit keys record markers (so drivers-off moves
0 -> X key), gamepad A records a marker (home moves A -> Y), and Start
finishes the session (XY invert stays on the ` key only).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from jog import (
    DEFAULT_SPEED_US,
    POLL_MS,
    SPEED_MAX_US,
    SPEED_MIN_US,
    SPEED_REPEAT_S,
    SPEED_STEP_US,
    VK,
    clear_active_motion,
    drain_async,
    flush_console_input,
    gripper_sweep_close,
    gripper_sweep_open,
    gripper_sweep_stop,
    gripper_key_state,
    j4_roll_state,
    key_down,
    pressed_cart_vector,
    run_home,
    stop_jog,
    sync_cart_jog,
    speed_us_from_stick_factor,
    wait_until_released,
    CJ_RESEND_S,
    GRIP_RESEND_S,
)
from mt4_jog.console import print_status
from mt4_jog.gamepad import A, B, BACK, START, X, Y, XboxGamepad
from mt4_jog.joints import DEFAULT_BAUD, J1_HOME_CENTER_STEPS, J2_HOME_PULLOFF_STEPS
from mt4_jog.ports import Mt4PortError, port_display, resolve_port
from mt4_jog.serial import FirmwareNotReadyError, SerialGoneError, await_firmware_alive, open_serial, send, send_quick
from mt4_jog.status import Mt4Status, parse_status_lines

# Calibration-only keys, merged into jog's VK table.
VK.update(
    {
        "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35,
        "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
        "enter": 0x0D,
        "g": 0x47,
        "x": 0x58,
    }
)
DIGIT_KEYS = tuple(str(d) for d in range(10))

STATUS_WAIT_S = 1.5


def print_help(*, gamepad: bool) -> None:
    print("MT4 vision calibration — jog the TCP onto each reachable marker")
    print()
    print("Jog (as jog.py):")
    print("  I/K  A/D  S/W   world Z / X / Y      J / L   J4 roll")
    print("  Q / E           gripper open / close  - / =  speed")
    print("  `               toggle invert world X + Y")
    print()
    print("Calibration:")
    print("  0-9     record current TCP as marker <digit>")
    print("  G       record grip pose (pick Z + gripper S, while gripping a cube)")
    print("  H       re-home")
    print("  SPACE   status")
    print("  X       stop, drivers off")
    print("  ENTER   finish: fit calibration and save")
    print("  ESC     quit without saving")
    if gamepad:
        print()
        print("Xbox controller:")
        print("  Sticks / triggers           jog, gripper (as jog.py)")
        print(f"  Stick throw                 jog speed (full = {SPEED_MIN_US} µs)")
        print("  A       record marker (auto-identified: the one the arm hides)")
        print("  Y       home")
        print("  B       stop, drivers off")
        print("  X       status")
        print("  Start   finish: fit calibration and save")
        print("  Back    quit without saving")
    print()


def prompt_float(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} [{default:g}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Enter a number")


def query_status(ser, buf: list[str], verbose: bool) -> Mt4Status:
    lines = send(ser, "?", wait=STATUS_WAIT_S)
    drain_async(ser, buf, verbose)
    return parse_status_lines(lines)


def record_marker(
    ser,
    buf: list[str],
    verbose: bool,
    marker_id: int,
    recorded: dict[int, tuple[float, float, float]],
) -> None:
    status = query_status(ser, buf, verbose)
    if status.tcp is None:
        print("Record failed: no TCP pose in status reply, try again")
        return
    if not status.homed:
        print("Record failed: arm reports not homed -- press H to home first")
        return
    tcp = status.tcp
    verb = "updated" if marker_id in recorded else "recorded"
    recorded[marker_id] = (tcp.x, tcp.y, tcp.z)
    print(
        f"marker {marker_id} {verb}: robot ({tcp.x:.1f}, {tcp.y:.1f}, z {tcp.z:.1f})"
        f" -- {len(recorded)} marker(s) so far"
    )


def autodetect_touched_marker(cap, dict_name: str, ref_ids: set[int]) -> int | None:
    """The marker the TCP is touching is the one the arm now hides: diff the
    currently visible ids against the reference frame's."""
    from mt4_vision.camera import grab_frame
    from mt4_vision.detect import detect_markers

    frame = grab_frame(cap)
    seen = {m.marker_id for m in detect_markers(frame, dict_name)}
    missing = sorted(ref_ids - seen)
    if len(missing) == 1:
        return missing[0]
    if not missing:
        print("Record failed: all markers still visible -- is the TCP on one? "
              "Use a digit key to record explicitly.")
    else:
        print(f"Record ambiguous: markers {missing} all hidden -- "
              "use a digit key to record explicitly.")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="MT4 camera->robot calibration")
    parser.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index (MT4_CAMERA_INDEX, else 0, if omitted; "
                             "-1 auto-detects via markers)")
    parser.add_argument("--dict", default="4x4_50", help="ArUco dictionary name")
    parser.add_argument("--output", default=None, help="calibration JSON path")
    parser.add_argument("--poll-ms", type=int, default=POLL_MS)
    parser.add_argument("--j1-center", type=int, default=J1_HOME_CENTER_STEPS)
    parser.add_argument("--j2-pull", type=int, default=J2_HOME_PULLOFF_STEPS)
    parser.add_argument("--no-gamepad", action="store_true")
    parser.add_argument("--gamepad-deadzone", type=int, default=9000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("Requires Windows (GetAsyncKeyState / XInput)", file=sys.stderr)
        return 1

    # Import late so a missing cv2 fails with a clear message, not at startup
    # of unrelated arg errors.
    from mt4_vision.calib import (
        DEFAULT_CALIB_PATH,
        Calibration,
        load_calibration,
    )
    from mt4_vision.camera import DEFAULT_CAMERA_INDEX, CameraError, grab_frame, open_camera
    from mt4_vision.detect import detect_markers

    output = Path(args.output) if args.output else DEFAULT_CALIB_PATH
    camera_index = DEFAULT_CAMERA_INDEX if args.camera is None else args.camera

    # Carry setup-specific tuning across re-calibrations.
    prev = None
    try:
        prev = load_calibration(output)
        print(f"Existing calibration at {output} -- grip/color settings carried over")
    except Exception:  # noqa: BLE001 -- absent or unreadable both mean "fresh"
        pass

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

    try:
        cap = open_camera(camera_index)
    except CameraError as exc:
        print(exc, file=sys.stderr)
        return 1

    print_help(gamepad=gamepad is not None)
    print(port_display(port, baud=args.baud, explicit=args.port is not None))
    print()

    poll_s = args.poll_ms / 1000.0
    buf: list[str] = [""]
    recorded: dict[int, tuple[float, float, float]] = {}
    grip_pose: tuple[float, float] | None = None  # (grip_z, grip_close_s)
    saved = False

    with open_serial(port, args.baud) as ser:
        try:
            await_firmware_alive(ser, port_label=port)
            send(ser, "all f", wait=0.5)
            send(ser, "orient on", wait=0.3)

            status = query_status(ser, buf, args.verbose)
            if status.homed:
                print("Already homed (press H anytime to re-home and clear the view)")
            else:
                print("Homing first (clear the workspace)...")
                run_home(ser, buf, args.j1_center, args.j2_pull, args.verbose)
        except (FirmwareNotReadyError, SerialGoneError) as exc:
            print(exc, file=sys.stderr)
            return 1

        # Reference frame: prefer a clear view of every marker. If we skipped
        # home, the arm may still be over the pad -- H rehomes if needed.
        frame = grab_frame(cap)
        ref_markers = detect_markers(frame, args.dict)
        if len(ref_markers) < 3:
            print(
                f"only {len(ref_markers)} markers visible with dict {args.dict}; "
                "need >=3 (check camera view / --dict)",
                file=sys.stderr,
            )
            return 1
        ref_ids = {m.marker_id for m in ref_markers}
        ref_px = {m.marker_id: (m.px, m.py) for m in ref_markers}
        print(f"Markers in view: {sorted(ref_ids)} -- record at least 3 reachable ones")
        print()

        active_cart: tuple[tuple[int, int, int] | None, int] | None = None
        grip_state: str | None = None
        last_cj_send = last_grip_send = 0.0
        keyboard_speed_us = DEFAULT_SPEED_US
        applied_speed_us = keyboard_speed_us
        last_speed_adjust = 0.0
        invert_xy = False
        grave_was_down = False
        digit_was_down: dict[str, bool] = {d: False for d in DIGIT_KEYS}
        g_was_down = False
        finish = False

        try:
            while True:
                drain_async(ser, buf, args.verbose)

                pad = gamepad.poll() if gamepad is not None else None
                pad_cart = pad.cart if pad and pad.connected else None
                pad_j4 = pad.j4 if pad and pad.connected else None
                pad_grip = pad.grip if pad and pad.connected else None
                pad_edges = pad.edges if pad and pad.connected else 0
                now = time.monotonic()

                if key_down("esc") or pad_edges & BACK:
                    # If we already recorded everything visible/expected,
                    # explicitly confirm quitting without saving.
                    if recorded and ref_ids and set(recorded) >= ref_ids:
                        flush_console_input()
                        total = len(ref_ids)
                        got = len(recorded)
                        raw = input(
                            f"\nAll markers recorded ({got}/{total}). "
                            "Quit without saving? [y/N]: "
                        ).strip().lower()
                        if raw not in ("y", "yes"):
                            continue
                    break

                if key_down("enter") or pad_edges & START:
                    active_cart, grip_state = clear_active_motion(ser)
                    wait_until_released(
                        ser, buf, args.verbose, poll_s, gamepad=gamepad,
                        keyboard_keys=("enter",), gamepad_mask=START,
                    )
                    finish = True
                    break

                # Record marker: explicit id via digit key ...
                for digit in DIGIT_KEYS:
                    down = key_down(digit)
                    if down and not digit_was_down[digit]:
                        marker_id = int(digit)
                        active_cart, grip_state = clear_active_motion(ser)
                        if marker_id not in ref_ids:
                            print(f"Marker {marker_id} is not in the camera's view "
                                  f"(visible: {sorted(ref_ids)})")
                        else:
                            record_marker(ser, buf, args.verbose, marker_id, recorded)
                    digit_was_down[digit] = down

                # ... or auto-identified via occlusion on gamepad A.
                if pad_edges & A:
                    active_cart, grip_state = clear_active_motion(ser)
                    marker_id = autodetect_touched_marker(cap, args.dict, ref_ids)
                    if marker_id is not None:
                        record_marker(ser, buf, args.verbose, marker_id, recorded)

                g_down = key_down("g")
                if g_down and not g_was_down:
                    active_cart, grip_state = clear_active_motion(ser)
                    status = query_status(ser, buf, args.verbose)
                    if status.tcp is None:
                        print("Grip record failed: no TCP pose in status reply")
                    else:
                        grip_pose = (status.tcp.z, status.tcp.grip)
                        print(f"Grip pose recorded: grip_z {grip_pose[0]:.1f}, "
                              f"gripper S {grip_pose[1]:.0f}")
                g_was_down = g_down

                grave_down = key_down("grave")
                if grave_down and not grave_was_down:
                    invert_xy = not invert_xy
                    print(f"XY invert: {'on' if invert_xy else 'off'}")
                    if active_cart is not None:
                        stop_jog(ser)
                        active_cart = None
                grave_was_down = grave_down

                if key_down("space") or pad_edges & X:
                    active_cart, grip_state = clear_active_motion(ser)
                    print_status(send(ser, "?", wait=1.0))
                    wait_until_released(
                        ser, buf, args.verbose, poll_s, gamepad=gamepad,
                        keyboard_keys=("space",), gamepad_mask=X,
                    )
                    continue

                if key_down("x") or pad_edges & B:
                    active_cart, grip_state = clear_active_motion(ser)
                    send(ser, "e0", wait=0.2)
                    send(ser, "all f", wait=0.3)
                    wait_until_released(
                        ser, buf, args.verbose, poll_s, gamepad=gamepad,
                        keyboard_keys=("x",), gamepad_mask=B,
                    )
                    continue

                if key_down("h") or pad_edges & Y:
                    active_cart, grip_state = clear_active_motion(ser)
                    run_home(ser, buf, args.j1_center, args.j2_pull, args.verbose)
                    wait_until_released(
                        ser, buf, args.verbose, poll_s, gamepad=gamepad,
                        keyboard_keys=("h",), gamepad_mask=Y,
                    )
                    continue

                minus = key_down("minus")
                plus = key_down("plus")
                if (minus or plus) and now - last_speed_adjust >= SPEED_REPEAT_S:
                    # Keyboard -/+ sets the fallback speed. During active stick
                    # movement, firmware speed is driven by `pad.speed_factor`
                    # (ephemeral) instead.
                    keyboard_speed_us += -SPEED_STEP_US if plus else SPEED_STEP_US
                    keyboard_speed_us = max(
                        SPEED_MIN_US, min(SPEED_MAX_US, keyboard_speed_us)
                    )
                    last_speed_adjust = now
                    print(f"Speed: {keyboard_speed_us}")

                # Thumbpad/stick throw speed: while sticks are active, drive
                # firmware speed from `pad.speed_factor`. When sticks return to
                # center, restore the keyboard -/+ fallback speed.
                stick_factor = (
                    pad.speed_factor
                    if pad is not None and pad.connected
                    else None
                )
                if stick_factor is not None:
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
            cap.release()

    if finish and len(recorded) >= 3:
        flush_console_input()
        from mt4_vision.table_fit import fit_table_map

        marker_corners = {
            m.marker_id: m.corners for m in ref_markers if m.corners is not None
        }
        touch_px = {mid: ref_px[mid] for mid in sorted(recorded)}
        touch_robot = {mid: recorded[mid][:2] for mid in sorted(recorded)}
        matrix, report = fit_table_map(marker_corners, touch_px, touch_robot)
        print(f"\n{report.kind} fit from markers {sorted(recorded)}")
        if report.corner_rms_px is not None:
            print(f"Corner-bundle RMS: {report.corner_rms_px}px "
                  f"(~{report.corner_rms_mm}mm; >1px suggests lens distortion)")
        print(f"Per-marker touch residual (mm): {report.touch_residuals_mm}")
        if report.touch_loo_mm:
            print(f"Per-marker leave-one-out error (mm): {report.touch_loo_mm}")
        for note in report.notes:
            print(f"NOTE: {note}")
        # A touch that disagrees with the camera's own geometry by this much
        # is almost never a sloppy touch -- it's the wrong digit pressed at
        # that marker (marker ids aren't human-readable; this exact mistake,
        # ids 2 and 3 swapped, once cost a full recalibration).
        suspects = [m for m, e in report.touch_residuals_mm.items() if e > 25.0]
        if suspects:
            print(f"WARNING: markers {suspects} disagree with the camera geometry by >25mm")
            print("  Most likely the wrong digit was pressed at those markers --")
            print("  Re-jog to each and re-record before trusting this calibration")

        # TCP Z while touching a marker IS the table height at that spot.
        table_z_default = round(statistics.median(z for _x, _y, z in recorded.values()), 1)
        cube_default = prev.cube_height_mm if prev else 20.0
        print("\nHeights (robot-frame Z, mm) and gripper -- Enter accepts defaults")
        # table_z and the cube edge length are prompted first: safe_z's default
        # (table_z + 1.5*cube) and the parallax model both depend on them.
        table_z = prompt_float("table_z", table_z_default)
        cube = prompt_float("cube edge length", cube_default)
        safe_z = prompt_float("safe_z", prev.safe_z if prev else round(table_z + 1.5 * cube, 1))
        cam_height_default = (
            prev.cam_height_mm if prev and prev.cam_height_mm is not None else 240.0
        )
        # Camera lens height above the table -- a starting value for the overlay
        # parallax model (calib.robot_to_pixel). For an accurate overlay run
        # calibrate_camera_nadir.py afterwards: it derives BOTH this height and
        # the camera nadir (cam_xy_robot) from vision + the arm (grip a cube,
        # sweep heights, fit), no tape measure needed, and overwrites this. This
        # prompt only seeds a plausible value / bootstraps setups without the
        # nadir fit yet.
        cam_height = prompt_float("camera height above table", cam_height_default)
        grip_close_default = int(grip_pose[1]) if grip_pose else (prev.grip_close_s if prev else 240)
        grip_close = int(prompt_float("grip_close_s", grip_close_default))
        grip_open = int(prompt_float("grip_open_s", prev.grip_open_s if prev else 140))

        import cv2
        import numpy as np

        hull = cv2.convexHull(
            np.array([[m.px, m.py] for m in ref_markers], dtype=np.float32)
        ).reshape(-1, 2)
        calib = Calibration(
            homography=matrix,
            table_z=table_z,
            safe_z=safe_z,
            grip_open_s=grip_open,
            grip_close_s=grip_close,
            cube_height_mm=cube,
            bundle_homography=report.bundle_h,
            cam_xy_robot=prev.cam_xy_robot if prev else None,
            cam_height_mm=cam_height,
            color_ranges=prev.color_ranges if prev else {},
            workspace_hull_px=hull.tolist(),
            raw_marker_observations={
                str(mid): {
                    "pixel": list(touch_px[mid]),
                    "corners": marker_corners.get(mid),
                    "robot": list(touch_robot[mid]),
                }
                for mid in sorted(recorded)
            },
        )
        calib.save(output)
        saved = True
        print(f"\nSaved to {output}")
    elif finish:
        print(f"\nOnly {len(recorded)} marker(s) recorded; need >=3 -- nothing saved")
    elif recorded:
        print(f"\nQuit without saving ({len(recorded)} marker(s) had been recorded)")

    print("Bye")
    return 0 if saved or not finish else 1


if __name__ == "__main__":
    try:
        _exit_code = main()
    finally:
        flush_console_input()
    raise SystemExit(_exit_code)
