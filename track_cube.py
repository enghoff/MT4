#!/usr/bin/env python3
"""Visually servo the gripper to hover over a single moving cube.

Holds the TCP a fixed height above the table (default 5cm) directly over
one cube and follows it as it's moved by hand. Only one cube is tracked at
a time -- if several are in view, whichever reachable one is detected first
is locked on and identity is then held via nearest-neighbor gating (not
color alone, since two cubes can share a color), never opportunistically
re-targeted.

Two phases:

  1. Setup (`Mt4Client`, blocking `mp`): home if needed, find an initial
     reachable cube, and move the TCP to hover directly above it. This
     reuses the well-tested absolute-move path for the one large motion in
     the whole run.
  2. Tracking (raw serial, streaming `mp`): `Mt4Client.move_to()` blocks
     until the whole move completes, so this script instead re-sends a
     bounded absolute move (`mp x y z j4 g speed_us`) every tick, retargeting
     it to the cube's latest position before the previous one finishes. This
     used to be done with the firmware's Cartesian jog (`cj dx dy dz`)
     instead -- direction-only, no notion of a destination, so *something*
     external always has to decide when to stop it, and that something was
     this host loop's ~100ms-stale glance at position: overshoot wasn't a
     stepper coasting past the target (steppers don't coast), it was the
     arm faithfully stepping in the last commanded direction for the entire
     gap between one position sample and the next. `mp` moves the stop
     decision into firmware, which tracks its own remaining step count and
     halts exactly there, no host reaction time involved. The one thing that
     took firmware work to support was retargeting *mid-flight* without an
     abrupt stop -- see start_absolute_move()'s in-flight-retarget path in
     motion.cpp -- so a moving cube gets a continuously updated destination
     instead of finish-then-redirect. This script closes `Mt4Client` and
     opens its own serial connection for the loop, since only one process
     may own the COM port at a time.

Each tick (paced by the camera, ~10Hz measured on this setup):
  - grab the freshest frame and detect cubes
  - match the tracked cube by nearest position (same color, within a gate);
    the target is just that raw matched position, no velocity estimate or
    smoothing filter over it (deliberately dropped -- see git history if
    reintroducing lead-time prediction is ever worth revisiting)
  - request a fresh real TCP pose every tick (`pos`, fire-and-forget) and
    pick up the reply whenever it lands -- kept off the critical path so it
    never blocks the loop, but still only a tick or so stale
  - re-send `mp` toward the target with a distance-scheduled speed_us (see
    speed_for_distance), but only once the previous `mp` has actually been
    acknowledged (see awaiting_mp_ack) -- this is bang-bang + a gain
    schedule, not PID, there's no integral or derivative term -- the
    firmware absorbs a resend's retargeting by ramping the applied step
    rate toward whatever speed_us was just requested instead of snapping to
    it, so a changing distance-to-target (and therefore a changing
    requested speed) doesn't itself produce a jerk
  - once inside the deadband, stop re-sending `mp` and let whatever move is
    already in flight arrive and hold on its own -- no explicit `stop`,
    since `mp` decelerates into its own destination and forcing a stop here
    would trade that smooth arrival for an abrupt one

If the cube is lost from view for longer than a short grace period, or Ctrl+C
is hit, or a re-home is requested, the arm sends an explicit `stop` and holds
position, watching for the same cube to reappear anywhere in view -- it never
switches to a different cube.

Ctrl+C (or any error) always sends `stop` before exiting -- there is no
firmware watchdog that stops `mp` on its own if the host goes quiet.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from jog import console_focused, key_down, run_home
from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.joints import (
    DEFAULT_BAUD,
    J1_HOME_CENTER_STEPS,
    J2_HOME_PULLOFF_STEPS,
    JOG_SPEED_MIN_US,
)
from mt4_jog.serial import (
    FirmwareNotReadyError,
    SerialGoneError,
    await_firmware_alive,
    drain_lines,
    open_serial,
    send_quick,
)
from mt4_jog.status import TcpPose, parse_tcp_line
from mt4_vision.calib import (
    DEFAULT_CALIB_PATH,
    Calibration,
    CalibrationError,
    load_calibration,
)
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, CameraError, FrameStream
from mt4_vision.detect import CubeDetection, detect_cubes
from mt4_vision.pickplace import ensure_homed
from mt4_vision.preview import VideoRecorder, draw_cube_marker, draw_lock_ring, draw_outlined_text
from mt4_vision.workspace import KEEPOUT_RADIUS_MM, MAX_REACH_MM

DEFAULT_HOVER_MM = 50.0
# Extra margin beyond the firmware's own keep-out check -- `mp` rejects a
# target inside the keep-out cylinder outright ("err mp keepout"), but has
# no analogous explicit max-reach clamp (an out-of-reach target just fails
# IK feasibility, "err mp unreachable"), so both edges of the workspace are
# clamped defensively here before ever sending a target.
KEEPOUT_MARGIN_MM = 5.0
REACH_MARGIN_MM = 5.0

MAX_JUMP_MM = 40.0  # per-tick gate while actively tracking
# No position gate once actually lost (frozen, past the grace period): the
# cube can reappear anywhere in view (picked up and set down elsewhere) and
# still count as the same tracked cube -- color is the only identity signal
# available anyway, and match_detection() still prefers the nearest same-
# color candidate to last_known when more than one is in view.
REACQUIRE_GATE_MM = math.inf
# Brief coast on the last known position (no velocity estimate anymore, so
# this just holds last_known frozen) before stop + hold.
LOST_GRACE_S = 0.5
DEADBAND_MM = 3.0  # inside this, stop rather than jitter
# Real TCP feedback is requested every tick but never blocked on: a `pos`
# query is fired fire-and-forget (send_quick) and its reply is picked up
# opportunistically whenever it shows up in the buffer already being
# drained for firmware chatter (report_firmware_lines). This gives TCP
# feedback that's stale by about one tick (~40-60ms) instead of the several
# hundred ms a synchronous send(wait=...) would cost every tick.
#
# A synchronous per-tick poll was tried first and did fix a real bug (a
# multi-tick blind window where the arm kept jogging on a stale position
# estimate, overshot the target during it, and reversed hard once the next
# poll revealed the overshoot -- a feedback-deadtime limit cycle, confirmed
# live as TCP y oscillating ~40-70mm around a perfectly stationary target).
# But blocking ~150-200ms per tick throttled the whole loop from
# ~15-30Hz (camera-bound) to ~5-6Hz, and a bang-bang controller that looks
# smooth at high update rate looks like discrete steps at a low one --
# nothing about the motion logic changed, just how coarse each visible
# correction segment became. The async version keeps the fast loop rate
# and keeps the blind window short enough not to matter.
# Requested speed ramps with distance-to-target: fast (the firmware's
# quickest step rate) while far away, slower as it converges. This IS
# load-bearing, confirmed by removing it: even with fresh per-tick TCP
# feedback, a constant max-speed approach overshoots the 3mm deadband within
# a single ~100ms tick (real linear speed at max jog covers more than that
# easily), and the correction the next tick overshoots the other way -- a
# sustained oscillation with no feedback-staleness involved at all, just
# velocity too high relative to deadband width and tick period.
#
# Each resend's requested speed_us comes from this schedule -- motion.cpp's
# in-flight-retarget path (dda_continue_ramp) is what smooths the applied
# step rate toward each newly requested speed_us on the firmware's own
# clock, so a changing schedule value never itself produces a jerk, the
# same way `cjramp` did for `cjspeed` before this script switched from `cj`
# to `mp`. That smoothing is cheap -- but unlike `cj` (a direction+speed
# re-arm), an `mp` call itself is NOT: it solves IK and plans a
# keep-out-clear route (in-flight retargets now skip the route's IK
# feasibility sweep, reusing the in-flight path's detour radius when the
# chord doesn't clear -- see plan_mp_xy_route -- but target IK, route
# geometry, and parsing remain), real work on an 8-bit AVR with no FPU. An earlier version of this constant gated resends
# on target drift (skip unless the target moved >= 5mm since the last `mp`
# actually sent) to protect against that cost -- confirmed necessary at the
# time: resending every ~100ms tick regardless of whether the target moved
# was blocking firmware's loop() long enough (55-420ms, scaling with
# distance -- motion.cpp's mp_estimate_path_ticks() was solving IK once per
# ~2mm segment of the whole path just to time the accel/decel ramp) that
# the AVR's UART RX buffer filled and silently dropped bytes, corrupting
# `mp` lines (`err mp <x> <y> <z> <j4> <g> [speed_us]`) or splicing two
# lines together (`err unknown`). But gating on drift instead of just
# eating the cost had its own failure mode, also confirmed live: it let the
# arm coast to the end of its current (now slightly stale) target and sit
# there -- since `mp` is a bounded move that finishes and holds on its own
# once nothing new is sent -- until enough drift finally accumulated to
# unblock the next send, which then had no in-flight move left to splice
# onto and had to cold-start, seen as the arm stopping dead and jerking
# forward again while being tracked continuously in one direction. Fixing
# mp_estimate_path_ticks() to use a cheap straight joint-space chord
# instead of a per-segment IK sum (see motion.cpp) cut the block to a flat
# ~65-110ms independent of distance -- but that's still comparable to this
# loop's own ~100ms tick, and send_quick never waits for a reply, so
# resending unconditionally again just traded byte-loss for a different
# problem: Python outpacing what firmware could actually drain, queuing up
# an ever-growing backlog of already-stale targets that landed in an
# uneven rhythm (confirmed live as tcp alternating small/big steps between
# polls -- the arm executing whatever the queue caught up to, not the
# current camera-fresh target). See awaiting_mp_ack in run_tracking_loop:
# gating resends on the firmware's own ack self-paces to its real
# throughput without an arbitrary constant, and unlike the drift gate it
# never withholds a send once firmware is actually ready.
SPEED_FAR_US = JOG_SPEED_MIN_US  # fastest, used beyond SPEED_FAR_MM
SPEED_NEAR_US = 2400  # gentle final approach, used within SPEED_NEAR_MM
SPEED_FAR_MM = 40.0
SPEED_NEAR_MM = 8.0  # just outside DEADBAND_MM, so it's already slow by the time it stops
INITIAL_LOCK_TIMEOUT_S = 30.0

PREVIEW_WINDOW = "track_cube preview (q or Esc to stop)"
COLOR_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 220, 220),
    "green": (0, 170, 0),
    "blue": (220, 100, 0),
}
UNKNOWN_COLOR_BGR = (200, 200, 200)
LOCK_BGR = (0, 255, 0)
LOST_BGR = (0, 0, 255)


def clamp_to_envelope(x: float, y: float) -> tuple[float, float]:
    """Pull (x, y) back inside the keep-out/reach annulus if it's outside."""
    r = math.hypot(x, y)
    r_min = KEEPOUT_RADIUS_MM + KEEPOUT_MARGIN_MM
    r_max = MAX_REACH_MM - REACH_MARGIN_MM
    if r < r_min and r > 1e-6:
        scale = r_min / r
        return x * scale, y * scale
    if r > r_max:
        scale = r_max / r
        return x * scale, y * scale
    return x, y


def is_reachable(x: float, y: float) -> bool:
    r = math.hypot(x, y)
    return (KEEPOUT_RADIUS_MM + KEEPOUT_MARGIN_MM) <= r <= (MAX_REACH_MM - REACH_MARGIN_MM)


def match_detection(
    detections: list[CubeDetection],
    color: str,
    last_xy: tuple[float, float],
    gate_mm: float,
) -> CubeDetection | None:
    best: CubeDetection | None = None
    best_d = gate_mm
    for det in detections:
        if det.color != color or det.x is None or det.y is None:
            continue
        d = math.hypot(det.x - last_xy[0], det.y - last_xy[1])
        if d <= best_d:
            best = det
            best_d = d
    return best


def find_initial_target(
    stream: FrameStream, calib: Calibration, timeout_s: float
) -> CubeDetection | None:
    """Lock onto the first reachable cube seen; None if none shows up in time."""
    deadline = time.monotonic() + timeout_s
    last_print = 0.0
    while time.monotonic() < deadline:
        frame = stream.fresh()
        detections = detect_cubes(frame, calib)
        candidates = [
            d for d in detections if d.x is not None and d.y is not None
            and is_reachable(d.x, d.y)
        ]
        if candidates:
            return candidates[0]
        now = time.monotonic()
        if now - last_print > 2.0:
            print(
                f"Waiting for a reachable cube in view... "
                f"({len(detections)} detected, none reachable)"
            )
            last_print = now
    return None


def speed_for_distance(dist_mm: float) -> int:
    if dist_mm >= SPEED_FAR_MM:
        return SPEED_FAR_US
    if dist_mm <= SPEED_NEAR_MM:
        return SPEED_NEAR_US
    t = (dist_mm - SPEED_NEAR_MM) / (SPEED_FAR_MM - SPEED_NEAR_MM)
    return int(round(SPEED_NEAR_US + t * (SPEED_FAR_US - SPEED_NEAR_US)))


def report_firmware_lines(
    ser, buf: list[str], verbose: bool
) -> tuple[TcpPose | None, bool]:
    """Drain and print firmware chatter; return the newest TCP pose seen (if
    any) and whether an `mp` reply (`ok mp` or any `err ...`) showed up.

    A `?` query's reply (a `tcp x=... y=... ...` line among others) may show
    up here on a later tick than the one that sent it -- that's the point:
    the query is fired async (send_quick) rather than blocked on, so this is
    where the answer gets picked up whenever it actually arrives. Same idea
    for `mp`'s reply: run_tracking_loop uses it to know when the firmware is
    actually done with the last one and ready for another (see
    awaiting_mp_ack there).
    """
    latest_tcp: TcpPose | None = None
    mp_acked = False
    for line in drain_lines(ser, buf):
        tcp = parse_tcp_line(line)
        if tcp is not None:
            latest_tcp = tcp
        if line == "ok mp" or line.startswith("err "):
            mp_acked = True
        if line.startswith("err ") or line.startswith("lim "):
            print(line)
        elif verbose:
            print(line, file=sys.stderr)
    return latest_tcp, mp_acked


def draw_preview(
    frame: np.ndarray,
    detections: list[CubeDetection],
    match: CubeDetection | None,
    *,
    color: str,
    lost_since: float | None,
    dist: float | None,
) -> np.ndarray:
    """Annotate a copy of `frame` for live troubleshooting.

    Draws every detected blob (so a flickering or wrong-color match is
    visible), highlights whichever one is currently locked, and overlays the
    tracker's own state (lost/tracking, distance-to-target) so it's obvious
    *why* the arm is or isn't moving, not just where the cube is.
    """
    out = frame.copy()
    for det in detections:
        bgr = COLOR_BGR.get(det.color, UNKNOWN_COLOR_BGR)
        draw_cube_marker(out, det.px, det.py, bgr, det.color)
    if match is not None:
        draw_lock_ring(out, match.px, match.py, LOCK_BGR)

    state = "LOST" if lost_since is not None else "tracking"
    state_bgr = LOST_BGR if lost_since is not None else LOCK_BGR
    lines = [
        f"{color}: {state}",
        f"dist={dist:.1f}mm" if dist is not None else "dist=n/a",
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        draw_outlined_text(out, line, (10, y), scale=0.6, color=state_bgr)
    return out


def run_tracking_loop(
    ser,
    stream: FrameStream,
    calib: Calibration,
    color: str,
    *,
    seed: tuple[float, float, float],
    seed_tcp_xyz: tuple[float, float, float],
    seed_tcp_j4: float,
    hover_z: float,
    deadband_mm: float,
    lost_grace_s: float,
    verbose: bool,
    deadline: float | None = None,
    preview: bool = False,
    recorder: VideoRecorder | None = None,
    j1_center: int = J1_HOME_CENTER_STEPS,
    j2_pull: int = J2_HOME_PULLOFF_STEPS,
) -> None:
    buf: list[str] = [""]
    _t0, x0, y0 = seed
    current_tcp_x, current_tcp_y, current_tcp_z = seed_tcp_xyz
    # Passed straight back as `mp`'s j4_deg every tick so start_absolute_move
    # treats orientation as held (hold_ws_orient) rather than something to
    # actively rotate toward -- there's no wrist target here, tracking only
    # ever moves xyz.
    current_tcp_j4 = seed_tcp_j4

    last_known = (x0, y0)
    lost_since: float | None = None
    moving = False
    # True from the moment an `mp` is sent until its reply (`ok mp` or any
    # `err ...`) is seen -- see the dist>=deadband_mm branch below for why.
    awaiting_mp_ack = False
    last_telemetry = 0.0

    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                print("Reached --max-seconds, stopping.")
                return
            if console_focused() and key_down("h"):
                if moving:
                    send_quick(ser, "stop")
                    moving = False
                    awaiting_mp_ack = False
                print("Homing (h)...")
                run_home(ser, buf, j1_center, j2_pull, verbose)
                new_tcp, _mp_acked = report_firmware_lines(ser, buf, verbose)
                if new_tcp is not None:
                    current_tcp_x, current_tcp_y, current_tcp_z = new_tcp.x, new_tcp.y, new_tcp.z
                    current_tcp_j4 = new_tcp.j4
                lost_since = time.monotonic()
                continue
            # min_advance=1 (not FrameStream's default of 2): the camera
            # only delivers ~10fps here, and the default requires waiting
            # for 2 full new frames past the call, which halves our
            # achievable rate to ~5fps when called every tick like this.
            # min_advance=1 accepts a frame that may have already been
            # mid-capture at call time (up to ~1 frame period, ~100ms,
            # staler) in exchange for roughly doubling the loop rate to
            # ~10Hz -- a good trade for a slowly hand-moved cube.
            frame = stream.fresh(min_advance=1)
            now = time.monotonic()
            detections = detect_cubes(frame, calib)

            gate = MAX_JUMP_MM if lost_since is None else REACQUIRE_GATE_MM
            match = match_detection(detections, color, last_known, gate)
            if match is not None:
                assert match.x is not None and match.y is not None
                if lost_since is not None:
                    print(f"Reacquired {color} cube at ({match.x:.1f}, {match.y:.1f})")
                last_known = (match.x, match.y)
                lost_since = None
            elif lost_since is None:
                lost_since = now

            if lost_since is not None and now - lost_since > lost_grace_s:
                if moving:
                    send_quick(ser, "stop")
                    moving = False
                    awaiting_mp_ack = False
                    print(
                        f"Lost {color} cube -- holding position, watching for "
                        f"reacquire near ({last_known[0]:.1f}, {last_known[1]:.1f})"
                    )
                new_tcp, _mp_acked = report_firmware_lines(ser, buf, verbose)
                if new_tcp is not None:
                    current_tcp_x, current_tcp_y, current_tcp_z = new_tcp.x, new_tcp.y, new_tcp.z
                    current_tcp_j4 = new_tcp.j4
                if _emit_preview(
                    frame, detections, match, color, lost_since, None,
                    preview=preview, recorder=recorder,
                ):
                    print("Preview window closed (q/Esc) -- stopping.")
                    return
                continue

            target_x, target_y = clamp_to_envelope(*last_known)

            # `pos` derives the same `tcp ...` line as `?` (both come from
            # print_joint_pos()'s live forward-kinematics computation) but
            # skips `?`'s multi-line status dump (mode, limits, gripper,
            # header/footer) that we don't need every tick.
            send_quick(ser, "pos")  # fire-and-forget; picked up below whenever it lands
            new_tcp, mp_acked = report_firmware_lines(ser, buf, verbose)
            if mp_acked:
                awaiting_mp_ack = False
            if new_tcp is not None:
                current_tcp_x, current_tcp_y, current_tcp_z = new_tcp.x, new_tcp.y, new_tcp.z
                current_tcp_j4 = new_tcp.j4

            err_x = target_x - current_tcp_x
            err_y = target_y - current_tcp_y
            dist = math.hypot(err_x, err_y)

            if dist < deadband_mm:
                # Close enough: stop re-targeting and let whatever `mp` is
                # already in flight arrive and hold on its own. No `stop`
                # here -- `mp` is a bounded move with its own decel ramp, so
                # forcing a stop would trade a smooth arrival for an abrupt
                # one for no reason.
                moving = False
            else:
                # Only send a new `mp` once the previous one has actually
                # been acknowledged. `mp` now costs ~65-110ms of blocking
                # firmware time (see motion.cpp) -- right at the edge of
                # this loop's own ~100ms camera-paced tick, so firing a new
                # one every tick regardless (send_quick never waits for a
                # reply) let Python outpace what firmware could actually
                # absorb: commands queued up faster than they drained,
                # and the arm executed an ever-growing backlog of already-
                # stale targets in an uneven rhythm -- confirmed live as a
                # steady small-move/big-move alternation in tcp between
                # ticks, felt as the arm micro-stepping/jerking even though
                # each individual retarget is itself smooth. Gating on the
                # firmware's own ack (rather than a fixed delay or a
                # position deadzone -- see git history for why a drift-based
                # gate was tried and reverted) self-paces to whatever rate
                # firmware can actually sustain and always sends the
                # freshest known target the moment it's ready, so it can't
                # build a backlog and can't stall waiting for accumulated
                # drift either.
                if not awaiting_mp_ack:
                    send_quick(
                        ser,
                        f"mp {target_x:.1f} {target_y:.1f} {hover_z:.1f} "
                        f"{current_tcp_j4:.1f} 0 {speed_for_distance(dist)}",
                    )
                    awaiting_mp_ack = True
                moving = True

            if _emit_preview(
                frame, detections, match, color, lost_since, dist,
                preview=preview, recorder=recorder,
            ):
                print("Preview window closed (q/Esc) -- stopping.")
                return

            if verbose and now - last_telemetry > 1.0:
                print(
                    f"track: target=({target_x:.1f},{target_y:.1f}) "
                    f"tcp=({current_tcp_x:.1f},{current_tcp_y:.1f},{current_tcp_z:.1f}) "
                    f"dist={dist:.1f}",
                    file=sys.stderr,
                )
                last_telemetry = now
    finally:
        if preview:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except cv2.error:
                pass
        if recorder is not None:
            recorder.close()


def _emit_preview(
    frame: np.ndarray,
    detections: list[CubeDetection],
    match: CubeDetection | None,
    color: str,
    lost_since: float | None,
    dist: float | None,
    *,
    preview: bool,
    recorder: VideoRecorder | None,
) -> bool:
    """Draw+record/show one frame if requested; True if the user hit q/Esc.

    The tracking loop itself already runs at camera rate (~10Hz, paced by
    ``stream.fresh()``), so recording just writes whatever this tick already
    annotated -- no separate background feed is needed the way stack_cubes.py
    needed one to keep updating through its multi-second blocking moves.
    """
    if not preview and recorder is None:
        return False
    annotated = draw_preview(
        frame, detections, match,
        color=color, lost_since=lost_since, dist=dist,
    )
    if recorder is not None:
        recorder.write(annotated)
    if not preview:
        return False
    cv2.imshow(PREVIEW_WINDOW, annotated)
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Servo the gripper to hover over and track a single moving cube"
    )
    parser.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index (MT4_CAMERA_INDEX, else 0, if omitted; -1 auto-detects)")
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument(
        "--hover-mm", type=float, default=DEFAULT_HOVER_MM,
        help="height above the table to hover (mm); default 50 (5cm)",
    )
    parser.add_argument("--deadband-mm", type=float, default=DEADBAND_MM)
    parser.add_argument("--lost-grace-s", type=float, default=LOST_GRACE_S)
    parser.add_argument("--j1-center", type=int, default=J1_HOME_CENTER_STEPS)
    parser.add_argument("--j2-pull", type=int, default=J2_HOME_PULLOFF_STEPS)
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="exit cleanly (stop + close) after this many seconds; default runs until Ctrl+C",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="show a live annotated camera preview window (q or Esc to stop)",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="record the live annotated preview to this video path (e.g. track_run.mp4)",
    )
    parser.add_argument(
        "--record-fps", type=float, default=10.0,
        help="playback-rate metadata for --record (default 10, matching the "
        "camera-paced tracking loop; does not change the loop's actual pace)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    try:
        calib = load_calibration(Path(args.calib))
    except CalibrationError as exc:
        print(exc, file=sys.stderr)
        return 1

    camera_index = args.camera if args.camera is not None else DEFAULT_CAMERA_INDEX
    print("Opening camera...")
    try:
        stream = FrameStream(camera_index)
    except CameraError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        return _run(args, calib, stream)
    finally:
        stream.close()


def _run(args: argparse.Namespace, calib: Calibration, stream: FrameStream) -> int:
    hover_z = calib.table_z + args.hover_mm
    client = Mt4Client(port=args.port, baud=args.baud)
    target: CubeDetection | None = None
    seed_tcp_xyz: tuple[float, float, float] | None = None
    seed_tcp_j4: float | None = None
    seed: tuple[float, float, float] | None = None
    try:
        client.ensure_connected()
        ensure_homed(client)
        print("Homed. Looking for a cube to track...")
        target = find_initial_target(stream, calib, INITIAL_LOCK_TIMEOUT_S)
        if target is None:
            print(
                "No reachable cube found within the timeout -- place a cube "
                "on the table within the arm's reach and in camera view.",
                file=sys.stderr,
            )
            return 1
        assert target.x is not None and target.y is not None
        seed = (time.monotonic(), target.x, target.y)
        tx, ty = clamp_to_envelope(target.x, target.y)
        print(
            f"Locked onto {target.color} cube at ({tx:.1f}, {ty:.1f}); "
            f"moving to hover at z={hover_z:.1f}"
        )
        # travel_speed_us, not approach_speed_us: this is a lateral transit
        # at hover height (nothing gripped, nothing near the work surface),
        # so it gets the fast travel speed -- approach_speed_us is the
        # deliberately slow pick/place descent speed and made this opening
        # move crawl at 3.4x slower than the arm's max.
        client.move_to(tx, ty, hover_z, speed_us=calib.travel_speed_us)
        tcp = client.get_tcp()
        seed_tcp_xyz = (tcp.x, tcp.y, tcp.z)
        seed_tcp_j4 = tcp.j4
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()

    assert (
        target is not None and seed is not None and seed_tcp_xyz is not None
        and seed_tcp_j4 is not None
    )
    port = args.port
    baud = args.baud

    print("Switching to direct jog control for tracking. Ctrl+C to stop, H (this window/terminal focused) to re-home.")
    try:
        with open_serial(port, baud) as ser:
            try:
                await_firmware_alive(ser, port_label=port or "auto-detected port")
                drain_lines(ser, [""])
            except FirmwareNotReadyError as exc:
                print(exc, file=sys.stderr)
                return 1
            recorder = (
                VideoRecorder(video_path=args.record, fps=args.record_fps)
                if args.record
                else None
            )
            try:
                deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
                run_tracking_loop(
                    ser, stream, calib, target.color,
                    seed=seed, seed_tcp_xyz=seed_tcp_xyz, seed_tcp_j4=seed_tcp_j4,
                    hover_z=hover_z,
                    deadband_mm=args.deadband_mm,
                    lost_grace_s=args.lost_grace_s, verbose=args.verbose,
                    deadline=deadline, preview=args.preview, recorder=recorder,
                    j1_center=args.j1_center, j2_pull=args.j2_pull,
                )
            except KeyboardInterrupt:
                print()
            finally:
                try:
                    send_quick(ser, "stop")
                except SerialGoneError:
                    pass
    except SerialGoneError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
