#!/usr/bin/env python3
"""Reverse a cube stack built by stack_cubes.py: scatter it back on the desk.

The site is a marker id passed on the CLI (required, like stack_cubes.py),
paired with ``--stack-height`` -- the number of cubes currently standing
there. The operator is trusted about that count, same trust model as
stack_cubes.py's ``--resume``: a wrong value will crash the gripper into or
release above/below the real stack.

Cubes come off the *top* of the column first, by dead reckoning: the
marker's calibrated XY plus the known per-level grip height locate each
cube directly (mirroring how stack_cubes.py places by dead reckoning, no
vision needed to find the target). Each one is then carried clear of the
column and released at a randomly chosen open spot on the table at a
randomly chosen orientation, kept clear of every marker, every cube
scattered earlier in this run, and any pre-existing loose cube seen in
one initial scan by at least ``DROP_SPACING_FALLBACKS_MM[0]`` (degrading
to a tighter fallback only if the desk is too crowded for that). As with
stack_cubes.py's placement, there is no visual alignment or post-place
verification of the drop itself.

Column-safe transit reuses the same ``mt4_vision.stackpath.StackPlanner``
+ ``routed_travel``/``go_camera_park`` machinery stack_cubes.py uses to
build the column (both now live in ``mt4_vision.pickplace``), so a fix to
one direction's path planning can't silently diverge from the other's.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path

from jog import console_focused, flush_console_input, key_down
from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import DEFAULT_CALIB_PATH, CalibrationError, load_calibration
from mt4_vision.camera import FrameStream, capture_frame
from dataclasses import replace

from mt4_vision.landing import (
    LANDING_MAX_RADIUS_MM,
    LANDING_MIN_RADIUS_MM,
    random_landing as _random_landing,
)
from mt4_vision.motion import Leg, barrier, plan_route_legs, send_legs, station
from mt4_vision.pickplace import (
    CAMERA_PARK_X,
    CAMERA_PARK_Y,
    ensure_homed,
    go_camera_park,
    home_arm,
    j4_for_face_align,
    near_camera_park,
    place,
    routed_travel,
)
from mt4_vision.preview import LiveFeed, PreviewStopped
from mt4_vision.scene import capture_scene
from mt4_vision.stackpath import StackPlanner
from mt4_vision.workspace import (
    MAX_REACH_MM,
    MarkerSlot,
    dist_mm,
    is_mp_reachable_xy,
    marker_slots_from_calibration,
)

# Minimum clearance between two scattered cubes. PICK_CLEARANCE_MM (45mm,
# workspace.py) is the bare minimum before fingers clip a neighbor on a
# later pick; this adds margin so release drag can't reintroduce a close
# neighbor (same reasoning as stack_cubes.py's CLEAR_PARK_MM). Degrades
# through the fallbacks, tightest last, if the desk is too full for the
# preferred spacing -- better a tighter-than-ideal drop than a stall.
DROP_SPACING_FALLBACKS_MM = (75.0, 60.0, 45.0)

# Scatter radius band (mm, robot frame). Floor/ceiling live on
# mt4_vision.landing so stack clear and unstack scatter can't drift apart
# again (field cases 2026-07-24 / 2026-07-29).
SCATTER_MIN_RADIUS_MM = LANDING_MIN_RADIUS_MM
SCATTER_MAX_RADIUS_MM = LANDING_MAX_RADIUS_MM

# Cubes this close to the unstack site are left alone -- clear of the
# column while it's still standing, and out of the way once it's gone.
SITE_AVOID_MM = 90.0

CAMERA_SETTLE_S = 0.8
LANDING_ATTEMPTS = 300


def marker_by_id(calib, marker_id: int) -> MarkerSlot:
    slots = marker_slots_from_calibration(calib)
    for m in slots:
        if m.marker_id == marker_id:
            return m
    known = [m.marker_id for m in slots]
    raise SystemExit(
        f"marker {marker_id} not in calibration; known ids: {known}"
    )


class _HomeKeyWatcher:
    """Catch a tap of H (same binding as jog.py) on a background thread and
    abort whatever the arm is doing right now (same mechanism shuffle.py
    uses via ``Mt4Client.request_interrupt``).

    ``key_down`` (GetAsyncKeyState) only reports the key at the instant it's
    polled, and calling ``request_interrupt`` only helps if something polls
    for it -- neither happens on its own inside this script's seconds-apart
    loop checkpoints. This thread polls at 20Hz so a normal tap is always
    caught, and calls ``request_interrupt`` immediately on the press edge so
    an in-flight ``move_to``/``gripper`` call aborts within a fraction of a
    second instead of running to completion first. Gated on
    ``console_focused`` so an H press in another window doesn't re-home the
    arm mid-unstack.
    """

    def __init__(self, client: Mt4Client) -> None:
        self._client = client
        self._requested = threading.Event()
        self._h_down = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="unstack-home-key", daemon=True
        )

    def start(self) -> None:
        if sys.platform == "win32":
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.3)

    def consume(self) -> bool:
        if self._requested.is_set():
            self._requested.clear()
            return True
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            down = console_focused() and key_down("h")
            if down and not self._h_down:
                self._requested.set()
                self._client.request_interrupt()
            self._h_down = down
            self._stop.wait(0.05)


def _run_home(client: Mt4Client, watcher: _HomeKeyWatcher) -> None:
    print("Homing (H)...")
    watcher.consume()
    client.clear_interrupt()
    try:
        client.stop()
    except Mt4ClientError:
        pass
    home_arm(client)
    print("Home ok")


def _check_home(client: Mt4Client, watcher: _HomeKeyWatcher) -> bool:
    """Checkpoint catch: an H tap that landed while no client call was in
    flight, so there was no in-progress move for request_interrupt to abort."""
    if not watcher.consume():
        return False
    _run_home(client, watcher)
    return True


def _home_requested(watcher: _HomeKeyWatcher, exc: Mt4ClientError) -> bool:
    """True when a Mt4ClientError was this H-triggered abort, not a real failure."""
    return watcher.consume() or "interrupted" in str(exc)


def random_landing(
    rng: random.Random,
    *,
    sx: float,
    sy: float,
    markers: list[MarkerSlot],
    avoid: list[tuple[float, float]],
    spacing_mm: float,
    attempts: int = LANDING_ATTEMPTS,
) -> tuple[float, float] | None:
    """A random reachable table XY at least ``spacing_mm`` from everything
    in ``avoid`` (prior drops, pre-existing cubes) and every marker paper,
    or None when nothing turned up in ``attempts`` tries."""
    return _random_landing(
        rng, sx=sx, sy=sy, markers=markers, avoid=avoid,
        spacing_mm=spacing_mm, site_avoid_mm=SITE_AVOID_MM,
        attempts=attempts,
        min_radius_mm=SCATTER_MIN_RADIUS_MM,
        max_radius_mm=SCATTER_MAX_RADIUS_MM,
    )


def find_landing(
    rng: random.Random,
    *,
    sx: float,
    sy: float,
    markers: list[MarkerSlot],
    avoid: list[tuple[float, float]],
) -> tuple[tuple[float, float], float]:
    """Best-effort landing spot: try the preferred spacing first, then
    degrade through ``DROP_SPACING_FALLBACKS_MM`` before giving up."""
    # Loop here (not via landing.find_landing) so tests can monkeypatch
    # this module's random_landing.
    for spacing in DROP_SPACING_FALLBACKS_MM:
        landing = random_landing(
            rng, sx=sx, sy=sy, markers=markers, avoid=avoid, spacing_mm=spacing,
        )
        if landing is not None:
            return landing, spacing
    raise Mt4ClientError(
        "no free landing spot on the table -- desk is too cluttered to unstack"
    )


def random_place_j4(x: float, y: float, rng: random.Random) -> float:
    """A joint-limit-safe world J4 at a random visual orientation.

    Cubes are square (90°-periodic), so folding a uniform draw over a full
    turn through ``j4_for_face_align`` still lands anywhere on that face
    while respecting the joint-J4 soft limits at this bearing -- the same
    helper ``place()``'s axis-align path uses, just seeded with a random
    angle instead of 0.
    """
    return j4_for_face_align(
        rng.uniform(0.0, 360.0), current_j4_deg=None, x=x, y=y,
    )


def pick_from_stack(
    client: Mt4Client,
    calib,
    planner: StackPlanner,
    level: int,
    *,
    approach_prefer_xy: tuple[float, float],
) -> None:
    """Take the level-``level`` cube off the top of the column.

    Mirrors ``place_on_stack`` (stack_cubes.py) in reverse: hover in over
    the column at the same clearance height that cube was placed at,
    descend to the same grip line stack_cubes.py used to seat it
    (``grip_top_z(level - 1)`` -- identical to a table ``table_z`` grip shifted
    up by the cubes still below it), close as a firmware station, then lift
    straight back out along the column axis before exiting. Column obstacle
    height is ``level`` (this cube hasn't left yet) for the approach and
    grasp, dropping to ``level - 1`` once it's lifted clear. The whole
    approach → barrier → descend → close → lift → exit is one `mq` queue.
    """
    sx, sy = planner.sx, planner.sy
    ensure_homed(client)
    hz = planner.hover_z(level)
    if hz is None:
        raise Mt4ClientError(f"level {level}: hover height unreachable")
    grip_z = planner.grip_top_z(level - 1)
    j4 = j4_for_face_align(0.0, current_j4_deg=None, x=sx, y=sy)
    tcp = client.get_tcp()
    if tcp is None:
        raise Mt4ClientError("stack pick: could not read TCP")
    start = (float(tcp.x), float(tcp.y), float(tcp.z))
    stage = planner.stage_point(hz, level, prefer_xy=start[:2])
    if stage is None:
        raise Mt4ClientError(f"level {level}: no reachable hover stage")
    # Same shape as motion.plan_pick_legs, but column-routed: open on the
    # first transit leg, barrier at hover before the slow descend, close
    # station at grip height, lift clear, optional exit hop -- one mq, no
    # blocking gripper round trips.
    legs = plan_route_legs(
        calib, planner, start, stage[0], stage[1], hz, level,
        j4=j4, then=[(sx, sy, hz)], descend=(sx, sy, grip_z),
        step=f"level {level} approach",
    )
    if legs:
        legs[0] = replace(legs[0], grip=int(calib.grip_open_s))
        legs.insert(-1, barrier((sx, sy, hz), j4=j4))
    else:
        legs = [station((sx, sy, grip_z), int(calib.grip_open_s), j4=j4)]
    legs.append(
        station((sx, sy, grip_z), int(calib.grip_close_s), j4=j4)
    )
    legs.append(
        Leg(sx, sy, hz, j4=j4, speed_us=int(calib.travel_speed_us))
    )
    exit_pt = planner.stage_point(hz, level - 1, prefer_xy=approach_prefer_xy)
    if exit_pt is not None:
        legs.append(
            Leg(
                exit_pt[0], exit_pt[1], hz,
                j4=j4, speed_us=int(calib.travel_speed_us),
            )
        )
    send_legs(client, legs, step=f"level {level} pick")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unstack cubes off a calibrated marker and scatter them on the table"
    )
    parser.add_argument(
        "--marker",
        type=int,
        required=True,
        help="calibration ArUco marker id the stack stands on (required)",
    )
    parser.add_argument(
        "--stack-height",
        type=int,
        required=True,
        metavar="N",
        help="number of cubes currently standing on the marker (required -- "
        "the operator is trusted about N, same as stack_cubes.py's --resume; "
        "a wrong value will crash the gripper into or release above/below "
        "the real stack)",
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument(
        "--preview", action="store_true",
        help="show a live annotated camera preview window (q or Esc to stop)",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="record a live annotated video to this path (e.g. unstack_run.mp4)",
    )
    parser.add_argument(
        "--feed-fps",
        type=float,
        default=10.0,
        help="capture/annotate rate for --preview and --record (default 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the random placement/orientation draw (default: unseeded)",
    )
    args = parser.parse_args()

    try:
        calib = load_calibration(Path(args.calib))
    except CalibrationError as exc:
        print(exc, file=sys.stderr)
        return 1

    marker = marker_by_id(calib, args.marker)
    sx, sy = marker.x, marker.y
    if not is_mp_reachable_xy(sx, sy) or math.hypot(sx, sy) > MAX_REACH_MM:
        print(
            f"marker {marker.marker_id} at ({sx:.1f},{sy:.1f}) is out of reach",
            file=sys.stderr,
        )
        return 1
    if near_camera_park(sx, sy):
        print(
            f"marker {marker.marker_id} at ({sx:.1f},{sy:.1f}) sits under the "
            f"camera park ({CAMERA_PARK_X:.0f},{CAMERA_PARK_Y:.0f}) -- the arm "
            "parks there between moves and would hit the stack; use another "
            "marker",
            file=sys.stderr,
        )
        return 1
    if args.stack_height < 1:
        print(f"--stack-height must be >= 1, got {args.stack_height}", file=sys.stderr)
        return 1

    camera_kwargs = {} if args.camera is None else {"index": args.camera}
    client = Mt4Client() if not args.port else Mt4Client(port=args.port)
    planner = StackPlanner(calib, sx, sy)
    rng = random.Random(args.seed)
    remaining = int(args.stack_height)
    watcher: _HomeKeyWatcher | None = None

    # A live feed and this script's own "look now" capture can't both open
    # the camera independently (one device, one owner) -- when either
    # --preview or --record is on, both pull frames from this one shared
    # stream instead of a one-shot capture_frame().
    stream = FrameStream(**camera_kwargs) if (args.preview or args.record) else None
    live_feed = (
        LiveFeed(
            calib=calib, stream=stream, fps=args.feed_fps,
            video_path=args.record, show_preview=args.preview,
        )
        if stream is not None
        else None
    )
    client.trajectory_sink = live_feed

    def snap_obstacles(levels: int) -> list[tuple[float, float]]:
        """Pre-existing loose cubes to steer clear of, from one clear-view scan."""
        go_camera_park(client, calib, planner, levels)
        time.sleep(CAMERA_SETTLE_S)
        frame = stream.fresh(min_advance=2) if stream is not None else capture_frame(**camera_kwargs)
        scene = capture_scene(calib, frame)
        if live_feed is not None:
            live_feed.set_status(["Scanning for obstacles", scene.summary_line()])
        return [
            (float(c.x), float(c.y))
            for c in scene.raw_cubes
            if c.x is not None and c.y is not None
            and dist_mm(float(c.x), float(c.y), sx, sy) >= SITE_AVOID_MM
        ]

    try:
        client.ensure_connected()
        status = client.get_status()
        if not status.homed:
            print("Homing...")
            home_arm(client)
        else:
            print("Already homed")
        status = client.get_status()
        # Max cruise (lowest step period) for the session.
        client.move_to(
            status.tcp.x, status.tcp.y, status.tcp.z,
            speed_us=calib.travel_speed_us,
        )

        print(
            f"Unstack site: marker {marker.marker_id} at ({sx:.1f},{sy:.1f}), "
            f"{remaining} level(s) standing, cube_height={calib.cube_height_mm:.1f}mm"
        )

        # Preflight: every level must have a reachable hover/grip height --
        # if it was buildable it should still be liftable, but confirm
        # before touching anything.
        for level in range(1, remaining + 1):
            if planner.hover_z(level) is None:
                print(
                    f"joint z ceiling at this site is {planner.site_max_z:.0f}mm "
                    f"-- level {level} is not reachable to lift",
                    file=sys.stderr,
                )
                return 1

        all_markers = marker_slots_from_calibration(calib)
        obstacles = snap_obstacles(remaining)
        if obstacles:
            print(f"Avoiding {len(obstacles)} pre-existing loose cube(s) on the table")

        print("Ctrl+C to stop, H (this window/terminal focused) to re-home before the next step.")
        watcher = _HomeKeyWatcher(client)
        watcher.start()

        placed: list[tuple[float, float]] = []
        while remaining > 0:
            level = remaining
            if live_feed is not None and live_feed.stopped_by_user.is_set():
                raise PreviewStopped()
            if _check_home(client, watcher):
                continue
            try:
                landing, spacing = find_landing(
                    rng, sx=sx, sy=sy, markers=all_markers, avoid=placed + obstacles,
                )
                tx, ty = landing
                j4 = random_place_j4(tx, ty, rng)
                print(
                    f"\nLevel {level}: lifting off stack -> "
                    f"({tx:.0f},{ty:.0f}) yaw={j4:.0f}deg (spacing={spacing:.0f}mm)"
                )
                if live_feed is not None:
                    live_feed.set_status([
                        f"Level {level}/{args.stack_height}: unstacking -> "
                        f"({tx:.0f},{ty:.0f})",
                    ])
                pick_from_stack(
                    client, calib, planner, level,
                    approach_prefer_xy=(CAMERA_PARK_X, CAMERA_PARK_Y),
                )
                routed_travel(
                    client, calib, planner, tx, ty, calib.safe_z, level - 1,
                    j4=j4, step=f"level {level} carry to landing",
                )
                place(client, calib, tx, ty, j4=j4, travel_z=calib.safe_z)
            except Mt4ClientError as exc:
                if _home_requested(watcher, exc):
                    _run_home(client, watcher)
                    continue
                print(f"  level {level} failed: {exc}", file=sys.stderr)
                return 1
            placed.append((tx, ty))
            remaining = level - 1
            print(f"  placed level {level} at ({tx:.0f},{ty:.0f})")

        print(f"\nUnstacked {args.stack_height} cube(s) from marker {marker.marker_id}")
        try:
            go_camera_park(client, calib, planner, 0)
        except Mt4ClientError as exc:
            if _home_requested(watcher, exc):
                _run_home(client, watcher)
            else:
                print(exc, file=sys.stderr)
        return 0
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    except PreviewStopped:
        print("Preview window closed (q/Esc) -- stopping.")
        go_camera_park(client, calib, planner, remaining)
        return 0
    finally:
        if watcher is not None:
            watcher.close()
        client.close()
        if live_feed is not None:
            live_feed.close()
        if stream is not None:
            stream.close()
        if args.record:
            print(f"Recording saved to {args.record}")


if __name__ == "__main__":
    try:
        _exit_code = main()
    finally:
        flush_console_input()
    raise SystemExit(_exit_code)
