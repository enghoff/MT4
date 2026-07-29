"""Pick and place sequences for cubes on the calibrated work surface."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.joints import JOINT_SOFT_MAX_STEPS, JOINT_SOFT_MIN_STEPS, JOG_SPEED_MIN_US
from mt4_jog.kinematics import (
    HOME_J1_DEG,
    HOME_J2_DEG,
    HOME_J3_DEG,
    HOME_J4_DEG,
    STEPS_PER_DEG,
    JointAnglesDeg,
    ik_position,
)
from mt4_vision.calib import Calibration
from mt4_vision.detect import CubeDetection
from mt4_vision.stackpath import StackPlanner
from mt4_vision.workspace import KEEPOUT_RADIUS_MM, is_mp_reachable_xy


def fold_square_yaw_deg(yaw_deg: float) -> float:
    """Map any angle into (-45, 45] -- one face of a square (90° period)."""
    return (yaw_deg + 45.0) % 90.0 - 45.0


def j4_for_face_align(
    cube_yaw_deg: float,
    *,
    current_j4_deg: float | None = None,
    x: float | None = None,
    y: float | None = None,
    j4_margin_steps: int = 200,
) -> float:
    """World-frame J4 (deg) so the jaws meet a cube face, not a corner.

    Assumes firmware ``j4zero``: jaws along the arm ⇒ world J4 = 0.
    ``cube_yaw_deg`` is a robot-frame edge angle from detection. Squares are
    90°-periodic; when ``current_j4_deg`` is given, pick the equivalent that
    minimizes wrist travel -- but only among candidates whose *joint* J4
    (world − j1) stays inside soft limits at (x, y). Preferring nearest
    world yaw alone can pin joint J4 past ±8100 on far −Y picks after the
    wrist has drifted near 90° (stack level-4: world 109° at j1≈−72° →
    joint 181° / 8130 steps → ``err mp joints``).
    """
    base = fold_square_yaw_deg(cube_yaw_deg)
    # Face-aligned lattice: base + k*90.
    candidates = [base + 90.0 * k for k in range(-4, 5)]
    if x is not None and y is not None:
        j1 = math.degrees(math.atan2(y, x))
        lo = JOINT_SOFT_MIN_STEPS[3] / STEPS_PER_DEG[3] + j4_margin_steps / STEPS_PER_DEG[3]
        hi = JOINT_SOFT_MAX_STEPS[3] / STEPS_PER_DEG[3] - j4_margin_steps / STEPS_PER_DEG[3]
        feasible = [w for w in candidates if lo <= (w - j1) <= hi]
        if feasible:
            candidates = feasible
    if current_j4_deg is None:
        # Prefer the folded representative (or the feasible one closest to it).
        return min(candidates, key=lambda w: abs(w - base))
    return min(candidates, key=lambda w: abs(w - current_j4_deg))


def j4_preserve_wrist(
    x: float,
    y: float,
    *,
    from_x: float,
    from_y: float,
    from_j4: float,
) -> float:
    """World-frame J4 that keeps joint J4 fixed across a J1 swing.

    ``Mt4Client.move_to(j4=None)`` holds *world* yaw, which commands
    ``joint_j4 = world_j4 - j1``. Large base swings (e.g. to marker 0 at
    j1≈−80°) then drive J4 past soft limits → ``err mp joints``. Holding
    the wrist joint instead yields ``world_j4 = j1_tgt + (from_j4 - j1_from)``.
    """
    j1_from = math.degrees(math.atan2(from_y, from_x))
    j1_to = math.degrees(math.atan2(y, x))
    return j1_to + (from_j4 - j1_from)


def _resolve_travel_j4(j4: float | str | None) -> float | str:
    """Explicit j4 passes through; None becomes the firmware `w` sentinel.

    `w` holds the J4 *joint* angle across the leg's J1 swing, resolved
    on-device at leg-plan time -- the firmware-native version of the old
    host-side TCP probe + j4_preserve_wrist() computation (kept above for
    reference and tests), with identical endpoint behavior, one less serial
    round trip per travel, and correct per-leg resolution on queued
    (`mq`/move_path) waypoints.
    """
    return "wrist" if j4 is None else float(j4)


def _check(result: dict[str, object], step: str) -> dict[str, object]:
    """move_to/home/gripper report failure via {"ok": False, ...}, not an
    exception -- callers here must not chain past a step that never
    happened, so turn a failed result into one."""
    if not result.get("ok"):
        raise Mt4ClientError(f"{step} failed: {result.get('error', result)}")
    return result


@dataclass(frozen=True)
class MoveEvent:
    """One completed client.move_to()/gripper() call within pick()/place().

    Fired via ``on_waypoint`` for callers (e.g. mt4_pi.collect) that want
    the actual commanded waypoints instead of reconstructing them after the
    fact. Any field left None means "unchanged from the previous event" --
    the same convention Mt4Client.move_to's own ``grip=0`` already uses.

    ``started_at`` is ``time.monotonic()`` sampled immediately *before* the
    blocking call, so a consumer can tell motion apart from the dead time
    that precedes it. That dead time is not small: place() alone spends
    ensure_homed + resolve_place_j4 + get_tcp -- up to three ``?`` round
    trips at ~200ms each (mt4_jog.serial.send: 50ms write settle + reply +
    100ms quiet gap) -- standing still before its first move goes out.
    Without this field a consumer interpolating between event *arrival*
    times smears that stall across the leg as if the arm were flying.

    ``ended_at`` is when this waypoint was actually reached. It is None for
    a plain blocking move -- the consumer can use its own receive time,
    which is the same instant. It is set only when several waypoints are
    reported after ONE queued call (move_path/mq): those all return
    together, so without an explicit per-leg end time a consumer would see
    the arm teleport through the whole route at the moment the call
    returned. See ``_emit_path_waypoints``.
    """

    step: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    j4: float | None = None
    grip: int | None = None
    started_at: float | None = None
    ended_at: float | None = None


def _travel(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    z: float,
    step: str,
    *,
    j4: float | str | None = None,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> dict[str, object]:
    """Horizontal or lift move at safe travel speed (firmware mp ramp active)."""
    started = time.monotonic()
    result = _check(
        client.move_to(
            x, y, z, j4=_resolve_travel_j4(j4),
            speed_us=calib.travel_speed_us,
        ),
        step,
    )
    if on_waypoint is not None:
        # j4 may be the "hold"/"wrist" firmware sentinel (None or "wrist"),
        # resolved on-device -- only report it when the caller gave us a
        # concrete world-frame angle.
        on_waypoint(MoveEvent(
            step, x, y, z,
            j4 if isinstance(j4, (int, float)) else None,
            started_at=started,
        ))
    return result


def _leg_step_counts(
    start_xyz: tuple[float, float, float],
    wps: list[tuple[float, float, float]],
    j4_list: list[object],
    start_j4_world: float | None,
) -> list[float] | None:
    """Motor steps on the busiest axis for each leg of a queued path.

    The firmware paces a leg with a DDA clocked at ``speed_us`` per step, so
    a leg's duration is set by whichever axis has the most steps to take --
    NOT by how far the TCP travels in millimetres. Those two disagree
    violently for a leg that mostly rotates the wrist: measured on a live
    level-1 carry, the 1.2mm nudge from safe_z to hover height also swings
    J4 by 80.8 deg (the face-align angle takes effect on the leg after the
    "wrist"-held lift). By distance it is 0.2% of the path and gets 30ms of
    the 12s carry; by steps it is 3634 of them and takes real time. That gap
    is what put an 80.8 deg single-tick discontinuity into every recorded
    stacking episode.

    j4 bookkeeping: a concrete world-frame angle commands joint
    ``j4 - j1``, while the ``"wrist"`` sentinel holds the *joint* angle
    across the leg's J1 swing (so world j4 moves with j1 and joint J4 does
    not). Steps are counted on the joint, which is what the motor turns.

    Returns None when any waypoint has no IK solution -- the caller then
    falls back to distance weighting rather than guessing.
    """
    near = JointAnglesDeg(HOME_J1_DEG, HOME_J2_DEG, HOME_J3_DEG, HOME_J4_DEG)
    start = ik_position(*start_xyz, near=near, hold_orientation=False)
    if start is None:
        return None
    prev = (start.j1, start.j2, start.j3)
    # Joint J4 at the start of the path. Unknown world j4 (no TCP read) means
    # we cannot track the wrist at all; treat it as stationary rather than
    # inventing a swing.
    prev_j4_joint = (
        0.0 if start_j4_world is None else float(start_j4_world) - start.j1
    )
    near = JointAnglesDeg(start.j1, start.j2, start.j3, prev_j4_joint)

    counts: list[float] = []
    for i, wp in enumerate(wps):
        sol = ik_position(*wp, near=near, hold_orientation=False)
        if sol is None:
            return None
        cmd = j4_list[i] if i < len(j4_list) else None
        # "wrist"/None hold the joint; a number is a world angle to convert.
        j4_joint = (
            float(cmd) - sol.j1 if isinstance(cmd, (int, float)) else prev_j4_joint
        )
        deltas = (
            abs(sol.j1 - prev[0]), abs(sol.j2 - prev[1]),
            abs(sol.j3 - prev[2]), abs(j4_joint - prev_j4_joint),
        )
        counts.append(max(d * STEPS_PER_DEG[j] for j, d in enumerate(deltas)))
        prev = (sol.j1, sol.j2, sol.j3)
        prev_j4_joint = j4_joint
        near = JointAnglesDeg(sol.j1, sol.j2, sol.j3, j4_joint)
    return counts


def _emit_path_waypoints(
    on_waypoint: Callable[[MoveEvent], None],
    step: str,
    start_xyz: tuple[float, float, float],
    wps: list[tuple[float, float, float]],
    j4_list: list[object],
    speeds: list[int],
    started: float,
    ended: float,
    start_j4_world: float | None = None,
) -> None:
    """Report one MoveEvent per leg of a completed queued (mq) path.

    A queued path is ONE blocking call covering several legs, so there is no
    per-leg completion time to observe. Reporting a single event at the end
    would make a consumer interpolate a straight line from the start pose to
    the final pose -- but these routes are deliberately non-straight (lift
    up, route across, hop over the column, descend down its axis), so that
    line runs through poses the arm was never at, and any frame captured
    mid-carry gets labelled with one. That is worse than coarse: it is
    confidently wrong, in a way a model trained on it would learn.

    So split the measured wall duration across legs by ``steps * speed_us``
    -- speed_us is microseconds per step, so a leg's time scales with both
    how much the motors turn and how slow the leg is run. The speed term
    matters because the final ``descend`` leg runs at approach speed
    (several times slower than travel), and ignoring it would give that leg
    a small slice of time it does not take.

    Steps, not millimetres: see ``_leg_step_counts``. Distance weighting
    silently zeroes any leg that rotates the wrist without moving the TCP,
    and those legs are common -- ``routed_travel`` puts the face-align angle
    on the leg *after* the wrist-held lift, which is often a 1mm height
    adjustment. Distance falls back in only when IK cannot solve the route.

    Still an approximation: the firmware's accel ramp across the queue is
    not modeled, same caveat as WaypointLog.state_at.
    """
    counts = _leg_step_counts(start_xyz, wps, j4_list, start_j4_world)
    legs: list[tuple[tuple[float, float, float], float]] = []
    prev = start_xyz
    for i, wp in enumerate(wps):
        speed = float(speeds[i]) if i < len(speeds) else 1.0
        effort = counts[i] if counts is not None else math.dist(prev, wp)
        legs.append((wp, effort * max(speed, 1.0)))
        prev = wp
    total = sum(w for _, w in legs)
    span = max(ended - started, 0.0)
    # Degenerate (zero-length path): report the endpoint once rather than
    # dividing by zero or emitting nothing at all.
    if total <= 0.0:
        wp = wps[-1]
        j4 = j4_list[-1] if j4_list else None
        on_waypoint(MoveEvent(
            step, wp[0], wp[1], wp[2],
            j4 if isinstance(j4, (int, float)) else None,
            started_at=started, ended_at=ended,
        ))
        return
    elapsed = 0.0
    leg_start = started
    for i, (wp, weight) in enumerate(legs):
        elapsed += weight
        leg_end = started + span * (elapsed / total)
        j4 = j4_list[i] if i < len(j4_list) else None
        on_waypoint(MoveEvent(
            f"{step} leg {i + 1}/{len(legs)}", wp[0], wp[1], wp[2],
            # "wrist"/None are firmware sentinels resolved on-device -- only
            # report a concrete world-frame angle.
            j4 if isinstance(j4, (int, float)) else None,
            started_at=leg_start, ended_at=leg_end,
        ))
        leg_start = leg_end


# Within this radius of a stack axis, motion must be pure horizontal or pure
# vertical (never a 3D diagonal) so the gripper cannot clip a tall column.
STACK_AXIS_CLEAR_MM = 50.0
_XY_EPS_MM = 0.5
_Z_EPS_MM = 0.5


def stack_clear_xy(
    sx: float,
    sy: float,
    from_x: float,
    from_y: float,
    radius_mm: float = STACK_AXIS_CLEAR_MM,
) -> tuple[float, float] | None:
    """Reachable XY at ``radius_mm`` from (sx, sy), preferring the approach ray."""
    dx, dy = from_x - sx, from_y - sy
    if math.hypot(dx, dy) < 1.0:
        dx, dy = sx, sy
    if math.hypot(dx, dy) < 1.0:
        dx, dy = 1.0, 0.0
    scale = math.hypot(dx, dy)
    ux, uy = dx / scale, dy / scale
    for angle_deg in (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
        ang = math.radians(angle_deg)
        ca, sa = math.cos(ang), math.sin(ang)
        vx, vy = ux * ca - uy * sa, ux * sa + uy * ca
        px, py = sx + vx * radius_mm, sy + vy * radius_mm
        if is_mp_reachable_xy(px, py):
            return (px, py)
    return None


def travel_orthogonal(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    z: float,
    step: str,
    *,
    j4: float | None = None,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> None:
    """Reach (x, y, z) via vertical-then-horizontal segments (no XYZ diagonal).

    When both segments are needed they go out as one queued firmware path
    (move_path) -- same orthogonal track, no stop/settle/reaccel or serial
    round trip at the corner.
    """
    tcp = client.get_tcp()
    if tcp is None:
        raise Mt4ClientError(f"{step}: could not read TCP")
    same_xy = math.hypot(float(tcp.x) - x, float(tcp.y) - y) < _XY_EPS_MM
    same_z = abs(float(tcp.z) - z) < _Z_EPS_MM
    if same_xy and same_z:
        return
    if not same_z and not same_xy:
        started = time.monotonic()
        _check(
            client.move_path(
                [(float(tcp.x), float(tcp.y), z), (x, y, z)],
                j4=_resolve_travel_j4(j4),
                speed_us=calib.travel_speed_us,
            ),
            step,
        )
        if on_waypoint is not None:
            _emit_path_waypoints(
                on_waypoint, step,
                (float(tcp.x), float(tcp.y), float(tcp.z)),
                [(float(tcp.x), float(tcp.y), z), (x, y, z)],
                [j4 if isinstance(j4, (int, float)) else None] * 2,
                [calib.travel_speed_us, calib.travel_speed_us],
                started, time.monotonic(),
                start_j4_world=float(tcp.j4),
            )
        return
    if not same_z:
        _travel(
            client, calib, float(tcp.x), float(tcp.y), z,
            f"{step}: vertical", j4=j4, on_waypoint=on_waypoint,
        )
    else:
        _travel(client, calib, x, y, z, f"{step}: horizontal", j4=j4, on_waypoint=on_waypoint)


def _approach(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    z: float,
    step: str,
    *,
    j4: float | str | None = None,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> dict[str, object]:
    """Slow final descent near the table (firmware ramp off)."""
    started = time.monotonic()
    result = _check(
        client.move_to(
            x, y, z, j4=_resolve_travel_j4(j4),
            speed_us=calib.approach_speed_us,
        ),
        step,
    )
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            step, x, y, z,
            j4 if isinstance(j4, (int, float)) else None,
            started_at=started,
        ))
    return result


# Camera-clear parking spot for between-move captures: the homed TCP pose.
# From the front-mounted camera the arm parked here only occludes the strip
# behind it -- essentially the mp keep-out region, where nothing pickable or
# placeable ever sits. Anywhere over the workspace, the forearm hides cubes
# and markers AND reads as cube-sized red blobs inside the workspace hull.
CAMERA_PARK_X = 200.0
CAMERA_PARK_Y = 0.0
CAMERA_PARK_Z = 260.0
CAMERA_PARK_CLEARANCE_MM = 80.0


def near_camera_park(x: float, y: float) -> bool:
    """True when (x, y) is too close to the camera-park TCP to place/pick."""
    return (
        (x - CAMERA_PARK_X) ** 2 + (y - CAMERA_PARK_Y) ** 2
    ) < CAMERA_PARK_CLEARANCE_MM**2


def retreat_for_camera(client: Mt4Client, calib: Calibration) -> dict[str, object]:
    """Move the TCP to the camera-clear park pose (post-move capture prep).

    Same orthogonal lift / traverse / drop track as always (a depart from
    over a stack must never diagonal into the column), but sent as ONE
    queued firmware path: one TCP read, one blocking call, no
    stop/settle/reaccel at the two corners -- this used to be up to three
    probe+move round trips per capture, on the hottest path in every
    vision loop.
    """
    tcp = client.get_tcp()
    if tcp is None:
        raise Mt4ClientError("retreat to camera park: could not read TCP")
    cx, cy, cz = float(tcp.x), float(tcp.y), float(tcp.z)
    z_hi = max(cz, CAMERA_PARK_Z, float(calib.safe_z))
    wps: list[tuple[float, float, float]] = []
    if z_hi - cz > _Z_EPS_MM:
        wps.append((cx, cy, z_hi))
    if math.hypot(cx - CAMERA_PARK_X, cy - CAMERA_PARK_Y) > _XY_EPS_MM:
        wps.append((CAMERA_PARK_X, CAMERA_PARK_Y, z_hi))
    if z_hi - CAMERA_PARK_Z > _Z_EPS_MM:
        wps.append((CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z))
    if wps:
        _check(
            client.move_path(wps, j4="wrist", speed_us=calib.travel_speed_us),
            "retreat to camera park",
        )
    return {"ok": True, "parked_at": [CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z]}


def routed_travel(
    client: Mt4Client,
    calib: Calibration,
    planner: StackPlanner,
    x: float,
    y: float,
    z: float,
    levels: int,
    *,
    j4: float | None = None,
    then: list[tuple[float, float, float]] | None = None,
    final_j4: float | None = None,
    lift_to: float | None = None,
    descend: tuple[float, float, float] | None = None,
    step: str = "stack transit",
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> None:
    """Travel to (x, y, z) along a StackPlanner route (direct when safe).

    The whole route goes out as one firmware-side `mq` waypoint queue
    (Mt4Client.move_path()) -- no stop/re-accelerate between waypoints
    (see the `mq` protocol doc in firmware/mt4_jog/src/main.cpp for what
    that does and doesn't smooth out). `j4=None` maps to the firmware `w`
    sentinel: the wrist *joint* angle is held leg-by-leg across each J1
    swing, resolved on-device from wherever the previous leg actually
    ended -- the per-leg behavior the old per-waypoint _travel() fallback
    loop existed to emulate.

    ``lift_to`` prepends a vertical lift-off to the route: the TCP rises
    straight up from wherever it is (e.g. still at grab height holding a
    cube) to this z BEFORE the column-aware route plans from there, so a
    post-grip lift folds into the carry mq with no stop between. The lift
    stays a pure vertical (held at "wrist" so the gripped orientation is
    preserved -- no J1 swing anyway); a diagonal lift-off from grab height
    could clip a neighbouring cube, which route()'s safety model does not
    cover (it guards the column, not table neighbours). No-op when the arm
    is already at/above ``lift_to``.

    ``then`` appends extra waypoints AFTER the planned route, in the same
    queued `mq` so there is no stop/settle at the route's end. These are
    NOT run through the column-safety planner -- the caller guarantees the
    continuation from (x, y, z) is safe (e.g. a horizontal hop over the
    stack top at hover height, where the fingertips already clear the
    column). Used to fold the "hover over stack" hop into the carry.

    ``descend`` appends a final SLOW leg (to this XYZ) run at
    ``calib.approach_speed_us`` while every other leg runs at travel speed,
    so the transit flows straight into the precise final descent with no
    stop/settle at hover/safe height. The firmware ramps travel->approach as
    it enters this leg (dda_continue_ramp is bidirectional), so the speed
    where the gripper actually meets the cube/stack matches a standalone
    ``_approach()`` -- only the fast upper part of the drop is saved. Like
    ``then``, it is NOT column-checked: the caller guarantees a pure
    vertical drop straight below the arrival XY.

    ``final_j4`` sets the wrist (world-frame yaw, deg) from the leg that
    ARRIVES at (x, y, z) onward (that arrival leg plus any ``then``/
    ``descend`` legs) while earlier transit legs hold ``j4``/wrist -- so the
    arm is already oriented before any hop or drop and no separate
    wrist-rotation move is needed. Only honored when the move actually runs
    (not on early-return).

    Shared by stack_cubes.py (levels grows as cubes are added) and
    unstack_cubes.py (levels shrinks as cubes come off) -- both route
    around the same column, so the safety model must stay identical.

    ``on_waypoint``, when given, fires once per leg of the resulting queued
    path (lift + route + then + descend), NOT once for the whole call -- see
    ``_emit_path_waypoints`` for why a single end-of-call event would be
    actively misleading for a route this shaped.
    """
    tcp = client.get_tcp()
    if tcp is None:
        raise Mt4ClientError(f"{step}: could not read TCP")
    cur = (float(tcp.x), float(tcp.y), float(tcp.z))
    lift_wps: list[tuple[float, float, float]] = []
    a = cur
    if lift_to is not None and float(lift_to) - cur[2] > _Z_EPS_MM:
        a = (cur[0], cur[1], float(lift_to))
        lift_wps = [a]
    tail = [(float(px), float(py), float(pz)) for px, py, pz in (then or [])]
    descend_wps = (
        []
        if descend is None
        else [(float(descend[0]), float(descend[1]), float(descend[2]))]
    )
    final = descend_wps[-1] if descend_wps else (tail[-1] if tail else (x, y, z))
    if not lift_wps and math.dist(cur, final) < 1.0:
        return
    wps = planner.route(a, (x, y, z), levels)
    if wps is None:
        raise Mt4ClientError(
            f"{step}: no stack-safe route from "
            f"({a[0]:.0f},{a[1]:.0f},{a[2]:.0f}) to ({x:.0f},{y:.0f},{z:.0f})"
        )
    all_wps = lift_wps + wps + tail + descend_wps
    transit_j4 = j4 if j4 is not None else "wrist"
    arrival_j4 = float(final_j4) if final_j4 is not None else transit_j4
    # Lift legs hold "wrist" (preserve the gripped orientation up the vertical
    # lift). Route transit legs hold the transit wrist. The leg that ARRIVES
    # at (x, y, z) and everything after it (then hops, the final descend) hold
    # arrival_j4, so the wrist is settled before any hop or drop rather than
    # turning into it (arrival_j4 == transit_j4 when no final_j4, i.e. uniform).
    n_lift = len(lift_wps)
    n_route_transit = len(wps) - 1
    j4_list = (
        ["wrist"] * n_lift
        + [transit_j4] * n_route_transit
        + [arrival_j4] * (len(all_wps) - n_lift - n_route_transit)
    )
    speeds = [calib.travel_speed_us] * len(all_wps)
    if descend_wps:
        speeds[-1] = int(calib.approach_speed_us)
    # Collapse to a scalar when uniform (the common case) so the wire form
    # and existing behavior are unchanged; only mixed legs go out as lists.
    j4_arg: object = j4_list[0] if all(v == j4_list[0] for v in j4_list) else j4_list
    speed_arg: object = speeds[0] if all(s == speeds[0] for s in speeds) else speeds
    started = time.monotonic()
    _check(client.move_path(all_wps, j4=j4_arg, speed_us=speed_arg), step)
    if on_waypoint is not None:
        _emit_path_waypoints(
            on_waypoint, step, cur, all_wps, j4_list, speeds,
            started, time.monotonic(), start_j4_world=float(tcp.j4),
        )


def go_camera_park(
    client: Mt4Client, calib: Calibration, planner: StackPlanner, levels: int
) -> dict[str, object]:
    """Move to the camera park pose; column-aware once a stack exists."""
    if levels > 0:
        routed_travel(
            client, calib, planner,
            CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z, levels,
            step="park transit",
        )
        return {"ok": True, "parked_at": [CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z]}
    return retreat_for_camera(client, calib)


def ensure_homed(client: Mt4Client) -> None:
    status = client.get_status()
    if not status.homed:
        home_arm(client)


def home_arm(client: Mt4Client) -> None:
    """Run firmware homing regardless of the session homed flag."""
    _check(client.home(), "home")


def _require_mp_reachable(x: float, y: float, step: str) -> None:
    if not is_mp_reachable_xy(x, y):
        raise Mt4ClientError(
            f"{step}: ({x:.1f}, {y:.1f}) is inside the {KEEPOUT_RADIUS_MM:.0f}mm "
            f"J1 keep-out zone (mp cannot move there)"
        )


def resolve_pick_j4(
    client: Mt4Client,
    calib: Calibration,
    yaw_deg: float | None,
    *,
    face_align: bool = True,
    x: float | None = None,
    y: float | None = None,
) -> float | None:
    """Face-align world J4, or None so ``_travel`` preserves joint J4 instead.

    None must not mean world-yaw hold: that trips J4 soft limits on large
    J1 swings (marker 0 / far −Y). ``_travel``/``_approach`` map None to
    ``j4_preserve_wrist``. Pass pick (x, y) so face-align stays inside
    joint-J4 soft limits at the target bearing.
    """
    if not face_align or yaw_deg is None:
        return None
    tcp = client.get_tcp()
    current = tcp.j4 if tcp is not None else None
    return j4_for_face_align(
        yaw_deg, current_j4_deg=current, x=x, y=y,
    )


def resolve_place_j4(
    client: Mt4Client,
    calib: Calibration,
    *,
    axis_align: bool = True,
    x: float | None = None,
    y: float | None = None,
) -> float | None:
    """World-frame J4 that lands the held cube square to the X/Y axes.

    A gripped cube's orientation relative to the jaws is fixed at pick time,
    so driving J4 to 0° (mod 90°, closest to the current wrist) squares
    whatever face is held to the world axes — assumes ``j4zero``.

    Defaults on unconditionally (validated safe on hardware): even for a
    pick that wasn't face-aligned, squaring the wrist costs nothing worse
    than the unaligned yaw it would otherwise land at.
    """
    if not axis_align:
        return None
    return resolve_pick_j4(client, calib, 0.0, face_align=True, x=x, y=y)


def pick(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    *,
    yaw_deg: float | None = None,
    face_align: bool | None = None,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> dict[str, object]:
    """Grip a cube at robot-frame (x, y): open, descend, close, lift.

    When ``yaw_deg`` is set (robot-frame cube-edge angle from detection) and
    face-align is enabled, world-frame J4 is commanded so the jaws meet a
    face rather than a corner. Face-align defaults on and assumes firmware
    ``j4zero`` (``calibrate_j4.py``): world J4 = 0 means jaws along the arm.

    ``on_waypoint``, when given, is called with a ``MoveEvent`` after every
    completed move/grip in the sequence -- see that class for the "None
    means unchanged" convention.
    """
    ensure_homed(client)
    _require_mp_reachable(x, y, "pick target")
    if face_align is None:
        face_align = bool(getattr(calib, "face_align_picks", True))
    j4 = resolve_pick_j4(
        client, calib, yaw_deg, face_align=face_align, x=x, y=y,
    )
    open_started = time.monotonic()
    client.gripper(calib.grip_open_s)
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            "open before descend", grip=calib.grip_open_s, started_at=open_started,
        ))
    _travel(client, calib, x, y, calib.safe_z, "move to safe height", j4=j4, on_waypoint=on_waypoint)
    _approach(client, calib, x, y, calib.table_z, "descend to pick height", j4=j4, on_waypoint=on_waypoint)
    close_started = time.monotonic()
    result = client.gripper(calib.grip_close_s)
    if not result.get("ok"):
        # Lift clear anyway so a failed grip doesn't leave the TCP parked
        # against the cube.
        _travel(client, calib, x, y, calib.safe_z, "lift after failed grip")
        raise Mt4ClientError(f"gripper close failed: {result}")
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            "grip", grip=calib.grip_close_s, started_at=close_started,
        ))
    _travel(client, calib, x, y, calib.safe_z, "lift after grip", on_waypoint=on_waypoint)
    out: dict[str, object] = {"ok": True, "picked_at": [x, y]}
    if j4 is not None:
        out["j4"] = round(j4, 2)
        out["yaw_deg"] = None if yaw_deg is None else round(float(yaw_deg), 2)
    return out


def pick_cube(
    client: Mt4Client,
    calib: Calibration,
    cube: CubeDetection,
    *,
    face_align: bool | None = None,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> dict[str, object]:
    """Vision pick from a ``CubeDetection`` (central entry for shuffle/MCP/etc.)."""
    if cube.x is None or cube.y is None:
        raise Mt4ClientError("pick_cube: detection has no robot XY")
    result = pick(
        client,
        calib,
        float(cube.x),
        float(cube.y),
        yaw_deg=cube.yaw_deg,
        face_align=face_align,
        on_waypoint=on_waypoint,
    )
    result["color"] = cube.color
    return result


def place(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    *,
    on_released: Callable[[], None] | None = None,
    axis_align: bool = True,
    along_arm: bool = False,
    j4: float | None = None,
    lift_after: bool = True,
    release_z: float | None = None,
    travel_z: float | None = None,
    axis_clear_mm: float | None = None,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> dict[str, object]:
    """Release the held cube at robot-frame (x, y).

    Releases slightly above pick height so the cube drops the last couple of
    mm instead of being pressed into the table. By default world-frame J4 is
    driven square to the X/Y axes during the approach (world J4 = 0 after
    ``j4zero``) so the released cube lands aligned rather than at whatever
    orientation it happened to be picked in.

    ``along_arm`` forces jaws along the arm (world J4 = 0, soft-limit
    safe) instead of the nearest 90° square to the current wrist -- needed
    after ``pick_centered``'s ±90° rotate, which otherwise leaves place at
    world ~90° (jaws across the arm).

    ``j4`` overrides both of the above with an explicit world-frame angle
    (e.g. a random landing orientation for unstack_cubes.py) -- the caller
    is responsible for keeping it within joint-J4 soft limits at (x, y)
    (see ``j4_for_face_align``).

    ``release_z`` overrides the table release height (stacking uses
    ``table_z + (level-1)*cube_height_mm``). ``travel_z`` overrides the
    transit height (defaults to ``max(safe_z, release_z)``).

    When ``axis_clear_mm`` is set (stacking), approach and depart use
    vertical-then-horizontal segments and finish with a horizontal move to
    that radius from (x, y) so later diagonals cannot clip the column.

    When ``lift_after`` is False the TCP stays at release height over the
    target (for in-place centering immediately after).

    ``on_waypoint``, when given, is called with a ``MoveEvent`` after every
    completed move/release -- see that class.
    """
    ensure_homed(client)
    _require_mp_reachable(x, y, "place target")
    if j4 is not None:
        pass
    elif along_arm:
        # Prefer world 0 (jaws along arm after j4zero), not nearest-to-current.
        j4 = j4_for_face_align(0.0, current_j4_deg=None, x=x, y=y)
    else:
        j4 = resolve_place_j4(client, calib, axis_align=axis_align, x=x, y=y)
    rz = calib.table_z + 3.0 if release_z is None else float(release_z)
    tz = max(float(calib.safe_z), rz) if travel_z is None else float(travel_z)
    tcp0 = client.get_tcp()
    if tcp0 is None:
        raise Mt4ClientError("place: could not read TCP")
    if axis_clear_mm is not None and axis_clear_mm > 0:
        travel_orthogonal(
            client, calib, float(tcp0.x), float(tcp0.y), tz,
            "stack approach height", j4=j4, on_waypoint=on_waypoint,
        )
        travel_orthogonal(
            client, calib, x, y, tz, "horizontal over place XY", j4=j4,
            on_waypoint=on_waypoint,
        )
    else:
        _travel(client, calib, x, y, tz, "move to safe height", j4=j4, on_waypoint=on_waypoint)
    _approach(client, calib, x, y, rz, "descend to release height", j4=j4, on_waypoint=on_waypoint)
    release_started = time.monotonic()
    client.gripper(calib.grip_open_s)
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            "release", grip=calib.grip_open_s, started_at=release_started,
        ))
    if on_released is not None:
        on_released()
    if lift_after:
        _travel(client, calib, x, y, tz, "lift after release", on_waypoint=on_waypoint)
        if axis_clear_mm is not None and axis_clear_mm > 0:
            clear = stack_clear_xy(
                x, y, float(tcp0.x), float(tcp0.y), float(axis_clear_mm),
            )
            if clear is not None:
                _travel(
                    client, calib, clear[0], clear[1], tz,
                    "horizontal clear of stack axis", on_waypoint=on_waypoint,
                )
    return {"ok": True, "placed_at": [x, y], "release_z": rz}


def pick_centered(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    *,
    yaw_deg: float | None = None,
    face_align: bool | None = None,
    lift_after: bool = True,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> dict[str, object]:
    """Center under TCP then take the cube (calibrate_height-style align).

    Does **not** call ``pick()`` (that lifts after the first grip and forces
    an extra descend). Sequence:

    1. Face-aligned approach, descend, grab
    2. Release in place (still at pick height)
    3. Lift to ``safe_z``
    4. Rotate J4 ±90°
    5. Lower, grab, lift — cube remains held for transport

    When ``lift_after`` is False the final lift is skipped and the arm is
    left holding the cube at grab height -- the caller then folds the lift
    into its next transit (e.g. place_on_stack's ``lift_to`` carry mq),
    saving the stop/settle at safe_z between the grip and the carry.

    ``on_waypoint``, when given, is called once per move/grip -- mirrors
    ``pick()``'s convention, including for the ±90° rotate (see
    ``_rotate_j4_90_in_place``, which reads back the actual resulting pose
    since that move is joint-space, not an (x,y,z) target).
    """
    ensure_homed(client)
    _require_mp_reachable(x, y, "pick_centered target")
    if face_align is None:
        face_align = bool(getattr(calib, "face_align_picks", True))
    # One TCP read, reused for both the face-align wrist (same result
    # resolve_pick_j4 would give) and the skip check below.
    tcp = client.get_tcp()
    current_j4 = tcp.j4 if tcp is not None else None
    if not face_align or yaw_deg is None:
        j4 = None
    else:
        j4 = j4_for_face_align(yaw_deg, current_j4_deg=current_j4, x=x, y=y)
    open_started = time.monotonic()
    client.gripper(calib.grip_open_s)
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            "align: open before descend", grip=calib.grip_open_s,
            started_at=open_started,
        ))
    # The stack loop's routed approach can deliver the arm already
    # face-aligned AND already descended (the descend folded into the transit
    # mq), or just aligned at safe height, or nothing (first cube / non-stack
    # caller). Read the fresh TCP and skip whatever is already done; any
    # missed precondition just falls back to running the move.
    def _here(z_target: float) -> bool:
        return (
            tcp is not None
            and abs(float(tcp.x) - x) < _XY_EPS_MM
            and abs(float(tcp.y) - y) < _XY_EPS_MM
            and abs(float(tcp.z) - float(z_target)) < _Z_EPS_MM
            and (j4 is None or abs(float(tcp.j4) - float(j4)) < 0.3)
        )

    if _here(calib.table_z):
        pass  # fused transit already descended us to the grab pose
    elif _here(calib.safe_z):
        _approach(
            client, calib, x, y, calib.table_z, "align: descend to grab",
            j4=j4, on_waypoint=on_waypoint,
        )
    else:
        _travel(client, calib, x, y, calib.safe_z, "align: approach", j4=j4, on_waypoint=on_waypoint)
        _approach(
            client, calib, x, y, calib.table_z, "align: descend to grab",
            j4=j4, on_waypoint=on_waypoint,
        )
    close1_started = time.monotonic()
    _check(client.gripper(calib.grip_close_s), "align: grab")
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            "align: grab", grip=calib.grip_close_s, started_at=close1_started,
        ))
    open2_started = time.monotonic()
    _check(client.gripper(calib.grip_open_s), "align: release")
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            "align: release", grip=calib.grip_open_s, started_at=open2_started,
        ))
    _travel(client, calib, x, y, calib.safe_z, "align: lift before rotate", on_waypoint=on_waypoint)
    _rotate_j4_90_in_place(client, on_waypoint=on_waypoint)
    _approach(
        client, calib, x, y, calib.table_z, "align: descend to re-grip",
        on_waypoint=on_waypoint,
    )
    close2_started = time.monotonic()
    _check(client.gripper(calib.grip_close_s), "align: grip")
    if on_waypoint is not None:
        on_waypoint(MoveEvent(
            "align: grip", grip=calib.grip_close_s, started_at=close2_started,
        ))
    if lift_after:
        _travel(client, calib, x, y, calib.safe_z, "align: lift after grip", on_waypoint=on_waypoint)
    out: dict[str, object] = {"ok": True, "picked_at": [x, y], "centered": True}
    if yaw_deg is not None:
        out["yaw_deg"] = round(float(yaw_deg), 2)
    return out


def place_here(client: Mt4Client, calib: Calibration) -> dict[str, object]:
    """Release the held cube at the current TCP xy."""
    tcp = client.get_tcp()
    return place(client, calib, tcp.x, tcp.y)


def _rotate_j4_90_in_place(
    client: Mt4Client,
    *,
    on_waypoint: Callable[[MoveEvent], None] | None = None,
) -> None:
    """Rotate J4 ±90° via ``m``, picking the direction with more soft-limit headroom."""
    started = time.monotonic()
    dj4_90 = round(90.0 * STEPS_PER_DEG[3])
    j4_min, j4_max = JOINT_SOFT_MIN_STEPS[3], JOINT_SOFT_MAX_STEPS[3]
    j4 = client.get_status().joints.get("j4", 0)
    options: list[tuple[int, int]] = []
    for dj4 in (dj4_90, -dj4_90):
        end = j4 + dj4
        if j4_min <= end <= j4_max:
            margin = min(end - j4_min, j4_max - end)
            options.append((margin, dj4))
    if not options:
        raise Mt4ClientError("center: no j4 ±90° rotation within soft limits")
    options.sort(key=lambda item: item[0], reverse=True)
    status = client.get_status()
    prev_speed = status.speed_us or (
        int(status.tcp.speed) if status.tcp is not None else JOG_SPEED_MIN_US
    )
    _check(client.set_speed(JOG_SPEED_MIN_US), "center: max speed for j4 rotate")
    try:
        last_err: object = None
        for _, dj4 in options:
            result = client.move_relative(0, 0, 0, dj4)
            if result.get("ok"):
                if on_waypoint is not None:
                    # A joint-space move (relative J4 steps), not an (x,y,z)
                    # target like every other leg here -- read back the
                    # resulting world-frame pose rather than compute it, so
                    # the reported j4 is what the arm actually reached, not
                    # what was commanded.
                    tcp = client.get_tcp()
                    if tcp is not None:
                        on_waypoint(MoveEvent(
                            "align: rotate j4 90",
                            float(tcp.x), float(tcp.y), float(tcp.z),
                            float(tcp.j4), started_at=started,
                        ))
                return
            last_err = result.get("error", result)
        raise Mt4ClientError(f"center: rotate j4 ±90° failed: {last_err}")
    finally:
        if prev_speed != JOG_SPEED_MIN_US:
            _check(client.set_speed(prev_speed), "center: restore speed")


def center_placed_cube(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    *,
    lift_before_rotate: bool = False,
) -> dict[str, object]:
    """Re-grip a placed cube after rotating J4 90° and release in place.

    Centers the cube under the TCP (corrects placement/release drag). The
    gripper closes and opens at pick height, then lifts straight up.

    Expects ``place(..., lift_after=False)`` to have left the TCP at release
    height over (x, y). The wrist is rotated in place with a relative joint
    move (`m`); commanding absolute j4+90 through ``mp`` can exceed soft
    limits because firmware sets joint_j4 = world_j4 - j1.

    When ``lift_before_rotate`` is True (first calibration placement only),
    lift to ``safe_z`` before the wrist rotation, then descend for grip/release.
    """
    ensure_homed(client)
    _require_mp_reachable(x, y, "center target")
    if lift_before_rotate:
        tcp = client.get_tcp()
        _travel(
            client, calib, tcp.x, tcp.y, calib.safe_z,
            "center: lift before rotate",
        )
    _rotate_j4_90_in_place(client)
    _approach(client, calib, x, y, calib.table_z, "center: descend to cube")
    _check(client.gripper(calib.grip_close_s), "center: grip")
    _check(client.gripper(calib.grip_open_s), "center: release")
    tcp = client.get_tcp()
    _travel(
        client, calib, tcp.x, tcp.y, calib.safe_z,
        "center: lift straight after release",
    )
    return {"ok": True, "centered_at": [x, y]}


def goto_marker(
    client: Mt4Client, calib: Calibration, x: float, y: float, *, touch: bool = False
) -> dict[str, object]:
    """Move the TCP over robot-frame (x, y) -- a calibration accuracy check:
    hover at the safe travel height by default (won't crash into the table
    even if the calibration is off), or descend to the measured table
    surface with `touch=True` for a physical go/no-go check.
    """
    ensure_homed(client)
    _travel(client, calib, x, y, calib.safe_z, "move to safe height")
    if touch:
        _approach(client, calib, x, y, calib.table_z, "descend to table")
    return {"ok": True, "moved_to": [x, y], "touched": touch}
