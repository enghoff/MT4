"""Central pose-and-yaw motion primitives on the firmware `mq` queue.

One layer, three operations -- ``pick_at``, ``place_at``, ``transfer`` -- each
taking a ``Grasp`` (x, y, yaw, and how that yaw repeats) rather than bare
coordinates. Everything above this (``pickplace``'s pick/place, shuffle, the
entity layer, MCP) composes these instead of hand-rolling leg sequences.

**Why this exists.** Motion used to be split in two. ``routed_travel`` sent a
whole column-aware route as one `mq` queue with per-leg wrist and speed, and
only stack_cubes/unstack_cubes used it. Everything else -- ``pick``, ``place``,
shuffle, MCP -- issued one blocking `mp` per segment with a stop/settle and a
serial round trip each, plus separate blocking gripper calls, so a shuffle
pick-and-place cost ~6 `mp` round trips and 2 gripper sleeps for a move the
stacking path did in two queues. Meanwhile yaw was handled five different ways
across three conventions, and no primitive took a target *pose* at all.

**What makes one queue possible.** Firmware grip stations (a queued entry with
``dwell_ms`` > 0: no motion, grip, hold until the sweep finishes plus a settle
-- see the `mq` protocol doc in firmware/mt4_jog/src/main.cpp). Without them a
grasp had to break the queue, because a leg's own ``grip`` fires at leg *start*
and sweeps while the arm moves, whereas the jaws must close after the descent
with the arm stopped.

**Leg vocabulary.** A plan is a flat ``list[Leg]``. Two ways to move the
gripper, and the difference matters:

* ``Leg.grip`` -- fires as that leg *starts*, sweeping while the arm travels.
  Free, and correct for opening the jaws on the way to a pick.
* a station (``Leg.dwell_ms`` > 0) -- no motion, arm stopped, queue held until
  the sweep completes. The only correct way to close on an object, and also
  usable as a pure barrier (``grip=0``) that waits out an in-flight sweep.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.joints import MQ_QUEUE_CAPACITY
from mt4_vision.calib import Calibration
from mt4_vision.stackpath import StackPlanner
from mt4_vision.workspace import KEEPOUT_RADIUS_MM, is_mp_reachable_xy
from mt4_vision.wrist import j4_for_face_align, j4_for_long_axis

# Mechanical settle a station waits *after* the firmware's own sweep-complete
# signal. The servo reaching its commanded S is not the jaws having seated on
# the object; this covers the difference (and gripper compliance) without the
# host modelling sweep duration at all.
DEFAULT_STATION_DWELL_MS = 150
# A barrier station: no grip of its own, minimum settle. Its whole job is the
# firmware's stage-1 wait, which holds the queue until *any* in-flight sweep
# finishes -- including one started by an earlier leg's `grip`. That is what
# lets the jaws open during transit and still be guaranteed open before the
# descent, on a transit too short to cover the sweep.
BARRIER_DWELL_MS = 1
# Max legs per move_path call: the firmware queue holds MQ_QUEUE_CAPACITY
# behind the leg currently executing.
MAX_LEGS_PER_QUEUE = MQ_QUEUE_CAPACITY + 1

_XY_EPS_MM = 0.5
_Z_EPS_MM = 0.5

# Square faces vs an elongated long axis -- see mt4_vision.wrist.
YAW_PERIOD_SQUARE = 90.0
YAW_PERIOD_LONG_AXIS = 180.0


@dataclass(frozen=True)
class Leg:
    """One queued `mq` entry: a waypoint, or a grip station when dwell_ms > 0."""

    x: float
    y: float
    z: float
    # World-frame yaw in deg, or a firmware sentinel ("wrist" / "hold" / None).
    j4: float | str | None = None
    speed_us: int = 0
    # Fires at leg START and sweeps while the arm moves (see module docstring).
    grip: int = 0
    # >0 makes this a station: no motion, grip with the arm stopped, hold.
    dwell_ms: int = 0

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def is_station(self) -> bool:
        return self.dwell_ms > 0


@dataclass(frozen=True)
class Grasp:
    """Where and how to take (or release) something.

    ``yaw_deg`` is the object's own robot-frame angle, not a wrist angle: a
    cube-face edge for a square, the long axis for something elongated.
    ``yaw_period_deg`` says which -- 90 aligns the jaws *to* a face, 180 closes
    them *across* the axis. None means "no yaw opinion"; the wrist is preserved.

    ``z`` defaults to ``calib.table_z``, which grips an object resting on the
    table (the jaws straddle its lower half). ``center`` runs the ±90° re-grip
    that squares a cube under the TCP -- see ``pickplace.pick_centered``.
    """

    x: float
    y: float
    yaw_deg: float | None = None
    yaw_period_deg: float = YAW_PERIOD_SQUARE
    z: float | None = None
    center: bool = False
    grip_open_s: int | None = None
    grip_close_s: int | None = None
    dwell_ms: int = DEFAULT_STATION_DWELL_MS

    def grasp_z(self, calib: Calibration) -> float:
        return float(calib.table_z) if self.z is None else float(self.z)

    def open_s(self, calib: Calibration) -> int:
        return int(calib.grip_open_s if self.grip_open_s is None else self.grip_open_s)

    def close_s(self, calib: Calibration) -> int:
        return int(
            calib.grip_close_s if self.grip_close_s is None else self.grip_close_s
        )


def square_place(x: float, y: float, **kwargs: object) -> Grasp:
    """Destination grasp that squares the held object to the world X/Y axes.

    ``Grasp(x, y)`` means "no yaw opinion, keep the wrist wherever it is", which
    lands a cube at whatever angle the pick left -- NOT what "put it down here"
    means. Squaring is what ``pickplace.place`` has always done by default
    (``resolve_place_j4`` = face-align to 0° on the 90° lattice, nearest the
    current wrist), and a bare ``Grasp`` silently dropped it when the MCP and
    shuffle paths moved onto ``place_at``/``transfer``. A destination -- a
    marker, a slot, a bare XY -- has no orientation of its own, so this is the
    right default for one; an object being *picked* does have one, and keeps it.
    """
    return Grasp(x, y, yaw_deg=0.0, yaw_period_deg=YAW_PERIOD_SQUARE, **kwargs)  # type: ignore[arg-type]


def check(result: dict[str, object], step: str) -> dict[str, object]:
    """move_to/home/gripper report failure via {"ok": False, ...}, not an
    exception -- callers must not chain past a step that never happened, so
    turn a failed result into one."""
    if not result.get("ok"):
        raise Mt4ClientError(f"{step} failed: {result.get('error', result)}")
    return result


def home_arm(client: Mt4Client) -> None:
    """Run firmware homing regardless of the session homed flag."""
    check(client.home(), "home")


def ensure_homed(client: Mt4Client) -> None:
    status = client.get_status()
    if not status.homed:
        home_arm(client)


def require_mp_reachable(x: float, y: float, step: str) -> None:
    if not is_mp_reachable_xy(x, y):
        raise Mt4ClientError(
            f"{step}: ({x:.1f}, {y:.1f}) is inside the {KEEPOUT_RADIUS_MM:.0f}mm "
            f"J1 keep-out zone (mp cannot move there)"
        )


def resolve_yaw_j4(
    client: Mt4Client,
    calib: Calibration,
    yaw_deg: float | None,
    *,
    period_deg: float = YAW_PERIOD_SQUARE,
    x: float | None = None,
    y: float | None = None,
    current_j4: float | None = None,
) -> float | None:
    """World-frame J4 for a grasp yaw, reading the current wrist to minimize
    travel. The single answer to "what wrist angle does this grasp want".

    None means "no opinion, preserve the wrist" (``yaw_deg is None``), which
    the leg builders map to the firmware wrist sentinel. An *infeasible* 180°
    grasp is a different thing and must not collapse into that same None: it
    raises, so nothing can silently grasp a pen along its shaft instead of
    across it. That raise is unreachable at this arm's stock soft limits (the
    joint-J4 window is wider than the 180° period) and exists for a rig whose
    ``MT4_JOINT_SOFT_*`` window is narrower -- see ``wrist._j4_on_lattice``.
    Call ``wrist.j4_for_long_axis`` directly (pure geometry, returns None) to
    pre-check without raising.

    ``current_j4`` supplies the wrist instead of a live TCP read, for a caller
    planning legs the arm has not executed yet. The live read is the *wrong*
    wrist there -- it is where the arm is now, not where the plan will have left
    it -- so a chained plan would minimize travel from a stale angle, and pay a
    serial round trip per stage to do it.
    """
    if yaw_deg is None:
        return None
    if current_j4 is not None:
        current: float | None = float(current_j4)
    else:
        tcp = client.get_tcp()
        current = tcp.j4 if tcp is not None else None
    if period_deg == YAW_PERIOD_LONG_AXIS:
        j4 = j4_for_long_axis(yaw_deg, current_j4_deg=current, x=x, y=y)
        if j4 is None:
            raise Mt4ClientError(
                f"no feasible wrist angle for a long-axis grasp at "
                f"({0.0 if x is None else x:.1f}, {0.0 if y is None else y:.1f}) "
                f"with axis {yaw_deg:.1f}deg -- joint J4 would leave its soft "
                f"limits at that bearing"
            )
        return j4
    return j4_for_face_align(yaw_deg, current_j4_deg=current, x=x, y=y)


# -- leg planning ---------------------------------------------------------


def plan_route_legs(
    calib: Calibration,
    planner: StackPlanner,
    start: tuple[float, float, float],
    x: float,
    y: float,
    z: float,
    levels: int,
    *,
    j4: float | str | None = None,
    then: list[tuple[float, float, float]] | None = None,
    final_j4: float | None = None,
    lift_to: float | None = None,
    descend: tuple[float, float, float] | None = None,
    step: str = "transit",
) -> list[Leg]:
    """Legs from ``start`` to (x, y, z) along a StackPlanner route.

    The leg composition ``routed_travel`` has always produced, lifted out so
    plans can be *composed* (a pick's legs plus a place's legs in one queue)
    instead of each being sent on its own. ``routed_travel`` now calls this and
    sends the result, so there is still exactly one implementation of the
    routing and wrist-assignment rules; tests/test_routed_travel.py pins them.

    ``lift_to`` prepends a pure vertical lift-off from ``start`` (e.g. rising
    off a grab with a cube held) BEFORE the column-aware route plans from
    there. It stays vertical and holds "wrist": a diagonal lift-off from grab
    height could clip a neighbouring cube, which route()'s model does not cover
    (it guards the column, not table neighbours).

    ``then`` appends waypoints after the route and ``descend`` appends a final
    slow leg; neither is column-checked -- the caller guarantees them (a
    horizontal hop at hover height where the fingertips already clear, a pure
    vertical drop straight below the arrival XY).

    ``final_j4`` sets the wrist from the ARRIVING leg onward (arrival plus any
    then/descend legs) while earlier transit legs hold ``j4``/wrist, so the arm
    is oriented before any hop or drop rather than turning into it.

    Returns [] when the move is already satisfied (same guard routed_travel
    has always had); raises when no stack-safe route exists.
    """
    cur = (float(start[0]), float(start[1]), float(start[2]))
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
        return []
    wps = planner.route(a, (x, y, z), levels)
    if wps is None:
        raise Mt4ClientError(
            f"{step}: no stack-safe route from "
            f"({a[0]:.0f},{a[1]:.0f},{a[2]:.0f}) to ({x:.0f},{y:.0f},{z:.0f})"
        )
    all_wps = lift_wps + wps + tail + descend_wps
    transit_j4: float | str = j4 if j4 is not None else "wrist"
    arrival_j4: float | str = (
        float(final_j4) if final_j4 is not None else transit_j4
    )
    n_lift = len(lift_wps)
    n_route_transit = len(wps) - 1
    j4_list: list[float | str] = (
        ["wrist"] * n_lift
        + [transit_j4] * n_route_transit
        + [arrival_j4] * (len(all_wps) - n_lift - n_route_transit)
    )
    speeds = [int(calib.travel_speed_us)] * len(all_wps)
    if descend_wps:
        speeds[-1] = int(calib.approach_speed_us)
    return [
        Leg(wp[0], wp[1], wp[2], j4=j4_list[i], speed_us=speeds[i])
        for i, wp in enumerate(all_wps)
    ]


def station(
    xyz: tuple[float, float, float],
    grip: int,
    *,
    j4: float | str | None = None,
    dwell_ms: int = DEFAULT_STATION_DWELL_MS,
) -> Leg:
    """A grip station at ``xyz`` -- no motion, arm stopped, queue held."""
    return Leg(xyz[0], xyz[1], xyz[2], j4=j4, grip=int(grip), dwell_ms=int(dwell_ms))


def barrier(xyz: tuple[float, float, float], *, j4: float | str | None = None) -> Leg:
    """A station that grips nothing: waits out any in-flight gripper sweep.

    Cheap insurance before a descent whose transit may have been shorter than
    the jaws-opening sweep it overlapped -- without it, a short-transit pick
    descends with the jaws still closing and shoves the object.
    """
    return station(xyz, 0, j4=j4, dwell_ms=BARRIER_DWELL_MS)


# -- sending --------------------------------------------------------------


def split_legs(legs: list[Leg], *, limit: int = MAX_LEGS_PER_QUEUE) -> list[list[Leg]]:
    """Chunk ``legs`` into queues of at most ``limit``, preferring station cuts.

    A chunk boundary is a real stop: move_path returns only once the firmware
    queue drains, and the next call cold-starts. Cutting right after a station
    is therefore free -- the arm is already stopped there -- while cutting
    mid-transit costs a stop/settle/reaccel. So within each window we take the
    last station boundary if there is one, and only fall back to a hard cut at
    the limit when there is not.
    """
    chunks: list[list[Leg]] = []
    rest = list(legs)
    while len(rest) > limit:
        window = rest[:limit]
        cut = 0
        for i in range(len(window) - 1, -1, -1):
            if window[i].is_station:
                cut = i + 1
                break
        if cut == 0 or cut == len(window):
            cut = limit
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


def send_legs(
    client: Mt4Client,
    legs: list[Leg],
    *,
    step: str = "motion",
    on_queue: Callable[[list[Leg]], None] | None = None,
) -> int:
    """Send ``legs`` as as few `mq` queues as the firmware depth allows.

    Returns the number of move_path calls made. Uniform j4/speed collapse to
    scalars so the wire form of an ordinary transit is byte-identical to what
    it was before this layer existed; only genuinely mixed legs go out as
    lists. Raises on the first failing queue rather than sending the rest,
    since later legs were planned against poses the arm never reached.
    """
    if not legs:
        return 0
    chunks = split_legs(legs)
    for chunk in chunks:
        wps = [leg.xyz for leg in chunk]
        j4s: list[float | str | None] = [leg.j4 for leg in chunk]
        speeds = [leg.speed_us for leg in chunk]
        grips = [leg.grip for leg in chunk]
        dwells = [leg.dwell_ms for leg in chunk]
        kwargs: dict[str, object] = {
            "j4": j4s[0] if all(v == j4s[0] for v in j4s) else j4s,
            "speed_us": speeds[0] if all(s == speeds[0] for s in speeds) else speeds,
        }
        # Only mention grip/dwell when a leg actually uses them, so a plain
        # transit's call is byte-identical to what routed_travel sent before
        # this layer existed -- including for duck-typed clients (test fakes,
        # mt4_pi) that predate the station fields. When they ARE used they go
        # as lists: a scalar grip means "first leg only" to move_path, which is
        # not what a plan means.
        if any(grips):
            kwargs["grip"] = grips
        if any(dwells):
            kwargs["dwell_ms"] = dwells
        result = client.move_path(wps, **kwargs)
        if not result.get("ok"):
            raise Mt4ClientError(
                f"{step} failed: {result.get('error', result)}"
            )
        if on_queue is not None:
            on_queue(chunk)
    return len(chunks)


# -- primitives -----------------------------------------------------------


def _planner_for(calib: Calibration, planner: StackPlanner | None) -> StackPlanner:
    return StackPlanner.free_space(calib) if planner is None else planner


def plan_pick_legs(
    client: Mt4Client,
    calib: Calibration,
    g: Grasp,
    *,
    planner: StackPlanner | None = None,
    levels: int = 0,
    start: tuple[float, float, float] | None = None,
    lift_after: bool = True,
    hover_z: float | None = None,
    current_j4: float | None = None,
) -> tuple[list[Leg], float | None]:
    """Legs that take the object at ``g``, plus the wrist angle used.

        transit (jaws opening en route) -> barrier -> slow descend
        -> close station -> lift

    The open rides on the first transit leg's ``grip`` so the sweep overlaps
    travel for free, and the barrier then guarantees it finished before the
    descent. Compare the old sequence: a blocking gripper call, a blocking
    travel `mp`, a blocking descend `mp`, a blocking gripper call, a blocking
    lift `mp`.
    """
    plan = _planner_for(calib, planner)
    gz = g.grasp_z(calib)
    tz = float(calib.safe_z) if hover_z is None else float(hover_z)
    j4 = resolve_yaw_j4(
        client, calib, g.yaw_deg, period_deg=g.yaw_period_deg, x=g.x, y=g.y,
        current_j4=current_j4,
    )
    if start is None:
        tcp = client.get_tcp()
        if tcp is None:
            raise Mt4ClientError("pick: could not read TCP")
        start = (float(tcp.x), float(tcp.y), float(tcp.z))

    legs = plan_route_legs(
        calib, plan, start, g.x, g.y, tz, levels,
        final_j4=j4, descend=(g.x, g.y, gz), step="pick approach",
    )
    if legs:
        # Open the jaws on the very first leg: the sweep then overlaps the
        # whole transit instead of serializing ahead of it.
        legs[0] = replace(legs[0], grip=g.open_s(calib))
        legs.insert(-1, barrier((g.x, g.y, tz), j4=j4))
    else:
        # Already at the grasp pose (a fused transit descended us here, or a
        # caller placed the arm). Still open, still wait for it.
        legs = [
            station((g.x, g.y, gz), g.open_s(calib), j4=j4),
        ]
    legs.append(
        station((g.x, g.y, gz), g.close_s(calib), j4=j4, dwell_ms=g.dwell_ms)
    )
    if lift_after:
        legs.append(
            Leg(g.x, g.y, tz, j4=j4, speed_us=int(calib.travel_speed_us))
        )
    return legs, j4


def plan_place_legs(
    client: Mt4Client,
    calib: Calibration,
    g: Grasp,
    *,
    planner: StackPlanner | None = None,
    levels: int = 0,
    start: tuple[float, float, float] | None = None,
    lift_from: float | None = None,
    lift_after: bool = True,
    release_z: float | None = None,
    travel_z: float | None = None,
    j4: float | None = None,
    current_j4: float | None = None,
) -> tuple[list[Leg], float | None]:
    """Legs that release the held object at ``g``, plus the wrist angle used.

        carry (optionally lifting off first) -> slow descend -> open station
        -> lift clear

    ``release_z`` defaults to 3mm above the grasp height so the object drops
    the last couple of mm instead of being pressed into the table.
    ``lift_from`` folds a vertical lift-off into the carry (what a transfer
    uses so the post-grip lift and the carry are one queue).

    ``j4`` is an explicit *wrist* angle that bypasses the resolver entirely --
    for callers that already solved the lattice themselves and want that exact
    world yaw (unstack_cubes' random landing orientation). Unlike
    ``Grasp.yaw_deg``, which is the object's angle, this is not folded or
    re-fitted, so the caller owns keeping it inside the joint soft limits at
    (x, y).
    """
    plan = _planner_for(calib, planner)
    rz = g.grasp_z(calib) + 3.0 if release_z is None else float(release_z)
    tz = max(float(calib.safe_z), rz) if travel_z is None else float(travel_z)
    if j4 is None:
        j4 = resolve_yaw_j4(
            client, calib, g.yaw_deg, period_deg=g.yaw_period_deg, x=g.x, y=g.y,
            current_j4=current_j4,
        )
    if start is None:
        tcp = client.get_tcp()
        if tcp is None:
            raise Mt4ClientError("place: could not read TCP")
        start = (float(tcp.x), float(tcp.y), float(tcp.z))

    legs = plan_route_legs(
        calib, plan, start, g.x, g.y, tz, levels,
        final_j4=j4, lift_to=lift_from, descend=(g.x, g.y, rz),
        step="place carry",
    )
    if not legs:
        legs = []
    legs.append(
        station((g.x, g.y, rz), g.open_s(calib), j4=j4, dwell_ms=g.dwell_ms)
    )
    if lift_after:
        legs.append(
            Leg(g.x, g.y, tz, j4=j4, speed_us=int(calib.travel_speed_us))
        )
    return legs, j4


def pick_at(
    client: Mt4Client,
    calib: Calibration,
    g: Grasp,
    *,
    planner: StackPlanner | None = None,
    levels: int = 0,
    lift_after: bool = True,
) -> dict[str, object]:
    """Take the object at ``g``. One queue (chunked only if the route is long).

    ``g.center`` runs the ±90° re-grip that squares a cube under the TCP. That
    step cannot be queued -- it is a relative joint (`m`) move on purpose,
    since an absolute j4+90 through `mp` can exceed soft limits (firmware sets
    joint_j4 = world_j4 - j1) -- so it costs extra round trips by nature. See
    ``pickplace.pick_centered``, which this defers to.
    """
    # The one backward dependency in this module, and deferred so the import
    # graph stays wrist -> motion -> pickplace. pick_centered is a legacy
    # multi-`mp` sequence that cannot be expressed as legs (see above); moving
    # it down here would buy nothing but churn.
    from mt4_vision.pickplace import pick_centered

    ensure_homed(client)
    require_mp_reachable(g.x, g.y, "pick target")
    if g.center:
        return pick_centered(
            client, calib, g.x, g.y,
            yaw_deg=g.yaw_deg, lift_after=lift_after,
        )
    legs, j4 = plan_pick_legs(
        client, calib, g, planner=planner, levels=levels, lift_after=lift_after
    )
    queues = send_legs(client, legs, step="pick")
    out: dict[str, object] = {
        "ok": True, "picked_at": [g.x, g.y], "queues": queues,
    }
    if j4 is not None:
        out["j4"] = round(j4, 2)
        out["yaw_deg"] = None if g.yaw_deg is None else round(float(g.yaw_deg), 2)
    return out


def place_at(
    client: Mt4Client,
    calib: Calibration,
    g: Grasp,
    *,
    planner: StackPlanner | None = None,
    levels: int = 0,
    lift_after: bool = True,
    release_z: float | None = None,
    travel_z: float | None = None,
    j4: float | None = None,
    on_released: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Release the held object at ``g``. One queue.

    ``on_released`` fires after the queue completes, not at the instant of
    release -- the release is inside a firmware station, so there is no
    host-side moment to hook. Callers that need something to happen while the
    arm still hovers there should pass ``lift_after=False``.
    """
    ensure_homed(client)
    require_mp_reachable(g.x, g.y, "place target")
    legs, j4 = plan_place_legs(
        client, calib, g, planner=planner, levels=levels,
        lift_after=lift_after, release_z=release_z, travel_z=travel_z, j4=j4,
    )
    queues = send_legs(client, legs, step="place")
    if on_released is not None:
        on_released()
    rz = g.grasp_z(calib) + 3.0 if release_z is None else float(release_z)
    out: dict[str, object] = {
        "ok": True, "placed_at": [g.x, g.y], "release_z": rz, "queues": queues,
    }
    if j4 is not None:
        out["j4"] = round(j4, 2)
    return out


def plan_transfer_legs(
    client: Mt4Client,
    calib: Calibration,
    src: Grasp,
    dst: Grasp,
    *,
    planner: StackPlanner | None = None,
    levels: int = 0,
    start: tuple[float, float, float] | None = None,
    release_z: float | None = None,
    travel_z: float | None = None,
    current_j4: float | None = None,
) -> tuple[list[Leg], tuple[float, float, float], float | None, float | None]:
    """Composable legs for one ``src`` → ``dst`` transfer, plus the end pose.

    Same pick+place fusion ``transfer`` sends, lifted out so a caller can
    concatenate several transfers into one ``send_legs`` (shuffle lookahead)
    without a stop/settle between them. ``src.center`` cannot be planned this
    way (the ±90° re-grip is an unqueueable relative ``m``) -- raise and let the
    caller fall back to ``transfer``.

    Chaining needs BOTH carried-forward states, not just the pose: pass the
    previous plan's end pose as ``start`` and its ending wrist (the returned
    ``place_j4``, falling back to ``pick_j4``) as ``current_j4``. Otherwise each
    stage resolves its wrist against a live TCP read -- the angle the arm is at
    *now*, before any of the queue has run -- which both costs a serial round
    trip per stage and minimizes travel from the wrong angle. The place's wrist
    is likewise resolved against the pick's, not against the live TCP.
    """
    if src.center:
        raise Mt4ClientError(
            "plan_transfer_legs: centered grasp cannot be queued; use transfer()"
        )
    require_mp_reachable(src.x, src.y, "pick target")
    require_mp_reachable(dst.x, dst.y, "place target")
    if start is None:
        tcp = client.get_tcp()
        if tcp is None:
            raise Mt4ClientError("transfer: could not read TCP")
        start = (float(tcp.x), float(tcp.y), float(tcp.z))

    pick_legs, pick_j4 = plan_pick_legs(
        client, calib, src, planner=planner, levels=levels,
        start=start, lift_after=False, current_j4=current_j4,
    )
    # The carry's own lift-off replaces the pick's lift leg: one vertical rise,
    # planned by the router, with no stop at the top.
    grab_xyz = (src.x, src.y, src.grasp_z(calib))
    place_legs, place_j4 = plan_place_legs(
        client, calib, dst, planner=planner, levels=levels,
        start=grab_xyz, lift_from=float(calib.safe_z),
        release_z=release_z, travel_z=travel_z,
        current_j4=pick_j4 if pick_j4 is not None else current_j4,
    )
    legs = pick_legs + place_legs
    if not legs:
        end = start
    else:
        end = legs[-1].xyz
    return legs, end, pick_j4, place_j4


def transfer(
    client: Mt4Client,
    calib: Calibration,
    src: Grasp,
    dst: Grasp,
    *,
    planner: StackPlanner | None = None,
    levels: int = 0,
    on_grasped: Callable[[], None] | None = None,
    on_released: Callable[[], None] | None = None,
    release_z: float | None = None,
    travel_z: float | None = None,
) -> dict[str, object]:
    """Move the object at ``src`` to ``dst`` as one planned operation.

    The pick's lift and the carry are the same leg (``lift_from``), so there is
    no stop between the grip and the transit -- the fusion stack_cubes.py used
    to hand-roll. Both grips are firmware stations, so the whole thing is one
    queue unless the route is long enough to need chunking, against ~7 blocking
    `mp` moves and 2 gripper sleeps before.

    ``src.center`` forces the split form: the ±90° re-grip cannot be queued.
    """
    ensure_homed(client)
    require_mp_reachable(src.x, src.y, "pick target")
    require_mp_reachable(dst.x, dst.y, "place target")

    if src.center:
        # Centering breaks the queue anyway, so run the two halves as they are.
        picked = pick_at(
            client, calib, src, planner=planner, levels=levels, lift_after=False
        )
        if on_grasped is not None:
            on_grasped()
        placed = place_at(
            client, calib, dst, planner=planner, levels=levels,
            release_z=release_z, travel_z=travel_z, on_released=on_released,
        )
        return {
            "ok": True,
            "picked_at": picked.get("picked_at"),
            "placed_at": placed.get("placed_at"),
            "queues": int(picked.get("queues", 0)) + int(placed.get("queues", 0)),
            "centered": True,
        }

    legs, _end, pick_j4, place_j4 = plan_transfer_legs(
        client, calib, src, dst, planner=planner, levels=levels,
        release_z=release_z, travel_z=travel_z,
    )
    queues = send_legs(client, legs, step="transfer")
    if on_grasped is not None:
        on_grasped()
    if on_released is not None:
        on_released()
    out: dict[str, object] = {
        "ok": True,
        "picked_at": [src.x, src.y],
        "placed_at": [dst.x, dst.y],
        "queues": queues,
    }
    if pick_j4 is not None:
        out["pick_j4"] = round(pick_j4, 2)
    if place_j4 is not None:
        out["place_j4"] = round(place_j4, 2)
    return out


# -- grasp verification ---------------------------------------------------
#
# Post-move "did the physical action happen?" checks. Callers supply an
# iterable of (color, x, y) detections -- scene.cubes for shuffle,
# scene.raw_cubes for stack's miss check, or an occupancy probe for thin
# objects later. Keeping this scene-agnostic is what lets every consumer
# share one helper without motion importing the camera stack.

VERIFY_ORIGIN_RADIUS_MM = 30.0
VERIFY_PLACED_RADIUS_MM = 35.0

# (color, x_mm, y_mm) -- whatever the caller can see after the move.
DetectionXY = tuple[str, float, float]


def detection_near(
    detections: Iterable[DetectionXY],
    *,
    color: str,
    x: float,
    y: float,
    radius_mm: float,
) -> tuple[float, float] | None:
    """First same-color detection within ``radius_mm`` of (x, y), or None."""
    for c, dx, dy in detections:
        if c == color and math.hypot(dx - x, dy - y) <= radius_mm:
            return (dx, dy)
    return None


def grasp_failed_at(
    detections: Iterable[DetectionXY],
    *,
    pick_x: float,
    pick_y: float,
    pick_color: str,
    radius_mm: float = VERIFY_ORIGIN_RADIUS_MM,
) -> tuple[float, float] | None:
    """XY of a same-color detection still at the pick origin, or None.

    A miss signature: the grab shoved or missed the target and it is still
    sitting near where we aimed. Used by stack_cubes' ``pick_missed``.
    """
    return detection_near(
        detections, color=pick_color, x=pick_x, y=pick_y, radius_mm=radius_mm,
    )


def verify_pick_place(
    detections: Iterable[DetectionXY],
    *,
    pick_x: float,
    pick_y: float,
    pick_color: str,
    place_x: float,
    place_y: float,
    origin_radius_mm: float = VERIFY_ORIGIN_RADIUS_MM,
    placed_radius_mm: float = VERIFY_PLACED_RADIUS_MM,
) -> str:
    """Judge an atomic pick+place from post-move detections.

    Returns ``placed``, ``grasp_failed``, or ``lost``. Occupancy-based thin-
    object checks can feed the same verdicts by synthesizing a one-entry
    detection list from a lift-height probe.
    """
    if detection_near(
        detections, color=pick_color, x=place_x, y=place_y,
        radius_mm=placed_radius_mm,
    ) is not None:
        return "placed"
    if grasp_failed_at(
        detections, pick_x=pick_x, pick_y=pick_y, pick_color=pick_color,
        radius_mm=origin_radius_mm,
    ) is not None:
        return "grasp_failed"
    return "lost"
