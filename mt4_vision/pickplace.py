"""Pick and place sequences for cubes on the calibrated work surface.

The queued-motion core moved to ``mt4_vision.motion``: ``pick``/``place`` here
are thin wrappers over ``motion.pick_at``/``motion.place_at`` (same signatures,
so every existing caller is untouched), and ``routed_travel`` now sends legs
built by ``motion.plan_route_legs`` rather than composing them itself. What
stays here is the cube-specific and legacy-sequence material: the camera park,
``pick_centered``'s ±90° re-grip, ``center_placed_cube``, and the
``CubeDetection``-shaped entry points.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.joints import JOINT_SOFT_MAX_STEPS, JOINT_SOFT_MIN_STEPS, JOG_SPEED_MIN_US
from mt4_jog.kinematics import STEPS_PER_DEG
from mt4_vision.calib import Calibration
from mt4_vision.detect import CubeDetection
from mt4_vision import motion
from mt4_vision.motion import (  # noqa: F401 -- home_arm is a re-export (see below)
    Grasp,
    YAW_PERIOD_SQUARE,
    ensure_homed,
    home_arm,
    plan_route_legs,
    require_mp_reachable,
    resolve_yaw_j4,
    send_legs,
)
from mt4_vision.motion import check as _check
from mt4_vision.stackpath import StackPlanner
from mt4_vision.workspace import is_mp_reachable_xy

# The wrist-angle lattice lives in mt4_vision.wrist (pure geometry, so the
# motion primitives can build on it without importing this module). Re-exported
# here, along with the motion-layer names above, because callers and tests have
# always reached for them via pickplace.
from mt4_vision.wrist import (  # noqa: F401
    fold_square_yaw_deg,
    fold_yaw_deg,
    j4_for_face_align,
    j4_for_long_axis,
    j4_preserve_wrist,
)


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


def _travel(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    z: float,
    step: str,
    *,
    j4: float | str | None = None,
) -> dict[str, object]:
    """Horizontal or lift move at safe travel speed (firmware mp ramp active)."""
    return _check(
        client.move_to(
            x, y, z, j4=_resolve_travel_j4(j4),
            speed_us=calib.travel_speed_us,
        ),
        step,
    )


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
        _check(
            client.move_path(
                [(float(tcp.x), float(tcp.y), z), (x, y, z)],
                j4=_resolve_travel_j4(j4),
                speed_us=calib.travel_speed_us,
            ),
            step,
        )
        return
    if not same_z:
        _travel(
            client, calib, float(tcp.x), float(tcp.y), z,
            f"{step}: vertical", j4=j4,
        )
    else:
        _travel(client, calib, x, y, z, f"{step}: horizontal", j4=j4)


def _approach(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    z: float,
    step: str,
    *,
    j4: float | str | None = None,
) -> dict[str, object]:
    """Slow final descent near the table (firmware ramp off)."""
    return _check(
        client.move_to(
            x, y, z, j4=_resolve_travel_j4(j4),
            speed_us=calib.approach_speed_us,
        ),
        step,
    )


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

    The leg composition itself now lives in ``motion.plan_route_legs`` so a
    pick's and a place's legs can be planned together and sent as one queue;
    this reads the TCP and sends. Behaviour is unchanged
    (tests/test_routed_travel.py pins it).
    """
    tcp = client.get_tcp()
    if tcp is None:
        raise Mt4ClientError(f"{step}: could not read TCP")
    legs = plan_route_legs(
        calib, planner, (float(tcp.x), float(tcp.y), float(tcp.z)),
        x, y, z, levels,
        j4=j4, then=then, final_j4=final_j4, lift_to=lift_to,
        descend=descend, step=step,
    )
    send_legs(client, legs, step=step)


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
    if not face_align:
        return None
    return resolve_yaw_j4(client, calib, yaw_deg, period_deg=90.0, x=x, y=y)


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
) -> dict[str, object]:
    """Grip a cube at robot-frame (x, y): open, descend, close, lift.

    When ``yaw_deg`` is set (robot-frame cube-edge angle from detection) and
    face-align is enabled, world-frame J4 is commanded so the jaws meet a
    face rather than a corner. Face-align defaults on and assumes firmware
    ``j4zero`` (``calibrate_j4.py``): world J4 = 0 means jaws along the arm.

    A cube-shaped wrapper over ``motion.pick_at``: the sequence is now one
    queued `mq` path with a firmware grip station instead of two blocking
    gripper calls and three blocking `mp` moves. Same signature, same return
    keys.
    """
    if face_align is None:
        face_align = bool(getattr(calib, "face_align_picks", True))
    return motion.pick_at(
        client,
        calib,
        Grasp(
            x, y,
            yaw_deg=yaw_deg if face_align else None,
            yaw_period_deg=YAW_PERIOD_SQUARE,
        ),
    )


def pick_cube(
    client: Mt4Client,
    calib: Calibration,
    cube: CubeDetection,
    *,
    face_align: bool | None = None,
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

    The ordinary path is a cube-shaped wrapper over ``motion.place_at`` -- one
    queued `mq` path with a firmware release station. ``axis_clear_mm`` keeps
    the older per-segment orthogonal route: it is the stacking approach, whose
    vertical-then-horizontal legs and stack-axis clear hop are not expressible
    as a routed plan, and migrating the column path is deliberately staged
    behind proving the queued path on the shuffle loop first.
    """
    if j4 is not None:
        wrist: float | None = float(j4)
    elif along_arm:
        # Prefer world 0 (jaws along arm after j4zero), not nearest-to-current.
        wrist = j4_for_face_align(0.0, current_j4_deg=None, x=x, y=y)
    else:
        wrist = resolve_place_j4(client, calib, axis_align=axis_align, x=x, y=y)

    if axis_clear_mm is not None and axis_clear_mm > 0:
        ensure_homed(client)
        require_mp_reachable(x, y, "place target")
        rz = calib.table_z + 3.0 if release_z is None else float(release_z)
        tz = max(float(calib.safe_z), rz) if travel_z is None else float(travel_z)
        tcp0 = client.get_tcp()
        if tcp0 is None:
            raise Mt4ClientError("place: could not read TCP")
        travel_orthogonal(
            client, calib, float(tcp0.x), float(tcp0.y), tz,
            "stack approach height", j4=wrist,
        )
        travel_orthogonal(
            client, calib, x, y, tz, "horizontal over place XY", j4=wrist,
        )
        _approach(client, calib, x, y, rz, "descend to release height", j4=wrist)
        client.gripper(calib.grip_open_s)
        if on_released is not None:
            on_released()
        if lift_after:
            _travel(client, calib, x, y, tz, "lift after release")
            clear = stack_clear_xy(
                x, y, float(tcp0.x), float(tcp0.y), float(axis_clear_mm),
            )
            if clear is not None:
                _travel(
                    client, calib, clear[0], clear[1], tz,
                    "horizontal clear of stack axis",
                )
        return {"ok": True, "placed_at": [x, y], "release_z": rz}

    return motion.place_at(
        client,
        calib,
        Grasp(x, y),
        j4=wrist,
        lift_after=lift_after,
        release_z=release_z,
        travel_z=travel_z,
        on_released=on_released,
    )


def pick_centered(
    client: Mt4Client,
    calib: Calibration,
    x: float,
    y: float,
    *,
    yaw_deg: float | None = None,
    face_align: bool | None = None,
    lift_after: bool = True,
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
    """
    ensure_homed(client)
    require_mp_reachable(x, y, "pick_centered target")
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
    client.gripper(calib.grip_open_s)
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
        _approach(client, calib, x, y, calib.table_z, "align: descend to grab", j4=j4)
    else:
        _travel(client, calib, x, y, calib.safe_z, "align: approach", j4=j4)
        _approach(client, calib, x, y, calib.table_z, "align: descend to grab", j4=j4)
    _check(client.gripper(calib.grip_close_s), "align: grab")
    _check(client.gripper(calib.grip_open_s), "align: release")
    _travel(client, calib, x, y, calib.safe_z, "align: lift before rotate")
    _rotate_j4_90_in_place(client)
    _approach(client, calib, x, y, calib.table_z, "align: descend to re-grip")
    _check(client.gripper(calib.grip_close_s), "align: grip")
    if lift_after:
        _travel(client, calib, x, y, calib.safe_z, "align: lift after grip")
    out: dict[str, object] = {"ok": True, "picked_at": [x, y], "centered": True}
    if yaw_deg is not None:
        out["yaw_deg"] = round(float(yaw_deg), 2)
    return out


def place_here(client: Mt4Client, calib: Calibration) -> dict[str, object]:
    """Release the held cube at the current TCP xy."""
    tcp = client.get_tcp()
    return place(client, calib, tcp.x, tcp.y)


def _rotate_j4_90_in_place(client: Mt4Client) -> None:
    """Rotate J4 ±90° via ``m``, picking the direction with more soft-limit headroom."""
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
    require_mp_reachable(x, y, "center target")
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
