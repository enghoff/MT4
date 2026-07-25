#!/usr/bin/env python3
"""Build a cube stack on a calibrated ArUco marker.

The stack site is a marker id passed on the CLI (required -- no default).
Any cubes within SITE_CLEAR_MM of that marker are nudged aside along the
marker→cube direction to CLEAR_PARK_MM (keep-clear + margin) first, preferring
landings that stay inside the pick hull and out of the stack camera-shadow
corridor. A full-circle angle sweep backs up the primary push direction, and
open-table free slots (or, failing those, a full annulus grid scan) are the
fallback when nothing near the site itself qualifies. Each stack cube is taken
with the calibrate_height centering
sequence (yaw-pick → release → lift → rotate J4 90° → re-grip) via
``pick_centered``, then placed at the marker's calibrated XY by dead
reckoning. Placement Z steps by ``cube_height_mm`` from the calibration;
there is no visual alignment or post-place verification.

Stack motion is planned by ``mt4_vision.stackpath.StackPlanner``: transits
route the gripper *and* forearm around the growing column, the carry lifts
diagonally to a stage point beside the stack and hops over the top at a
hover height fitted under the joint-limit z ceiling (~315mm at the marker
radii), and the retreat lifts the fingers free of the placed cube while the
ceiling allows it, switching to a slide-out perpendicular to the jaw axis
for the top level(s). That slide is what makes level 9 buildable: lifting
the fingers clear of a 9th cube needs ~324mm of TCP height, which the J3
soft limit cannot reach.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from jog import console_focused, flush_console_input, key_down
from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import DEFAULT_CALIB_PATH, CalibrationError, load_calibration
from mt4_vision.camera import FrameStream, capture_frame
from mt4_vision.detect import CubeDetection

# _travel/_approach/_check are pickplace's single-segment movers; the stack
# executor below sequences them along StackPlanner routes.
from mt4_vision.pickplace import (
    CAMERA_PARK_X,
    CAMERA_PARK_Y,
    CAMERA_PARK_Z,
    _approach,
    _check,
    _travel,
    go_camera_park,
    home_arm,
    j4_for_face_align,
    near_camera_park,
    pick,
    pick_centered,
    place,
    retreat_for_camera,
    routed_travel,
)
from mt4_vision.preview import LiveFeed, PreviewStopped
from mt4_vision.scene import Scene, capture_scene, within_pick_hull
from mt4_vision.stackpath import StackPlanner
from mt4_vision.workspace import (
    MAX_REACH_MM,
    MarkerSlot,
    dist_mm,
    is_mp_reachable_xy,
    marker_slots_from_calibration,
)

# Cubes this close to the stack marker are moved aside before building.
SITE_CLEAR_MM = 70.0
# Clear landings sit this far past the keep-clear radius so release drag /
# vision scatter can't bounce them straight back into the zone (the old
# free-slot path parked at ~70mm and re-cleared the same cube forever).
CLEAR_MARGIN_MM = 40.0
CLEAR_PARK_MM = SITE_CLEAR_MM + CLEAR_MARGIN_MM  # 110mm from marker
# Finger clearance from other cubes when parking a cleared cube.
CLEAR_SEP_MM = 45.0
# Cleared cubes must stay visible: the parked arm occludes a strip over the
# J1 keep-out, so a cube dropped near it vanishes from later scans (field
# case 2026-07-24: a clear to (134,49) at r=143 was never seen again and the
# build stalled "no reachable cube" with the cube sitting right there).
CLEAR_MIN_RADIUS_MM = 170.0
# Settle after retreat before a fresh scene capture.
CAMERA_SETTLE_S = 0.8
SITE_CLEAR_ATTEMPTS = 6
# Camera line-of-sight shadow behind the stack: raised stack tops map
# (via the 1-cube cube-top homography) to phantom table cubes further from
# the camera than the site. Ignore pick candidates in that corridor.
# Measured 2026-07-21 on marker 3, level 4: true (179,180) -> phantom
# ~(115,227) (~79mm along, ~8mm lateral). The phantom spreads laterally as
# well as along as the stack grows, not just along -- field case 2026-07-24
# marker 2 level 6 put a phantom at ~49mm lateral, past the old fixed 45mm
# cutoff, and it slipped through as a real pick candidate. Both axes now
# scale with stack height.
STACK_SHADOW_LATERAL_MIN_MM = 45.0
STACK_SHADOW_LATERAL_PER_LEVEL_MM = 10.0
STACK_SHADOW_ALONG_MIN_MM = 25.0
STACK_SHADOW_ALONG_PER_LEVEL_MM = 35.0
STACK_SHADOW_ALONG_FLOOR_MM = 90.0


def _expanding_angles_deg(step: float = 20.0) -> tuple[float, ...]:
    """0, then +-step, +-2*step, ... out to +-180 (a full-circle sweep that
    tries the straightest push first)."""
    angles = [0.0]
    a = step
    while a < 180.0:
        angles.append(a)
        angles.append(-a)
        a += step
    angles.append(180.0)
    return tuple(angles)


# Field case 2026-07-24: the old +-110 deg cap only ever tried a narrow fan
# off the marker->cube bearing. At a corner site (marker 0) that fan mostly
# fell outside the pick hull, leaving exactly one valid landing -- which the
# previously-cleared cube had just taken, stranding the next clear with
# "no reachable clear spot" even though open table space existed elsewhere
# around the site. A full-circle sweep finds those spots.
_CLEAR_ANGLES_DEG = _expanding_angles_deg()


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
    for it -- neither happens on its own inside stack_cubes.py's seconds-
    apart loop checkpoints. This thread polls at 20Hz so a normal tap is
    always caught, and calls ``request_interrupt`` immediately on the press
    edge so an in-flight ``move_to``/``gripper`` call aborts within a
    fraction of a second instead of running to completion first. Gated on
    ``console_focused`` so an H press in another window doesn't re-home the
    arm mid-stack.
    """

    def __init__(self, client: Mt4Client) -> None:
        self._client = client
        self._requested = threading.Event()
        self._h_down = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="stack-home-key", daemon=True
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


def cubes_near_site(
    scene: Scene, sx: float, sy: float, radius_mm: float = SITE_CLEAR_MM
) -> list[CubeDetection]:
    return [
        c
        for c in scene.raw_cubes
        if c.x is not None
        and c.y is not None
        and dist_mm(float(c.x), float(c.y), sx, sy) < radius_mm
    ]


def _clear_landing_ok(
    tx: float,
    ty: float,
    *,
    sx: float,
    sy: float,
    occupied: list[tuple[float, float]],
    markers: list[MarkerSlot] | None,
    behind_u: tuple[float, float] | None,
    shadow_levels: int,
) -> bool:
    """True when a clear drop should remain a later pick candidate."""
    if not is_mp_reachable_xy(tx, ty):
        return False
    r = math.hypot(tx, ty)
    if r > MAX_REACH_MM or r < CLEAR_MIN_RADIUS_MM:
        return False
    if any(dist_mm(tx, ty, ox, oy) < CLEAR_SEP_MM for ox, oy in occupied):
        return False
    # Stay inside the same hull gate pick uses, so clears are not dropped
    # as "phantoms" on the next capture.
    if markers is not None and not within_pick_hull(tx, ty, markers):
        return False
    # Avoid the camera-LOS corridor behind the site (ignored once stacked).
    if behind_u is not None and in_stack_camera_shadow(
        tx, ty, sx, sy, behind_u, stack_levels=shadow_levels,
    ):
        return False
    return True


def clear_aside_xy(
    sx: float,
    sy: float,
    cx: float,
    cy: float,
    occupied: list[tuple[float, float]],
    *,
    markers: list[MarkerSlot] | None = None,
    behind_u: tuple[float, float] | None = None,
    shadow_levels: int = 8,
) -> tuple[float, float] | None:
    """Push a cube away from the marker along marker→cube, with margin.

    Lands at ~CLEAR_PARK_MM from the site (not on a barely-outside free
    slot that vision will still read as "near site"). Tries a few angles
    and radii if the primary landing is blocked or unreachable. Prefers
    landings that stay pickable (marker hull + not in stack camera shadow).
    """
    dx, dy = cx - sx, cy - sy
    r = math.hypot(dx, dy)
    if r < 1.0:
        # Sitting on the tag: push outward along the marker's base bearing.
        dx, dy = sx, sy
        r = math.hypot(dx, dy) or 1.0
    ux, uy = dx / r, dy / r
    for dist in (CLEAR_PARK_MM, CLEAR_PARK_MM + 25.0, CLEAR_PARK_MM + 45.0):
        for angle_deg in _CLEAR_ANGLES_DEG:
            ang = math.radians(angle_deg)
            ca, sa = math.cos(ang), math.sin(ang)
            vx, vy = ux * ca - uy * sa, ux * sa + uy * ca
            tx, ty = sx + vx * dist, sy + vy * dist
            if _clear_landing_ok(
                tx, ty, sx=sx, sy=sy, occupied=occupied,
                markers=markers, behind_u=behind_u,
                shadow_levels=shadow_levels,
            ):
                return (tx, ty)
    return None


def _park_grid_candidates(step_mm: float = 40.0) -> list[tuple[float, float]]:
    """Rings of points spanning the whole reachable annulus, not just the
    8 fixed PLACEMENT_SLOTS. Those slots double as where stock cubes for the
    build get staged, so all 8 can be occupied (or, near a corner marker,
    outside the pick hull) at once -- see _CLEAR_ANGLES_DEG above for the
    matching field case. This is the fallback of last resort, so it trades
    a few dozen cheap arithmetic checks for never reporting "no clear spot"
    while open table space remains."""
    pts: list[tuple[float, float]] = []
    r = CLEAR_MIN_RADIUS_MM
    while r <= MAX_REACH_MM:
        steps = max(1, round(2 * math.pi * r / step_mm))
        for i in range(steps):
            ang = 2 * math.pi * i / steps
            pts.append((r * math.cos(ang), r * math.sin(ang)))
        r += step_mm
    return pts


_PARK_GRID = _park_grid_candidates()


def choose_park_slot(
    scene: Scene,
    sx: float,
    sy: float,
    *,
    avoid: list[tuple[float, float]] | None = None,
    markers: list[MarkerSlot] | None = None,
    behind_u: tuple[float, float] | None = None,
    shadow_levels: int = 8,
) -> tuple[float, float] | None:
    """Nearest free open-table slot well clear of the stack site.

    Prefers the fixed PLACEMENT_SLOTS (cheap, and already known to be sane
    staging spots); falls back to a full annulus grid scan when none of
    those are free.
    """
    avoid = avoid or []

    def _valid(x: float, y: float) -> bool:
        return dist_mm(x, y, sx, sy) >= CLEAR_PARK_MM and _clear_landing_ok(
            x, y, sx=sx, sy=sy, occupied=avoid,
            markers=markers, behind_u=behind_u,
            shadow_levels=shadow_levels,
        )

    candidates = [(x, y) for x, y in scene.free_slots if _valid(x, y)]
    if not candidates:
        candidates = [(x, y) for x, y in _PARK_GRID if _valid(x, y)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: dist_mm(p[0], p[1], sx, sy))


def stack_shadow_behind_unit(
    calib, sx: float, sy: float
) -> tuple[float, float] | None:
    """Unit XY vector from the stack site away from the camera.

    Derived from table vs cube-top maps at the site: mapping the site's
    table pixel through the cube-top homography shifts toward the camera;
    the opposite direction is "behind the stack" along the camera LOS.
    """
    if not calib.cube_top_homography:
        return None
    ht_inv = np.linalg.inv(np.array(calib.homography, dtype=np.float64))
    v = ht_inv @ np.array([sx, sy, 1.0])
    px, py = float(v[0] / v[2]), float(v[1] / v[2])
    cx, cy = calib.pixel_to_robot(px, py, on_cube_top=True)
    # cube-top reading of the table-site pixel sits toward the camera.
    toward_cam_x, toward_cam_y = cx - sx, cy - sy
    length = math.hypot(toward_cam_x, toward_cam_y)
    if length < 1.0:
        return None
    return (-toward_cam_x / length, -toward_cam_y / length)


def in_stack_camera_shadow(
    x: float,
    y: float,
    sx: float,
    sy: float,
    behind_u: tuple[float, float],
    *,
    stack_levels: int,
) -> bool:
    """True when (x, y) lies behind the stack along the camera LOS.

    A real cube there would be occluded by the stack; detections in this
    corridor are almost always raised stack tops mis-mapped as table cubes.
    """
    dx, dy = x - sx, y - sy
    ux, uy = behind_u
    along = dx * ux + dy * uy
    lateral = abs(dx * uy - dy * ux)
    along_max = max(
        STACK_SHADOW_ALONG_FLOOR_MM,
        stack_levels * STACK_SHADOW_ALONG_PER_LEVEL_MM,
    )
    lateral_max = max(
        STACK_SHADOW_LATERAL_MIN_MM,
        stack_levels * STACK_SHADOW_LATERAL_PER_LEVEL_MM,
    )
    return (
        along >= STACK_SHADOW_ALONG_MIN_MM
        and along <= along_max
        and lateral <= lateral_max
    )


# Off-corridor detections this close to the site mid-build mean a cube is
# lying beside / leaning against the column. Closer readings are the stack's
# own base; corridor-aligned ones are mapped side faces of the stacked cubes
# themselves (levels 1..built each cast one, at ~26mm/level along the
# camera LOS) and must never be treated as fallen -- they hold perfectly
# still as the stack grows, unlike the top phantom. A "static in corridor
# means fallen" rule was tried 2026-07-24 and false-positived on exactly
# those side faces; only the off-corridor test is trustworthy.
NEAR_SITE_MIN_MM = 25.0
CORRIDOR_ALIGN_COS = math.cos(math.radians(35.0))


def stack_integrity_issues(
    scene: Scene,
    sx: float,
    sy: float,
    behind_u: tuple[float, float] | None,
) -> list[str]:
    """Evidence that the column shed a cube: a detection beside the site
    that is not corridor-aligned (i.e. cannot be the stack's own mapped
    side faces)."""
    issues: list[str] = []
    for c in scene.raw_cubes:
        if c.x is None or c.y is None:
            continue
        dx, dy = float(c.x) - sx, float(c.y) - sy
        d = math.hypot(dx, dy)
        if not (NEAR_SITE_MIN_MM <= d < SITE_CLEAR_MM):
            continue
        aligned = (
            behind_u is not None
            and (dx * behind_u[0] + dy * behind_u[1]) / max(d, 1e-6)
            >= CORRIDOR_ALIGN_COS
        )
        if not aligned:
            issues.append(
                f"{c.color} cube {d:.0f}mm from the site at "
                f"({c.x:.0f},{c.y:.0f}) -- likely shed from the stack"
            )
    return issues


# A pick "succeeded" but a same-color cube still sits within this radius of
# the pick spot on the next scan: the grab missed (edge-of-hull vision skew
# or lost steps) and usually just shoved the cube. Pick targets require
# 45mm clearance from every other cube, so a same-color detection this
# close can only be the target itself. Missed picks correlate with lost
# steps, so recovery is a re-home before retrying.
PICK_FAIL_RADIUS_MM = 30.0
PICK_FAIL_MAX_RETRIES = 3


def pick_missed(
    scene: Scene, last_pick: tuple[str, float, float] | None
) -> tuple[float, float] | None:
    """XY of the not-actually-picked cube, or None when the pick took."""
    if last_pick is None:
        return None
    color, px, py = last_pick
    for c in scene.raw_cubes:
        if (
            c.color == color
            and c.x is not None
            and c.y is not None
            and dist_mm(float(c.x), float(c.y), px, py) <= PICK_FAIL_RADIUS_MM
        ):
            return (float(c.x), float(c.y))
    return None


def stack_candidates(
    scene: Scene,
    sx: float,
    sy: float,
    *,
    calib=None,
    stack_levels: int = 0,
) -> list[CubeDetection]:
    """Reachable pickable cubes outside the site keep-clear radius.

    When the stack already has cubes, also drop detections in the camera
    line-of-sight shadow behind the site (stack-top phantoms).
    """
    behind_u = None
    if stack_levels > 0 and calib is not None:
        behind_u = stack_shadow_behind_unit(calib, sx, sy)
    out: list[CubeDetection] = []
    for c in scene.pickable(scene.cubes):
        if dist_mm(float(c.x), float(c.y), sx, sy) < SITE_CLEAR_MM:
            continue
        if (
            behind_u is not None
            and in_stack_camera_shadow(
                float(c.x), float(c.y), sx, sy, behind_u,
                stack_levels=stack_levels,
            )
        ):
            continue
        out.append(c)
    return out


def select_stack_cube(
    cands: list[CubeDetection], current_color: str | None
) -> tuple[CubeDetection, str]:
    """Pick from the most-abundant color among reachable candidates.

    Stacking by color reads cleaner: e.g. 3 reachable red vs. 2 blue/green
    each -> take red first. ``current_color`` is the color committed to by
    the previous pick; as long as that color still has a reachable cube we
    stick with it, so a scan-to-scan count tie (3 red -> 2 red/2 blue/2
    green after one red is placed) can't bounce the pick to another color
    before red is actually exhausted. Only once ``current_color`` has no
    candidates left do we re-rank by abundance and commit to a new color.
    """
    if current_color is not None:
        same_color = [c for c in cands if c.color == current_color]
        if same_color:
            return same_color[0], current_color
    counts: dict[str, int] = {}
    for c in cands:
        counts[c.color] = counts.get(c.color, 0) + 1
    best_color = max(counts, key=lambda color: counts[color])
    for c in cands:
        if c.color == best_color:
            return c, best_color
    return cands[0], cands[0].color


def release_z_for_level(calib, level: int) -> float:
    """TCP release height: 4mm above the current stack top.

    Stack top before placing ``level`` (1-based) is the top of the uppermost
    cube already seated -- ``pick_z + (level-1)*cube_height_mm`` in the same
    TCP frame as table grips (empty marker when level==1). Mirrors
    ``StackPlanner.release_z``.
    """
    stack_top = float(calib.pick_z) + (level - 1) * float(calib.cube_height_mm)
    return stack_top + 4.0


def place_on_stack(
    client: Mt4Client,
    calib,
    planner: StackPlanner,
    level: int,
    *,
    park_xy: tuple[float, float],
) -> None:
    """Carry the held cube to the site and seat it as ``level``.

    Assumes ``level - 1`` cubes are already stacked and the cube is held at
    the pick location around ``safe_z``. Sequence:

    1. carry: routed (usually one diagonal ``mp``) to a stage point
       STAGE_OFFSET_MM beside the stack, arriving at hover height
    2. hop over the stack top at hover, slow descend, release
    3. retreat: lift free when the z ceiling allows the fingertips above
       the placed cube (levels <= ~8), else lift a few mm and slide out
       perpendicular to the jaw axis so the open fingers sweep off the
       cube faces without pushing it (level 9's only option)
    """
    sx, sy = planner.sx, planner.sy
    built = level - 1
    rz = planner.release_z(level)
    hz = planner.hover_z(level)
    if hz is None:
        raise Mt4ClientError(f"level {level}: hover height unreachable")
    # Same axis-square wrist as the old along-arm place (assumes j4zero).
    j4 = j4_for_face_align(0.0, current_j4_deg=None, x=sx, y=sy)
    tcp = client.get_tcp()
    if tcp is None:
        raise Mt4ClientError("stack place: could not read TCP")
    stage = planner.stage_point(
        hz, built, prefer_xy=(float(tcp.x), float(tcp.y))
    )
    if stage is None:
        raise Mt4ClientError(f"level {level}: no reachable hover stage")
    # Carry to the stage point beside the column, then hop horizontally over
    # the stack top to (sx, sy) -- both at hover height, folded into one
    # queued mq via ``then`` so there's no stop/settle at the stage point.
    # The hop clears the column by construction (hz = hover height, fingers
    # already above the top cube), the same guarantee the old plain move_to
    # relied on.
    routed_travel(
        client, calib, planner, stage[0], stage[1], hz, built,
        j4=j4, then=[(sx, sy, hz)], step=f"level {level} carry",
    )
    _approach(client, calib, sx, sy, rz, "descend to stack release", j4=j4)
    _check(client.gripper(calib.grip_open_s), "stack release")
    fz = planner.free_retreat_z(level)
    if fz is not None:
        # Retreat AND transit to the camera park in one queued mq: lift
        # straight up clear of the placed cube, hop off-axis to the stage
        # exit, then route column-aware to the park pose -- no stop/settle at
        # the top of the lift, the exit, or (crucially) between the retreat
        # and the park transit. Folding the transit in here leaves the arm
        # already parked, so the next capture's go_camera_park() is a
        # get_tcp + early return instead of a whole separate move.
        retreat: list[tuple[float, float, float]] = [(sx, sy, fz)]
        exit_pt = planner.stage_point(fz, level, prefer_xy=park_xy)
        if exit_pt is not None:
            retreat.append((exit_pt[0], exit_pt[1], fz))
        from_xy = exit_pt if exit_pt is not None else (sx, sy)
        park_route = planner.route(
            (from_xy[0], from_xy[1], fz),
            (CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z),
            level,
        )
        if park_route is not None:
            retreat += park_route
        # "wrist" holds the J4 *joint* angle across the big base swing to the
        # park (a fixed world yaw would drive J4 past its soft limits there);
        # the cube is already released, so wrist orientation is free, and the
        # short lift/hop legs have no swing so it holds them steady too. If
        # the park route came back empty, the arm just stops at the exit and
        # the next go_camera_park() does the transit as before.
        _check(
            client.move_path(retreat, j4="wrist", speed_us=calib.travel_speed_us),
            f"level {level} retreat to park",
        )
    else:
        sz = planner.slide_z(level)
        exits = planner.slide_exits(j4, level, prefer_xy=park_xy)
        if not exits:
            raise Mt4ClientError(f"level {level}: no jaw-safe slide retreat")
        if sz > rz + 0.5:
            _travel(client, calib, sx, sy, sz, "slide lift", j4=j4)
        # Slow and wrist-locked: the fingers still straddle the placed cube.
        _approach(
            client, calib, exits[0][0], exits[0][1], sz,
            "slide clear of stack", j4=j4,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stack cubes on a calibrated marker (cartesian place)"
    )
    parser.add_argument(
        "--marker",
        type=int,
        required=True,
        help="calibration ArUco marker id to build the stack on (required)",
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument(
        "--max-levels",
        type=int,
        default=9,
        help="stop after this many levels (default 9; capped by the "
        "joint-limit z ceiling at the site)",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        metavar="N",
        help="a stack of N levels already stands on the marker: skip the "
        "site-clear phase and continue with level N+1 (the operator is "
        "trusted about N -- a wrong value will crash the gripper into or "
        "release cubes above the real stack)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="show a live annotated camera preview window (q or Esc to stop)",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="record a live annotated video to this path (e.g. stack_run.mp4)",
    )
    parser.add_argument(
        "--feed-fps",
        type=float,
        default=10.0,
        help="capture/annotate rate for --preview and --record (default 10)",
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

    if args.resume < 0 or args.resume >= args.max_levels:
        print(
            f"--resume {args.resume} must be in 0..{args.max_levels - 1}",
            file=sys.stderr,
        )
        return 1

    camera_kwargs = {} if args.camera is None else {"index": args.camera}
    client = Mt4Client() if not args.port else Mt4Client(port=args.port)
    cube_h = float(calib.cube_height_mm)
    planner = StackPlanner(calib, sx, sy)
    built = int(args.resume)
    watcher: _HomeKeyWatcher | None = None

    # A live feed and this script's own "look now" captures can't both open
    # the camera independently (one device, one owner) -- when either
    # --preview or --record is on, both pull frames from this one shared
    # stream instead of the default one-shot capture_frame() per look.
    stream = FrameStream(**camera_kwargs) if (args.preview or args.record) else None
    live_feed = (
        LiveFeed(
            calib=calib, stream=stream, fps=args.feed_fps,
            video_path=args.record, show_preview=args.preview,
        )
        if stream is not None
        else None
    )

    def snap_scene(stage: str) -> Scene:
        if live_feed is not None:
            if live_feed.stopped_by_user.is_set():
                raise PreviewStopped()
            live_feed.clear_target()
        go_camera_park(client, calib, planner, built)
        time.sleep(CAMERA_SETTLE_S)
        frame = stream.fresh(min_advance=2) if stream is not None else capture_frame(**camera_kwargs)
        scene = capture_scene(calib, frame)
        if live_feed is not None:
            live_feed.set_status([stage, scene.summary_line()])
        return scene

    def snap_decision(target: CubeDetection, stage: str) -> None:
        if live_feed is not None:
            live_feed.set_target(target.color, float(target.x), float(target.y))
            live_feed.set_status([stage])

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
            f"Stack site: marker {marker.marker_id} at "
            f"({sx:.1f},{sy:.1f}), cube_height={cube_h:.1f}mm"
        )

        # Joint-limit preflight: how many levels fit under the z ceiling
        # here, and from which level the retreat must slide instead of lift.
        target_levels = 0
        first_slide: int | None = None
        for level in range(1, args.max_levels + 1):
            if planner.hover_z(level) is None:
                break
            if first_slide is None and planner.free_retreat_z(level) is None:
                first_slide = level
            target_levels = level
        if target_levels == 0:
            print(
                f"joint z ceiling at this site is {planner.site_max_z:.0f}mm "
                "-- cannot even hover level 1",
                file=sys.stderr,
            )
            return 1
        retreat_note = (
            "lift-free retreats"
            if first_slide is None
            else f"slide retreat from level {first_slide}"
        )
        cap_note = (
            ""
            if target_levels == args.max_levels
            else f" (z ceiling caps requested {args.max_levels})"
        )
        print(
            f"Plan: {target_levels} level(s), z ceiling "
            f"{planner.site_max_z:.0f}mm, {retreat_note}{cap_note}"
        )
        print("Ctrl+C to stop, H (this window/terminal focused) to re-home before the next step.")
        watcher = _HomeKeyWatcher(client)
        watcher.start()

        all_markers = marker_slots_from_calibration(calib)
        behind_u = stack_shadow_behind_unit(calib, sx, sy)
        shadow_levels = max(1, int(target_levels))

        # --- Clear cubes near the stack marker ---------------------------------
        if built > 0:
            print(
                f"Resuming with {built} level(s) standing -- skipping site clear"
            )
        last_clear: tuple[str, float, float] | None = None
        for attempt in range(1, SITE_CLEAR_ATTEMPTS + 1):
            if built > 0:
                break  # resuming onto a standing stack; skips the else too
            if _check_home(client, watcher):
                continue
            try:
                scene = snap_scene(f"Clearing site (attempt {attempt}/{SITE_CLEAR_ATTEMPTS})")
                miss = pick_missed(scene, last_clear)
                last_clear = None
                if miss is not None:
                    print(
                        f"  clear pick missed -- cube still at "
                        f"({miss[0]:.0f},{miss[1]:.0f}); homing before retry"
                    )
                    _run_home(client, watcher)
                    continue
                near = cubes_near_site(scene, sx, sy)
                if not near:
                    print("Site clear")
                    break
                # Prefer pickable detections; fall back to raw occupants.
                pickable_near = [
                    c for c in scene.pickable(scene.cubes)
                    if dist_mm(float(c.x), float(c.y), sx, sy) < SITE_CLEAR_MM
                ]
                target = (pickable_near or near)[0]
                occupied = [
                    (float(c.x), float(c.y))
                    for c in scene.raw_cubes
                    if c is not target and c.x is not None and c.y is not None
                ]
                dest = clear_aside_xy(
                    sx, sy, float(target.x), float(target.y), occupied,
                    markers=all_markers, behind_u=behind_u,
                    shadow_levels=shadow_levels,
                )
                if dest is None:
                    dest = choose_park_slot(
                        scene, sx, sy, avoid=occupied,
                        markers=all_markers, behind_u=behind_u,
                        shadow_levels=shadow_levels,
                    )
                if dest is None:
                    print(
                        f"No reachable clear spot for {target.color} at "
                        f"({target.x:.0f},{target.y:.0f})",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    f"Clearing {target.color} at ({target.x:.0f},{target.y:.0f}) "
                    f"-> ({dest[0]:.0f},{dest[1]:.0f}) "
                    f"[attempt {attempt}/{SITE_CLEAR_ATTEMPTS}]"
                )
                snap_decision(
                    target,
                    f"Clearing {target.color} -> ({dest[0]:.0f},{dest[1]:.0f})",
                )
                pick(
                    client, calib, float(target.x), float(target.y),
                    yaw_deg=target.yaw_deg,
                )
                place(client, calib, dest[0], dest[1])
                last_clear = (target.color, float(target.x), float(target.y))
            except Mt4ClientError as exc:
                if _home_requested(watcher, exc):
                    _run_home(client, watcher)
                    continue
                print(f"  clear failed: {exc}", file=sys.stderr)
                return 1
        else:
            try:
                scene = snap_scene("Verifying site clear")
                still = cubes_near_site(scene, sx, sy)
                if still:
                    print(
                        "Site still occupied after clear attempts: "
                        + ", ".join(f"{c.color}({c.x:.0f},{c.y:.0f})" for c in still),
                        file=sys.stderr,
                    )
                    return 1
            except Mt4ClientError as exc:
                if _home_requested(watcher, exc):
                    _run_home(client, watcher)
                    print("Interrupted during final site-clear check -- rerun to build.", file=sys.stderr)
                else:
                    print(exc, file=sys.stderr)
                return 1

        # --- Build the stack ---------------------------------------------------
        # Single loop keyed off ``built``: each pass scans, verifies the
        # previous pick actually took its cube (homing + retrying the level
        # when it missed -- a miss usually means lost steps), then picks and
        # places ``built + 1``. The loop runs one extra pass after the last
        # placement so the final level is verified too.
        current_color: str | None = None
        integrity_failed = False
        last_pick: tuple[str, float, float] | None = None
        pick_fail_streak = 0
        while built < target_levels or last_pick is not None:
            level = built + 1
            if _check_home(client, watcher):
                continue
            try:
                stage_label = (
                    "Final verification"
                    if built >= target_levels
                    else f"Level {level}/{target_levels}: scanning"
                )
                scene = snap_scene(stage_label)

                # Did the previous pick actually take its cube?
                miss = pick_missed(scene, last_pick)
                if miss is not None:
                    missed_color = last_pick[0]
                    last_pick = None
                    built -= 1
                    pick_fail_streak += 1
                    print(
                        f"level {built + 1}: pick of {missed_color} missed -- "
                        f"cube still at ({miss[0]:.0f},{miss[1]:.0f}); homing "
                        f"and retrying "
                        f"({pick_fail_streak}/{PICK_FAIL_MAX_RETRIES})",
                        file=sys.stderr,
                    )
                    if pick_fail_streak >= PICK_FAIL_MAX_RETRIES:
                        print(
                            "Too many missed picks in a row -- stopping.",
                            file=sys.stderr,
                        )
                        break
                    _run_home(client, watcher)
                    continue
                last_pick = None
                pick_fail_streak = 0
                if built >= target_levels:
                    break

                rz = planner.release_z(level)
                hz = planner.hover_z(level)
                if hz is None:  # preflight guarantees this; belt and braces
                    print(f"level {level}: hover height unreachable -- stopping")
                    break

                # Drop stack-top phantoms behind the site along the camera LOS
                # (they appear once level 1+ is built).
                shadowed = []
                behind_u = (
                    stack_shadow_behind_unit(calib, sx, sy) if built > 0 else None
                )
                if built > 0:
                    issues = stack_integrity_issues(scene, sx, sy, behind_u)
                    if issues:
                        print(
                            "Stack integrity check failed:\n  "
                            + "\n  ".join(issues),
                            file=sys.stderr,
                        )
                        print(
                            f"Stopping -- the stack likely shed a cube "
                            f"around level {built}; the physical stack "
                            "may be shorter than reported.",
                            file=sys.stderr,
                        )
                        integrity_failed = True
                        break
                if behind_u is not None:
                    for c in scene.pickable(scene.cubes):
                        if dist_mm(float(c.x), float(c.y), sx, sy) < SITE_CLEAR_MM:
                            continue
                        if in_stack_camera_shadow(
                            float(c.x), float(c.y), sx, sy, behind_u,
                            stack_levels=built,
                        ):
                            shadowed.append(c)
                if shadowed:
                    print(
                        "Ignoring stack-shadow phantom(s): "
                        + ", ".join(
                            f"{c.color}({c.x:.0f},{c.y:.0f})" for c in shadowed
                        )
                    )
                cands = stack_candidates(
                    scene, sx, sy, calib=calib, stack_levels=built,
                )
                if not cands:
                    print(f"level {level}: no reachable cube outside site")
                    try:
                        reply = input(
                            "Move cubes into arm reach, then Enter to resume "
                            "(q to quit): "
                        ).strip().lower()
                    except EOFError:
                        break
                    if reply in ("q", "quit"):
                        break
                    print("Resuming...")
                    continue

                cube, current_color = select_stack_cube(cands, current_color)
                print(
                    f"\nLevel {level}: align-pick {cube.color} at "
                    f"({cube.x:.1f},{cube.y:.1f}) yaw={cube.yaw_deg:.0f}"
                )
                snap_decision(
                    cube, f"Level {level}/{target_levels}: picking {cube.color}",
                )
                if built > 0:
                    # Column-aware transit to the pick before descending.
                    routed_travel(
                        client, calib, planner,
                        float(cube.x), float(cube.y), calib.safe_z,
                        built, step="approach pick",
                    )
                pick_centered(
                    client, calib, float(cube.x), float(cube.y),
                    yaw_deg=cube.yaw_deg,
                )
                print(
                    f"  placing at marker ({sx:.1f},{sy:.1f}) "
                    f"release_z={rz:.1f} hover_z={hz:.1f}"
                )
                place_on_stack(
                    client, calib, planner, level,
                    park_xy=(CAMERA_PARK_X, CAMERA_PARK_Y),
                )
            except Mt4ClientError as exc:
                if _home_requested(watcher, exc):
                    _run_home(client, watcher)
                    continue
                print(f"  level {level} failed: {exc}", file=sys.stderr)
                return 1
            last_pick = (cube.color, float(cube.x), float(cube.y))
            built = level
            print(f"  placed level {level}")

        print(f"\nBuilt {built} level(s) on marker {marker.marker_id}")
        if integrity_failed:
            print(
                "Warning: stack integrity was compromised -- the physical "
                "stack is likely shorter than the built count.",
                file=sys.stderr,
            )
        try:
            go_camera_park(client, calib, planner, built)
        except Mt4ClientError as exc:
            if _home_requested(watcher, exc):
                _run_home(client, watcher)
            else:
                print(exc, file=sys.stderr)
        return 1 if integrity_failed else 0
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    except PreviewStopped:
        print("Preview window closed (q/Esc) -- stopping.")
        go_camera_park(client, calib, planner, built)
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
