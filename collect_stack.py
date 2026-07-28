#!/usr/bin/env python3
"""Autonomous pi0.5 demonstration collection: repeated stack/unstack cycles.

Runs stack_cubes.py/unstack_cubes.py's own building blocks -- site
clearing, ``StackPlanner``-routed carries, ``place_on_stack``/
``pick_from_stack`` -- but wraps each single-level pick+place as one
recorded episode, the same way collect_shuffle.py wraps a shuffle
pick+place: a background camera thread samples frames through the move
while an ``on_waypoint`` hook (now threaded through the stacking path too,
see mt4_vision/pickplace.py's ``_emit_path_waypoints``) records every real
commanded waypoint, tagged with a language prompt and a verdict.

One cycle = build a stack of ``--levels`` cubes on ``--marker``, then take
it back down to loose cubes on the table. The loose cubes from one
cycle's unstack become the raw material for the next cycle's stack, so
cycles chain without any manual re-scattering.

Verdicts reuse the two checks stack_cubes.py's own build loop already
trusts for its own error recovery, rather than inventing new vision logic:
``pick_missed`` (did the source cube actually leave -- same check used for
table picks) and ``stack_integrity_issues`` (did the column shed a cube).
The unstack phase reuses ``verify_pick_place`` directly, valid there
because a landing spot is a normal table position, not one hidden behind
the stacked-cube camera-parallax shadow that placements above level 1 are
subject to (see stack_cubes.py's ``stack_shadow_behind_unit`` docstring for
why that shadow rules out reusing the same check for the STACK phase).

Safety posture is deliberately more conservative than collect_shuffle.py's:
a table shuffle's state is always re-established by vision on the next
scan, but a stack's state is dead-reckoned from a *count* of what should be
standing, and unstack_cubes.py's own docstring is explicit that a wrong
count "will crash the gripper into or release above/below the real stack".
So anything that makes that count untrustworthy -- a detected shed cube, or
any hard Mt4ClientError mid-column -- ends the run rather than trying to
resume from an assumed height. A soft miss (pick_missed fires but the
column itself still checks out) is fully recoverable, same as
stack_cubes.py treats it: home and retry the same level.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import DEFAULT_CALIB_PATH, CalibrationError, load_calibration
from mt4_vision.camera import DEFAULT_CAMERA_INDEX
from mt4_vision.pickplace import (
    CAMERA_PARK_X,
    CAMERA_PARK_Y,
    go_camera_park,
    home_arm,
    near_camera_park,
    pick,
    place,
    resolve_pick_j4,
    retreat_for_camera,
    routed_travel,
)
from mt4_vision.scene import Scene, capture_scene, verify_pick_place
from mt4_vision.stackpath import StackPlanner
from mt4_vision.workspace import (
    MAX_REACH_MM,
    MarkerSlot,
    dist_mm,
    is_mp_reachable_xy,
    marker_slots_from_calibration,
)

from stack_cubes import (
    CAMERA_SETTLE_S,
    SITE_CLEAR_ATTEMPTS,
    clear_aside_xy,
    choose_park_slot,
    cubes_near_site,
    marker_by_id,
    pick_centered,
    pick_missed,
    place_on_stack,
    select_stack_cube,
    stack_candidates,
    stack_integrity_issues,
    stack_shadow_behind_unit,
)
from unstack_cubes import find_landing, pick_from_stack, random_place_j4

from mt4_pi.collect.prompts import build_stack_prompt, build_unstack_prompt
from mt4_pi.collect.recorder import DEFAULT_HZ, EpisodeCamera, EpisodeRecorder, OpenEpisode

# Consecutive missed picks (grasp_failed, column itself still fine) before
# giving up on this cycle's stack phase and moving on to unstack whatever
# built -- mirrors stack_cubes.py's own PICK_FAIL_MAX_RETRIES.
PICK_FAIL_MAX_RETRIES = 3

# Re-home every N recorded episodes (stack or unstack level), regardless of
# verdict -- same rationale and value as collect_shuffle.py's
# HOME_EVERY_PICKS: successful picks still accumulate small alignment
# drift that a fail-streak counter alone would miss.
HOME_EVERY_PICKS = 8


def _snap_scene(
    client: Mt4Client, calib, cam: EpisodeCamera, planner: StackPlanner, levels: int
) -> Scene:
    """Park for the camera (column-aware once a stack exists), settle, capture."""
    go_camera_park(client, calib, planner, levels)
    time.sleep(CAMERA_SETTLE_S)
    return capture_scene(calib, cam.capture_scene_frame())


def _discard_partial(episode: OpenEpisode | None) -> None:
    """Remove an episode directory abandoned mid-cycle (frames on disk, no
    meta.json/ticks.jsonl yet) -- unusable, and a converter would trip on it."""
    if episode is None or (episode.dir / "meta.json").exists():
        return
    shutil.rmtree(episode.dir, ignore_errors=True)
    print(f"  discarded partial {episode.dir.name}")


def _clear_site(
    client: Mt4Client,
    calib,
    cam: EpisodeCamera,
    planner: StackPlanner,
    *,
    sx: float,
    sy: float,
    all_markers: list[MarkerSlot],
    behind_u: tuple[float, float] | None,
) -> bool:
    """Push any cube near the stack site aside. Not recorded as an episode
    -- this is desk setup, not a demonstration (collect_shuffle.py already
    covers plain table pick/place). Returns False on a hard failure."""
    last_clear: tuple[str, float, float] | None = None
    for attempt in range(1, SITE_CLEAR_ATTEMPTS + 1):
        scene = _snap_scene(client, calib, cam, planner, 0)
        miss = pick_missed(scene, last_clear)
        last_clear = None
        if miss is not None:
            print(f"  clear pick missed at ({miss[0]:.0f},{miss[1]:.0f}) -- homing")
            home_arm(client)
            continue
        near = cubes_near_site(scene, sx, sy)
        if not near:
            return True
        pickable_near = [
            c for c in scene.pickable(scene.cubes)
            if dist_mm(float(c.x), float(c.y), sx, sy) < 70.0
        ]
        target = (pickable_near or near)[0]
        occupied = [
            (float(c.x), float(c.y))
            for c in scene.raw_cubes
            if c is not target and c.x is not None and c.y is not None
        ]
        dest = clear_aside_xy(
            sx, sy, float(target.x), float(target.y), occupied,
            markers=all_markers, behind_u=behind_u, shadow_levels=8,
        )
        if dest is None:
            dest = choose_park_slot(
                scene, sx, sy, avoid=occupied, markers=all_markers,
                behind_u=behind_u, shadow_levels=8,
            )
        if dest is None:
            print(f"No reachable clear spot for {target.color}", file=sys.stderr)
            return False
        print(
            f"  clearing site: {target.color} ({target.x:.0f},{target.y:.0f}) "
            f"-> ({dest[0]:.0f},{dest[1]:.0f}) [{attempt}/{SITE_CLEAR_ATTEMPTS}]"
        )
        pick(client, calib, float(target.x), float(target.y), yaw_deg=target.yaw_deg)
        place(client, calib, dest[0], dest[1])
        last_clear = (target.color, float(target.x), float(target.y))
    scene = _snap_scene(client, calib, cam, planner, 0)
    still = cubes_near_site(scene, sx, sy)
    if still:
        print(
            "Site still occupied after clear attempts: "
            + ", ".join(f"{c.color}({c.x:.0f},{c.y:.0f})" for c in still),
            file=sys.stderr,
        )
        return False
    return True


def _run_stack_phase(
    client: Mt4Client,
    calib,
    cam: EpisodeCamera,
    recorder: EpisodeRecorder,
    planner: StackPlanner,
    *,
    sx: float,
    sy: float,
    target_levels: int,
    hz: float,
    on_episode: Callable[[str], None],
) -> tuple[int, list[str], bool]:
    """Build up to ``target_levels`` on the (already cleared) site.

    Returns ``(built, level_colors, shed_detected)``. ``level_colors[i]`` is
    the color placed as level ``i + 1`` -- tracked here (not re-identified
    visually later) because the unstack phase needs to know what's coming
    off the top, and nothing in this codebase visually reads a stack's
    composition once it's built. ``shed_detected`` True means a placement
    dislodged an earlier cube -- the caller must stop the whole run (see
    module docstring): the built count can no longer be trusted for
    dead-reckoned unstacking.
    """
    behind_u = stack_shadow_behind_unit(calib, sx, sy)
    level_colors: list[str] = []
    current_color: str | None = None
    built = 0
    fail_streak = 0
    empty_scan_streak = 0
    scene = _snap_scene(client, calib, cam, planner, 0)
    while built < target_levels:
        level = built + 1
        cands = stack_candidates(scene, sx, sy, calib=calib, stack_levels=built)
        if not cands:
            empty_scan_streak += 1
            print(
                f"  level {level}: no reachable cube outside site -- waiting "
                f"({empty_scan_streak}/{PICK_FAIL_MAX_RETRIES})"
            )
            if empty_scan_streak >= PICK_FAIL_MAX_RETRIES:
                print(
                    f"  no reachable cube after {empty_scan_streak} scans -- "
                    f"ending stack phase at level {built}",
                    file=sys.stderr,
                )
                return built, level_colors, False
            time.sleep(5.0)
            scene = _snap_scene(client, calib, cam, planner, built)
            continue

        empty_scan_streak = 0
        cube, current_color = select_stack_cube(cands, current_color)
        prompt = build_stack_prompt(cube.color, level)
        print(f"[stack level {level}/{target_levels}] {prompt!r}: {cube.color} ({cube.x:.0f},{cube.y:.0f})")

        pre_status = client.get_status()
        episode = recorder.start(prompt, pre_status)
        if episode is None:
            print("  pre-move status didn't parse -- skipping this attempt")
            time.sleep(1.0)
            continue

        cam.begin_recording(episode.frames_dir, hz=hz)
        try:
            failed_exc: Mt4ClientError | None = None
            try:
                if built > 0:
                    approach_j4 = resolve_pick_j4(
                        client, calib, cube.yaw_deg,
                        face_align=bool(getattr(calib, "face_align_picks", True)),
                        x=float(cube.x), y=float(cube.y),
                    )
                    routed_travel(
                        client, calib, planner, float(cube.x), float(cube.y), calib.safe_z,
                        built, final_j4=approach_j4,
                        descend=(float(cube.x), float(cube.y), calib.table_z),
                        step="approach pick", on_waypoint=episode.on_waypoint,
                    )
                pick_centered(
                    client, calib, float(cube.x), float(cube.y),
                    yaw_deg=cube.yaw_deg, lift_after=False, on_waypoint=episode.on_waypoint,
                )
                place_on_stack(
                    client, calib, planner, level,
                    park_xy=(CAMERA_PARK_X, CAMERA_PARK_Y), on_waypoint=episode.on_waypoint,
                )
            except Mt4ClientError as exc:
                failed_exc = exc
            ticks = cam.end_recording()
            post_status = None
            if failed_exc is None:
                try:
                    post_status = client.get_status()
                except Mt4ClientError as exc:
                    failed_exc = exc
                    print(f"  post-move status unavailable: {exc}")

            if failed_exc is not None:
                episode_dir = recorder.finish(
                    episode, ticks, post_status, success=False, verdict="error",
                    pick={"color": cube.color, "x": float(cube.x), "y": float(cube.y)},
                    place={"x": sx, "y": sy, "kind": "stack", "level": level},
                    error=str(failed_exc),
                )
                print(f"  -> {episode_dir.name}: error ({failed_exc})")
                on_episode(episode_dir.name)
                raise failed_exc

            # Optimistic count for this scan's routing (place_on_stack
            # completed without raising) -- corrected below if the verify
            # says otherwise. Mirrors stack_cubes.py's own build loop, which
            # increments `built` the same way right after a place_on_stack
            # call returns and only backs off on the FOLLOWING scan's
            # pick_missed check.
            scene = _snap_scene(client, calib, cam, planner, level)
            miss = pick_missed(scene, (cube.color, float(cube.x), float(cube.y)))
            issues = stack_integrity_issues(scene, sx, sy, behind_u) if built > 0 else []
            if miss is not None:
                verdict = "grasp_failed"
            elif issues:
                verdict = "lost"
            else:
                verdict = "placed"

            episode_dir = recorder.finish(
                episode, ticks, post_status, success=(verdict == "placed"), verdict=verdict,
                pick={"color": cube.color, "x": float(cube.x), "y": float(cube.y)},
                place={"x": sx, "y": sy, "kind": "stack", "level": level},
                error=None,
            )
            print(f"  -> {episode_dir.name}: {verdict} ({len(ticks)} frames)")
            on_episode(episode_dir.name)
        finally:
            # No-op once recorder.finish() has run (meta.json exists) --
            # only fires for an exception that skipped it entirely, e.g. a
            # KeyboardInterrupt landing between begin_recording and finish.
            _discard_partial(episode)

        if verdict == "placed":
            built = level
            level_colors.append(cube.color)
            fail_streak = 0
            continue
        if issues:
            print(
                "Stack integrity compromised: "
                + "; ".join(issues)
                + " -- stopping (built count is no longer trustworthy)",
                file=sys.stderr,
            )
            return built, level_colors, True
        # grasp_failed: column itself is fine, just retry this level.
        fail_streak += 1
        print(f"  pick missed at ({miss[0]:.0f},{miss[1]:.0f}) -- homing ({fail_streak}/{PICK_FAIL_MAX_RETRIES})")
        home_arm(client)
        if fail_streak >= PICK_FAIL_MAX_RETRIES:
            print(f"  {fail_streak} missed picks in a row -- ending stack phase at level {built}")
            return built, level_colors, False
        scene = _snap_scene(client, calib, cam, planner, built)

    return built, level_colors, False


def _run_unstack_phase(
    client: Mt4Client,
    calib,
    cam: EpisodeCamera,
    recorder: EpisodeRecorder,
    planner: StackPlanner,
    *,
    sx: float,
    sy: float,
    built: int,
    level_colors: list[str],
    all_markers: list[MarkerSlot],
    rng: random.Random,
    hz: float,
    on_episode: Callable[[str], None],
) -> None:
    """Take ``built`` cubes off the top, scattering each at a random open
    landing. Raises Mt4ClientError on a hard failure -- the caller treats
    that as run-ending, same reasoning as the stack phase.
    """
    remaining = built
    placed: list[tuple[float, float]] = []
    if remaining > 0:
        obstacle_scene = _snap_scene(client, calib, cam, planner, remaining)
        obstacles = [
            (float(c.x), float(c.y))
            for c in obstacle_scene.raw_cubes
            if c.x is not None and c.y is not None
            and dist_mm(float(c.x), float(c.y), sx, sy) >= 90.0
        ]
    else:
        obstacles = []

    while remaining > 0:
        level = remaining
        color = level_colors[level - 1]
        landing, spacing = find_landing(rng, sx=sx, sy=sy, markers=all_markers, avoid=placed + obstacles)
        tx, ty = landing
        j4 = random_place_j4(tx, ty, rng)
        prompt = build_unstack_prompt(color)
        print(f"[unstack level {level}] {prompt!r}: {color} -> ({tx:.0f},{ty:.0f})")

        pre_status = client.get_status()
        episode = recorder.start(prompt, pre_status)
        if episode is None:
            print("  pre-move status didn't parse -- skipping this attempt")
            time.sleep(1.0)
            continue

        cam.begin_recording(episode.frames_dir, hz=hz)
        try:
            failed_exc: Mt4ClientError | None = None
            try:
                pick_from_stack(
                    client, calib, planner, level,
                    approach_prefer_xy=(CAMERA_PARK_X, CAMERA_PARK_Y), on_waypoint=episode.on_waypoint,
                )
                routed_travel(
                    client, calib, planner, tx, ty, calib.safe_z, level - 1,
                    j4=j4, step=f"level {level} carry to landing", on_waypoint=episode.on_waypoint,
                )
                place(
                    client, calib, tx, ty, j4=j4, travel_z=calib.safe_z,
                    on_waypoint=episode.on_waypoint,
                )
            except Mt4ClientError as exc:
                failed_exc = exc
            ticks = cam.end_recording()
            post_status = None
            if failed_exc is None:
                try:
                    post_status = client.get_status()
                except Mt4ClientError as exc:
                    failed_exc = exc
                    print(f"  post-move status unavailable: {exc}")

            if failed_exc is not None:
                episode_dir = recorder.finish(
                    episode, ticks, post_status, success=False, verdict="error",
                    pick={"color": color, "x": sx, "y": sy},
                    place={"x": tx, "y": ty, "kind": "unstack", "level": level, "j4": j4},
                    error=str(failed_exc),
                )
                print(f"  -> {episode_dir.name}: error ({failed_exc})")
                on_episode(episode_dir.name)
                raise failed_exc

            scene = _snap_scene(client, calib, cam, planner, level - 1)
            verdict = verify_pick_place(
                scene, pick_x=sx, pick_y=sy, pick_color=color, place_x=tx, place_y=ty,
            )
            episode_dir = recorder.finish(
                episode, ticks, post_status, success=(verdict == "placed"), verdict=verdict,
                pick={"color": color, "x": sx, "y": sy},
                place={"x": tx, "y": ty, "kind": "unstack", "level": level, "j4": j4},
                error=None,
            )
            print(f"  -> {episode_dir.name}: {verdict} ({len(ticks)} frames)")
            on_episode(episode_dir.name)
        finally:
            _discard_partial(episode)

        # The cube conclusively left the stack (pick_from_stack raises on a
        # real grip failure) regardless of whether the landing itself was
        # where planned -- verify_pick_place only judges landing accuracy.
        placed.append((tx, ty))
        remaining = level - 1


def _run(
    client: Mt4Client,
    calib,
    *,
    camera_index: int,
    calib_path: str,
    marker_id: int,
    levels: int,
    cycles: int | None,
    out_dir: Path,
    hz: float,
    home_every: int,
) -> int:
    marker = marker_by_id(calib, marker_id)
    sx, sy = marker.x, marker.y
    if not is_mp_reachable_xy(sx, sy) or dist_mm(sx, sy, 0.0, 0.0) > MAX_REACH_MM:
        print(f"marker {marker.marker_id} at ({sx:.1f},{sy:.1f}) is out of reach", file=sys.stderr)
        return 1
    if near_camera_park(sx, sy):
        print(
            f"marker {marker.marker_id} at ({sx:.1f},{sy:.1f}) sits under the camera "
            "park -- the arm would hit the stack between moves; use another marker",
            file=sys.stderr,
        )
        return 1

    planner = StackPlanner(calib, sx, sy)
    target_levels = 0
    for level in range(1, levels + 1):
        if planner.hover_z(level) is None:
            break
        target_levels = level
    if target_levels == 0:
        print(f"joint z ceiling at this site cannot even hover level 1", file=sys.stderr)
        return 1
    if target_levels < levels:
        print(f"z ceiling caps requested {levels} levels to {target_levels}")

    if client.get_status().homed:
        print("Already homed")
    else:
        print("Homing...")
        home_arm(client)
        print("Home ok")

    all_markers = marker_slots_from_calibration(calib)
    cam = EpisodeCamera(camera_index)
    recorder = EpisodeRecorder(out_dir, camera_index=camera_index, calib_path=calib_path)
    rng = random.Random()
    picks_since_home = 0

    def on_episode(_name: str) -> None:
        nonlocal picks_since_home
        picks_since_home += 1
        if home_every and picks_since_home >= home_every:
            print(f"  {picks_since_home} picks since last home -- homing")
            home_arm(client)
            picks_since_home = 0

    try:
        cycle = 0
        while cycles is None or cycle < cycles:
            cycle += 1
            print(f"\n=== cycle {cycle}{f'/{cycles}' if cycles else ''} ===")
            behind_u = stack_shadow_behind_unit(calib, sx, sy)
            if not _clear_site(client, calib, cam, planner, sx=sx, sy=sy, all_markers=all_markers, behind_u=behind_u):
                print("Could not clear the stack site -- stopping.", file=sys.stderr)
                return 1

            built, level_colors, shed = _run_stack_phase(
                client, calib, cam, recorder, planner,
                sx=sx, sy=sy, target_levels=target_levels, hz=hz, on_episode=on_episode,
            )
            if built == 0 and not shed:
                print("Built nothing this cycle (no reachable cubes) -- stopping.", file=sys.stderr)
                return 1

            _run_unstack_phase(
                client, calib, cam, recorder, planner,
                sx=sx, sy=sy, built=built, level_colors=level_colors,
                all_markers=all_markers, rng=rng, hz=hz, on_episode=on_episode,
            )

            if shed:
                print(
                    "Stopping after this cycle: a stack placement shed a cube, "
                    "so the built count could not be trusted for the rest of "
                    "this cycle's build -- what stood was unstacked, but the "
                    "session ends here rather than starting a new build on an "
                    "uncertain physical state.",
                    file=sys.stderr,
                )
                go_camera_park(client, calib, planner, 0)
                return 1
    except Mt4ClientError as exc:
        print(f"Stopping: {exc}", file=sys.stderr)
        try:
            home_arm(client)
            retreat_for_camera(client, calib)
        except Mt4ClientError:
            pass
        return 1
    finally:
        cam.close()

    try:
        go_camera_park(client, calib, planner, 0)
    except Mt4ClientError:
        pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=int, required=True, help="calibration ArUco marker id to build the stack on")
    parser.add_argument("--levels", type=int, default=4, help="target stack height per cycle (default 4)")
    parser.add_argument("--cycles", type=int, default=None, help="stop after N build+unstack cycles (default: run until Ctrl+C)")
    parser.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--out", default="data/pi_stack_demos", help="output root (default data/pi_stack_demos)")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help=f"frame sample rate during motion (default {DEFAULT_HZ})")
    parser.add_argument(
        "--home-every", type=int, default=HOME_EVERY_PICKS,
        help=f"re-home every N recorded episodes regardless of verdict (default {HOME_EVERY_PICKS}, 0 disables)",
    )
    args = parser.parse_args()

    try:
        calib = load_calibration(Path(args.calib))
    except CalibrationError as exc:
        print(exc, file=sys.stderr)
        return 1

    client = Mt4Client() if args.port is None else Mt4Client(port=args.port)
    try:
        time.sleep(1.0)
        return _run(
            client, calib,
            camera_index=args.camera,
            calib_path=args.calib,
            marker_id=args.marker,
            levels=args.levels,
            cycles=args.cycles,
            out_dir=Path(args.out),
            hz=args.hz,
            home_every=max(0, args.home_every),
        )
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
