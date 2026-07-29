#!/usr/bin/env python3
"""Autonomous pi0.5 demonstration collection: shuffle cubes and record it.

Runs the same vision-guided pick/place loop as shuffle_blocks.py -- pick an
open-table cube onto a free calibrated marker, or a marker cube onto open
table -- but wraps each pick+place cycle as one recorded demonstration
episode: a background camera thread samples frames through the move while
mt4_vision.pickplace's on_waypoint hook records every real commanded
waypoint (see mt4_pi/collect/recorder.py's WaypointLog), tagged with a
language prompt and the existing verify_pick_place() success verdict.

Output is NOT a LeRobot dataset -- it's the raw per-episode trace a later
conversion step (mirroring openpi's
examples/libero/convert_libero_data_to_lerobot.py) turns into one. See
mt4_pi/collect/recorder.py for the on-disk layout.

Unattended-safe: same H-to-rehome / Ctrl+C handling as shuffle_blocks.py,
plus a fail-streak counter that forces a re-home after repeated failures
instead of hammering a possibly-stuck arm for hours unattended, and an
unconditional re-home every HOME_EVERY_PICKS picks to clear the slow
alignment drift that successful picks still accumulate.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.status import Mt4Status
from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration
from mt4_vision.camera import DEFAULT_CAMERA_INDEX
from mt4_vision.pickplace import home_arm, pick_cube, place, retreat_for_camera
from mt4_vision.policy import plan_shuffle
from mt4_vision.scene import capture_scene, verify_pick_place
from mt4_vision.workspace import free_placement_slots
from unstack_cubes import random_place_j4
from mt4_vision.shuffle import POST_MOVE_RECHECK_ATTEMPTS, POST_MOVE_RECHECK_DELAY_S

from mt4_pi.collect.balance import (
    COLLECTION_SLOTS,
    count_cells,
    deficits,
    format_cells,
    plan_balanced,
)
from mt4_pi.collect.prompts import build_prompt
from mt4_pi.collect.recorder import (
    DEFAULT_HZ,
    EpisodeCamera,
    EpisodeRecorder,
    OpenEpisode,
)

# Consecutive failed/rejected episodes before forcing a re-home -- mirrors
# stack_cubes.py's PICK_FAIL_MAX_RETRIES; a run left overnight should not
# spend hours retrying against lost steps or a stuck grip.
FAIL_STREAK_HOME = 3

# Picks between unconditional re-homes, regardless of verdict. The arm
# occasionally clips a cube on approach or retreat -- not hard enough to
# fail the pick, so FAIL_STREAK_HOME never fires, but enough to leave a
# small standing offset that accumulates over a long run. Every episode
# recorded while offset carries that error into its state labels, so the
# damage is silent and cumulative rather than a visible failure. Homing
# clears it, so do it on a fixed cadence instead of waiting for a failure
# to reveal drift that is already in the data.
HOME_EVERY_PICKS = 8

# A CH340 USB-serial dropout (or an MCU reset) makes the port vanish
# mid-cycle and every client call raise until it re-enumerates. Do NOT poll
# through that window: after an MCU reset the bootloader stays resident for
# ~13s while the host keeps talking to it, eating the commands that follow.
# Go fully silent for longer than that, then try once per round.
SERIAL_SILENCE_S = 15.0
SERIAL_RECOVER_TRIES = 5


def _verify_with_recheck(
    client: Mt4Client,
    cam: EpisodeCamera,
    calib,
    *,
    pick_x: float,
    pick_y: float,
    pick_color: str,
    place_x: float,
    place_y: float,
) -> str:
    """verify_pick_place() with shuffle.py's post-move recheck.

    Retreats first. ``place()`` leaves the arm hovering at safe_z directly
    over the cube it just set down -- measured, 0.0mm away in every episode
    of a run -- and under a camera this oblique the gripper hides the
    landing site often enough to matter. The recheck loop alone cannot
    recover from that: the arm does not move between attempts, so all three
    captures see the same occlusion and the verdict lands on "lost" (cube at
    neither end) or "grasp_failed". The very next cycle then retreats,
    spots the cube exactly where it was placed, and happily picks it up
    again. Measured on a live run: 3 of 7 consecutive episodes were
    mislabelled this way, and whether a given placement survives is pure
    geometry -- which is why the verdicts looked random rather than
    correlated with anything about the pick.

    A capture taken right after the arm clears can also lag the real desk by
    a few hundred ms (driver frame backlog after the multi-second gap during
    arm motion) -- see mt4_vision.shuffle's POST_MOVE_RECHECK_* comment.
    That is what the retries are for, and they stay.

    shuffle.py only loses a redundant retry to either problem; here it would
    stamp success=False onto an otherwise good episode's meta.json, and a
    mislabelled demonstration hurts the fine-tune more than a missing one --
    it teaches the model that a correct trajectory was a failure. A real
    failure still reads the same way after the retreat and the retries.

    The retreat is close to free: the collection loop already retreats at
    the top of the next cycle, which now finds the arm parked and no-ops.
    """
    try:
        retreat_for_camera(client, calib)
    except Mt4ClientError as exc:
        # Verify from wherever the arm is rather than abandoning the
        # episode -- a possibly-occluded check still beats no check.
        print(f"  retreat before verify failed: {exc}")
    verdict = "error"
    for attempt in range(POST_MOVE_RECHECK_ATTEMPTS + 1):
        if attempt:
            time.sleep(POST_MOVE_RECHECK_DELAY_S)
        scene = capture_scene(calib, cam.capture_scene_frame())
        verdict = verify_pick_place(
            scene,
            pick_x=pick_x, pick_y=pick_y, pick_color=pick_color,
            place_x=place_x, place_y=place_y,
        )
        if verdict == "placed":
            break
    return verdict


def _discard_partial(episode: OpenEpisode | None) -> None:
    """Remove an episode directory abandoned mid-cycle.

    Its frames are on disk but meta.json/ticks.jsonl were never written, so
    it has no state labels and no verdict -- unusable for training, and a
    converter walking episodes/ would trip over it. Episode ids are
    monotonic, so the resulting gap in the numbering is harmless.
    """
    if episode is None or (episode.dir / "meta.json").exists():
        return
    shutil.rmtree(episode.dir, ignore_errors=True)
    print(f"  discarded partial {episode.dir.name}")


def _run(
    client: Mt4Client,
    calib,
    *,
    camera_index: int,
    calib_path: str,
    out_dir: Path,
    hz: float,
    max_episodes: int | None,
    home_every: int,
    retry_s: float,
    balance_target: int = 0,
    count_roots: list[Path] | None = None,
    random_yaw_prob: float = 1.0,
    seed: int | None = None,
) -> None:
    """`random_yaw_prob` is the chance each place lands at a random cube
    orientation instead of square to the world axes.

    `place()` defaults to `axis_align=True`, which squares the released cube
    to the X/Y axes. That is right for tidy shuffling and wrong as the ONLY
    behaviour for a dataset: every place drives the table toward yaw = 0
    (mod 90), so the corpus concentrates near axis-aligned and the policy
    rarely sees an angled cube. Measured over the shuffle corpus: mean |yaw|
    at grasp 10.0 deg, 43.8% of picks within 5 deg of axis-aligned, only
    18.6% beyond 20 deg. The already-randomising stack collector reaches
    34.2% beyond 20 deg through the same helper.

    Note the gripper alignment itself was never missing: `resolve_pick_j4`
    estimates the cube's edge angle from detection and `j4_for_face_align`
    squares the jaws to a face, so every demonstration does contain wrist
    alignment. The problem is narrower -- that behaviour is only ever
    demonstrated at angles near zero, so the j4 channel is under-exercised
    rather than absent. Measured over 201 pre-change episodes, the commanded
    place angle was **exactly 0.0 deg every single time** (std 0.0).

    A probability rather than a flag because BOTH regimes are wanted.
    Axis-aligned picks are cleaner and their orientation is known, which
    makes them the easier examples worth keeping; randomised ones supply the
    variation. Since the 210 episodes already collected are overwhelmingly
    axis-aligned, running the remainder at 1.0 lands the *combined* corpus
    on a mix rather than replacing one bias with another.
    """
    rng = random.Random(seed)
    if client.get_status().homed:
        print("Already homed")
    else:
        print("Homing...")
        home_arm(client)
        print("Home ok")

    cam = EpisodeCamera(camera_index)
    recorder = EpisodeRecorder(out_dir, camera_index=camera_index, calib_path=calib_path)

    # Balance mode counts the WHOLE corpus, not just this run's output, so a
    # restart keeps filling the same gaps instead of re-balancing from zero.
    cells = Counter()
    if balance_target:
        roots = count_roots or [out_dir]
        cells = count_cells(*roots)
        print(f"balance target {balance_target}/cell over {[str(r) for r in roots]}")
        print(f"  current: {format_cells(cells, balance_target)}")

    last_place: tuple[float, float] | None = None
    fail_streak = 0
    picks_since_home = 0
    n = 0

    def rehome(reason: str) -> None:
        """Home and clear both counters -- any home, for any reason, resets
        the periodic cadence too, so a fail-streak home never leaves a
        stale picks_since_home to trigger a redundant home right after."""
        nonlocal fail_streak, picks_since_home
        print(f"  {reason} -- homing")
        home_arm(client)
        fail_streak = 0
        picks_since_home = 0

    def recover_serial(exc: Mt4ClientError) -> bool:
        """Wait out a serial dropout, then re-establish a homed arm.

        A CH340 re-enumeration or MCU reset takes the port away entirely, so
        there is nothing to do but stay quiet and let it come back -- see
        SERIAL_SILENCE_S on why polling through that window is actively
        harmful. Returns False only when the link never returned, the one
        case worth ending the run over.
        """
        nonlocal fail_streak, picks_since_home, last_place
        print(f"serial link lost: {exc}")
        for attempt in range(1, SERIAL_RECOVER_TRIES + 1):
            print(
                f"  silent {SERIAL_SILENCE_S:.0f}s, then retry "
                f"{attempt}/{SERIAL_RECOVER_TRIES}"
            )
            time.sleep(SERIAL_SILENCE_S)
            try:
                status = client.get_status()
            except Mt4ClientError as retry_exc:
                print(f"  still down: {retry_exc}")
                continue
            # Position is unknown after a reset regardless of what `homed`
            # claims, so always home rather than trusting the flag.
            print(f"  link back (homed={status.homed}) -- homing before resuming")
            try:
                home_arm(client)
            except Mt4ClientError as home_exc:
                print(f"  home failed: {home_exc}")
                continue
            fail_streak = 0
            picks_since_home = 0
            last_place = None
            print("  recovered -- resuming collection")
            return True
        return False

    try:
        while max_episodes is None or n < max_episodes:
            # Tracks a cycle that opened an episode dir but has not written
            # its meta.json yet, so the handler below can clean up.
            in_flight: OpenEpisode | None = None
            try:
                try:
                    retreat_for_camera(client, calib)
                except Mt4ClientError as exc:
                    print(f"camera retreat failed: {exc}")

                scene = capture_scene(calib, cam.capture_scene_frame())
                if balance_target:
                    # Finishing is a real end state here, unlike free
                    # shuffling: once every cell is at target the planner can
                    # only return "wait", and the loop would spin on the
                    # retry timer forever.
                    if not any(deficits(cells, balance_target).values()):
                        print(f"all cells at target {balance_target} -- collection complete")
                        print(f"  final: {format_cells(cells, balance_target)}")
                        return
                    # Re-check the collection grid against this frame:
                    # free_placement_slots drops anything unreachable, too
                    # close to a marker's paper, or already holding a cube.
                    # raw_cubes, not cubes: `cubes` is the phantom-filtered
                    # pick-quality subset, and a detection dropped for being
                    # unpickable still occupies the space. rebuild_workspace_state
                    # uses raw for exactly this reason.
                    occupancy = scene.raw_cubes if scene.raw_cubes is not None else scene.cubes
                    slots = free_placement_slots(
                        calib, scene.markers, occupancy, slots=COLLECTION_SLOTS
                    )
                    action = plan_balanced(
                        scene, cells, target=balance_target,
                        avoid_xy=last_place, slots=slots,
                    )
                else:
                    action = plan_shuffle(scene, avoid_xy=last_place)

                if action.kind != "pick" or action.cube is None or action.place_x is None:
                    print(f"Waiting {retry_s:.0f}s for a clearer scene ({action.reason})")
                    time.sleep(retry_s)
                    continue

                cube = action.cube
                place_x, place_y = float(action.place_x), float(action.place_y)
                prompt = build_prompt(cube.color, action.place_kind)
                print(
                    f"[ep {n:06d}] {prompt!r}: {cube.color} "
                    f"({cube.x:.0f},{cube.y:.0f}) -> ({place_x:.0f},{place_y:.0f})"
                )

                pre_status = client.get_status()
                episode = recorder.start(prompt, pre_status)
                if episode is None:
                    print("  pre-move status didn't parse -- skipping this cycle")
                    time.sleep(1.0)
                    continue
                in_flight = episode

                cam.begin_recording(episode.frames_dir, hz=hz)
                failed_exc: Mt4ClientError | None = None
                try:
                    pick_cube(client, calib, cube, on_waypoint=episode.on_waypoint)
                    # Land at a random orientation rather than square to the
                    # axes -- see random_yaw in _run's signature.
                    place_j4 = (
                        random_place_j4(place_x, place_y, rng)
                        if rng.random() < random_yaw_prob else None
                    )
                    place(
                        client, calib, place_x, place_y,
                        j4=place_j4, on_waypoint=episode.on_waypoint,
                    )
                except Mt4ClientError as exc:
                    failed_exc = exc
                ticks = cam.end_recording()
                # Must never abort the cycle. pick/place failures above are
                # already caught, but if the serial link died during them
                # this read dies too -- and it used to propagate straight out
                # of _run, killing the whole run at the one point where
                # frames were already on disk but meta.json did not exist
                # yet, leaving an unreadable half-episode behind. A missing
                # end pose is worth far less than the episode and the rest of
                # the run.
                post_status: Mt4Status | None = None
                try:
                    post_status = client.get_status()
                except Mt4ClientError as exc:
                    if failed_exc is None:
                        failed_exc = exc
                    print(f"  post-move status unavailable: {exc}")

                verdict = "error"
                if failed_exc is None:
                    verdict = _verify_with_recheck(
                        client, cam, calib,
                        pick_x=float(cube.x), pick_y=float(cube.y),
                        pick_color=cube.color,
                        place_x=place_x, place_y=place_y,
                    )

                episode_dir = recorder.finish(
                    episode, ticks, post_status,
                    success=(verdict == "placed"),
                    verdict=verdict,
                    pick={"color": cube.color, "x": float(cube.x), "y": float(cube.y)},
                    place={
                        "x": place_x, "y": place_y,
                        "kind": action.place_kind, "marker_id": action.place_marker_id,
                    },
                    error=str(failed_exc) if failed_exc else None,
                )
                # Complete on disk from here -- no longer the handler's to
                # clean up if a later call in this cycle loses the link.
                in_flight = None
                print(f"  -> {episode_dir.name}: {verdict} ({len(ticks)} frames)")
                n += 1
                # Counts attempted picks, not successful ones: a pick that
                # missed and shoved its cube is at least as likely to have
                # left the arm offset as one that worked.
                picks_since_home += 1

                if verdict == "placed":
                    fail_streak = 0
                    # Only successful episodes survive conversion, so only
                    # they may retire a cell's deficit.
                    if balance_target:
                        cells[(cube.color, action.place_kind)] += 1
                        print(f"  cells: {format_cells(cells, balance_target)}")
                    last_place = (place_x, place_y)
                    if home_every and picks_since_home >= home_every:
                        rehome(f"{picks_since_home} picks since last home")
                    continue

                fail_streak += 1
                if failed_exc is not None and "interrupted" in str(failed_exc):
                    rehome("interrupted")
                    last_place = None
                    continue
                if fail_streak >= FAIL_STREAK_HOME:
                    rehome(f"{fail_streak} failures in a row")
                elif home_every and picks_since_home >= home_every:
                    rehome(f"{picks_since_home} picks since last home")
                else:
                    print(f"  move did not verify ({verdict}) -- retrying next cycle")
                last_place = None
                time.sleep(retry_s if failed_exc is not None else 0.0)

            except Mt4ClientError as exc:
                # Whatever the inner handlers did not already absorb: a lost
                # link during pre-move status, or inside one of the rehome
                # calls. Unwind this cycle, then wait the link out rather
                # than ending a run that still has episodes to collect.
                if cam.is_recording:
                    cam.end_recording()
                _discard_partial(in_flight)
                if not recover_serial(exc):
                    print("serial link did not come back -- stopping")
                    return
    finally:
        cam.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--out", default="data/pi_demos", help="output root (default data/pi_demos)")
    parser.add_argument(
        "--hz", type=float, default=DEFAULT_HZ,
        help=f"frame sample rate during motion (default {DEFAULT_HZ}, matches the DROID reference rate)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="stop after N episodes (default: run until Ctrl+C)")
    parser.add_argument(
        "--home-every", type=int, default=HOME_EVERY_PICKS,
        help=f"re-home every N picks regardless of verdict, to clear accumulated "
             f"alignment drift (default {HOME_EVERY_PICKS}, 0 disables)",
    )
    parser.add_argument("--retry", type=float, default=5.0, help="seconds to wait when no valid move is visible")
    parser.add_argument(
        "--random-yaw-prob", type=float, default=1.0, metavar="P",
        help="chance each place lands the cube at a random orientation rather than square "
             "to the world axes (default 1.0; 0 restores place()'s axis-align default). "
             "Axis-aligned picks are cleaner and already dominate the existing corpus, so "
             "collecting the remainder at 1.0 yields a mixed corpus rather than a new bias.",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed the placement-orientation RNG")
    parser.add_argument(
        "--balance", type=int, default=0, metavar="N",
        help="collect toward N successful episodes per (colour, place-kind) cell instead of "
             "shuffling freely. The default planner cannot produce a balanced corpus: it only "
             "reaches to_slot when no marker is free, and picks colour uniformly over whatever "
             "is pickable. 0 (default) keeps the original behaviour.",
    )
    parser.add_argument(
        "--count-root", action="append", dest="count_roots", default=None,
        help="collection root to count toward --balance; repeat to include earlier corpora "
             "(default: --out only). Pass the existing roots so the run fills real gaps.",
    )
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    client = Mt4Client() if args.port is None else Mt4Client(port=args.port)
    try:
        time.sleep(1.0)
        print("Collection loop started (Ctrl+C to stop)")
        _run(
            client, calib,
            camera_index=args.camera,
            calib_path=args.calib,
            out_dir=Path(args.out),
            hz=args.hz,
            max_episodes=args.episodes,
            home_every=max(0, args.home_every),
            retry_s=args.retry,
            random_yaw_prob=min(1.0, max(0.0, args.random_yaw_prob)),
            seed=args.seed,
            balance_target=max(0, args.balance),
            count_roots=[Path(r) for r in args.count_roots] if args.count_roots else None,
        )
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
