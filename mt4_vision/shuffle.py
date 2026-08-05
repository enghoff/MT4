"""Indefinite cube shuffle loop (see shuffle_blocks.py CLI).

Plans only from the latest camera frame via detection-as-state Scene.
No synthetic cubes, no camera-park retreat: if the arm obscures a cube or
marker, that object simply isn't in the scene. Ghosts must be filtered out
of pick candidates; they are never "handled" as grips.

The one deliberate exception is `last_place`: the destination of the last
completed move is passed to plan_shuffle so it deprioritizes (never
excludes) re-picking that same cube next cycle. It only prefers variety, and
always falls back to the same cube if it is the only pickable one: a stigma
that refused to repeat a move outright would hide grasp and vision bugs.

Lookahead: before executing a planned move, also ask plan_shuffle (via
_lookahead_action) whether a *second*, independent move is already visible
in that same capture. If so, both moves' legs are planned and sent through
one ``send_legs`` call -- no capture, settle pause, verification, OR
stop/settle between the two transfers. This trades verifying the first move
immediately for fewer pauses; a first move that actually failed still shows
up as a pickable cube on the next real capture, just one cycle later than
it otherwise would.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import Calibration
from mt4_vision.camera import grab_frame, open_camera
from mt4_vision.detect import CubeDetection
from mt4_vision.motion import (
    Grasp,
    ensure_homed,
    plan_transfer_legs,
    send_legs,
    square_place,
)
from mt4_vision.pickplace import home_arm, retreat_for_camera
from mt4_vision.policy import Action, plan_shuffle
from mt4_vision.scene import Scene, capture_scene, verify_pick_place

# A post-move capture taken right after the arm clears sometimes lags the
# real desk by a few hundred ms (camera driver frame backlog after the
# multi-second gap during arm motion) -- a genuinely completed move can
# briefly still look like "grasp_failed" (cube seen at the old spot, none at
# the new one). Re-check a couple of times before trusting that reading and
# repeating the same pick; a real grasp failure still reads the same way
# after these retries and proceeds exactly as before.
POST_MOVE_RECHECK_ATTEMPTS = 2
POST_MOVE_RECHECK_DELAY_S = 0.4


class _HomeKeyWatcher:
    """Detect a tap of H (same binding as jog.py) without blocking.

    Gated on this process's console having OS focus -- GetAsyncKeyState is
    global, so without that check an H press in any other window (browser,
    another terminal) would re-home the arm mid-shuffle.
    """

    def __init__(self, client: Mt4Client) -> None:
        self._client = client
        self._requested = threading.Event()
        self._h_down = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="shuffle-home-key", daemon=True
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
        from jog import console_focused, key_down

        while not self._stop.is_set():
            down = console_focused() and key_down("h")
            if down and not self._h_down:
                self._requested.set()
                self._client.request_interrupt()
            self._h_down = down
            self._stop.wait(0.05)


class _VisionPrefetch:
    """Drain the camera during arm motion; capture once motion ends."""

    def __init__(
        self,
        calib: Calibration,
        camera: int,
        *,
        settle_s: float,
    ) -> None:
        self._calib = calib
        self._settle_s = settle_s
        self._cap = open_camera(camera)
        self._motion_done = threading.Event()
        self._motion_done.set()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="shuffle-vision"
        )
        self._future: Future[Scene] | None = None

    def close(self) -> None:
        self._motion_done.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._cap.release()

    def capture(self) -> Scene:
        frame = grab_frame(self._cap)
        return capture_scene(self._calib, frame)

    def begin_motion(self) -> None:
        """Discard buffered frames while the arm moves."""
        self._motion_done.clear()
        if self._future is None or self._future.done():
            self._future = self._executor.submit(self._run)

    def end_motion(self) -> None:
        """Arm move finished -- capture a fresh frame on the worker thread."""
        self._motion_done.set()

    def take(self) -> Scene:
        if self._future is None:
            return self.capture()
        return self._future.result()

    def _run(self) -> Scene:
        while not self._motion_done.wait(timeout=0.05):
            self._cap.grab()
        if self._settle_s > 0:
            time.sleep(self._settle_s)
        return self.capture()


def _print_scene(scene: Scene) -> None:
    print(f"Scene: {scene.summary_line()}")
    for line in scene.cube_lines():
        print(line)


def _print_action(action: Action) -> None:
    print(f"Action: {action.kind} -- {action.reason}")


def _action_targets(
    action: Action,
) -> tuple[CubeDetection, float, float] | None:
    """Pick cube + place coords from one planned action, or None to wait."""
    if action.kind != "pick" or action.cube is None:
        return None
    if action.place_x is None or action.place_y is None:
        return None
    return (action.cube, float(action.place_x), float(action.place_y))


def _lookahead_action(
    scene: Scene, first_action: Action, avoid_xy: tuple[float, float] | None
) -> Action | None:
    """A second, independent move already visible in the same capture as
    `first_action`, or None. Lets the caller skip the capture+settle pause
    between two moves planned from the same frame."""
    if first_action.kind != "pick":
        return None
    exclude_slot = (
        (first_action.place_x, first_action.place_y)
        if first_action.place_kind == "to_slot"
        else None
    )
    candidate = plan_shuffle(
        scene,
        avoid_xy=avoid_xy,
        exclude_cube=first_action.cube,
        exclude_marker_id=first_action.place_marker_id,
        exclude_slot=exclude_slot,
    )
    return candidate if candidate.kind == "pick" else None


def _plan_from_capture(
    prefetch: _VisionPrefetch,
    client: Mt4Client,
    calib: Calibration,
    avoid_xy: tuple[float, float] | None = None,
) -> tuple[Scene, Action]:
    # Closer camera: forearm over a marker hides ArUco tags → every marker
    # becomes "unknown" and the planner wedges with free_markers=0. Clear
    # the view before every planning capture.
    try:
        retreat_for_camera(client, calib)
    except Mt4ClientError as exc:
        print(f"camera retreat failed: {exc}")
    scene = prefetch.capture()
    return scene, plan_shuffle(scene, avoid_xy=avoid_xy)


def _plan_after_move(
    prefetch: _VisionPrefetch,
    client: Mt4Client,
    calib: Calibration,
    *,
    pick_x: float,
    pick_y: float,
    pick_color: str,
    place_x: float,
    place_y: float,
) -> tuple[Scene, Action]:
    scene = prefetch.take()
    verdict = verify_pick_place(
        scene,
        pick_x=pick_x,
        pick_y=pick_y,
        pick_color=pick_color,
        place_x=place_x,
        place_y=place_y,
    )
    attempts = 0
    while verdict != "placed" and attempts < POST_MOVE_RECHECK_ATTEMPTS:
        attempts += 1
        time.sleep(POST_MOVE_RECHECK_DELAY_S)
        scene = prefetch.capture()
        verdict = verify_pick_place(
            scene,
            pick_x=pick_x,
            pick_y=pick_y,
            pick_color=pick_color,
            place_x=place_x,
            place_y=place_y,
        )
    print(f"Post-move check: {verdict} (after {attempts} recheck(s))")
    return _plan_from_capture(prefetch, client, calib, avoid_xy=(place_x, place_y))


def _check_home_request(
    watcher: _HomeKeyWatcher,
    client: Mt4Client,
    prefetch: _VisionPrefetch,
    calib: Calibration,
    avoid_xy: tuple[float, float] | None = None,
) -> tuple[Scene, Action] | None:
    if watcher.consume():
        return _run_home(client, prefetch, calib, avoid_xy)
    return None


def _sleep_or_home(
    seconds: float,
    watcher: _HomeKeyWatcher,
    client: Mt4Client,
    prefetch: _VisionPrefetch,
    calib: Calibration,
    avoid_xy: tuple[float, float] | None = None,
) -> tuple[Scene, Action] | None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if watcher.consume():
            return _run_home(client, prefetch, calib, avoid_xy)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    return _check_home_request(watcher, client, prefetch, calib, avoid_xy)


def _run_home(
    client: Mt4Client,
    prefetch: _VisionPrefetch,
    calib: Calibration,
    avoid_xy: tuple[float, float] | None = None,
) -> tuple[Scene, Action]:
    print("Homing (H)...")
    client.clear_interrupt()
    try:
        client.stop()
    except Mt4ClientError:
        pass
    prefetch.end_motion()
    home_arm(client)
    print("Home ok")
    return _plan_from_capture(prefetch, client, calib, avoid_xy)


def _home_was_requested(watcher: _HomeKeyWatcher, exc: Mt4ClientError) -> bool:
    return watcher.consume() or "interrupted" in str(exc)


def run_shuffle_loop(
    client: Mt4Client,
    calib: Calibration,
    *,
    camera: int,
    pause_s: float = 2.0,
    retry_s: float = 5.0,
) -> None:
    """Home if needed, then shuffle cubes until interrupted."""
    if client.get_status().homed:
        print("Already homed")
    else:
        print("Homing...")
        home_arm(client)
        print("Home ok")

    watcher = _HomeKeyWatcher(client)
    watcher.start()
    prefetch = _VisionPrefetch(calib, camera, settle_s=pause_s)
    # Destination of the last completed move -- passed to plan_shuffle so it
    # doesn't move the same cube twice in a row when another is pickable.
    last_place: tuple[float, float] | None = None
    try:
        scene, action = _plan_from_capture(prefetch, client, calib, last_place)
        while True:
            refreshed = _check_home_request(watcher, client, prefetch, calib, last_place)
            if refreshed is not None:
                scene, action = refreshed
                continue

            _print_scene(scene)
            _print_action(action)

            targets = _action_targets(action)
            if targets is None:
                print(f"Waiting {retry_s:.0f}s for a clearer scene")
                refreshed = _sleep_or_home(
                    retry_s, watcher, client, prefetch, calib, last_place
                )
                if refreshed is not None:
                    scene, action = refreshed
                else:
                    scene, action = _plan_from_capture(
                        prefetch, client, calib, last_place
                    )
                continue

            moves = [targets]
            lookahead = _lookahead_action(scene, action, last_place)
            if lookahead is not None:
                lookahead_targets = _action_targets(lookahead)
                assert lookahead_targets is not None
                moves.append(lookahead_targets)
                print(
                    f"Lookahead: {lookahead.reason} -- also visible in this "
                    f"capture, chaining it in (no capture in between)"
                )

            # One continuous motion span across all queued moves: plan every
            # transfer's legs up front and send them as one (station-aware
            # chunked) queue, so chained lookahead has no stop/settle between
            # moves. Capture+verify still runs once after the last move.
            executed: list[tuple[float, float, str, float, float]] = []
            failed_exc: Mt4ClientError | None = None
            prefetch.begin_motion()
            try:
                ensure_homed(client)
                all_legs: list = []
                start: tuple[float, float, float] | None = None
                # Wrist the plan will have left behind, carried across chained
                # transfers -- the live TCP is the wrong angle to minimize
                # travel from once the first transfer is planned but unsent.
                wrist: float | None = None
                planned: list[tuple[float, float, str, float, float]] = []
                for cube, mplace_x, mplace_y in moves:
                    yaw = (
                        f" yaw={cube.yaw_deg:.0f}°"
                        if cube.yaw_deg is not None
                        else ""
                    )
                    print(
                        f"Pick-and-place: {cube.color} "
                        f"({cube.x:.0f},{cube.y:.0f}){yaw} "
                        f"-> ({mplace_x:.0f},{mplace_y:.0f})"
                    )
                    src = Grasp(
                        float(cube.x), float(cube.y),
                        yaw_deg=(
                            cube.yaw_deg
                            if getattr(calib, "face_align_picks", True)
                            else None
                        ),
                    )
                    # square_place, not a bare Grasp: the cube lands square to
                    # the world axes, which is what place() did before shuffle
                    # moved onto the queued transfer.
                    dst = square_place(mplace_x, mplace_y)
                    legs, start, pick_j4, place_j4 = plan_transfer_legs(
                        client, calib, src, dst, start=start, current_j4=wrist,
                    )
                    if place_j4 is not None:
                        wrist = place_j4
                    elif pick_j4 is not None:
                        wrist = pick_j4
                    all_legs.extend(legs)
                    planned.append(
                        (
                            float(cube.x),
                            float(cube.y),
                            cube.color,
                            mplace_x,
                            mplace_y,
                        )
                    )
                send_legs(
                    client, all_legs,
                    step="shuffle" if len(moves) == 1 else "shuffle lookahead",
                )
                executed = planned
            except Mt4ClientError as exc:
                failed_exc = exc
            prefetch.end_motion()

            if executed:
                last_place = (executed[-1][3], executed[-1][4])

            if failed_exc is not None:
                if _home_was_requested(watcher, failed_exc):
                    scene, action = _run_home(client, prefetch, calib, last_place)
                    continue
                print(f"Move failed: {failed_exc} -- retrying next cycle")
                refreshed = _sleep_or_home(
                    retry_s, watcher, client, prefetch, calib, last_place
                )
                if refreshed is not None:
                    scene, action = refreshed
                else:
                    prefetch.take()
                    scene, action = _plan_from_capture(
                        prefetch, client, calib, last_place
                    )
                continue

            last_pick_x, last_pick_y, last_color, last_place_x, last_place_y = executed[-1]
            scene, action = _plan_after_move(
                prefetch,
                client,
                calib,
                pick_x=last_pick_x,
                pick_y=last_pick_y,
                pick_color=last_color,
                place_x=last_place_x,
                place_y=last_place_y,
            )
    finally:
        watcher.close()
        prefetch.close()
