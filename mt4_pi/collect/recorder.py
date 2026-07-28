"""Per-episode camera sampling and trace writer for autonomous collection.

Threading: a single cv2.VideoCapture is not safe to read from two threads
at once, and mt4_vision.pickplace.pick_cube()/place() never touch the
camera -- so EpisodeCamera hands the one open capture to a background
thread for exactly one pick+place window (mirrors the begin_motion/
end_motion split in mt4_vision.shuffle._VisionPrefetch) while the main
thread owns the one serial client and is the only caller of
Mt4Client.get_status(). Never call capture_scene_frame() while a recording
is active, or vice versa -- collect_shuffle.py's loop already respects this
(plan -> begin_recording -> pick/place -> end_recording -> verify).

Per-tick state comes from REAL commanded waypoints, not a guess:
mt4_vision.pickplace.pick()/place() accept an ``on_waypoint`` callback (see
mt4_vision.pickplace.MoveEvent) fired after every completed move/grip, on
the same main thread that calls them. WaypointLog records each one, seeded
from the pre-motion Mt4Status so a frame captured before the first waypoint
still resolves to a real measured pose. A tick's state is a linear
interpolation, by real elapsed time, between the pair of waypoints
bracketing it -- not zero-order-hold (which would freeze a tick for a whole
leg's duration and then jump when the leg completes) and not a single
interpolation across the whole episode (which ignores that _travel and
_approach run at very different speeds over very different distances, so
leg durations vary a lot). See WaypointLog.state_at for what this still
approximates. Every episode's meta.json carries a ``state_source`` field
(``"commanded_linear_interp"``) so a consumer never has to guess which
regime a given batch of data came from.

Clock: everything timed here uses time.monotonic(), never time.time().
On this host time.time() is GetSystemTimeAsFileTime -- 15.6ms nominal
resolution (a quarter of a 15Hz tick) and, worse, adjustable=True, so a
single NTP step during an unattended overnight run would make the waypoint
log non-monotonic and silently corrupt every interpolation after it.
meta.json still records one wall-clock ``started_at`` anchor for absolute
time; all arithmetic is monotonic-relative to it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN
from mt4_jog.status import Mt4Status, TcpPose
from mt4_vision.camera import grab_frame, open_camera
from mt4_vision.pickplace import MoveEvent

DEFAULT_HZ = 15.0  # matches the DROID reference rate the rest of mt4_pi assumes


class EpisodeCamera:
    """Owns the one camera handle for a collection run."""

    def __init__(self, index: int) -> None:
        self._cap = open_camera(index)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ticks: list[tuple[float, str]] = []
        self._frames_dir: Path | None = None
        self._recording = False

    @property
    def is_recording(self) -> bool:
        """Whether the background thread currently owns the capture. Lets an
        error handler unwind a cycle that died mid-motion without having to
        guess whether end_recording() is legal to call."""
        return self._recording

    def capture_scene_frame(self) -> np.ndarray:
        if self._recording:
            raise RuntimeError("camera is owned by the recording thread")
        return grab_frame(self._cap)

    def begin_recording(self, frames_dir: Path, *, hz: float = DEFAULT_HZ) -> None:
        if self._recording:
            raise RuntimeError("already recording")
        frames_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir = frames_dir
        self._ticks = []
        self._stop.clear()
        self._recording = True
        self._thread = threading.Thread(
            target=self._run, args=(hz,), daemon=True, name="episode-camera"
        )
        self._thread.start()

    def end_recording(self) -> list[tuple[float, str]]:
        if not self._recording:
            raise RuntimeError("not recording")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._recording = False
        return self._ticks

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()

    def _run(self, hz: float) -> None:
        # .read() on a continuously-drained capture blocks ~1 native frame
        # (commonly ~33ms @30fps) -- that alone paces the loop; we just
        # subsample down to `hz` and skip the write on the frames we drop.
        interval = 1.0 / hz
        idx = 0
        next_due = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            # Stamp the instant read() handed us the frame, BEFORE the JPEG
            # encode below: imwrite of a 720p frame costs milliseconds that
            # vary with image content, and folding that into the timestamp
            # biases every tick's interpolated pose forward along the
            # trajectory. (A fixed residual remains -- the frame was exposed
            # roughly one native frame period before read() returned -- but
            # that is a constant offset, not content-dependent jitter.)
            now = time.monotonic()
            if not ok or frame is None:
                continue
            if now < next_due:
                continue
            assert self._frames_dir is not None
            name = f"{idx:06d}.jpg"
            # Write first, append second: a tick must never reference a
            # frame file that isn't on disk.
            cv2.imwrite(str(self._frames_dir / name), frame)
            self._ticks.append((now, name))
            idx += 1
            # Re-phase from the frame we just took. `next_due + interval`
            # would, after one late frame, leave next_due in the past and
            # take the following frames back-to-back to "catch up".
            next_due = now + interval


def _gripper_fraction(servo_position: float) -> float:
    """Firmware gripper servo units (GRIPPER_S_OPEN..GRIPPER_S_CLOSED) -> [0, 1].

    Same formula as mt4_pi.observation._gripper_fraction; duplicated rather
    than imported since that one is private to the DROID observation shape
    and this module's state representation is MT4-native, not DROID-padded.
    """
    span = GRIPPER_S_CLOSED - GRIPPER_S_OPEN
    return max(0.0, min(1.0, (servo_position - GRIPPER_S_OPEN) / span))


class WaypointLog:
    """Time-weighted trace of the TCP pose actually commanded during an
    episode, built from real MoveEvents and interpolated per-leg by their
    real measured duration -- see module docstring. Only ever touched from
    the main thread (on_waypoint fires synchronously inside
    pick_cube()/place()), so no locking.
    """

    def __init__(self, seed_tcp: TcpPose) -> None:
        self._current = {
            "x": seed_tcp.x, "y": seed_tcp.y, "z": seed_tcp.z,
            "j4": seed_tcp.j4, "grip": _gripper_fraction(seed_tcp.grip),
        }
        self._log: list[tuple[float, dict[str, float]]] = [
            (time.monotonic(), dict(self._current))
        ]
        self._n_events = 0

    def on_waypoint(self, event: MoveEvent) -> None:
        """Pass this bound method as pick()/pick_cube()/place()'s
        on_waypoint= to record their real commanded waypoints."""
        # Two samples per event, not one. MoveEvent.started_at is when the
        # blocking call was dispatched; everything between the previous
        # waypoint and that instant is dead time -- serial round trips for
        # ensure_homed/get_tcp, j4 resolution, planning -- with the arm
        # provably stationary. Holding the previous pose across it keeps
        # state_at from ramping through a stall: the pick->place handover
        # alone spends ~0.5s there, which at 15Hz is ~7 frames that would
        # otherwise be labelled tens of mm along a move that hadn't begun.
        started = event.started_at
        if started is not None and started > self._log[-1][0]:
            self._log.append((started, dict(self._current)))
        if event.x is not None:
            self._current["x"] = event.x
        if event.y is not None:
            self._current["y"] = event.y
        if event.z is not None:
            self._current["z"] = event.z
        if event.j4 is not None:
            self._current["j4"] = event.j4
        if event.grip is not None:
            self._current["grip"] = _gripper_fraction(event.grip)
        # event.ended_at is the leg's real completion time when several
        # legs are reported after one queued move_path() call (see
        # MoveEvent's docstring) -- those legs all return together, so
        # time.monotonic() here would stamp every leg of a multi-leg
        # transit at ~the same instant (the moment the whole call
        # returned) and collapse the route into a single late jump.
        ended = event.ended_at if event.ended_at is not None else time.monotonic()
        self._log.append((ended, dict(self._current)))
        self._n_events += 1

    def state_at(self, t: float) -> dict[str, float]:
        """Pose at monotonic time t: linear interpolation, by real elapsed
        time, between the pair of logged samples bracketing t.

        _travel (calib.travel_speed_us) and _approach (calib.approach_speed_us,
        the firmware's ramp-off slow final descent) run at very different
        speeds over very different distances, so leg durations vary a lot --
        a zero-order hold ("state = whatever was last commanded") would
        freeze a tick for a whole leg's duration and then jump
        discontinuously, out of step with the camera frames actually
        showing continuous motion. Interpolating by each leg's own measured
        span (not an assumed constant speed) tracks that correctly
        regardless of which leg a tick lands in, and on_waypoint's paired
        dead-time samples keep the ramp confined to the leg rather than
        letting it start during the preceding serial stall.

        It still approximates constant velocity *within* a leg -- the
        firmware's accel-ramp/cruise/decel shape isn't modeled, which would
        require mirroring its ramp constants from firmware. Not worth it
        for legs this short; this removes the dominant errors (whole-leg
        staleness, dead-time smear), not the residual one (in-leg ramp
        shape).

        Returns a fresh dict every call -- the samples in _log are the
        recorder's own state and must not be handed out by reference.
        """
        if t <= self._log[0][0]:
            return dict(self._log[0][1])
        for i in range(len(self._log) - 1):
            t0, s0 = self._log[i]
            t1, s1 = self._log[i + 1]
            if t <= t1:
                frac = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
                return {k: s0[k] + (s1[k] - s0[k]) * frac for k in s0}
        return dict(self._log[-1][1])

    def waypoint_count(self) -> int:
        """Real MoveEvents seen -- not len(_log), which also holds the seed
        and one dead-time hold sample per event."""
        return self._n_events


@dataclass
class OpenEpisode:
    dir: Path
    frames_dir: Path
    prompt: str
    started_wall: float  # absolute anchor for meta.json only
    started_mono: float  # base for every tick offset / duration
    pre_status: Mt4Status
    waypoints: WaypointLog = field(repr=False)

    @property
    def on_waypoint(self):
        return self.waypoints.on_waypoint


def _next_episode_id(episodes_dir: Path) -> int:
    ids = []
    for p in episodes_dir.glob("ep_*"):
        suffix = p.name.split("_", 1)[-1]
        if p.is_dir() and suffix.isdigit():
            ids.append(int(suffix))
    return max(ids, default=-1) + 1


class EpisodeRecorder:
    """Writes one directory per episode: meta.json + ticks.jsonl + frames/."""

    def __init__(self, root: Path, *, camera_index: int, calib_path: str) -> None:
        self.root = root
        self.episodes_dir = root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self._next_id = _next_episode_id(self.episodes_dir)
        with (root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "started_at": time.time(),
                "camera_index": camera_index,
                "calib_path": calib_path,
            }) + "\n")

    def start(self, prompt: str, pre_status: Mt4Status) -> OpenEpisode | None:
        """None when pre_status has no TCP pose -- caller should re-read
        status and try again rather than open an episode with unknown
        starting state."""
        if pre_status.tcp is None:
            return None
        ep_dir = self.episodes_dir / f"ep_{self._next_id:06d}"
        self._next_id += 1
        ep_dir.mkdir(parents=True, exist_ok=False)
        return OpenEpisode(
            dir=ep_dir,
            frames_dir=ep_dir / "frames",
            prompt=prompt,
            started_wall=time.time(),
            started_mono=time.monotonic(),
            pre_status=pre_status,
            waypoints=WaypointLog(pre_status.tcp),
        )

    def finish(
        self,
        episode: OpenEpisode,
        ticks: list[tuple[float, str]],
        post_status: Mt4Status | None,
        *,
        success: bool,
        verdict: str,
        pick: dict[str, object],
        place: dict[str, object],
        error: str | None,
    ) -> Path:
        ended_mono = time.monotonic()
        lines = []
        for t, frame_name in ticks:
            state = episode.waypoints.state_at(t)
            lines.append(json.dumps({
                "t": round(t - episode.started_mono, 4),
                "frame": f"frames/{frame_name}",
                "state": {k: round(v, 3) for k, v in state.items()},
            }))
        # Trailing newline: ticks.jsonl is line-delimited, and a reader that
        # appends or concatenates must not fuse the last record with a first.
        (episode.dir / "ticks.jsonl").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )

        final_state = episode.waypoints.state_at(ended_mono)
        meta = {
            "prompt": episode.prompt,
            "success": success,
            "verdict": verdict,
            "error": error,
            "pick": pick,
            "place": place,
            "pre_state": episode.pre_status.tcp.as_dict(),
            # None when the serial link dropped before post-move status could
            # be read -- the episode is still written (its frames and
            # waypoints are real), just without a measured end pose.
            "post_state": (
                post_status.tcp.as_dict()
                if post_status is not None and post_status.tcp
                else None
            ),
            "final_commanded_state": {k: round(v, 3) for k, v in final_state.items()},
            # Names the regime, not just the source: ticks are a linear
            # interpolation between real MoveEvents, with dead time held.
            # See the module docstring.
            "state_source": "commanded_linear_interp",
            "n_ticks": len(ticks),
            "n_waypoints": episode.waypoints.waypoint_count(),
            "started_at": episode.started_wall,
            "duration_s": round(ended_mono - episode.started_mono, 3),
        }
        (episode.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return episode.dir
