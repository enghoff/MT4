"""Stage 1 of the ACT pipeline: raw collector traces -> a compact, portable export.

    python -m mt4_pi.act.export_episodes --out data/act_export

Runs locally, where the raw data and `mt4_jog.kinematics` live. Produces one
H.264 clip plus one `.npz` of states/actions per episode, and a `manifest.json`
describing the whole set.

Why a separate export step instead of writing the LeRobot dataset directly
-------------------------------------------------------------------------
The pi05 converter writes LeRobot directly, which forced the whole machine to
run wherever `lerobot` was importable at exactly the pinned commit. That bit
us repeatedly. Here the split falls along the real seam:

  * This stage needs the raw 10 GB of JPEGs, `mt4_jog.kinematics` for IK, and
    ffmpeg -- all of which exist only on the workstation.
  * `pack_lerobot.py` needs `lerobot` at whatever version the trainer uses --
    which exists only on the GPU box.

Neither stage needs the other's dependencies, and the artefact between them is
~25x smaller than the raw frames, so it is also the thing worth uploading.

Frames are re-encoded by piping the stored JPEG bytes straight into ffmpeg
(`-f image2pipe`), so Python never decodes an image. The scale filter does the
1280x720 -> 640x360 downsample inside ffmpeg too.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mt4_jog.kinematics import (
    HOME_J1_DEG,
    HOME_J2_DEG,
    HOME_J3_DEG,
    HOME_J4_DEG,
    JointAnglesDeg,
    ik_position,
)
from mt4_pi.act.schema import (
    ACTION_COLUMNS,
    CAMERA_KEY,
    FPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    STATE_COLUMNS,
    build_state,
)

DEFAULT_SOURCES = (Path("data/pi_demos"), Path("data/pi_stack_demos"))
DEFAULT_OUT = Path("data/act_export")

#: Bumped whenever the on-disk export layout or the state/action convention
#: changes, so `pack_lerobot.py` can refuse a stale export rather than train on
#: silently mismatched columns.
EXPORT_VERSION = 1

#: A tick counts as static when neither the commanded joints nor the gripper
#: move appreciably. 0.05 deg is well under the ~0.35 deg the coarsest joint
#: moves in one step at survey speed, and the gripper is normalised [0, 1].
STATIC_JOINT_TOL_DEG = 0.05
STATIC_GRIP_TOL = 0.002


@dataclass
class EpisodeRecord:
    """One row of `manifest.json["episodes"]`."""

    episode_id: str
    source: str
    src_dir: str
    prompt: str
    color: str
    place_kind: str
    goal_xy_mm: list[float]
    n_frames: int
    n_static: int
    video: str
    arrays: str


def _load_episode(ep_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    ticks = [
        json.loads(line)
        for line in (ep_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return meta, ticks


def solve_joint_track(ticks: list[dict]) -> list[JointAnglesDeg] | None:
    """IK each recorded Cartesian pose into joint angles, seeding each solve
    with the previous result so the arm stays on one IK branch.

    Mirrors `mt4_pi.collect.convert_to_lerobot._solve_joint_track` -- kept
    separate rather than imported because that module pulls in the pinned old
    `lerobot` and the openpi client, and the whole point of `mt4_pi.act` is
    that it does not depend on the pi0.5 stack.

    j4 is read straight from the log, not re-derived: it is already the
    commanded WORLD-FRAME wrist angle, which is both what the training labels
    should express and what `queue_move` consumes. (Under the firmware's
    ORIENT=hold the raw joint sits at `tcp_j4 - j1`; conflating the two is the
    bug that cost us round 1 -- see `mt4_pi/observation.joint_state_from_status`.)
    """
    near = JointAnglesDeg(HOME_J1_DEG, HOME_J2_DEG, HOME_J3_DEG, HOME_J4_DEG)
    out: list[JointAnglesDeg] = []
    for tick in ticks:
        s = tick["state"]
        sol = ik_position(s["x"], s["y"], s["z"], near=near, hold_orientation=False)
        if sol is None:
            return None
        q = JointAnglesDeg(sol.j1, sol.j2, sol.j3, s["j4"])
        out.append(q)
        near = q
    return out


def _joints_rad(q: JointAnglesDeg) -> np.ndarray:
    return np.array([math.radians(v) for v in (q.j1, q.j2, q.j3, q.j4)], dtype=np.float32)


def build_arrays(meta: dict, ticks: list[dict], joints: list[JointAnglesDeg]) -> dict[str, np.ndarray]:
    """State/action/static arrays for one episode.

    The final tick is dropped: it has no successor, so there is no action to
    label it with. Synthesising a hold-in-place action instead would add one
    explicit "freeze" example per episode, which is the last behaviour we want
    to reinforce given 24% of the corpus is already stationary.
    """
    color = meta["pick"]["color"]
    kind = meta["place"]["kind"]
    goal = (float(meta["place"]["x"]), float(meta["place"]["y"]))

    n = len(ticks) - 1
    state = np.zeros((n, len(STATE_COLUMNS)), dtype=np.float32)
    action = np.zeros((n, len(ACTION_COLUMNS)), dtype=np.float32)
    static = np.zeros(n, dtype=bool)

    for i in range(n):
        grip_t = float(ticks[i]["state"]["grip"])
        grip_next = float(ticks[i + 1]["state"]["grip"])
        state[i] = build_state(_joints_rad(joints[i]), grip_t, color, kind, goal)
        # Absolute target, not a delta -- see schema.py.
        action[i, :4] = _joints_rad(joints[i + 1])
        action[i, 4] = grip_next

        q_t, q_n = joints[i], joints[i + 1]
        moved = max(
            abs(q_n.j1 - q_t.j1), abs(q_n.j2 - q_t.j2), abs(q_n.j3 - q_t.j3), abs(q_n.j4 - q_t.j4)
        )
        static[i] = moved < STATIC_JOINT_TOL_DEG and abs(grip_next - grip_t) < STATIC_GRIP_TOL

    return {"state": state, "action": action, "static": static}


def encode_video(ep_dir: Path, ticks: list[dict], n_frames: int, dest: Path, ffmpeg: str) -> None:
    """Concatenate the episode's stored JPEGs into one H.264 clip.

    `-f image2pipe` lets ffmpeg demux a raw concatenation of JPEG files from
    stdin, so the frames never round-trip through Python. CRF 18 is
    near-transparent; the clip is re-encoded once more by LeRobot at pack
    time, and two generations at this quality are not a meaningful loss for
    policy learning.
    """
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(FPS), "-i", "-",
        "-vf", f"scale={IMAGE_WIDTH}:{IMAGE_HEIGHT}",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(dest),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for tick in ticks[:n_frames]:
            proc.stdin.write((ep_dir / tick["frame"]).read_bytes())
        proc.stdin.close()
    except BrokenPipeError:  # ffmpeg died early; the stderr below explains why
        pass
    stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {ep_dir.name}: {stderr.strip()}")


def export(
    sources: list[Path], out_root: Path, *, limit: int | None = None, ffmpeg: str = "ffmpeg"
) -> Path:
    if shutil.which(ffmpeg) is None:
        raise SystemExit(f"ffmpeg not found on PATH as {ffmpeg!r}; install it or pass --ffmpeg")

    ep_out = out_root / "episodes"
    ep_out.mkdir(parents=True, exist_ok=True)

    records: list[EpisodeRecord] = []
    n_skipped = 0
    for source in sources:
        episodes_root = source / "episodes"
        if not episodes_root.is_dir():
            print(f"  {source}: no episodes/ directory, skipping source")
            continue
        for ep_dir in sorted(p for p in episodes_root.glob("ep_*") if p.is_dir()):
            if limit is not None and len(records) >= limit:
                break
            # A collector killed mid-cycle leaves frames on disk with no
            # meta.json. The collector cleans these up on its own error
            # paths, but not if the process is hard-killed -- and one such
            # directory should not take down an export of 200 good episodes.
            if not (ep_dir / "meta.json").is_file():
                print(f"  {source.name}/{ep_dir.name}: no meta.json (partial), skipping")
                n_skipped += 1
                continue
            meta, ticks = _load_episode(ep_dir)
            # Only successful demonstrations: a failed grasp is not a
            # demonstration of the behaviour we want imitated.
            if not meta.get("success"):
                continue
            if len(ticks) < 2:
                print(f"  {source.name}/{ep_dir.name}: fewer than 2 ticks, skipping")
                n_skipped += 1
                continue
            joints = solve_joint_track(ticks)
            if joints is None:
                print(f"  {source.name}/{ep_dir.name}: IK failed on a recorded pose, skipping")
                n_skipped += 1
                continue

            episode_id = f"{source.name}__{ep_dir.name}"
            arrays = build_arrays(meta, ticks, joints)
            n_frames = len(arrays["state"])

            video = ep_out / f"{episode_id}.mp4"
            encode_video(ep_dir, ticks, n_frames, video, ffmpeg)
            npz = ep_out / f"{episode_id}.npz"
            np.savez_compressed(npz, **arrays)

            records.append(
                EpisodeRecord(
                    episode_id=episode_id,
                    source=source.name,
                    src_dir=str(ep_dir).replace("\\", "/"),
                    prompt=meta["prompt"],
                    color=meta["pick"]["color"],
                    place_kind=meta["place"]["kind"],
                    goal_xy_mm=[float(meta["place"]["x"]), float(meta["place"]["y"])],
                    n_frames=n_frames,
                    n_static=int(arrays["static"].sum()),
                    video=f"episodes/{video.name}",
                    arrays=f"episodes/{npz.name}",
                )
            )
            if len(records) % 25 == 0:
                print(f"  ... {len(records)} episodes exported")

    manifest = {
        "export_version": EXPORT_VERSION,
        "fps": FPS,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "camera_key": CAMERA_KEY,
        "state_columns": list(STATE_COLUMNS),
        "action_columns": list(ACTION_COLUMNS),
        "robot_type": "mt4",
        "episodes": [asdict(r) for r in records],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_frames = sum(r.n_frames for r in records)
    total_static = sum(r.n_static for r in records)
    size_mb = sum(p.stat().st_size for p in ep_out.iterdir()) / 1e6
    print(
        f"\nExported {len(records)} episodes ({n_skipped} skipped), {total_frames} frames, "
        f"{total_static} static ({total_static / max(1, total_frames):.1%}), {size_mb:.0f} MB "
        f"-> {out_root}"
    )
    return out_root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source", action="append", dest="sources", default=None,
        help="raw collection root; repeat to merge (default: data/pi_demos, data/pi_stack_demos)",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="export destination root")
    ap.add_argument("--limit", type=int, default=None, help="stop after N episodes (smoke test)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    args = ap.parse_args(argv)

    sources = [Path(s) for s in args.sources] if args.sources else [p for p in DEFAULT_SOURCES if p.exists()]
    if not sources:
        raise SystemExit("no source directories found; pass --source")
    export(sources, Path(args.out), limit=args.limit, ffmpeg=args.ffmpeg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
