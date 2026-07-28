"""Convert mt4_pi.collect's raw per-episode traces (see recorder.py) into a
LeRobot-format dataset matching pi05_droid's raw column schema, so openpi's
existing LeRobotDROIDDataConfig repack/transform pipeline (RepackTransform ->
DroidInputs) consumes it unmodified -- see
src/openpi/training/config.py:LeRobotDROIDDataConfig and
src/openpi/policies/droid_policy.py:DroidInputs in Physical-Intelligence/openpi.

Only successful ("placed") episodes are converted -- a failed grasp is not a
demonstration of the behavior we want the policy to imitate.

Unit choice: joint_position/actions are radians (math.radians of the MT4
kinematics module's native degrees), not degrees. adapter.integrate_joint_target
clips the model's raw action to [-1, 1] before integrating, which only makes
sense as a per-joint radians/s safety cap (matching real Franka/DROID joint
speeds) -- treating our labels as degrees/s would make that clip reject nearly
all real motion. mt4_jog.kinematics.JointAnglesDeg itself stays degrees-native
throughout; conversion to radians happens only at the boundary, in
_pad_joint_position_rad / the per-tick action loop below.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# Pinned to the exact commit openpi vendors (its pyproject.toml git dependency,
# not the PyPI `lerobot` package): openpi's own lerobot refuses to load a
# dataset written by a newer lerobot with a ForwardCompatibilityError, so the
# writer and reader must be the same commit, not just "some recent version".
# pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" --no-deps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from mt4_jog.kinematics import (
    HOME_J1_DEG,
    HOME_J2_DEG,
    HOME_J3_DEG,
    HOME_J4_DEG,
    JointAnglesDeg,
    ik_position,
)
from mt4_pi.adapter import DT_S
from mt4_pi.observation import DROID_NUM_JOINTS, IMAGE_SIZE, MT4_NUM_JOINTS, _prep_image

DEFAULT_OUT_ROOT = Path("data/lerobot_dataset")
FPS = 15  # matches recorder.DEFAULT_HZ / adapter.DT_S


def _episode_dirs(episodes_root: Path) -> list[Path]:
    return sorted(p for p in episodes_root.glob("ep_*") if p.is_dir())


def _load_episode(ep_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    ticks = [
        json.loads(line)
        for line in (ep_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return meta, ticks


def _solve_joint_track(ticks: list[dict]) -> list[JointAnglesDeg] | None:
    """IK each tick's recorded Cartesian pose into joint angles, chaining each
    solve's result as the next tick's branch-selection seed for continuity
    (see mt4_jog.kinematics.ik_q2_q3's nearest-branch doc). j4 is taken
    directly from the log rather than re-derived, since it's already the real
    commanded world-frame wrist angle, not something IK needs to guess.
    """
    near = JointAnglesDeg(HOME_J1_DEG, HOME_J2_DEG, HOME_J3_DEG, HOME_J4_DEG)
    out = []
    for tick in ticks:
        s = tick["state"]
        sol = ik_position(s["x"], s["y"], s["z"], near=near, hold_orientation=False)
        if sol is None:
            return None
        q = JointAnglesDeg(sol.j1, sol.j2, sol.j3, s["j4"])
        out.append(q)
        near = q
    return out


def _pad_joint_position_rad(q: JointAnglesDeg) -> np.ndarray:
    joints = np.zeros(DROID_NUM_JOINTS, dtype=np.float32)
    joints[:MT4_NUM_JOINTS] = [math.radians(v) for v in (q.j1, q.j2, q.j3, q.j4)]
    return joints


def _build_features() -> dict:
    image_spec = {"dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
    return {
        "exterior_image_1_left": image_spec,
        # DROID's RepackTransform plucks this key even though DroidInputs
        # never reads it for PI0/PI05 -- keep it present (zeroed) so the
        # repack step doesn't KeyError on a genuinely single-camera rig.
        "exterior_image_2_left": image_spec,
        "wrist_image_left": image_spec,  # zeroed -- MT4 has no wrist camera, same stand-in as observation.py
        "joint_position": {"dtype": "float32", "shape": (DROID_NUM_JOINTS,), "names": ["joint_position"]},
        "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
    }


def convert(
    data_dirs: list[Path],
    out_root: Path,
    repo_id: str,
    *,
    push_to_hub: bool = False,
) -> Path:
    """Merge one or more raw collection roots (e.g. collect_shuffle.py's
    data/pi_demos and collect_stack.py's data/pi_stack_demos) into a single
    LeRobot dataset. Both write the same meta.json/ticks.jsonl schema and
    differ only in prompt/place.kind, which the model is meant to condition
    on -- see mt4_pi/collect/prompts.py's _TO_MARKER/_TO_SLOT vs
    _STACK_BASE/_STACK_ON_TOP -- so one combined corpus is the intended
    training set, not per-task datasets.
    """
    output_path = out_root / repo_id
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=_build_features(),
        root=output_path,
        robot_type="mt4",
        image_writer_threads=10,
        image_writer_processes=5,
    )

    zero_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    n_episodes = 0
    n_skipped = 0
    for data_dir in data_dirs:
        episodes_root = data_dir / "episodes"
        for ep_dir in _episode_dirs(episodes_root):
            meta, ticks = _load_episode(ep_dir)
            if not meta.get("success"):
                continue
            if len(ticks) < 2:
                print(f"  {data_dir.name}/{ep_dir.name}: fewer than 2 ticks, skipping")
                n_skipped += 1
                continue

            joints = _solve_joint_track(ticks)
            if joints is None:
                print(f"  {data_dir.name}/{ep_dir.name}: IK failed on a recorded waypoint, skipping")
                n_skipped += 1
                continue

            prompt = meta["prompt"]
            for i in range(len(ticks) - 1):  # last tick has no outgoing action -- dropped
                q_t, q_next = joints[i], joints[i + 1]
                grip_t = float(ticks[i]["state"]["grip"])
                grip_next = float(ticks[i + 1]["state"]["grip"])

                dq_rad_s = np.array(
                    [math.radians(q_next.j1 - q_t.j1), math.radians(q_next.j2 - q_t.j2),
                     math.radians(q_next.j3 - q_t.j3), math.radians(q_next.j4 - q_t.j4)],
                    dtype=np.float32,
                ) / DT_S
                actions = np.zeros(8, dtype=np.float32)
                actions[:MT4_NUM_JOINTS] = dq_rad_s
                actions[7] = grip_next

                frame_bgr = cv2.imread(str(ep_dir / ticks[i]["frame"]))
                image_rgb = _prep_image(frame_bgr)  # same resize_with_pad/uint8 prep used at inference, for train/serve parity

                dataset.add_frame({
                    "exterior_image_1_left": image_rgb,
                    "exterior_image_2_left": zero_image,
                    "wrist_image_left": zero_image,
                    "joint_position": _pad_joint_position_rad(q_t),
                    "gripper_position": np.asarray([grip_t], dtype=np.float32),
                    "actions": actions,
                    "task": prompt,
                })
            dataset.save_episode()
            n_episodes += 1

    print(f"Converted {n_episodes} episodes ({n_skipped} skipped) from {len(data_dirs)} source dir(s) -> {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["mt4", "pi05", "cube-shuffle"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", action="append", dest="data_dirs", default=None,
        help="raw collection output root; repeat to merge multiple sources (default: all of data/pi_demos, data/pi_stack_demos that exist)",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="local LeRobot dataset output root")
    parser.add_argument("--repo-id", required=True, help="e.g. your_hf_username/mt4-cube-shuffle")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    if args.data_dirs is None:
        data_dirs = [p for p in (Path("data/pi_demos"), Path("data/pi_stack_demos")) if p.exists()]
    else:
        data_dirs = [Path(d) for d in args.data_dirs]

    convert(data_dirs, Path(args.out_root), args.repo_id, push_to_hub=args.push_to_hub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
