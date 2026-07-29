"""Stage 2 of the ACT pipeline: `data/act_export/` -> a LeRobot dataset.

    python -m mt4_pi.act.pack_lerobot --export data/act_export \
        --repo-id you/mt4-act-v1 --out data/act_lerobot

Runs wherever `lerobot` is installed -- normally the GPU box, right before
training. Depends only on `lerobot`, `numpy` and `cv2`; nothing from
`mt4_jog`, the openpi stack, or the serial layer. `schema.py` is imported for
the column layout but is numpy-only by design.

Version tolerance
-----------------
This repo already has one `lerobot` pinned to the exact commit openpi vendors
(`lerobot.common.datasets`, for the pi0.5 path) and will have another, much
newer one for ACT (`lerobot.datasets`). Rather than pin a third time and
re-litigate it on every upgrade, the import and the two call sites that
actually moved between versions are probed at runtime.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from mt4_pi.act.schema import (
    ACTION_COLUMNS,
    CAMERA_KEY,
    STATE_COLUMNS,
    prune_constant_columns,
)

#: Written into the packed dataset root. Serving reads it to rebuild exactly
#: the state vector the model was trained on -- once constant columns have
#: been pruned the layout is no longer derivable from `schema.py` alone.
SIDECAR_NAME = "mt4_act_meta.json"

FAMILY_SOURCES = {
    "shuffle": ("pi_demos",),
    "stack": ("pi_stack_demos",),
    "all": ("pi_demos", "pi_stack_demos"),
}


def _import_lerobot_dataset():
    """`LeRobotDataset`, from whichever module path this version uses."""
    errors = []
    for module in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
        try:
            return getattr(__import__(module, fromlist=["LeRobotDataset"]), "LeRobotDataset")
        except Exception as exc:  # ImportError, but also transitive failures
            errors.append(f"  {module}: {exc}")
    raise SystemExit("could not import LeRobotDataset from any known path:\n" + "\n".join(errors))


def _create_dataset(cls, **kwargs):
    """`LeRobotDataset.create`, dropping kwargs this version does not accept.

    The image-writer knobs in particular have come and gone; none of them are
    load-bearing for correctness, so silently omitting an unsupported one is
    better than failing the run.
    """
    accepted = set(inspect.signature(cls.create).parameters)
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        print(f"  note: this lerobot's create() does not accept {dropped}; omitting")
    return cls.create(**{k: v for k, v in kwargs.items() if k in accepted})


def _add_frame(dataset, frame: dict, task: str) -> None:
    """`add_frame`, whether this version wants the task inside the frame dict
    or as a keyword argument."""
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        dataset.add_frame({**frame, "task": task})


def _save_episode(dataset, task: str) -> None:
    try:
        dataset.save_episode()
    except TypeError:
        dataset.save_episode(task=task)


def read_video(path: Path, expected: int) -> np.ndarray:
    """Decode an exported clip to (N, H, W, 3) RGB uint8."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != expected:
        raise RuntimeError(f"{path.name}: decoded {len(frames)} frames, manifest says {expected}")
    return np.stack(frames)


def static_keep_mask(static: np.ndarray, max_run: int | None) -> np.ndarray:
    """Keep at most `max_run` frames from each maximal run of static frames.

    24% of the corpus is fully stationary and ~5% is entire chunks of nothing
    -- an artefact of the collector waiting on serial round-trips, not of the
    task. Trimming it is safe here only because actions are ABSOLUTE targets:
    the caller re-points each surviving frame's action at the next surviving
    state (see `retarget_actions`). The same edit on the pi0.5 velocity labels
    would have been silently wrong, since a velocity is defined against a
    specific elapsed time.
    """
    if max_run is None:
        return np.ones(len(static), dtype=bool)
    keep = np.ones(len(static), dtype=bool)
    run = 0
    for i, is_static in enumerate(static):
        run = run + 1 if is_static else 0
        if is_static and run > max_run:
            keep[i] = False
    return keep


def retarget_actions(state: np.ndarray, action: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """After dropping frames, re-point each kept action at the next kept state.

    Absolute targets make this exact: the label for frame i is simply "the
    pose the arm should be in next", and after a deletion the next pose is the
    next survivor. The final kept frame has no successor, so it retains its
    original label.
    """
    idx = np.flatnonzero(keep)
    out = action[keep].copy()
    if len(idx) > 1:
        out[:-1, : len(ACTION_COLUMNS)] = state[idx[1:], : len(ACTION_COLUMNS)]
    return out


def episode_max_jump_deg(state: np.ndarray, action: np.ndarray) -> float:
    """Largest single-step joint discontinuity in an episode's labels, degrees.

    A label is a discontinuity when the commanded pose at t+1 is further from
    the pose at t than the arm could plausibly travel in one 100 ms tick.

    The stack corpus is full of these -- 100% of `stack` episodes and 45% of
    `unstack` ones, median worst-jump 68 deg, up to 171. The cause is the
    carry: a stacking carry goes out as ONE queued firmware path (mq) of
    five or six legs, and the recorder has to split that single measured
    duration back across the legs after the fact. It splits it wrong, so an
    early leg absorbs nearly the whole span (measured on ep_000030: a 27.8mm
    vertical lift was given 12.59s of a 12.69s carry) and the remaining
    400mm of travel lands inside a single 0.09s tick. Shuffle is clean by
    comparison (median 3.6 deg, worst 17.1) because pick()/place() issue
    their moves as separate blocking calls, each individually timed.

    NOT the retreat to camera park, which earlier notes blamed: that runs
    after `end_recording()`, so no tick is ever labelled with it.

    Such a frame cannot be imitated: `mt4_pi.safety` caps a step at 5 deg, so
    the arm would refuse to execute it even if the policy reproduced it
    perfectly. Training on them only teaches the model to hedge.
    """
    return float(np.abs(np.degrees(action[:, :4]) - np.degrees(state[:, :4])).max())


def _features(
    state_dim: int, height: int, width: int, state_names: list[str], *, use_videos: bool = True
) -> dict:
    return {
        CAMERA_KEY: {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(ACTION_COLUMNS),), "names": list(ACTION_COLUMNS)},
    }


def pack(
    export_root: Path,
    out_root: Path,
    repo_id: str,
    *,
    family: str = "all",
    max_static_run: int | None = None,
    max_jump_deg: float | None = None,
    prune_constant: bool = True,
    use_videos: bool = True,
    push_to_hub: bool = False,
    limit: int | None = None,
) -> Path:
    manifest = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["state_columns"] != list(STATE_COLUMNS) or manifest["action_columns"] != list(ACTION_COLUMNS):
        raise SystemExit(
            "export was written with a different column layout than mt4_pi.act.schema defines.\n"
            f"  export: {manifest['state_columns']}\n  schema: {list(STATE_COLUMNS)}\n"
            "Re-run export_episodes.py."
        )

    sources = FAMILY_SOURCES[family]
    episodes = [e for e in manifest["episodes"] if e["source"] in sources]

    n_excluded = 0
    if max_jump_deg is not None:
        kept_eps = []
        for e in episodes:
            arrays = np.load(export_root / e["arrays"])
            if episode_max_jump_deg(arrays["state"], arrays["action"]) <= max_jump_deg:
                kept_eps.append(e)
        n_excluded = len(episodes) - len(kept_eps)
        print(f"  excluding {n_excluded} episode(s) with a label jump > {max_jump_deg} deg")
        episodes = kept_eps

    episodes = episodes[:limit]
    if not episodes:
        raise SystemExit(f"no episodes matched family {family!r}")

    # Two passes: the first decides the column layout (which needs statistics
    # over the whole subset), the second writes. Loading only the small .npz
    # arrays for pass one keeps this cheap -- no video is touched.
    all_state = np.concatenate([np.load(export_root / e["arrays"])["state"] for e in episodes])
    if prune_constant:
        _, kept_columns, dropped = prune_constant_columns(all_state)
        if dropped:
            print(f"  pruning {len(dropped)} constant state column(s): {list(dropped)}")
    else:
        kept_columns, dropped = tuple(STATE_COLUMNS), ()
    keep_idx = [STATE_COLUMNS.index(c) for c in kept_columns]

    output_path = out_root / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    cls = _import_lerobot_dataset()
    dataset = _create_dataset(
        cls,
        repo_id=repo_id,
        fps=manifest["fps"],
        features=_features(
            len(kept_columns),
            manifest["image_height"],
            manifest["image_width"],
            list(kept_columns),
            use_videos=use_videos,
        ),
        root=output_path,
        robot_type=manifest["robot_type"],
        use_videos=use_videos,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    n_frames_in = n_frames_out = 0
    for n, ep in enumerate(episodes, 1):
        arrays = np.load(export_root / ep["arrays"])
        state, action, static = arrays["state"], arrays["action"], arrays["static"]
        images = read_video(export_root / ep["video"], ep["n_frames"])

        keep = static_keep_mask(static, max_static_run)
        action_kept = retarget_actions(state, action, keep)
        state_kept = state[keep][:, keep_idx]
        images_kept = images[keep]
        n_frames_in += len(state)
        n_frames_out += len(state_kept)

        for i in range(len(state_kept)):
            _add_frame(
                dataset,
                {
                    CAMERA_KEY: images_kept[i],
                    "observation.state": state_kept[i].astype(np.float32),
                    "action": action_kept[i].astype(np.float32),
                },
                task=ep["prompt"],
            )
        _save_episode(dataset, ep["prompt"])
        if n % 25 == 0:
            print(f"  ... {n}/{len(episodes)} episodes packed")

    sidecar = {
        "export_version": manifest["export_version"],
        "repo_id": repo_id,
        "family": family,
        "fps": manifest["fps"],
        "camera_key": CAMERA_KEY,
        "image_height": manifest["image_height"],
        "image_width": manifest["image_width"],
        "state_columns": list(kept_columns),
        "dropped_state_columns": list(dropped),
        "action_columns": list(ACTION_COLUMNS),
        "max_static_run": max_static_run,
        "max_jump_deg": max_jump_deg,
        "frame_storage": "video" if use_videos else "image",
        "n_episodes": len(episodes),
        "n_episodes_excluded_by_jump": n_excluded,
        "n_frames": n_frames_out,
    }
    (output_path / SIDECAR_NAME).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    dropped_pct = 1 - n_frames_out / max(1, n_frames_in)
    print(
        f"\nPacked {len(episodes)} episodes, {n_frames_out} frames "
        f"({dropped_pct:.1%} dropped as static), state dim {len(kept_columns)} -> {output_path}"
    )

    if push_to_hub:
        print("pushing to the Hub ...")
        dataset.push_to_hub(tags=["mt4", "act", family], private=True, license="apache-2.0")
        # push_to_hub uploads the dataset's own files; the sidecar is ours, so
        # it needs a separate call or the GPU box cannot rebuild the layout.
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(output_path / SIDECAR_NAME),
            path_in_repo=SIDECAR_NAME,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"pushed {repo_id} (including {SIDECAR_NAME})")

    return output_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="data/act_export", help="export root from export_episodes.py")
    ap.add_argument("--out", default="data/act_lerobot", help="local dataset output root")
    ap.add_argument("--repo-id", required=True, help="e.g. you/mt4-act-v1")
    ap.add_argument("--family", choices=sorted(FAMILY_SOURCES), default="all", help="task subset to include")
    ap.add_argument(
        "--max-static-run", type=int, default=None,
        help="keep at most N frames from each run of stationary frames (default: keep all)",
    )
    ap.add_argument(
        "--max-jump-deg", type=float, default=None,
        help="exclude episodes whose labels contain a single-step joint jump larger than this "
             "(recommended 15.0; drops 88 of 114 stack episodes and 1 of 111 shuffle episodes)",
    )
    ap.add_argument(
        "--frame-storage", choices=("video", "image"), default="video",
        help="how frames are stored. Keep 'video' unless you have a reason not to: LeRobot "
             "encodes AV1 with a keyframe every 2 frames, and random access measures 2.9 ms "
             "(346 frames/s per worker), so decode is nowhere near the bottleneck. 'image' "
             "trades ~10x the disk for removing the video stack from the dependency list.",
    )
    ap.add_argument("--no-prune-constant-state", action="store_true", help="keep zero-variance state columns")
    ap.add_argument("--limit", type=int, default=None, help="stop after N episodes (smoke test)")
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args(argv)

    pack(
        Path(args.export), Path(args.out), args.repo_id,
        family=args.family,
        max_static_run=args.max_static_run,
        max_jump_deg=args.max_jump_deg,
        prune_constant=not args.no_prune_constant_state,
        use_videos=args.frame_storage == "video",
        push_to_hub=args.push_to_hub,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
