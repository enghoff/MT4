"""Upload `data/act_export/` to a private HF dataset repo, code included.

    python -m mt4_pi.act.push_export --repo-id you/mt4-act-export-v1

This is the hand-off between the workstation (which has the raw frames) and
the GPU box (which has `lerobot`). It exists for the same reason the pi0.5
pipeline grew one: paying a GPU instance to sit idle acting as a file server
cost ~110x more wall-clock than routing the bytes through the Hub, and the
transfer no longer has to overlap with the instance's uptime at all. Push
first, start the pod second -- the pod should find its data already waiting.

The packing code travels with the data. `pack_lerobot.py` and `schema.py` are
the only modules the GPU box needs, they are small and dependency-light, and
bundling them makes the upload self-describing: whatever version of the
column layout produced these arrays is the version that will read them back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Repo-relative source files copied into the upload under `code/`.
BUNDLED_CODE = (
    "mt4_pi/__init__.py",
    "mt4_pi/act/__init__.py",
    "mt4_pi/act/schema.py",
    "mt4_pi/act/pack_lerobot.py",
)
BUNDLED_SCRIPTS_DIR = "mt4_pi/act/runpod"

README = """\
# MT4 ACT export

Intermediate artefact between `mt4_pi.act.export_episodes` (workstation) and
`mt4_pi.act.pack_lerobot` (GPU box). Not a LeRobot dataset -- run the packing
step to get one.

- `manifest.json` — episode index plus the state/action column layout
- `episodes/<id>.mp4` — {w}x{h} H.264, one clip per episode
- `episodes/<id>.npz` — `state`, `action`, `static` arrays for that episode
- `code/` — the exact packing code this export was written for

Actions are **absolute joint targets** (radians) plus a gripper command, not
velocities. State carries the arm pose, the gripper, and a one-hot task
encoding. See `code/mt4_pi/act/schema.py` for the full rationale.

## Use

```bash
hf download {repo_id} --repo-type dataset --local-dir export
PYTHONPATH=export/code python -m mt4_pi.act.pack_lerobot \\
    --export export --repo-id you/mt4-act-v1 --out dataset
```

Episodes: {n_episodes} · frames: {n_frames} · fps: {fps}
"""


def _link_episodes(source: Path, dest: Path) -> None:
    """Expose `source` at `dest` without duplicating 271 MB of video.

    Three strategies, cheapest first. `upload_large_folder` follows all of
    them, so this only affects local disk:

    1. A directory symlink -- one inode, but Windows refuses it outside
       developer mode or an elevated shell.
    2. Hardlinks per file -- works on NTFS with no special privilege and
       still costs no data blocks. The files are write-once artefacts of the
       export, so sharing inodes is safe.
    3. A real copy, if the two paths are on different volumes.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return
    try:
        dest.rmdir()
        dest.symlink_to(source.resolve(), target_is_directory=True)
        return
    except OSError:
        dest.mkdir(parents=True, exist_ok=True)

    import shutil

    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        target = dest / path.name
        try:
            target.hardlink_to(path)
        except OSError:
            shutil.copy2(path, target)


def build_upload_dir(export_root: Path, repo_root: Path, staging: Path, repo_id: str) -> dict:
    """Assemble the upload tree: export payload + code bundle + README.

    Symlinks the bulky `episodes/` rather than copying it -- `upload_large_folder`
    follows them, and duplicating 271 MB to stage an upload is pointless.
    """
    manifest = json.loads((export_root / "manifest.json").read_text(encoding="utf-8"))
    staging.mkdir(parents=True, exist_ok=True)

    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _link_episodes(export_root / "episodes", staging / "episodes")

    for rel in BUNDLED_CODE:
        dest = staging / "code" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((repo_root / rel).read_bytes())
    for script in sorted((repo_root / BUNDLED_SCRIPTS_DIR).glob("*")):
        if script.is_file():
            dest = staging / "code" / "runpod" / script.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(script.read_bytes())

    (staging / "README.md").write_text(
        README.format(
            repo_id=repo_id,
            w=manifest["image_width"],
            h=manifest["image_height"],
            fps=manifest["fps"],
            n_episodes=len(manifest["episodes"]),
            n_frames=sum(e["n_frames"] for e in manifest["episodes"]),
        ),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="data/act_export")
    ap.add_argument("--repo-id", required=True, help="e.g. you/mt4-act-export-v1")
    ap.add_argument("--staging", default=None, help="upload staging dir (default: <export>/../act_upload)")
    ap.add_argument("--public", action="store_true", help="create a public repo (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="stage the tree but do not upload")
    args = ap.parse_args(argv)

    export_root = Path(args.export)
    repo_root = Path(__file__).resolve().parent.parent.parent
    staging = Path(args.staging) if args.staging else export_root.parent / "act_upload"

    manifest = build_upload_dir(export_root, repo_root, staging, args.repo_id)
    n_frames = sum(e["n_frames"] for e in manifest["episodes"])
    print(f"staged {len(manifest['episodes'])} episodes / {n_frames} frames at {staging}")

    if args.dry_run:
        print("--dry-run: not uploading")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    # Chunked, parallel and resumable -- the plain upload_folder stalls on a
    # few hundred MB of video and cannot resume a broken run.
    api.upload_large_folder(
        repo_id=args.repo_id, repo_type="dataset", folder_path=str(staging), print_report=True
    )
    print(f"\npushed -> https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
