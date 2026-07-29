#!/usr/bin/env bash
# Pull the ACT export from the Hub and pack it into a LeRobot dataset.
#
#   MT4_EXPORT_REPO=you/mt4-act-export-v1 bash fetch_and_pack.sh
#
# Run after setup.sh, before train.sh. Both the download and the packed
# dataset go to the container disk (see setup.sh for why the network volume
# is unusable as a live training path).
set -euo pipefail

WORK="${MT4_ACT_WORK:-/root/mt4act}"
EXPORT_REPO="${MT4_EXPORT_REPO:?set MT4_EXPORT_REPO to the HF dataset repo id}"
DATASET_REPO="${MT4_DATASET_REPO:-local/mt4-act}"

# Defaults describe the RECOMMENDED FIRST RUN, not the whole corpus.
#
#   family=shuffle    111 clean episodes of one coherent behaviour. The stack
#                     corpus is excluded because its labels are largely
#                     unusable: median worst single-step jump 72 deg (up to
#                     171), against 3.6 deg for shuffle. A stack episode
#                     retreats to the camera park between phases and the
#                     recorder interpolates that traverse as one transition,
#                     so the label asks for a move the safety layer caps at
#                     5 deg and the arm would refuse. Fix the recorder before
#                     trying `all`.
#   max_jump_deg=15   Belt and braces: drops the 1 shuffle episode that also
#                     has a discontinuity.
FAMILY="${MT4_FAMILY:-shuffle}"
MAX_JUMP_DEG="${MT4_MAX_JUMP_DEG:-15}"
# Keep every frame by default. Trimming the ~10% stationary frames is a real
# ablation (see docs/ACT_PIPELINE.md) but it should be a deliberate choice,
# not something that quietly happens to the baseline.
MAX_STATIC_RUN="${MT4_MAX_STATIC_RUN:-}"

export HF_HOME="${HF_HOME:-$WORK/hf}"

echo "=== download $EXPORT_REPO ==="
hf download "$EXPORT_REPO" --repo-type dataset --local-dir "$WORK/export"

echo
echo "=== manifest ==="
python - "$WORK/export/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
eps = m["episodes"]
print(f"  episodes {len(eps)}  frames {sum(e['n_frames'] for e in eps)}  fps {m['fps']}")
print(f"  image    {m['image_width']}x{m['image_height']}  state dim {len(m['state_columns'])}")
PY

echo
echo "=== pack -> LeRobot ==="
ARGS=(--export "$WORK/export" --out "$WORK/dataset" --repo-id "$DATASET_REPO" --family "$FAMILY")
[ -n "$MAX_STATIC_RUN" ] && ARGS+=(--max-static-run "$MAX_STATIC_RUN")
[ -n "$MAX_JUMP_DEG" ] && ARGS+=(--max-jump-deg "$MAX_JUMP_DEG")

PYTHONPATH="$WORK/export/code" python -m mt4_pi.act.pack_lerobot "${ARGS[@]}"

echo
echo "=== verify the packed dataset loads and decodes ==="
# A dataset that writes cleanly but cannot be read back is the failure mode
# worth catching here, not 40 minutes into training.
DATASET_ROOT="$WORK/dataset/$DATASET_REPO" python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

side = json.loads((root / "mt4_act_meta.json").read_text())
ds = LeRobotDataset(side["repo_id"], root=root)
print(f"  frames {ds.num_frames}  episodes {ds.num_episodes}  fps {ds.fps}")

sample = ds[0]
img = sample[side["camera_key"]]
state, action = sample["observation.state"], sample["action"]
print(f"  image  {tuple(img.shape)} {img.dtype}")
print(f"  state  {tuple(state.shape)}  action {tuple(action.shape)}")
assert state.shape[-1] == len(side["state_columns"]), "state dim disagrees with the sidecar"
assert action.shape[-1] == len(side["action_columns"]), "action dim disagrees with the sidecar"
assert float(img.max()) > 0, "decoded an all-black frame -- video backend is broken"
print("  decode OK")
PY

echo
echo "packed at $WORK/dataset/$DATASET_REPO -- next: bash train.sh"
