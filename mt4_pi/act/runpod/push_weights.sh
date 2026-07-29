#!/usr/bin/env bash
# Push a trained ACT checkpoint to the Hub, then the pod can be destroyed.
#
#   MT4_WEIGHTS_REPO=you/mt4-act-v1 bash push_weights.sh [checkpoint_dir]
#
# Same lesson as the pi0.5 pipeline: do not pay an instance to act as a file
# server. Pulling weights over SSH from the pod measured ~2.3 MB/s; routing
# them through the Hub measured ~263 MB/s, roughly 110x faster, and it
# decouples the transfer from the instance's uptime entirely. Push, verify the
# push, then stop the pod -- in that order.
set -euo pipefail

WORK="${MT4_ACT_WORK:-/root/mt4act}"
REPO="${MT4_WEIGHTS_REPO:?set MT4_WEIGHTS_REPO to the HF model repo id}"
RUN_NAME="${MT4_RUN_NAME:-mt4-act-v1}"

CKPT="${1:-}"
if [ -z "$CKPT" ]; then
  CKPT=$(ls -1d "$WORK/outputs/$RUN_NAME"/checkpoints/*/ 2>/dev/null | sort | tail -1 || true)
  [ -n "$CKPT" ] || { echo "!! no checkpoint under $WORK/outputs/$RUN_NAME" >&2; exit 1; }
fi
# lerobot nests the loadable model under pretrained_model/; prefer it if present.
[ -d "$CKPT/pretrained_model" ] && CKPT="$CKPT/pretrained_model"

echo "=== uploading $CKPT -> $REPO ==="
du -sh "$CKPT"
ls -1 "$CKPT"

# The sidecar records which state columns survived pruning. Without it the
# serving side cannot build the input vector, and the weights are unusable.
if [ ! -f "$CKPT/mt4_act_meta.json" ]; then
  SIDE=$(find "$WORK/dataset" -name mt4_act_meta.json | head -1 || true)
  [ -n "$SIDE" ] && cp "$SIDE" "$CKPT/" && echo "  added mt4_act_meta.json from $SIDE"
fi
[ -f "$CKPT/mt4_act_meta.json" ] || { echo "!! mt4_act_meta.json missing; refusing to push unusable weights" >&2; exit 1; }

CKPT="$CKPT" REPO="$REPO" python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
repo, ckpt = os.environ["REPO"], os.environ["CKPT"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=ckpt, print_report=True)

files = {f.rfilename for f in api.model_info(repo, files_metadata=False).siblings}
missing = {"mt4_act_meta.json"} - files
if missing:
    raise SystemExit(f"upload incomplete, missing {missing}")
print(f"\nverified on the Hub: {len(files)} files")
print(f"https://huggingface.co/{repo}")
PY

echo
echo "Weights are on the Hub and verified. Safe to stop the pod now."
