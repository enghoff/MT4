#!/usr/bin/env bash
# Train ACT on the packed MT4 dataset.
#
#   bash train.sh smoke     # ~200 steps, proves the loop runs end to end
#   bash train.sh           # the real run
#
# Always do the smoke run first. It costs about a minute and catches the
# dimension, decode and dataloader failures that otherwise surface an hour in,
# on a clock that bills by uptime.
set -euo pipefail

WORK="${MT4_ACT_WORK:-/root/mt4act}"
DATASET_REPO="${MT4_DATASET_REPO:-local/mt4-act}"
DATASET_ROOT="$WORK/dataset/$DATASET_REPO"
export HF_HOME="${HF_HOME:-$WORK/hf}"

MODE="${1:-full}"
if [ "$MODE" = "smoke" ]; then
  STEPS="${MT4_SMOKE_STEPS:-200}"; SAVE_FREQ=$STEPS; LOG_FREQ=25
  RUN_NAME="mt4-act-smoke"
else
  STEPS="${MT4_STEPS:-100000}"; SAVE_FREQ="${MT4_SAVE_FREQ:-10000}"; LOG_FREQ=250
  RUN_NAME="${MT4_RUN_NAME:-mt4-act-v1}"
fi
OUT="$WORK/outputs/$RUN_NAME"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# chunk_size / n_action_steps are the two that must NOT be copied from an
# ALOHA recipe unexamined. ALOHA's defaults (100/100) are tuned for 50 Hz,
# where 100 steps is 2 s. This dataset is 10 Hz, so 100 steps would be 10 s --
# most of an entire episode, predicted open-loop from the first frame.
#
#   chunk_size=32      3.2 s of context, the same horizon ALOHA actually uses
#   n_action_steps=8   0.8 s executed before re-planning. On the MT4 that is
#                      ~1.6 s of wall clock, because each queued waypoint
#                      costs ~204 ms on the serial link -- the arm, not the
#                      GPU, sets the closed-loop rate here.
CHUNK_SIZE="${MT4_CHUNK_SIZE:-32}"
N_ACTION_STEPS="${MT4_N_ACTION_STEPS:-8}"
# 16, not 32. Measured on an RTX 3070 (8 GB) at 640x360, chunk 32:
#   batch 16 -> 5.9 step/s, 4.07 GB peak  =  94 samples/s
#   batch 32 -> 2.4 step/s, 7.02 GB peak  =  76 samples/s
# Bigger batches are slower in samples/s here and leave only 1 GB of headroom.
# On a 24 GB card 32 is fine, but 16 is not the compromise it looks like.
BATCH_SIZE="${MT4_BATCH_SIZE:-16}"
# Explicit rather than derived: `nproc` reports host-level values inside a
# RunPod container and cannot be trusted to size the dataloader.
NUM_WORKERS="${MT4_NUM_WORKERS:-8}"

if [ ! -d "$DATASET_ROOT" ]; then
  echo "!! no dataset at $DATASET_ROOT -- run fetch_and_pack.sh first" >&2; exit 1
fi

# `lerobot-train` is the modern entry point; older trees only ship the script.
if command -v lerobot-train >/dev/null 2>&1; then
  TRAIN=(lerobot-train)
else
  TRAIN=(python -m lerobot.scripts.train)
fi

echo "=== $MODE run: $RUN_NAME ==="
echo "  dataset  $DATASET_ROOT"
echo "  steps    $STEPS   batch $BATCH_SIZE   chunk $CHUNK_SIZE   exec $N_ACTION_STEPS"
echo "  output   $OUT"
echo

"${TRAIN[@]}" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=act \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name="$RUN_NAME" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --num_workers="$NUM_WORKERS" \
  --wandb.enable="${MT4_WANDB:-false}" \
  "${@:2}"

echo
echo "=== checkpoints ==="
ls -1 "$OUT/checkpoints" 2>/dev/null || echo "  (none written)"

# The sidecar has to travel with the weights: it is the only record of which
# state columns survived pruning, and serving cannot rebuild the input vector
# without it.
LAST=$(ls -1d "$OUT"/checkpoints/*/ 2>/dev/null | sort | tail -1 || true)
if [ -n "$LAST" ]; then
  cp "$DATASET_ROOT/mt4_act_meta.json" "$LAST/pretrained_model/" 2>/dev/null \
    || cp "$DATASET_ROOT/mt4_act_meta.json" "$LAST/" 2>/dev/null || true
  echo "  copied mt4_act_meta.json into $LAST"
fi

if [ "$MODE" = "smoke" ]; then
  echo
  echo "smoke run finished. If loss decreased and a checkpoint exists, start the real run:"
  echo "  bash train.sh"
fi
