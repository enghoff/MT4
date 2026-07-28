#!/bin/bash
# norm stats -> smoke test -> automated gate -> detached 4000-step full run.
# Emits ===MARKER=== lines; every failure path emits a ===FAIL_*=== marker so
# a monitor never has to infer failure from silence.
set -o pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/openpi" || { echo "===FAIL_NO_OPENPI==="; exit 1; }

echo "===DISK_BEFORE==="
df -h / | tail -1

# uv's download cache is dead weight once .venv is built; the root volume here
# is 117G (not the 500G brev's search implied), so reclaim it before pulling
# the 11.6GB base checkpoint + dataset + checkpoints.
echo "===CLEAN_CACHE==="
uv cache clean 2>&1 | tail -2
sudo apt-get clean 2>/dev/null
df -h / | tail -1

echo "===NORM_STATS==="
date
uv run scripts/compute_norm_stats.py --config-name pi05_mt4_lora_v3 2>&1 | tail -25
if [ $? -ne 0 ]; then echo "===FAIL_NORM_STATS==="; exit 1; fi
echo "===NORM_STATS_DONE==="
df -h / | tail -1

echo "===SMOKE_START==="
date
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mt4_lora_v3 \
  --exp-name=smoke_test_v3 --overwrite --num-train-steps=50 --no-wandb-enabled \
  --log-interval=5 --save-interval=1000000 2>&1 | tee "$HOME/smoke_v3.log" | tail -25
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then echo "===FAIL_SMOKE_EXIT_$SMOKE_RC==="; exit 1; fi

# Real gate, not just "did it exit 0": the units-bug class of failure trains
# happily to completion with wrong-scale actions. Require finite, sane loss.
echo "===SMOKE_GATE==="
if grep -qiE 'loss=(nan|inf|-inf)' "$HOME/smoke_v3.log"; then
  echo "===FAIL_SMOKE_NONFINITE_LOSS==="; exit 1
fi
LOSS=$(grep -oE 'loss=[0-9]+\.?[0-9]*' "$HOME/smoke_v3.log" | tail -1 | cut -d= -f2)
echo "final smoke loss: ${LOSS:-<none found>}"
if [ -z "$LOSS" ]; then echo "===FAIL_SMOKE_NO_LOSS_LOGGED==="; exit 1; fi
# Round 1 started ~1.5-1.9 and fell; anything wildly above that after 50 steps
# means something is structurally wrong, not just undertrained.
if ! awk -v l="$LOSS" 'BEGIN{exit !(l+0 > 0 && l+0 < 10)}'; then
  echo "===FAIL_SMOKE_LOSS_OUT_OF_RANGE==="; exit 1
fi
echo "===SMOKE_GATE_PASSED==="

# openpi writes a checkpoint for the smoke run even with
# --save-interval=1000000, so this costs ~8.7GB that is pure dead weight once
# the gate has passed. On a 117GB root volume (see 4b -- brev's advertised
# 500GB is not what you get) that margin matters for a 4000-step run.
echo "===CLEAN_SMOKE_CKPT==="
rm -rf "$HOME/openpi/checkpoints/pi05_mt4_lora_v3/smoke_test_v3"
df -h / | tail -1

echo "===FULL_LAUNCH==="
date
setsid nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_mt4_lora_v3 \
  --exp-name=full_run_v3 --overwrite --num-train-steps=4000 --no-wandb-enabled \
  --log-interval=25 --save-interval=500 \
  </dev/null >"$HOME/full_run_v3.log" 2>&1 &
disown
sleep 5
echo "===FULL_LAUNCHED==="
date
