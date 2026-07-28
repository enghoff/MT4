#!/bin/bash
# Unattended openpi setup for a fresh Brev instance, reproducing the round-1
# environment documented in docs/PI05_FINETUNING_PIPELINE.md.
# Emits ===MARKER=== lines so a monitor can track progress / spot failures.
set -o pipefail

log() { echo "===$1==="; }

log START
date

# --- 0. sanity: GPU present -------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { log FAIL_NO_GPU; exit 1; }

# --- 1. cloud-init holds the apt lock on a fresh box; wait it out -----------
log WAIT_APT_LOCK
for i in $(seq 1 150); do
  if ! sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 && \
     ! sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
    break
  fi
  sleep 4
done

log APT_INSTALL
sudo apt-get update -qq 2>&1 | tail -2
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl build-essential 2>&1 | tail -3

# --- 2. uv ------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log INSTALL_UV
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || { log FAIL_UV; exit 1; }

# --- 3. openpi checkout -----------------------------------------------------
if [ ! -d "$HOME/openpi/.git" ]; then
  log CLONE_OPENPI
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$HOME/openpi" 2>&1 | tail -5
fi
cd "$HOME/openpi" || { log FAIL_NO_OPENPI; exit 1; }
git rev-parse --short HEAD

# --- 4. deps ----------------------------------------------------------------
log UV_SYNC
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -15 || { log FAIL_UV_SYNC; exit 1; }

# The lerobot version-pinning trap: openpi vendors an exact old lerobot commit,
# and writer/reader must match exactly or the dataset fails to load. Pin the
# same commit + datasets/pyarrow versions the local writer used.
log PIN_LEROBOT
uv pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" --no-deps 2>&1 | tail -3
uv pip install "datasets==3.6.0" "pyarrow==20.0.0" --no-deps 2>&1 | tail -3

# --- 5. HF auth (token file staged separately by the caller) ----------------
log HF_AUTH
if [ -f "$HOME/.hf_token" ]; then
  mkdir -p "$HOME/.cache/huggingface"
  cp "$HOME/.hf_token" "$HOME/.cache/huggingface/token"
  chmod 600 "$HOME/.cache/huggingface/token"
  uv run python -c "
from huggingface_hub import HfApi
info = HfApi().dataset_info('senghoff/mt4-cube-shuffle-v3')
print('dataset reachable:', info.id, '| files:', len(info.siblings))
" 2>&1 | tail -3 || { log FAIL_HF_AUTH; exit 1; }
else
  log FAIL_NO_TOKEN; exit 1
fi

# --- 6. MT4 TrainConfig patch ----------------------------------------------
log ADD_CONFIG
python3 "$HOME/add_mt4_config.py" --name pi05_mt4_lora_v3 \
  --repo-id senghoff/mt4-cube-shuffle-v3 --num-train-steps 4000 2>&1 | tail -3
uv run python -c "
from openpi.training import config as _c
assert 'pi05_mt4_lora_v3' in _c._CONFIGS_DICT, 'config missing'
print('config registered ok')
" 2>&1 | tail -3 || { log FAIL_CONFIG; exit 1; }

# --- 7. norm stats (downloads the dataset; no GPU needed) -------------------
# NOTE: on current openpi HEAD compute_norm_stats.py takes --config-name as a
# *flag*; older revisions (round 1) took it positionally. train.py still takes
# the config name positionally, so the two scripts genuinely differ -- don't
# assume one from the other.
log NORM_STATS
uv run scripts/compute_norm_stats.py --config-name pi05_mt4_lora_v3 2>&1 | tail -20 || { log FAIL_NORM_STATS; exit 1; }

log SETUP_DONE
date
