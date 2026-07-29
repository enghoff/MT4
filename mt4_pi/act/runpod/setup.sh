#!/usr/bin/env bash
# Prepare a fresh RunPod container for ACT training. Idempotent -- safe to
# re-run after a pod restart.
#
#   bash setup.sh
#
# Everything lands on the CONTAINER DISK, never the network volume. RunPod
# network volumes are MooseFS over FUSE: PyTorch DataLoader workers that mmap
# or randomly seek into a dataset stored there hang indefinitely rather than
# erroring, and the usual diagnostics lie (`df`, `nproc`, `free` and
# `loadavg` all report host-level or bogus values inside the container). Use
# /workspace only as an archive you copy to and from, never as a live
# training path.
set -euo pipefail

WORK="${MT4_ACT_WORK:-/root/mt4act}"
export HF_HOME="${HF_HOME:-$WORK/hf}"

echo "=== target layout (container disk) ==="
echo "  work dir : $WORK"
echo "  HF cache : $HF_HOME"
mkdir -p "$WORK"/{export,dataset,outputs} "$HF_HOME"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "!! no GPU visible; ACT training needs one" >&2; exit 1; }

echo
echo "=== system deps ==="
# torchcodec links against FFmpeg's shared libraries and fails at import with
# "libavutil.so.56: cannot open shared object file" without them. The dataset
# then writes fine and only fails when something tries to READ a frame back --
# a confusing place to discover a missing apt package. Most CUDA images ship
# these, but not all.
if ! ldconfig -p | grep -q libavutil; then
  echo "  installing ffmpeg (torchcodec needs libavutil)"
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
fi
ldconfig -p | grep -q libavutil && echo "  libavutil present" || echo "  !! libavutil still missing"

echo
echo "=== python deps ==="
# lerobot pulls its own torch pin; let it. opencv-headless is what
# pack_lerobot.py decodes the exported clips with, and the headless build
# avoids the libGL dependency that the normal wheel needs in a bare container.
pip install --quiet --upgrade pip
pip install --quiet "lerobot[act]" || pip install --quiet lerobot
pip install --quiet opencv-python-headless "huggingface_hub[cli]"

echo
echo "=== versions ==="
python - <<'PY'
import importlib, torch
print(f"torch      {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device     {torch.cuda.get_device_name(0)}")
try:
    import lerobot
    print(f"lerobot    {getattr(lerobot, '__version__', 'unknown')}")
except Exception as exc:
    raise SystemExit(f"lerobot import failed: {exc}")
for path in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
    try:
        importlib.import_module(path)
        print(f"dataset API {path}")
        break
    except Exception:
        continue
else:
    raise SystemExit("no known LeRobotDataset module path -- pack_lerobot.py will fail")
PY

echo
echo "setup complete. Next: bash fetch_and_pack.sh"
