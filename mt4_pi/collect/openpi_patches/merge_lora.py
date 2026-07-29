"""Merge a pi05 LoRA fine-tune checkpoint into base-shaped params, so it can be
served under the stock `pi05_droid` config with no server-side code changes.

Run this on MEDIA, never on the training instance -- an earlier out-of-place
version peaked at ~23GB RSS and got OOM-killed on the (swap-less) training box
while training was also running, which took the whole instance UNHEALTHY. See
docs/PI05_FINETUNING_PIPELINE.md 4a/6a.

LoRA math (confirmed from openpi's src/openpi/models/lora.py + the
gemma_2b_lora / gemma_300m_lora variant definitions in gemma.py):

    merged = w + scaling * (lora_a @ lora_b)   # batched matmul over leading axes
    gemma_2b_lora:   rank=16, alpha=16.0 -> scaling = alpha/rank = 1.0
    gemma_300m_lora: rank=32, alpha=32.0 -> scaling = alpha/rank = 1.0

Both resolve to scaling=1.0, so it is exactly `w + lora_a @ lora_b`.

Two LoRA parameter layouts appear in the checkpoint and both are handled:
  - Einsum-style:       sibling keys {"w", "lora_a", "lora_b"} in one dict
  - FeedForward-style:  flat siblings "<name>", "<name>_lora_a", "<name>_lora_b"
                        (e.g. gating_einsum / gating_einsum_lora_a / ..._lora_b)

Usage (paths must be absolute -- orbax fails deep in its tensorstore backend
with "Checkpoint path should be absolute" otherwise, hence the .resolve()):

    JAX_PLATFORMS=cpu /opt/openpi/.venv/bin/python3 merge_lora.py \
        --in  /root/mt4_merge/checkpoints/pi05_mt4_lora_v3/full_run_v3/3999 \
        --out /root/mt4_merge/checkpoints/pi05_mt4_lora_v3_merged/3999/params
"""

import argparse
import gc
import pathlib

import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as _model


def merge_lora_inplace(d):
    """Mutate `d` in place -- never build a parallel copy of the whole tree.

    The out-of-place version (a dict comprehension rebuilding each level) kept
    the entire ~11.6GB restored pytree alive for the duration of the rebuild
    plus per-tensor float32 upcast temporaries, peaking ~23GB. In place, the
    only extra memory beyond the base checkpoint is one transient float32 copy
    of whichever single tensor is mid-merge.
    """
    if not isinstance(d, dict):
        return
    for v in d.values():
        merge_lora_inplace(v)
    for key in list(d.keys()):
        if key == "lora_a":
            base, b_key = "w", "lora_b"
        elif key.endswith("_lora_a"):
            base = key[: -len("_lora_a")]
            b_key = base + "_lora_b"
        else:
            continue
        if base not in d or b_key not in d:
            raise ValueError(f"found {key!r} without matching {base!r}/{b_key!r} in {list(d.keys())}")
        w = d[base]
        merged = (w.astype(np.float32) + np.matmul(d[key].astype(np.float32), d[b_key].astype(np.float32))).astype(
            w.dtype
        )
        assert merged.shape == w.shape, (merged.shape, w.shape)
        del d[base], d[key], d[b_key], w
        d[base] = merged
        gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True, help="checkpoint step dir (contains params/)")
    ap.add_argument("--out", dest="out_dir", required=True, help="output params dir")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.in_dir).resolve()
    out = pathlib.Path(args.out_dir).resolve()

    print(f"Restoring LoRA params from {ckpt / 'params'} ...")
    params = _model.restore_params(ckpt / "params", restore_type=np.ndarray)

    n_pairs = 0

    def _count(d):
        nonlocal n_pairs
        if isinstance(d, dict):
            for k in d:
                if k == "lora_a" or k.endswith("_lora_a"):
                    n_pairs += 1
            for v in d.values():
                _count(v)

    _count(params)
    print(f"Found {n_pairs} LoRA (a,b) pairs to merge.")
    if n_pairs == 0:
        # Refuse to emit a "merged" checkpoint that is silently just the base
        # weights -- that would serve as a plausible-looking no-op fine-tune.
        print("ERROR: no LoRA pairs found; refusing to write a no-op merge.")
        return 1

    merge_lora_inplace(params)

    def _assert_clean(d):
        if isinstance(d, dict):
            for k, v in d.items():
                assert k != "lora_a" and k != "lora_b" and not k.endswith("_lora_a") and not k.endswith("_lora_b"), k
                _assert_clean(v)

    _assert_clean(params)
    print("Merge complete, no lora_a/lora_b keys remain.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(out, ocp.args.PyTreeSave(item={"params": params}))
    print(f"Saved merged params to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
