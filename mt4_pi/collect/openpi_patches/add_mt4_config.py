"""Idempotently add the MT4 pi0.5 LoRA `TrainConfig` (or a new named variant,
e.g. for a later round trained on a different dataset repo_id) to a Brev
instance's `~/openpi/src/openpi/training/config.py`.

Previously this was a one-off edit re-derived by hand on each fresh
instance; committing it here means a fresh `mt4-pi05-lora*` instance -- or
a new training round on an updated dataset -- doesn't require re-deriving
the exact `TrainConfig` fields from openpi's example configs again.

Run from the openpi checkout root on the instance (i.e. ~/openpi):

    # round 1 (the config the currently-deployed step-1999 checkpoint used)
    python add_mt4_config.py

    # a later round, trained on a different dataset
    python add_mt4_config.py --name pi05_mt4_lora_v3 \\
        --repo-id senghoff/mt4-cube-shuffle-v3 --num-train-steps 4000

Idempotent: exits without writing if a TrainConfig with --name already
exists in config.py. --num-train-steps only sets the config's *default*;
every train.py invocation in docs/PI05_FINETUNING_PIPELINE.md overrides it
via its own --num-train-steps flag regardless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG_PATH = "src/openpi/training/config.py"

# `_CONFIGS`'s last two entries before this comment are RoboArena/PolaRiS's
# own generated lists -- inserting immediately before it keeps our configs
# together, after the debug configs, without needing to parse the list.
INSERT_BEFORE = "    # RoboArena & PolaRiS configs.\n"

TEMPLATE = '''    #
    # MT4 pi05 LoRA fine-tune (cube-shuffle{extra_note} demonstrations).
    #
    TrainConfig(
        name="{name}",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotDROIDDataConfig(
            repo_id="{repo_id}",
            base_config=DataConfig(prompt_from_task=True),
            # No assets override: compute fresh norm stats scoped to this
            # repo_id rather than reusing pi05_droid's real-Franka stats --
            # our action/state distributions are MT4-native (see
            # convert_to_lerobot.py's unit-choice docstring), a different
            # embodiment wearing the DROID wire-format costume, not the
            # same robot pi05_droid was pretrained on (norm_stats.md's
            # reuse case).
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        num_train_steps={num_train_steps},
        batch_size=16,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pi05_mt4_lora")
    parser.add_argument("--repo-id", default="senghoff/mt4-cube-shuffle-v2")
    parser.add_argument("--num-train-steps", type=int, default=2_000)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    path = Path(args.config_path)
    text = path.read_text(encoding="utf-8")

    if f'name="{args.name}"' in text:
        print(f"{args.name!r} already present in {path}, nothing to do.")
        return 0

    if INSERT_BEFORE not in text:
        print(
            f"Anchor line {INSERT_BEFORE!r} not found in {path} -- "
            "has config.py's structure changed upstream?",
            file=sys.stderr,
        )
        return 1

    extra_note = "" if args.name == "pi05_mt4_lora" else " + stacking"
    block = TEMPLATE.format(
        name=args.name,
        repo_id=args.repo_id,
        num_train_steps=args.num_train_steps,
        extra_note=extra_note,
    )
    text = text.replace(INSERT_BEFORE, block + INSERT_BEFORE, 1)
    path.write_text(text, encoding="utf-8")
    print(f"Added TrainConfig {args.name!r} (repo_id={args.repo_id}, num_train_steps={args.num_train_steps}) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
