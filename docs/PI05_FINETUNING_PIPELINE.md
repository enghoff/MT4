# pi0.5 LoRA Fine-Tuning Pipeline — MT4 Cube Shuffle

End-to-end runbook for turning recorded MT4 pick/place demonstrations into a
LoRA fine-tune of `pi05_droid`, trained on a rented Brev GPU, deployable to
the `MEDIA` inference server. Companion to
[`PI05_DEPLOYMENT.md`](PI05_DEPLOYMENT.md) (which covers the already-solved
serving side); this doc covers data → training.

**Status (2026-07-28): first real-hardware validation done — grasp
targeting/timing is not yet reliable. Round 2 (combined shuffle+stack data,
4000 steps) is underway to address it.** The round-1 2000-step run
(resumed once after an instance outage, see §4a) finished cleanly — loss
dropped from ~1.6 to ~0.02, no errors — and the merged/deployed checkpoint
served correctly (27s first-call compile, then steady ~285ms, matching
base `pi05_droid` latency, §6a-6b). Real-arm testing (§7a) then found two
distinct failure modes across two runs: a premature mid-air grasp, and a
full descent to table height that never grasped and drifted to a location
matching none of the actual cube positions. Motion quality itself (smooth,
continuous, safety-clean) is good — targeting/grasp-timing is not, which
reads as a data-volume/diversity problem more than a wiring bug (see §7a
for what was ruled out). Round 2 combines the original
`data/pi_demos` shuffle demos with a new `data/pi_stack_demos` stacking set
into one corpus (`senghoff/mt4-cube-shuffle-v3`) and doubles training to
4000 steps. See §0 for the default procedure this and future training
requests follow.

---

## 0. Default scope for a requested training run

**Unless the user narrows the scope explicitly** (e.g. "just resume", "only
the shuffle data", "reuse the existing dataset"), "run a training run" /
"fine-tune again" means the *whole* pipeline below, not just `train.py`:

1. **Collect, if asked** — otherwise use whatever's already on disk under
   `data/` (every `data/<name>/episodes/` directory that exists, not just
   the most recently touched one — e.g. both `pi_demos` and
   `pi_stack_demos`).
2. **Convert + merge all of it into one fresh dataset**, one call to
   `mt4_pi.collect.convert_to_lerobot` (no `--data-dir` needed — it
   defaults to every known source dir that exists; see §3). **Mint a new
   `repo_id`** (increment the `-vN` suffix) rather than overwriting the
   previous one — cheap, and keeps the currently-deployed checkpoint's
   dataset reproducible.
3. **Push to HF Hub and confirm it landed** *before* touching Brev — do
   not start/restart the GPU instance until the dataset push has actually
   finished. The GPU instance bills by wall-clock uptime regardless of
   whether anything is using the GPU, so starting it "in parallel" to save
   a few minutes of wall-clock time just pays for idle GPU while local
   conversion/upload (which needs no GPU at all) finishes anyway.
4. **Only then** restart the Brev instance if stopped, refresh SSH
   (`brev refresh` — the IP changes every stop/start cycle), and **point a
   new/updated `TrainConfig` at the new `repo_id`** (see §5) — add a new
   named config rather than mutating the deployed one's config in place,
   same reasoning as the repo_id.
5. **Smoke test (50 steps) before the full run, always** — this is not
   optional even when the pipeline is "well understood"; it's what would
   have caught the units bug (§3) at zero cost instead of after a full run.
6. **Full run at the requested step count** (ask only if truly unspecified
   — default to at least the previous run's step count, scaling up if the
   dataset grew materially, as happened going from round 1 → round 2).
7. **Merge on MEDIA, never the training instance** (§6a) — deploy, verify
   latency against a known-good baseline (§6b), then **re-validate on the
   real arm** (§7a) — a clean loss curve and serving latency say nothing
   about whether the learned behavior actually works.
8. Stop (don't delete) the Brev instance when done, per standing
   instruction.

This section exists so a future request to "run a training run" doesn't
default to just re-running `train.py` on stale data — the whole chain
above is the default unit of work.

---

## 1. Pipeline stages

```
collect_shuffle.py    ⟍
                        ⟩→  convert_to_lerobot.py (merges both)  →  HF Hub dataset repo  →
collect_stack.py      ⟋
  Brev GPU instance (openpi) → compute_norm_stats → train.py (LoRA) →
  checkpoint → merge/copy → MEDIA → validate on hardware
```

| Stage | Where | Output |
|---|---|---|
| Collect (shuffle) | `collect_shuffle.py` (this repo, real arm+camera) | `data/pi_demos/episodes/ep_NNNNNN/` (raw frames + waypoint trace) |
| Collect (stack) | `collect_stack.py` (this repo, real arm+camera) | `data/pi_stack_demos/episodes/ep_NNNNNN/` (same schema) |
| Convert | `mt4_pi/collect/convert_to_lerobot.py` | merges every source dir into one LeRobot-format dataset, local + pushed to HF Hub |
| Train | Brev cloud GPU, `openpi` checkout | LoRA checkpoint under `~/openpi/checkpoints/pi05_mt4_lora/<exp-name>/` |
| Deploy | `MEDIA` (192.168.1.3), see `PI05_DEPLOYMENT.md` | running policy server |

---

## 2. Collect demonstrations

`collect_shuffle.py` runs the same vision-guided pick/place loop as
`shuffle_blocks.py` but wraps each cycle as a recorded episode:

- A background `EpisodeCamera` thread (`mt4_pi/collect/recorder.py`) samples
  frames at `DEFAULT_HZ = 15` (matches the DROID reference rate) during the
  move — never overlap this with `capture_scene_frame()`, they share one
  `cv2.VideoCapture`.
- `mt4_vision.pickplace.pick_cube()`/`place()` fire a real `MoveEvent` per
  completed waypoint via `on_waypoint=`; `WaypointLog` turns those into a
  per-frame pose by **linear interpolation over each leg's own measured
  duration**, holding the pose flat across the dead time between legs
  (serial round trips, `ensure_homed`, IK — the arm is provably stationary
  there). This is deliberately *not* zero-order-hold-per-leg and *not* one
  interpolation across the whole episode — see the docstrings in
  `recorder.py` and `tests/test_collect_waypoints.py` for why both of those
  are wrong.
- Only `verdict == "placed"` episodes are kept as demonstrations (mislabeling
  a real success as a failure, or vice versa, is worse for the fine-tune than
  dropping the episode — see `_verify_with_recheck`'s post-move recheck,
  reusing `mt4_vision.shuffle.POST_MOVE_RECHECK_*` for the same reason: a
  capture taken right after the arm clears can lag the real desk by a few
  hundred ms, briefly misreading a genuine success as `grasp_failed`.
- Prompts are randomized per-episode from templates in
  `mt4_pi/collect/prompts.py` (color + marker-vs-table phrasing) — logging
  one literal string for every episode would teach the model that string,
  not the concept.
- Unattended-safe: re-homes after `FAIL_STREAK_HOME = 3` consecutive
  non-placed verdicts, same pattern as `stack_cubes.py`.

```bash
python collect_shuffle.py --episodes 200   # or omit --episodes to run until Ctrl+C
```

Output root defaults to `data/pi_demos/`; each episode is
`meta.json` + `ticks.jsonl` + `frames/NNNNNN.jpg`.

**Collection tips for next time:**
- Run in short batches and spot-check `meta.json["verdict"]` distribution —
  a batch that's mostly `"grasp_failed"` means the vision/pick tuning has
  drifted, not that the fine-tune data is fine to use anyway.
- Vary cube placement / lighting across sessions if possible; a single
  sitting's episodes are highly correlated (same lighting, same starting
  cube layout region).

### 2a. Collect stacking demonstrations

`collect_stack.py` wraps `stack_cubes.py`/`unstack_cubes.py` the same way
`collect_shuffle.py` wraps the plain shuffle loop — same recorder, same
`meta.json`/`ticks.jsonl` schema (only `prompt` and `place.kind`/`place.level`
differ), so it converts through the exact same path. One cycle builds a
`--levels`-cube stack then takes it back down; the loose cubes from
unstacking become the next cycle's raw material, so cycles chain
unattended. Safety posture is more conservative than shuffle: a stack's
state is dead-reckoned from a count of what should be standing (not
re-established by vision every cycle), so anything that makes that count
untrustworthy — a shed cube, any hard `Mt4ClientError` mid-column — ends
the run rather than guessing.

```bash
python collect_stack.py --levels 3 --marker 2   # or omit for defaults
```

Output root defaults to `data/pi_stack_demos/`, same per-episode layout as
`data/pi_demos/`.

---

## 3. Convert to LeRobot format

**Both collectors' output merges into one dataset by default** — the whole
point of `prompts.py` having separate `_TO_MARKER`/`_TO_SLOT` (shuffle) and
`_STACK_BASE`/`_STACK_ON_TOP` (stacking) template pools is that pi0.5
conditions on the prompt string, so one prompt-conditioned policy is meant
to cover both behaviors, not two separate fine-tunes:

```bash
# Default: merges every data/*_demos dir that exists (currently pi_demos +
# pi_stack_demos) into one dataset. Pass --data-dir (repeatable) to override.
python -m mt4_pi.collect.convert_to_lerobot \
    --repo-id <hf_user>/<dataset-name> --push-to-hub
```

Key decisions baked into `mt4_pi/collect/convert_to_lerobot.py`:

- **Only successful episodes convert** (`meta["success"]` must be true).
- **Joint track comes from IK, not direct logging.** The recorder logs
  Cartesian TCP pose (+ raw `j4`), not joint angles — `_solve_joint_track`
  re-solves `ik_position` per tick, chaining each solution as the next
  tick's branch-selection seed (`near=`) for continuity, per
  `ik_q2_q3`'s nearest-branch convention. `j4` is taken directly from the
  log since it's already a real commanded angle, not something IK needs to
  guess.
- **Units: radians, not degrees.** `joint_position`/`actions` are written in
  radians (`math.radians` of the kinematics module's native-degrees
  `JointAnglesDeg`). This matters because `adapter.integrate_joint_target`
  clips the model's raw action to `[-1, 1]` before integrating — that only
  makes sense as a ~57°/s (≈1 rad/s) per-joint speed cap, matching real
  DROID/Franka joint speeds. **This was a real bug caught mid-session**: the
  first version of `adapter.py` added the clipped `[-1,1]` delta straight
  onto a degrees-native pose (i.e. treated it as ~1 degree/s), which would
  have made the fine-tuned policy's commanded motion ~57x too slow relative
  to what the training labels encode. Fixed by converting the integrated
  delta to degrees at the point it's added onto `JointAnglesDeg`; see
  `mt4_pi/adapter.py:integrate_joint_target`.
- **Image prep matches inference exactly** — `_prep_image` from
  `mt4_pi/observation.py` (BGR→RGB, `resize_with_pad` to 224×224, uint8) is
  reused for train-time frames, for train/serve parity.
- **Schema padding for DROID compatibility**: `exterior_image_2_left` and
  `wrist_image_left` are zeroed stand-ins (MT4 has one camera, no wrist
  cam) — kept present because `DroidInputs`'s `RepackTransform` plucks the
  key even when PI0.5 never reads it, and a missing key is a `KeyError`, not
  a silent skip.
- `push_to_hub()` is called directly on the in-memory write-mode dataset
  object right after `convert()` builds it, rather than re-opening it in
  read mode first — avoids one extra, version-sensitive round trip through
  the lerobot dataset loader (see the version trap below).

### The lerobot version-pinning trap (read this before reinstalling anything)

**openpi vendors an exact old lerobot git commit, not the PyPI `lerobot`
package**, and its dataset reader is strict about matching writer format.
This cost the most debugging time in the whole pipeline (4 distinct root
causes, one after another, all presenting as varyingly-worded loader
errors). The rule going forward:

> Every machine that writes or reads the LeRobot dataset — this repo's conda
> env, and the Brev training instance — must have the **exact same commit**
> installed:
> ```bash
> pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" --no-deps
> ```
> A newer PyPI `lerobot` (e.g. 0.6.0) writes a "3.0"-style format
> (`datasets` `Feature` type `"List"`) that this pinned commit's reader
> rejects with `ForwardCompatibilityError` / `ValueError: Feature type
> 'List' not found`.

It is not enough to just pin `lerobot` — the underlying `datasets`/`pyarrow`
versions matter too, because the parquet schema they embed differs even
between compatible-looking lerobot versions. The versions that worked,
matching Brev's `openpi` environment: **`datasets==3.6.0`, `pyarrow==20.0.0`**
(installed with `--no-deps` after the lerobot pin, to avoid pulling
newer transitive versions back in).

Symptoms if this drifts again, roughly in the order they'll appear:
1. `ForwardCompatibilityError: dataset ... only available in 3.0 format` —
   wrong lerobot commit on the writer.
2. `ValueError: Feature type 'List' not found` — wrong `datasets`/`pyarrow`
   version on writer vs. reader, even with the right lerobot commit.
3. Same error persisting after a fix — **stale local cache**. Clear
   `~/.cache/huggingface/lerobot/<repo_id>` and
   `~/.cache/huggingface/hub/datasets--<org>--<repo>` on whichever machine
   is failing to load.
4. `ValueError: One or several timestamps unexpectedly violate the
   tolerance inside episode range` — stale **HF Hub-side** files from an
   earlier broken push still present in the repo (old-format
   `meta/episodes/...`, `meta/tasks.parquet`, `meta/stats.json`,
   `data/chunk-000/...` sitting alongside the corrected ones), causing
   duplicated/misaligned rows on load. Diagnose via
   `HfApi().dataset_info(repo_id).siblings` and delete the stale ones with
   `HfApi().delete_file()` / `delete_folder()`.
5. **If the row count comes back exactly 2x what it should be even after
   cleanup**, that's a caching/duplication artifact on HF Hub's side that
   was never fully root-caused. Loading the same parquet from a local path
   (`LeRobotDataset(repo_id=..., root=local_path)`) gave the correct count,
   proving the local data was fine — only the previously-touched
   `repo_id` was cursed. **The reliable fix was pushing to a brand-new
   `repo_id`** with zero history (this repo's dataset is
   `senghoff/mt4-cube-shuffle-v2`, not `-v1`, for exactly this reason).
   **Lesson for next time: if a push goes wrong partway through, don't try
   to clean up the same repo_id — mint a new one.** It's free and far
   faster than hunting the caching bug further.

---

## 4. Brev GPU instance

Org: `nca-86490`. CLI from WSL Ubuntu (`brev create/exec/copy/ls/start/stop/delete`,
`brev search gpu`). GPU type used: `g2-standard-8:nvidia-l4:1` (~$1.02/hr,
24GB L4).

- **If an instance gets stuck in `STARTING`** for 10+ minutes with SSH
  unreachable (happened once, `mt4-pi05-lora`): don't `brev reset` if
  you've already done real setup work on it — reset only preserves
  `/home/brev/workspace/`, not arbitrary paths like `~/openpi`. Just
  **delete and recreate** on the same GPU type; a fresh instance came up
  clean in a couple minutes.
- **A stopped instance that won't restart is usually GPU capacity, not a
  broken instance** — see §4b. Don't delete it reflexively.
- **Fresh instance apt lock**: cloud-init runs its own `apt` in the
  background; `E: Could not get lock /var/lib/apt/lists/lock` on your first
  `apt install` is normal — poll and wait rather than retrying blindly:
  ```bash
  while sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do sleep 2; done
  ```
- **`uv`/`brev exec` PATH**: non-login remote shells don't have
  `~/.local/bin` on `PATH`. Every script run via `brev exec` needs
  `export PATH=/home/ubuntu/.local/bin:$PATH` as its first line, or `uv`
  is "not found".
- **Windows/WSL/Brev quoting**: chaining `Bash → wsl.exe -d Ubuntu -- bash
  -lc '...' → brev exec` mangles anything with `$`-based expansion in the
  inner script (PATH exports, heredocs, etc.). **The pattern that always
  works**: `Write` the script to the scratchpad dir, `cp` it across the
  `/mnt/c/...` boundary into the WSL home, then
  `brev exec <instance> @$HOME/script.sh`. Don't try to inline multi-line
  shell through three layers of quoting again — it's not worth the retries.
- **JAX compile time is real, not a hang.** First-step compile for the 2B
  backbone can take 4-5+ minutes with no log output. Verify it's alive via
  growing CPU time / active `ptxas` child processes (`ps aux`), not by
  waiting for output.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` pre-allocates that fraction of GPU
  memory up front — `nvidia-smi` will show most of the 24GB "used" from the
  first step onward regardless of actual load; don't read that as
  near-OOM by itself.
- **SSH keys under `/mnt/c/...` get refused.** WSL mounts Windows drives
  with permissive (0777) permissions, and OpenSSH refuses any private key
  file that open (`bad permissions`, silently ignores the key, then
  `Permission denied (publickey)`). Fix: `cp` the key into WSL's native
  filesystem (`~/.ssh/`) and `chmod 600` it there — applies to any key
  used from WSL, not just Brev's (hit this for MEDIA's `id_ed25519_media`
  too).
- **A background-task/Monitor command that shells out via `wsl.exe` is
  itself running under Git Bash**, not WSL — so `~` and bare
  `/absolute/paths` in the command string get expanded/mangled by Git
  Bash *before* `wsl.exe` ever sees them (e.g. `~/foo.sh` became
  `C:/Users/.../foo.sh`, and `/home/user/foo.sh` became
  `C:/Program Files/Git/home/user/foo.sh`). The fix that's worked
  everywhere in this pipeline: always wrap the whole remote command as one
  string inside `wsl.exe -d Ubuntu -- bash -lc '...'`, and reference any
  Windows-side file via its explicit `/mnt/c/...` path inside that string
  — never as a bare top-level argument.

**Recommendation for next time**: the local `openpi` checkout needs a config
addition (`pi05_mt4_lora` in `src/openpi/training/config.py`) that currently
only exists as a one-off script (`add_mt4_config.py`) run once on the
instance and not committed anywhere durable. **Worth saving that
patch/script into this repo** (e.g. `mt4_pi/collect/openpi_patches/` or
similar) so a fresh instance doesn't require re-deriving the exact
`TrainConfig` fields from openpi's example configs again.

### 4a. Instance went UNHEALTHY mid-run — diagnosis and recovery

Partway through the full run, the instance stopped accepting new SSH
connections entirely — `brev exec`'s own long-lived session died with a
Brev OAuth error ("You are currently logged out"), and every fresh `ssh`
attempt afterward (from this machine *and* independently confirmed from the
user's own terminal) timed out **during the SSH banner exchange**, even
though a raw TCP connect to port 22 still succeeded. `brev ls` eventually
confirmed it plainly: `STATUS UNHEALTHY`.

**Root cause, likely**: running a second, memory-heavy JAX/orbax process
(the LoRA-merge script, see §6a) on the *same* instance *while training was
still running*, with no swap configured. This OOM-killed the merge script
once already (confirmed via `sudo dmesg | grep -i oom`), and a second
attempt is the leading suspect for tipping the whole box into a state where
even `sshd` couldn't fork new connection handlers. **Lesson: never run a
second heavy job on the training instance. Do memory-heavy one-off work
(merging, inspecting) on a different machine — see §6a.**

**Diagnosis steps that actually distinguished "instance dead" from "just a
connection hiccup"** (in order, cheapest first):
1. Raw TCP check: `bash -c 'echo > /dev/tcp/<ip>/22'` — succeeds even when
   SSH itself won't complete the handshake; rules out a fully-down host.
2. Ask the user to try connecting from their *own* terminal — if theirs
   also hangs, it's not an IP-specific rate-limit on your one connection.
3. Brev's web dashboard "History" tab — shows VM lifecycle transitions
   (Pending/Provisioning/Building/Running), *not* CPU/GPU utilization
   graphs (there's no resource monitor there, despite the name). Still
   useful: it confirms whether the VM itself ever restarted.
4. `brev ls` STATUS column — the real signal. `UNHEALTHY` here matched the
   Jupyter-notebook health-check on port 8888 also going unhealthy at the
   same time, i.e. a whole-box symptom, not an SSH-specific one.

**Recovery**: `brev stop <name>` then restart from the web portal (or
`brev start`). This preserves the persistent disk — the in-progress
checkpoint (`full_run/500` at the time) survived intact — but **kills the
training process**; there is no live-migration, just a clean reboot.

**Gotcha after restart: the public IP changes.** Post-restart, the instance
came back with a *different* IP (`35.252.193.87` → `136.67.49.97`), so the
cached `~/.brev/ssh_config` pointed at a dead address (raw TCP connect just
failed outright, a different symptom than the banner-exchange hang above —
useful for telling the two failure modes apart). Fix: `brev refresh`
regenerates `~/.brev/ssh_config` with the current IP. Always run this once
after any stop/start cycle before assuming SSH is still broken.

**Resuming training after the restart** — see §6c.

### 4b. A stopped instance that won't start again (GPU capacity)

Hit on 2026-07-28 starting round 2: `brev start mt4-pi05-lora2` reported
"Instance is starting", then the instance reverted to `STOPPED` within
~90s, and the dashboard showed a red **Start failed** badge. Two attempts,
same result.

**This is a scheduling failure, not a damaged instance.** The distinction
matters because the reflex — delete and recreate — is the one irreversible
option available:

- Reverting to `STOPPED` in ~60-90s means the VM never booted far enough to
  break anything. The persistent disk (and everything on it: `~/openpi`,
  the venv, checkpoints, the patched `config.py`) is untouched.
- The cause is almost always **the cloud provider having no GPU of that
  type free in that zone**. Nothing about your instance is wrong; there is
  simply nothing to schedule it onto.
- **Never hit Delete to "fix" this.** It destroys the disk and costs a full
  environment rebuild (uv, deps, the pinned lerobot commit, and an 11.6 GB
  base-checkpoint re-download).
- A stopped instance bills only for its disk (~$0.02/hr here), which is
  precisely what's preserving the setup. Leave it stopped.

**Recovery options**, cheapest first:
1. **Wait and retry.** Capacity frees up; retries are free since a failed
   start never runs. A retry loop every ~5 min is enough.
2. **Create a *second* instance on a different provider/zone**, leaving the
   original stopped and intact. This is what was actually done — see below.
3. Delete only if you've positively concluded the instance itself is
   wedged, which the 90s-revert signature argues against.

**Choosing the replacement type** — `brev search gpu` lists everything, but
the **`FEATURES` column is the one to read first**: `S`=stoppable,
`R`=rebootable, `P`=pausable. GCP shows `SP`, AWS `SRP`, but the cheapest
options (shadeform resellers — A6000 48GB at $0.68/hr, RTX4090 at $0.72,
both *cheaper than the L4*) show `-`: **delete-only, no stop support**.
That's disqualifying for this workflow, where the whole point of stopping
between rounds is to preserve a ~1hr environment build. Pay the premium for
a stoppable instance.

`brev create` handles the fallback automatically, and defaults to a 500GB
disk (no disk flag needed — `--min-disk` is only a *filter*):

```bash
# --stoppable filters out delete-only providers; --type is an ordered
# fallback chain, tried left to right until one succeeds.
brev create mt4-pi05-r2 --stoppable --min-vram 20 --min-capability 8.0 --dry-run
brev create mt4-pi05-r2 --timeout 900 \
  --type g6.2xlarge,g5.2xlarge,g6.4xlarge,g5.4xlarge,g6e.xlarge,g2-standard-8:nvidia-l4:1
```

Round 2 landed on AWS **`g6.2xlarge`** (L4 22GB, 8 vCPU, 500GB, $1.17/hr) —
same L4 GPU family as the known-good round-1 setup and the same vCPU count,
but a different provider entirely, which is what actually dodges the
capacity problem. Note 22GB vs GCP's 24GB: with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` that's ~19.8GB usable against round
1's observed ~17.5GB peak, so it fits, but with less headroom.

### 4c. Rebuilding the environment from scratch, unattended

`mt4_pi/collect/openpi_patches/setup_openpi.sh` reproduces the whole
round-1 environment on a fresh instance in one detached pass: apt-lock
wait, `uv`, `openpi` clone, `uv sync`, the §3 lerobot/`datasets`/`pyarrow`
pins, HF auth check, the `add_mt4_config.py` TrainConfig patch, and
`compute_norm_stats`. It emits `===MARKER===` progress lines and explicit
`FAIL_*` markers at each failure point, so a monitor can distinguish
"still working" from "died" — see the coverage note in §6c about filters
that only match the happy path.

Two things it needs staged first, neither of which lives in git:
- **`~/.hf_token`** — the dataset is *private*, so `compute_norm_stats`
  can't read it without auth. Copy the local token
  (`huggingface_hub.get_token()`) to `~/.hf_token` on the instance; the
  script installs it to `~/.cache/huggingface/token` and verifies it can
  actually reach the dataset before continuing (a `FAIL_HF_AUTH` here is
  much cheaper than discovering it mid-`compute_norm_stats`).
- **`~/add_mt4_config.py`** — from this repo's `openpi_patches/`.

Launch it detached, never over a live pipe (§6c):
```bash
setsid nohup bash ~/setup_openpi.sh </dev/null >~/setup.log 2>&1 & disown
```

---

## 5. TrainConfig, norm stats, and training

**Round-1 config (`pi05_mt4_lora`, below) is left untouched** — it's what
the currently-deployed step-1999 checkpoint was trained against, and stays
reproducible. Round 2 adds a **second, separately-named** entry
(`pi05_mt4_lora_v3`) rather than editing this one in place — same reasoning
as minting a new dataset `repo_id` instead of overwriting: the old
combination stays reconstructable. Only `data.repo_id` differs between the
two; `num_train_steps` here is just the config default and is overridden
per-invocation by `train.py`'s own `--num-train-steps` flag either way, so
it doesn't need to change in the config itself for a step-count bump alone
— it only earns a new config entry because the *dataset* changed.

Added to (remote) `~/openpi/src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_mt4_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotDROIDDataConfig(
        repo_id="senghoff/mt4-cube-shuffle-v2",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_droid/params"  # 11.6 GiB base checkpoint
    ),
    num_train_steps=2_000,
    batch_size=16,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
),
```

**`senghoff/mt4-cube-shuffle-v2` (and the abandoned `-v1`) were deleted from
HF Hub on 2026-07-28**, once `-v3` was confirmed live and round 2 started
— per user instruction, to stop accumulating superseded dataset repos.
This config block and the stats below are kept as a historical record of
what actually trained the round-1 checkpoint; **the repo_id itself no
longer resolves**, so don't expect `compute_norm_stats.py pi05_mt4_lora`
to work against HF Hub again without re-pushing. A local copy of both
still sits at `data/lerobot_dataset/senghoff/mt4-cube-shuffle-v2` (and
`-v1`) as of the same date, if ever needed for re-upload.

The patch that adds this (previously only a one-off script run directly on
the instance, per the "recommendation for next time" this doc used to
carry) is now saved in this repo at
`mt4_pi/collect/openpi_patches/add_mt4_config.py` — run it once on any
fresh instance's `~/openpi` checkout before the steps below, rather than
re-deriving the `TrainConfig` fields from openpi's example configs again.

Steps, in order:

```bash
export PATH=/home/ubuntu/.local/bin:$PATH
cd ~/openpi

# 1. norm stats (reads the HF dataset, writes assets/<config-name>/<repo_id>/)
#    NOTE: --config-name is a *flag* here on current openpi HEAD. Round 1 ran
#    on an older revision that took it positionally; a fresh clone will fail
#    with "The following arguments are required: --config-name" if you copy
#    the old form. train.py did NOT change -- it still takes the config name
#    positionally (see step 2). Don't infer one script's CLI from the other's.
uv run scripts/compute_norm_stats.py --config-name pi05_mt4_lora_v3

# 2. smoke test — cheap correctness check before spending real money
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mt4_lora \
  --exp-name=smoke_test --overwrite --num-train-steps=50 --no-wandb-enabled \
  --log-interval=5 --save-interval=1000000

# 3. full run
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mt4_lora \
  --exp-name=full_run --overwrite --num-train-steps=2000 --no-wandb-enabled \
  --log-interval=25 --save-interval=500
```

**Smoke test first, always.** 50 steps caught the units bug's absence of a
crash (it would *not* have errored — it would have silently trained a
policy with wrong-scale actions) and confirmed real throughput before
committing to the full run's cost/time.

**Automating the smoke gate.** For an unattended run,
`mt4_pi/collect/openpi_patches/finish_pipeline.sh` chains norm stats →
smoke → gate → detached full run. The gate deliberately checks more than
`exit 0`, because the failure class the smoke test exists to catch (wrong
scale actions) exits 0 happily: it also requires a finite (`nan`/`inf`
rejected) final loss that's actually present in the log and within a sane
range. A gate that only checked the exit code would wave through exactly
the bug this step is for.

Observed on `g2-standard-8:nvidia-l4:1`:
- Steady-state throughput: **~7.5-8.5s/step**, settling to ~7.6s/step after
  the first ~20 steps (first two log ticks are slower — compile still
  amortizing).
- Loss: finite, 1.5-1.9 range at start (expect this to fall over the run;
  hasn't been checked at completion yet).
- `grad_norm`: 8-10, no spikes/NaNs in the smoke test.
- No OOM, despite a tight ~17.5GB/~20.7GB moment during compile.
- 2000 steps × ~7.6s/step ≈ **4.2 hours**, ≈ **$4.30** at $1.02/hr.
- **Watch disk, and don't trust the advertised size.** The round-2 AWS box
  reported a **117GB** root volume despite `brev search`/`create` implying
  500GB. Steady-state consumers: `.venv` ~7.9GB, `~/.cache/openpi` ~12GB
  (the base checkpoint), `~/.cache/huggingface` ~4.7GB (the dataset), plus
  ~8.7GB per saved checkpoint. **The smoke test writes a full ~8.7GB
  checkpoint even with `--save-interval=1000000`** — delete
  `checkpoints/<config>/smoke_test*/` once the gate passes, or it silently
  eats the margin the real run needs (this took free space from 32GB back
  to 40GB). `finish_pipeline.sh` now does this automatically.
- **`CheckpointManagerOptions` defaults to `max_to_keep=1`** — only the
  *most recent* `--save-interval` checkpoint is retained on disk at any
  time, not one per interval. If a run dies late, you get the last
  multiple-of-500 checkpoint before the crash, not a history to roll back
  through. Worth knowing before assuming you can recover an earlier step.

Dataset stats sanity-check (from the real run's logged `norm_stats`,
`senghoff/mt4-cube-shuffle-v2`, 2675 examples / 167 episodes): action means
are near zero with std ~0.25-0.4 rad/s per joint, gripper channel
mean/std ~0.44/0.33 — consistent with a mix of moving and holding, not a
degenerate all-zero or all-saturated action distribution.

---

## 6. LoRA size and deployment implications

From the smoke test's logged parameter shapes:
- Backbone LoRA (`gemma_2b_lora`, rank 16): ≈ 27.87M extra elements.
- Action-expert LoRA (`gemma_300m_lora`, rank 32): ≈ 22.12M extra elements.
- Total ≈ **50M params ≈ 200MB at float32** — under **2%** of the 11.6GB
  `pi05_droid` base checkpoint.

Two deployment shapes, both fitting easily on MEDIA's 8GB 3070 (which
already runs the unmodified 11.6GB-on-disk / ~7.9GiB-loaded base checkpoint
today, per `PI05_DEPLOYMENT.md`):

- **Merged** (LoRA folded into the base weights): identical size/shape to
  `pi05_droid` — a drop-in replacement checkpoint dir for
  `deploy_checkpoint.sh`.
- **Unmerged** (adapter kept separate): ~200MB addition, loaded alongside
  the frozen base at serve time (if the serving code path supports it —
  `[UNVALIDATED]`, hasn't been checked against `pi0.5-server`'s
  `serve_policy.py` yet).

Either way, GPU memory headroom is not a concern for this fine-tune.

### 6a. Merging LoRA into base-shaped params (real script, real math)

There's no merge utility in openpi itself. Confirmed from
`src/openpi/models/lora.py` + the `gemma_2b_lora`/`gemma_300m_lora` variant
definitions in `src/openpi/models/gemma.py`:

```
merged = w + scaling * (lora_a @ lora_b)     # batched matmul over leading axes
gemma_2b_lora:   rank=16, alpha=16.0  -> scaling = alpha/rank = 1.0
gemma_300m_lora: rank=32, alpha=32.0  -> scaling = alpha/rank = 1.0
```

Both variants resolve to `scaling=1.0`, so it's just `w + lora_a @ lora_b`.
Two LoRA parameter shapes appear in the checkpoint and both need handling:
`Einsum`-style (sibling keys `w`/`lora_a`/`lora_b` under one dict) and
`FeedForward`-style (flat siblings `gating_einsum`/`gating_einsum_lora_a`/
`gating_einsum_lora_b`, same for `linear`). A single recursive function
that looks for any `lora_a`-suffixed key and its matching base/`_lora_b`
sibling handles both without hardcoding layer names.

**Load/save uses openpi's own `restore_params`/`ocp.PyTreeCheckpointer`**,
not hand-rolled serialization — mirrors exactly what
`policy_config.create_trained_policy` and `weight_loaders.py` already do,
so the merged output is byte-for-byte structurally what `pi05_droid`'s own
released checkpoint looks like. Gotcha: `ocp.PyTreeCheckpointer().save()`
requires an **absolute path** — a relative `OUT_DIR` fails deep inside
orbax's tensorstore backend with `ValueError: Checkpoint path should be
absolute`, not at the point you'd expect.

**The OOM lesson (the actual cause of §4a's outage):** a first version that
rebuilt the params tree via `{k: merge_lora(v) for k, v in d.items()}`
(out-of-place) held the *entire* original ~11.6GB restored pytree alive
for the whole recursive rebuild, plus per-tensor float32 upcast temporaries
— peaked at ~23GB RSS and got OOM-killed (`dmesg`: `Out of memory: Killed
process ... python3 ... anon-rss:23225540kB`) on a 31GB-RAM, **swap-less**
instance that was also running training (~6.6GB RSS). Rewriting to mutate
the params dict **in place** (`merge_lora_inplace`, `del`-ing old
arrays + `gc.collect()` after each merge) avoids ever holding two full
copies of the tree. **Even so, do this on a different machine than
training** — see below.

**Where to actually run it: not the training instance.** MEDIA already had
everything needed (`/opt/openpi/.venv` has jax/orbax/openpi installed,
since it's what serves `pi05_droid` — check via
`.venv/bin/python3 -c "import jax; import orbax.checkpoint; import openpi"`)
plus 20GB+ free RAM and 8GB swap as a safety net, with no training
competing for memory. Force `JAX_PLATFORMS=cpu` before running the merge
regardless of which machine — the operation is pure array arithmetic, no
GPU needed, and you don't want a second process touching whatever GPU is
in use (training's L4, or MEDIA's already-tight 8GB 3070 if a policy server
happens to be running there).

**Transfer path**: no direct network route from the Brev cloud instance to
MEDIA's private LAN address, so it's a 2-hop relay: Brev instance → local
machine → MEDIA. Only the `params/` subdirectory is needed for
merging/serving (skip `train_state/`, which is ~2x larger and holds
optimizer momentum you don't need outside of resuming training) —
`du -sh .../params` vs `train_state` before copying anything. The
Brev→local hop rode the user's home internet connection (~4-40MB/s,
climbing over the transfer — watch this like the earlier dataset-copy
"is it stalled?" check, since this link has been slow before); the
local→MEDIA hop is LAN-speed (~30-40MB/s), effectively instant for a few
GB.

### 6b. Deploying the merged checkpoint — the norm_stats trap

`deploy_checkpoint.sh <checkpoint-dir> pi05_droid gpu 8000` loads params
fine (merged checkpoint is structurally identical to stock `pi05_droid`,
same `params/` shape) but **serving also needs
`<checkpoint-dir>/assets/droid/norm_stats.json`** — `config=pi05_droid`'s
`asset_id` is hardcoded to `"droid"`, so `serve_policy.py` looks for that
exact path regardless of which checkpoint dir it's pointed at, and fails
with `FileNotFoundError` if it's missing.

**This is not just a missing file — using the wrong norm_stats would be a
silent correctness bug.** The fine-tune trained against *our own* MT4
dataset's norm stats (written by `compute_norm_stats.py` to
`~/openpi/assets/pi05_mt4_lora/<repo_id>/norm_stats.json` on the training
instance — the exact same file logged as `Loaded norm stats from ...` at
training start). Copying the *original* DROID/Franka norm stats into that
path instead would un-normalize every action at the wrong scale — the
model would produce plausible-shaped but wrong-magnitude commands, with no
error or crash to flag it. **Always fetch and place our own
`norm_stats.json`** at `<merged-checkpoint>/assets/droid/norm_stats.json`,
never assume the config's default assets are correct for a fine-tune.

**A failed first warm-up call is not necessarily a broken checkpoint.**
`deploy_checkpoint.sh`'s own post-deploy warm-up call failed once with a
"1011 keepalive ping timeout" (the documented first-call-JAX-compile-vs-
default-websocket-timeout race from `PI05_DEPLOYMENT.md`). The *server*
process was still alive and `/healthz` still passed — only the optional
warm-up call itself timed out. But real inference calls right after were
all uniformly ~6.5s (not dropping after the first, as a one-time compile
cost would) — genuinely degraded, not just an unwarmed first call.
**Fix: redeploy clean** (`NO_WARMUP=1`, then warm it up yourself via
`mt4_pi.policy_client`) — this resolved it completely: first real call ate
the full ~26s compile, every call after was ~300ms, identical to stock
`pi05_droid`'s steady-state latency. Root cause of the degraded state
wasn't chased further; a clean redeploy is cheap enough not to bother.

**This recurs — it's non-deterministic timing, not a one-off.** Deploying
the *final* (step-1999) checkpoint hit the exact same race again: even with
`NO_WARMUP=1` and `mt4_pi.policy_client`'s own `ping_interval=-1.0`
workaround already in place, the very first call still occasionally times
out with `1011 keepalive ping timeout` (confirmed via the server's own log
that the checkpoint + norm_stats loaded fine and the connection opened —
it's specifically the first inference call's client-side keepalive racing
the compile, not a loading/config problem). Simply retrying the client
against the *same* still-running server reproduces the degraded ~6.5s/call
plateau again, same as before — retrying alone does not fix it. **The only
reliable fix found so far is a full redeploy** (kill + fresh
`deploy_checkpoint.sh` invocation), which has now cleanly resolved this
twice. Budget for possibly needing one redeploy-and-retry cycle every time
you stand up a fresh server, rather than assuming the first attempt will
land cleanly.

**Verify with real inference, not just `/healthz`**:
```bash
python -m mt4_pi.policy_client --host 192.168.1.3 --port 8000 --iters 3
```
Expect `(15, 8)` finite action chunks; steady-state latency should match a
freshly-deployed stock `pi05_droid` on the same hardware (redeploy the
stock checkpoint side-by-side if in doubt — this is exactly how the ~6.5s
regression above was caught, by comparing against a known-good baseline
rather than trusting a single absolute number against documentation).

### 6c. Resuming training after an interruption

`scripts/train.py` has an explicit `--resume` flag ("resume training from
the last checkpoint", default off) — check with
`uv run scripts/train.py <config> --help`. **Do not pass `--overwrite`**
when resuming — that flag wipes the checkpoint directory you're trying to
resume from. Keep the same `--exp-name` (it's how `train.py` finds the
existing `checkpoints/<config>/<exp-name>/` dir to resume from), keep the
same `--num-train-steps` (it's the *total* target, not "how many more").
The log confirms a real resume with `'step': 500, 'event_type': 'restore'`
and the `tqdm` counter starting from `501/2000` rather than `0/2000`.

**Launch it detached this time** — `setsid nohup ... </dev/null
>logfile 2>&1 & disown`, writing to a log *file* rather than piping over a
live SSH/`brev exec` connection. This is the direct fix for §4a's outage:
the original run's local `brev exec | tee` pipe dying took the *connection*
down, and while the training process itself likely wasn't killed by that
alone (it had no controlling TTY), every subsequent log line the training
script tried to print risked a `BrokenPipeError` on the now-dead channel —
Python's `logging` module swallows that safely, but plain `print()` calls
(which `train.py` uses for its periodic `Step N: loss=...` lines) do not,
and an unhandled exception there would kill the whole run. Detaching with
`setsid`/`nohup`/`disown` and writing to a file removes this dependency on
the local connection entirely — poll the remote log file periodically
instead of tailing a live pipe.

---

## 7. Remaining work

Done: the round-1 2000-step run completed cleanly, the final (step-1999)
checkpoint was merged and deployed to MEDIA and verified via real
inference, and real-arm validation (§7a) actually ran — the missing piece
flagged here previously. `mt4-pi05-lora2` was stopped (not deleted) between
rounds to end billing while preserving the environment.

1. **Round 2**: combined shuffle+stack dataset (`senghoff/mt4-cube-shuffle-v3`),
   4000 steps, to address §7a's targeting/timing findings with more data
   and more task diversity. Merge on MEDIA, deploy, re-validate on the real
   arm exactly as in §7a — a better loss curve alone would not be evidence
   this actually fixed anything.
2. If round 2 still mis-targets: the next lever is almost certainly *more
   and more-varied* demonstrations (lighting, cube starting positions,
   session diversity per §2's collection tips) rather than more training
   steps on the same data — loss was already down to ~0.02 in round 1,
   so the model fit the training distribution fine; the failure is
   generalization/precision, not underfitting.

### 7a. Real-hardware validation results (round 1, 2026-07-28)

First-ever real-arm runs of a fine-tuned checkpoint, both with
`client.queue_move()` enabled in `mt4_pi/runtime.py` (previously
deliberately inert) and the safety layer (`mt4_pi/safety.py`) active
throughout — zero safety-layer rejections in either run, motion was smooth
and continuous in both.

- **Run 1** (2 ticks / 16 waypoints, prompt `"put the blue cube on the
  marker"`): smooth descent, but the gripper closed at z≈171mm — well
  above table height, in midair, with no cube there. A premature grasp.
- **Run 2** (8 ticks / 64 waypoints, gripper explicitly opened first to
  match the training distribution's episode-start convention — episodes
  are collected starting from `retreat_for_camera`'s camera-park pose with
  the gripper open, per `collect_shuffle.py`): smooth continuous descent
  from camera park (200,0,260) down to table height (230.5, 16.8, 134.6),
  gripper stayed open the whole time — never grasped. Checked the scene
  afterward with `scripts/validate_scene_live.py --move`: the 3 detected
  blue cubes (at robot-frame (114,-167), (43,175), (31,-199)) were
  untouched, and none is anywhere near where the arm ended up — a
  150-230mm miss, not a near-miss.

**Ruled out as the cause** (so this reads as a model-quality/data-volume
issue, not a wiring bug): prompt phrasing exactly matches the
`_TO_MARKER` templates used at collection time; the camera index
(`capture_frame`'s `DEFAULT_CAMERA_INDEX`) is the same shared default used
at collection time, no override drift; the observation/action unit
conventions (radians, `_prep_image`'s resize_with_pad/BGR→RGB) were already
validated when the base `pi05_droid` checkpoint was confirmed producing
dimensionally-valid finite outputs and the fine-tune's own `norm_stats`
were confirmed correctly wired (§6b) — none of that changed between
deployment-verification and this test.

**Read on the result**: the fine-tune clearly learned the demonstrations'
*motion style* (smooth, plausible reach-and-descend shape, not noise) but
not yet reliable *grasp targeting or timing* — consistent with a small
fine-tune (2000 steps on one collection session's ~167-episode shuffle-only
corpus) that hasn't seen enough visual/spatial diversity to generalize
precisely. This is the direct motivation for round 2 (§0, §5): more data
(stacking demos added), doubled step count.

---

## See also

- [`PI05_DEPLOYMENT.md`](PI05_DEPLOYMENT.md) — MEDIA server topology, start/stop/verify
- `mt4_pi/adapter.py` — action → waypoint mapping, units
- `mt4_pi/observation.py` — observation dict shape, image prep
- `mt4_pi/runtime.py` — the control loop that actually drives the arm
- `mt4_pi/collect/recorder.py`, `mt4_pi/collect/convert_to_lerobot.py`,
  `mt4_pi/collect/prompts.py` — collection + conversion
- `mt4_pi/collect/openpi_patches/add_mt4_config.py` — the remote
  `TrainConfig` patch, committed here instead of re-derived per instance
- `collect_shuffle.py`, `collect_stack.py` — the two demo collectors
- `tests/test_collect_waypoints.py` — `WaypointLog` interpolation behavior
