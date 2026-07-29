# ACT training pipeline for the MT4

**Status (2026-07-29):** pipeline verified end to end on MEDIA — dataset
packed (13,762 frames / 110 episodes) and a 200-step training run completed
at 5.9 step/s with loss falling 24.09 → 3.05. No full training run has been
launched yet. The 225-episode export is also on the Hub at
`senghoff/mt4-act-export-v1` (private, 462 files, verified). Nothing has been
started on RunPod, and on these numbers nothing needs to be — see §2a.

ACT (Action Chunking Transformer) trains from scratch on our own data. That
is the whole reason it is here: every compromise the pi0.5 path had to make
existed to keep a DROID-pretrained checkpoint happy, and none of them survive
contact with a method that has no pretraining distribution to violate.

| | pi0.5 (`mt4_pi/`) | ACT (`mt4_pi/act/`) |
|---|---|---|
| Action | joint velocities, rad/s | **absolute joint targets, rad** |
| Action dim | 4 padded to 7 padded to 32 | **5** (4 joints + gripper) |
| State dim | 4 padded to 7 | **14** (4 joints + gripper + task) |
| Wrist camera | zero-filled tensor | **absent** |
| Assumed rate | 15 Hz (wrong; see below) | **10 Hz, measured** |
| Task conditioning | language prompt | one-hot columns in the state |
| Params | ~3B | ~80M |
| Gripper | binarised at 0.5 -> servo 285 | continuous inverse -> servo 140..255 |

## Why the design differs where it does

### Actions are absolute targets, not velocities

The pi0.5 labels were per-tick deltas divided by `DT_S`. That single choice
produced three separate bugs, and all three vanish here:

- **`DT_S` was wrong.** It was `1/15`, but the recorder actually ticks at
  ~10 Hz — measured mean 0.1003 s, median 0.0940 s over 44,475 ticks. Every
  velocity label was ~1.5x too large. Worse, execution runs at ~4.5 Hz
  because each `queue_move` costs ~204 ms on the serial link, so even a
  correctly scaled velocity would have been integrated over the wrong
  interval. An absolute target does not care how long the trip takes.
- **Integration drift.** Velocities accumulate error against a live state
  estimate. An absolute target is wrong once.
- **The ±1.0 rad/s clip** in `adapter.integrate_joint_target` was silently
  throttling predictions, since the model's outputs ran 2–5x larger than
  truth. There is nothing to clip here beyond the joint limits.

It also makes static-frame trimming *safe*: dropping a frame and re-pointing
the previous label at the next survivor (`pack_lerobot.retarget_actions`) is
exact for a target and meaningless for a velocity.

### Task conditioning rides in the state vector

ACT has no language input. Our corpus is not one task — 4 place kinds across
3 cube colours — and given the same scene, "put the red cube on the marker"
and "take the top block off the stack" demand different behaviour. Trained
unconditioned, ACT could only learn the marginal over tasks.

Rather than split 225 episodes into per-task sets of ~50, or modify the
architecture, the task is appended to `observation.state` as one-hot columns.
ACT treats state as an opaque vector, so this is task conditioning at zero
model-code cost.

The goal placement `(x, y)` is included because it is genuinely available at
inference time — `mt4_vision` computes the free slot. The **pick** location
deliberately is not: finding the cube to grasp is the part we want learned
from pixels.

Constant columns are pruned at pack time (`schema.prune_constant_columns`),
because a zero-variance column divided by `std + 1e-8` is a landmine, and a
constant column carries no information anyway. The surviving layout is
recorded in `mt4_act_meta.json` and **must travel with the weights** — the
serving side cannot rebuild the input vector without it.

### The gripper has no threshold

The pi0.5 adapter binarised column 7 at 0.5 and snapped closed to
`GRIPPER_S_CLOSED = 285`. Both were wrong for this data: the labels are
bimodal at **0.121 and 0.818**, so 0.5 is not the midpoint, and 0.818 is
servo **255**, not 285 — the old path commanded a harder squeeze than any
demonstration used. `act/adapter.gripper_target` is an exact continuous
inverse of `gripper_fraction`, so it simply replays grips that already
worked. No threshold to get wrong.

## Data quality: the stack corpus is largely unusable as labels

Measured over the export, per-step joint discontinuities:

| | median worst jump | p90 | max | episodes >30° |
|---|---|---|---|---|
| shuffle (`pi_demos`) | **3.6°** | 5.2° | 16.7° | **0 / 111** |
| stack (`pi_stack_demos`) | **72.3°** | 131.2° | 171.0° | **88 / 114** |

### Root cause (found and fixed 2026-07-29)

An earlier version of this section blamed the retreat to camera park. That
was wrong: the retreat runs after `end_recording()`, so it never labels a
tick. The real mechanism is per-leg timing inside a queued path.

A stacking carry goes out as **one** `mq` call covering five or six legs
(lift, route, hop, descend), so there is no per-leg completion time to
observe — `_emit_path_waypoints` splits the single measured duration back
across the legs afterwards. It split by *Cartesian distance × speed*.

`routed_travel` holds the wrist (`"wrist"` sentinel) through the lift and
applies the face-align angle from the **next** leg onward. On a level-1 carry
that next leg is the 1.2 mm nudge from `safe_z` to hover height — 0.2% of the
path by distance, so it was allotted **0.03 s of a 12 s carry**, while the
80.8° wrist sweep it carries is 3634 motor steps, the largest single piece of
work in the whole move. Hence a one-tick 80° discontinuity in every stacking
episode, and up to 171° once a long J1 swing rode the same leg.

Legs are now weighted by `steps × speed_us` on the busiest axis
(`_leg_step_counts`), which is what the firmware's DDA actually paces:
per-waypoint IK for J1–J3, and J4 counted on the *joint* (a concrete world
angle commands `j4 − j1`; the `"wrist"` sentinel holds the joint across a J1
swing). Distance weighting remains only as a fallback when IK cannot solve.

Validation: the step model predicts 7.42 s for a carry measured at 9.00 s
(18% low, all of it the unmodelled accel ramp — which can only add time).
Distance weighting was off by ~130× on the wrist leg. A fresh 6-episode
hardware cycle at the same site worst-jumps **4.7°**, below the 5° safety cap.

**Consequence:** the recorded stack corpus is still unusable and is not
recoverable — only the interpolated ticks were stored, not the raw waypoint
log, so the timing cannot be re-derived. The 88 bad episodes are quarantined
under `data/pi_stack_demos/rejected/label_jump/` by
`python -m mt4_pi.collect.prune`. Stacking data must be **re-collected**.
`fetch_and_pack.sh` still defaults to `MT4_FAMILY=shuffle` and
`MT4_MAX_JUMP_DEG=15`; keep the jump filter on regardless, as a tripwire.

This is also worth revisiting for pi0.5: round 2 trained on the merged
corpus, so roughly half its episodes carried these labels.

Aside from that, the corpus is 225 successful episodes (of 268 recorded, 84%
success), 37,852 usable frames, ~81 minutes, 10.2% of frames fully
stationary.

## Collecting a balanced corpus

`collect_shuffle.py --balance N` collects toward N successful episodes per
(colour, place-kind) cell, using `mt4_pi/collect/balance.py` instead of
`mt4_vision.policy.plan_shuffle`.

```bash
python -u collect_shuffle.py --out data/pi_demos --balance 40 --episodes 300
```

The skew in the original corpus was structural, not bad luck:

* `plan_shuffle` only reaches `to_slot` when there are **no free markers**.
  With 5 markers that essentially never happens — 7 `to_slot` against 104
  `to_marker`.
* Colour is `random.choice` over pickable cubes, so a colour that spends the
  session parked on a marker barely appears. Blue got 5 episodes, red 58.

The balanced planner enumerates marker and slot destinations *together* and
takes the move whose cell is furthest below target.

### The far-slot trap

**Five of the eight `PLACEMENT_SLOTS` are reachable but not verifiable**, and
using them is actively destructive. `MAX_VERIFIABLE_RADIUS_MM = 240` in
`balance.py` filters them out.

| slot | radius | usable |
|---|---|---|
| (150, 100) | 180 | yes |
| (200, ±60) | 209 | yes |
| (240, ±150) | 283 | **no** |
| (280, 0) | 280 | **no** |
| (150, ±250) | 292 | **no** |

Every reliably-detected ArUco marker sits at r = 211–236 mm. Beyond that the
cube falls outside the camera's detection hull, and the cost is doubled:

1. The placement *succeeds mechanically* — the cube lands exactly where
   commanded — but `capture_scene` cannot see it, so `verify_pick_place`
   returns `lost` and a perfectly good demonstration is stamped
   `success=False` and dropped by the converter.
2. The cube is now invisible to the planner and cannot be picked again.

Measured: four consecutive far-slot placements removed **4 of 7 cubes** from
the table before anything looked wrong from the log alone. Recovery is
open-loop against the known slot coordinates (vision cannot help — being
invisible is the problem), with `face_align=False` since there is no
detection to read yaw from.

Because only three slots survive the filter, the planner can run out of
`to_slot` destinations. It then emits a **filler** `to_marker` move rather
than waiting, which lifts a cube off a slot and frees it for the next cycle;
waiting would deadlock, since the scene only changes when the arm moves.

Goal diversity is capped by physical layout: 5 markers + 3 verifiable slots =
8 sites, so the `goal_x/goal_y` state columns are memorised rather than
generalised. Widening that needs the markers physically repositioned and
re-calibrated.

## Running it

### 1. Export (workstation, done)

```bash
python -m mt4_pi.act.export_episodes --out data/act_export
python -m mt4_pi.act.push_export --repo-id senghoff/mt4-act-export-v1
```

Reads the raw traces, solves IK, downsamples 1280x720 -> 640x360 and encodes
one H.264 clip per episode. **271 MB, down from 10.1 GB of JPEGs** — a 37x
reduction, which is why the Hub hand-off is cheap.

The export bundles `schema.py`, `pack_lerobot.py` and the RunPod scripts
under `code/`, so the artefact is self-describing: whatever produced these
arrays is what reads them back.

### 2a. Train on MEDIA (measured — this is the recommended route)

MEDIA is an RTX 3070 (8 GB), Ryzen 7 5700G, 8 cores, 21 GB RAM, 898 GB free,
Ubuntu 22.04. lerobot 0.4.4 is installed at `/root/act_venv` (uv-created,
Python 3.11, isolated from `/opt/openpi/.venv`).

The whole pipeline has been run there end to end and measured:

| | result |
|---|---|
| Export copied over LAN | 271 MB in 6.2 s (43.5 MB/s) |
| Pack (shuffle, jump filter) | 13,762 frames / 110 episodes, 1 excluded |
| State columns kept | 12 (`kind_stack`, `kind_unstack` pruned) |
| Dataset random read | 5.0 ms/sample, 200 samples/s single-threaded |
| **Training, batch 16** | **5.9 step/s, 4.07 GB peak VRAM** |
| Training, batch 32 | 2.4 step/s, 7.02 GB peak — *slower per sample* |
| Step breakdown | `updt_s 0.162`, `data_s 0.005` — GPU-bound, 3% dataloading |
| Loss over 200 steps | 24.09 → 3.05 |

At 5.9 step/s and ~860 steps per epoch:

- **50k steps ≈ 2.4 h** (58 epochs)
- **100k steps ≈ 4.7 h** (116 epochs)

Use batch 16. Batch 32 fits but is worse on both axes — fewer samples/s and
only 1 GB of headroom.

```bash
# from the workstation, push the export over the LAN (no HF auth on MEDIA)
rsync -a data/act_upload/ root@192.168.1.3:/root/mt4act/export/

# on MEDIA
V=/root/act_venv; W=/root/mt4act
PYTHONPATH=$W/export/code $V/bin/python -m mt4_pi.act.pack_lerobot \
    --export $W/export --out $W/dataset --repo-id local/mt4-act-shuffle \
    --family shuffle --max-jump-deg 15
MT4_ACT_WORK=$W MT4_DATASET_REPO=local/mt4-act-shuffle \
    PATH=$V/bin:$PATH bash $W/export/code/runpod/train.sh
```

**`apt install ffmpeg` is required.** torchcodec links against FFmpeg's shared
libraries and dies with `libavutil.so.56: cannot open shared object file`
without them. The failure is confusing because packing *succeeds* — only
reading a frame back fails. `setup.sh` now checks for this.

Note the pi0.5 server cannot run at the same time: it holds ~7.7 GB of the
3070's 8 GB. Training at batch 16 needs 4.07 GB, so the two do not coexist.

### 2b. Train on RunPod

Only start the pod once step 1 is verified complete — it bills by uptime, not
by GPU use, and there is nothing for it to do while data uploads.

```bash
hf download senghoff/mt4-act-export-v1 --repo-type dataset --local-dir export
cd export/code/runpod

bash setup.sh
MT4_EXPORT_REPO=senghoff/mt4-act-export-v1 bash fetch_and_pack.sh
bash train.sh smoke      # ~200 steps; catches dimension/decode/dataloader failures in a minute
bash train.sh            # the real run
MT4_WEIGHTS_REPO=senghoff/mt4-act-v1 bash push_weights.sh
```

Everything lands on the **container disk**, never `/workspace`. RunPod
network volumes are MooseFS over FUSE; DataLoader workers that seek into a
dataset stored there hang indefinitely rather than erroring, and the usual
diagnostics (`df`, `nproc`, `free`, loadavg) all report host-level or bogus
values inside the container.

Push weights through the Hub rather than pulling over SSH. Measured on the
pi0.5 work: ~2.3 MB/s over SSH against ~263 MB/s via the Hub, roughly 110x,
and it decouples the transfer from the instance's uptime entirely.

**Hyperparameters that must not be copied from an ALOHA recipe:**
`chunk_size` and `n_action_steps`. ALOHA's defaults (100/100) are tuned for
50 Hz, where 100 steps is 2 s. This dataset is 10 Hz, so 100 would be 10 s —
most of an entire episode, open-loop from the first frame. Defaults here are
`chunk_size=32` (3.2 s, the horizon ALOHA actually uses) and
`n_action_steps=8` (0.8 s trained, ~1.6 s wall clock at 204 ms per queued
waypoint).

### 3. Serve and run on the arm

```bash
# on the GPU host (MEDIA)
python -m mt4_pi.act.serve --checkpoint /path/to/pretrained_model --port 8100

# on the workstation
python -m mt4_pi.act.client --host 192.168.1.3      # smoke test, no arm needed
```

Plain HTTP with a base64 JPEG, not the pi0.5 WebSocket protocol — that exists
to survive keepalive timeouts across a 300 ms JAX call, and ACT infers in
~10–20 ms. The client needs only stdlib, numpy and cv2; the arm-side machine
should not need torch or lerobot installed.

`mt4_pi/act/runtime.py` is the control loop. It takes the task as structured
arguments (`color`, `place_kind`, `goal_xy_mm`) rather than a prompt, because
conditioning is in the state vector.

## Safety

The existing `mt4_pi/safety.py` is reused unchanged and turns out to be
well-calibrated for absolute targets at 10 Hz — measured against all 37,852
real transitions, **99.4% pass**:

| cap | value | real p99 | fraction exceeding |
|---|---|---|---|
| `MAX_JOINT_STEP_DEG` (j1–j3) | 5.0° | 3.2–4.1° | 0.44–0.63% |
| `MAX_J4_STEP_DEG` | 10.0° | 3.1° | 0.17% |
| `MAX_TCP_STEP_MM` | 15.0 mm | 10.5 mm | 0.58% |

The transitions that exceed the caps are precisely the stack-corpus
discontinuities described above.

`runtime.py` adds one check the pi0.5 loop did not need: if the first row of
a chunk is more than 20° from where the arm actually is, the policy is not
tracking the arm and the loop stops. With absolute targets that is a
meaningful, cheap sanity test.

## Normalisation lives outside the policy (lerobot 0.4.x)

**Any code that loads an ACT checkpoint must also load its processors.**

`ACTPolicy` in lerobot 0.4.4 has no normalize modules — `named_children()` is
just `model`, and there is no `normalize_inputs`/`unnormalize_outputs`.
Normalisation was moved into a pair of pipelines saved beside the weights:

```
policy_preprocessor.json   policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor.json  policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

```python
from lerobot.policies.factory import make_pre_post_processors
policy = ACTPolicy.from_pretrained(model_dir)
pre, post = make_pre_post_processors(policy.config, pretrained_path=model_dir)
chunk = post(policy.predict_action_chunk(pre(batch)))
```

`predict_action_chunk` applies neither, so calling it directly feeds raw
tensors to a model trained on normalised ones *and* reads normalised outputs
back as if they were radians. This is not a subtle degradation, and it is
nastily plausible-looking: the raw outputs land in roughly the right numeric
range (j2 ≈ 1.56 against a true 1.60 rad), so nothing looks obviously wrong.

Measured on the same 20k-step checkpoint:

| | h=1 | h=8 | h=32 |
|---|---|---|---|
| without processors | 21.1° | 23.6° | 34.5° |
| **with processors** | **3.2°** | **3.1°** | **3.5°** |
| hold-pose baseline | 0.65° | 3.14° | 9.83° |

Without them the model looks like it learned nothing. With them it beats the
baseline 2.8x over the full chunk.

## Held-out result (20k steps, 80 train / 20 val episodes)

| split | h=1 | h=8 | h=16 | h=32 |
|---|---|---|---|---|
| train | 3.15° | 3.13° | 3.26° | 3.47° |
| val | 3.96° | 4.16° | 4.62° | 5.57° |
| hold baseline | 0.85° | 3.48° | 6.34° | 10.90° |

Reading:

* **It learns.** Val beats the trivial baseline 2x at the full chunk horizon
  and 1.4x at h=16.
* **There is a real generalisation gap** — 3.47° train vs 5.57° val at h=32,
  about 1.6x. That is the "more data" signal, and it is what the balanced
  collection is for.
* **Error is almost flat across the horizon** (~3-4° at every step), so the
  residual is a roughly constant per-step offset rather than drift. The model
  knows where the trajectory goes; it is imprecise step to step.
* **At the executed horizon (8) it is only break-even with "don't move."**
  That matters for deployment: `EXECUTION_HORIZON = 8` sits exactly where the
  policy has the least advantage. Executing a longer slice, or enabling ACT's
  temporal ensembling, is worth trying before blaming the policy.
* 10k -> 20k improved val h=32 only 5.70° -> 5.57°, so it is close to
  converged. More steps will not close the gap; more and better-balanced data
  might.

## Open questions

1. **Does it learn at all?** This is the point of the exercise. If ACT
   succeeds on 110 clean shuffle episodes, pi0.5's trouble was embodiment and
   pretraining mismatch. If ACT also fails, the problem is the data — the
   interpolated state or the missing wrist view — and no model choice
   rescues it.
2. **The state labels are partly fiction.** `state_source` is
   `commanded_linear_interp` for all 268 episodes: poses are interpolated
   between commanded waypoints, not measured encoder feedback. Every method
   trained on this inherits it.
3. **Still no wrist camera.** ACT degrades gracefully without one where
   pi0.5 fell off a cliff, but close-range grasping from a single steeply
   oblique view is genuinely hard.
4. **Dead time.** 10.2% of frames are fully stationary, an artefact of the
   collector waiting on serial round-trips. `--max-static-run` trims it; left
   off by default so the baseline stays faithful.
5. **MCU resets on COM10** remain the dominant obstacle to *any* hardware
   validation — 3 in one session, each destroying the J4 zero, which has no
   home switch and needs manual re-alignment.

## Files

| Path | Runs where | Needs |
|---|---|---|
| `mt4_pi/act/schema.py` | anywhere | numpy |
| `mt4_pi/act/export_episodes.py` | workstation | `mt4_jog`, ffmpeg |
| `mt4_pi/act/pack_lerobot.py` | GPU box | `lerobot`, cv2 |
| `mt4_pi/act/push_export.py` | workstation | `huggingface_hub` |
| `mt4_pi/act/serve.py` | GPU host | torch, `lerobot`, cv2 |
| `mt4_pi/act/client.py` | workstation | numpy, cv2 |
| `mt4_pi/act/adapter.py` | workstation | `mt4_jog` |
| `mt4_pi/act/runtime.py` | workstation | the above + `mt4_vision` |
| `mt4_pi/jointstate.py` | anywhere | `mt4_jog` |

`pack_lerobot.py` probes for both `lerobot.datasets` and
`lerobot.common.datasets` and drops `create()` kwargs the installed version
does not accept. This repo already pins one lerobot to the exact commit
openpi vendors; version-probing beats pinning a second one and
re-litigating it at every upgrade.
