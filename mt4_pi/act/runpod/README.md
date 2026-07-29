# RunPod scripts for MT4 ACT training

Run in order. Full rationale in `docs/ACT_PIPELINE.md`.

```bash
bash setup.sh                                              # deps + GPU check
MT4_EXPORT_REPO=senghoff/mt4-act-export-v1 bash fetch_and_pack.sh
bash train.sh smoke                                        # ~200 steps, ~1 min
bash train.sh                                              # the real run
MT4_WEIGHTS_REPO=senghoff/mt4-act-v1 bash push_weights.sh
```

**Do the smoke run.** It costs about a minute and catches the dimension,
decode and dataloader failures that otherwise surface an hour into a run, on
a clock that bills by uptime.

## Two things that will bite

**Never put the dataset or a cache on `/workspace`.** RunPod network volumes
are MooseFS over FUSE. DataLoader workers that seek into a dataset stored
there hang indefinitely rather than erroring, and the container's own
diagnostics lie — `df`, `nproc`, `free` and loadavg all report host-level or
bogus values. Everything here defaults to `/root/mt4act` on the container
disk. Use `/workspace` only as an archive you copy to and from.

**Move weights through the Hub, not SSH.** Measured on the pi0.5 work: ~2.3
MB/s pulling over SSH against ~263 MB/s via the Hub, roughly 110x. It also
decouples the transfer from the instance's uptime, so the pod can be stopped
as soon as `push_weights.sh` verifies.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `MT4_ACT_WORK` | `/root/mt4act` | container disk, not `/workspace` |
| `MT4_EXPORT_REPO` | *(required)* | HF dataset repo holding the export |
| `MT4_DATASET_REPO` | `local/mt4-act` | local id for the packed dataset |
| `MT4_FAMILY` | `shuffle` | `shuffle` \| `stack` \| `all` |
| `MT4_MAX_JUMP_DEG` | `15` | exclude episodes with discontinuous labels |
| `MT4_MAX_STATIC_RUN` | *(unset)* | trim runs of stationary frames |
| `MT4_STEPS` | `100000` | |
| `MT4_BATCH_SIZE` | `32` | |
| `MT4_CHUNK_SIZE` | `32` | 3.2 s at 10 Hz — **not** ALOHA's 100 |
| `MT4_N_ACTION_STEPS` | `8` | executed before re-planning |
| `MT4_NUM_WORKERS` | `8` | explicit; `nproc` is unreliable here |
| `MT4_WANDB` | `false` | |
| `MT4_WEIGHTS_REPO` | *(required by push)* | HF model repo |

## Why `family=shuffle` by default

The stack corpus has a recording defect: its labels contain single-step joint
jumps with a median worst case of 72° (up to 171°), against 3.6° for shuffle.
A stack episode retreats to the camera park between phases and the recorder
interpolates that traverse as one transition, producing a label the arm's
safety layer caps at 5° and would refuse to execute. 88 of 114 stack episodes
are affected, against 0 of 111 shuffle episodes.

Training on `all` needs the recorder fixed, not a flag change.

## Cost discipline

Data first, pod second. The export is already on the Hub and verified — the
pod should find its data waiting rather than being billed to watch an upload.
Stop the pod as soon as `push_weights.sh` reports the weights verified.
