# SAM 2.1 server — setup

Prompted segmentation for the MT4 desk camera. The segmenter is a small
FastAPI service wrapping `facebook/sam2.1-hiera-small`; the arm side talks to
it over HTTP and needs nothing else.

**Every measurement that starts from a detector box goes through it**, so it is
not an optional extra for the open-vocabulary half of this repo — see
[What depends on the service being up](#what-depends-on-the-service-being-up).
Cube pick/place, stacking, calibration and the jog client do not touch it.

**What it buys you:** a point or a box becomes a *silhouette*. Grounding DINO
says the pen is somewhere in this rectangle; this says which pixels are pen. On
the reference desk a click on the Statue of Liberty figurine returns its outline
including the raised torch — a shape no HSV cube detector and no bounding box
describes.

---

## Where to run it

The service is just an HTTP endpoint, so put it wherever your GPU is. Pick the
row that matches your hardware:

| Your setup | Where the service runs | Client URL |
|---|---|---|
| **GPU in the machine driving the arm** | Same machine, on loopback | `http://127.0.0.1:8767` (the default — nothing to configure) |
| **GPU on another host** | That host, reached by SSH tunnel or a LAN bind | Tunnel: the default. LAN bind: `http://<gpu-host>:8767` |
| **No GPU** | Same machine, on CPU | The default, with `SAM2_DEVICE=cpu`. Expect ~1 s per frame |

```
                                       ┌─────────────────────────────┐
 mt4_vision.sam                        │ server.py (uvicorn)         │
   MT4_SAM_URL           HTTP          │   GET  /health              │
   ─────────────────────────────────>  │   POST /embed               │
                                       │   POST /segment             │
 same host, or across the network      │   sam2.1-hiera-small, fp16  │
                                       └─────────────────────────────┘
```

The only thing the arm side needs to know is `MT4_SAM_URL` (default
`http://127.0.0.1:8767`). Everything below is about standing the service up.

---

## Install

Needs Python 3.10+ and, for useful speed, a CUDA GPU. The model is small — 38.5
million parameters, 74 MiB of weights in fp16 and 363 MiB at peak — so it fits
beside a much larger model on the same card. On the reference 8 GB card it runs
alongside Qwen3-VL with 2 GB still free.

Copy `server.py`, `requirements.txt` and (on Linux) `sam2.service` from
[services/sam2/](../services/sam2/) to wherever you want it to live, then:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # several GB of wheels, mostly torch

# check the GPU is visible before going further
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# first run downloads ~180 MB of weights into the HuggingFace cache, then
# spends ~100 s compiling the image encoder
.venv/bin/python server.py                     # Ctrl-C once it logs "Uvicorn running"
```

Do that first run in the foreground, before putting the service under a
supervisor. It pays for both the weight download and the first compile, and a
restart-on-failure supervisor turns either into what looks like a crash-loop.
Once both are cached the service starts in about 7 s, offline.

`requirements.txt` is deliberately unpinned. `transformers` must be ≥ 4.57,
which is where `Sam2Model` and `Sam2Processor` arrived. `torchvision` and
`accelerate` are listed because transformers reaches for them internally —
torchvision on the fast image-processor path that GPU preprocessing needs —
though `server.py` imports neither directly.

The checkpoint is published as a video model, so loading it into the image-only
`Sam2Model` logs a note about instantiating a `sam2` from a `sam2_video`
checkpoint. That is expected: the image model is a subset of the video one, and
the weights it needs all load.

### Keeping it running

On a Linux or WSL2 host with systemd, use the bundled unit:

```bash
install -m 644 sam2.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now sam2.service
systemctl status sam2.service
curl http://127.0.0.1:8767/health
```

Edit `WorkingDirectory`, `ExecStart` and `TORCHINDUCTOR_CACHE_DIR` in the unit
if you did not install to `/opt/sam2`. `enable` is what makes it come back on
its own after the host reboots. On Windows or macOS there is no unit file — run
`server.py` under whatever supervisor you already use (NSSM, a Scheduled Task,
`launchd`, a terminal you leave open).

A healthy `/health` reports `ok=True`, `loaded=True`, `device=cuda`, and the
compile status.

### Configuration

All environment variables; there is no config file:

| Variable | Default | Notes |
|----------|---------|-------|
| `SAM2_HOST` | `127.0.0.1` | Loopback. See [Reaching another host](#reaching-a-service-on-another-host) before changing it |
| `SAM2_PORT` | `8767` | Must match whatever the client's `MT4_SAM_URL` points at |
| `SAM2_MODEL` | `facebook/sam2.1-hiera-small` | `…-tiny` is smaller, `…-base-plus` and `…-large` more accurate and slower |
| `SAM2_DEVICE` | `cuda` if available, else `cpu` | The bundled unit pins `cuda` |
| `SAM2_DTYPE` | `fp16` on cuda, `fp32` on cpu | `bf16` is available and measurably worse here — see below |
| `SAM2_COMPILE` | `1` | `0` skips compiling the image encoder: ~8 ms slower per new frame, ~5 s faster to start |
| `SAM2_COMPILE_MODE` | `max-autotune-no-cudagraphs` | Any `torch.compile` mode |
| `SAM2_CACHE_FRAMES` | `8` | Encoder outputs kept, 8.4 MB of VRAM each |
| `TORCHINDUCTOR_CACHE_DIR` | torch's default (often `/tmp`) | Point it somewhere persistent or every restart recompiles from scratch |

---

## What makes it fast

Measured on the reference RTX 3070 (sm86, 8 GB) with a 1280×720 desk frame,
while Qwen3-VL was resident on the same card. Each of these is in the deployed
configuration; the ones deliberately *not* taken are at the end.

**A request is two costs, not one.** The image encoder is the expensive half
and it does not depend on the prompt, so `/segment` splits them and keeps the
encoder's output:

| | encoder | prompt decoder | request total |
|---|---|---|---|
| Frame the service has not seen | 26–38 ms | 9–19 ms | **45–60 ms** |
| Frame already in the cache | — | 9–17 ms | **17–21 ms** |
| Three objects × three candidates, cached frame | — | — | **19–21 ms** |

Those are steady-state, over six consecutive frames each. The first request or
two after the card has sat idle cost about twice as much — 79 and 80 ms of
encoder against the 26–33 ms that followed — because GPU clocks ramp under
load. A request that has to wait for a Qwen3-VL generation on the same card
waits longer still.

The cache is keyed by the bytes of the submitted image, so no caller has to
manage identity — asking twice about one frame is enough, and the second ask
skips the JPEG decode as well as the encoder. `/embed` exists for the case
where you know the frame early: encode it while the arm is still moving, then
prompt it by `image_id` when the answer is needed.

**fp16, not bf16.** Encoder time on the same frame, and the worst mask
agreement against fp32 over four exposures of the desk plus a noise frame:

| dtype | encoder | worst IoU vs fp32 |
|---|---|---|
| fp32 | 101 ms | — |
| bf16 | 39 ms | 0.996, but **0.499** on a frame darkened to 0.25× |
| fp16 | 33 ms | 0.998 |

The bf16 outlier is not a corrupted mask. Its coarser mantissa flipped which of
the three candidates won a near-tied predicted-IoU score, so a different, larger
part of the object was returned — with a confident-looking score attached, which
is exactly the failure a caller cannot detect. fp16 is both the faster and the
closer of the two, so there is nothing to trade.

**Compile the encoder, leave the decoder alone.** The encoder always sees
1024×1024×3 — the processor resizes every frame to the model's square input —
so one static shape compiles once and replays with no recompilation: 33 → 27 ms.
The decoder's shapes follow the prompt count, which changes per request, and at
7 ms it has little to give.

Compiling from nothing costs ~100 s. Against a populated
`TORCHINDUCTOR_CACHE_DIR` it costs 5 s, which is why the unit sets one; the
service is ready 7.4 s after a restart (1.2 s to load, 6.2 s to compile and
warm up). `/health` reports `loaded: false` until then.

**Preprocess and post-process on the GPU.** The processor accepts
`device="cuda"` and does the resize and normalize there: 15.6 → 3.4 ms. Mask
logits come back at 256×256 and are upsampled to the frame before leaving the
GPU: 0.1 ms there against 2–3 ms on the host.

**Warm up at load, not on the first request.** Compilation is not the only
first-touch cost — cuDNN and cuBLAS pick kernels on first use, and the
processor and decoder allocate. A first real request paid 2.9 s for all of it
where the next cost 0.03 s, so the service runs one synthetic 1280×720 request
before it starts listening.

**Run-length encoded masks.** A 1280×720 mask is 921,600 booleans; the one
measured here becomes 3,619 runs in 0.2 ms. PNG would be 7.9 KB but 8.5 ms.

### Deliberately not done

- **channels_last** (33.7 ms) and **TF32** (34.1 ms) against 33.1 ms baseline —
  no change worth the lines.
- **CUDA graphs** (`mode="reduce-overhead"`) reached 25.1 ms, 2.5 ms better than
  the mode in use. Replay reuses the output buffers, and this service holds
  encoder outputs in a cache across requests; add the pinned memory pool on a
  card shared with another model and it is the wrong 2.5 ms to buy.
- **Automatic "segment everything"** is not exposed. The transformers
  `mask-generation` pipeline raises `RuntimeError: dets should have the same
  type as scores` in fp16; in fp32 it works but takes 13.5 s and peaks at
  2802 MiB — more than the ~2.3 GB free beside Qwen3-VL — and returned 19 masks
  on the desk frame. Prompt it with points instead.

---

## Reaching a service on another host

Skip this section entirely if the service runs on the machine driving the arm.

**Option A — SSH tunnel.** The service stays on loopback and the tunnel is your
access control. Nothing is exposed to the network.

```bash
ssh -N -L 8767:127.0.0.1:8767 user@gpu-host
```

This repo ships a helper for it,
[scripts/start_tunnel.ps1](../scripts/start_tunnel.ps1), which checks
whether the port is already listening (printing `/health` if so), verifies the
key exists, and reconnects when the link drops. Its defaults match the
reference deployment below, so override them for yours: `-RemoteHost`, `-User`,
`-IdentityFile`, `-LocalPort`, `-RemoteBind`.

**Option B — bind to the LAN.** Set `SAM2_HOST=0.0.0.0` and point `MT4_SAM_URL`
at `http://<gpu-host>:8767`. Simpler, but **the service has no authentication of
any kind** — anyone who can reach the port can post images to your GPU. Only do
this on a network you trust, and prefer Option A otherwise.

The WSL2 host prerequisites are the same ones the Grounding DINO service needs
(systemd enabled inside the distro, `networkingMode=mirrored` and
`vmIdleTimeout=-1` on the Windows side):
[docs/GROUNDING_DINO.md § If your GPU host runs WSL2](GROUNDING_DINO.md#if-your-gpu-host-runs-wsl2).

### Re-deploying after editing server.py

```bash
scp server.py requirements.txt sam2.service user@gpu-host:/opt/sam2/
ssh user@gpu-host "install -m 644 /opt/sam2/sam2.service /etc/systemd/system/ \
  && systemctl daemon-reload && systemctl restart sam2.service \
  && sleep 10 && curl -s http://127.0.0.1:8767/health"
```

`scp` alone does not update `/etc/systemd/system/`, so a restart without the
`install` + `daemon-reload` keeps running the old unit. An SSH tunnel survives a
service restart; only in-flight requests fail. The embedding cache does not
survive — the first request after a restart re-encodes its frame.

---

## HTTP API

Three endpoints, all in [services/sam2/server.py](../services/sam2/server.py).

### `GET /health`

```json
{"ok": true, "model": "facebook/sam2.1-hiera-small", "device": "cuda",
 "cuda": true, "loaded": true, "dtype": "fp16",
 "compile": "enabled (max-autotune-no-cudagraphs)",
 "cache": {"frames": 2, "capacity": 8, "hits": 7, "misses": 2}}
```

### `POST /embed` — `multipart/form-data`

| Field | Type | Meaning |
|-------|------|---------|
| `image` | file | JPEG/PNG frame |

Encodes the frame and keeps it. Returns `image_id`, `width`, `height`,
`cached` (whether it was already held) and `encode_s`.

### `POST /segment` — `multipart/form-data`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `image` | file | — | JPEG/PNG frame. Omit only when passing `image_id` |
| `image_id` | str | — | A frame the service still holds |
| `points` | JSON | — | `[[x, y], ...]`, or `[[[x, y], [x, y]], ...]` to put several points on one object |
| `labels` | JSON | all 1 | Same shape as `points`: 1 for a point on the object, 0 for one off it |
| `boxes` | JSON | — | `[[x1, y1, x2, y2], ...]` |
| `multimask` | bool | true | Three candidates per object instead of one |

At least one of `points` and `boxes` is required; both together are passed to
the model as one prompt set per object. **All coordinates are pixels in the
frame as submitted.** The service rescales them to the model's 1024×1024 input
and maps the masks back, so a caller never works in the model's space.

```json
{"ok": true, "model": "facebook/sam2.1-hiera-small",
 "image_id": "b0dbd73623e6b07bf5d741b6", "width": 1280, "height": 720,
 "objects": 1, "cached": false, "encode_s": 0.0384, "decode_s": 0.0135,
 "total_s": 0.0615,
 "masks": [{"object_index": 0, "score": 0.963, "area": 3061,
            "bbox": [714, 534, 764, 598], "rle": [0, 12, 1268, ...]}]}
```

Masks arrive best-first within each `object_index`, ranked by the model's own
predicted IoU. `rle` is row-major run lengths that alternate background,
foreground, background, …, always starting on background — a leading zero-length
run appears when the top-left pixel is inside the mask.
`mt4_vision.sam.decode_rle` is the three-line inverse.

Errors are HTTP status codes: `400` for a missing or malformed prompt, an
undecodable image, or ragged point groups; `404` for an `image_id` that has
aged out of the cache; `503` before the model finishes loading.

### Three candidates, and why

A single point is genuinely ambiguous — the cube, its top face, or the stack it
sits on are all correct readings of one click. With `multimask` the model
returns all three and scores them, which is more useful than picking for you.
Measured on the desk frame, one point on the blue cube returns 3,061 px at
score 0.96 (the cube), 4,546 px at 0.61 (cube plus the marker under it) and
10,640 px at 0.26 (cube, marker and shadow). Take the first unless you have a
reason not to.

---

## Consumers in this repo

| Entry point | What it does |
|-------------|--------------|
| [mt4_vision/sam.py](../mt4_vision/sam.py) | Client. `health()`, `embed(frame)`, `segment(frame, points=…, boxes=…)` → `Mask` list with `mask`, `score`, `area`, `bbox` and a centre-of-area `cx`/`cy`. `best_per_object()` takes the top candidate each. Raises `SamError` when unreachable |
| `python -m mt4_vision sam --pixel PX PY` | Segment at a pixel or `--box X1 Y1 X2 Y2`, print the numbers, save `sam_frame.jpg` with the mask tinted on. `--candidates` draws all three |
| [mt4_vision/locate.py](../mt4_vision/locate.py) | **Every measurement that starts from a box.** `measure_sam` prompts with the box; `measure_with_box_fallback` is the ladder Grounding DINO, Qwen3-VL, the MCP tools and `relocate` all reach it through |

Measured from the arm host through an SSH tunnel, a call takes 30–95 ms; the
spread is the link, not the model, whose own accounting is the table above.

A mask is not yet a grasp. `mt4_vision.locate` is what turns pixels into a
centre, a long axis and millimetres, and `grasp_feasibility` decides whether
the jaws can take it.

### What depends on the service being up

Anything that measures an object **from a box**: `mt4_locate_by_prompt`,
`ask_dino.py`, `ask_qwen.py` and the instruction worker, `python -m mt4_vision
grounding --locate`, and re-acquiring a registered object with
`locate.relocate`. All but the last already need a GPU service for the box
itself, and this one runs beside either of those on the same card.

Untouched by an outage: cube pick/place, stacking, calibration, the jog client,
and `locate.measure` from a bare pixel, which segments by how far each pixel
sits from the local desk colour and needs no model at all.

When the service is unreachable, a box measurement **refuses and says so**. It
does not quietly fall to the rungs below — the desk-deviation path cannot
segment a stapler at all, and the last rung reports the box's own dimensions as
the object's (150 × 74 mm for a stapler, measured 2026-08-02), which is a wrong
number that looks like a right one.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sam service unreachable at http://127.0.0.1:8767` | Service not running, or (remote setups) no tunnel | Check the service; start the tunnel |
| Port already listening but `/health` fails | Stale tunnel, or the service died behind a live one | Check the service on the GPU host; kill the old `ssh` and restart |
| `503 model not loaded` | Requested during startup | Wait ~8 s; `/health` reports `loaded` |
| Startup takes 100 s and `/health` is dead throughout | Compiling with a cold or non-persistent inductor cache | Set `TORCHINDUCTOR_CACHE_DIR` somewhere that survives reboots, or `SAM2_COMPILE=0` |
| `404 image_id … is not cached` | The frame aged out (8 frames by default) | Send the image again, or raise `SAM2_CACHE_FRAMES` |
| Mask covers the object *and* its shadow | The point was ambiguous and the top candidate is the wide one | Add a second point with label 0 on the shadow, or prompt with a box |
| Mask is a small part of the object | The point landed on a distinct sub-part | Move the point toward the object's middle, or use a box |
| `/health` shows `device=cpu` unexpectedly | No CUDA torch, or the driver is not visible | `torch.cuda.is_available()` in the venv; check the GPU driver |
| OOM at load beside another model | Card too full | Check what else is on the GPU; the service itself needs ~600 MiB including its CUDA context |

---

## Reference deployment

Not requirements — just the setup this was developed and verified against, for
comparison if something behaves differently for you.

| | |
|---|---|
| GPU host | a LAN machine (`media`) running WSL2 under Windows, reached by SSH |
| GPU | NVIDIA GeForce RTX 3070, 8 GB, driver 591.86 |
| Service dir | `/opt/sam2` (`server.py`, `requirements.txt`, the unit, `.venv/`, `.inductor/`) |
| Python | 3.10.12 |
| Resolved packages | torch 2.13.0+cu130, torchvision 0.28.0, transformers 5.14.1, fastapi 0.141.1, uvicorn 0.52.1 |
| Weight cache | `/root/.cache/huggingface`, ~180 MB for this model |
| Bind | `127.0.0.1:8767`, loopback only, reached by SSH local forward |
| Sharing the card | Runs alongside `qwen3-vl.service`; both enabled, 6175 MiB of 8192 in use with Qwen3-VL mid-generation |
| Autostart | `systemctl enable`, so it comes up with the distro — which a Windows scheduled task (`OpenPI-WSL-Autostart`) starts at boot |

`grounding-dino.service` and `qwen3-vl.service` cannot both be resident on this
card, which is what `services/switch_vision_service.sh` is for. SAM 2.1 is not
part of that trade: at ~600 MiB it stays up beside whichever of the two is
running.

In that deployment the GPU host resolves by mDNS to both a LAN IPv4 address and
a link-local IPv6 one. The v6 record does not route, so a bare `ping` or `ssh`
picks it and hangs — which is why the `ssh`/`scp` invocations for it pass `-4`.

---

## See also

- [services/sam2/README.md](../services/sam2/README.md) — what each deployed file is, plus the day-to-day segment commands
- [docs/GROUNDING_DINO.md](GROUNDING_DINO.md) — the box-producing service whose output makes a good prompt here, and the shared WSL2 host notes
- [docs/CALIBRATION.md](CALIBRATION.md) — the measurement layers a mask feeds into
