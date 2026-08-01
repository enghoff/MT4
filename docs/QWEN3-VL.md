# Qwen3-VL service

A local HTTP wrapper around `Qwen/Qwen3-VL-4B-Instruct`, loaded in 4-bit (NF4) via
bitsandbytes. Lives at `/opt/qwen3-vl`, in the same WSL distro as the Grounding
DINO service (`/opt/grounding_dino`) — on the reference deployment (`media`),
that's the `openpi`-named distro (see [docs/GROUNDING_DINO.md](GROUNDING_DINO.md)
for the WSL2 host-level gotchas that apply here too).

- **Port:** `8766`, bound to `127.0.0.1` only (not reachable outside the distro/host
  without a tunnel — see below).
- **VRAM footprint:** ~6.1GB of the RTX 3070's 8GB with the static cache and CUDA
  graphs enabled (~3.7GB without — see [Decode throughput](#decode-throughput)).
  That leaves ~2GB headroom, so it is even less of a co-tenant than before.
- **Not meant to run at the same time as Grounding DINO** — it's set up as a separate,
  disabled-by-default systemd service you start on demand, not something that
  auto-starts alongside `grounding-dino.service`.

## Starting / stopping

```bash
# from inside the GPU host's WSL distro
sudo systemctl start qwen3-vl      # first request after this takes ~15-20s to load the model
sudo systemctl stop qwen3-vl
sudo systemctl status qwen3-vl
journalctl -u qwen3-vl -f          # tail logs
```

It is **not enabled** (won't auto-start on boot), unlike `grounding-dino.service`.
If you want it running after every boot, `sudo systemctl enable qwen3-vl` — but
remember both services loaded together fit in 8GB VRAM only with limited headroom,
so keep an eye on `nvidia-smi` if you do that.

## API

### `GET /health`

```bash
curl http://127.0.0.1:8766/health
```

```json
{"ok": true, "model": "Qwen/Qwen3-VL-4B-Instruct", "device": "cuda", "cuda": true, "loaded": true, "quantization": "nf4-4bit"}
```

### `POST /generate`

Multipart form fields:

| Field | Required | Default | Notes |
|---|---|---|---|
| `prompt` | yes | — | The text prompt / question |
| `image` | no | — | Image file. **Repeatable** — send it N times for N independent images. Omit for text-only |
| `video` | no | — | Frame file, **repeatable**, 2+ required. Sent as one time sequence. Mutually exclusive with `image` |
| `fps` | no | `2.0` | Video only. The *real* capture rate; timestamps are built from it |
| `max_new_tokens` | no | `256` | Generation length cap |
| `do_sample` | no | model config | `false` for greedy/reproducible. The model's own config samples at temperature 0.7 |
| `temperature` | no | model config | Implies `do_sample=true` |

Every response reports what the model actually received — `mode`, `frames_sent`,
`prompt_tokens`, plus `images_encoded` or `temporal_groups` and `timestamps_s`.
That is not decoration: see the frame-dropping trap below.

It also reports what generation cost, split the way the fixes are split:
`completion_tokens`, `prefill_s`, `decode_s`, `decode_tok_s`, `generate_s`, and
`cache` (`static` or `dynamic`). Prefill and decode want opposite optimizations
and differ by two orders of magnitude per token, so one wall time hides which
of them a change moved — see [Decode throughput](#decode-throughput).

**With an image:**

```bash
curl -X POST http://127.0.0.1:8766/generate \
  -F "image=@photo.jpg" \
  -F "prompt=Describe what's happening in this image." \
  -F "max_new_tokens=200"
```

**Text-only (no image):**

```bash
curl -X POST http://127.0.0.1:8766/generate \
  -F "prompt=Summarize the difference between a VLM and an LLM in one sentence."
```

**Response:**

```json
{
  "ok": true,
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "prompt": "Describe what's happening in this image.",
  "had_image": true,
  "response": "..."
}
```

Errors come back as standard FastAPI JSON (`400` for a bad/empty image or empty
prompt, `422` if `prompt` is missing entirely, `503` if the model hasn't finished
loading yet).

## Decode throughput

**Essentially all of a request's time is the token loop, not the image.** Measured
on the reference deployment, one 1280x720 frame:

| | cost |
|---|---|
| prefill (vision encoder + one batched forward, 912 prompt tokens) | **0.45 s** |
| decode | **29 ms/token** |

So a 350-token `/objects` reply is ~10s, of which the image accounts for 4%.
Nothing about resizing frames or trimming vision tokens will move that number
meaningfully; only the token loop will.

Two knobs do, and they only work together:

| Config | decode | GPU util | power |
|---|---|---|---|
| dynamic cache, eager (the original setup) | 11.4 tok/s | 48% | 140 W |
| static cache, eager | 11.0 tok/s | 61% | 168 W |
| `COMPILE=1` alone — **silently does nothing** | 10.8 tok/s | 56% | 158 W |
| **static cache + CUDA-graphed decode** | **34 tok/s** | **86%** | 192 W |

A **3x** speedup, and the utilization number that prompted the exercise goes
48% → 86%. Note the static cache *on its own is a pessimization* — it is worth
having only as the precondition for compilation.

### Why `COMPILE=1` alone does nothing

`transformers` compiles only the decode step (prefill stays eager, which is what
keeps a variable-size vision input from recompiling per image), but
`_valid_auto_compile_criteria` gates that on `hf_quantizer.is_compileable`. In
5.14.1 **only `TorchAoHfQuantizer` and `FineGrainedFP8HfQuantizer` return True** —
bitsandbytes, GPTQ, HQQ, AWQ and the rest all return False. A blocked compile
only logs `unable to meet the criteria for compilation`, so a config that looks
enabled can be doing nothing at all. `/health` reports the gate's actual verdict
in its `compile` field for exactly that reason; check it, don't assume it.

`QWEN_VL_FORCE_COMPILE=1` patches the property. Measured, that gate is
conservative for this combination (transformers 5.14.1 / bnb 0.50 / torch 2.13 /
sm86): output stays coherent and is byte-identical across repeated passes in one
process.

### What it costs

- **~52 s of compilation at startup**, warmed at load rather than on the first
  request. `/health` reports `loaded: false` for ~80 s after a restart, not ~30 s.
- **~2.4GB more VRAM** (6.1GB total), from the preallocated cache and the graph pools.
- **Greedy replies no longer match the ones the uncompiled service gave.** This
  is *not* caused by compilation — the static cache alone diverged from
  dynamic-cache-eager on 3 of 4 probe prompts. Changing floating-point reduction
  order sends one near-tied argmax the other way and the rest of the sequence
  follows; every output involved stayed coherent. Reproducibility *within* a
  configuration holds, which is what `/freeze` and `/again` depend on. Only
  comparisons against answers recorded before this change are affected — if you
  have such a baseline, re-run it.

### Sizing `QWEN_VL_CACHE_LEN`

Attention runs over the **whole** window every step regardless of how much is
real, so an oversized cache is pure waste: 4096 gave 23.6 tok/s where 2048 gave
34 on the same 912-token request. The default 2048 fits one image plus a
700-token reply.

Requests that would overrun it fall back to the dynamic cache, and therefore to
eager — correct, just slow, and reported as `"cache": "dynamic"` in the response.
That covers multi-frame `images` mode (6 frames measured at 5310 prompt tokens →
8.4 tok/s) and the explicit `/watch <question>` before/after pair. Raise the knob
if that is your normal workload, and expect the single-frame case to get slower
in exchange.

### What's left

The remaining ceiling is the quantizer, not the plumbing. 34 tok/s is still well
short of what a 4B int4 model can do on this card, because bitsandbytes NF4
dequantizes weights on every matmul — the GPU draws 192 W to produce those
tokens. Switching to **torchao int4 weight-only** would replace those kernels
*and* pass the compile gate legitimately (no `FORCE_COMPILE`), and unlike
AWQ/GPTQ it needs no calibration pass. That is the next real step.

## Reaching it from outside the distro

The service only binds to `127.0.0.1` inside the distro, the same pattern as
Grounding DINO — see [docs/GROUNDING_DINO.md § Reaching a service on another
host](GROUNDING_DINO.md#reaching-a-service-on-another-host) for the full
options (SSH tunnel vs. LAN bind) and the WSL2-specific gotchas
(`systemd=true`, `networkingMode=mirrored`, mDNS/IPv6).

For the reference deployment (`media`), use
[scripts/start_qwen_tunnel.ps1](../scripts/start_qwen_tunnel.ps1) — same
pattern as `start_grounding_tunnel.ps1`, on port 8766 instead of 8765, and
kept as a separate tunnel since the two services aren't meant to run at the
same time on the GPU:

```powershell
.\scripts\start_qwen_tunnel.ps1
# leave running, then from another terminal:
curl http://127.0.0.1:8766/health
```

## Consumers in this repo

Nothing in the arm stack depends on the VLM — cube pick/place, calibration, the
MCP tools and Grounding DINO all work with this service absent.

| Entry point | What it does |
|-------------|--------------|
| [mt4_vision/qwen.py](../mt4_vision/qwen.py) | Client. `ask(prompt, frames, mode=...)` → `Reply` (text plus what the service really encoded, incl. `frame_warning()`); `generate(prompt, frame)` → just text for the one-frame case; `parse_regions(text)` → the boxes/points Qwen named; `health()`. Raises `QwenError` when unreachable. `MT4_QWEN_URL` overrides the default URL |
| [ask_qwen.py](../ask_qwen.py) | Interactive harness — see below |
| [services/qwen3_vl/](../services/qwen3_vl/) | The deployed service itself: `server.py`, `requirements.txt`, systemd unit |

### The interactive harness

```powershell
python ask_qwen.py --camera 1          # interactive
python ask_qwen.py --prompt "how many cubes are on the desk?"   # one-shot, scriptable
```

Type a question, get the answer drawn beside **the exact frame that was
POSTed**, with any coordinates it returned drawn on that frame. Showing the
submitted frame rather than the live feed is the point: a wrong answer about a
frame the arm was blocking is a different problem from a wrong answer about a
clean frame, and the text alone never tells you which happened. A live inset
stays in the corner so you can keep aiming the camera.

`/help` lists the commands. The ones that earn their keep:

| Command | Why |
|---|---|
| `/freeze` | Pin one frame so repeat asks use an identical image — separates model nondeterminism from scene change |
| `/again` | Re-ask the same question; with `/freeze`, the same question on the same pixels |
| `/coords abs\|norm` | Reinterpret returned coordinates — see below |
| `/grid` | Draw a labelled pixel grid on the sent image (`annotate_for_pointing`), which measurably helps pointing |
| `/frames N [gap]` | Capture N frames, `gap` seconds apart |
| `/mode M` | How they reach the model: `single`, `montage`, `images`, `video`. Set `/frames` once, then flip `/mode` and `/again` to A/B the representations on the same question |
| `/sample` | Toggle greedy (default) vs the model's temperature 0.7. Greedy is what makes two answers comparable |
| `/watch` | Watching is **on by default** — whatever you type becomes the standing question, re-asked on every movement. `/watch off` stops it; see below |
| `/once <q>` | Ask without changing what the watcher is watching for |
| `/sens X` | Motion trigger threshold |
| `/preset` | A capability checklist: description, inventory, counting, colors, grounding, pointing, OCR, fiducial tags, spatial relations, graspability, arm-occlusion |
| `/save` | Write the sent frame, the annotated view and a JSON record to `qwen_probes/` |

Questions queue FIFO, one on the GPU at a time (generation is serialized
server-side, so overlapping requests only make both slower). Piping works —
`printf '/preset 3\n/preset 5\n' | python ask_qwen.py --camera 1 --no-preview`
answers both and exits.

`--camera` defaults to the repo-wide ArUco auto-detect, which asks "which
camera sees the calibrated desk". This harness needs no calibration and no
markers, and the desk tags are routinely occluded by the very objects being
probed — so if auto-detect gives up it prints the openable indices and you pass
one explicitly.

### Multiple frames: images vs video (measured)

The same frames cost very different amounts and produce different answers, so
the two are not packaging choices. Measured on the reference deployment, 6
frames at 1280x720:

| Sent as | Prompt tokens | What the model gets |
|---|---|---|
| `montage` (tiled into one image) | ~950 | one still; sequence only from drawn numbers |
| `video` | ~1830 | temporal patching, real timestamps, N/2 groups |
| `images` | ~2700 | N independent full-budget images, nothing temporal |

Video is about half the cost of images because the encoder pairs adjacent
frames into one temporal patch (`temporal_patch_size: 2`).

**Capability, tested against ground truth** — a synthetic moving square, and
real desk frames with the arm driven to known positions:

| Question | montage | images | video |
|---|---|---|---|
| Did anything move? (vs. static control) | correct | correct | correct |
| Which direction did it move? | — | **correct**, and flips when frames are reversed | **wrong, and invariant to reversal** |
| Which numbered frame is the object furthest left? | wrong | wrong | wrong |

So: **video for "did anything change"** (same answer, ~2/3 the tokens of
images), **images for "which way"**, and do not trust frame-index or
superlative questions in any mode. All three detect change reliably; none of
them reliably index frames.

Video's direction failure is the model, not the plumbing — verified by
comparing tensors for forward vs reversed frame lists (they differ correctly,
all frames present, order preserved).

### Watching for movement (on by default)

```powershell
python ask_qwen.py --camera 1              # already watching
python ask_qwen.py --camera 1 --no-watch   # request/response only
```

The harness is a **monitor by default**: it watches the desk from startup and
asks about anything that moves. Until you type something it asks the generic
"what changed between these two frames"; after that, **whatever you last typed
becomes the standing question**, asked again on every event. So

```
ask: is the red cube still on its marker?
```

answers immediately *and* keeps answering that same question each time the desk
changes — no second command. Commands are the exception: `/preset`, `/again`,
`/noimage` and `/once <q>` are one-offs and leave the standing question alone.
`/watch off` stops; a bare `/watch` reports state when on and resumes when off
(it is deliberately not the way to turn it off, now that it starts on).

**A typed question is asked about the new frame alone, not the pair.** Almost
every question is about the state of the scene — "where is the stapler", "how
many cubes" — and handing that a before/after pair silently reframes it as a
comparison. That is how the pair framing kept answering *"it has not moved"*
for changes the frame diff had already proven. Only two things use both frames:
the built-in "what changed" default, and an explicit `/watch <question>`.

For tracking one object, `/track <object>` arms the same single-frame path with
a grounding prompt, so each event redraws its box:

```
ask: /track stapler
```

An answer costs 3-5s and the GPU serializes them, so polling the model to ask
"has anything moved" would run at ~0.2 Hz and keep the GPU busy permanently. A
frame diff answers that question at camera rate for nothing, so it gates
everything: the model is asked once the scene has **moved and settled**, and is
handed the last quiet frame plus the first new quiet one, as two images. While
disarmed the watcher thread does not even read the camera, so `--no-watch`
costs nothing.

Waiting for the settle is the part that matters — firing on the first changed
frame catches the arm mid-sweep or a hand still over the desk. Events that
arrive while an answer is still generating are dropped and counted, not queued,
because a backlog of stale before/after pairs reads as current.

Threshold defaults to `0.0005`, from measurement on the desk camera: 60 static
frames gave a noise floor of mean `0.00008` / max `0.00016`, while a 25x25px
object moving produces ~`0.0014`. So it sits ~3x above the worst noise and ~3x
below the smallest real event. `/sens` retunes it; the live score is in the
panel, and changed regions are outlined on the view so you can see whether it
fired on the object you care about or on a shadow.

**The gate is more reliable than the model.** Verified with arm-driven motion:
a 0.063 event (390x the noise floor) was described correctly, but a 0.027 event
(169x noise) came back as *"no discernible change between the two frames"*.
Treat the diff score as the authority on **whether** something moved, and the
model only as the answer to **what** — if it contradicts a trigger, believe the
trigger.

For actually tracking an object's position, use Grounding DINO instead
([docs/GROUNDING_DINO.md](GROUNDING_DINO.md)): ~3 Hz, real boxes, and already
running here. This is for the semantics — what changed, did it fall over, is
the arm in the way.

### Getting a JSON object list instead of prose

`identify all objects` returns a paragraph. Asking for "JSON" returns JSON of
the wrong shape. **The schema has to be named.** Measured on one frame, greedy,
3 runs each:

| Prompt | Boxes returned |
|---|---|
| `identify all objects` | **0** — prose, "Based on the image provided…" |
| `Identify all objects. Reply in JSON.` | **0** — valid JSON, but `{"objects":[{"name","description"}]}` |
| Explicit `bbox_2d` schema + "no prose, no markdown" | **10** ✓ |

All three were 3/3 reproducible under greedy decoding, so this is a property of
the prompt, not luck. No constrained decoding needed — the model complies once
told the keys. `/objects` has the working prompt baked in (`OBJECTS_PROMPT` in
`ask_qwen.py`), and `/objects watch` re-lists on every scene change.

Budget tokens for it: roughly 35 per boxed object, so the 256 default truncates
a busy desk mid-array. `parse_regions` recovers the complete entries, but the
tail goes missing silently — use `/tokens 600` or more.

### The frame-dropping trap

The video processor defaults to `do_sample_frames=True`, `fps=2`,
`min_frames=4`, and *resamples* whatever it is handed as though it were raw
footage needing decimation. With no metadata it assumes the source was 24 fps,
so **5, 6, 8 and 9 frames all collapsed to an identical 2-group tensor** —
frames silently discarded, answer still fluent. Timestamps came from the same
false assumption: 0.2 s reported for frames actually spanning 2.5 s.

`server.py` fixes both (`do_sample_frames=False` plus real `video_metadata`),
but note `video_metadata` is only honoured on a **direct processor call** —
through `apply_chat_template` it is ignored or raises, which is why the video
path builds text and tensors in two steps. If you ever rewrite that function,
check `temporal_groups` in the response against `ceil(frames/2)`;
`mt4_vision.qwen.Reply.frame_warning()` is that assertion.

### Coordinate space (measured, not assumed)

**This build answers in 0-1000 normalized coordinates, not pixels.** Verified
against the desk camera at 1280x720: a box reported at `x=807` sits at 1033px,
and asking for the whole desk returns exactly `[0, 433, 1000, 1000]`. So
`ask_qwen.py` scales by default (`--coords norm`).

Do not carry that assumption to another build. Qwen2-VL used 0-1000, 2.5-VL
onward is *documented* as absolute pixels, and "absolute" can still mean the
processor's internally resized image rather than the one you submitted.
`parse_regions` therefore returns coordinates exactly as written and guesses
nothing; the harness flags regions that fall outside the frame and suggests the
other interpretation, which settles it by eye in one keystroke.

Accuracy on the reference desk, for calibration of expectations: named objects
(stapler, red cube, toy figurine) box tightly and usefully. But it labelled the
red cube "green cube" while boxing it correctly, and counted "2 cubes ... red,
red" with a green one in frame — localization is stronger than instance/colour
binding. Treat it as a describer and a rough locator, not a detector; for
detection there is [Grounding DINO](GROUNDING_DINO.md).

## Configuration (environment variables)

Set these in `/etc/systemd/system/qwen3-vl.service` (then `sudo systemctl daemon-reload`):

| Variable | Default | Purpose |
|---|---|---|
| `QWEN_VL_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | HF repo id to load |
| `QWEN_VL_HOST` | `127.0.0.1` | Bind address |
| `QWEN_VL_PORT` | `8766` | Bind port |
| `QWEN_VL_DEVICE` | `cuda` (auto-detected) | `cuda` or `cpu` |
| `QWEN_VL_LOAD_IN_4BIT` | `1` | Set to `0` to load full-precision (needs ~16GB+ VRAM, won't fit this card) |
| `QWEN_VL_MAX_NEW_TOKENS` | `256` | Default generation cap if the request omits `max_new_tokens` |
| `QWEN_VL_CACHE_LEN` | `2048` | Static KV-cache window. `0` for the dynamic cache (and no compilation). Size it tightly — see [Decode throughput](#decode-throughput) |
| `QWEN_VL_COMPILE` | `1` | CUDA-graph the decode step. Needs a static cache; **a no-op on its own with bitsandbytes** |
| `QWEN_VL_FORCE_COMPILE` | `1` | Override `is_compileable=False` on the bnb quantizer. Without it `COMPILE=1` is silently skipped |

Set `QWEN_VL_COMPILE=0` to revert to the original 11.4 tok/s behaviour, e.g. if you
need to reproduce an answer recorded before the change.

## Layout

```
/opt/qwen3-vl/
├── .venv/              # uv-managed venv (torch, transformers, bitsandbytes, fastapi, ...)
├── requirements.txt
├── server.py
└── README.md           # this file
```

Model weights are cached under `~/.cache/huggingface/hub/` the first time the
service loads them (~8GB download), not inside `/opt/qwen3-vl` itself.

### Re-deploying after editing server.py

The source of truth is [services/qwen3_vl/](../services/qwen3_vl/) in this repo;
`/opt/qwen3-vl` is a deployment of it. Push, restart, confirm:

```powershell
scp services/qwen3_vl/server.py root@media:/opt/qwen3-vl/server.py
ssh root@media "systemctl restart qwen3-vl && sleep 30 && curl -s http://127.0.0.1:8766/health"
```

Model load takes ~20-30s and compilation another ~52s, so allow ~80s; a `/health`
before then reports `loaded: false` rather than failing. If the unit file changed,
`install -m 644` it into `/etc/systemd/system/` and `systemctl daemon-reload`
first — `scp` alone leaves the old unit in force.

Confirm the perf knobs actually engaged, since a blocked compile is silent:

```powershell
curl -s http://127.0.0.1:8766/health   # want cache:"static", compile:"forced: ..."
```

One more trap when restarting by hand: `pkill -f qwen3-vl/server.py` over SSH
matches the remote shell's *own* command line and kills the session before your
next command runs. Use `pkill -f '[q]wen3-vl/server.py'`.

## Known gotcha

`transformers`' Qwen3-VL processor pulls in a video-processor class that requires
`torchvision` even for pure image use — it's in `requirements.txt` now, but if you
rebuild the venv from scratch and see `ImportError: Qwen3VLVideoProcessor requires
the Torchvision library`, that's why.

## See also

- [docs/GROUNDING_DINO.md](GROUNDING_DINO.md) — the other vision service on this
  GPU host: open-vocab *detection* rather than description, and the source of the
  deployment/tunnel/WSL2 details this doc refers to
