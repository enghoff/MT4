# Qwen3-VL service

A local HTTP wrapper around `Qwen/Qwen3-VL-4B-Instruct`, loaded in 4-bit (NF4) via
bitsandbytes. Lives at `/opt/qwen3-vl`, in the same WSL distro as the Grounding
DINO service (`/opt/grounding_dino`) — on the reference deployment (`media`),
that's the `openpi`-named distro (see [docs/GROUNDING_DINO.md](GROUNDING_DINO.md)
for the WSL2 host-level gotchas that apply here too).

- **Port:** `8766`, bound to `127.0.0.1` only (not reachable outside the distro/host
  without a tunnel — see below).
- **VRAM footprint:** ~4.7GB of the RTX 3070's 8GB with int4 weights, the static
  cache and CUDA graphs (~3.7GB for the original nf4 eager setup — see
  [Decode throughput](#decode-throughput)).
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
on the reference deployment, one 1280x720 frame at 912 prompt tokens: prefill is
**0.85 s** and decode is **22 ms/token**, so a 350-token reply is ~9 s with the
image accounting for under 10% of it. Nothing about resizing frames or
trimming vision tokens will move that meaningfully; only the token loop will.

Three things do, and the ordering is not the intuitive one:

| Config | decode | GPU util | power | VRAM |
|---|---|---|---|---|
| nf4, dynamic cache, eager (the original setup) | 11.4 tok/s | 48% | 140 W | 3.7 GB |
| nf4, static cache, eager | 11.0 tok/s | 61% | 168 W | — |
| nf4, `COMPILE=1` alone — **silently does nothing** | 10.8 tok/s | 56% | 158 W | — |
| int4, dynamic cache, eager | 12.2 tok/s | — | — | 4.8 GB |
| nf4 + static cache + CUDA graphs | 34 tok/s | 86% | 192 W | 6.1 GB |
| **int4 + static cache + CUDA graphs** (current) | **45 tok/s** | 81% | 185 W | **4.7 GB** |

**4x on decode**, and the utilization number that prompted the exercise goes
48% → 81%. End to end a 200-token request went 17.9 s → 5.3 s.

Two of those rows are the interesting ones. The static cache *on its own is a
pessimization* — it earns its place only as the precondition for compilation.
And **the quantizer is worth nothing without compilation**: nf4 and int4 differ
by 11.4 vs 12.2 tok/s eager, because per-token dispatch overhead swamps any
difference between the kernels, and only once CUDA graphs remove that overhead
does the kernel quality surface as 34 vs 45. Anyone benchmarking quantizers in
eager mode is measuring Python.

### Why `COMPILE=1` alone does nothing

`transformers` compiles only the decode step (prefill stays eager, which is what
keeps a variable-size vision input from recompiling per image), but
`_valid_auto_compile_criteria` gates that on `hf_quantizer.is_compileable`. In
5.14.1 **only `TorchAoHfQuantizer` and `FineGrainedFP8HfQuantizer` return True** —
bitsandbytes, GPTQ, HQQ, AWQ and the rest all return False. A blocked compile
only logs `unable to meet the criteria for compilation`, so a config that looks
enabled can be doing nothing at all. `/health` reports the gate's actual verdict
in its `compile` field for exactly that reason; check it, don't assume it.

**torchao passes that gate legitimately**, which is half the reason to prefer it:
`QWEN_VL_QUANT=int4` needs no `FORCE_COMPILE`, and `/health` reports
`compile: "enabled"` rather than `"forced: ..."`.

`QWEN_VL_FORCE_COMPILE=1` remains for the nf4 path, where it patches the
property. Measured, the gate is conservative for that combination (transformers
5.14.1 / bnb 0.50 / torch 2.13 / sm86): output stays coherent and is
byte-identical across repeated passes in one process.

### int4 trades prefill for decode

Weight-only int4 has to unpack for the wide GEMMs prefill is made of, where bf16
just multiplies, so **prefill went 0.45 s → 0.85 s** while decode went 34 → 45
tok/s. That puts the crossover at about **56 output tokens**: below it nf4 was
marginally quicker, above it int4 wins and keeps winning. The only workload here
is one decision per step — an action, a box, a destination and however much the
model writes into `reason`, capped at 640 tokens by `instruct.MAX_NEW_TOKENS`.
A cap is a ceiling and most decisions decode far below it, so this workload
still sits nearer the crossover than far above it; measure before assuming int4
is the faster choice for this loop in particular.

`QWEN_VL_INT4_SKIP` (default `visual`) keeps the vision tower in bf16. That is
better on every axis at once — 43 → 45 tok/s, VRAM 5.0 → 4.7 GB, and compile time
62 s → 20 s — so the prefill cost above is the *language model's*, not the vision
encoder's. Set it empty to quantize everything.

Quality spot-check: int4 and nf4 put the same grounding boxes within 1-4 units of
each other on the 0-1000 scale, and prose stayed coherent. That was four prompts
on one synthetic frame, not the capability matrix above — **re-check grounding
against the real desk** before trusting int4 for anything new.

### What it costs

- **~20 s of compilation at startup** on top of a 4 s load (int4 quantizes at
  load, so there is no separate quantized checkpoint to manage). Warmed at load
  rather than on the first request, so `/health` reports `loaded: false` for ~25 s.
- **~1GB more VRAM** (4.7GB total), from the preallocated cache and graph pools.
- **Greedy replies no longer match the ones the uncompiled service gave.** This
  is *not* caused by compilation — the static cache alone diverged from
  dynamic-cache-eager on 3 of 4 probe prompts. Changing floating-point reduction
  order sends one near-tied argmax the other way and the rest of the sequence
  follows; every output involved stayed coherent. Reproducibility *within* a
  configuration holds, which is what the loop's greedy decoding relies on: the
  same frame and the same instruction give the same decision. Only comparisons
  against answers recorded before this change are affected — if you have such a
  baseline, re-run it.

### Sizing `QWEN_VL_CACHE_LEN`

Attention runs over the **whole** window every step regardless of how much of it
is real, so an oversized cache is pure waste. Measured on the same 912-token
request — with nf4, 4096 gave 23.6 tok/s and 2048 gave 34; with int4, 2048 gave
40 and 1664 gave 43. The default **1664** is the smallest that still fits one
image (912 prompt tokens) plus a full 700-token reply.

Requests that would overrun it fall back to the dynamic cache, and therefore to
eager — correct, just slow, and reported as `"cache": "dynamic"` in the response.
That covers any multi-frame send — `images` mode with 6 frames measured at 5310
prompt tokens → 8.4 tok/s. Raise the knob if that is your normal workload, and
expect the single-frame case to get slower in exchange.

### What's left

Decode is no longer the obvious bottleneck, and the next steps are structural
rather than a knob:

- **The multi-frame fallback.** `images`/`video` mode gets none of this, because
  one static cache can only be one size. Two caches (one single-frame, one wide)
  would fix it at the cost of a second CUDA graph.
- **Concurrency.** The service is strictly one request at a time. The instruction
  loop is serial anyway — one arm, one decision per step — so nothing queues
  behind itself today, but a second client would. vLLM or SGLang would give
  continuous batching and prefix caching; both would want an int4 checkpoint,
  which is what this is.
- **45 tok/s is still ~20% of this card's roofline**, so there is real headroom
  left in the kernels if it ever matters.

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
| [mt4_vision/instruct.py](../mt4_vision/instruct.py) | The policy layer: builds the decision prompt, reads the action and the box back out. Nothing in it computes a robot coordinate |
| [ask_qwen.py](../ask_qwen.py) | The harness: type an English instruction and the arm carries it out — see below |
| [services/qwen3_vl/](../services/qwen3_vl/) | The deployed service itself: `server.py`, `requirements.txt`, systemd unit |

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

### The schema has to be named

`identify all objects` returns a paragraph. Asking for "JSON" returns JSON of
the wrong shape. **Naming the keys is what works.** Measured on one frame,
greedy, 3 runs each:

| Prompt | Boxes returned |
|---|---|
| `identify all objects` | **0** — prose, "Based on the image provided…" |
| `Identify all objects. Reply in JSON.` | **0** — valid JSON, but `{"objects":[{"name","description"}]}` |
| Explicit `bbox_2d` schema + "no prose, no markdown" | **10** ✓ |

All three were 3/3 reproducible under greedy decoding, so this is a property of
the prompt, not luck. No constrained decoding needed — the model complies once
told the keys. `instruct.build_prompt` spells its own keys out for exactly this
reason, forbids prose and markdown fences, and ends with "Begin your reply with
`{` and end it with `}`".

Budget tokens accordingly: roughly 35 per boxed object. `MAX_NEW_TOKENS` is 640
in the instruction loop. The arm-facing fields need a few dozen tokens of that;
the rest is headroom for `reason`, which is the only channel back from the
model and carries the answer when a task asks a question rather than for a
movement.

**The decision prompt does not get the service's fast path, and never did.**
`server.py` takes the reused StaticCache only when `prompt_len + max_new_tokens
<= CACHE_LEN` (1664). Measured on a 1280x720 frame, `build_prompt` alone comes
to **1815 prompt tokens**, so every decision already falls back to the dynamic
cache and eager decode regardless of the reply budget. Restoring the fast path
means raising `QWEN_VL_CACHE_LEN` past prompt + reply, and the cache is sized
tightly on purpose — 1664 measured 43 tok/s where 2048 gave 34 and 4096 gave
23.6, so a bigger cache slows every other request the service serves. That is a
service-level trade, not one the loop can make for itself.

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
and asking for the whole desk returns exactly `[0, 433, 1000, 1000]`. So the
instruction loop scales: `instruct.to_frame_pixels` reads a reply as 0-1000
against the frame's own width and height, `x * w / 1000` and `y * h / 1000`,
two different factors on a non-square frame. Measured over 3 targets × 2 prompt
styles on one 1280x720 frame, box centres land **2-13px** from truth read that
way, 6 of 6 — and **264-363px** away read as raw pixels.

Do not carry that assumption to another build. Qwen2-VL used 0-1000, 2.5-VL
onward is *documented* as absolute pixels, and "absolute" can still mean the
processor's internally resized image rather than the one you submitted.
`parse_regions` therefore returns coordinates exactly as written and guesses
nothing, and `Region.in_bounds` flags a region falling outside the frame —
which is what a 0-1000 reply read as pixels looks like. The instruction loop
keeps the raw-pixel reading as a retry, used only when the normalized one fails
to segment.

Accuracy on the reference desk, for calibration of expectations: named objects
(stapler, red cube, toy figurine) box tightly and usefully. But it labelled the
red cube "green cube" while boxing it correctly, and counted "2 cubes ... red,
red" with a green one in frame — localization is stronger than instance/colour
binding. Treat it as a describer and a rough locator, not a detector; for
detection there is [Grounding DINO](GROUNDING_DINO.md).

### The instruction loop

`ask_qwen.py` is a prompt pinned to the bottom of the terminal, a background
worker, and a window that keeps redrawing while the worker is busy. The worker
moves real hardware, and that is what shapes the rest of the pieces.

```powershell
python ask_qwen.py                                  # interactive
python ask_qwen.py "put the red cube on marker 3"   # one-shot, exit 0 = DONE
python ask_qwen.py "find all the pickable objects"  # a report, nothing moves
python ask_qwen.py --dry-run "pick up the stapler"  # decide, never move
python ask_qwen.py --record run.mp4 "..."           # the window, to a video
```

**The window is three things at once.** The left pane is the frame the last
decision was made from, with the model's own answer drawn over it
(`preview.annotate_qwen`: the box it named, the same numbers in the other
coordinate space, the segmented silhouette, and a line from the point it gave
to the entity the stack bound that point to). The right is a state panel —
instruction, step, phase, what the jaws are believed to hold, what has been
done. The corner inset is the live feed. The left pane deliberately does not
update during a move: those are the pixels a decision was made from, and a
re-capture would silently answer a different question.

**The two pictures trade places while the hardware moves** — a carry, a park, a
homing, the jaws opening. Standing still, the decision frame is large, because
the question then is what the model pointed at and that is a question about the
pixels a box was drawn on. Moving, the live feed takes the large pane and the
overlay drops to the inset: the desk is changing, and the frame with the
overlay is describing a moment that has passed. Both pictures stay on screen
throughout; only which one gets the space changes, and the inset's label says
which is which.

**The live pane draws the move it is watching** — a green ring where the jaws
will close, another where they will open, an arrow from one to the other, in
the same colours `annotate_qwen` uses on the decision frame. They are the poses
handed to `transfer` / `pick_at` / `place_at`, projected through the
calibration, so the large pane shows the arm against its own plan rather than
against a memory of an overlay that is now two inches wide in the corner.
Either ring can be absent and is labelled `from` / `to`, since a pick has no
destination and a place no source.

**`--record run.mp4` writes that picture to a video** for the whole session,
and `--record-fps` sets the rate (default 10). The compositing thread is what
records, so the window is not required: `--no-preview`, or a machine whose
OpenCV has no GUI, still gets the file — which is the case where a recording is
the only way to see what happened. The loop turns over at camera rate (30.7/s
measured, 26.3 ms of it waiting for the next frame and 6.0 ms drawing), so the
recorder writes on a wall clock rather than once per tick and repeats the
canvas across a late one. That is what keeps a move at real speed: 8.1 s of
file for an 8.0 s run, where one frame per due tick gave 7.4 s. A 1740×720
canvas of the desk costs about 165 KiB/s, so roughly 10 MB per minute of
recorded motion.

**Standing still is recorded a tenth as often**, so waiting on the model plays
back ten times faster. It is the same predicate the panes swap on: motion is
worth real time and a still picture of a desk is not, and most of a session is
the still picture. A pace change takes effect on the tick that caused it rather
than after the deadline already pending — otherwise the switch out of a wait
would swallow the first second of the move, which is the second worth having.

**Typing does not block on the arm.** A transfer is seconds of motion and a
decision is seconds of GPU, and through all of it the prompt still takes input.
Instructions queue FIFO — one arm, so one at a time. The commands worth knowing:

| Command | Why |
|---|---|
| `/stop` | End after the current step and drop the queue. The arm finishes what it is doing, so it stops somewhere it chose to be |
| `/abort` | Also interrupt the move in flight (`Mt4Client.request_interrupt`). The arm halts where it stands with the jaws as they are — the recovery, not the tidy exit |
| `/held [thing]` | What the loop believes is in the jaws, and how to correct it. **Nothing on this rig can sense this**, so after an `/abort` mid-carry the belief and the world can differ and only you can say so. It is reported to the model and drawn in the panel; it never refuses an action, because a belief that cannot be checked must not gate one |
| `/open` | Release the jaws — what `/abort` mid-carry leaves you needing |
| `/scene` | The entity list as the model will be shown it, with the reason each blocked one is blocked |
| `/status` `/park` `/home` | Arm state and recovery, queued behind the run rather than racing it down one serial port |
| `/dry on\|off` | Flip between deciding and moving without restarting |
| `/save` | The frame the model saw, the whole window, and a JSON record of the decision |

**What the gripper holds outlives an instruction**, because the jaws are
physical and finishing a sentence does not empty them. `held` is session state:
stop a transfer halfway and the next instruction is told the arm is still
carrying something.

**One camera, one stream.** The window and the decision captures come off a
single `FrameStream` held open for the session — only one consumer can hold the
device, and on Windows DSHOW a second open simply fails. It is also much
faster: `capture_frame` reopens the device and burns 20 exposure warm-up reads,
2-3 seconds, and the loop used to pay that on every step. `instruct.observe`
therefore takes an optional `frame=`, and the caller owns the freshness
guarantee that reopening used to provide — pull the frame *after* the arm has
parked, which is what `FrameStream.fresh` blocks for.

Everything the loop reports about a move is what it **sent**, never what it
observed; outcome lines say "commanded, not checked". There is no sensor in the
jaws, and the vision test that once stood in for one was measured reporting a
completed pick as a failure because it matched the ArUco tag beside a stapler
instead of the stapler. What is still checked, before the gripper opens, is
reach, keep-out, ground, the jaw-width plan, the desk edge, and both ends of a
transfer.

**What the loop deliberately will not refuse.** Three checks the rest of the
stack applies are off in the instruction path, because each one substitutes a
judgement for the operator's instead of reporting a physical limit. None of
them can produce an illegal pose, and each failure they used to prevent is
visible on the desk and recoverable.

| Off | Why | What it costs |
|---|---|---|
| Camera coverage on a destination | Says the result would leave the frame and could not be re-detected, not that the arm cannot place there. Still on wherever the stack picks a spot itself — a slot nothing can re-detect is a bad autonomous choice | Nothing. It was vetoing **316 cm² of the 1541 cm²** the arm can hold a grasp pose over, 21% |
| Neighbour clearance on a pick | Counts `scene.cubes` and nothing else, so it refuses a grasp beside a cube and permits the same grasp beside a pen — in a loop that never tells the model cubes are a category | A tight pick can nudge a neighbour |
| The gripper state | `held` is typed in by hand, not sensed. A stale belief is the common case after an `/abort` mid-carry, and gating on it blocked a fine action until someone typed `/held` | A PICK while genuinely holding drops the carried object from grab height |

**A destination can carry an orientation.** `dest_2d` says where, and the
optional `dest_axis_2d` says which way: a second image point, such that the
line between them is the direction the object should lie in. Two points rather
than an angle, for the same reason every destination is a point — an angle
would need a frame to be measured in and the model has no access to the
robot's. Absent, the destination squares to the world axes as before. Two
guards matter: the points must be **8mm apart in robot space** to mean a
direction, and `[0, 0]` reads as absent, because this build fills an unused
optional field with zeros rather than null (observed live beside a real
`dest_2d`, where it would otherwise mean "lay it along the line to the frame's
top-left corner").

**A question about the desk is answered in boxes, from a second call.**
`REPORT` is the action for a task whose answer is a list of things rather than a
movement — *"find all the pickable objects"*, *"how many cubes are on the
desk"*. The decision reply only chooses the action; the list comes from
`instruct.enumerate_objects`, a second call on the same frame whose prompt does
nothing but enumerate. Every box it returns is validated exactly as a PICK's
single box is — the same two coordinate readings, the same whole-frame rejection
— then segmented and put through `instruct.source_entity`: reach, the J1
keep-out, ground Z, the desk polygon and the jaw-width plan, the same predicate a
pick has to satisfy. Each object becomes a numbered row with its robot position,
the width the jaws will close across, and either "pickable" or the gate that
stopped it, with the same numbers drawn on the frame in green and red. Nothing
moves and nothing is registered — the ids are the report's own rows and do not
outlive it.

**Why a second call.** Measured on a nine-cube desk (2026-08-04, greedy, one
frame):

| prompt | objects returned |
|---|---|
| decision prompt with an `objects` field | **1** of 9 |
| the same, worded harder | **0** of 9 |
| the same, `objects` first in the schema | **8** of 9 |
| a prompt that only enumerates | **9** of 9 |

The mechanism is field order. Greedy decoding writes the keys in the order the
schema lists them, so with `box_2d` first the reply commits to a single target
and the list becomes an echo of it — the 1-of-9 reply boxed one red cube in
`box_2d` and copied it into `objects`, reasoning that it "is the only object
identified as being on the desk that can be picked up". Putting `objects` first
fixes the count and **breaks the movement actions instead**: asked to *"put a
blue cube on marker 2"*, that ordering answered TRANSFER with marker_2's box in
`box_2d`, which is the field the pick target is read from — the arm would have
tried to pick up the printed tag. A silent wrong target is worse than a short
list, so the decision prompt keeps its shape and has no list in it at all.

The model is not the limitation: its own bare grounding phrasing
(*"Locate all the cubes in the image, output their bbox in JSON format"*)
returns 9 of 9 on the same pixels, raw or annotated. The enumeration prompt is
that shape with the task text woven in, and it reads back through
`qwen.parse_regions` — the model answers in its own convention (`bbox_2d`, in a
markdown fence), so nothing imposes a schema on it.

On the same nine cubes the reported positions land **0.3-8.2mm** from the HSV
detector's reading, and every colour label was right.

**Pickability is not the model's to judge**, and the enumeration prompt says so.
Whether the arm can pick something up depends on how far it reaches, how close
to its own base it can work, where the desk ends and how wide the jaws open, and
none of that is in the photograph. List every candidate; the gates rule out the
failures. An object left out because it looked hard to grasp would be gone from
the answer with nothing able to notice — the same argument that keeps the cube
list out of the decision prompt.

**The reported width is `grip_mm`, not the silhouette's extent.** GrabCut from a
box takes in the object's shadow: on those nine cubes the extent read 38-64mm
long for 20mm cubes while the planned grip width read 20.5-28.7mm. The position
is sound either way, but a size column two to three times over is worse in an
operator's hands than none, and the grip width is what the pick is planned on.

The list is never quietly shortened. A box that will not read — outside the
frame, or covering it — is reported as a named gap (`Action.report_notes`)
rather than dropped, and one that reads but will not segment keeps its row with
the segmenter's complaint. An empty list is a real answer and not a refusal: it
says there are none. If the service fails on the enumeration call the report
comes back empty *with the reason as a note*, because the decision has already
been made and reported by then and losing the list must not lose the reply.

A REPORT ends the instruction, counted as success. The report *is* the answer,
so another step would put the same question to the same unchanged desk — the
argument a chosen STOP already runs on. A task that wants something reported
*and* moved is two instructions.

One wart: `reason` comes from the decision call, which has not enumerated
anything, so its prose can disagree with the list ("There are two green cubes
visible" above three rows). The prompt tells it not to count there. The
authoritative count is the measured one, printed under the rows.

**A STOP the model chose is terminal.** It is the model's answer to the
request, so re-asking is asking the same question again. Live, `open the
gripper` — an instruction naming no object to see — was answered STOP six times
running, each time in different words, spending six park-capture-decide cycles
to arrive where step 1 already was. Full jaws do not buy a retry: the object
stays in them either way, so the loop says so on the first step rather than the
sixth. `Action.declined` marks it, and it is set only where the model's own
`action` field says STOP.

**Every other refusal retries once, and the same one twice stops.** Those are
the refusals the model did not choose — a malformed or missing reply, and the
physical gates on a measurement or a pose. A refusal caused by the frame rather
than the request goes away on a fresh park and capture, so one more look is
worth it whatever the jaws hold. A second identical refusal is worth nothing —
nothing moved in between, so the reason is about the request — and repeating it
spends the whole step budget arriving where step 1 already was. Sameness is
judged with the numbers stripped, because a re-measured object jitters: one
out-of-reach blob refused at `r=367mm` and then `r=373mm`, two strings for one
problem. That test works on the gates' fixed wording and would not have worked
on model prose, which is the other reason a chosen STOP is not re-asked.

`decide` re-asks once **sampled** when a reply carries no JSON at all — greedy
re-asking is the same prompt against the same weights and returns the same
reply, so that retry is the one place this loop does not decode greedily.

**The action list depends on what the jaws hold.** With them empty, TRANSFER is
described as preferable to PICK-then-PLACE, which it is — one motion against
two, with a park and a second decision in between. With them full that
preference is false, because TRANSFER there means picking up a *second* object
and abandoning the first, so it is not stated and both pick actions say plainly
that they act on a different object. The model cannot see into the gripper:
given a standing "prefer TRANSFER" and the task *"place the held object on
marker 2"*, it was measured matching "the held object" to a lookalike on the
desk and transferring that instead — twice in a row, each attempt refused for
max reach on a blob at r=367mm, a message that names nothing to do with the
real problem.

## Configuration (environment variables)

Set these in `/etc/systemd/system/qwen3-vl.service` (then `sudo systemctl daemon-reload`):

| Variable | Default | Purpose |
|---|---|---|
| `QWEN_VL_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | HF repo id to load |
| `QWEN_VL_HOST` | `127.0.0.1` | Bind address |
| `QWEN_VL_PORT` | `8766` | Bind port |
| `QWEN_VL_DEVICE` | `cuda` (auto-detected) | `cuda` or `cpu` |
| `QWEN_VL_LOAD_IN_4BIT` | `1` | Set to `0` to load full-precision (needs ~16GB+ VRAM, won't fit this card). Equivalent to `QWEN_VL_QUANT=none` |
| `QWEN_VL_MAX_NEW_TOKENS` | `256` | Default generation cap if the request omits `max_new_tokens` |
| `QWEN_VL_QUANT` | `int4` | 4-bit backend: `int4` (torchao, compileable), `nf4` (bitsandbytes), `none` |
| `QWEN_VL_INT4_GROUP` | `128` | int4 quantization group size |
| `QWEN_VL_INT4_PACKING` | `tile_packed_to_4d` | torchao packing format. Its own default (`plain`) targets newer architectures; this is the tinygemm layout with a fast sm86 kernel |
| `QWEN_VL_INT4_SKIP` | `visual` | Modules left in bf16, comma-separated. Empty to quantize everything |
| `QWEN_VL_CACHE_LEN` | `1664` | Static KV-cache window. `0` for the dynamic cache (and no compilation). Size it tightly — see [Decode throughput](#decode-throughput) |
| `QWEN_VL_COMPILE` | `1` | CUDA-graph the decode step. Needs a static cache; **a no-op on its own with `nf4`** |
| `QWEN_VL_FORCE_COMPILE` | `0` | Override `is_compileable=False`. Only needed for `nf4`; torchao passes the gate on its own |

Set `QWEN_VL_COMPILE=0` to revert to the original ~12 tok/s behaviour, e.g. if you
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

Load plus int4 quantization takes ~4s and compilation another ~20s, so allow ~30s;
a `/health` before then reports `loaded: false` rather than failing. If the unit file changed,
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

The venv is uv-managed and has **no `pip`**, so `python -m pip install` fails with
`No module named pip`. Use uv, with the venv named explicitly (it isn't on `PATH`
for a non-login SSH shell either):

```bash
cd /opt/qwen3-vl && VIRTUAL_ENV=/opt/qwen3-vl/.venv ~/.local/bin/uv pip install <pkg>
```

## See also

- [docs/GROUNDING_DINO.md](GROUNDING_DINO.md) — the other vision service on this
  GPU host: open-vocab *detection* rather than description, and the source of the
  deployment/tunnel/WSL2 details this doc refers to
