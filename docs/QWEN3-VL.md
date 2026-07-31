# Qwen3-VL service

A local HTTP wrapper around `Qwen/Qwen3-VL-4B-Instruct`, loaded in 4-bit (NF4) via
bitsandbytes. Lives at `/opt/qwen3-vl`, in the same WSL distro as the Grounding
DINO service (`/opt/grounding_dino`) — on the reference deployment (`media`),
that's the `openpi`-named distro (see [docs/GROUNDING_DINO.md](GROUNDING_DINO.md)
for the WSL2 host-level gotchas that apply here too).

- **Port:** `8766`, bound to `127.0.0.1` only (not reachable outside the distro/host
  without a tunnel — see below).
- **VRAM footprint:** ~3.7GB loaded and idle, leaving ~4.2GB headroom on the RTX 3070
  for KV-cache and vision tokens.
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

Model load takes ~20-30s, so a `/health` immediately after the restart reports
`loaded: false` rather than failing. If the unit file changed, `install -m 644`
it into `/etc/systemd/system/` and `systemctl daemon-reload` first — `scp` alone
leaves the old unit in force.

## Known gotcha

`transformers`' Qwen3-VL processor pulls in a video-processor class that requires
`torchvision` even for pure image use — it's in `requirements.txt` now, but if you
rebuild the venv from scratch and see `ImportError: Qwen3VLVideoProcessor requires
the Torchvision library`, that's why.

## See also

- [docs/GROUNDING_DINO.md](GROUNDING_DINO.md) — the other vision service on this
  GPU host: open-vocab *detection* rather than description, and the source of the
  deployment/tunnel/WSL2 details this doc refers to
