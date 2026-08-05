# SAM 2.1 — in-process segmentation

Prompted segmentation for the MT4 desk camera. `mt4_vision.sam` loads
`facebook/sam2.1-hiera-small` in the same Python process that calls it; there
is no HTTP service and no MCP tool.

**Every measurement that starts from a detector box goes through it**, so it is
not an optional extra for the open-vocabulary half of this repo — see
[What depends on SAM being available](#what-depends-on-sam-being-available).
Cube pick/place, stacking, calibration and the jog client do not touch it.

**What it buys you:** a point or a box becomes a *silhouette*. Grounding DINO
says the pen is somewhere in this rectangle; this says which pixels are pen. On
the reference desk a click on the Statue of Liberty figurine returns its outline
including the raised torch — a shape no HSV cube detector and no bounding box
describes.

```
 mt4_vision.locate / CLI
        │
        ▼
 mt4_vision.sam          Sam2Model + Sam2Processor (lazy, first use)
   segment(frame, …) ──► embeddings cache (keyed by frame bytes)
   embed(frame)          fp16 on CUDA, fp32 on CPU
```

---

## Install

Needs Python 3.10+. A CUDA GPU is much faster; CPU works at about a second per
new frame.

```powershell
pip install -r requirements-sam.txt
```

`transformers` must be ≥ 4.57, which is where `Sam2Model` and `Sam2Processor`
arrived. `torchvision` and `accelerate` are listed because transformers reaches
for them internally.

The first call downloads ~180 MB of weights into the HuggingFace cache, then
loads and warms the model. On CUDA with compile enabled, a cold inductor cache
costs ~100 s once; later starts reuse that work. Importing `mt4_vision.sam`
does not load anything — load happens on the first `segment` / `embed`, or when
you force it.

```powershell
python -c "from mt4_vision.sam import health, segment; print(health())"
python -m mt4_vision --camera 1 sam --pixel 737 570
```

The checkpoint is published as a video model, so loading it into the image-only
`Sam2Model` logs a note about instantiating a `sam2` from a `sam2_video`
checkpoint. That is expected: the image model is a subset of the video one.

### Configuration

All environment variables; there is no config file:

| Variable | Default | Notes |
|----------|---------|-------|
| `SAM2_MODEL` | `facebook/sam2.1-hiera-small` | `…-tiny` is smaller, `…-base-plus` and `…-large` more accurate and slower |
| `SAM2_DEVICE` | `cuda` if available, else `cpu` | |
| `SAM2_DTYPE` | `fp16` on cuda, `fp32` on cpu | `bf16` is available and measurably worse here — see below |
| `SAM2_COMPILE` | `1` | `0` skips compiling the image encoder (CUDA only): ~8 ms slower per new frame, faster to start |
| `SAM2_COMPILE_MODE` | `max-autotune-no-cudagraphs` | Any `torch.compile` mode |
| `SAM2_CACHE_FRAMES` | `8` | Encoder outputs kept, ~8.4 MB of VRAM each on CUDA |
| `TORCHINDUCTOR_CACHE_DIR` | torch's default | Point it somewhere persistent or every restart recompiles from scratch |

---

## What makes it fast

Measured on the reference RTX 3070 (sm86, 8 GB) with a 1280×720 desk frame
(when this code still ran as a side service on that card). Each of these is
still in the in-process path; the ones deliberately *not* taken are at the end.

**A call is two costs, not one.** The image encoder is the expensive half and
it does not depend on the prompt, so embeddings are cached by the bytes of the
BGR frame:

| | encoder | prompt decoder | request total |
|---|---|---|---|
| Frame not yet cached | 26–38 ms | 9–19 ms | **45–60 ms** |
| Frame already in the cache | — | 9–17 ms | **17–21 ms** |

Those are steady-state CUDA numbers. On CPU expect about a second per new frame.
The first real call after idle can cost more while clocks and kernels settle —
warmup at load absorbs the multi-second first-touch cost.

`embed(frame)` encodes early (for example while the arm is still moving); later
`segment(None, image_id=…, …)` prompts that cache entry.

**fp16, not bf16.** Encoder time on the same frame, and the worst mask
agreement against fp32 over four exposures of the desk plus a noise frame:

| dtype | encoder | worst IoU vs fp32 |
|---|---|---|
| fp32 | 101 ms | — |
| bf16 | 39 ms | 0.996, but **0.499** on a frame darkened to 0.25× |
| fp16 | 33 ms | 0.998 |

The bf16 outlier is not a corrupted mask. Its coarser mantissa flipped which of
the three candidates won a near-tied predicted-IoU score. fp16 is both the
faster and the closer of the two.

**Compile the encoder, leave the decoder alone.** The encoder always sees
1024×1024×3 after the processor, so one static shape compiles once. The
decoder's shapes follow the prompt count. Compiling from nothing costs ~100 s;
against a populated `TORCHINDUCTOR_CACHE_DIR` it costs a few seconds.

**Preprocess and post-process on the GPU when available.** The processor accepts
`device="cuda"` for resize/normalize; mask logits are upsampled to the frame
before leaving the device.

### Deliberately not done

- **channels_last** and **TF32** — no change worth the lines on the reference card.
- **CUDA graphs** (`mode="reduce-overhead"`) — replay reuses output buffers while
  this module holds encoder outputs across calls.
- **Automatic "segment everything"** — not exposed; prompt with points or boxes.

---

## API

[`mt4_vision/sam.py`](../mt4_vision/sam.py):

| Call | Meaning |
|------|---------|
| `health()` | `ok`, `model`, `device`, `loaded`, `dtype`, `compile`, cache stats. Does not load the model |
| `embed(frame)` | Encode now; return `image_id` |
| `segment(frame, points=…, labels=…, boxes=…, multimask=True, image_id=…)` | Masks for the prompts. `frame=None` with `image_id` uses the cache |
| `best_per_object(masks)` | Top candidate per prompt |
| `decode_rle(runs, h, w)` | RLE helper (tests / tooling) |

`points` is `[[x, y], ...]`, one object per point, or `[[[x, y], [x, y]], ...]`
to put several points on one object. `labels` matches that shape: 1 on the
object, 0 off it. `boxes` is `[[x1, y1, x2, y2], ...]`. All coordinates are
pixels in the submitted frame.

With `multimask` the model returns three candidates per object, ranked by its
own predicted IoU. A single point is genuinely ambiguous — the cube, its top
face, or the stack. Take the first unless you have a reason not to.

Raises `SamError` when deps are missing, the model fails to load, prompts are
malformed, or an `image_id` has aged out of the cache.

### Day-to-day CLI

```powershell
python -m mt4_vision --camera 1 sam --pixel 737 570
python -m mt4_vision --camera 1 sam --box 671 523 787 647
python -m mt4_vision --camera 1 sam --pixel 737 570 --candidates
```

---

## Consumers in this repo

| Entry point | What it does |
|-------------|--------------|
| [`mt4_vision/sam.py`](../mt4_vision/sam.py) | In-process model. `segment` → `Mask` list with `mask`, `score`, `area`, `bbox` and centre-of-area `cx`/`cy` |
| `python -m mt4_vision sam …` | Segment at a pixel or box, print numbers, save `sam_frame.jpg` |
| [`mt4_vision/locate.py`](../mt4_vision/locate.py) | **Every measurement that starts from a box.** `measure_sam` prompts with the box; `measure_with_box_fallback` is the ladder Grounding DINO, Qwen3-VL, the MCP tools and `relocate` all reach it through |

A mask is not yet a grasp. `mt4_vision.locate` turns pixels into a centre, a
long axis and millimetres, and `grasp_feasibility` decides whether the jaws can
take it.

### What depends on SAM being available

Anything that measures an object **from a box**: `mt4_locate_by_prompt`,
`ask_dino.py`, `ask_qwen.py` and the instruction worker, `python -m mt4_vision
grounding --locate`, and re-acquiring a registered object with
`locate.relocate`.

Untouched by a SAM outage: cube pick/place, stacking, calibration, the jog
client, and `locate.measure` from a bare pixel, which segments by how far each
pixel sits from the local desk colour.

When the model cannot load or run, a box measurement **refuses and says so**.
It does not quietly fall to the rungs below — the desk-deviation path cannot
segment a stapler at all, and the last rung reports the box's own dimensions as
the object's (150 × 74 mm for a stapler, measured 2026-08-02), which is a wrong
number that looks like a right one.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `SAM 2.1 needs torch, transformers…` | Deps not installed | `pip install -r requirements-sam.txt` |
| First call takes ~100 s | Cold `torch.compile` / inductor cache | Set `TORCHINDUCTOR_CACHE_DIR` somewhere persistent, or `SAM2_COMPILE=0` |
| `image_id … is not cached` | Frame aged out (8 frames by default) | Pass the frame again, or raise `SAM2_CACHE_FRAMES` |
| Mask covers the object *and* its shadow | Ambiguous point; top candidate is the wide one | Add a label-0 point on the shadow, or prompt with a box |
| Mask is a small part of the object | Point landed on a sub-part | Move toward the middle, or use a box |
| `health()` shows `device=cpu` unexpectedly | No CUDA torch / driver | `torch.cuda.is_available()`; check the GPU driver |
| Slow (~1 s) per frame | Running on CPU | Expected without a local GPU |

---

## See also

- [docs/GROUNDING_DINO.md](GROUNDING_DINO.md) — the box-producing HTTP service whose output makes a good prompt here
- [docs/CALIBRATION.md](CALIBRATION.md) — the measurement layers a mask feeds into
