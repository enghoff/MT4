# Qwen3-VL Policy Mapped to the MT4 Repository

Revised 2026-08-02 against the code and against the measurements in
[docs/QWEN3-VL.md](QWEN3-VL.md). Two revision passes are marked:
**[r1]** for corrections forced by measured VLM behaviour, **[r2]** for changes
forced by the cube-agnostic objective below.

## Objective

Add Qwen3-VL-based instruction understanding and target selection to the
existing MT4 stack without replacing its calibrated geometry, object
measurement, motion planning, or safety checks.

**The target is a cube-agnostic flow.** Any pickable object on the desk —
coloured cube, pen, key, stapler, eraser — should reach the arm through the same
path, with the same accuracy and the same failure modes. Cubes are a
**benchmark control**, not the design centre: they are the one object class with
free, automatic ground truth (HSV gives the label, ArUco gives the frame), which
makes them ideal for measuring a general path and misleading as a target for
one.

That objective has a sharp consequence for this document: **every place the
stack treats a cube specially is a place the flow is not yet general**, and each
one needs either a general replacement or an explicit, measured justification
for staying special. The audit is below.

The policy decides:

```text
What operation should happen next?
Which entity should it act on?
```

It does not predict robot coordinates or low-level motion.

---

## Existing MT4 Execution Stack

```text
Camera frame
    ↓
mt4_vision.scene / mt4_vision.locate
    ↓
mt4_vision.entities.build_snapshot(...)
    ↓
Snapshot: cube_N | marker_N | slot_N | obj_N
    ↓
Policy selects action + entity
    ↓
mt4_mcp.server: mt4_pick(entity_id) / mt4_place(entity_id)
    ↓
fresh-frame re-acquisition
    ↓
Entity.as_grasp(...)
    ↓
mt4_vision.motion: pick_at / place_at / transfer
    ↓
calibrated and validated MT4 movement
```

| Responsibility | Existing implementation |
|---|---|
| Addressable scene objects | [mt4_vision/entities.py](../mt4_vision/entities.py) |
| Cube/marker detection | [mt4_vision/scene.py](../mt4_vision/scene.py), [detect.py](../mt4_vision/detect.py) |
| Object segmentation and measurement | [mt4_vision/locate.py](../mt4_vision/locate.py) |
| Calibration and parallax | [mt4_vision/calib.py](../mt4_vision/calib.py) |
| Motion primitives | [mt4_vision/motion.py](../mt4_vision/motion.py) |
| Entity tools and re-acquisition | [mt4_mcp/server.py](../mt4_mcp/server.py) |
| Pointing overlay (grid + drawn ids) | [mt4_vision/preview.py](../mt4_vision/preview.py) |
| Open-vocabulary detection | [mt4_vision/grounding.py](../mt4_vision/grounding.py) |
| Qwen client and region parsing | [mt4_vision/qwen.py](../mt4_vision/qwen.py) |

> **Name collision.** [mt4_vision/policy.py](../mt4_vision/policy.py) is already
> the shuffle planner. The VLM policy needs a different module name —
> `mt4_vision/instruct.py` or similar.

---

## The Cube-Specific Surface **[r2]**

Where a cube is treated differently from any other object today, what the
general equivalent is, and whether the general path is already good enough.
This table is the work list for cube-agnosticism; the sections after it expand
the entries that are not one-liners.

| # | Cube path | General path | Status |
|---|---|---|---|
| 1 | **Detection.** HSV colour blobs, enumerated every frame, stateless ([detect.py](../mt4_vision/detect.py)) | Objects must be *registered* by pixel hint or DINO, then template-matched, and drop out on failure ([server.py:305](../mt4_mcp/server.py#L305)) | **The real asymmetry.** No general enumerator exists. See below |
| 2 | **Parallax.** Fitted `cube_top_homography` at a fixed `cube_height_mm = 20` ([calib.py:154](../mt4_vision/calib.py#L154)) | `_unproject` at any height, with height inferred from silhouette inflation ([locate.py:432](../mt4_vision/locate.py#L432), [locate.py:449](../mt4_vision/locate.py#L449)) | **Already general.** The cube path is the legacy special case. Cross-validate, then consider retiring |
| 3 | **Re-acquisition.** Colour re-detect within 15mm ([server.py:584-615](../mt4_mcp/server.py#L584-L615)) | Template match, strict, drops on failure ([locate.py:849](../mt4_vision/locate.py#L849)) | Two mechanisms, different failure modes. Unify or measure both against one standard |
| 4 | **Pickability ladder.** `pick_block_reason` — blob-area floor/ceiling, reach, hull, neighbour clearance ([entities.py:201](../mt4_vision/entities.py#L201)) | `object_entity` — reach, hull, jaw-width-aware clearance ([entities.py:461](../mt4_vision/entities.py#L461)), plus `grasp_feasibility` — jaw span, wrist angle, camera radius ([locate.py:903](../mt4_vision/locate.py#L903)) | **Three ladders that disagree.** See below |
| 5 | **Grip.** Calibrated `grip_close_s` | Full open, full close, servo stops on resistance ([entities.py:132](../mt4_vision/entities.py#L132)) | General path is strictly more general; cube value is an optimization |
| 6 | **Yaw periodicity.** Hard-coded `YAW_PERIOD_SQUARE` | `is_compact(long, short)` → 90° or 180° | General path yields 90° for a cube anyway. Unify for free |
| 7 | **Label.** Free and correct from HSV (`"red cube"`) | From a VLM or DINO — unreliable at attributes | Not fixable. Shapes the resolver design, see below |
| 8 | **Deduplication.** `is_own_colour_blob` stops a coloured object being listed twice ([entities.py:440](../mt4_vision/entities.py#L440)) | — | Becomes load-bearing the moment both detectors run every frame |

Rows 2, 5 and 6 are the good news: the general mechanism already exists and
already subsumes the cube case. Row 1 is the structural problem. Row 4 is a
correctness problem that the objective exposes.

---

## Measured Results, 2026-08-02

First implementation pass. One live frame (1280x720, 9 cubes, 5 markers, plus a
figurine, a remote, cable runs and the arm parked over the desk), saved and
worked offline so every number below is reproducible against the same pixels.

### Phase 0 gate: FAILED

`locate.measure` was pointed at each HSV cube centroid -- exactly what a hint
from a human or a VLM does -- and compared against the HSV +
`cube_top_homography` answer for the same cube:

| cube | HSV robot | generic robot | error |
|---|---|---|---|
| blue | (179, -186) | (174, -199) | 14.2mm |
| blue | (157, -77) | (164, -83) | 9.7mm |
| green | (166, 103) | (176, 95) | 12.5mm |
| red | (143, 263) | (161, 242) | 27.7mm |
| red | (84, -155) | (112, -150) | 29.1mm |

n=5, mean **18.6mm**, median 14.2mm, max 29.1mm, against a ~10mm pick
tolerance. One further cube raised `LocateError` outright. **The generic path
cannot currently replace the cube path**, so nothing that acts on the arm
should be switched over.

**Root cause, and it is not the unprojection.** `_assumed_height_mm` divides by
`1 + gain * cos_radial`, where `cos_radial` is measured across the object's
*long axis*. A cube has no long axis -- `minAreaRect` picks one arbitrarily --
so `cos_radial` came out 0.03, 0.06, 0.15, 0.38, 0.58 across five identical
cubes. Where it is near zero the divisor collapses to 1 and the whole silhouette
width is attributed to height: inferred heights of 7.2 to 32.3mm for objects
that are all 20mm. The two worst cubes (cos_radial 0.03 and 0.06) are exactly
the two with ~28mm error.

So the estimator is unstable *for compact objects specifically*, which is
unfortunate because compact is what the benchmark object is. A radial /
tangential formulation would be axis-independent, and a first look at it
(silhouette extent along the nadir direction minus the extent across it) gave
heights of 1.9 to 15.9mm -- still scattered, because the silhouettes also
include cast shadow. **Both would need fixing, and both change where the arm
goes for every pointed-at object, so neither was attempted here.**

### Phase 1 detection parity: works, and already outperforms HSV

`mt4_vision/discover.py` proposes hints with no prior knowledge and hands each
to the ordinary `measure` path. On the same frame it found **all 9 cubes**;
`mt4_scene` reports **6**. The three it added expose two separate pre-existing
bugs:

1. **The desk polygon is ~20mm too tight at the back edge.** Two real,
   reachable cubes are discarded by `detect_cubes`' `on_table` test, missing the
   polygon by **17.1mm** and **0.2mm**. `off_table_blobs=7` in the scene summary
   is the signature CLAUDE.md already names. The other five discards are the
   arm and the wall, correctly rejected at x = -148 to -6990mm. Fix is to re-run
   `calibrate_table_edge.py`, which rewrites the calibration and needs the arm
   parked clear -- not done here.
2. **One green cube is invisible to HSV entirely**, not even a raw blob. A
   colour-range or saturation miss at the back of the desk, unrelated to (1).

Timing: 0.18s to propose, 0.46s to measure everything, single-threaded.

### The desk does not have to be uniform

The concern that this method needs a plain desk is measurably wrong, and the
tests now pin it: an illumination gradient of 90 grey levels corner to corner
(larger than the object-to-desk contrast) plus horizontal grain changes neither
which objects are found nor where they measure, to within 2mm. A bare desk with
both nuisances turned on proposes **nothing**. That is what the local median
background buys, and it is why a single reference colour was abandoned.

### The real limit is width, and it bites at cube scale

An object wider than the background kernel does not disappear cleanly -- its
interior is absorbed but its **corners still deviate**, and each corner, cut out
and measured alone, comes back as a small, plausible, stable object. Measured on
a synthetic 183px square: four phantom 12x6mm objects, all four surviving the
two-window stability check. On a real desk that is a book offering the arm four
fake grasp points, which is worse than not seeing the book. `discover` now
vetoes it with a second, coarser deviation pass.

Tuning that veto exposed how tight the margin is. Coarse widths on the live
frame, in full-res pixels:

```text
cubes                     24, 40, 40, 48, 57, 64
arm base / remote / arm   94, 101, 242
```

A 20mm cube is only ~40px of footprint, but this mount is steeply oblique, so
its side face projects out beside it and the silhouette reaches 64px -- past
the 61px kernel. **Vetoing at exactly one kernel deletes cubes.** The threshold
sits at 1.25 kernels, in the 64-94px gap. The practical scope is therefore
narrower than "30mm": it is 30mm of *silhouette*, and an oblique camera inflates
a footprint by 1.5-2x, so the honest object-width limit is nearer 15-20mm with a
cube right at the boundary.

The veto also has to be a per-pixel width test, not a per-component one. Judging
whole connected components was tried first and deleted two real cubes, because
at the coarse scale a cube 100px from the arm's base merges into the base's blob
and inherits its width.

### Known false positive

One 40x4mm sliver at pixel (491, 392), robot (45, -140) -- a shadow edge or
cable between the remote and a cube. It clears the keep-out by 7mm and is
therefore reported as pickable. Nothing currently refuses it: the jaw-span gate
in `grasp_feasibility` is inert, because `grip_span_s_at_zero_mm` and
`grip_span_s_per_mm` are `null` in `vision_calibration.json`. Calibrating the
jaw span would refuse a 4mm object on its own terms, which is better than
inventing a width floor in the enumerator.

---

## Detection Parity Is the Real Asymmetry **[r2]**

Today `mt4_scene` lists **every cube automatically** and **only those objects
someone already registered**. So the flows are not the same length:

```text
cube:      point → bind → pick                    (1 model call)
anything:  point → locate/register → bind → pick  (2 model calls, and
                                                   only if the operator
                                                   already knew to look)
```

Worse, the entity list the policy points into is cube-complete and
object-sparse. A stapler that nobody registered is not merely hard to select —
it is *not in the world model at all*, and the correct answer to "pick up the
stapler" becomes `STOP: no such entity`. That is a scene-completeness failure
dressed as a referent failure, and no amount of policy work fixes it.

Equal ability requires the snapshot to enumerate arbitrary objects on every
frame. Three options, in the order I would try them:

**(a) Class-agnostic desk segmentation — preferred.** [`_segment`](../mt4_vision/locate.py#L182)
already estimates the local desk with a large median filter and finds the
foreground component by deviation from it; [`marker_paper_mask`](../mt4_vision/locate.py#L153)
already excludes the ArUco papers, which were *"every segmentation failure that
survived a decent hint"*. Today it runs from a hint. Generalizing it to
enumerate all desk-deviating components turns it into a deterministic,
VLM-free, camera-rate object proposer that feeds the measurement code already
written for it. Geometry stays deterministic; the VLM only labels and selects.
Risks to handle: the arm's own body, shadows, and the desk edge — the last of
which `in_work_region` already gates on the measured desk polygon.

**(b) Qwen as the enumerator.** `OBJECTS_PROMPT` is measured at 10 boxes, 3/3
reproducible under greedy ([QWEN3-VL.md § Getting a JSON object
list](QWEN3-VL.md)), and `measure_with_box_fallback` already turns a box into a
measured `obj_N` — it is exactly what the DINO path does at
[server.py:505](../mt4_mcp/server.py#L505). Costs 150–350 tokens, so ~5–9 s per
observation, and it **moves the VLM from advisory to load-bearing for
perception**. That is a real architectural change: today *"nothing in the arm
stack depends on the VLM"*. Worth doing knowingly, not by drift.

**(c) Grounding DINO with a generic prompt.** ~3 Hz and real boxes, but it
cannot co-run with Qwen on the 8GB card, so it forces a service switch per
observation. Useful as an offline ground-truth generator for the benchmark;
impractical in the live loop alongside Qwen.

Whichever is chosen, `is_own_colour_blob` (row 8) becomes load-bearing: with a
general enumerator running alongside HSV, every coloured cube is detected twice
and must resolve to one id.

---

## Geometry Is Already Object-General **[r2]**

Worth stating plainly because it inverts the obvious worry. A single oblique
scene camera cannot see height, and height aliases into XY error — but this
is solved, and solved generically:

- [`_parallax_gain`](../mt4_vision/locate.py#L416) gives millimetres of
  table-plane error per millimetre of object height at (x, y). On this mount the
  nadir sits 360–490mm away, so the gain is **1.4–2.0**.
- [`_unproject`](../mt4_vision/locate.py#L432) inverts the parallax for a point
  at any height — the exact inverse of `robot_to_pixel`'s forward scaling.
- [`_assumed_height_mm`](../mt4_vision/locate.py#L449) infers the height from
  how much the object's own height inflated its silhouette, assuming a
  cross-section as tall as it is wide.

The stakes are documented and hardware-verified: a 12mm pen is reported ~9mm
from where it is, *"past the ~10mm the jaws tolerate — the arm shoved the pen
instead of gripping it, in exactly the direction this predicts."*

The `h ≈ w` assumption is where this can fail, and it fails in known directions:
a flat sheet is reported thicker and nearer than it is; a tall narrow object has
its height under-attributed. Both are scope boundaries to state, not bugs.

**A cube satisfies `h ≈ w` exactly** (20 × 20mm), which yields a clean, cheap
validation experiment and the single best evidence for cube-agnosticism:

> Run the same physical cube through both paths — HSV + `cube_top_homography`,
> and the generic segment + `_assumed_height_mm` + `_unproject` path — and
> compare the reported XY. Agreement within a few mm means the general path
> handles the benchmark object as well as its dedicated pipeline does, on the
> one object where an independent, empirically-fitted answer exists.

If that holds, `cube_top_homography` is an optimization rather than a
requirement, and rows 2, 5 and 6 of the audit collapse. If it does not hold, the
gap is quantified and the reason is findable — the fitted homography also
absorbs blob-centroid bias, which the generic path does not model.

Run this on hardware before any policy work. It needs no VLM, no instruction
parsing, and no benchmark suite, and it gates whether the objective is
achievable with this camera at all.

---

## One Pickability Ladder **[r2]**

Three predicates currently decide whether a thing can be picked, and they
disagree:

**Partly fixed already.** Commit `4ca5542` (2026-08-02) replaced the marker hull
with one region predicate, `workspace.in_work_region` /
`work_region_block_reason`, and all three ladders now call it
([entities.py:227](../mt4_vision/entities.py#L227),
[entities.py:494](../mt4_vision/entities.py#L494),
[locate.py:913](../mt4_vision/locate.py#L913)). Keep-out, reach, desk polygon
and camera coverage are therefore no longer a source of disagreement, and
`MAX_VERIFIABLE_RADIUS_MM` is gone. What is left:

| Check | `pick_block_reason` (cube) | `object_entity` (obj) | `grasp_feasibility` |
|---|---|---|---|
| Work region (`work_region_block_reason`) | ✓ | ✓ | ✓ |
| Blob area 400–5000px | ✓ | — | — |
| Neighbour clearance | 45mm fixed | jaw-width-aware | — |
| Jaw span vs object width | — | — | ✓ (inert, see below) |
| Wrist angle in soft limits | — | — | ✓ |

Blob area is legitimately detector-specific and should stay declared as an HSV
add-on. The other two rows are the remaining drift.

**The jaw-span check is currently a no-op.** `_span_mm` returns `None` unless
`grip_span_s_at_zero_mm` and `grip_span_s_per_mm` are set, and neither is
present in `vision_calibration.json` today. So nothing refuses an object too
wide for the jaws. That matters more under a general enumerator, which will
propose objects HSV never would.

And `grasp_feasibility` is only ever called **at registration** — in
`mt4_locate_at_pixel` and `mt4_locate_by_prompt`
([server.py:444](../mt4_mcp/server.py#L444),
[server.py:517](../mt4_mcp/server.py#L517)) — never from `build_snapshot` or
`_reacquire`. So on every later snapshot, an object's `pickable` comes from
`object_entity` alone, and the jaw-span and wrist-angle gates are silently not
re-applied.

For a cube-agnostic flow this needs to be **one predicate over a common
`Entity`**, evaluated everywhere `pickable` is set, with detector-specific
checks (the HSV blob-area gates) declared as add-ons rather than as part of the
definition. `pick_block_reason`'s docstring already states the principle — that
it must not *"drift into a second, disagreeing definition of pickable"*. There
are currently three.

Note also that the cube ladder never checks jaw span or wrist angle, and the
object ladder never checks camera-verifiable radius. Neither omission is
deliberate.

---

## The Action Contract

The executed action carries an **entity reference**, never robot coordinates:

```json
{
  "action": "PICK_ENTITY",
  "entity_id": "obj_3",
  "confidence": 0.94
}
```

`mt4_pick("obj_3")` ([server.py:633](../mt4_mcp/server.py#L633)) does more than
execute at stored coordinates. It verifies the id exists in the current
snapshot, checks pickability, captures a fresh frame, re-acquires the same
physical object, **refuses nearest-match substitution** — a target that moved
more than 15mm is an error, not an approximation
([server.py:594](../mt4_mcp/server.py#L594)) — converts to a `Grasp`, and runs
the validated `motion.pick_at` sequence. A coordinate-only contract discards all
of it.

### Action vocabulary **[r1]**

```text
PICK_ENTITY(entity_id)
PLACE_ENTITY(entity_id)
LOCATE_AT_PIXEL(view, label, point)     # registration fallback, see below
DONE(summary)
STOP(reason)
```

`DONE` is separate from `STOP` deliberately: the first draft had `STOP` carrying
both "I cannot resolve this" and "the task is finished", which leaves the closed
loop with no termination signal and makes refusal rate unmeasurable.

Once detection parity (a) or (b) lands, `LOCATE_AT_PIXEL` stops being the normal
route to a non-cube object and becomes a fallback for something the enumerator
missed. That demotion is the point — it is the step that makes the non-cube flow
longer than the cube flow.

Not exposed as policy outputs: `OPEN_GRIPPER`, `CLOSE_GRIPPER`, `MOVE_ABOVE`,
Δx/Δy/Δz, joint commands. `pick_at` and `place_at` already control approach,
gripper, lift, wrist orientation, and routing.

---

## How the Entity ID Is Produced **[r1]**

The first draft asked the model to read a text entity list and return the
matching id. That relies on the capability this build measurably does worst.
From [QWEN3-VL.md § Coordinate space](QWEN3-VL.md):

> named objects (stapler, red cube, toy figurine) box tightly and usefully. But
> it labelled the red cube "green cube" while boxing it correctly, and counted
> "2 cubes ... red, red" with a green one in frame — **localization is stronger
> than instance/colour binding**.

The failure is invisible: two cubes are both `pickable`, so every downstream gate
passes and the arm confidently picks the wrong one. Nothing in this stack can
detect that.

So use localization and let deterministic code do the binding:

```text
one capture
    ↓
Snapshot (entity list)  +  annotate_for_pointing(frame, entities)
                              grid + a drawn circle and id per entity
    ↓
Qwen: "point at the referent"  →  point_2d
    ↓
adapter: scale coordinates, then nearest entity by pixel distance
    ↓
  unique match within threshold  →  obj_3   →  mt4_pick("obj_3")
  no match / two within threshold →  STOP, naming the ambiguity
```

[`annotate_for_pointing`](../mt4_vision/preview.py#L483) already draws the
numbered grid and each entity's id at its pixel; its docstring says the grid
exists so *"a model reading coordinates off drawn, numbered gridlines is
self-correcting about the coordinate space"*. This is the intended consumer.

This design is also **the cube-agnostic one**, which is why it survives r2
unchanged: pointing works identically on a cube, a pen and a stapler, whereas
list-reading leans on the label quality that only cubes have (audit row 7).

The final action still carries an id. Pixels are how the referent is *derived*;
the id is what is *executed*. The first draft framed these as alternatives; they
are different layers.

**Binding threshold.** Derive it from `PICK_CLEARANCE_MM` (45mm) through the
calibration rather than hard-coding pixels — two objects closer than the finger
clearance are already not independently pickable, so an ambiguous point among
them is a genuine ambiguity. Pixel-per-mm varies across this oblique frame, so
the conversion must be positional.

---

## Coordinate Space Is a Hard Requirement **[r1]**

**This build answers in 0–1000 normalized coordinates, not pixels.** Verified
against the desk camera at 1280x720: a box reported at `x=807` sits at 1033px,
and asking for the whole desk returns exactly `[0, 433, 1000, 1000]`.

The first draft emitted `"pixel": {"x": 511, "y": 326}` and validated with
"pixel coordinates are within the frame". On a 1280x720 frame that check
**passes for every normalized coordinate** while pointing at the wrong place. It
lands on the desk, so segmentation finds *something* and registers a confident
`obj_N` for the wrong object — the most dangerous defect in the original
proposal, because it looks like success at every layer.

Required handling:

1. Name the space in the schema (`point_2d`, 0–1000), never "pixel".
2. Scale in the adapter — the `--coords norm` conversion `ask_qwen.py` already
   applies by default.
3. Re-check the convention rather than trusting it. `parse_regions` returns
   coordinates exactly as written and guesses nothing; `Region.in_bounds` is the
   assertion. A whole answer failing bounds is a convention mismatch, not a bad
   detection.
4. Reject a scaled point whose segmented mask does not contain it. A hint beside
   the object is not recoverable by segmentation, and refusing is correct.

Do not carry the 0–1000 assumption to another model build. Re-run `/preset` on
the real desk after any service change — the int4 quality spot-check on record
was four prompts on one synthetic frame.

---

## Observation Contract **[r1]**

Robot state is required in every path, not optional. Each observation carries:

| Field | Why |
|---|---|
| The annotated frame | the pointing target; one capture, see below |
| Entity list from that same capture | ids drawn on the frame must match the list |
| **Held object** | whether the gripper has something, and what |
| **Task progress** | what was asked and what has been executed |
| Gripper state | from `mt4_status` |

Without held state a policy will emit `PICK` while holding, or `PLACE` with an
empty gripper — neither of which any current gate refuses.

**One capture, not two.** `mt4_scene` and `mt4_camera_view` capture
independently ([server.py:296](../mt4_mcp/server.py#L296),
[server.py:375](../mt4_mcp/server.py#L375)), and `mt4_camera_view` draws the
*previous* snapshot's pixels onto a *new* frame. Harmless for a static desk;
not harmless for point-and-bind, which compares the model's point against pixel
positions from a different frame. Add one tool returning the annotated frame and
the snapshot from a single capture, under one token.

---

## Repository Prerequisites (Phase 0) **[r1]**

Gaps in the current MCP server, not in the policy design. All small; the closed
loop is unsound without them.

**1. The MCP server never retreats for the camera.** `retreat_for_camera` exists
at [pickplace.py:195](../mt4_vision/pickplace.py#L195) but only
[shuffle.py:208](../mt4_vision/shuffle.py#L208) calls it. In a
one-action-per-observation loop, observation *N+1* is taken with the arm
wherever `pick_at` left it — over the pick point, in frame, casting a shadow.
That is the condition the known blind-zone and occlusion problems live in, and
it hits a general segmentation enumerator harder than it hits HSV.

**2. `held_cube_px` is never passed.** `_build_snapshot` calls
`capture_scene(calib, frame)` with no held-cube hint
([server.py:297](../mt4_mcp/server.py#L297)), although
[scene.py:327-331](../mt4_vision/scene.py#L327-L331) supports it. A cube in the
jaws can reappear as a pickable `cube_N`. Note this hint is itself cube-specific
(audit row 8's cousin) — the general version is "mask the gripper region".

**3. `obj_N` disappears when picked.** `relocate` is deliberately strict; a
lifted object fails re-acquisition and is dropped
([server.py:305-309](../mt4_mcp/server.py#L305-L309)). "Pick the pen, put it on
marker 4" loses the pen's identity between steps. The held-object field carries
it across.

**4. `mt4_place` does not check that anything is held.** It validates the
destination and executes.

---

## Referent Resolution **[r2 — role changed]**

The first draft, and r1, put a deterministic referent resolver first: label,
colour, and marker-number matching over the snapshot, with the VLM as fallback.
The reasoning was that HSV and ArUco supply ground-truth labels, so matching
"the red cube" to label `"red cube"` is a string match, not perception.

**That reasoning is exactly a cube privilege** (audit row 7). For a pen or a
stapler the label comes from a VLM or DINO and carries no such guarantee, so the
resolver's advantage evaporates on precisely the objects the objective cares
about.

So the role changes:

- **Point-and-bind is the primary path for everything**, cube included. It is
  the only resolution mechanism that behaves identically across object classes.
- **The deterministic resolver becomes a benchmark control**, not a production
  shortcut. Run it on the cube subset to establish the accuracy ceiling that
  free ground-truth labels buy, then measure how far point-and-bind falls below
  that ceiling. The gap is a direct readout of the VLM's grounding quality,
  isolated from every other error source.
- Keep it in the loop only for referents that are *structurally* unambiguous
  and object-class-independent — "marker 4" resolves from the printed ArUco id
  and always will. That is a property of markers, not of cubes.

Spatial relations ("the leftmost one", "nearest the base") can be computed
deterministically from the snapshot's robot-frame x/y for any object class, so
they belong in the resolver too — they are general, not cube-specific.

---

## When the Enumerator Misses Something

Fallback registration by pixel hint, unchanged in mechanism and demoted in role:

```text
annotated view (grid + existing ids)
        ↓
Qwen returns point_2d on the object, 0-1000
        ↓
adapter scales to frame pixels, bounds-checks, mask-checks
        ↓
mt4_locate_at_pixel(px, py, label, view)
        ↓
segmentation → centre, axis, extent, inferred height, robot XY
        ↓
new obj_N → mt4_pick(obj_N)
```

```json
{
  "action": "LOCATE_AT_PIXEL",
  "view": "v12",
  "label": "stapler",
  "point_2d": [399, 453],
  "space": "0-1000",
  "confidence": 0.88
}
```

The point is only a **hint that lands on the object**; `locate` recovers the
real centre, axis, size and position. Drawn ids do the other half — they stop
the model pointing at something already in the table.

Track how often this fires. **A high rate is the metric that says the enumerator
is not yet general** — it is the cube-agnosticism gap made visible at runtime.

---

## One Action Per Observation

```text
observe (single capture: frame + snapshot + held state)
    ↓
select one action
    ↓
execute through MT4
    ↓
retreat for camera, capture again
    ↓
select the next action
```

*"Pick up the stapler and put it on marker 4."* — step 1 `PICK_ENTITY obj_3`;
step 2, after retreat and re-observe with `held: obj_3 (stapler)`,
`PLACE_ENTITY marker_4`; step 3 `DONE`.

Entity ids belong to the snapshot that produced them. Ids are assigned by
spatial order ([entities.py:280](../mt4_vision/entities.py#L280)) and are not
stable across snapshots; nothing carries an id forward except the held-object
field.

---

## Confidence and Ambiguity Gating **[r1]**

The first draft gated on a model-emitted `confidence` scalar. Nothing supports
that number being calibrated, and the service returns text only — no logprobs.
Three gates that do work, cheapest first:

1. **Geometric ambiguity** (free). Two entities within the binding threshold of
   the returned point → `STOP`, naming both. The primary gate, and fully
   object-class-agnostic.
2. **Sampled self-consistency.** k=3 at temperature, require agreement on the
   bound id. A policy call is ~0.85 s prefill + ~1 s decode for a short JSON
   reply at 45 tok/s, so k=3 costs a few seconds against a pick that takes
   longer anyway.
3. **Logprobs.** A small addition to
   [services/qwen3_vl/server.py](../services/qwen3_vl/server.py) returning
   per-token logprobs, or scoring candidate ids directly, gives a calibrated
   distribution over entities — the deferred pointer head with no training.

Keep `confidence` in the schema as a log field, not a gate.

---

## Latency and Token Budget **[r1]**

Measured ([QWEN3-VL.md § Decode throughput](QWEN3-VL.md)):

| Quantity | Value |
|---|---|
| Prefill, one 1280x720 frame (912 prompt tokens) | 0.85 s |
| Decode | 45 tok/s |
| Static KV cache window (`QWEN_VL_CACHE_LEN`) | 1664 |
| Multi-image mode — falls back to dynamic cache | 8.4 tok/s |

- **Never send two frames.** Multi-image and video overrun the static cache and
  drop to eager decode, ~5x slower. One capture per decision.
- **Trim the entity table.** `reason` fields are prose sentences (*"blob is
  312px2, under the 400px2 pick floor -- glare or an arm-paint fleck, not a
  cube"*). A general enumerator will make the table longer than HSV's does, so
  this gets tighter, not looser: send `reason` only for plausible targets, or
  raise the knob deliberately.
- A short JSON action is ~40 tokens → ~1.7 s per decision. Fine against a
  multi-second pick.
- **If Qwen becomes the enumerator** (detection option b), add 150–350 tokens
  and ~5–9 s per observation, and check the total against the 1664 window.
- **Qwen and Grounding DINO do not co-run** on the 8GB card. Comparing grounding
  paths means switching services with
  [services/switch_service.sh](../services/switch_service.sh).

---

## Benchmark Design **[r2]**

Cubes are the control, not the test. A cube-only benchmark measures plumbing and
flatters the system: uniform colour, known 20mm height, square, 90°-periodic,
sitting on marked slots, with automatic ground truth. Every one of those is an
advantage the target objects do not have.

**Stratify the object set:**

| Stratum | Purpose | Ground truth |
|---|---|---|
| Cubes | control; regression; the `h ≈ w` validation | free (HSV + ArUco) |
| Thin flat (pen, key, eraser, screwdriver) | the current de facto scope | hand-labelled |
| Compact non-cube (stapler, tape roll, dice) | tests `is_compact` and the grip path | hand-labelled |
| Adversarial (flat sheet, tall narrow) | the known `h ≈ w` failure directions | hand-labelled |

**The headline metric is the generality gap:**

```text
generality_gap = accuracy(cube stratum) - accuracy(non-cube strata)
```

measured separately for each stage — detection recall, referent binding, pick
success. A gap near zero is the objective. A large gap says the flow is still
special-cased, and *which* stage's gap is large says where.

Report these, and do not average them together:

| Metric | Why |
|---|---|
| **Wrong-entity rate** | the silent failure; no downstream gate catches it |
| **Refusal rate** | the safe failure; costs a retry, not a mis-pick |
| **Enumeration recall** | fraction of real objects present in the snapshot at all |
| **`LOCATE_AT_PIXEL` fallback rate** | how often the enumerator missed |
| Point-binding distance distribution | is the threshold right |
| XY error vs measured object height | the `h ≈ w` assumption, quantified |
| Coordinate-convention violations | should be zero after scaling |
| Task completion, prompt sensitivity | end to end |

**The decision gate is not "can Qwen emit JSON".** That is answered — measured
greedy, 3 runs each: `identify all objects` → 0 boxes (prose); *"reply in JSON"*
→ 0 boxes (valid JSON, wrong shape); an explicit `bbox_2d` schema plus "no
prose, no markdown" → **10 boxes, 3/3**. Format compliance is a solved prompting
problem. The gate is selection accuracy and the generality gap.

---

## Validation Before Execution

Every action is validated in the adapter before it reaches the MCP layer:

- the action is in the vocabulary
- the bound id exists in the snapshot that produced the frame
- entity capability matches the action
- the action is consistent with held state
- the view token matches the current frame
- coordinates are in the declared space and inside the frame after scaling
- the point binds to exactly one entity within threshold
- no target is silently substituted

The last is the invariant the rest serve. Picking a different object than the
one asked for is worse than doing nothing.

---

## Safety Boundary

The policy is advisory. It must not bypass entity `pickable`/`placeable` flags,
`reason` fields, snapshot lookup, fresh-frame re-acquisition, object
measurement, image-to-robot calibration, J1 keep-out, reachability limits,
stack/obstacle routing, wrist soft limits, safe travel and approach heights, or
deterministic gripper sequencing.

Raw tools — `mt4_pick_at`, `mt4_place_at`, `mt4_transfer` — remain for probing
and calibration, never as policy output.

Capability masks from a snapshot are advisory in the same sense: `pickable` was
true *when the frame was taken*. The gate that holds is `_reacquire` on a fresh
frame ([server.py:535](../mt4_mcp/server.py#L535)). Under detection option (b)
this matters more, not less: if the VLM is in the perception loop, the
deterministic gates are the only thing standing between a hallucinated object
and a motion command.

---

## Optional Atomic Entity Transfer

The repository has a raw-coordinate `mt4_transfer(...)` and a deterministic
`motion.transfer(...)` ([motion.py:683](../mt4_vision/motion.py#L683)), but no
entity-aware transfer at the MCP layer:

```python
mt4_transfer_entities(source_entity_id, destination_entity_id, camera=1)
```

Look up and validate both, re-acquire the source, validate the destination,
convert both to `Grasp`, call `motion.transfer(...)`.

```json
{
  "action": "TRANSFER_ENTITY",
  "source_entity_id": "obj_1",
  "destination_entity_id": "marker_4",
  "confidence": 0.91
}
```

This also sidesteps prerequisite 3 for the common two-step case: the object never
has to survive a snapshot boundary while held — which matters more for a
template-matched object than for a cube, since the object is the one that gets
dropped from the registry.

---

## Repository Mapping by Action

### `PICK_ENTITY` / `PLACE_ENTITY`

```text
Qwen point_2d → scaled → bound to entity_id
    ↓
mt4_pick(entity_id)                    mt4_place(entity_id)
    ↓                                      ↓
_lookup → _reacquire → as_grasp        _lookup → destination validation
    ↓                                      ↓
motion.pick_at(...)                    square_place / as_grasp → motion.place_at(...)
```

Destinations are fixed calibrated positions, so `_reacquire` returns
`square_place(x, y)` with no fresh capture
([server.py:553-561](../mt4_mcp/server.py#L553-L561)).

### `LOCATE_AT_PIXEL`

```text
point_2d (0-1000) → scaled → bounds- and mask-checked
    ↓
mt4_locate_at_pixel(...) → locate.measure(...) → object_entity(...) → obj_N
```

### `DONE` / `STOP`

No physical action. `mt4_stop()` cancels motion already in progress and is
neither of these.

---

## Build Order **[r2]**

### Phase 0 — hardware gate: DONE, and it failed

Run 2026-08-02; see Measured Results above. Mean 18.6mm error against a ~10mm
tolerance, caused by `_assumed_height_mm` measuring inflation across an
arbitrary axis on compact objects. **Nothing that commands the arm may switch to
the generic path until this is fixed.**

Still outstanding from this phase: the XY-error-versus-known-height sweep, using
the rig [calibrate_camera_nadir.py](../calibrate_camera_nadir.py) already
provides (grip a cube, sweep known heights). That needs the arm and was not run.

### Phase 0a — fix the height estimator

Two stacked causes, both needing hardware verification because both move where
the arm goes for every pointed-at object:

1. **Axis dependence.** Replace the across-the-long-axis inflation measurement
   with a radial/tangential one, which needs no axis and so behaves the same on
   a cube as on a pen.
2. **Shadow in the silhouette.** `L_WEIGHT = 0.35` is not enough on this desk;
   tangential extents of 28-48mm for 20mm cubes say the cast shadow is inside
   the mask.

Re-run the Phase 0 gate after each. The gate passes when cube error is within a
few mm, not merely better.

### Phase 0b — repository prerequisites

Retreat for the camera before snapshots; mask the gripper region (generalizing
`held_cube_px`); add held-object state and a held check to `mt4_place`; one tool
returning frame + snapshot from a single capture. Unify the three pickability
ladders into one predicate.

### Phase 1 — detection parity: BUILT, not wired

[mt4_vision/discover.py](../mt4_vision/discover.py) implements option (a) and
[tests/test_discover.py](../tests/test_discover.py) covers it (16 tests, plus
309 existing still green). Recall on the live frame is 9/9 cubes against
`mt4_scene`'s 6.

**Deliberately not wired into `build_snapshot` or `mt4_scene`.** Doing so would
put objects carrying 9-29mm position error into the entity list, where
`mt4_pick` would act on them. It stays a standalone module until Phase 0a
passes. Remaining work once it does:

- decide HSV/discover precedence for the same physical cube, via
  `is_own_colour_blob` (audit row 8)
- a width floor or a calibrated jaw span, so 4mm slivers are refused
- recall per stratum, not just on cubes

### Phase 2 — Qwen point-and-bind

Annotated frame + instruction → `point_2d` → scale → nearest-entity binding →
`PICK_ENTITY` / `PLACE_ENTITY` / `DONE` / `STOP`, with `LOCATE_AT_PIXEL` as the
enumerator-miss fallback. Report wrong-entity and refusal rates **per stratum**,
and the generality gap.

### Phase 3 — logprob scoring

If the ambiguity gate refuses too often, or the wrong-entity rate is material,
add logprobs and score candidate ids. A calibrated pointer distribution, no
training.

### Phase 4 — learned policy, only if 0–3 measurably fail

Deferred, for reasons worth stating rather than rediscovering:

- Under the cube-agnostic objective the head would have to learn from labels
  that are only trustworthy on cubes — so it would learn the cube shortcut and
  report a small validation error while doing the opposite of the objective.
  A stratified benchmark is what catches that; an averaged one would not.
- The residual failure is instance/attribute binding, which lives **inside the
  frozen encoder**. A head on frozen features cannot repair a representation
  that does not reliably separate red from green.
- 30–50 episodes is ~100 decision points. That will not train a transformer over
  Qwen token features; it will memorize the benchmark.

If a learned component is genuinely necessary, LoRA on the VLM is the honest
step, not a head on frozen features. The pointer-head design is preserved below.

---

## Appendix: Learned Pointer-Head Design (deferred)

Kept for reference. Do not build before Phases 0–3 are measured.

Entity ids are local to a snapshot and must never be learned as fixed output
vocabulary. Represent the scene as a variable list and predict a pointer over it:

```text
Camera image + instruction        Current entity list       Robot state
            ↓                             ↓                      ↓
      Frozen Qwen3-VL               Entity encoder         State encoder
            ↓                             ↓                      ↓
   Task tokens [B, Q, d]        Entity tokens [B, N, d]    State token(s)
                          ↘            ↓            ↙
                            Policy transformer
                                     ↓
              action head + entity pointer head + point head + confidence
```

```text
action_logits:  [B, 5]     # PICK / PLACE / LOCATE / DONE / STOP
entity_logits:  [B, N]
point_xy:       [B, 2]
confidence:     [B, 1]
```

Capability masks: `PICK_ENTITY` over `pickable == true`, `PLACE_ENTITY` over
`placeable == true`. The selected index converts back to the current id before
calling the repository API.

Entity token features: kind, label embedding, pixel x/y, robot x/y, yaw, yaw
periodicity, **measured extent and inferred height**, pickable, placeable,
source, confidence, holds/on. Include `source` explicitly so the training set
can be audited for cube bias — and hold out non-cube strata, never a random
split. Do not use stored robot x/y as an execution target; the execution path
still re-acquires from a fresh frame.

Training records store the snapshot presented and the **index** selected within
it, never the id as a global class:

```json
{
  "instruction": "Pick up the stapler",
  "image": "frame_0032.png",
  "snapshot": "s18",
  "held": null,
  "entities": [
    {"id": "obj_1", "kind": "object", "label": "stapler", "source": "vlm", "pickable": true},
    {"id": "cube_2", "kind": "cube", "label": "red cube", "source": "hsv", "pickable": true},
    {"id": "marker_4", "kind": "marker", "label": "marker 4", "placeable": true}
  ],
  "target": {"action": "PICK_ENTITY", "entity_index": 0},
  "success": true
}
```

Losses, applied only when relevant to the target action:

```text
total_loss = cross_entropy(action_logits, target_action)
           + λ_entity     × masked_cross_entropy(entity_logits, target_index)
           + λ_point      × huber(point_xy, target_point)
           + λ_confidence × bce(confidence, success)
```

---

## Summary of Revisions

### r1 — forced by measured VLM behaviour

| First draft | Revised | Forced by |
|---|---|---|
| Model returns `entity_id` from a text list | Model points; code binds the id geometrically | measured colour/instance binding failure |
| `"pixel": {"x", "y"}`, validated as in-frame | `point_2d` in 0–1000, scaled, bounds- and mask-checked | measured 0–1000 output; the in-frame check cannot catch it |
| Gate: "can Qwen emit valid JSON?" | Gate: selection accuracy on a fixed benchmark | schema-naming already gives 10/10 boxes, 3/3 |
| `STOP` covers refusal and completion | `DONE` added | closed loop had no termination signal |
| Robot state optional, learned path only | Held object and task progress required everywhere | no gate refuses `PICK` while holding |
| `confidence` scalar as the gate | Geometric ambiguity → self-consistency → logprobs | no logprobs; self-reported scalar uncalibrated |
| Learned head on 30–50 episodes | Deferred to Phase 4 | ~100 decision points |
| — | Phase 0b server prerequisites | no camera retreat, no held mask, `obj_N` drops when picked |
| — | Single capture for frame + snapshot | two independent captures; overlay uses stale pixels |
| — | Latency and token budget constraints | 1664-token static cache; multi-image drops to 8.4 tok/s |

### r2 — forced by the cube-agnostic objective

| r1 | Revised | Reason |
|---|---|---|
| Cubes as the working object class | Cubes as a benchmark **control**; stratified object set; **generality gap** as headline metric | a cube-only benchmark measures plumbing, not capability |
| Deterministic resolver as Phase 1 primary | Demoted to a cube-only control that measures the accuracy ceiling | its advantage is free HSV/ArUco labels — a cube privilege |
| Detection asymmetry unexamined | **Phase 1 is detection parity**: class-agnostic desk enumerator from `_segment` | cubes auto-listed, objects only if pre-registered — a scene-completeness failure, not a referent failure |
| Height assumed to be the blocker | Height is **already handled generically** (`_parallax_gain`, `_unproject`, `_assumed_height_mm`); the cube homography is the legacy special case | reading the code; the general path predates this document |
| — | Phase 0 hardware gate: same cube through both geometry paths, plus XY-error-vs-height sweep | a cube satisfies `h ≈ w` exactly, so it validates the general path against an independent fit |
| — | Audit table of all eight cube special cases | makes "where is the flow still cube-shaped" a checklist rather than a judgement |
| — | One pickability predicate | three ladders disagree; `grasp_feasibility` never re-runs after registration |
| — | `LOCATE_AT_PIXEL` demoted to enumerator-miss fallback, with its rate as a metric | it is the extra step that makes the non-cube flow longer |
