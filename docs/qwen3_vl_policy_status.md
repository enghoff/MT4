# Qwen3-VL action policy — build status, 2026-08-02 … 08-03

Companion to `qwen3_vl_mt4_repository_mapped_policy.md`. That document is the
design; this one is the ledger. What exists, what was measured on hardware,
what is broken, and what has to be decided.

Nothing described here is committed. Branch `feature/qwen3-vl-harness`, working
tree:

```
 M CLAUDE.md
 M docs/qwen3_vl_mt4_repository_mapped_policy.md
 M mt4_vision/entities.py
 M mt4_vision/grasp.py
 M mt4_vision/locate.py
 M mt4_vision/preview.py
 M tests/test_entities.py
 M tests/test_locate.py
 M tests/test_preview.py
?? docs/qwen3_vl_policy_status.md
?? mt4_vision/discover.py
?? mt4_vision/instruct.py
?? run_instruction.py
?? tests/test_discover.py
?? tests/test_grasp.py
?? tests/test_instruct.py
```

Full suite last run: **387 passed** (2026-08-03).

`vision_calibration.json` is **not** version-controlled and was written to —
`grip_span_s_at_zero_mm` and `grip_span_s_per_mm`, see §2f.

> **Regression, 2026-08-03 07:22.** A `calibrate_*` run overwrote the file and
> dropped four measured values to `null`: `cam_xy_robot` (was `[518.1, -35.0]`),
> `cam_height_mm` (244.0), `grip_span_s_at_zero_mm` (212.3) and
> `grip_span_s_per_mm` (1.881). The jaw-width refusal gate of §2f fails **open**
> when unmeasured, so it is currently switched off; the nadir and height feed
> the parallax and height correction in `locate.measure`, which is a candidate
> for the mask instability in §2b. The previous file survives as
> `backups/vision_calibration_pre_recalibrate_20260803_072222.json`. Whichever
> script did this needs to merge rather than rewrite.

---

## 1. What was built

### `mt4_vision/instruct.py` — the policy layer (new, no tests)

Turns one English instruction plus one camera frame into one validated action.
Everything it emits is checked against the snapshot before it can reach a
motor.

| Piece | What it does |
|---|---|
| `Observation` | `frame`, `annotated`, `snapshot`, `calib`, `held`, `history`. One capture produces both the snapshot and the pointing overlay, so the pixels the model reasons over are the pixels that were measured. |
| `Action` | `kind`, `ok`, `reason`, `entity_id`, `label`, `point_px`, `model_entity_id`, the `dest_*` mirror of those for a `TRANSFER`'s destination, and `raw`. `ok` means *safe to act on*: every `STOP` is `ok=False`, `DONE` is `ok=True`. `model_entity_id` preserves what the model said when validation redirected it. |
| `observe()` | Capture → `entities.build_snapshot` → grid overlay, one frame. |
| `decide()` | One Qwen call. `ACTIONS = (TRANSFER, PICK_ENTITY, PLACE_ENTITY, LOCATE_AT_PIXEL, DONE, STOP)`. Dispatch only: every target is resolved by `_resolve_target`. |
| `_resolve_target()` | Resolves ONE target, or refuses with prose. `kind` selects the entire rule set — candidate entity kinds, which of the task's attributes may describe this target, whether a "not on a marker" exclusion applies. A `TRANSFER` calls it twice against the same snapshot, `PICK_ENTITY` then `PLACE_ENTITY`, so both halves answer to exactly the rules the two separate actions always did. See §2r. |
| `locate_target()` | A *separate* single-purpose grounding call for a noun the detector has no word for. Measured: asked to choose an action **and** ground an unknown noun in one call, the model does neither reliably. Asks for `bbox_2d` and returns a `Grounding`. |
| `to_frame_pixels()` / `alternate_reading()` | The single model-coordinate → pixel conversion, and the sanctioned way to second-guess it. See §2a. |
| `Grounding` / `measure_grounding()` | A located box plus its centre, and the measurement that prefers GrabCut from the box over the desk-deviation point path. See §2b. |
| `unmatched_nouns()` | Words in the instruction that name a target nothing in the snapshot could satisfy, filtered through a `_FILLER` stopword set. This is what triggers grounding. |
| `bind()` | Nearest entity to a model-supplied pixel, thresholded at `BIND_RADIUS_MM = PICK_CLEARANCE_MM` (45 mm) converted to pixels. Refuses on ambiguity rather than picking the closer of two. |
| `grasp_for()` | The motion-layer pose for an entity, taken **from the snapshot the decision was made against**. Markers and slots → `square_place`; everything else keeps its own measured angle. Replaced `reacquire()` on 2026-08-03 — the fresh-frame re-measure it did was re-measuring an unchanged scene, and the template match it used to do it with tracked an ArUco tag instead of the object. See §2r. |
| `instruction_attributes()` / `_matches_attributes()` | Guards against "the *red* cube" resolving to a blue one. Kind-generic words are excluded; the remaining attributes must **all** match. Abstains (rather than fails open) when the noun is unknown to the detector. |
| `build_prompt()` | Includes the progress history, the held object **by label only**, an explicit statement that ids were assigned just now and mean nothing in any earlier step, and a request for pixels read off the drawn grid. |

Constants: `COORD_SCALE = 1000.0`, `MAX_NEW_TOKENS = 220`, `BIND_RADIUS_MM = 45`.
(`MOVED_TOLERANCE_MM` went with `reacquire` — nothing re-measures now.)

### `run_instruction.py` — the end-to-end script (new)

```
python run_instruction.py --camera 1 "pick up the red cube and place it on marker 2"
python run_instruction.py --dry-run "pick up the stapler"
```

Per step: `retreat_for_camera` → `observe` → ground any unmatched noun into
`obj_N` → `decide` → `grasp_for` off that same snapshot →
`transfer` / `pick_at` / `place_at` → `retreat_for_camera`. Loops to
`--max-steps` (default 6) or until `DONE`/refusal. Moving something somewhere
is **one** step, and nothing is verified after a motion — see §2r.

**It owns its own snapshot for both the decision and the motion.** Deciding in
one snapshot and then calling the MCP server's `mt4_pick` would resolve the id
inside a *different* capture. Ids are assigned by spatial order so the two
usually agree, and "usually" is the wrong guarantee when the failure mode is
picking the wrong object silently.

### `mt4_vision/discover.py` — class-agnostic desk enumerator (new, **unwired**)

Proposes one grasp-point candidate per desk-deviating blob with no prior
knowledge of what anything is, and hands each to the ordinary `locate.measure`
path — same stability check, same plausibility band, same work-region gate.

- Constants: `MIN_BLOB_PX = 100`, `MAX_BLOB_PX = 40000`, `MERGE_MM = 12`,
  `EXCLUDE_DILATE_PX = 9`, `OVERSIZE_SCALE = 8`, `OVERSIZE_KERNEL_PX = 61`,
  `OVERSIZE_WIDTH_KERNELS = 1.25`.
- `oversize_mask()` uses a **per-pixel** width test (morphological opening with
  a disk). Per-component was tried and deleted two real cubes: at coarse scale
  a cube 100 px from the arm base merges into the arm.
- Real limit is **scale, not colour**. `BG_MEDIAN_KERNEL_PX = 61` means
  anything wider than roughly 30 mm at this mount survives its own background
  estimate and disappears. A pen, key or cube is found; a book, phone or
  mousepad is not, on any desk. `MAX_BLOB_PX` enforces that honestly instead of
  failing silently.
- On a live frame it found **all 9 cubes** where `mt4_scene` reported 6.

It is deliberately not wired into `build_snapshot`. See §4.

### `tests/test_discover.py` — 16 tests, all passing

Includes the uniform-desk robustness tests: a 90-grey-level gradient plus grain
must not change which objects are found or where they measure (within 2 mm),
and a bare textured desk must propose nothing.

### `mt4_vision/locate.py` — one extraction, behaviour identical

`desk_deviation()` is now a shared public helper so `_segment` and `discover`
cannot drift apart:

```python
def desk_deviation(crop, kernel=BG_MEDIAN_KERNEL_PX):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    desk = cv2.medianBlur(lab, int(kernel) | 1)
    scale = np.array([L_WEIGHT, 1.0, 1.0], np.float32)
    return np.linalg.norm((lab.astype(np.float32) - desk.astype(np.float32)) * scale, axis=2)
```

The background is estimated **locally**, per pixel, from a 61 px median. That
is why wood grain, illumination gradients and soft shadows are absorbed rather
than flagged — a single global reference colour was tried and let a Sharpie's
mask swallow half the desk.

### `docs/qwen3_vl_mt4_repository_mapped_policy.md` — two revisions

r1 folded in measured VLM behaviour; r2 reframed around the cube-agnostic
objective, added the 8-row cube-specific-surface audit, the
`generality_gap = accuracy(cubes) − accuracy(non-cubes)` benchmark, and build
phases 0–4.

**This document is now partly stale.** It argues point-and-bind should be
primary with the deterministic resolver demoted. Live measurement contradicted
that: name-and-verify first, with pointing as the fallback for unknown nouns,
is what `instruct.py` actually implements and what works. It needs a third pass.

---

## 2. What was measured on hardware

### Bugs found and fixed during live runs

| Symptom (observed live) | Root cause | Fix |
|---|---|---|
| "stapler" bound to `cube_8`, reported `ok=True` | No id-existence check before binding | Check the id exists in the snapshot first |
| Every cube matched "the **red** cube" | `_matches_attributes` used `any()` over `{red, cube}`; "cube" matched everything | Exclude kind-generic words, require `all()` |
| "put the **blue** cube on **marker 3**" always refused | Attributes `{blue, marker3}` demanded one entity satisfy both | Filter attributes to the kind being resolved |
| JSON truncated mid-object | `max_new_tokens=120` overrun | Raised to 220 |
| Cube numbering changed mid-run; a dead id fed forward | `held` was passed as `"cube_3 (red cube)"` | Pass the **label only**; prompt states ids are per-snapshot |
| Task completed successfully **three times** | Loop was memoryless | Added `Observation.history`; `DONE` now fires |
| Picked a blue cube for "stapler", then falsely reported `DONE` | Attribute check abstained on unknown nouns | `unmatched_nouns` + a dedicated grounding call before the decision |
| Gripped the stapler, then named `obj_1` as the **place destination** | The held object stayed in the registry and was still listed as an entity | `objects.pop(action.entity_id, None)` on a successful pick |
| "could not segment an object at (700, 687)" for a stapler in plain view | **Coordinate space read backwards** — see §2a | `to_frame_pixels`, and grounding now asks for `bbox_2d` |
| Step 2 re-grounded the held stapler onto the gripper, registering a phantom at robot (−50, −97), r = 109 mm — inside the J1 keep-out | The noun that started the task stayed "unmatched" once the object left the desk, so every later step searched for it again | `unmatched_nouns(..., held=held)` treats the held label as known vocabulary |
| "place it on **marker 0**" moved a clamp onto marker 3, then onto marker 1, then reported `DONE` — on a desk with no marker 0 — see §2s | Every marker-number check derived its requirement by walking the *snapshot's* markers, so a number the desk lacks produced **no** requirement | `named_place_targets()` reads the number from the instruction; `missing_place_target_reason()` refuses before any binding |
| A reply with `entity_id: null` picked whatever it pointed at, unchecked — see §2s | `_matches_attributes` ran against the id the model *wrote*, and every branch that settles on the point instead skipped it | One gate on the entity finally chosen, in `_resolve_target` |

The last one is the third variant of the same underlying mistake — the model
naming the held object instead of the destination (`cube_3`, then `obj_1`).
Validation caught it each time, which is the system working, but the pattern
says the prompt still invites it.

### 2a. The coordinate space was read backwards — fixed

Reported symptom:

```
python run_instruction.py --camera 1 "pick up the stapler and place it on marker 0"
found a stapler but could not measure it: could not segment an object at (700, 687)
```

**Qwen was not wrong. It had located the stapler to within 4 px.** The reply
`[700, 700]` was in a 0–1000 normalized space and meant pixel (896, 504). The
code read it as pixel (700, 700) — 280 px away, on bare desk near the bottom
edge, where segmentation correctly found nothing.

Measured on the live rig, four objects × two prompt forms (`point_2d`,
`bbox_2d`) × two frame variants (raw, grid-annotated):

| Reading | Error vs. known pixel positions |
|---|---|
| 0–1000 normalized | **1–10 px** |
| absolute pixels | 206–285 px |

**8 replies out of 8 normalized.** Asking for pixels does not change it — the
old prompt said "coordinates are pixels of this 1280x720 image; the grid drawn
on it is labelled in pixels" and the model answered normalized anyway. The grid
overlay makes no measurable difference either: box replies on the raw and
annotated frames differed by 2–4 units.

**Why the old heuristic could not have worked.** It was "use the numbers as
pixels if they land in frame, rescale only if they don't". At 1280×720 a
normalized coordinate is *almost always* in frame: x ≤ 1000 < 1280 always, and
y ≤ 720 across the lower 72 % of the range. So the test fires the wrong way on
nearly every reply. This run failed loudly only by luck. The quieter version of
the same bug lands on a *different object* and measures it confidently, which
is precisely the silent wrong-target failure the entity layer exists to stop.

**The fix.** `instruct.to_frame_pixels` is now the single conversion, and it
defaults to normalized. The only thing that can rule normalized out is a
coordinate above `COORD_SCALE`, since no normalized coordinate exceeds 1000.
That leaves a blind spot — a true pixel answer whose coordinates all happen to
fall under 1000 — so `alternate_reading` exposes the other reading and
`measure_grounding` retries with it rather than reporting a failure. The retry
still has to survive segmentation, the two-window stability check and the
work-region gate, so a wrong reading cannot earn a target by being tried twice.

Note the module docstring had said "Coordinates are 0-1000, not pixels" all
along, while `_to_pixels` beneath it was a no-op returning raw pixels with a
docstring arguing the opposite. Two contradicting comments, and the code
followed the wrong one.

### 2b. Grounding now asks for a box, not a point

Asked for `bbox_2d` on the same frame, the boxes land tightly on their objects
and unlock the GrabCut path. Measured:

| Path | Objects segmented (of 4) |
|---|---|
| `measure` from the point (desk deviation) | **1** |
| `measure_grabcut` from the box | **4** |

Where both succeed, the box result also agrees better with the reference. The
box-derived positions sit **6.3–12.4 mm** (mean 9.1 mm) from where the HSV cube
detector puts the same three cubes — against the 18.6 mm mean / 29.1 mm max the
point path produced in the Phase 0 gate.

The stapler is the clearest case. From the box: 61.6 × 17.8 mm at robot
(165.4, 80.7). From the bare point: 73.5 × **6.6 mm** at (150.6, 84.0) — a
6.6 mm short axis is not a stapler, it is a sliver of mask, and it puts the
grasp 15 mm away.

A box buys three things a point cannot:

1. **GrabCut**, which finds low-contrast silhouettes desk-deviation misses.
2. **A bound on the mask**, which otherwise floods into shadow and neighbours.
3. **An extent to sanity-check.** A point has no size, so nothing about it can
   be checked before the arm moves.

There is a fourth, which is how the bug was found: a normalized box often
*overflows the frame* when misread as pixels (the stapler's y2 = 827 > 720),
and that overflow is the only self-evidence a reply carries about its own
coordinate space. A point carries none.

### 2c. Live verification, 2026-08-02

The command that started this, re-run on hardware after the fixes:

```
python run_instruction.py --camera 1 "pick up the stapler and place it on marker 0"

[1] cubes=2 blockers=0 free_markers=2 occupied=2 unknown=1 free_slots=15 ...
    registered obj_1: stapler at (169, 85) 68x16mm
    PICK_ENTITY  obj_1 (stapler)
    -> picked, holding the stapler
[2] cubes=2 blockers=0 free_markers=3 occupied=2 unknown=0 free_slots=15 ...
    PLACE_ENTITY  marker_0 (marker 0 (free))
    -> placed on marker_0
[3] ...
    DONE
```

Confirmed by camera: the stapler sits on marker 0's paper, and marker 0 now
reports `unknown` because the stapler's body covers the ArUco tag — which is
the correct reading of "something resting on the paper", not a fault.

**One real finding from the run.** "Place on X" puts the *grasp point* on X,
not the object's centre. For a cube those are the same point. For a stapler
they are not: the pick-time mask was 68 × 16 mm (part of the body), while the
same stapler measured on the desk afterwards is 150 × 73 mm. The grasp point
therefore sat well off centre, and the placed stapler's centroid landed ~87 mm
from the marker centre even though the placement itself was accurate.

This is not a bug in the placement, it is an unstated definition. For
cube-agnostic operation it needs deciding: does "place on the marker" mean
put the grasp point there, or centre the object there? Centring needs the
offset between the grasp point and the object centroid, which is measurable
(both are known at pick time) but is not currently carried through the move.

### 2d. A GrabCut-found object could never be re-acquired — fixed

```
PICK_ENTITY obj_1 (stapler)
-> refused after re-acquiring: obj_1 (stapler) could not be re-acquired in a
   fresh frame -- it moved, or the arm is over it
```

The stapler had not moved. Splitting `relocate` into its two halves showed
why: the template matched at **0.993 on the identical pixel**, and then the
function threw that away and finished with plain `measure` — the
desk-deviation path that cannot segment this stapler at all.

So any object that only GrabCut could find in the first place was permanently
un-re-acquirable: registration succeeded, and every attempt to act on it was
refused with a message blaming the desk.

**Fix.** `relocate` now re-measures through the matched template's own bounds.
That box is a tight, true outline of the object in this frame — it is the crop
the object was registered from, found again — so it seeds GrabCut exactly the
way a detector box does. `relocate_detail` returns which of the four failures
happened, and `reacquire` quotes it, because "it moved" and "found it, could
not re-measure it" call for opposite responses from whoever reads the message.

### 2e. Motion verification wired in — and it immediately caught a lie

```
[1] registered obj_1: stapler at (135, -200) 150x75mm
    PICK_ENTITY obj_1 (stapler)   -> picked, holding the stapler
[2] PLACE_ENTITY marker_4         -> placed on marker_4
[3] DONE
```

The stapler had not moved a millimetre. Every line above was printed from the
command that was *sent*; nothing was ever observed. This is the gap flagged
since the first draft of this document, reproduced end to end.

`instruct.verify_pick` / `verify_place` now run after each motion, and history
records the outcome rather than the command. Cubes are checked by colour within
`VERIFY_ORIGIN_RADIUS_MM`, the same rule `motion.verify_pick_place` uses for
stacking; objects have no colour rule, so they are template-matched at the
origin — which is what makes it work on a stapler and not only on a cube.

The direction matters. *Still at the origin* is strong evidence of a failed
grasp. *Not at the origin* is weaker, because `relocate` also misses an object
the grasp merely rotated — but that error costs a spurious refusal, and a
spurious refusal is a retry, while the opposite error is a false `DONE`.

Immediately on re-run:

```
-> the pick did not take: the stapler is still at (136, -199), within 30mm of
   where the grasp aimed (template match 0.99) -- it was never lifted
```

### 2f. Why that grasp failed: the jaw-width gate was inert *and* absent

Two separate holes, both now closed.

**The model was uncalibrated.** `grip_span_s_at_zero_mm` and
`grip_span_s_per_mm` were `null`, so `grasp_feasibility` skipped its width
check entirely. Measured on the rig by parking the TCP over clear desk at table
height — so the jaws sit *in* the table plane and the homography converts pixel
separation to millimetres with no parallax correction — and photographing
S = 140…230:

| S | 140 | 150 | 160 | 170 | 180 | 190 | 200+ |
|---|---|---|---|---|---|---|---|
| span (mm) | 33 | 32 | 27 | 21 | 13 | 7 | ~0 |

Fitted over the region where the jaws are still apart:
`span_mm = (205.0 − S) / 1.797`, rms residual 1.9 mm. **Maximum opening at
`grip_open_s` = 140 is 36 mm.** The stapler measured 75 mm across the grasp; it
was never going to fit. Both coefficients are now written to
`vision_calibration.json`.

Writing them is safe by inspection: `grip_s_for_span_mm`, the only function
that would turn the model into a *grip value*, has **no callers**. The
coefficients feed nothing but the refusal gate, so grip force is unchanged.

**The gate did not exist on the path that matters.** `grasp_feasibility` is
called by the MCP server and the CLI. The policy loop acts on
`entities.object_entity`, which had no width test at all. Both now call one
shared `locate.jaw_span_block_reason`, so they cannot drift.

```
STOP obj_1 (stapler)
reason: obj_1 (stapler) cannot be picked up: it measures 71mm across the grasp
        and the jaws open to 36mm -- they would close beside it, not on it
```

Note the gate stays silent when the model is unmeasured. That is a deliberate
fail-open — inventing a jaw width would refuse real objects on a rig nobody has
measured — and it is exactly why measuring it mattered.

### 2g. A 28-prompt sweep over one static scene

One observation, 28 instructions, no motion, so every prompt saw an identical
world. The scene was chosen to be awkward: two red cubes neither of which can
be picked (one at r = 363 mm beyond reach, one at r = 130 mm inside the
keep-out), two pickable cubes, an occupied marker, a marker whose tag would not
decode, a too-wide stapler and a too-wide stone.

**What already worked:** all six ordinary pick/place prompts resolved to the
right entity. Every physical refusal quoted the real constraint. Absent objects
(screwdriver, coffee mug, banana) were refused with no substitution.

**Four defects found, all fixed:**

| Found | Fix |
|---|---|
| Abstract words reached the grounder. "somewhere", "clear", "tidy", "anything", "everything", "there", "not" all got a model call, and Qwen answers rather than declining. **"clear" was registered as a physical object at robot (24, −335) 83×50 mm, "there" at (−293, 6) 55×33 mm.** Others returned whole-frame boxes projecting to 315037×312990 mm, caught only by the plausibility band. | `_FILLER` widened from 60 to ~200 words. Erring long is safe: a real noun wrongly listed degrades to "not in the snapshot", while a filler word wrongly omitted invents an entity. |
| Attributes were harvested from the whole sentence, so a landmark became a requirement. "pick up the grey rock next to the blue cube" → `the task says ['blue','grey','rock'] but obj_5 is 'rock'`. | `noun_phrase()` brackets the target's own phrase. When the entity's label head is absent from the instruction it **abstains** rather than falling back to the sentence — the fallback is what manufactured the contradiction. |
| Multi-word targets registered one object per word: `obj_3` "grey" and `obj_4` "rock" at the identical (109, 179). Two ids for one stone splits the attribute check across entities that can never both match. | `register_object()` merges within `DUPLICATE_MM` = 12 mm and folds the new word into the label, so the entity answers to both. |
| A **question** moved the arm. "is there anything on the desk that is not a cube" returned `PICK_ENTITY cube_2` — the action set has no way to answer, so the model picked something. | `is_question()` guard: nothing moves for a question. Deterministic rather than prompted, because a guard cannot be talked out of it. |
| "pick up **slot 5**" returned `PICK_ENTITY cube_2` — a silent substitution of exactly the kind this layer exists to stop. | `wrong_kind_block_reason()`: refuse a pick whose chosen kind the instruction never names, when it does name a never-pickable kind. |
| "put the blue cube on marker 2" (occupied) picked first and would have discovered at the place step that the destination was never available, stranding a cube in the jaws. | `named_destination_block_reason()` pre-flights the named marker before the pick — exempting a destination occupied by the very thing being moved, so "put the green cube on marker 2" with it already there does not refuse itself. |

**Known and left alone.** "put it down" with two equally free markers refuses on
name/point disagreement. That is the ambiguity rule working as designed; the
alternative is guessing which marker the user meant.

### 2h. Live confirmation, everything wired

```
python run_instruction.py --camera 1 "pick up the blue cube and put it on marker 3"

[1] PICK_ENTITY cube_2 (blue cube)
    -> picked, holding the blue cube -- the origin is clear
[2] PLACE_ENTITY marker_3 (marker 3 (free))
    -> placed on marker_3 -- a blue cube is at (152, 157)
[3] DONE
```

`marker_3` is at (153.6, 156.9); the cube landed **2 mm** from its centre.
Confirmed by camera.

### 2i. A regression the campaign introduced, and why it slipped through

The kind guard from §2g broke the main non-cube case within the hour:

```
python run_instruction.py --camera 1 "pick up the stapler and place it on marker 4"
STOP  reason: the task asks for a marker, which is a place target and cannot be
      picked up, and obj_1 is a object the task never mentions
```

It compared the chosen entity's **kind** against the instruction. A registered
object's kind is the literal string `"object"`, which no real instruction ever
contains, and "marker 4" tripped the never-pickable trigger. So every
"pick up the *thing* and put it on marker N" was refused.

**Cubes hid it**, which is the part worth remembering: `"cube"` is both their
internal kind and a word people type, so every cube test passed. The one
combination that fails — a non-cube target plus a named marker destination —
was in neither the campaign (no marker in "pick up the stapler") nor the live
runs (all cubes).

Fixed to test the entity's **label** words instead: labels are what users name
things by, kinds are an internal taxonomy. Pinned by
`test_the_kind_guard_does_not_block_a_named_object_bound_for_a_marker`.

### 2j. The stapler's earlier success was a lucky mask, not a capability

Worth stating plainly, because the run history reads as a regression and is not
one:

| Run | Mask measured | Grasp width | Outcome |
|---|---|---|---|
| → marker 0 | 68 × 16 mm | 16 mm | **worked**, stapler moved |
| → marker 4 | 150 × 73 mm | 73 mm | jaws closed on air |
| → marker 4, now | 149 × 73 mm | 73 mm | refused by the width gate |

The first run segmented only the stapler's **metal rail** and aimed the grasp
there, across 16 mm — comfortably inside the 36 mm jaws. Later runs segmented
the whole stapler and aimed at its centroid, across 73 mm, which cannot be
gripped. Nothing changed about the arm; the segmentation was inconsistent, and
the inconsistency is what made the behaviour look random.

So the width gate is telling the truth about the grasp it is offered, and the
refusal is correct. But "the stapler cannot be picked up" is *not* true — the
arm demonstrably picked it up by the rail. What is missing is a stage that
chooses **where on an object to grasp**, rather than always taking the
silhouette centroid.

**That is the next real capability gap for cube-agnostic operation.** A cube
has one grasp; anything else has many, most of them bad. The measurements
needed are already in hand — `measure_grabcut` returns a mask, and the jaw span
is now known — so the missing piece is scanning the mask for the narrowest
spannable cross-section near the centre of mass instead of assuming one exists
at the centroid.

### 2k. Grasp-point selection — built, unit-tested, not yet validated on hardware

`mt4_vision/grasp.py`. An antipodal-grasp search over the object's footprint in
robot millimetres: rotate through every jaw angle, slide a **finger-width
window** along the jaw axis, and keep the narrowest window whose two contact
lines are flat enough to be facing surfaces — preferring one near the centre of
mass so the object does not pivot out as the jaws close.

Wired through `locate.plan_object_grasp` into `entities.object_entity`, so the
entity's `x`, `y` and `yaw_deg` **are** the planned grasp rather than the
silhouette centroid, and a new `grip_mm` field reports what the jaws will
actually close on. Clearance and the work-region test now run at the planned
point, because for anything elongated that is tens of millimetres from the
centroid.

**The bug worth recording** is one the tests caught before hardware could:
minimising width alone returns **corners**. A diagonal slice near the corner of
a 25 mm cube is 22 mm across — narrower than any face — so the search picked it
every time (yaw 140°, 12 mm off centre, on an object whose correct grasp is
25 mm dead centre). Closing there squeezes a wedge the jaws push out.
`CONTACT_FLATNESS_MM` requires both contact lines to be flat across the finger,
which is what makes a grasp antipodal rather than merely narrow.

**Known limit, stated plainly.** This is a *plan-view* method. It finds narrow
features that are narrow seen from above — a pen, a screwdriver shaft, a
saucepan handle, a mug handle. It cannot find a narrow feature that is only
reachable because it is *raised* above a wider base, which is what the
stapler's metal rail is. So the stapler stays refused, and that is honest for a
top-down grasp at table height; recovering the rail grasp needs height
information this stack does not have.

13 tests in `tests/test_grasp.py` on synthetic shapes with arithmetic answers:
cube at its centre, pen across its short axis at any rotation, saucepan handle
found away from the centroid, corner never returned, thin neck not reported as
the grip when the fingers straddle it onto the blobs either side.

### 2l. Jaw span, re-measured over bare desk

The first pass parked the gripper over a spot where an ArUco marker lay on the
desk, and the detector's "largest bright run between dark pixels" was bounded
by the **white paper**, which is narrower than the gap between the finger
faces. Re-measured over bare wood, walking outward from the jaw axis until a
dark finger face is found on each side:

| S | 140 | 155 | 170 | 185 | 200 |
|---|---|---|---|---|---|
| span (mm) | 36.8 | 32.3 | 23.7 | 13.1 | 0 |

`span_mm = (212.3 − S) / 1.881`, rms 1.6 mm, **38.4 mm at `grip_open_s` = 140**.
Both coefficients written to `vision_calibration.json`.

Three separate detectors disagreed on this before it settled, all for the same
reason: the frame contains other dark and bright structure — marker paper, the
controller box, shadow — and any rule of the form "find the widest bright run"
finds one of those instead. Anchoring the search at the jaw axis and walking
outward is what fixed it. Worth remembering next time a pixel measurement of
something small looks plausible but wrong.

### 2m. Grasp planning on hardware — it works

```
python run_instruction.py --camera 1 "pick up the stone and place it on marker 3"

[1] registered obj_1: stone at (213, -72) 49x28mm
    PICK_ENTITY obj_1 (stone)
    -> picked, holding the stone -- the origin is clear
[2] PLACE_ENTITY marker_3 (marker 3 (free))
    -> placed on marker_3 -- it is no longer at its old position
[3] DONE
```

Confirmed by camera: the stone is on marker 3. This is the instruction that
opened this section refusing, on an object the centroid rule could not hold.

Two planner defects were found and fixed against live masks before it worked:

**It reached for a shadow.** GrabCut's silhouette of the stone included a spill
of shadow, and a 14 mm-wide sliver of shadow 22 mm off centre beat the stone's
own 28 mm body — so the plan landed on bare desk. Fixed with `MIN_BAND_SHARE`
(the finger band must hold at least 15 % of the material in the fullest band at
that angle) and by raising `CENTRE_PENALTY` from 0.25 to 0.6. Density is what
separates "a thin part of this thing" from "a thin thing next to it" without
knowing what either is.

**It reached for corners** — caught by unit tests, see §2k.

Planning against four real objects afterwards, with the plans drawn on the
frame: the stapler plan lands on its narrow front anvil (16.8 mm), the red cube
and the stone on their own bodies. All are places the jaws can actually close.

### 2n. The "segmentation instability" was the marker-paper exclusion — fixed

The measurements below looked like an unstable segmenter. The same stapler,
unmoved, across four consecutive measurements:

| Run | Outline reported | Mask? |
|---|---|---|
| a | 123.2 × 72.1 mm | none — fell through to `measure_box` |
| b | 149 × 73 mm | GrabCut |
| c | 19.4 × 16.9 mm | GrabCut, front anvil only |
| d | 128 × 67 mm | GrabCut |

**That diagnosis was wrong, and the instrumentation that produced it was the
reason.** `measure_with_box_fallback` is a ladder: GrabCut, then desk-deviation,
then the raw detector box. Each rung is tried silently, so a run that failed all
the way to the bottom reported a number in the same format as one that
succeeded, and four runs at different rungs read as one flaky segmenter.

Instrumented properly — GrabCut called directly so a failure raises — the truth
is the opposite of unstable. Qwen's box is **identical** across four calls on a
fixed frame, `(284, 351, 403, 467)`, and near-identical across four fresh frames
(x1 varies 282–284, nothing else moves). GrabCut then failed **8 times out of
8**, always with `GrabCut found no foreground in box`.

The cause is `marker_paper_mask`. It draws a geometric quad around each
calibrated marker position and forces it to sure-background, which is right when
the marker paper is bare and wrong the moment something is resting on it — and
the arm had placed the stapler on marker 0 in an earlier run. The exclusion
covered **53.8 % of the detector box**, including the stapler. With the
exclusion GrabCut failed; with it removed the same box measured 90.0 × 41.5 mm.

Also worth stating plainly, since it was the question that found this: **the
DINO path does nothing different.** `mt4_mcp/server.py:506` makes the same
`measure_with_box_fallback` call with the same arguments. DINO looked fine
because its objects were not sitting on markers.

**Fix, in `_segment_grabcut`:** the exclusion is now a prior inside the detector
box rather than a law. Outside the box, marker paper stays `GC_BGD`. Inside it,
the paper becomes `GC_PR_BGD`, so GrabCut's colour model can reclaim the pixels
that are not paper-coloured — the object on top — while still being steered off
the high-contrast tag that made the exclusion necessary in the first place. The
sure-foreground seed is re-stamped afterwards so it always survives. Verified:
the box that failed 8/8 now returns a mask.

One measurement still needs a second look. Post-fix the stapler comes out at
55.8 × 38.4 mm and the planner refuses it — "the narrowest grip anywhere on it
is 43 mm and the jaws give 35 mm". That is an honest refusal for a top-down
grasp, but the mask probably still includes shadow, so the true width may be
narrower. Shadow rejection in the GrabCut path is the remaining work here.

### 2o. Destinations were never drawn on the image the model sees

Found while re-running "pick up the green cube and place it on a non-marker
location". Four separate defects, all in the destination half of the loop.

**1. Place-holder nouns for a position reached the grounder.** `_FILLER` had
"somewhere" but not "location", "spot", "position", "space", "site", "square" or
"gap". So `unmatched_nouns` treated "location" as a thing to find, spent a model
call on it, got a whole-frame box back, projected it to **315037 × 312990 mm**,
and then refused the entire task with `the task names ['location'], which
nothing in this snapshot is` — on an instruction whose destination the snapshot
was offering fourteen candidates for. Fixed by adding the positional
place-holders, plus "unoccupied" and "vacant".

**2. A whole-frame box is a non-answer and was treated as a reading.** Asked to
locate something that is not there, this build sometimes returns the whole image
instead of the empty list the prompt asks for. Its centre is in frame, so
nothing rejected it until the projection did — and by then the complaint was
about the camera's horizon, three layers from the cause. `_grounding` now
rejects a box covering more than `MAX_BOX_FRAME_SHARE` = 55 % of the frame. The
largest thing this stack will measure is 200 mm, roughly a third of the frame's
width, so the threshold has room.

**3. Markers and slots had no pixel, so they were invisible.** This is the big
one. `Entity.pixel` was populated for cubes and objects only. Two consumers read
it and both silently dropped everything else: `annotate_for_pointing` skips any
entity without one, and `_entity_lines` appends "at image point (x, y)" only
when there is one. The decision prompt meanwhile asserts *"Each is circled in
the image with its id written beside it"* and asks for the target's centre in
pixels. **For every marker and every slot that sentence was false.** The model
was being asked to point at things it had never been shown.

The symptom was exact and repeatable. Four runs of the same instruction: named
`marker_0` — the first destination in the list — every time, and pointed at
(1255, 402), (1261, 403), (1258, 402). That is the far right edge of the frame,
nowhere near any destination, the same wrong answer each time.

Fixed with `entities._desk_pixel_projector`. Markers and slots are flat on the
desk, so the desk-plane homography is exact for them and none of the height
correction cubes need applies. Verified against the live frame: every projected
circle lands on its printed tag. After the fix the model's point moved from the
frame edge to (936, 420) — 91 px from `slot_5`, inside the desk, wrong but no
longer nonsense.

**4. "Non-marker" could not be expressed.** Both destination kinds reach the
model as "can be placed on" and nothing else distinguishes them. Two changes:
the prompt now says what each kind *is*, and `excluded_destination_kind` reads
the constraint out of the instruction so it can be enforced rather than hoped
for. When the task rules markers out, markers stop being candidates for the
point, a name/point disagreement where only the name is a marker resolves to the
point, and a destination that is still a marker is refused by kind.

**Measured after all four fixes.** The pick half is now solid — 5 runs, 5
successful picks, each verified by camera, `-0, +0 mm` between the snapshot
position and the executed grasp every time. A named marker destination completes
the whole task: "put the green cube on marker 2" ran pick → place → DONE with
both steps verified, landing at (164, −154) against a marker at (162, −150).

**A slot destination still does not work**, and it is now cleanly isolated as a
model-capability limit rather than a plumbing gap. "Move the green cube to an
open slot" picked correctly and then answered `PLACE_ENTITY cube_2` — the id of
the cube it had just picked up, which by then named a different cube — pointing
at (750, 405), the arm base. See §5 for the decision this raises.

**Also fixed, message quality only.** Three refusals named a true fact about the
point while saying nothing about what went wrong: a reply that named `marker_0`
was refused with "it does not name anything in the list", and a `PLACE_ENTITY`
naming a cube was refused for the point being 177 px from a slot. Both now say
what they mean. And `run_instruction.py` now prints the executed grasp and its
offset from the snapshot position, because without it a failed grasp gives no
way to tell a mis-measured target from a good one badly executed.

### 2p. The decision point was being rescaled out from under itself — fixed

Found by looking at the preview (§2q) and asking why the magenta point was
always off. It was off by a constant factor, which is the signature of a
coordinate-space error, not of a model guessing badly.

**The convention is a property of the prompt, not of the model.** The two
prompts in `instruct.py` differ in one respect that turns out to decide it:

| Call | Does the prompt show a pixel? | Answer space |
|---|---|---|
| `locate_target` (`bbox_2d`) | no — it deliberately says nothing | 0–1000 normalized, 8/8 |
| `decide` (`point_2d`) | **yes** — every entity's pixel is in the list | pixels, 5/5 |

`to_frame_pixels` applied the normalized rule to both. So a decision reply of
`[320, 473]` — the named entity's own pixel, copied out of the list — was
rescaled to (410, 341), **160 mm-scale pixels away**, onto bare desk.

Measured 2026-08-03, five consecutive decisions on one fixed frame: the reply
was the named entity's pixel to **0.4 px** every time, and the same five read as
normalized land **160 px** away. After the fix, 4/4 at 0.4 px.

Worth being precise about the cause, because it is nearly the opposite of the
2026-08-02 finding in §2a. Asking for pixels in words does nothing — that was
measured and is still true. **Showing a worked example does.** The entity list
only started printing "at image point (x, y)" for markers and slots today
(§2o), and the reply switched space as a result. The earlier 8/8-normalized
measurement was correct for the prompt that existed when it was taken.

**Fix:** `point_readings` returns both in-frame readings, pixels first, and
`decide` resolves them **against the snapshot** — a reading wins by landing on
something that is actually there. If the two readings bind to *different*
entities the reply does not identify one, and it is refused rather than settled
by whichever convention flatters the answer. `Action.alt_point_px` carries the
loser so the preview can draw it.

Not covered: `LOCATE_AT_PIXEL`, whose target is by definition *not* in the list,
so the model has no pixel to copy and must read the grid. Which space it answers
in there is unmeasured. It now takes the pixels-first reading, and a wrong guess
degrades to "could not measure it" rather than a silent mis-measurement — but it
should be measured, and offered the alternate reading as a retry the way
`measure_grounding` is.

### 2q. A preview window, on by default

`run_instruction.py` opens a live window showing the frame the model looked at
with its own answer drawn on it (`--no-preview` to suppress; suppressed
automatically when OpenCV has no GUI, since `LivePreview`'s PIL fallback would
spawn a system image viewer per update).

Deliberately the frame the model saw, never a fresh capture: every question
worth asking of a reply is a question about *that* frame, and re-capturing
changes the subject.

| Mark | What it answers |
|---|---|
| Solid yellow box + frame share | Is this grounding a real object or the whole image? |
| Dashed orange box / cross | Same numbers in the other coordinate space |
| Spring-green fill + window share | Did GrabCut take the object, a fragment, or everything? |
| Magenta ring | Where the model pointed |
| Green/red ring + labelled line | What the stack resolved it to, and the gap |
| Caption band | Instruction, action, full reason, word-wrapped |

It paid for itself twice within an hour of existing. §2p was found by looking at
it. And on the first live frame, the stapler's mask came back **95 × 62 mm
filling 16 % of its window** — visibly the top half of the stapler plus a leak
onto a cable, for an object that is roughly 150 × 40 mm. That is the shadow
problem at the end of §2n, no longer an inference from a plausible-looking
number.

### Pre-existing bugs surfaced but not fixed

- **Desk polygon is ~20 mm too tight.** Two real cubes fall outside it by
  17.1 mm and 0.2 mm and are dropped from the scene entirely.
- **One green cube is invisible to HSV** at current lighting.
- ~~**Jaw span is uncalibrated.**~~ Measured and written, see §2f.

### Phase 0 geometry gate — FAILED

Cross-validating the generic `locate.measure` path against the cube path gave
**18.6 mm mean, 29.1 mm max** disagreement, against a ~10 mm tolerance.

Root cause: `_assumed_height_mm` divides by `1 + gain * cos_radial`, where
`cos_radial` is measured across an arbitrary axis. For cubes it came out
0.03–0.58 depending on orientation — so the same cube estimates a different
height depending on how it happens to be rotated, and the parallax correction
moves the target accordingly. Camera geometry: nadir ≈ (518.1, −35.0) robot mm,
height 244 mm, parallax gain 1.4–2.0 mm of XY error per mm of object height.

**Not fixed.** Any change here moves where the arm goes for every pointed-at
object, so it needs hardware time and a re-run of the gate.

Important scope note: the gate compared the generic path against the cube path.
The policy loop currently uses the cube path for cubes and the generic path for
objects, so this is a *consistency* failure, not a blocker on either one
individually. Picking works to roughly ±10 mm today.

### `calibrate_table_edge.py` — attempted, fully reverted

"Only 0 clean columns (need >=15)" was diagnosed as a wood colour-model
failure: measured desk saturation 15–27 against a threshold of 40, hue 165–180
and 0–15 against a window of 8–30, matching 1.6 % of the desk. The saturation
fix gave an unstable 10–28 columns. The hue-wraparound fix gave 156 columns but
broke the wall test, putting the edge at +79 mm — GrabCut latching onto the
black ArUco pads. Reverted with `git checkout`; the file is unmodified.

### 2r. Motion verification removed; pick and place fused into one `TRANSFER`

The check added in §2e survived one day. On 2026-08-03 it reported this, with
the stapler visibly clamped in the jaws:

```
python run_instruction.py --camera 1 "pick up the stapler and place it on marker 3"

[1] registered obj_1: stapler at (-4, 258) 82x36mm
    PICK_ENTITY  obj_1 (stapler)
    -> (16, 241) yaw 156, +20,-17mm from the snapshot
    -> the pick did not take: the stapler is still at (32, 237), within 30mm of
       where the grasp aimed (template match 0.65) -- it was never lifted
```

**It had matched the ArUco marker.** `marker_1` measures (34.7, 233.7) — 4.2 mm
from the position reported as the stapler's, and 3.6 px from it in the frame.
`mt4_scene` on the same desk listed `marker_1` as *free*, tag decoding cleanly,
nothing resting on it.

The mechanism, measured on the live frame. `locate` stores an object's template
as a plain axis-aligned crop of the bounding box (`locate.py:626`), not as its
silhouette, so anything else inside the box is stored with it. The stapler was
lying against marker 1's paper, and for its 59 × 59 px crop:

| | share of crop area | share of crop's squared deviation from its mean |
|---|---|---|
| ArUco tag | 30 % | **87 %** |
| everything else | 70 % | 13 % |

The rest is bare desk at a standard deviation of **1.0 grey level**.
`TM_CCOEFF_NORMED` correlates mean-subtracted patches, so a flat region
contributes nothing either way and the tag is effectively the whole template.
After the stapler left, the crop still correlated with the tag at 0.65, over the
`RELOCATE_MIN_SCORE = 0.6` threshold and inside the ±80 px search window.
`relocate_detail` then re-segmented through the matched box — a crisp dark tag
on uniform desk being the easiest GrabCut there is — and returned the tag's
centroid as the stapler's position.

`relocate`'s own docstring already forbade the converse use ("None is not
evidence that the object left the table"). The mirror hazard is the one that
fired: a **match** is not evidence the object stayed, when the template carries
stationary background. Any object registered touching a marker, a cable or a
strip of tape inherits it.

**Decision: the loop no longer verifies anything after a motion.** There is no
sensor in the jaws (see `grasp.py`), and the only vision test available answers
wrongly in both directions — a false negative here, and a false positive
whenever a grasp merely rotates the object. A check that lies both ways is worse
than none, because it is acted on. Outcome lines now read
`(commanded, not checked)` and history records what was sent.

**What replaced the two-step flow.** Once nothing is verified between the pick
and the place, the park-capture-decide cycle in between answers no question:
the object is in the jaws, it has left the entity list, and where it should go
was already settled. So `TRANSFER` names both ends in one decision and
`motion.transfer` carries them out as one planned operation — the pick's lift
and the carry are the same leg, so the arm never stops between having the object
and being on its way. `run_instruction` then recalls the arm to camera park.

Everything that was ever a *pre*-motion check still runs, and now runs on both
ends before the gripper opens: reach, keep-out, ground, finger clearance, jaw
span, the desk polygon, camera framing, attribute agreement, the point/name
disagreement rule, and the destination-occupancy rule. A transfer onto an
occupied or undecoded marker is refused while the object is still on the desk,
which is strictly better than the old order — that refused after the grasp and
stranded the object in the jaws.

`decide()` is now dispatch only; `_resolve_target()` is its former tail lifted
out whole and parameterised by `kind`, so a transfer's two halves are resolved
by exactly the rules `PICK_ENTITY` and `PLACE_ENTITY` always used and cannot
drift from them. Gripper-state rules moved up to `decide`, because whether the
jaws may be full is a property of the action, not of one target.

**Live, 2026-08-03**, on the command that started this:

```
[1] cubes=3 blockers=2 free_markers=2 occupied=1 unknown=1 free_slots=14
    registered obj_1: stapler at (180, -80) 122x24mm
    TRANSFER  obj_1 (stapler)  -> marker_3 (marker 3 (free))
    -> pick (180, -80) yaw 172  ->  place (154, 157) squared
    -> moved the stapler onto marker 3 (free)  (commanded, not checked)
[2] registered obj_2: stapler at (146, 157) 91x26mm
    DONE
```

Confirmed by camera: the stapler is on marker 3's paper, and it re-measures 7 mm
from the marker centre. A second run moved it to `slot_4` and a dry run of
"put the stapler on marker 2" refused before any motion, quoting the undecoded
tag. One decision and one motion per move, against two decisions, two motions
and three captures before.

**Two things this does not fix.** The place still puts the *grasp point* on the
destination rather than the object's centroid (§2c), so an elongated object
lands off-centre by the grasp offset. And the mask remains unstable: the same
stapler measured 82 × 36, 122 × 24, 91 × 26, 67 × 36 and 59 × 44 mm across five
frames in one session (§2b in the ranked list below).

### 2s. Identity checks that were no-ops exactly when they mattered — fixed

Reported symptom, 2026-08-03. The desk's calibration carries markers 1, 2, 3 and
4, and no marker 0:

```
python run_instruction.py --camera 1 "pick up the clamp and place it on marker 0"

[1] registered obj_1: clamp at (48, 234) 153x25mm
    TRANSFER  obj_1 (clamp)  -> marker_3 (marker 3 (free))
    reason: clamp is on marker_3
    -> moved the clamp onto marker 3 (free)  (commanded, not checked)
[2] registered obj_2: clamp at (104, 163) 148x20mm
    TRANSFER  obj_2 (clamp)  -> marker_1 (marker 1 (free))
    reason: clamp is on marker_1 as per task
    -> moved the clamp onto marker 1 (free)  (commanded, not checked)
[3] DONE
    reason: task already completed
```

Three `ok=True` decisions, two real arm moves, and not one of them touched
anything the instruction named. This is the outcome the whole entity layer
exists to prevent, and it went through every gate.

**Two independent defects, both of the same shape: a check that abstains
precisely when it has something to say.**

**1. The requirement was derived from the snapshot, not from the instruction.**
`instruction_attributes` built its marker attribute by iterating the snapshot's
markers and asking whether the text mentioned each one:

```python
for e in snapshot.entities:
    if e.kind == "marker":
        num = e.id.split("_")[-1]
        if f"marker {num}" in whole:
            found.add(f"marker{num}")
```

With markers 1–4 in the snapshot and "marker 0" in the text, that loop matches
nothing and returns an **empty** attribute set. `_resolve_target` then computes
`verified = named is not None and attrs and _matches_attributes(...)`, which is
false for an empty `attrs`, falls through to the point, and accepts whatever the
point landed on. Every destination satisfies an empty requirement vacuously. So
the guard against putting the object on the wrong marker worked for every number
the desk has and switched itself off for every number it does not — which is the
only case where the model has no correct answer available and will invent one.
`named_destination_block_reason` had the identical structure and the identical
blind spot: it pre-flights whether the named marker is *occupied*, and a marker
that does not exist is not occupied.

Numbers are now read from the instruction by `named_place_targets()`
(`\b(marker|slot)s?\s*#?\s*(\d+)\b`), which is snapshot-independent by
construction, and `missing_place_target_reason()` refuses before any binding
runs, naming the numbers that do exist:

```
the task names marker 0, which is not on this desk -- the markers here are
1, 2, 3, 4. Nothing else may stand in for it, so the task cannot be carried
out as written
```

It reads existence from the **calibration**, not from this frame's decode:
`capture_scene` lists every calibrated marker whether or not its tag was
legible, so a marker in shadow — or under the very object being moved — is still
"on this desk" and never trips this. What trips it is a number nobody taped
down. The same check now also runs on `DONE`, because `DONE` is the one claim in
this layer nothing downstream re-examines: `run_instruction` returns success on
it without looking at the desk, and step 3 above was making it.

**2. The attribute check ran against the id the model wrote, never against the
entity chosen.** `_matches_attributes` was applied only to `named`. But
`_resolve_target` has three branches that settle on `pointed` instead — a reply
with `entity_id: null`, a reply whose name and point disagree in a way the task
resolves, and the excluded-kind branch — and none of them checked anything.
Reproduced offline against the live snapshot shape:

| reply | task | old result |
|---|---|---|
| `PICK_ENTITY`, `entity_id: null`, point on the blue cube | "pick up the **red** cube" | `ok=True`, `cube_2`, the blue one |
| `TRANSFER`, `dest_entity_id: null`, `dest_2d` on marker 1 | "…place it on **marker 3**" | `ok=True`, dest `marker_1` |

The module docstring claims "neither signal is trusted alone". Pointing alone
was in fact trusted, and silently, whenever the model declined to name. One gate
on the entity finally chosen — the same predicate the named half was already
held to — closes all three branches at once.

**Scope.** Both fixes now cover `slot_N` as well as `marker_N`, since both are
place targets named by number, and a number is an identity rather than a
description: `slot 3` is not satisfied by `marker_3`. The decision prompt gained
one sentence saying so, which is not what makes this safe — the deterministic
guard is — but stops the loop spending a park, a capture and a decision on a
reply that was always going to be refused.

**Not fixed, and worth knowing.** Step 3 also shows the mask flooding:
`the box measured 15362x7582mm, outside the plausible 4-200mm range`. That is
caught and reported honestly by `locate.measure_box`'s plausibility band, and it
is the same segmentation instability as §2b — the clamp measured 153 × 25 then
148 × 20 mm on consecutive frames before failing outright on the third.

**13 regression tests** in `tests/test_instruct.py` pin all of it, including the
live transcript above and the controls that must still pass: a transfer to a
marker the task actually names, a point-only reply that lands on the right
thing, and an unnumbered destination the model remains free to choose.

### 2t. A run abandoned with the object in the jaws — three causes, fixed

Reported symptom, 2026-08-03, on the newly interactive `run_instruction.py`:

```
> move stapler to center aruco marker

[1] cubes=0 ... free_markers=3 unknown=1
    registered obj_1: aruco at (180, -142) 54x20mm
    registered obj_2: stapler at (48, 234) 157x25mm
    PICK_ENTITY  obj_2 (stapler)
    -> holding the stapler  (commanded, not checked)

[2] ... objects=1
    STOP  [model said stapler]
    reason: 'stapler' is not in this snapshot (ids: cube_1, cube_2, marker_1, ...)
    -> refused, stopping. The scene or the request needs to change.
```

The stapler was in the jaws and the run gave up. Three separate defects, in
causal order.

**1. "aruco" was treated as a noun to go find.** `unmatched_nouns` built its
vocabulary from entity *labels* plus the literal kind names, and a marker's label
is `marker 2 (free)` — so "aruco" matched nothing, the pre-pass spent a grounding
call on it, and registered `obj_1` at (180, −142): **19.6 mm from `marker_2` at
(162, −150)**, the very marker meant. An ArUco tag *is* what a `marker_N` is —
the repo says so from `detect_markers` to the "ArUco tag did not decode" refusal
— so the word cannot name anything else on this desk.

Fixed with `_KIND_SYNONYMS`, credited only for kinds the snapshot actually holds:
with no marker listed, "the aruco tag" stays unmatched and reaches the grounder,
which is the honest answer. Deliberately **not** extended to "block"/"brick" for
cubes: a wooden block the HSV detector never saw is a thing the grounder should
still be sent after, and crediting the word would turn that into a silent pick of
the nearest coloured cube.

**2. A destination id that is not an id aborted the task.** With `obj_1` a
*duplicate object* rather than the marker, no legal `TRANSFER` destination existed
— a transfer's destination must be a `marker_N` or `slot_N` — so the one-step move
degraded into PICK then PLACE, and the extra step is where it died. At step 2 the
model answered `PLACE_ENTITY` with `entity_id: "stapler"`, the label of the thing
in its own jaws, which the prompt explicitly says is never a destination. The
existence check refused before the accompanying point was ever read.

That check is right for a **pick** and wrong for a **destination**, and the
difference is not a matter of strictness. Destinations are enumerated
exhaustively — markers from ArUco detection, slots from fixed geometry — so an id
not in the list cannot mean "the desk does not have it"; it can only mean the
model wrote a string that is not an id. A pick id is the opposite: absent really
can mean the detector never found the thing, and substituting the nearest cube is
the one outcome nothing downstream can detect. So free text is now discarded and
the point decides; the pick gate is untouched.

**An id-*shaped* string is still refused.** `marker_9` is not a mislabelled
field, it is a claim about the desk, and "a number is an identity, not a hint"
(§2s) applies to the model's reply exactly as it does to the instruction. Quietly
placing on `marker_2` instead is the failure that gate exists for.
`_looks_like_entity_id` draws the line, and the existing
`test_a_transfer_refusal_says_which_end_failed` caught the first version of this
fix for being too broad.

The discarded string is still reported as `model_entity_id`, printed as
`[model said stapler]`, so a reply resolved this way is never silent about it.

**3. A refusal ended the run while the gripper was full.** `run_instruction`
broke out of the loop on any `ok=False`, which strands whatever is held. With the
jaws full the step is now retried instead: the next one is a fresh park, a fresh
frame and a fresh decision, and a refusal caused by the frame rather than the
request goes away on its own. This run is the case in point — step 1 saw
`unknown=1`, a marker whose tag had not decoded, and step 2 saw `unknown=0`; the
desk had not changed, only the exposure. Bounded by `max_steps`, arm parked
between tries.

**Measured after the fixes, same instruction, same desk.** No phantom was
registered for "aruco". The model still answered `PLACE_ENTITY` with
`entity_id: "stapler"` — twice, in two separate runs — and both times the point
resolved it to `marker_3` and the run continued to `DONE`, exit 0, where before
it exited 2 with the stapler held.

**What the fixes do not touch, and should not be read as fixing.**

- The model still chose `PICK_ENTITY` over `TRANSFER` for a plainly two-ended
  task, with three free markers listed. Removing the phantom made a transfer
  *available*; it did not make a 4B model choose it.
- **"center" is not a checkable attribute.** It is in `_FILLER`, so nothing
  constrains which marker gets used, and the destination was whichever marker the
  model's point landed on (`marker_3` at (154, 157)) — not the most central one
  (`marker_4` at (211, 7)). The reply's own prose said "slot_2" while its point
  said `marker_3`; the point is the only channel carrying coordinates, so the
  point won.
- **The physical outcome was not confirmed.** Inspected by camera afterwards, the
  stapler lies across `marker_1` in the top right, not on `marker_3` where the
  place was commanded. Its two grounded measurements a full task apart —
  (133, 233) and (119, 229) — agree with each other and with neither the
  commanded destination nor a successful move. Mask size over the same object
  ranged 68×38, 82×28 and 111×65 mm across three measurements, so its centre is
  not reliable to better than a few centimetres, which is consistent with a
  marginal grasp. `(commanded, not checked)` means what it says.
- **A marker resting under the stapler stays `unknown`.** That is why
  `unknown=1` persists across runs, and it permanently removes one destination.

**7 regression tests** in `tests/test_instruct.py`: the label-as-destination
fallback, the `marker_9`/`marker 9`/`marker-9` refusals, a junk id with no usable
point, the untouched pick gate, the kind synonyms, the presence gating, and
"block" still reaching the grounder.

---

### 2u. The checks that double-guessed Qwen were removed — owner's decision

> "let's get the code working and introduce checks if things fail, rather than
> have the implementation fail because we don't trust what qwen is telling us."

Three consecutive live runs were abandoned by validation rather than by the arm
(§2t, and the statue run below). The gates were each defensible on their own and
collectively made the loop unusable, so the policy changed: **once the model has
named a target, that is the target.**

**The run that settled it.** `move green statue to center aruco marker`:

```
    registered obj_1: statue at (186, -230) 156x14mm
    STOP  [model said obj_1]
    reason: the task says ['green', 'statue'] but obj_1 is 'blue statue'
            -- the detector's own label contradicts the choice
```

The figurine is teal. Measured over its saturated mask pixels, the hue histogram
peaks at 90-95 with a **median of 94**; the green band ends at 88 and the blue
band starts at 90, adjacent with no neutral zone, so 3307 pixels landed in blue
against 108 in green — share 0.466 vs 0.015, far outside `COLOR_MARGIN`, so the
"not clearly any named colour" abstention could not fire. **No phrasing of
"green" could ever have passed.** The refusal was also unfalsifiable in the other
direction: "green" is only a checkable attribute because some *other* entity's
label carries it, so removing the green cube from the desk made the same
instruction succeed.

**What was removed from `_resolve_target` and `decide`:**

| Gate | What it refused |
|---|---|
| unknown `entity_id` | any id not in the snapshot — now falls back to the point |
| `unmatched_nouns` | a pick whose task named a word no label carried |
| `wrong_kind_block_reason` | a pick of a marker/slot, a place onto a cube |
| attribute contradiction (×2) | colour or noun disagreeing with the label, on `named` and on `hit` |
| rival ambiguity | two entities matching the words, point not singling one out |
| named-vs-pointed | id and point resolving to different entities |
| coordinate-space disagreement | the two readings landing on different entities |
| `excluded_destination_kind` | "not on a marker" violated by the reply |
| `missing_place_target_reason` | the instruction naming `marker 0` on a desk of 1-4 |
| `named_destination_block_reason` | a pick whose named destination was unavailable |
| DONE audit | the model reporting success on such a task |
| LOCATE_AT_PIXEL duplicate | a point already covered by an entity |

**What remains**, and why the removals are safe:

* **The physical envelope.** `Entity.pickable` / `Entity.placeable` carry reach,
  the J1 keep-out, ground Z, finger clearance and the desk polygon;
  `Entity.reason` names the one that failed. Verified live after the refactor:
  `put the red cube on marker 3` → `cube_2 (red cube) cannot be picked up:
  r=364mm is beyond the 350mm max reach`. Nothing in the policy layer can
  command a pose the envelope would have rejected.
* **Gripper state** — no transfer with full jaws, no place with empty ones.
* **Parse failures** — no JSON object, or an action outside `ACTIONS`.
* **A reply that identifies nothing** — neither a known id nor a point that
  binds. An absence, not a judgement.
* **`is_question`** — kept deliberately when everything around it went, because
  it reads the *instruction*, not the model's judgement, and no phrasing of a
  real pick-and-place trips it. Without it, "is there anything on the desk that
  is not a cube" moves the arm.

**The costs, stated plainly.** A red statue is now picked for "the green
statue" when it is the only statue. A green cube is picked for "the red cube"
when the reply names it — colour was the only thing telling two cubes apart. And
a reply whose id and point disagree resolves to the id with no warning beyond the
transcript's `[model said ...]`. Measured after the refactor against a frame
whose markers had not decoded, `put it on marker 4` (holding) resolved to
`slot_2`: the model named marker_4, no marker was placeable, and the point bound
to the nearest usable destination instead. Under the old code that was a refusal.

**Refusals no longer end a run while the gripper is full** (§2t item 3), so a
transient refusal costs a retry rather than a stranded object.

**Also in this pass.** The gates' helper functions were deleted rather than left
uncalled (384 lines), along with the tests that pinned them; 11 tests now assert
the new behaviour instead. Inline comments were rewritten to describe current
behaviour rather than the history of changes — that history lives in this
document and in git. Every file on the Qwen path is now under 1000 lines:
`instruct.py` 900 (was 1611, with `instruct_reply.py` split off),
`ask_qwen.py` 761 (was 1774, split into `qwen_panel`, `qwen_worker`,
`qwen_watch`, `qwen_prompts`), `run_instruction.py` 429 (was 1488, split into
`instruct_view` and `instruct_worker`). `locate.py` (1139) and `stack_cubes.py`
(1075) are still over and are not Qwen-specific.

---

---

## 3. Are we cube-agnostic? No — in one specific sense

The observed summary line is the evidence:

```
cubes=2 blockers=1 free_markers=3 occupied=1 unknown=1 free_slots=15 phantoms_dropped=1 off_table_blobs=3
```

Decoded, with the code that produces each field
(`scene.Scene.summary_line`, `entities.build_snapshot`):

| Field | Source | Class-agnostic? |
|---|---|---|
| `cubes=2` | `len(scene.cubes)` — HSV blobs from `detect_cubes` surviving `filter_phantoms` | **No.** `detect.COLOR_RANGES` |
| `blockers=1` | cubes not within `MARKER_OCCUPY_RADIUS_MM` (26 mm) of a marker | **No** |
| `free_markers=3` | ArUco markers, tag decoded, `PLACE_CLEARANCE_MM` (45 mm) clear **of cubes** | **No** |
| `occupied=1` | marker with a **cube** on it | **No** |
| `unknown=1` | calibrated marker, tag not decoded this frame | n/a (marker bookkeeping) |
| `free_slots=15` | `free_placement_slots(calib, markers, cubes)` — clearance **against cubes** | **No** |
| `phantoms_dropped=1` | a blob rejected by `is_phantom_detection` (area band or work region) | **No** |
| `off_table_blobs=3` | blobs `detect_cubes` discarded as behind the desk edge | **No** |
| `objects=N` | appended by `build_snapshot` **only when the registry is non-empty** | Yes — but absent here |

The missing `objects=` suffix is the tell. At that moment the world model
contained **zero** non-cube entities, and it will keep containing zero until a
human or the VLM points at something.

### Three senses of "cube-agnostic" — we are 2 for 3

**1. Manipulation — yes.** `locate.measure` → `object_entity` → `pick_at` knows
nothing about cubes. It measures a silhouette, takes the long axis from
`minAreaRect`, infers height from silhouette inflation, and applies a 180°
yaw period for elongated objects versus 90° for compact ones. The stapler was
gripped on hardware.

**2. Referent resolution — yes.** `unmatched_nouns` + `locate_target` +
`LOCATE_AT_PIXEL` register anything Qwen can name and point at as `obj_N`,
after which it is treated exactly like a cube by every downstream gate.

**3. Enumeration — no.** *Nothing that is not a coloured cube enters the world
model unprompted.* `observe` → `build_snapshot` → `capture_scene` →
`detect_cubes` (HSV) + ArUco + fixed slots. That is the whole world.

### Why enumeration matters beyond tidy reporting

Three concrete consequences, all in current code:

- **Negative and survey queries are wrong by construction.** "Is there anything
  else on the desk?" "Put it somewhere clear." The model can only answer from
  cubes, markers and slots.

- **Clearance is asymmetric, and the unsafe direction is unchecked.**
  `object_entity` (entities.py:499-512) *does* check a registered object
  against `scene.cubes` and refuses if a cube is inside the jaw span. But
  `pick_block_reason` (entities.py:238-240) and `Scene.pickable`
  (scene.py:248-253) iterate `self.cubes` only. **A cube sitting 20 mm from a
  stapler is still reported pickable**, and the jaws will close through the
  stapler. An unregistered object is not even a candidate for the check.

- **Place targets ignore non-cube occupancy.** `free_markers`
  (workspace.py:484-490) requires `PLACE_CLEARANCE_MM` from every **cube**.
  A marker with a pen lying on it reads `free`, and the arm will place onto it.
  `free_placement_slots` has the same signature and the same blind spot.

`discover.py` was written specifically to close sense 3 and is tested, but it
is not wired in, so none of the above is fixed yet.

---

## 4. Outstanding work, ranked

### Blocking a trustworthy live run

**1. ~~No motion verification.~~ Abandoned as unachievable on this rig** — see
§2r. `verify_pick` / `verify_place` were built (§2e), ran for one day, and were
removed for answering wrongly in both directions: they template-matched the
ArUco tag beside a stapler and called a completed pick a failure. With no jaw
sensor there is nothing left to build this on, so the loop reports commands and
says so. Everything that can be checked is now checked *before* the move
instead, on both ends of a transfer.

**2. Wire `discover` into `build_snapshot` / `mt4_scene`.** Closes sense 3
above. Needs three decisions (§5) and carries a real risk of over-detection
noise in the entity list, which is why it was left out pending review. **Now
the top remaining item.**

**2a. ~~Choose a grasp point within an object.~~** Done — `mt4_vision/grasp.py`,
validated on hardware, see §2k and §2m.

**2b. Stabilise the mask.** Now the biggest source of wrong answers: the same
object measures 19 mm one frame and 128 mm the next (§2n). Everything above it
is correct given a correct mask. Two concrete steps: reject shadow in the
GrabCut path, and add a cross-frame stability check like the one `measure`
already applies across window sizes.

**3. Extend the clearance predicates to objects.** `Scene.pickable`,
`pick_block_reason`, `free_markers` and `free_placement_slots` all need to see
registered objects, not just cubes. Independent of (2) and cheaper — worth
doing first, because (2) makes the blind spot much more likely to fire.

### Improvements with clear value

**4. ~~`locate_target` should ask for `bbox_2d`~~** — **done**, see §2b.

**5. Unit tests for `instruct.py`** — **started**. `tests/test_instruct.py`
covers the coordinate convention, the `Grounding` assembly, the
`measure_grounding` retry, the stopword and noun-phrase rules and the three
deterministic guards, 50 tests, using the live
replies verbatim as fixtures. `decide`, `bind` and `_matches_attributes` still
have none, and that is where the remaining §2 bugs came from.

**5a. Decide what "place on X" means for a non-cube.** See §2c. The grasp point
and the object centre coincide for a cube and diverge by tens of millimetres
for anything elongated.

**6. Update the design doc** to the name-verify-first architecture that was
actually built.

### Needs hardware time

**7. Fix `_assumed_height_mm`'s axis dependence** and re-run the Phase 0 gate.
Moves the target for every pointed-at object, so it must be measured, not
reasoned about.

**8. Re-run `calibrate_table_edge.py`** after reworking the wood colour model
(hue wraparound plus a saturation floor that matches the measured 15–27 range,
without the black-pad false positives). Recovers 2 cubes on its own.

**9. ~~Calibrate the jaw span~~ — done, see §2l.** `grip_span_s_at_zero_mm` =
212.3 and `grip_span_s_per_mm` = 1.881, giving 38.4 mm fully open. The
jaw-width gate is live.

**10. Reject shadow in the GrabCut path.** The last known mask error, and the
reason the stapler measures 55.8 × 38.4 mm and is refused. See the end of §2n.

---

## 4b. Review backlog, 2026-08-03

Four review passes over the whole branch (reuse, simplification, efficiency,
altitude). The contained findings are applied and committed. These are the ones
left, because each crosses a module boundary, changes behaviour, or was
contested between reviewers. Ranked by value.

**A. `locate.grasp_feasibility` and `entities.object_entity` are two answers to
"can the arm take this object", and they have drifted in opposite directions.**
`grasp_feasibility` tests region → jaw span → whether a J4 angle exists.
`object_entity` tests region → antipodal grasp plan → cube clearance → region
again at the moved grasp point. Each has a gate the other lacks. Five callers
run **both** and hand-reconcile the answers (`mt4_mcp/server.py:445`, `:518`,
`mt4_vision/__main__.py:147`, `:241`, `move_object_to_marker.py:133`), each with
its own `if not ok and entity.reason is None:` merge. The acting paths —
`mt4_pick`, `_reacquire`, `run_instruction` — read only `entity.pickable`, so
**the J4 gate never runs where it matters** and an infeasible long-axis grasp is
discovered by `motion.resolve_yaw_j4` raising mid-move. The two also disagree by
3 mm about the same jaws: `jaw_span_block_reason` has no margin,
`grasp.plan_grasp` uses `max_span - SPAN_MARGIN_MM`. Fix: delete
`grasp_feasibility`, make `object_entity` the single computation, let callers
read `entity.pickable` / `entity.reason`. Behaviour change — some objects that
are pickable today would start being refused, correctly.

**B. The pick-source / place-destination split is written out nine times in
three modules.** `instruct._NEVER_PICKABLE_KINDS`, `wrong_kind_block_reason`,
the `kinds = ("cube","object") if ... else ("marker","slot")` tuple,
`grasp_for`'s `kind in ("marker","slot")`, the "not somewhere to put things"
branch, the alternatives list, `capable = hit.pickable if ... else placeable`,
plus `server.mt4_pick` and `server.mt4_place`. Fix: `PICK_KINDS` / `PLACE_KINDS`
beside the `KIND_*` constants, and one `Entity.admits(role) -> (bool, reason)`
returning the category-error sentence when the kind is wrong and `self.reason`
when a physical gate failed.

**C. There is no single notion of "does this word name this entity" — seven
label parsers, none of them `Entity`'s.** `unmatched_nouns`' vocabulary build,
`instruction_attributes`' vocabulary build, `_matches_attributes`' three-way
dispatch, `wrong_kind_block_reason`'s label split, `register_object`'s word-wise
merge, the `f"marker {num}"` parse done twice, `run_instruction.py:208`'s
substring test (inconsistent — substring where the others are word-wise), and
`server.py:587`'s `label.split()[0]` to recover a colour `entity.color` already
holds. Fix: compute `Entity.names: frozenset[str]` once at build time from kind
+ measured colour + noun + printed marker number, and route every text match
through `entity.answers_to(word)`. `Entity.color` already proves the pattern —
it exists precisely because the label could not answer a question (§2r).

**D. Three object registries, three dedupe policies.**
`instruct.register_object` merges within `DUPLICATE_MM` and folds labels;
`discover`'s merge loop uses the same 12 mm but keeps the larger measurement and
never merges labels; `server._register_object` has **no** dedupe at all, so two
`mt4_locate_at_pixel` calls on one pen yield `obj_1` and `obj_2` — the exact
failure `register_object`'s docstring exists to prevent. `run_instruction`'s
LOCATE_AT_PIXEL path also writes `objects[f"obj_{n}"]` directly, bypassing the
deduplicating helper it uses on the grounding path.

**E. `measure_box` re-implements the whole silhouette-to-`LocatedObject`
transform**, so the parallax correction (now shared), the plausibility band (now
shared) and the **two-window stability check** exist in inconsistent copies. The
stability check runs only on the `measure` path, so a GrabCut or box measurement
is acted on with no stability evidence at all. `measure_box` also stores no
`mask`, which silently degrades `plan_object_grasp` back to the centroid grasp
that `grasp.py` was written to replace. Fix: one pipeline parameterised by
segmentation strategy.

**F. `instruct` depends on `run_instruction` having run the grounding
pre-pass.** The 75-line ground-measure-register loop lives in the CLI, and
`_resolve_target` assumes it has already run — so an MCP-driven agent gets
"register it with LOCATE_AT_PIXEL first" with no mechanism to satisfy it. The
`held` / `objects.pop` / `history.append` rules are likewise enforced in the
script and consumed by `build_prompt` and `unmatched_nouns` across the module
boundary. Fix: a `TaskState` and an `instruct.step()` in `mt4_vision/`; the
script keeps argparse, printing and the preview.

**G. `qwen.py`'s HTTP client is a line-for-line copy of `grounding.py`'s** —
health probe, hand-rolled multipart body, `URLError`/`HTTPError` mapping, the
tunnel hint, ~60 lines twice. When the transport changes, one service gets it
and the other silently does not.

**H. Smaller, all verified:** `bind` computes an off-frame pixel that
`entities._desk_pixel_projector` deliberately suppressed, undoing the one
rejection the projector makes; three implementations of entity → `Grasp`, with
`server._reacquire` honouring `calib.face_align_picks` and `Entity.as_grasp`
ignoring it, so the same cube lands at a different wrist angle depending on
which driver is running; two balanced-delimiter JSON scanners
(`instruct._first_json_object` and `qwen._json_spans`, **not** drop-in
equivalent — the fix is one scanner covering both); three entry points for
"which reading of these numbers" plus two unrelated strategies for choosing
between them; `ask_qwen.py` re-implements `preview.wrap_text`,
`preview.draw_lock_ring` and `preview.LiveFeed`'s threading skeleton.

**I. Grounding fires once per unmatched *word*, not per noun phrase.** A
two-word target costs ~1.5 s of extra model time and a second measurement, and
lands both hits on the same object for `register_object` to merge.
`instruct.noun_phrase` already brackets the phrase.

### Contested — do not act without re-measuring

`discover.desk_pixel_mask` maps all 921,600 pixels through the homography in
float64 (71 ms, 88.5 MB transient) where projecting the four desk-polygon
vertices and calling `fillPoly` would take 0.37 ms. The efficiency reviewer
measured them agreeing to 99.7%. The reuse reviewer ran both against the live
calibration independently and reported that the shortcut depends on every
polygon vertex staying in front of the camera, and that the vectorised version's
NaN horizon guard is doing real work. Both cannot be right. `discover` is
unwired, so nothing is paying the 71 ms today — settle it before wiring it in,
not before.

### Rejected

Removing `discover.py` as dead code. It has no production caller and that is
deliberate — §1 records the three decisions wiring it in needs first.

---

## 5. Decisions you own

**May the loop choose a destination the task deliberately left open?** This is
the one thing blocking "place it on a non-marker location", and it is a policy
question, not a bug. The standing rule is never to substitute a target, because
nothing downstream can detect that it happened. That rule is about targets the
*user named*. Here the user named no destination — they stated a constraint, and
fifteen free slots satisfy it equally. Three options:

- *Refuse, as now.* Safest, and consistent with the rest of the layer. The cost
  is that a perfectly clear instruction cannot be carried out, and the refusal
  is about the model's arbitrary point missing an arbitrary lattice point.
- *Place at the pixel the model pointed at,* gated on `in_work_region`. This
  honours the model's own answer rather than snapping it to a slot, and it is
  the semantically correct reading — the model is pointing at bare desk because
  bare desk is what was asked for. It needs a new action path; `PLACE_ENTITY` is
  entity-only by design.
- *Pick any placeable slot when the task names no destination.* Simplest to
  build, and the weakest guarantee.

Note that widening the bind radius is **not** an option: the slot lattice's
closest pair is 51 mm apart, so a radius large enough to catch a 91 px miss
makes neighbouring slots ambiguous instead.

**Motion verification is closed, not solved — see §2r.** It was built, it
misread an ArUco tag as the object it was checking for, and it is gone. Nothing
observes the outcome of a move on this rig, and nothing can until there is
either a jaw sensor or a re-detector that does not depend on a rectangular
template. What that leaves is a loop whose guarantees are all *pre*-motion, and
which says so on every outcome line. Whether that is a trustworthy live run is
now a judgement about the benchmark rather than a missing feature.

The nearest thing to a real fix, if it is wanted later: match through the stored
silhouette rather than the bounding box. `locate.measure` already keeps `mask`
and `mask_origin_px`, and `cv2.matchTemplate` takes a mask, so the marker pixels
would contribute nothing to the score and a lifted object would score near zero.
`measure_box` (the Grounding-DINO path) stores no mask and would need a
fallback. Two-sided origin testing — a match **and** the spot still looking
unlike bare desk — is the other half.

**How aggressively should `discover` feed the entity list?** Three sub-choices:

- *All proposals, or only those that survive `in_work_region`?* Showing
  out-of-region objects tells the model why it cannot act on them; hiding them
  keeps the list short. The cube path currently hides them and reports a count.
- *Merge with, or replace, HSV cube detection?* Running both means the same
  cube can appear as `cube_2` and `obj_5`. `is_own_colour_blob` already exists
  to suppress that for registered objects and would need extending.
- *What happens to over-detection?* Cables, tape and the arm's body all deviate
  from the desk. They survive `measure` sometimes. A 20-entity list is worse
  for the VLM than a 6-entity one even when every entry is real.

**Is the ~±10 mm pick accuracy acceptable for the benchmark**, or does the
height estimator get fixed first? Cubes are forgiving because they are wide;
a pen at 8 mm wide is not. If the benchmark includes narrow objects, item 7
moves above item 4.
