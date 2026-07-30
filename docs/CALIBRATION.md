# Calibration — step by step

Everything the vision stack knows about where things are lives in one file,
`vision_calibration.json` (path from `mt4_vision.calib.DEFAULT_CALIB_PATH`,
overridable with `MT4_VISION_CALIB`). This document is the operator's guide to
producing it: what each step measures, how to run it, how to read its output,
and what invalidates it later.

The design principle throughout: **the arm is the ground truth, not a ruler.**
Every number here is derived from the arm touching or placing something and the
camera observing the result. Nothing is hand-measured, and nothing needs camera
intrinsics.

---

## Which script do I need?

| Situation | Run |
|-----------|-----|
| Fresh setup, or the arm base moved | [Full sequence](#the-full-sequence) — all four steps |
| Camera moved / bumped / drifted, base and markers did not | [`recalibrate_camera.py`](#camera-moved-only), then steps 3 and 4 |
| Markers repositioned or re-printed | Full sequence from step 2 |
| Cube picks miss by 15–30 mm in a consistent direction | [step 3](#step-3-cube-top-parallax) (`calibrate_height.py`) — this is the classic symptom of a missing cube-top map |
| Overlay draws the arm's trajectory too low / offset | [step 4](#step-4-camera-nadir-and-lens-height) (`calibrate_camera_nadir.py`) |
| Arm was power-cycled or reflashed | [step 5](#step-5-j4-wrist-zero) (`calibrate_j4.py`) — MCU state, not in the file |
| Picks grab air on non-cube objects | [Gripper span model](#optional-gripper-span-model) — uncalibrated by default |
| Different cube size | Full sequence (the cube edge length feeds the parallax model) |
| Kinematic constants or step ratios changed | Flash, then full sequence |

---

## The five layers

Calibration is not one transform. It is five independent measurements, each
fixing a different error source. They stack, and a missing layer degrades
silently rather than failing loudly.

| # | Layer | Fields | Fixes | Missing → |
|---|-------|--------|-------|-----------|
| 1 | **Table plane** | `homography`, `bundle_homography`, `table_z`, `safe_z`, `workspace_hull_px` | Pixel → robot XY for anything lying on the table | Nothing works |
| 2 | **Cube top** | `cube_top_homography`, `cube_top_residual` | Cubes are detected by their *top* face, 20 mm up — parallax shifts it | ~15–30 mm pick error, warning on stderr |
| 3 | **Camera geometry** | `cam_xy_robot`, `cam_height_mm` | True parallax at *any* height — drives the trajectory overlay | Overlay draws low; also the cube-top fallback |
| 4 | **J4 wrist zero** | *(MCU step counter — not in the file)* | "Jaws along the arm" = world J4 0°, so face-aligned picks work | Wrist angles are meaningless; picks hit cube corners |
| 5 | **Gripper span** | `grip_span_s_at_zero_mm`, `grip_span_s_per_mm` | Closing distance for objects that are not 20 mm cubes | Jaws close on air, pick reports success |

Layers 1–3 live in the JSON file. Layer 4 lives in the microcontroller and is
lost on power cycle. Layer 5 is optional and currently unmeasured on this rig.

### Why layer 1 is fit the way it is

For a pinhole camera viewing a plane, pixel ↔ plane really is exactly a
homography — 8 degrees of freedom, the right model family. But fitting all 8
DOF to ~5 marker-center touches went badly wrong in practice: the two
perspective parameters are barely observable from that few points, and least
squares "explained" millimetre-level touch noise by planting the camera's
horizon line *inside the workspace*. That fit was exact at every data point and
divergent everywhere between them.

So [mt4_vision/table_fit.py](../mt4_vision/table_fit.py) sources each part of
the model from data rich enough to determine it:

1. **Perspective** comes from the marker **corners**. Every marker is a printed
   square of identical physical size, so 5 markers give 20 subpixel corner
   points. A small bundle adjustment jointly fits one homography (pixels → a
   metric table frame) plus each marker's unknown pose in that frame. The arm is
   not involved; accuracy is limited only by print quality and lens distortion.
2. **Alignment** (metric frame → robot frame) comes from the arm **touches**, as
   a 4-DOF similarity. N touches give 2N equations for 4 unknowns — heavily
   overdetermined, so touch noise averages out instead of leaking into
   perspective, and one bad touch shows up as an outlier residual instead of
   silently warping the map.

The composed result is still a single 3×3 homography. A solution whose
denominator changes sign across the workspace is rejected outright — that is the
physical-camera check the naive fit failed.

### A note on this rig's camera

Despite being called "overhead", the camera is **steeply oblique**: measured
nadir at robot ≈ **(518, −35)**, lens ≈ **244 mm** above the table. The nadir is
far off-desk, well outside the workspace, so raising the TCP shifts its image a
lot along the camera azimuth. Height parallax is radial from that far nadir and
grows with height — not the roughly-constant shift a single-height fit suggests.
This is why layer 3 exists as its own measurement.

The residual error at large ±Y is lens **barrel distortion**, which a single
radial model cannot express and nothing in this pipeline undistorts. A real fix
would need `cv2.calibrateCamera` intrinsics plus an undistort pass. Full detail,
including which rows of the older top-down analysis this superseded, is in
[ASSUMPTIONS.md](ASSUMPTIONS.md) section B.

---

## The full sequence

```powershell
python -m mt4_vision markers        # 1. verify the markers are seen
python calibrate_vision.py          # 2. table plane (interactive, ~5 min)
python calibrate_height.py          # 3. cube-top parallax (automatic, ~10 min)
python calibrate_camera_nadir.py    # 4. camera nadir + height (automatic, ~10 min)
python calibrate_j4.py              # 5. J4 zero (interactive, ~1 min)
python -m mt4_vision entities       # 6. verify
```

Steps 3 and 4 are fully autonomous — the arm places and sweeps its own probe
cube. Only steps 2 and 5 need a human.

### Prerequisites

- Custom firmware flashed (`python flash_jog.py --port COM6`).
- Serial port free. **Only one process can own it** — stop `jog.py`, the MCP
  server, and any other client first.
- Overhead camera connected and seeing the desk.
- At least 3 ArUco markers (DICT_4X4_50) taped flat on the work surface, spread
  out, all within the arm's reach if possible. Sheet:
  [ArUco Markers A4 5x5cm.pdf](ArUco%20Markers%20A4%205x5cm.pdf).
- Two or three coloured cubes on the desk for steps 3 and 4.
- Workspace clear enough to home safely.

---

## Step 1: verify the markers

```powershell
python -m mt4_vision markers
```

Detects markers and saves an annotated frame. Check that:

- **Every marker you intend to use is detected**, with the id you expect.
- Markers are **flat**. A curled or crumpled marker throws off both the corner
  bundle and the centre pixel, and glare on a crease reads as a phantom blob
  later.
- The markers **span the working area**. The table map is most accurate inside
  their convex hull; detections outside it are rejected outright
  (`workspace_hull_px`).

`--dict scan` tries every dictionary if the ids look wrong.

---

## Step 2: table plane

`calibrate_vision.py` — the interactive step. It homes the arm, captures a
reference frame of the markers, then drops into the jog controls from `jog.py` so
you can physically touch the TCP to each marker centre.

```powershell
python calibrate_vision.py
```

Options: `--camera N`, `--dict 4x4_50`, `--output PATH`, `--no-gamepad`,
`--j1-center` / `--j2-pull` (homing overrides).

### Controls

Jog is identical to `jog.py` (I/K = Z, A/D = X, S/W = Y, J/L = J4 roll,
Q/E = gripper, `-`/`=` = speed, `` ` `` = invert XY). Calibration adds:

| Key | Action |
|-----|--------|
| `0`–`9` | Record the current TCP as marker *digit* |
| `G` | Record grip pose (pick Z + gripper S) — press **while gripping a cube** |
| `H` | Re-home |
| `SPACE` | Status |
| `X` | Stop, drivers off (moved off `0`, which is now a digit key) |
| `ENTER` | Finish: fit and save |
| `ESC` | Quit without saving |

Gamepad: sticks/triggers jog as usual; **A** records a marker (auto-identified —
see below), **Y** homes, **B** stops, **X** status, **Start** finishes, **Back**
quits.

### Procedure

1. Let it home. If already homed it says so; press `H` any time to re-home and
   clear the camera's view.
2. It prints the markers in view. You need **at least 3**.
3. Jog the TCP down onto a marker's centre until it just touches, then record
   it — either with that marker's **digit key**, or with gamepad **A**.
4. Repeat for every reachable marker. **Order does not matter**, unreachable
   markers are simply skipped, and re-recording a marker overwrites it.
5. While gripping a cube sitting on the table, press **`G`** once. This captures
   `table_z` and the gripper S value that holds a cube.
6. Press **ENTER** (or **Start**).

**Gamepad A is worth using.** It identifies the marker automatically by
occlusion: the marker the TCP is touching is the one that just disappeared from
the camera's view relative to the reference frame. This eliminates the single
most expensive mistake in this whole process — pressing the wrong digit. Marker
ids are not human-readable, and swapping ids 2 and 3 once cost a full
recalibration. If more than one marker is hidden, or none, it refuses and tells
you to use a digit key.

### Reading the fit report

```
bundle+similarity fit from markers [0, 1, 2, 3, 4]
Corner-bundle RMS: 0.42px (~0.5mm; >1px suggests lens distortion)
Per-marker touch residual (mm): {0: 1.2, 1: 3.4, 2: 0.9, 3: 2.1, 4: 1.5}
Per-marker leave-one-out error (mm): {0: 2.1, 1: 6.8, 2: 1.4, 3: 3.0, 4: 2.2}
```

- **`kind`** — `bundle+similarity` is what you want. `affine` means the corner
  bundle was unavailable or failed (a `NOTE:` line says why); an affine map has
  no perspective correction and degrades away from the marker triangle.
- **Corner-bundle RMS** — geometric consistency of the printed squares. Above
  ~1 px points at lens distortion (nothing in this pipeline undistorts) or a
  non-flat marker.
- **Touch residual** — how far each touch sits from where the camera geometry
  says that marker is. A few mm is normal.
- **Leave-one-out** — refit without that marker, then predict it. This is the
  honest number; in-fit residuals always look good. Only computed with ≥4
  touched markers.
- **`WARNING: markers [...] disagree with the camera geometry by >25mm`** — take
  this seriously. It almost never means a sloppy touch. It means **the wrong
  digit was pressed** at those markers. Re-jog and re-record before trusting the
  result.

### The prompts

Then it asks for a few values, each with a default; Enter accepts.

| Prompt | Default | Notes |
|--------|---------|-------|
| `table_z` | Median TCP Z of the recorded touches | The TCP Z while touching a marker *is* the table height there. Also the grip height for a cube on the table — the jaws straddle its lower faces. Currently ≈ 127.2 |
| `cube edge length` | Previous, else 20 | Feeds the parallax model and stack place height |
| `safe_z` | Previous, else `table_z + 1.5 × cube` | Travel height. Keep modest — the arm belongs low over the desk |
| `camera height above table` | Previous, else 240 | **Only a seed.** Step 4 derives this properly and overwrites it |
| `grip_close_s` | The `G` reading, else previous, else 240 | Firmware S value that holds a cube (currently 255) |
| `grip_open_s` | Previous, else 140 | |

Saved with a `bundle+similarity` (or `affine`) homography, the marker-centre
convex hull, and `raw_marker_observations` — the raw pixel/corner/robot data for
every touched marker. **That raw block is why `recalibrate_camera.py` can exist**:
a fitted matrix is not invertible back to its inputs, which cost us dearly once.

### What this step resets

`calibrate_vision.py` builds a **fresh** calibration object. It carries over
`color_ranges` and `cam_xy_robot` from the previous file, and seeds several
prompts from it — but these fields go back to defaults:

| Reset to | Field |
|----------|-------|
| `None` | `cube_top_homography`, `cube_top_residual`, `probe_observations` |
| `None` | `grip_span_s_at_zero_mm`, `grip_span_s_per_mm` |
| `{}` | `color_xy_offset_mm` |
| `700` / `2400` | `travel_speed_us`, `approach_speed_us` |
| `True` | `face_align_picks` |

So **steps 3 and 4 are not optional after step 2.** Nothing downstream fails
when they are skipped; cube picks just silently regain 15–30 mm of parallax
error. There is a one-shot stderr warning at first use, and that is the only
signal you get.

---

## Step 3: cube-top parallax

`calibrate_height.py`. Cubes are detected by their **top face**, `cube_height_mm` above the table
plane the markers lie on. On this oblique camera that costs a measured
**15–19 mm** of pick error across most of the desk, and ~26 mm near the
low-x markers. This step measures the shift directly.

```powershell
python calibrate_height.py
```

Fully automatic — no human interaction. The arm picks up a cube, places it at a
grid of robot-frame targets, and photographs where each landing appears.

**The ground truth is the arm's own positioning**, not a marker: *we* choose the
target and command the arm there, so this needs no correspondence to
already-calibrated points. That is also why the grid points need not be anywhere
special.

### What it probes

12 built-in targets — the 8 `PLACEMENT_SLOTS`, plus 4 densifying the +y quadrant
and the near-base region — and any calibrated marker centre not already within
25 mm of a scheduled target. Ordered to start in the well-lit region near the
base and finish in the far, dimmer area, so a late detection failure costs one
point at the end of the chain instead of resetting it at the start.

There is deliberately no point near (100, 0): the homed arm's gripper hangs over
that spot in the camera view, so a cube placed there cannot be re-detected.

### Options

| Flag | Effect |
|------|--------|
| `--holdout N` | Reserve N points to validate instead of fit (default 1) |
| `--reps N` | Placements per target (default 1). Release drag moves the cube 2–6 mm per placement; extra reps average it out. Use `--holdout 0` with this — the split is by observation, so a held-out target would leak into the fit via its other reps |
| `--probe-at X Y` | Exact XY of a cube the arm previously placed — skips the less accurate vision bootstrap for the first pick |
| `--extra-target X Y` | Additional probe target (repeatable) |
| `--skip-grid` | Probe only `--extra-target` points |
| `--merge` | Include stored `probe_observations` in the fit |
| `--avoid X Y` | Skip targets within place clearance of this XY (repeatable) |

`--merge` is refused across probe colours. Red- and blue-cube centroids of the
same physical position measure ~10 px apart, because each HSV mask admits a
different mix of lit and shaded side faces. Merging them poisons the fit.

### Safety rails it applies

The script cannot see whether a grasp worked — `pick()` is open-loop, with no
force or current sensing, so a closed gripper is not proof of anything. Several
checks gate each data point:

- **Grasp verification** — if the cube is still within 30 px of where it sat
  before the pick, the grasp failed. (This really happens: multiple "successful"
  placements turned out to be the untouched cube still sitting at its original
  spot.) After two failures it rotates to a different cube, since the bootstrap
  map may be worse where the first one sits.
- **Placement sanity** — the raw table-plane estimate of the landing must be
  within 100 mm of the intended target. This is a slip detector, not a
  tolerance.
- **Match radius** — a candidate blob must be within 120 px of where the probe
  is expected, which rejects the arm's own orange body reading as red elsewhere
  in the frame.
- **Phantom filter** — the base's hardware intermittently reads as small blue
  blobs inside the keep-out cylinder; those are excluded from the probe pool.
- **Pose recovery** — before every leg it checks for a stranded TCP or a J4 near
  its soft limit (which `center_placed_cube`'s ±90° rotation can cause) and
  re-homes if needed.
- **Observations are always kept.** Raw `(pixel, robot, colour)` records are
  written on every exit path that has any, and failure paths *append* rather than
  replace — an aborted 2-point run must not destroy a prior full collection
  (which happened once).

### Reading the output

```
bundle+affine fit from 15 point(s)
In-fit reprojection error (mm): [0.9, 1.4, ...]
Held-out validation error (mm): [2.3]
(This is the number that matters -- in-fit error is expected to look good regardless)
With residual layer (mm): [0.2, 0.3, ...]
```

- **`bundle+affine`** (≥8 points) or **`bundle+similarity`** (fewer) — both reuse
  the corner bundle's perspective from step 2 and fit only a low-DOF alignment on
  top. An unconstrained 8-DOF homography from a handful of noisy blob centroids
  is exactly the overfit that broke the table map; it is not attempted.
- **Held-out validation error** is the honest number. Single-digit mm is good.
- **Residual layer** — the per-location error is stable to ~1 mm but nonlinear in
  position, beyond what an affine can express. So the probe residuals themselves
  are stored and interpolated at read time: Gaussian-weighted mean (σ = 60 mm)
  with a regulariser (0.25) that shrinks the correction toward zero away from the
  probes, so there is no extrapolation pathology.

A historical note worth knowing: the first version of this fit used only the 3
reachable markers, hit zero residual, and made pick accuracy **worse**. Marker
corners are subpixel-precise; a cube's colour-blob centroid is not, and a
3-point exact affine has no redundancy to average that noise out — it just bakes
it in. Sampling many points across the workspace is the whole point.

### Requirements

Needs ≥5 usable grid targets (1 with `--merge`) and ≥4 points after the holdout.
Below that it saves the observations and refuses to refit.

---

## Step 4: camera nadir and lens height

`calibrate_camera_nadir.py`. The trajectory overlay needs to know where a point at height *z* lands in the
image. For a pinhole camera that is pure geometry: a point at height *h*
projects, through the flat table homography, onto the table-plane intersection of
its camera ray, which sits radially outward from the camera **nadir** by
`cam_height / (cam_height − h)`. Two numbers describe it completely.

```powershell
python calibrate_camera_nadir.py
```

Fully automatic. The arm grips a cube and hovers it at a column of known heights
over several XY spread across the desk, detecting where each lands, then fits the
radial model to the apparent-vs-true offsets.

### Why fit it rather than measure it

**Bias-free by construction.** The constant grip/detection offset (blob centroid
vs TCP) is the *same* at every height in a column, so fitting the offset's
*growth with height* cancels it. The cube-top map from step 3 cannot serve this
purpose — it bakes that bias in.

The earlier approach prompted for a `cam_height_mm` and guessed the nadir. That
was wrong by a wide margin: the guess was 700 mm, the real lens height is ~244
mm, and the camera is oblique rather than overhead. The overlay drew the
trajectory far too low as a result. This fit recovers the height within a couple
of mm of a tape measure — but derives it, like every other number here, from
vision plus the arm.

### What it sweeps

10 columns at robot XY spread across **both** axes, 8 height rungs each from 8
to 148 mm above the table. The y-spread is what pins the nadir's y: a +y and a
−y column give opposite-sign vertical parallax that triangulate it. The wide ±Y
columns keep the fit honest toward the frame edges; the central ones anchor the
middle.

It is robust to this rig's two nuisances:

- **Self-occlusion** — the black gripper claw hides a low held cube from the
  oblique camera, so each column is tracked through whatever heights *are*
  visible and skips the rest, rather than demanding the lowest rung.
- **Probe identity** — the held cube is told from desk cubes by elimination: the
  desk is photographed before the grip, and the probe is the same-coloured blob
  that is *not* sitting where a desk cube was. Rungs then track by pixel
  continuity, re-acquiring against the model prediction if a skipped rung leaves
  a gap.

The cube is always set back down and the camera cleared on exit, including on
failure.

### Options

`--color red` (probe colour), `--dry-run` (fit and report without writing),
`--dump PATH` (raw per-rung records as JSON for offline analysis).

### Reading the output

```
Collected 62 rungs across 9 columns: [...]
nadir = (518.1, -35.0)   cam_height = 244.0 mm
robot-space rms = 18.42 mm   overlay pixel error mean=6.3 max=14.1px
```

- **nadir far off-desk** (large x, here ~518) is expected and correct on this
  rig — that is what "steeply oblique" means.
- **robot-space rms of ~20–30 mm is normal and fine.** A wide-Y fit is
  inherently distortion-limited: the oblique lens barrel-distorts the frame
  edges, which no single pinhole radial model can express. Above 35 mm it refuses
  to write, which catches a genuinely broken sweep (latched the wrong blob,
  absurd nadir) rather than an honest edge-distortion residual.
- **Overlay pixel error** is what you actually see. Under ~15 px tracks the arm
  well.

It refuses to write with fewer than 2 columns, fewer than 6 rungs, or under
40 mm of y-spread (the nadir's y would be unconstrained).

---

## Step 5: J4 wrist zero

`calibrate_j4.py`. J4 has no home switch. Its step counter starts at 0 wherever the wrist happened
to sit at boot. Face-aligned picks need a known relationship between "jaws along
the arm" and world-frame J4, and `face_align_picks` defaults to **true** —
meaning the stack assumes this has been done.

```powershell
python calibrate_j4.py
```

1. Homes if needed, then parks the TCP on the arm axis at (230, 0, 215). The
   `y = 0` requirement is structural: it puts J1 ≈ 0 so the arm axis *is* robot
   x. `--pose X Y Z` can move the park point, but y must still be 0.
2. Jog until the jaws look aligned with the arm:
   - Left thumbstick horizontal — world X nudge
   - **J / L** (or right stick X) — J4 wrist roll
   - **H / A** — home (leaves the park pose; re-park before aligning)
   - **ENTER / Start** — confirm
   - **ESC** — abort
3. On confirm it sends firmware `j4zero`, which rewrites J4's step counter so
   that pose reports **world J4 = 0**. No motion occurs. Then it retreats for the
   camera.

`--dry-run` reports the current world J4 without zeroing.

### This one is not in the file

`calibrate_j4.py` **writes nothing to `vision_calibration.json`**. The zero lives
in the microcontroller's step counter. It survives `home` (the firmware preserves
J4 across the J1–J3 rewrite) but is **lost on power cycle and on reflash**.

Re-run it at the start of any session where face-aligned picks matter. Nothing
detects that it is missing.

---

## Step 6: verify

```powershell
python -m mt4_vision entities            # ids, positions, pickable + why not
python -m mt4_vision goto-marker 4       # hover over a marker
python -m mt4_vision goto-marker 4 --touch   # descend and touch it
```

`goto-marker --touch` is the end-to-end check on layer 1: the TCP should land on
the marker centre. Without `--touch` it hovers at `safe_z`, which won't crash
into the desk even if the calibration is off — start there.

For a per-layer breakdown when something is off:

```powershell
python scripts/diagnose_pick_accuracy.py --port COM9
python scripts/diagnose_pick_accuracy.py --no-arm      # camera-only
```

It isolates each segment of the chain separately: table-plane residuals, live
camera drift vs the calibration snapshot, the cube-top parallax gap, multi-frame
detection noise, and commanded-vs-reported TCP consistency. Use it before
re-running any calibration — it tells you *which* layer moved.

---

## Camera moved only

`recalibrate_camera.py`. If the camera moved but **the robot base and the markers did not**, you do not
need to re-touch anything. The markers' robot XYs are already stored in
`raw_marker_observations` and are still correct. All the camera move invalidated
is the *pixel* side of each correspondence.

```powershell
python recalibrate_camera.py
python calibrate_height.py          # required -- cube-top map was cleared
python calibrate_camera_nadir.py    # required -- camera pose was cleared
```

It loads the stored robot XYs (no arm motion at all), captures a fresh frame,
auto-detects each marker's current pixel centre and corners, refits through the
same `fit_table_map()` step 2 uses, backs up the old calibration to
`backups/vision_calibration_pre_recalibrate_<timestamp>.json`, and saves.

`--dry-run` fits and reports without writing.

### The assumption is not self-checking

If "the robot and markers didn't move" is wrong, this produces a **confidently
wrong** calibration with no way to catch it from pixels alone. The residual
report is an internal consistency check — does one homography explain every
marker's old robot XY from its new pixel position — not proof the assumption
holds.

An outlier flagged above 25 mm usually *does* mean something real: a marker
moved, the wrong id matched, or the original touch was recorded off-centre.
Investigate before trusting the result. The check uses the worst of the in-fit
and leave-one-out errors, because least squares smears a single bad
correspondence across all markers — the culprit's own in-fit residual can look
innocent.

### `exclude_from_fit`

A marker can be flagged in `raw_marker_observations` with
`"exclude_from_fit": true`. It then keeps its stored robot XY as a usable *slot
coordinate* but stays out of the map fit. This exists because marker 1's touch
was recorded at the arm's reach limit, where arm-frame and camera geometry
disagree by ~30 mm; including it dragged the whole similarity. The flag is
carried through refits — dropping it would silently re-arm that trap.

---

## What each entry point preserves

| Field | `calibrate_vision` | `recalibrate_camera` | `calibrate_height` | `calibrate_camera_nadir` |
|-------|:---:|:---:|:---:|:---:|
| `homography`, `bundle_homography` | **writes** | **writes** | keeps | keeps |
| `workspace_hull_px` | **writes** | **writes** | keeps | keeps |
| `raw_marker_observations` | **writes** | **writes** (new pixels, old robot, flags kept) | keeps | keeps |
| `table_z`, `safe_z`, `cube_height_mm` | prompts | keeps | keeps | keeps |
| `grip_open_s`, `grip_close_s` | prompts | keeps | keeps | keeps |
| `cube_top_homography`, `cube_top_residual` | **cleared** | **cleared** | **writes** | keeps |
| `probe_observations` | **cleared** | **cleared** | **writes** | keeps |
| `cam_xy_robot` | kept | **cleared** | keeps | **writes** |
| `cam_height_mm` | prompts (seed) | **cleared** | keeps | **writes** |
| `color_ranges` | kept | kept | keeps | keeps |
| `color_xy_offset_mm` | **cleared** | **cleared** | keeps | keeps |
| `grip_span_s_*` | **cleared** | **cleared** | keeps | keeps |
| `travel_speed_us`, `approach_speed_us` | **reset to default** | keeps | keeps | keeps |
| `face_align_picks` | **reset to true** | keeps | keeps | keeps |

The clears are mostly deliberate: a cube-top map, a camera pose, and
colour-centroid offsets are all functions of the viewing geometry, so a
camera-side change invalidates them. Note that `grip_span_s_*` and the two speed
fields are also lost on a full recalibration — if you have measured a jaw-span
model, save the values before re-running step 2.

---

## Optional: gripper span model

`grip_span_s_at_zero_mm` and `grip_span_s_per_mm` are **currently unmeasured** on
this rig (both `null`). They model jaw opening as
`span_mm = (at_zero − S) / per_mm`.

This matters more than it looks. `grip_close_s` is the value for a **20 mm
cube**; closing to it on a ~10 mm pen never touches the object. With no
grip-retention sensing anywhere in the stack, the pick then **reports success
having done nothing**. `grip_s_for_span_mm()` warns once on stderr and falls
back to `grip_close_s`.

To calibrate: grip two or three objects of known width, note the S at which each
just holds, fit `S = at_zero − per_mm × width`, and store both coefficients. From
the one point we have (S 240 holds a 20 mm cube) plus S 285 = fully closed, the
slope is around 2.25 S/mm — a guess to check, not a substitute for measuring.

Vision reports a *silhouette* width, which on this oblique mount reads wide for
anything with height, so picks close `GRIP_SQUEEZE_MM` = 4 mm past the measured
width. Too-closed merely squeezes, which the jaws tolerate; too-open is the
failure mode with no detector.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `no calibration at <path> -- run: python calibrate_vision.py` | No file. Start at step 2 |
| `WARNING: cube_top_homography is not set` on stderr | Step 3 was skipped after a step-2 or `recalibrate_camera` run. ~15–30 mm of pick error |
| `WARNING: grip_span_s_* are not calibrated` | Expected — see [above](#optional-gripper-span-model). Only affects non-cube objects |
| `only N markers visible ... need >=3` | Camera view, marker placement, or wrong `--dict` |
| `WARNING: markers [...] disagree ... by >25mm` | Almost always the wrong digit key. Re-record those markers |
| Fit came out `affine`, not `bundle+similarity` | Fewer than 2 markers had detectable corners, or the bundle failed — read the `NOTE:` line |
| Corner RMS > 1 px | Lens distortion (nothing undistorts here) or a non-flat marker |
| `lost track of the probe cube -- aborting` (step 3) | Probe rolled out of view or out of reach. Observations are kept; re-run |
| `Grasp likely failed` repeatedly (step 3) | The bootstrap map is poor where that cube sits. It auto-rotates cubes after 2 failures; put a cube nearer the base |
| `Columns lack y-spread` (step 4) | Too few columns survived reach/occlusion. Clear the desk and retry |
| `Fit rms ... exceeds 35mm -- refusing to write` (step 4) | A mis-tracked sweep. Check for a second same-coloured cube confusing the elimination logic |
| `err mp segment` mid-calibration | Arm stranded low, often J4 at a soft limit. Home and park before retrying |
| Picks hit cube corners rather than faces | Step 5 not run this power cycle |
| Empty scene / no cubes | Arm blocking the camera, wrong camera index, or a cold first frame |

Calibration touches hardware, so the usual hardware-debugging rules in
[CLAUDE.md](../CLAUDE.md) apply: query live state, read `terminals/*.txt`, home
and park to recover, and treat a failure as a software or state bug before
suspecting the arm.

---

## The calibration file

`vision_calibration.json` is gitignored — it is specific to one physical setup.
`backups/` holds archived copies, including the automatic
`_pre_recalibrate_<timestamp>` ones.

| Field | Meaning |
|-------|---------|
| `homography` | 3×3 pixel → robot XY, table plane. Row-major |
| `bundle_homography` | Pixel → metric table frame (the perspective part alone), so low-DOF refits can reuse it without re-solving perspective |
| `table_z` | Robot Z of the table surface, and the TCP Z for gripping a cube on it |
| `safe_z` | Travel height between moves |
| `travel_speed_us` | Step period for `safe_z` transits. Must be under the firmware ramp's 1800 µs to engage accel/decel; 700 is fastest |
| `approach_speed_us` | Step period for descents and table touches — deliberately slow (≥1800) so the ramp stays off near the work surface |
| `grip_open_s` / `grip_close_s` | Firmware absolute gripper S, 120–285 |
| `grip_span_s_at_zero_mm` / `grip_span_s_per_mm` | Linear jaw-span model. `null` until measured |
| `cube_height_mm` | Cube edge length |
| `cube_top_homography` | Pixel → robot XY at cube-top height. Absorbs oblique parallax *and* the centroid bias at that one height, empirically |
| `cube_top_residual` | Interpolated residual layer: `points`, `deltas`, `sigma_mm`, `reg` |
| `probe_observations` | Raw step-3 `(pixel, robot, colour)` records, for offline refits and outlier checks |
| `raw_marker_observations` | Per marker: `pixel`, `corners`, `robot`, plus flags like `exclude_from_fit` |
| `cam_xy_robot` / `cam_height_mm` | Measured nadir and lens height. Drives `robot_to_pixel` at all heights; also the cube-top fallback when `cube_top_homography` is unset |
| `color_ranges` | Per-colour HSV overrides merged over `detect.COLOR_RANGES` |
| `color_xy_offset_mm` | Per-colour XY correction. Each HSV band admits a different mix of lit/shaded side faces, so centroid bias is colour-dependent — a map calibrated with one probe colour mis-locates the others by a constant few-to-15 mm |
| `face_align_picks` | Command J4 from the detection's `yaw_deg`. Assumes step 5 has been done |
| `workspace_hull_px` | Pixel convex hull of the marker centres. Detections outside are rejected |

---

## Further reading

| Doc | Contents |
|-----|----------|
| [ASSUMPTIONS.md](ASSUMPTIONS.md) | Every assumption the accuracy chain rests on, how well each is known, and where error actually leaks. Section B covers camera geometry |
| [MT4_ARCHITECTURE.md](MT4_ARCHITECTURE.md) | Hardware, pin map, flash path |
| [mt4_vision/calib.py](../mt4_vision/calib.py) | The `Calibration` dataclass and the transform functions |
| [mt4_vision/table_fit.py](../mt4_vision/table_fit.py) | The bundle + similarity fit, and why it is not one least-squares call |
| [firmware/mt4_jog/src/main.cpp](../firmware/mt4_jog/src/main.cpp) | Serial protocol, including `j4zero` |
