# Assumption audit — pick / place / stacking accuracy (2026-07-20; kinematics updated 2026-07-21)

Every row names an assumption the system currently relies on, what we
actually know about its validity (uncertainty), and how strongly system
performance depends on it (sensitivity). Evidence cites the 2026-07-19/20
instrumented runs where applicable.

## A. Arm / kinematics

| # | Assumption | Uncertainty | Sensitivity | Notes / evidence |
|---|---|---|---|---|
| 1 | Kinematic constants are exact: `LINKAGE1=130`, `LINKAGE2=150`, `CENCER_OFFSET=45`, `HEAD_OFFSET=35`, `HEAD_HEIGHT=14.43`, park angles **(107.0°, −9.3°)** at pull-off; J2/J3 step zeros at limit/interference (**≈135.57°, −23.59°**) | Links still ±1–2 mm hand-measured; park tape-fit 2026-07-21 to home TCP (r≈190, wrist z=240, pads≈226) — not a multi-radius hover fit. Limit-referenced zeros derived from that fit + pull-offs (1000/500) | **HIGH** (walk), **lower for absolute Z** | Pre-fit clinometer homes (103°, 4.7°) over-reported home Z by ~40 mm and table contact by ~20 mm. Absolute Z at home now matches tape; residual **z-walk** (XY vs commanded Z) and J2/J3 SPD ratio remain open — still want a hover-profile fit. |
| 2 | Steps/degree are integer and identical across J1–J3 (35.0) | A 1% ratio error ≈ 1° over 100° excursion ≈ several mm at the TCP, growing with joint travel | **HIGH**, position-dependent | Assumed from "shared gearbox design"; the EEPROM dump itself had 35.556 for another axis. **Differential** J2 vs J3 SPD is the strongest remaining z-walk lever after the home refit. |
| 3 | The arm is rigid (no sag, no elastic deflection) | Unmeasured; arm is demonstrably springy (exploited for press-releases), load = one 20mm cube | **MEDIUM** | "Hover-measured walk doesn't transfer to the place context" (documented) is consistent with load/pose-dependent deflection. A few mm, folded invisibly into every calibration. |
| 4 | Commanded moves position the TCP exactly (no backlash/stiction) | Measured ~6–9mm dead-zone on small reversing moves at r=209 (attempts 1, 5); apparent response "gain ~2" with sign flips in x even after take-up moves (attempt 10) | **HIGH** | Current convergence floor of the landing servo. Confounded with #11 — the "gain 2" could equally be measurement noise, not backlash. |
| 5 | Step counters = physical position (no lost steps in normal operation) | No encoders; `get_tcp` reports *commanded* steps, not reality | **CATASTROPHIC when violated** | r=315 stall precedent shows this fails under load+speed; slow loaded legs mitigate, but nothing *detects* a silent stall — every later arm-frame coordinate is wrong until re-homing. |
| 6 | Motion between waypoints is a Cartesian straight line | Firmware `mp` is piecewise Cartesian; mid-path sag still unmeasured | **MEDIUM** near the stack | Stack place/retreat now use **vertical-then-horizontal** segments within `STACK_AXIS_CLEAR_MM` (50 mm) of the site so diagonals cannot clip the column. Other paths (table pick/place) may still diagonal. |
| 7 | Commanded z is true z (`pick_z` / `table_z` from calibration, 120 mm) | Desk contact measured 2026-08-04 at ~120 mm by camera-tracked descent (±2 mm); monocular vision still cannot observe z error during a pick (it couples into XY) | **HIGH** | ±3mm decides contact-set vs. edge-drop. Soft `GROUND_Z_MM=115`, ~5 mm under the surface. A touch jogged down until the arm stops records the 115 clamp, not the desk. Edge-drops bounce cubes 30–60mm (attempts 5–7); current mitigation (release slightly above seat) validated once. |

## B. Camera / vision geometry

> **2026-07-25 update:** the "overhead" camera is actually **steeply oblique** —
> nadir far off-desk at ~(518, −35) robot, lens ~244 mm up, measured directly
> against the arm (`calibrate_camera_nadir.py`). Height parallax is therefore
> radial from that far nadir and grows with height; `Calibration.robot_to_pixel`
> now models it (isotropic scale = `cam_height/(cam_height − h)` about the
> nadir). The rows below were written top-down (2026-07-20) and are annotated
> where superseded. The residual error at large ±Y is now lens barrel distortion
> (row 9), which a single radial model cannot express — a real fix needs
> `cv2.calibrateCamera` intrinsics + undistort.

| # | Assumption | Uncertainty | Sensitivity | Notes / evidence |
|---|---|---|---|---|
| 8 | The camera does not move after calibration | Violated twice in two days (sun/heat); drifts of ~4px ≈ 4mm observed, shim-corrected to <0.6px | **HIGH** for absolute pick/place | Detectable only when markers are visible — parked cubes occluded 3 of 5 markers today. Anchor-relative stack logic is partially immune; picks and level-1 placement inherit the full error. |
| 9 | Pinhole camera, planar table, exact homographies — no lens distortion | Up to several px of position-dependent error; a 720p webcam typically has 1–3% radial distortion | **MEDIUM-HIGH, hidden** | No undistortion exists anywhere in the pipeline. Corrupts *all* derived geometry (residual layer, parallax direction `u`, px→mm Jacobian) a little, rather than one thing a lot. A one-time chessboard intrinsic calibration would remove it. |
| 10 | ~~Camera height Hc ≈ 700mm, "self-fitting" from observations~~ **(SUPERSEDED 2026-07-25)** | The 700 mm figure was simply wrong: lens is ~**244 mm** up and the camera is oblique (nadir far off-desk), measured against the arm. No live code path uses the old self-fit; the "700" now in the code is unrelated (step-periods, blob areas, grip timing). | **LOW** | `cam_height_mm`/`cam_xy_robot` now come from `calibrate_camera_nadir.py`; the radial model lives in `calib.robot_to_pixel`. See the section-B note above. |
| 11 | The top-face centroid is the cube's true center | ±2–5px per reading | **HIGH** | V-seeded segmentation assumes symmetric occlusion; the gripper fingers demonstrably occlude part of the held face (620px² vs ~2x expected). Likely alternative explanation for the ±5–8mm alternating hover readings currently attributed to backlash "gain 2" (#4) — currently confounded. Discriminating experiment: repeatedly measure a *static* cube through the servo path. |
| 12 | Cubes are the only colored blobs, separable, in fixed HSV bands | Frequent, environment-dependent failures | **MEDIUM-HIGH** | The arm itself is orange (reads red; `MAX_BLOB_AREA` + keep-out/hull the guards), hand/watch enter frame, adjacent cubes merge, lighting shifts move V/S. Area caps are mount-distance-sensitive: after moving the camera closer (2026-07-20) a real cube read ~2790px² and was dropped by the old 900/650 caps while arm flecks survived. Static-lock/misidentification class (attempts 3, 6, 8) patched with motion checks/exclusions/tight radii, not better perception. |
| 13 | A frame taken ≥0.8s after motion shows the settled scene | Unmeasured | **LOW-MEDIUM** | Vibration/pendulum sway of a held cube after a move is assumed damped within the settle sleep. Adds to #11's noise floor. |
| 14 | Partial occlusion does not materially shift reported cube XY | Centroid is the mean of *visible* mask pixels — asymmetric occlusion pulls it toward the unobstructed side (mm-scale on the desk). Gripper on held cube: fingers hide part of the top face; measured 2026-07-19 area 622px² (not the ~2× parallax-predicted size) and centroid ~54px from prediction. No geometric occlusion model anywhere in the pipeline. | **MEDIUM-HIGH** | On-desk cubes: open-loop picks use the last capture; a gripper or neighbour hiding part of the face reports a shifted `(x,y)` or drops below area filters. Held cube in the desk scene: excluded by color + predicted pixel (`is_held_cube_blob`), not repositioned. Held cube for inspect/servo: position *is* used and is biased — mitigated by tight/tracked search, motion checks, and re-measure loops (#24), not by correcting for fingers. Related to #11 (symmetric top-face) but distinct: occlusion is dynamic and pose-dependent. |

## C. Cube and grasp physics

| # | Assumption | Uncertainty | Sensitivity | Notes / evidence |
|---|---|---|---|---|
| 15 | Cubes are ideal rigid 20.000mm cubes | Real wooden cubes vary ±0.3–0.5mm with rounded corners; cumulative ±2–4mm over 8 dead-reckoned levels | **MEDIUM, grows with height** | The −3mm contact-release offset silently becomes a 0…−7mm press (or a drop) at higher levels. Worth switching release z to *measured* stack height once the model has observations. |
| 16 | A gripped cube stays fixed in the jaws | Release drag of 2–6mm proves the cube-jaw interface moves; slip during transport unmeasured | **MEDIUM** | Grip offset is measured once at inspect pose, assumed constant through transport. The hover servo re-measures at the stack and cancels most of it — one of its main contributions. |
| 17 | Opening the gripper is mechanically neutral | Measured ~4–5mm drag on a normal contact release; ~28mm flick opening under a deliberate 2mm press (attempt 10); 30–60mm bounce releasing from a >2–3mm drop onto an edge (attempts 5–7) | **EXTREME** | The least-controlled event in the pipeline, and today's dominant per-level failure mechanism. Mitigation: contact-height release without press; residual ~4–5mm scatter is the floor the verification/nudge loop must absorb. |
| 18 | Placed cubes stay put | Near-miss picks measured to shove neighbours (→ isolation-preferring selection added); a passing gripper 12mm above a stack assumed to clear it (see #6) | **MEDIUM** | |
| 19 | Cube yaw is irrelevant | After `calibrate_j4.py`, firmware `j4zero` makes jaws-along-arm report world J4 = 0. Survives `home`; lost on power cycle until re-zeroed. | **LOW** | Detection reports robot-frame edge yaw (`CubeDetection.yaw_deg`); `pick_cube` face-aligns via `pickplace.j4_for_face_align`. Keep `face_align_picks` on once zeroed; if the mount moves or zero was skipped this boot, re-run `calibrate_j4.py` (or flip `face_align_picks` off until then). |
| 20 | A ≤6mm-offset cube is a stable platform; tilt does not accumulate | Attempt 6 disproved the naive version: an on-column cube reading +11mm high was tilted, and the next level slid off | **HIGH for the max-height goal** | Now classified as perched → re-seat, but inter-face friction, lean accumulation, and the topple threshold of an 8-high column are all unmodeled. Expect a slow-lean failure class past ~5 levels. |

## D. Calibration, software logic, environment

| # | Assumption | Uncertainty | Sensitivity | Notes / evidence |
|---|---|---|---|---|
| 21 | Probe-point accuracy generalizes across the desk | 2–4mm at calibrated probe points; measured 10–65mm scatter at the reach edge; ±10mm pick tolerance assumed uniform | **HIGH outside the calibrated core** | Attempt-8 looped on a cube at r=284; attempt-9 crashed on an IK-unsolvable candidate (now guarded). |
| 22 | One grip force fits all (`grip_close_s=255`) | Per-color/per-cube differences suspected historically, never characterized | **LOW-MEDIUM** | |
| 23 | The inspect-pose reference frame is stable within a session | References motion-verified at creation now, but in-session drift is uncorrected | **MEDIUM** | A 3mm reference error biases every grip decision for that color. |
| 24 | A blob near a predicted pixel IS the object we reasoned about | Was the deepest, most-violated software assumption | **Was CATASTROPHIC, now MEDIUM** | Every past phantom failure (stack read as held cube, desk cube as reference, arm paint as cube, held cube as desk cube) is this assumption failing in a different costume. Current defenses (exclusion zones, tight/tracked search radii, motion checks, response-verified servo identity) are heuristic patches on a perception layer with no real object identity. |
| 25 | Serial acks mean motion finished | Gripper acks on parse, not arrival (patched with a settle sleep); moves are awaited; 64-byte line buffer bounds command size | **LOW now** | Any new command pattern must re-respect these constraints. |
| 26 | The desk is ours alone | User's hand appeared mid-session; cubes were re-arranged between runs | **LOW-MEDIUM, safety-relevant** | Per-level re-scanning makes the system mostly robust to this, but nothing guards against interference *during* a motion. |
| 27 | `cube_height_mm=20` serves two masters (physical z arithmetic AND vision parallax scale) | If real cubes average e.g. 19.6mm, calibration absorbs it on the vision side while the physical side accumulates error | **MEDIUM at height** | The two silently disagree as the stack grows — same mechanism as #15 but subtler (partial, inconsistent absorption). |
| 28 | Failures are software/state bugs, not hardware faults (working doctrine) | Right so far | **Unknown by construction** | A worn gripper servo, loosened finger, or degrading joint would masquerade as familiar "scatter" and be chased in software indefinitely. Periodic physical inspection is the only check. |

## Ranked: where accuracy actually leaks today

| Rank | Leak | Related # | Why |
|---|---|---|---|
| 1 | Release event physics | #17, #7 | Extreme, dominant per-level risk |
| 2 | Held-cube measurement noise vs. backlash / occlusion confound | #11, #4, #14 | Blocks servo convergence below ~3mm; needs the discriminating experiment |
| 3 | Residual z-walk / J2–J3 SPD ratio | #1, #2, #7 | Absolute home Z improved 2026-07-21; walk + height-from-parallax coupling remain |
| 4 | Blob identity as object identity | #24, #12 | Patched, still fragile |
| 5 | Silent lost steps | #5 | Rare but unbounded damage, undetected |
| 6 | Camera drift + occluded markers | #8 | Slow, recurring, partially detected |
| 7 | Cube tolerance & tilt accumulation at height | #15, #20, #27 | Will surface as the stack record climbs |
