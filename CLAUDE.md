# Claude / agent instructions — MT4 repo

This project controls a **real WLKATA MT4 arm** and **overhead USB camera** over serial. Treat hardware as part of the debugging loop, not a black box.

## Hardware autonomy (required)

When errors, unexpected behavior, or “why did this fail?” involve the arm, camera, gripper, or serial link:

1. **Investigate yourself** — query live state, read terminal logs, run diagnostics. Do not ask the user to check pose, COM port, or scene unless you are blocked (no serial, no camera, physical safety).
2. **Recover when needed** — home, retreat to camera park, free the serial port, then continue.
3. **Fix and verify** — patch code when the root cause is systemic; re-test on hardware when possible.

Cursor rules with full detail:

- `.cursor/rules/hardware-investigate.mdc` — investigation workflow, error mapping, recovery
- `.cursor/rules/flash-ok.mdc` — flash firmware without asking

## Primary tools

| Tool | Use |
|------|-----|
| MCP `mt4_status` | Homed flag, TCP, joint steps, gripper |
| MCP `mt4_scene` | Cubes, markers, free slots |
| MCP `mt4_home` / `mt4_move_to` | Recover pose, probe reachability |
| `Mt4Client` (Python) | Scripts: `calibrate_*.py`, `stack_cubes.py`, `jog.py` |
| `mt4_vision.camera.capture_frame` | Camera / detection issues |
| `terminals/*.txt` | Recent command output and firmware errors |

Homed FK TCP is about **(190, 0, 226)**; J1 keep-out **140 mm**; soft ground **115 mm**. After kinematics or keep-out changes, flash and re-run vision calibration.

## Typical failure patterns

- **`err mp segment` after aborted calibration** — arm often stranded low with **J4 at soft limit**; home + park before retrying.
- **Pick/place “failed” with no vision symptom** — motion planning failure, not mis-detection.
- **Empty scene / no cubes** — arm blocking camera, wrong camera index, or cold camera frame.
- **Serial busy** — stop MCP and other clients before flash or a second script.
- **`stack_cubes.py`: "No reachable clear spot for <color>"** — the clear/park search came up empty near the stack site; fixed 2026-07-24 (full-circle angle sweep in `clear_aside_xy` + annulus grid fallback in `choose_park_slot`, since corner markers and 8 fixed `PLACEMENT_SLOTS` could exhaust all candidates). If it recurs, the site is likely boxed in on all sides (occupied + hull + shadow), not a hardware fault.

## Project context

- Custom firmware: `firmware/mt4_jog/`
- Vision + pick/place: `mt4_vision/`
- MCP server: `mt4_mcp/` (stdio via `.cursor/mcp.json`)
- Calibration: `vision_calibration.json` (path from `mt4_vision.calib.DEFAULT_CALIB_PATH`)

## Learned-policy paths

Two, and they do **not** share conventions — check which one you are in before
touching an action vector.

| | pi0.5 (`mt4_pi/`) | ACT (`mt4_pi/act/`) |
|---|---|---|
| Actions | joint velocities, rad/s, padded to 32 | absolute joint targets, rad, 5-dim |
| Rate | assumes 15 Hz | 10 Hz (measured) |
| Task | language prompt | one-hot columns in the state vector |
| Docs | `docs/PI05_FINETUNING_PIPELINE.md` | `docs/ACT_PIPELINE.md` |

Both use the **world-frame** wrist angle (`status.tcp.j4`, via
`mt4_pi.jointstate.joint_state_from_status`), never the raw joint — under
`ORIENT=hold` they differ by j1, and conflating them compounds per tick.

**Recorder defect — fixed 2026-07-29, but the old `pi_stack_demos` data is
not recoverable.** Labels in that corpus contain single-step joint jumps up
to 171° (median worst 95° for `stack`, vs 3.6° for `pi_demos`). Cause: a
stacking carry goes out as ONE queued firmware path (`mq`), so
`_emit_path_waypoints` has to split the single measured duration back across
its legs — and it split by *Cartesian distance*. `routed_travel` applies the
face-align wrist angle on the leg **after** the wrist-held lift, and that leg
is often a ~1 mm height adjustment: 0.2% of the path by distance, 3634 motor
steps in reality. The whole 80° wrist sweep collapsed into one tick.
(It was **not** the retreat to camera park, as previously recorded here —
that runs after `end_recording()` and never labels a tick.)

Legs are now weighted by `steps × speed_us` on the busiest axis, which is how
the firmware's DDA actually paces them; validated against measured wall time
(predicted 7.42 s vs measured 9.00 s, the gap being the unmodelled accel
ramp). A fresh 6-episode cycle worst-jumps **4.7°**, under the 5° safety cap.
The 88 bad episodes are quarantined in `data/pi_stack_demos/rejected/`;
re-collect stacking data rather than trying to salvage them (the raw waypoint
log is not stored, only the interpolated ticks).

**Specular glare reads as a blue cube.** The laminated ArUco pads throw a
highlight whose faintly-tinted rim clips the blue band and passes every
geometric gate — cube-sized, square, reachable, inside the marker hull by
definition. Measured over 377 episodes: 28 picks dispatched at one, all blue,
all failed; in the last 100 episodes of a run, blue was failing 95% (36/38)
against ~20% for red/green, and the quota planner made it worse by
preferentially re-selecting starved blue. Two gates in `mt4_vision.scene`
(`is_glare_blob`): centroid inside a *decoded* marker outline (a cube on a
pad occludes it, so a decodable pad is provably empty), and
`CubeDetection.sat < 90` sampled at the top-face centroid, not over the
contour — the contour only holds pixels that already passed the band's S
floor, so its median is blind by construction.
