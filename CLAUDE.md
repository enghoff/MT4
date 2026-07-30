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
