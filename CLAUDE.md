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

## Explaining your work (required)

Chat replies are explanations for a competent colleague who has **not** read the code. Written deliverables — docs, plans, code comments — stay dense, thorough and complete. Simplify the prose *around* the work, never the work.

- **Lead with the headline or correction**, in ordinary words.
- **One idea per sentence.** Short sentences beat compressed ones.
- **Explain the mechanism before the term.** "The camera looks at the desk from an angle, so a tall object appears shifted sideways" — *then* say parallax.
- **No metaphors or wordplay standing in for an explanation.** Clever phrasing makes the reader decompress it before they can judge the claim.
- **Keep every number and observed fact.** Plain does not mean vague, and a measured value beats an adjective.
- **Bold lead-ins, short paragraphs.** Decisions the user owns go in their own section at the end, stated as choices with consequences.

## Code comments describe the present (required)

A comment says what the code does now, and why it has to be that way. It is not
a changelog. Git already records what changed and when — `git log`, `git blame`
and PR descriptions are the history. Someone reading the file wants the current
contract, not a diff narrative they have to subtract from what they see.

- **No before/after framing.** Do not write "previously", "used to", "no longer",
  "changed from", "old behaviour was", "as of <date>", "fixed <date>", "this was
  a literal `True` until", "two things used to stand here". If the sentence only
  makes sense to someone who saw the old version, cut it.
- **Keep the constraint, drop the history.** A rule learned from a past bug is
  worth documenting — as a rule, in the present tense. Write "The guard compares
  the predicted pose against the current one, so an already-violating pose can
  still move toward legality." Not "This used to refuse every violating pose,
  which froze the arm after a reset."
- **Never narrate removals.** A comment about code that is not there wastes the
  reader's attention. Delete the code and the comment together.
- **Keep every number and measured fact** (see "Explaining your work" above).
  Plain, present-tense comments are still dense. A date stays only when it is
  part of a live fact — when a calibration was measured, which firmware a
  constant was verified against — not as a marker of when an edit happened.
- **Nothing addressed to a reviewer.** No "note the fix here", "as requested",
  "per feedback", "this is the new version".
- **One exception:** a compatibility shim, migration path or deprecation really
  is about the past. State in one line what it is compatible with and the
  condition for deleting it.

Same rule for docstrings and for comments in firmware, tests and config. It does
**not** apply to project docs like this file, `docs/`, or commit messages, where
recording why a decision was made is the point.

## Reporting task completion (required)

This governs the reply once you've *finished doing* something — not planning, analysis, or investigation replies, which follow "Explaining your work" above.

- **Be brief.** A few sentences, not a report. Skip the step-by-step of what you tried.
- **Lead with what the user needs to do next.** If something is blocking — you need an approval, a physical check at the arm, a decision only they can make — say that first, in plain terms. If nothing is blocking, say so in one line or drop the section; don't invent a next step to fill space.
- **Summarize the result in one or two sentences** when it matches what was asked. Add detail only when the result differs from the request or something unexpected happened along the way.
- **Skip code-level detail.** The user reads code casually, not in depth. Don't name functions, files, or internal mechanisms unless they asked about them directly — say what changed in terms of what it now does, not how the code does it.

## Primary tools

| Tool | Use |
|------|-----|
| MCP `mt4_status` | Homed flag, TCP, joint steps, gripper |
| MCP `mt4_scene` | Cubes, markers, free slots |
| MCP `mt4_home` / `mt4_move_to` | Recover pose, probe reachability |
| `Mt4Client` (Python) | Scripts: `calibrate_*.py`, `stack_cubes.py`, `jog.py` |
| `mt4_vision.camera.capture_frame` | Camera / detection issues |

Homed FK TCP is about **(190, 0, 226)**; J1 keep-out **140 mm**; soft ground **115 mm**. After kinematics or keep-out changes, flash and re-run vision calibration.

## Envelope limits apply to every control path

Four ways to drive the arm, all four now gated on ground Z and the J1 keep-out:

| Path | Guard |
|------|-------|
| `mp` / `mq` | target, per-segment and routed-path checks |
| `cj` Cartesian jog | `setup_cartesian_jog` clamps, re-run every 40 ms |
| `j` joint jog | `refresh_envelope_guard_if_due`, polled every 10 ms |
| `m` relative move | same guard |

`motion_step_allowed` is **not** an envelope check — it only knows joint step
counters and the J2+J3 coupling, which say nothing about where the TCP is.
Measured 2026-08-02: 13% of the legal joint box puts the TCP below the desk
(worst 78 mm) and 7% inside the keep-out (worst r = 118 mm), so `j` and `m`
used to reach the desk with nothing objecting.

The guard compares a *predicted* pose against the current one and stops only
motion that makes a violation worse. It must stay that way — after an MCU
reset the arm sits at r = 124.6 mm, already inside the cylinder, so a guard
that refused every violating pose would freeze it there. Homing is unaffected:
it pulses the step pins directly and never touches the DDA.

**`GROUND_Z_MM` = 115 sits ~12 mm below actual desk contact (~127).** The
guard enforces the floor faithfully; the floor itself is deliberately slack so
picks at `table_z` = 127.2 have room. Raising it would make the guard prevent
contact rather than limit it, but it would also squeeze every pick.

## Where pick/place is allowed

One predicate: `mt4_vision.workspace.in_work_region(x, y, calib)`. Four things
must all hold, and `work_region_block_reason` names the first that fails:

1. the arm can hold the grasp pose (IK + joint soft limits + keep-out + reach)
2. it can lift `PICK_LIFT_MM` = 50 mm straight off it
3. the point is on the desk (`table_polygon_robot` in the calibration)
4. the point images inside the camera frame with a margin

Do not add a fifth gate somewhere else. The thing this replaced was a convex
hull of the ArUco marker centres applied twice with different allowances, in
two files; measured 2026-08-02 it admitted 828 cm² of a table where the arm can
safely work 2278 cm², and three cubes plainly on the desk were missing from
`mt4_scene` entirely. Marker positions describe where paper was taped down,
not where the desk, the arm, or the camera end.

Re-measure the desk edge with `python calibrate_table_edge.py` after moving the
arm, the desk, or the camera. It needs the wall visible above the desk, so park
the arm clear of the back of the frame first.

## Typical failure patterns

- **`err mp segment` after aborted calibration** — arm often stranded low with **J4 at soft limit**; home + park before retrying.
- **Pick/place “failed” with no vision symptom** — motion planning failure, not mis-detection.
- **Empty scene / no cubes** — arm blocking camera, wrong camera index, or cold camera frame.
- **Serial busy** — stop MCP and other clients before flash or a second script.
- **`stack_cubes.py`: "No reachable clear spot for <color>"** — the clear/park search came up empty near the stack site; fixed 2026-07-24 (full-circle angle sweep in `clear_aside_xy` + annulus grid fallback in `choose_park_slot`, since corner markers and 8 fixed `PLACEMENT_SLOTS` could exhaust all candidates). If it recurs, the site is likely boxed in on all sides (occupied + work region + shadow), not a hardware fault.
- **`no stack-safe route` while a stack is standing** — the target sits in the column's *forearm shadow*: further from the base than the stack and within ~40 mm of its bearing, so the forearm would cross over the column. Field case 2026-08-03, `unstack_cubes.py --marker 3`: landing (202, 239) is 93 mm beyond marker 3 and 16 mm off its bearing; with 3 cubes left the forearm reaches 187.3 mm where 192.2 mm is needed. `StackPlanner.column_shadow(levels)` is the shared up-front veto — unstack applies it to scatter landings, stack to pick candidates, so neither commits to a target that will fail routing. It costs ~1.5% of otherwise-valid landings. A recurrence means a *new* place that chooses a table XY without it.
- **A cube on the desk is missing from `mt4_scene`** — check the summary's `off_table_blobs` count. Non-zero means `detect_cubes` discarded blobs as behind the desk edge; if a real cube is among them the desk polygon is stale, so re-run `calibrate_table_edge.py`.

## Project context

- Custom firmware: `firmware/mt4_jog/`
- Vision + pick/place: `mt4_vision/`
- MCP server: `mt4_mcp/` (stdio via `.cursor/mcp.json`)
- Calibration: `vision_calibration.json` (path from `mt4_vision.calib.DEFAULT_CALIB_PATH`)
