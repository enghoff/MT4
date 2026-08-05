# WLKATA MT4 — custom control stack

A full replacement software stack for the WLKATA MT4 desktop robot arm
(ATmega2560, 115200 baud serial): custom firmware with on-device Cartesian
motion and queued multi-waypoint paths, interactive jog (keyboard + Xbox
gamepad), overhead-camera vision pick-and-place, open-vocabulary object
grounding, and an MCP server that lets an LLM drive the arm — "put the red
cube next to the blue one", "move the pen onto marker 3".

Everything on the work surface is addressed by **entity id** (`cube_2`,
`marker_1`, `slot_4`, `obj_1`) rather than raw coordinates, and an entity that
cannot be picked or placed says which physical constraint stopped it instead of
silently substituting another target.

The stock Grbl-derived firmware is replaced entirely (original images are
backed up and restorable, see [Restoring stock firmware](#restoring-stock-firmware)).

## Demo

[Autonomous cube stacking](https://youtu.be/1H_cvyK35i8) — `stack_cubes.py`
building a stack on a calibrated marker, with the live vision overlay visible throughout.

[Vision-language control](https://youtu.be/Upl5uVBjPy0) — `ask_qwen.py` turning
a typed English instruction into instructions for the MT4 arm.

## Requirements

- Python 3.10+ — `pip install -r requirements.txt`
- Windows (jog client uses `GetAsyncKeyState` / XInput)
- [PlatformIO](https://platformio.org/) + avrdude (only to flash firmware)
- For vision: an overhead USB camera and printed ArUco markers
  (DICT_4X4_50; sheet in [docs/ArUco Markers A4 5x5cm.pdf](docs/ArUco%20Markers%20A4%205x5cm.pdf))
- Optional, for open-vocab grounding: somewhere to run the Grounding DINO
  service — a CUDA GPU on this machine or another host (CPU works, slowly).
  See [Open-vocabulary objects](#open-vocabulary-objects)

Serial ports auto-detect the CH340 USB-UART when `--port` / `MT4_SERIAL_PORT`
are omitted (COM numbers often change after a re-plug). The camera index comes
from `--camera`, else `MT4_CAMERA_INDEX`, else **0**; set that variable to
whichever USB index your overhead camera landed on (`setx MT4_CAMERA_INDEX 1`
on a laptop whose built-in camera takes 0 — then restart any shell that was
already open, since it inherited an environment without it). Either can be
**-1** to auto-detect by scanning for the camera that sees the markers. The MCP
camera tools instead default to USB index **1** and take a `camera` argument.

## Quick start

```powershell
pip install -r requirements.txt

# Flash the custom firmware (one-time, or after firmware changes)
python flash_jog.py --port COM6

# Jog interactively (focus the terminal, hold keys; gamepad works unfocused)
python jog.py

# Calibrate vision (once per camera/arm placement -- see Calibration below)
python -m mt4_vision markers        # verify the markers are seen
python calibrate_vision.py          # jog-to-marker interactive calibration
python calibrate_height.py          # cube-top parallax probe-fit
python calibrate_camera_nadir.py    # camera nadir + lens height (overlay)
python calibrate_table_edge.py      # where the desk ends (gates pick/place)
python calibrate_j4.py              # J4 zero -- redo after every power cycle

# Check what the arm can see and reach
python -m mt4_vision entities       # entity table: ids, pickable, why not

# Run a task
python stack_cubes.py --marker 4 --preview
python ask_dino.py                  # interactive: name an object, it moves it

# MCP server for LLM control
python -m mt4_mcp                   # HTTP at http://127.0.0.1:8787/mcp
```

Only one process can own the serial port — stop `jog.py` and the MCP server
before running a script (and vice versa).

## Tasks

Top-level scripts, each driving the arm end to end:

| Script | What it does |
|--------|--------------|
| [stack_cubes.py](stack_cubes.py) | Build a cube stack on a calibrated marker (`--marker` required). Clears the site first, picks with the centering re-grip, places by dead reckoning; transits route the gripper *and* forearm around the growing column. `--resume N`, `--max-levels` (default 9), `--preview`, `--record` |
| [unstack_cubes.py](unstack_cubes.py) | Reverse a stack: take cubes off the top by dead reckoning and scatter them at random open spots/orientations. `--marker` and `--stack-height N` both required — the operator is trusted about N |
| [shuffle_blocks.py](shuffle_blocks.py) | Live loop: shuffle cubes between free markers and open-table slots (Ctrl+C to stop, H to re-home) |
| [ask_dino.py](ask_dino.py) | Interactive open-vocab mover: type an object description, Grounding DINO finds it, and it lands on a free marker. Live preview refreshed by a background detection thread. `--marker` (repeatable), `--dry-run`, `--no-preview` |
| [track_cube.py](track_cube.py) | Visually servo the gripper to hover above a single cube and follow it as you move it by hand |
| [jog.py](jog.py) | Keyboard + Xbox gamepad jog client — see [Jog](#jog) |

## Vision

An overhead USB camera watches the work surface, which carries ArUco markers.
A one-time calibration maps camera pixels to robot-frame XY on the table plane
— no camera intrinsics needed. Despite the name, the camera is **steeply
oblique** on this rig (measured nadir ≈ (518, −35) robot, lens ≈ 244 mm up), so
height parallax is modelled radially from that nadir and grows with height —
see [docs/CALIBRATION.md](docs/CALIBRATION.md).

### Entity model

`mt4_vision.entities` turns a camera frame into a snapshot of addressable
entities. This is what both the MCP tools and the task scripts work in terms of:

| Kind | Meaning |
|------|---------|
| `cube_N` | A colour-detected cube |
| `marker_N` | A printed ArUco tag — `N` is the tag number, so "marker 1" is `marker_1` |
| `slot_N` | An open spot on the table |
| `obj_N` | Anything registered by pixel hint or text prompt (a pen, a key) |

Every entity carries robot-frame `x`/`y` in mm plus `pickable` / `placeable`
flags. When a flag is false, `reason` names the physical constraint — J1
keep-out, out of reach, finger clearance, off the desk, outside the camera
frame, a tag that
did not decode this frame. Detections that are real but unusable are still
listed with their reason rather than dropped.

Ids belong to the snapshot that produced them, so re-query after anything
moves. There is deliberately **no nearest-match fallback**: an unresolvable
referent is reported, never substituted, because nothing in the stack can
detect that the wrong object was picked.

```powershell
python -m mt4_vision entities
```

### Calibration

Step-by-step guide, including how to read each fit report and what invalidates
what: **[docs/CALIBRATION.md](docs/CALIBRATION.md)**.

```powershell
python -m mt4_vision markers        # verify the markers are seen
python calibrate_vision.py          # jog-to-marker interactive calibration
python calibrate_height.py          # cube-top parallax map (automatic)
python calibrate_camera_nadir.py    # camera nadir + lens height (automatic)
python calibrate_table_edge.py      # desk edge -> table polygon (automatic)
python calibrate_j4.py              # J4 zero (manual jog; see below)
python -m mt4_vision scene          # sanity-check detections in robot coords
```

**`calibrate_vision.py`** homes the arm, then drops into the jog controls from
`jog.py`. Jog the TCP onto any reachable marker (any order; unreachable markers
are skipped) and record it with its digit key — or with gamepad **A**, which
identifies the marker automatically as the one the arm is hiding from the
camera. **G** records the pick height and gripper S while physically gripping a
cube; **Enter**/**Start** fits and saves. Because digits and A are taken,
drivers-off moves to **X** and home to gamepad **Y**. Three recorded markers
give an affine fit (accurate within the marker triangle); four or more give a
full perspective homography.

**`calibrate_height.py`** auto-probes the cube-top homography — the parallax
correction between a cube's top face and its footprint. Fully automatic.

**`calibrate_camera_nadir.py`** grips a cube and hovers it at a column of known
heights across the desk to fit the camera's nadir (`cam_xy_robot`) and lens
height (`cam_height_mm`). These drive the trajectory overlay and are the
parallax fallback when no cube-top map is set. `--dry-run` fits and reports
without writing.

**`calibrate_table_edge.py`** measures where the desk ends and stores it as
`table_polygon_robot`, the gate deciding whether a point is on the desk at all.
Only the back edge is measurable; the other three run past the arm's reach and
are stored nominal. No arm motion, but the wall above the desk must be visible,
so park the arm clear of the back of the frame. Re-run after the arm, the desk
or the camera moves. `--dry-run` reports without writing.

**`calibrate_j4.py`** sets the J4 wrist origin via firmware `j4zero`. J4 has no
home switch, so its step counter starts wherever the wrist sat at boot. The
zero survives `home` but is **lost on power cycle / reflash** — re-run it at the
start of any session where face-aligned picks matter.

If the **camera** moves but the arm base and markers do not, skip re-touching
markers:

```powershell
python recalibrate_camera.py
python calibrate_height.py          # cube-top map (cleared by recalibrate)
python calibrate_camera_nadir.py    # camera pose (cleared by recalibrate)
```

`recalibrate_camera.py` keeps the desk polygon, since that is stored in robot
coordinates and a camera move does not shift the desk. `calibrate_vision.py`
clears it, because a full re-touch admits that the arm or the desk may have
moved too.

Calibration lands in `vision_calibration.json` (table transform, `table_z` /
`safe_z`, gripper S values, cube-top homography, camera nadir + height, desk
polygon, HSV overrides). Everything after `calibrate_vision.py` is **not
optional**: a fresh run of it clears the cube-top map, colour offsets and the
desk polygon, and nothing downstream fails when they are missing. Cube picks
silently regain 15–30 mm of parallax error, and the desk-edge gate goes inert
so places past the edge stop being refused. See
[docs/CALIBRATION.md](docs/CALIBRATION.md#what-each-entry-point-preserves) for
the full preserve/clear matrix.

Cubes are detected by HSV threshold, then filtered in three stages. Blob size
and squareness drop the arm's own orange body and elongated streaks. The desk
polygon drops anything past the back edge, counted in the scene summary as
`off_table_blobs`. Specular glare is dropped outright — a blob under saturation
90, or one whose centroid sits inside a marker outline that *decoded* this
frame, since a decodable tag has nothing standing on it. What survives is
listed; `in_work_region` then decides which of those are pick candidates, and a
detection it refuses keeps its `reason` rather than disappearing.

The marker positions do **not** bound detection. They record where paper was
taped down, which says nothing about where the desk, the arm or the camera end.

### CLI

`python -m mt4_vision [--camera N] [--calib PATH] <subcommand>`:

| Subcommand | Purpose |
|------------|---------|
| `markers` | Detect ArUco markers, save an annotated frame (`--dict`) |
| `scene` | Detect cubes, print robot-frame coordinates |
| `entities` | Print the addressable entity table (ids, pickable, why not) |
| `pick <color>` | Pick a cube by color (moves the arm) |
| `place <x> <y>` | Place the held cube at robot-frame x/y (moves the arm) |
| `place-here` | Place the held cube at the current TCP xy (moves the arm) |
| `transfer --from X Y --to X Y` | Queued pick+place between two robot-frame XYs (`--from-yaw`, `--to-yaw`, `--center`) |
| `locate --pixel PX PY` | Measure a non-cube object at a pixel hint (`--label`, `--pick`) |
| `grounding --prompt "pen"` | Open-vocab detect via Grounding DINO (`--locate`, `--pick`) |
| `sam --pixel PX PY` | Segment at a pixel or `--box X1 Y1 X2 Y2` via SAM 2.1 (`--candidates`) |
| `goto-marker <id>` | Move the TCP to a marker — calibration accuracy check (`--touch` descends to table height) |
| `shuffle` | Home, then shuffle cubes between markers and open table |

### Open-vocabulary objects

Anything the HSV cube detector cannot name (a pen, a key, a screwdriver) is
registered as an `obj_N` entity in one of two ways:

- **Pixel hint** — read the object's pixel off the annotated frame and call
  `locate` / `mt4_locate_at_pixel`. Segmentation recovers the true centre, long
  axis and size in mm, so the pixel only has to land *on* the object.
- **Text prompt** — Grounding DINO returns candidate boxes for a phrase, and
  the top hit goes through the same measurement path.

The detector is an HTTP service you run wherever your GPU is — on this machine
if it has one, or on another host reached over an SSH tunnel or a LAN bind (it
also runs on CPU, slowly). The arm side only needs `MT4_GROUNDING_URL`
(default `http://127.0.0.1:8765`):

```powershell
python -m mt4_vision grounding --prompt "pen" --locate
```

If the service is on another host and you tunnel to it, open the forward first
and leave it running — `.\scripts\start_grounding_tunnel.ps1` does that.

Model `IDEA-Research/grounding-dino-base`. Full server setup — install,
supervision, remote access, HTTP API, troubleshooting:
[docs/GROUNDING_DINO.md](docs/GROUNDING_DINO.md). Everything else in this repo
works without it.

### Silhouettes instead of boxes

A third optional GPU service wraps `facebook/sam2.1-hiera-small`. Give it a
pixel or a box and it returns the actual outline of what is there — on the
reference desk, a click on a small Statue of Liberty figurine comes back as its
silhouette including the raised torch, which no rectangle and no HSV colour
threshold describes.

```powershell
python -m mt4_vision sam --pixel 737 570              # mask at a pixel
python -m mt4_vision sam --box 671 523 787 647        # mask inside a box
```

A single point is ambiguous — the cube, its top face, or the stack it sits on
— so the model returns three candidates with its own confidence in each, and
the client takes the best. The service keeps the encoded frame for the last
eight images it saw, so a second question about one frame costs about 20 ms of
service time against 50 for the first. Setup, HTTP API and the measured
fp16 / compile / cache choices: [docs/SAM2.md](docs/SAM2.md).

### Telling the arm what to do in English

A second, optional GPU service wraps `Qwen/Qwen3-VL-4B-Instruct`. `ask_qwen.py`
hands it one frame and one typed instruction, and it answers with the single
next action and a box around the thing to act on. Cube pick/place, calibration,
stacking and the MCP tools all work with this service absent.

```powershell
python ask_qwen.py                                  # interactive prompt + window
python ask_qwen.py "put the red cube on marker 3"   # one-shot, exit 0 = DONE
python ask_qwen.py "find all the pickable objects"  # a report, nothing moves
python ask_qwen.py --dry-run "pick up the stapler"  # decide, never move
python ask_qwen.py --record run.mp4 "..."           # the window, to a video
```

The model is the eyes and nothing else. It gets the frame and the decoded ArUco
tag numbers — a tag's printed number is the one thing no vision-language model
can read off an image — and every other object on the desk is its job to see.
No cube list, no object registry, no preprocessing of what you type. A target
comes back as a box, which GrabCut turns into a position, a size and a wrist
angle in millimetres; reach, the J1 keep-out, ground Z, jaw clearance and the
desk polygon are all checked before the gripper opens.

A task that asks *what is on the desk* rather than for something to be moved is
answered the same way, one box per object — from a second call whose only job is
to enumerate, which on a nine-cube desk returns all nine where asking the
decision prompt for a list returned one. Each object is measured and put to those
same gates, so every row carries a position, the width the jaws will close
across, and either "pickable" or the gate that stopped it — the arm's verdict,
not the model's, because what the arm can reach and how wide its jaws open are
not visible in a photograph.

The window shows the exact frame each decision was made from with the model's
own answer drawn on it, so a wrong answer about a frame the arm was blocking
looks different from a wrong answer about a clean one. A recorded session is in
[Demo](#demo). Setup, commands, and the measured coordinate-space and accuracy
findings: [docs/QWEN3-VL.md](docs/QWEN3-VL.md).

## MCP server

`mt4_mcp` exposes the arm to any MCP client over Streamable HTTP or stdio.
Natural-language pick-and-place: connect an LLM and say "put the red cube next
to the blue one".

Work by entity id. `mt4_scene` lists what is there; `mt4_pick` / `mt4_place`
act on those ids and refuse rather than substitute.

| Tool | Purpose |
|------|---------|
| **Status / motion** | |
| `mt4_status` | Full arm status (homed flag, mode, joints, TCP, drivers, jog) |
| `mt4_tcp` | Current TCP pose only |
| `mt4_stop` | Stop jog / cancel any in-progress move |
| `mt4_home` | On-device homing; returns `homed` + `tcp` |
| `mt4_move_to` | Absolute TCP move (requires homing this session) |
| `mt4_move_relative` | Bounded relative per-joint move |
| `mt4_gripper` | Open / close / stop / set the gripper |
| **Entities** | |
| `mt4_scene` | Snapshot of the work surface: ids, robot-frame x/y, `pickable`/`placeable` + `reason` |
| `mt4_pick` | Pick the entity with this id (re-detects on a fresh frame first) |
| `mt4_place` | Place the held object at a `marker_N` or `slot_N` |
| **Finding what the cube detector can't see** | |
| `mt4_camera_view` | Return the frame as an image with a 100px pixel grid and existing ids |
| `mt4_locate_at_pixel` | Register the object at a pixel of that frame as `obj_N` |
| `mt4_locate_by_prompt` | Register the top Grounding DINO hit for a text prompt as `obj_N` |
| **Raw coordinates (probing / calibration)** | |
| `mt4_pick_at` | Pick at explicit robot-frame x/y, bypassing the entity layer |
| `mt4_place_at` | Place the held object at robot-frame x/y |
| `mt4_transfer` | Queued pick+place between two robot-frame XYs — one continuous path |
| `mt4_goto_marker` | Move the TCP to a calibration marker — accuracy check |

Only one process can own the serial port — stop `jog.py` (or any other client)
before starting the server.

### Local HTTP

```powershell
python -m mt4_mcp     # http://127.0.0.1:8787/mcp (Streamable HTTP)
```

Configuration via flags or environment (a `.env` file is loaded automatically —
copy `.env.example` to get started): `MT4_SERIAL_PORT`, `MT4_BAUD`,
`MT4_MCP_HOST`, `MT4_MCP_PORT`, `MT4_MCP_PATH`. Test with
[MCP Inspector](https://github.com/modelcontextprotocol/inspector): connect to
`http://127.0.0.1:8787/mcp`, transport **Streamable HTTP**.

### Editor clients (stdio)

Both register the server for this workspace as `python -m mt4_mcp --stdio`:

- **Cursor** — [.cursor/mcp.json](.cursor/mcp.json); enable **MT4** under
  Cursor Settings → MCP.
- **Claude Code** — `.mcp.json` (gitignored, since it holds a machine-specific
  interpreter path).

### Public access (ChatGPT / remote clients)

Set `MT4_MCP_PUBLIC=1` to bind publicly, tunnel with ngrok
(`scripts/start_ngrok.ps1`), and enable the OAuth 2.1 flow (FastMCP's Google
provider) for ChatGPT-compatible auth. Full setup:
[docs/OAUTH_CHATGPT.md](docs/OAUTH_CHATGPT.md).

## Jog

`jog.py` drives the arm in world-frame Cartesian jog (the sole motion mode —
direct per-joint jog was dropped), plus J4 wrist roll and the gripper.

### Keyboard

| Key | Action |
|-----|--------|
| I/K | World +Z / -Z |
| S/W | World +Y / -Y |
| A/D | World +X / -X |
| J/L | J4 wrist roll (also while moving XYZ) |
| Q/E | Gripper sweep open / close (S120–S285; release = stop) |
| -/= | Keyboard jog speed slower / faster (live; does not apply to stick throw) |
| H | Home (on-device) |
| SPACE | Status |
| 0 | Stop, drivers off |
| ESC | Quit |

### Xbox controller

Player 1, via Windows XInput; works without terminal focus.

| Control | Action |
|---------|--------|
| Left stick | World X / Y |
| Right stick Y | World Z |
| Right stick X | J4 wrist roll (also while moving XYZ) |
| Stick throw | Jog speed from max active stick (full throw = 700 µs; ephemeral, not keyboard setting) |
| LT / RT | Gripper open / close |
| Y short / long (>500 ms) | Goto / store TCP x,y,z + J4 (max speed; gripper unchanged) |
| A | Home |
| B | Stop, drivers off |
| X | Status |
| Back | Quit |

Use `--no-gamepad` for keyboard only; `--gamepad-deadzone` adjusts the stick
deadzone (default 9000).

### Behavior notes

- `--no-orient` disables J4 wrist unwind during Cartesian moves (also
  live-toggleable via serial `orient on|off`). When on, J4 counters J1's yaw
  1:1 so the gripper holds its world-frame orientation.
- Gripper and J4-roll commands resend on a ~50 ms timer while their key is
  held, so a single dropped serial line can't strand them mid-motion — the same
  fix applied to Cartesian jog's `cj` resend.

## Firmware

[firmware/mt4_jog/](firmware/mt4_jog/) is a 4-axis step/dir jog engine with an
on-device world-frame resolved-rate jog, closed-form IK for straight-line
moves, and a queued waypoint path. Build/flash with PlatformIO via
`flash_jog.py`.

### Serial protocol

Full reference lives in the header comment of
[firmware/mt4_jog/src/main.cpp](firmware/mt4_jog/src/main.cpp). Summary:

| Command | Effect |
|---------|--------|
| `cj +x\|-x\|+y\|-y\|+z\|-z\|<dx> <dy> <dz> [j4]` | Cartesian jog. Optional J4 roll `-1\|0\|1` layers onto the solved rates so the wrist rotates during the move; zero direction + nonzero j4 = pure wrist roll. At the keep-out cylinder the inward component is clamped so the jog slides along the boundary |
| `orient on\|off` | J4 wrist unwind during Cartesian moves |
| `speed <us>` | Live jog step period, 700–4000 µs (session state) |
| `pos` | Joint steps + derived TCP mm, world-frame J4 deg, gripper S, move speed |
| `setpos <j1> <j2> <j3> <j4>` | Overwrite step counters |
| `j4zero` | Rewrite J4 steps so the current pose reports world J4 = 0 (no motion; survives home, lost on power cycle) |
| `m <dj1> <dj2> <dj3> <dj4> [dg]` | Bounded relative move; all axes finish together |
| `mp <x> <y> <z> <j4\|h\|w> <g> [speed_us]` | Absolute move: TCP position (mm) + gripper S + optional step period. XYZ interpolated along straight world-frame lines in short segments with closed-form IK per segment. The J4 field is a world-frame yaw in degrees, or a sentinel resolved at leg-plan time: `h` holds the world yaw the arm has when the leg is planned, `w` holds the J4 *joint* angle across the leg's J1 swing (what big base swings need). Rejected with `err not homed` unless homed this session |
| `mq <x> <y> <z> <j4\|h\|w> <g> [speed_us] [dwell_ms]` | Queued absolute move: same args and full keep-out/soft-limit validation as `mp`. If idle, behaves like `mp` (cold start). If a move is already executing, the waypoint is appended to a pending queue (depth 8; `err mq full N` beyond that) and spliced in without stopping when the running leg finishes — no per-waypoint stop/settle/reaccel or serial round trip. `dwell_ms > 0` makes the entry a **grip station** instead of a leg: no motion, the gripper is driven to `<g>`, and the queue holds until the sweep finishes plus `dwell_ms` of settle — this is what lets a whole pick-and-place be one queue |
| `home [j1 j2]` | On-device homing (see below) |
| `g o\|c\|stop\|<120-285>` | Gripper open / close / stop / set; bare `g` queries |
| `?` / `s` | Status |

Keep-out, soft joint limits and the ground plane reject or clamp `cj` and
`mp`/`mq` targets; `mp` also routes paths that would cross the keep-out
cylinder via tangent-arc-tangent. Joint-space moves (`m`, homing) command raw
steps and are **not** covered.

### Kinematics and calibration

The MT4 is a parallel-link (palletizing) arm: J2 sets the upper-arm absolute
angle, J3 sets the forearm absolute angle through the link rods (independent of
J2), and the head platform stays level. The model uses EEPROM link/offset
geometry (L1 130, L2 150, base 45/140, head 35/14.43).

The post-home park pose is **q2 = 107.0°, q3 = −9.3°** at step counters
**(0, j2_pull, j3_pull, ·)** — tape-fit 2026-07-21 from measured home TCP
(shoulder 140 mm, wrist 240 mm, pads ≈226 mm, radial ≈190 mm). J2/J3 model
angles are zeroed at the **limit/interference reference** (≈135.57° / −23.59°),
so changing pull-off does not invalidate that fit. FK at the park pose reports
**(190.0, 0, 225.6)**. Soft desk floor `GROUND_Z_MM` is **115**. The J1
keep-out cylinder is **140 mm**.

Per-joint steps/deg are from direct measurement (phone clinometer for J2–J4,
direct yaw for J1): J1/J2/J3 = 35, J4 = 45 — still a z-walk co-candidate if
J2/J3 ratios differ. `MT4_STEPS_PER_DEG` / `J_STEP_SIGN` / homes are duplicated
in [firmware/mt4_jog/src/kinematics.h](firmware/mt4_jog/src/kinematics.h),
[mt4_jog/joints.py](mt4_jog/joints.py) and
[mt4_jog/kinematics.py](mt4_jog/kinematics.py) — edit all three together,
flash, then re-run `calibrate_vision.py` / `calibrate_height.py` (and
`calibrate_j4.py` after power cycle).

### Homing

Homing seeks J1/J2's limit switches directly. J3 has no switch of its own, so
it's homed indirectly by driving it into mechanical interference with J2 until
that displaces J2 enough to release J2's own limit switch. Defaults: J1 center
**4580** steps, J2 pull-off **1000**, J3 pull-off **500** (override J1/J2 with
`--j1-center` / `--j2-pull` on the clients). J4's counter is preserved across
the J1–J3 rewrite, so a `j4zero` calibration survives homing.

### Pin map

| Joint | G-code | Drive | DIR | Limit |
|-------|--------|-------|-----|-------|
| J1 base | X | D23 | D22 | I21 |
| J2 shoulder | Y | D25 | D24 | I20 |
| J3 elbow | Z | D27 | D26 | — |
| J4 wrist | A | D35 | D36 | — |

Shared enable: **D40** (active low). Gripper PWM: **D7** (Timer4 OC4B); limits
and sweep run on the MCU (S120–S285).

Full hardware detail (board, drivers, flash path) is in
[docs/MT4_ARCHITECTURE.md](docs/MT4_ARCHITECTURE.md).

## Repo layout

| Path | Purpose |
|------|---------|
| [firmware/mt4_jog/](firmware/mt4_jog/) | Custom Arduino firmware: `config`/`pins`/`gripper`/`dda`/`motion`/`homing`/`commands`/`kinematics` |
| [mt4_jog/](mt4_jog/) | Python client library: serial protocol, joint map, kinematics, gamepad |
| [mt4_vision/](mt4_vision/) | Vision + motion: calibration, detection, entity table, grounding, VLM client, grasp/place primitives, path planning, preview |
| [mt4_mcp/](mt4_mcp/) | MCP server (HTTP or stdio) + OAuth |
| [services/grounding_dino/](services/grounding_dino/) | Grounding DINO GPU service (deployed to a separate host) |
| [services/qwen3_vl/](services/qwen3_vl/) | Qwen3-VL GPU service (deployed to a separate host) |
| [scripts/](scripts/) | Diagnostics (`diagnose_pick_accuracy.py`, `validate_scene_live.py`), ngrok + grounding/Qwen tunnel launchers |
| [tests/](tests/) | Unit tests |
| [docs/](docs/) | Calibration guide, hardware reference, assumption audit, Grounding DINO and Qwen3-VL setup, OAuth setup, printable ArUco sheet |
| [backups/](backups/) | Stock flash/EEPROM images and archived calibrations |

Key `mt4_vision` modules: `calib` (calibration + pixel↔robot transforms),
`detect`/`scene` (cube detection), `entities` (the addressable snapshot),
`locate`/`grounding` (non-cube objects), `qwen` (VLM client),
`instruct`/`instruct_reply`/`instruct_view`/`instruct_worker` (the English
instruction loop behind `ask_qwen.py`), `grasp`/`wrist` (grasp geometry and J4
angles), `motion`/`pickplace` (grasp and place primitives),
`stackpath`/`landing`/`workspace` (path and site planning), `table_fit` (the
table-plane homography fit), `policy`/`shuffle` (the shuffle planner and loop),
`preview` (annotated overlay), `console` (bottom-pinned interactive UI).

## Tests

```powershell
python -m pytest        # no hardware required
```

## Restoring stock firmware

Original factory images are in [backups/](backups/):

```powershell
python restore_stock.py --port COM6 --yes
```

Optional EEPROM restore:

```powershell
python restore_stock.py --port COM6 --yes --eeprom backups/mt4_eeprom_2026-07-02.hex
# or directly:
avrdude -p atmega2560 -c wiring -P COM6 -b 115200 -U eeprom:w:backups\mt4_eeprom_2026-07-02.hex:i
```

## Further docs

| Doc | Contents |
|-----|----------|
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | Step-by-step calibration guide: the six layers, each script's procedure and output, re-calibration decision matrix, troubleshooting, field reference |
| [docs/MT4_ARCHITECTURE.md](docs/MT4_ARCHITECTURE.md) | Hardware and pin-map reference, ATmega2560 flash path |
| [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | Assumption audit for pick/place/stacking accuracy — what each layer relies on, how well it's known, and where accuracy actually leaks |
| [docs/OAUTH_CHATGPT.md](docs/OAUTH_CHATGPT.md) | OAuth 2.1 via Google + ngrok for public MCP access |
| [docs/GROUNDING_DINO.md](docs/GROUNDING_DINO.md) | Grounding DINO server setup: GPU-host install, WSL2 prerequisites, systemd unit, SSH tunnel, HTTP API, troubleshooting |
| [services/grounding_dino/README.md](services/grounding_dino/README.md) | What the deployed service files are, and the day-to-day detect commands |
| [docs/SAM2.md](docs/SAM2.md) | SAM 2.1 segmentation service: install, systemd unit, SSH tunnel, HTTP API, the measured fp16/compile/embedding-cache choices, troubleshooting |
| [services/sam2/README.md](services/sam2/README.md) | What the deployed SAM 2.1 files are, and the day-to-day segment commands |
| [docs/QWEN3-VL.md](docs/QWEN3-VL.md) | Qwen3-VL service: start/stop, HTTP API, SSH tunnel, the `ask_qwen.py` harness, measured coordinate space and accuracy |
| [docs/qwen3_vl_mt4_repository_mapped_policy.md](docs/qwen3_vl_mt4_repository_mapped_policy.md) | Design: how VLM instruction-following maps onto this repo's geometry, measurement and safety layers |
| [docs/qwen3_vl_policy_status.md](docs/qwen3_vl_policy_status.md) | Build ledger for that design, 2026-08-02 … 08-03: what was measured on hardware, what failed, what was decided. History — the code is the authority on current behaviour |
| [docs/ArUco Markers A4 5x5cm.pdf](docs/ArUco%20Markers%20A4%205x5cm.pdf) | Printable marker sheet (DICT_4X4_50) |
| [firmware/mt4_jog/src/main.cpp](firmware/mt4_jog/src/main.cpp) | Full serial protocol reference (header comment) |
| [CLAUDE.md](CLAUDE.md) | Agent instructions: hardware autonomy, primary tools, typical failure patterns |
