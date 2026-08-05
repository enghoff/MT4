#!/usr/bin/env python3
"""Type an instruction in English; the arm carries it out.

    python ask_qwen.py                       # interactive prompt + window
    python ask_qwen.py "put the red cube on marker 3"
    python ask_qwen.py "find all the pickable objects"
    python ask_qwen.py --dry-run "pick up the green cube"
    python ask_qwen.py --no-preview "..."    # no window (headless, CI)

Interactive is the default: a prompt line pinned to the bottom of the terminal,
a background worker doing the slow thing, and a window that keeps redrawing
while it does.

Type an instruction and it starts. Type another while one is running and it
queues behind it. ``/`` commands (``/help``) inspect and steer without
disturbing the run: ``/stop`` ends it after the current step, ``/abort`` stops
the arm where it stands, ``/scene`` and ``/status`` ask the camera and the arm
what they currently see and where they are.

The window is split three ways. The left pane is the frame the model last
decided on with its own answer drawn over it (see
``mt4_vision.preview.annotate_qwen``); the right is a state panel; and the
corner inset is the live feed, so the desk stays visible while a multi-second
move plays out. The left pane is deliberately **not** live: those are the exact
pixels a decision was made from, and re-capturing them would quietly answer a
different question than the one worth asking. q or Esc in that window stops the
run; the arm is still parked on the way out.

Each step is one capture, one decision, one motion:

    park clear of the camera
        -> one frame -> pixel grid + the decoded ArUco ids drawn on it
        -> Qwen3-VL picks an action and draws a box round what it means
        -> GrabCut inside that box -> position, size and angle in millimetres
        -> the physical gates: reach, keep-out, ground, jaw clearance, desk
        -> motion.transfer / pick_at / place_at, off that same frame
        -> park clear of the camera again

then round again, until the model says DONE or anything refuses.

**Qwen is the eyes.** It is handed the frame and the ArUco tag numbers, and
nothing else. Not a list of cubes, not a list of objects, not a list of free
slots -- a tag is named only because its printed number is a code no
vision-language model can read off an image, and everything else on the desk is
the model's job to see. Nor is the instruction preprocessed: no noun
extraction, no stopword list, no question detection. What you type is what it
reads.

That is why a target is a **box** rather than an id. There is no list to hold
ids, so there is nothing to bind a point to, nothing to disagree about and
nothing to refuse over a name. The box goes straight to the segmenter that
turns pixels into millimetres.

**Every object is gripped at table height.** The jaws close at ``table_z``, as
low as they go, oriented by the GrabCut mask's long axis. Nothing on this desk
is taller than the jaws' vertical clearance, so the lowest grip is always
available and no pick has to know how tall a thing is to decide its Z.

*Where* the jaws go does need the height. The camera views the desk from a
steep angle, so a tall object images smeared outward and the middle of that
smear is not above its footprint -- 18.1-22.4mm outward on 20mm cubes, against
the ~10mm the jaws tolerate. The measurement infers the height from the
silhouette and unprojects; see ``locate._height_corrected``.

**Moving something somewhere is ONE step.** ``TRANSFER`` names the object and
its destination together and the arm carries out both halves without stopping,
because the intervening park-look-decide answered a question that has no
answer: whether the jaws actually have the thing. There is no sensor in them,
and the vision test that stood in for one was measured misreading an ArUco tag
beside a stapler as the stapler itself -- reporting a completed pick as a
failure. So nothing here is verified after the fact; outcome lines say
"commanded, not checked" and mean it. What IS still checked is everything
before the move: reach, keep-out, ground, finger clearance, the desk polygon,
and both ends of a transfer, all before the gripper opens.

**One frame, one decision, one motion.** The script owns the frame it decides
over and executes against, rather than deciding over its own and then calling
the MCP server's ``mt4_pick``, which would resolve a target in a *different*
capture. The arm is parked while the model thinks and nothing else on this desk
moves, so the measurement taken from the box is still true when the jaws
arrive.

**A question about the desk is answered in boxes, not in prose.** "Find all the
pickable objects" gets a ``REPORT``, and its list comes from a second call on
the same frame whose only job is to enumerate. Every box it returns is segmented
and put through the same gates a pick goes through -- reach, the keep-out,
ground, the jaw-width plan, the desk polygon. Each object comes back as a
numbered row with its position, the width the jaws will close across, and either
"pickable" or the gate that stopped it, with the same numbers drawn on the frame
in green and red. Nothing moves.

Those rows are then handed to the next step, so a task can look before it acts
-- "find the cubes you can reach and move them" is one instruction, not two.
This is the only place a gate's verdict travels back to the model; everywhere
else the gates answer the operator and the model decides again blind. A report
that was the whole answer says so and stops there.

**A refused step is told to the next one.** Reach, the keep-out, ground, jaw
width and the desk edge all speak in the words of the gate that refused, and
that sentence goes into the same history. Without it the loop re-decides as
though nothing had been refused, which is how one out-of-reach blob came to be
chosen twice running.

Enumeration needs its own call because the decision prompt cannot do both jobs.
Measured on a nine-cube desk: asked for a list alongside its usual single-target
fields it returned **1 of 9** objects, having boxed one cube and copied it into
the list; the enumeration prompt returns **9 of 9**. Reordering the decision
schema to fix the count instead sent a transfer's *destination* into the pick
field, which is a silently wrong target -- so the two stay apart.

The model is told plainly not to decide pickability for itself, because it
cannot see it: what the arm can reach, how close to its own base it can work,
where the desk ends and how wide the jaws open are facts about the machine, not
about the photograph. So it lists every candidate and the gates rule out the
ones that fail. An object left out because it looked hard to grasp would be
missing from the answer with nothing able to notice.

**What the gripper holds outlives an instruction.** The jaws are physical and
one line of typing does not empty them, so ``held`` is session state, not
per-task: stop a transfer halfway and the next instruction is told the arm is
still carrying something. Nothing can sense this, so ``/held`` exists to
correct it by hand and ``/open`` to let go.

Nothing here computes a coordinate or a wrist angle. Target selection is the
only thing the model does; every millimetre and every safety gate is the
existing stack's.

Prereqs:
  * the Qwen3-VL service reachable (``.\\scripts\\start_qwen_tunnel.ps1``)
  * arm free on the serial port, vision calibration present
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision import instruct as I
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, CameraError, FrameStream
from mt4_vision.console import BottomUI
from mt4_vision.instruct_view import MAX_STEPS, RunPreview
from mt4_vision.instruct_worker import PlainUI, TaskWorker, save_record
from mt4_vision.pickplace import ensure_homed, home_arm, retreat_for_camera
from mt4_vision.preview import gui_available
from mt4_vision.qwen import DEFAULT_URL, QwenError, health

HELP = """commands (anything else is run as an instruction):
  <text>          carry it out now, or queue it behind what is running
  /help           this list
  /stop           finish the current step, then stop; drops the queue
  /abort          /stop, and interrupt the move in flight -- the arm halts
                    wherever it is, jaws as they are
  /cancel         drop queued instructions; the running one continues
  /again          re-run the last instruction
  /steps N        give up after N steps per instruction (default 6)
  /dry [on|off]   decide and validate, never move
  /scene          what the camera sees now: ids, labels, why each is blocked
  /status         arm status: homed, TCP, gripper
  /park           retreat to the camera park pose
  /home           home the arm
  /open           open the jaws (what /abort mid-carry leaves you needing)
  /held [thing]   what the loop believes is in the jaws; /held none to clear
  /save [name]    write the frame the model saw, the window, and a JSON record
  /health         re-query the Qwen service
  Ctrl+C / Ctrl+Z quit"""


# --------------------------------------------------------------------------- #
# commands


def handle_command(
    line: str,
    *,
    worker: TaskWorker,
    ui: BottomUI | PlainUI,
    view: RunPreview | None,
    client: Mt4Client | None,
    camera: int,
    url: str,
    outdir: Path,
) -> bool:
    """Run a ``/command``. Returns False if the line was not a command."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    rest = parts[1:]

    if cmd in ("help", "h", "?"):
        ui.emit("")
        for row in HELP.split("\n"):
            ui.emit(row)
        ui.set_status("ready")
    elif cmd in ("stop", "abort"):
        abort = cmd == "abort"
        running = worker.request_stop(abort=abort)
        if not running:
            ui.set_status("nothing was running")
        elif abort:
            ui.set_status("aborting -- the arm halts where it is, jaws unchanged")
        else:
            ui.set_status("stopping after the current step")
    elif cmd == "cancel":
        dropped = worker.cancel_queued()
        ui.set_status(
            f"dropped {dropped} queued instruction(s)" if dropped
            else "nothing queued (the running one needs /stop or /abort)"
        )
    elif cmd == "again":
        previous = worker.last_instruction()
        if not previous:
            ui.set_status("nothing run yet")
        else:
            ui.emit(f"  again: {previous}")
            worker.submit(previous)
    elif cmd == "steps":
        try:
            steps = max(1, int(rest[0]))
        except (IndexError, ValueError):
            ui.set_status("usage: /steps N")
        else:
            worker.set_max_steps(steps)
            ui.set_status(f"giving up after {steps} step(s) per instruction")
    elif cmd == "dry":
        want = (rest[0].lower() if rest else "")
        state, _view = worker.snapshot()
        if want in ("on", "yes", "1"):
            dry = True
        elif want in ("off", "no", "0"):
            dry = False
        else:
            dry = not state.dry_run
        if not dry and client is None:
            ui.set_status("no arm on this run -- dry-run cannot be turned off")
            return True
        worker.set_dry_run(dry)
        ui.set_status(
            "dry run: deciding, never moving" if dry else "live: the arm will move"
        )
    elif cmd == "scene":
        # Queued like any other camera work: the worker owns the frame stream's
        # "look now" calls, and reading the desk mid-transfer would show the
        # arm across it anyway.
        def look() -> None:
            obs = I.observe(camera, frame=worker.frame(), token="scene")
            ui.emit("")
            ui.emit(f"  {obs.snapshot.summary}")
            for row in obs.snapshot.table():
                ui.emit(f"  {row}")
            ui.set_status(obs.snapshot.summary)

        worker.submit_chore("looking at the desk", look)
    elif cmd == "status":
        if client is None:
            ui.set_status("no arm on this run (--dry-run)")
            return True

        def report() -> None:
            status = client.get_status()
            tcp = status.tcp
            where = (
                "no TCP" if tcp is None
                else f"tcp ({tcp.x:.0f}, {tcp.y:.0f}, {tcp.z:.0f}) "
                     f"j4 {tcp.j4:.0f} grip {tcp.grip}"
            )
            line = f"homed {status.homed}  {where}"
            ui.emit(f"  {line}")
            ui.set_status(line)

        worker.submit_chore("reading arm status", report)
    elif cmd == "park":
        if client is None:
            ui.set_status("no arm on this run (--dry-run)")
            return True
        worker.submit_chore(
            "parking", lambda: retreat_for_camera(client, I.load_calibration())
        )
    elif cmd == "home":
        if client is None:
            ui.set_status("no arm on this run (--dry-run)")
            return True

        def do_home() -> None:
            home_arm(client)
            retreat_for_camera(client, I.load_calibration())
            ui.emit("  homed and parked")
            ui.set_status("homed and parked")

        worker.submit_chore("homing", do_home)
    elif cmd == "open":
        if client is None:
            ui.set_status("no arm on this run (--dry-run)")
            return True

        def release() -> None:
            calib = I.load_calibration()
            client.gripper(calib.grip_open_s)
            # The jaws are open, so whatever the loop thought it was carrying
            # is on the desk (or the floor). Saying so beats leaving the model
            # to be told about an object the arm no longer has.
            worker.set_held(None)
            ui.emit("  jaws open -- nothing is held now")
            ui.set_status("jaws open")

        worker.submit_chore("opening the jaws", release)
    elif cmd == "held":
        want = " ".join(rest).strip()
        if not want:
            held = worker.held()
            ui.set_status(
                f"the loop believes the jaws hold: {held}" if held
                else "the loop believes the jaws are empty"
            )
        elif want.lower() in ("none", "nothing", "empty", "-"):
            worker.set_held(None)
            ui.set_status("noted: the jaws are empty")
        else:
            worker.set_held(want)
            ui.set_status(f"noted: the jaws hold a {want}")
    elif cmd == "save":
        decision = worker.last_decision()
        if decision is None:
            ui.set_status("nothing decided yet")
        else:
            canvas = view.latest_canvas() if view is not None else None
            path = save_record(outdir, decision, canvas, " ".join(rest))
            ui.set_status(f"saved {path}")
    elif cmd == "health":
        try:
            info = health(url)
        except QwenError as exc:
            ui.set_status(str(exc))
        else:
            ui.emit(f"  health: {json.dumps(info)}")
            ui.set_status("service ok")
    else:
        ui.set_status(f"unknown command /{cmd} -- try /help")
    return True


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instruction", nargs="*", help="what to do; omit for interactive")
    ap.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    ap.add_argument("--url", default=DEFAULT_URL, help="Qwen3-VL service base URL")
    ap.add_argument("--dry-run", action="store_true", help="decide, never move")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--save-view", default=None, help="write the frame the model saw")
    ap.add_argument("--outdir", default="instruction_runs", help="where /save writes")
    ap.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True,
        help="live window: the frame the model saw with its answer drawn on it, "
             "a state panel and a live inset; q or Esc stops the run (default: on)",
    )
    args = ap.parse_args()

    try:
        info = health(args.url)
    except QwenError as exc:
        print(f"qwen service: {exc}", file=sys.stderr)
        return 1
    svc = f"{info.get('model', '?')}  {info.get('device', '?')}/{info.get('quantization', '?')}"
    if not info.get("loaded", False):
        print("note: model still loading; the first decision will be slow", file=sys.stderr)

    client = None
    if not args.dry_run:
        try:
            client = Mt4Client()
            # Once per session. Homing every move is what wears the rig.
            ensure_homed(client)
        except Mt4ClientError as exc:
            print(f"arm: {exc}", file=sys.stderr)
            return 1

    # On by default, because the question this window answers -- "did the model
    # point at the thing it named" -- is one you want asked on every run, not
    # only the runs where you already suspected something.
    #
    # Gated on a real GUI rather than left to LivePreview's own fallback: that
    # fallback opens a system image viewer per update, which is a reasonable
    # last resort for someone who typed --preview and unusable as a default.
    want_preview = args.preview
    if want_preview and not gui_available():
        print(
            "no OpenCV GUI here, so no preview window (is "
            "opencv-python-headless installed?). --save-view still writes "
            "the frame the model saw.",
            file=sys.stderr,
        )
        want_preview = False

    # Held open for the whole session, and shared. Only one consumer can hold
    # the capture device, so the window and the decision captures have to come
    # off the same stream -- and it is faster besides: a per-step reopen costs
    # 2-3s of exposure warm-up that a continuously drained stream pays once.
    #
    # Opened before the pinned prompt takes over the terminal, so a camera that
    # will not open reports itself as an ordinary error message rather than
    # into a two-line footer that is about to be torn down.
    try:
        stream = FrameStream(args.camera)
    except CameraError as exc:
        print(f"camera: {exc}", file=sys.stderr)
        return 1

    interactive = not args.instruction
    ui: BottomUI | PlainUI = BottomUI("instruction") if interactive else PlainUI()
    worker: TaskWorker | None = None
    view: RunPreview | None = None
    try:
        worker = TaskWorker(
            ui=ui, stream=stream, camera=args.camera, client=client,
            max_steps=args.max_steps, save_view=args.save_view,
            dry_run=args.dry_run,
        )
        if want_preview:
            view = RunPreview(stream, worker, svc=svc, camera=args.camera, ui=ui)

        if not interactive:
            worker.submit(" ".join(args.instruction))
            worker.drain()
            return 0 if worker.last_result() else 2

        ui.emit(f"mt4 instruction loop -- {svc}")
        ui.emit("type what to do, or /help for commands")
        if args.dry_run:
            ui.emit("  dry run: every decision is made and validated, nothing moves")
        ui.set_status("ready")
        while True:
            if view is not None and view.stopped_by_user():
                break
            line = ui.read_line()
            if line is None:
                break
            if not line:
                continue
            if line in ("quit", "exit"):
                break
            if handle_command(
                line, worker=worker, ui=ui, view=view, client=client,
                camera=args.camera, url=args.url, outdir=Path(args.outdir),
            ):
                continue
            worker.submit(line)
        # Clean exit: let an instruction already moving the arm finish rather
        # than closing the serial port out from under it.
        if worker.busy():
            ui.set_status("finishing the current instruction before exiting...")
            worker.drain()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if worker is not None:
            worker.close()
        ui.close()
        if view is not None:
            view.close()
        stream.close()
        if client is not None:
            try:
                retreat_for_camera(client, I.load_calibration())
            except Mt4ClientError:
                pass
            client.close()


if __name__ == "__main__":
    code = main()
    # Hard exit after main()'s finally has already parked the arm, closed the
    # serial port, the window and the camera. This script holds a FrameStream
    # for the whole session, and OpenCV highgui can keep a Windows process
    # alive past the end of main -- a lingering process holds the capture
    # device open, stranding the camera for every other vision script in the
    # repo until it is killed by hand.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
