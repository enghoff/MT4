#!/usr/bin/env python3
"""Interactive Qwen3-VL harness: ask questions about what the camera sees.

Same shape as ``move_object_to_marker.py`` -- a pinned prompt line, a
background worker, a live annotated preview -- but the worker asks a
vision-language model instead of moving the arm, and the answer is text
rather than a box. Nothing here touches the arm unless you pass ``--park``.

The window is split: the submitted frame on the left (with any boxes/points
Qwen named drawn on it), an answer panel on the right, and a small live inset
so you can still aim the camera while an answer is on screen. What you see on
the left is always *the exact image that was POSTed* -- which is the point.
A wrong answer about a frame the arm was blocking is a different problem from
a wrong answer about a clean frame, and no amount of reading the text alone
tells you which happened.

Coordinates come back in whichever space this model build uses (see
``parse_regions``); ``/coords`` flips the interpretation so you can settle it
by eye in two keystrokes rather than guessing.

It is a monitor by default: the desk is watched from startup, and anything
that moves gets boxed and described (JSON, one box per object) until you type
something, which replaces that as the standing question. ``--no-watch`` makes
it request/response only.

Prereqs:
  * The service running, and ``.\\scripts\\start_qwen_tunnel.ps1`` if remote
  * A camera. No calibration, no serial, no arm (unless ``--park``)

Example::

    python ask_qwen.py --camera 1
    python ask_qwen.py --camera 1 --no-watch
    python ask_qwen.py --prompt "how many cubes are on the desk?" --save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from mt4_vision.camera import DEFAULT_CAMERA_INDEX, CameraError, FrameStream, capture_frame
from mt4_vision.console import BottomUI
from mt4_vision.preview import LivePreview, PreviewStopped, annotate_for_pointing
from mt4_vision.qwen import DEFAULT_URL, QwenError, ask, health, parse_regions
from mt4_vision.qwen_panel import (
    Answer,
    DEFAULT_TOKENS,
    Options,
    SEND_MODES,
    compose,
    draw_regions,
    montage,
)
from mt4_vision.qwen_prompts import IDENTIFY_PROMPT, OBJECTS_PROMPT, PRESETS, TRACK_PROMPT
from mt4_vision.qwen_watch import (
    DEFAULT_WATCH_QUESTION,
    MOTION_THRESHOLD,
    MotionWatcher,
)
from mt4_vision.qwen_worker import QwenWorker, format_region

HELP = """commands (anything else is asked verbatim):
  <text>            ask it now, and keep answering it on every scene change
  <Enter>           back to the live feed
  /help             this list
  /freeze  /thaw    pin the current frame, so repeat asks use one identical image
  /again            re-ask the last question (on the pinned frame if frozen)
  /cancel           drop queued questions that have not started yet
  /grid             toggle the labelled pixel grid drawn on the sent image
  /coords abs|norm  reinterpret returned coords as pixels or 0-1000 normalized
  /tokens N         max_new_tokens (default 700)
  /frames N [gap]   capture N frames, gap seconds apart (N=1 for one still)
  /mode M           how they reach the model: single|montage|images|video
                      images = N separate images; direction right, ~2x tokens
                      video  = one sequence; change right, direction not, ~1x
                      montage= tiled into one picture; needs no server support
  /sample           toggle greedy (default) vs sampling at temperature 0.7
  /objects [watch]  JSON list of every object with boxes ("identify all objects"
                      alone returns prose -- the schema has to be named).
                      Add 'watch' to re-list on every change
  /identify <what> [watch]  JSON list of <what> with boxes AND a one-sentence
                      description each, e.g. /identify paintings. Add 'watch'
                      to re-identify on every change
  /track <object>   locate that object on each NEW frame after movement, box
                      drawn on it
  /watch            ON BY DEFAULT. Anything you type becomes the standing
                      question, re-asked on the NEW frame after every change.
                      /watch off to stop; /watch <q> uses the before/after
                      PAIR instead, for explicitly comparative questions
  /once <question>  ask without changing what the watcher is watching for
  /sens X           motion trigger threshold (default 0.0005; noise floor is
                      0.00016, a 25px object moving is ~0.0014)
  /noimage          ask the next question text-only (no frame)
  /preset [N]       list capability probes, or run one
  /save [name]      write the sent image, the view and a JSON record
  /health           re-query the service
  Ctrl+C / Ctrl+Z   quit"""


# --------------------------------------------------------------------------- #
# worker


class HarnessPreview:
    """Composites the live feed, the submitted frame and the answer panel."""

    def __init__(
        self, stream: FrameStream, worker: QwenWorker, *, svc: str, ui: BottomUI,
        watcher: MotionWatcher | None = None,
    ) -> None:
        self._stream = stream
        self._worker = worker
        self._svc = svc
        self._ui = ui
        self._watcher = watcher
        self._draw_errors: set[str] = set()
        self._preview = LivePreview("qwen probe (q or Esc to stop)")
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._user_quit = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qwen-view", daemon=True)
        self._thread.start()

    def stopped_by_user(self) -> bool:
        return self._user_quit.is_set()

    def latest_canvas(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                live = self._stream.fresh(min_advance=1)
            except CameraError:
                time.sleep(0.2)
                continue
            answer, pending, elapsed = self._worker.snapshot()
            watch = self._watcher.snapshot() if self._watcher is not None else None
            try:
                canvas = compose(
                    live, answer=answer, pending=pending, elapsed=elapsed,
                    opts=self._worker.opts, svc=self._svc, watch=watch,
                )
            except Exception as exc:  # noqa: BLE001 -- a draw bug must not kill the feed
                # ...but it must not be invisible either. Swallowing this
                # silently is how a compose() crash presented as "the preview
                # feature just doesn't work": no window, no error, nothing to
                # search for. Report the first one (and each new kind after it)
                # while still keeping the feed alive.
                sig = f"{type(exc).__name__}: {exc}"
                if sig not in self._draw_errors:
                    self._draw_errors.add(sig)
                    self._ui.emit(f"  preview draw failed -- {sig}")
                time.sleep(0.2)
                continue
            with self._lock:
                self._latest = canvas
            try:
                self._preview.show(canvas)
            except PreviewStopped:
                self._user_quit.set()
                break

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._preview.close()


# --------------------------------------------------------------------------- #
# commands


def save_probe(
    outdir: Path, answer: Answer, canvas: np.ndarray | None, name: str = "",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S")
    if name:
        stem = f"{stem}_{name}"
    cv2.imwrite(str(outdir / f"{stem}_sent.jpg"), answer.sent)
    if canvas is not None:
        cv2.imwrite(str(outdir / f"{stem}_view.jpg"), canvas)
    record = {
        "prompt": answer.prompt,
        "sent_prompt": answer.sent_prompt,
        "response": answer.text,
        "error": answer.error,
        "latency_s": round(answer.latency_s, 3),
        "max_new_tokens": answer.tokens,
        "had_image": answer.had_image,
        "sent_size": [answer.sent.shape[1], answer.sent.shape[0]],
        "send_mode": answer.send_mode,
        "sent_label": answer.sent_label(),
        "service": answer.reply.raw if answer.reply is not None else None,
        "regions": [
            {
                "label": r.label, "kind": r.kind, "coords": list(r.coords),
                **({"description": r.description} if r.description else {}),
            }
            for r in answer.regions
        ],
    }
    path = outdir / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def handle_command(
    line: str, *, worker: QwenWorker, ui: BottomUI, view: HarnessPreview | None,
    url: str, outdir: Path, watcher: MotionWatcher | None = None,
) -> bool:
    """Run a ``/command``. Returns False if the line was not a command."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    rest = parts[1:]
    opts = worker.opts

    if cmd in ("help", "h", "?"):
        ui.emit("")
        for row in HELP.split("\n"):
            ui.emit(row)
        ui.set_status("ready")
    elif cmd == "freeze":
        if opts.frames > 1:
            ui.set_status("/frames is >1 -- /frames 1 first (a pinned frame is not a sequence)")
            return True
        try:
            frame = worker.grab()
        except CameraError as exc:
            ui.set_status(f"camera: {exc}")
            return True
        opts.pinned = frame
        ui.set_status("frame frozen -- asks now reuse this exact image (/thaw to release)")
    elif cmd == "thaw":
        opts.pinned = None
        ui.set_status("thawed -- asks capture a fresh frame")
    elif cmd == "again":
        prev = worker.last_submitted()
        if prev is None:
            ui.set_status("nothing asked yet")
        else:
            worker.ask(prev[0], with_image=prev[1])
    elif cmd == "cancel":
        dropped = worker.cancel_queued()
        ui.set_status(
            f"dropped {dropped} queued question(s)" if dropped
            else "nothing queued (the one in flight cannot be recalled)"
        )
    elif cmd == "grid":
        opts.grid = not opts.grid
        ui.set_status(f"grid {'on' if opts.grid else 'off'} (redrawn on the next ask)")
    elif cmd == "coords":
        mode = (rest[0].lower() if rest else "")
        if mode.startswith("n"):
            opts.coord_mode = "norm"
        elif mode.startswith("a"):
            opts.coord_mode = "abs"
        else:
            opts.coord_mode = "norm" if opts.coord_mode == "abs" else "abs"
        ui.set_status(f"coords read as {opts.coord_mode}")
    elif cmd == "tokens":
        try:
            opts.tokens = max(1, int(rest[0]))
        except (IndexError, ValueError):
            ui.set_status("usage: /tokens N")
        else:
            ui.set_status(f"max_new_tokens = {opts.tokens}")
    elif cmd == "frames":
        try:
            want = max(1, min(16, int(rest[0])))
            gap = max(0.05, float(rest[1])) if len(rest) > 1 else opts.frame_gap_s
        except (IndexError, ValueError):
            ui.set_status("usage: /frames N [seconds-between]")
            return True
        # A frozen frame would be repeated and answered as if it were a
        # sequence -- a wrong answer with no visible cause.
        if want > 1 and opts.pinned is not None:
            ui.set_status("frame is frozen -- /thaw first, a sequence needs live frames")
            return True
        opts.frames, opts.frame_gap_s = want, gap
        if want > 1 and opts.send_mode == "single":
            opts.send_mode = "images"
        ui.set_status(
            f"{want} frames {gap:g}s apart, sent as {opts.send_mode} "
            f"({opts.fps:g} fps)" if want > 1 else "single frame"
        )
    elif cmd == "mode":
        want = (rest[0].lower() if rest else "")
        match = [m for m in SEND_MODES if m.startswith(want)] if want else []
        if len(match) != 1:
            ui.set_status(f"usage: /mode {'|'.join(SEND_MODES)}")
            return True
        opts.send_mode = match[0]
        if opts.send_mode != "single" and opts.frames < 2:
            opts.frames = 4
        ui.set_status(
            f"mode {opts.send_mode}"
            + (f", {opts.frames} frames {opts.frame_gap_s:g}s apart"
               if opts.send_mode != "single" else "")
        )
    elif cmd == "watch":
        if watcher is None:
            ui.set_status("no watcher (needs the camera stream)")
        elif rest and rest[0].lower() in ("off", "stop", "no"):
            was = watcher.armed
            _st, _sc, events, skipped, *_ = watcher.snapshot()
            watcher.disarm()
            ui.set_status(
                f"watch off after {events} event(s)"
                + (f", {skipped} skipped while busy" if skipped else "")
                if was else "watch was already off"
            )
        elif not rest:
            # Bare /watch reports when on and resumes when off -- it must not
            # be the way to turn monitoring off, now that it starts on.
            if watcher.armed:
                state, score, events, skipped, *_ = watcher.snapshot()
                ui.emit(f"  watching for: {watcher.last_question}")
                ui.set_status(
                    f"{state}, score {score:.5f}, {events} event(s)"
                    + (f", {skipped} skipped" if skipped else "")
                    + f", threshold {watcher.threshold:.5f}"
                )
            else:
                # Reuse whatever mode was last active -- disarm() leaves
                # ``send`` untouched precisely so a bare resume doesn't
                # silently downgrade an explicit /watch <q> pair back to latest.
                watcher.arm(watcher.last_question, send=watcher.send)
                ui.emit(f"  watching for: {watcher.last_question}")
                ui.set_status("watching again (/watch off to stop)")
        elif opts.pinned is not None:
            ui.set_status("frame is frozen -- /thaw first, the watcher needs live frames")
        else:
            question = " ".join(rest)
            watcher.arm(question, send="pair")
            ui.emit(f"  watching for: {question}")
            ui.set_status(
                f"watching -- asks on motion above {watcher.threshold:.5f} (/watch off)"
            )
    elif cmd in ("objects", "obj"):
        # "identify all objects" gets prose; the schema has to be named. See
        # OBJECTS_PROMPT for the measurement behind that.
        standing = bool(rest) and rest[0].lower() in ("watch", "keep", "on")
        # ~35 tokens per boxed object, so a busy desk can still overrun a
        # lowered budget and cut the array mid-entry. The entries that
        # completed are recovered and the transcript flags the cut.
        if opts.tokens < 500:
            ui.emit(f"  note: /tokens {opts.tokens} may truncate the list; 600+ is safer")
        if standing and watcher is not None:
            watcher.arm(OBJECTS_PROMPT, send="latest")
            ui.set_status("re-listing objects on every change (/watch off to stop)")
        worker.ask(OBJECTS_PROMPT)
    elif cmd == "identify":
        standing = bool(rest) and rest[-1].lower() in ("watch", "keep", "on")
        obj = " ".join(rest[:-1] if standing else rest).strip()
        if not obj:
            ui.set_status("usage: /identify <object(s)> [watch], e.g. /identify paintings")
            return True
        if opts.tokens < 500:
            ui.emit(f"  note: /tokens {opts.tokens} may truncate the list; 600+ is safer")
        question = IDENTIFY_PROMPT.format(obj=obj)
        if standing and watcher is not None:
            watcher.arm(question, send="latest")
            ui.set_status(f"re-identifying {obj} on every change (/watch off to stop)")
        worker.ask(question)
        ui.emit(f"  identifying: {obj}")
    elif cmd == "track":
        if watcher is None:
            ui.set_status("no watcher (needs the camera stream)")
            return True
        obj = " ".join(rest)
        if not obj:
            ui.set_status("usage: /track <object>, e.g. /track stapler")
            return True
        if opts.pinned is not None:
            ui.set_status("frame is frozen -- /thaw first, tracking needs live frames")
            return True
        question = TRACK_PROMPT.format(obj=obj)
        watcher.arm(question, send="latest")
        ui.emit(f"  tracking: {obj} -- locating it on each new frame after movement")
        ui.set_status(f"tracking {obj} (motion > {watcher.threshold:.5f}); /watch off to stop")
        worker.ask(question)   # a box straight away, not only after the next move
    elif cmd == "once":
        question = " ".join(rest)
        if not question:
            ui.set_status("usage: /once <question>")
        else:
            worker.ask(question)
    elif cmd == "sens":
        if watcher is None:
            ui.set_status("no watcher (needs the camera stream)")
            return True
        try:
            watcher.threshold = max(0.0, float(rest[0]))
        except (IndexError, ValueError):
            _st, score, *_ = watcher.snapshot()
            ui.set_status(
                f"usage: /sens X (now {watcher.threshold:.5f}, live score {score:.5f})"
            )
        else:
            ui.set_status(f"motion threshold {watcher.threshold:.5f}")
    elif cmd == "sample":
        opts.greedy = not opts.greedy
        ui.set_status(
            "greedy (reproducible)" if opts.greedy
            else "sampling at the model's temperature 0.7 (answers will vary)"
        )
    elif cmd == "noimage":
        prev = worker.last_submitted()
        prompt = " ".join(rest) or (prev[0] if prev else "")
        if not prompt:
            ui.set_status("usage: /noimage <question>")
        else:
            worker.ask(prompt, with_image=False)
    elif cmd in ("preset", "p"):
        if not rest:
            ui.emit("")
            ui.emit("presets:")
            for i, (name, prompt) in enumerate(PRESETS, start=1):
                ui.emit(f"  {i:2d}. {name:<10} {prompt[:70]}")
            ui.set_status("run one with /preset N")
        else:
            try:
                name, prompt = PRESETS[int(rest[0]) - 1]
            except (ValueError, IndexError):
                ui.set_status(f"no such preset (1-{len(PRESETS)})")
            else:
                ui.set_status(f"preset {name}")
                worker.ask(prompt)
    elif cmd == "save":
        last = worker.last()
        if last is None:
            ui.set_status("nothing to save yet")
        else:
            canvas = view.latest_canvas() if view is not None else None
            path = save_probe(outdir, last, canvas, " ".join(rest))
            ui.set_status(f"saved {path}")
    elif cmd == "health":
        try:
            info = health(url)
        except QwenError as exc:
            ui.set_status(str(exc))
        else:
            ui.emit(f"  health: {json.dumps(info)}")
            ui.set_status("service ok")
    elif cmd in ("live", "clear"):
        worker.clear()
        ui.set_status("live")
    else:
        ui.set_status(f"unknown command /{cmd} -- try /help")
    return True


# --------------------------------------------------------------------------- #


def camera_hint() -> str:
    """Which indices can be opened at all, for when auto-detect gives up.

    ``--camera`` defaults to the shared ArUco auto-detect, which answers "which
    camera sees the calibrated desk" -- but this harness needs no calibration
    and no markers, and the tags are routinely occluded by the very objects
    being probed. So a failure here is usually "pick an index", not "fix the
    desk", and the useful reply is the list of indices that exist.
    """
    openable = []
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        try:
            if cap.isOpened() and cap.read()[0]:
                openable.append(i)
        finally:
            cap.release()
    if not openable:
        return "no camera index 0-5 could be opened at all"
    return f"openable camera indices: {openable} -- pass one with --camera"


def run_once(args: argparse.Namespace, svc: str) -> int:
    """Non-interactive single question, for scripting."""
    frames: list[np.ndarray] = []
    send_mode = "single"
    stream: FrameStream | None = None
    if not args.no_image:
        want = 1 if args.mode == "single" else max(1, args.frames)
        try:
            if want == 1:
                frames = [capture_frame(args.camera)]
            else:
                # One held stream, never repeated opens -- see QwenWorker.grab.
                stream = FrameStream(args.camera)
                frames = [stream.fresh(min_advance=1)]
                for _ in range(want - 1):
                    time.sleep(args.gap)
                    frames.append(stream.fresh(min_advance=1))
        except CameraError as exc:
            print(f"camera: {exc}\n{camera_hint()}", file=sys.stderr)
            return 1
        finally:
            if stream is not None:
                stream.close()
        send_mode = args.mode if len(frames) > 1 else "single"
        if send_mode == "montage":
            frames = [montage(frames)]
        if args.grid:
            frames = [annotate_for_pointing(f) for f in frames]

    started = time.monotonic()
    try:
        reply = ask(
            args.prompt, frames or None,
            mode="video" if send_mode == "video" else "images",
            fps=1.0 / args.gap if args.gap > 0 else 1.0,
            url=args.url, max_new_tokens=args.tokens,
            do_sample=args.sample,
        )
    except QwenError as exc:
        print(f"qwen: {exc}", file=sys.stderr)
        return 1
    latency = time.monotonic() - started
    regions = parse_regions(reply.text)

    print(f"# {svc}  ({latency:.1f}s, {reply.mode}, {reply.prompt_tokens} prompt tok)")
    warning = reply.frame_warning()
    if warning:
        print(f"# WARNING: {warning}", file=sys.stderr)
    print(reply.text)
    for r in regions:
        print(f"# {format_region(r)}")
    if args.save:
        display = frames[0] if len(frames) == 1 else (
            montage(frames) if frames else np.zeros((8, 8, 3), np.uint8)
        )
        answer = Answer(
            prompt=args.prompt, sent_prompt=args.prompt, sent=display,
            text=reply.text, regions=regions, latency_s=latency, tokens=args.tokens,
            had_image=bool(frames), send_mode=send_mode, reply=reply,
        )
        annotated = answer.sent.copy()
        draw_regions(annotated, regions, args.coords)
        print(f"# saved {save_probe(Path(args.outdir), answer, annotated)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Interactive Qwen3-VL harness: ask questions about the camera feed "
            "and see the answer (and any coordinates it returns) drawn on the "
            "exact frame that was sent"
        ),
    )
    p.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    p.add_argument("--url", default=DEFAULT_URL, help="service base URL")
    p.add_argument(
        "--tokens", type=int, default=DEFAULT_TOKENS, help="max_new_tokens",
    )
    p.add_argument(
        "--grid", action="store_true",
        help="draw a labelled pixel grid on the sent image (helps pointing)",
    )
    p.add_argument(
        "--coords", choices=("abs", "norm"), default="norm",
        help=(
            "read returned coords as 0-1000 normalized (norm, measured default "
            "for this build) or raw pixels (abs)"
        ),
    )
    p.add_argument(
        "--mode", choices=SEND_MODES, default="single",
        help=(
            "how multiple frames reach the model: images (direction right, ~2x "
            "tokens), video (change right, direction not, ~1x), montage (tiled "
            "into one picture)"
        ),
    )
    p.add_argument(
        "--watch", default=DEFAULT_WATCH_QUESTION, metavar="QUESTION",
        help=(
            "the standing question the motion watcher asks on the new frame "
            "whenever the desk moves and settles. Watching is on by default; "
            "whatever you type at the prompt replaces this"
        ),
    )
    p.add_argument(
        "--no-watch", action="store_true",
        help="start with the motion watcher disarmed (/watch <q> arms it later)",
    )
    p.add_argument(
        "--sens", type=float, default=MOTION_THRESHOLD,
        help=f"motion trigger threshold (default {MOTION_THRESHOLD})",
    )
    p.add_argument("--frames", type=int, default=1, help="frames to capture")
    p.add_argument("--gap", type=float, default=0.5, help="seconds between frames")
    p.add_argument(
        "--sample", action="store_true",
        help="sample at the model's temperature 0.7 instead of greedy (default greedy)",
    )
    p.add_argument(
        "--prompt", default="",
        help="ask one question, print the answer, exit (no window)",
    )
    p.add_argument(
        "--no-image", action="store_true",
        help="with --prompt: text-only, do not send a frame",
    )
    p.add_argument("--save", action="store_true", help="with --prompt: save the record")
    p.add_argument("--outdir", default="qwen_probes", help="where /save writes")
    p.add_argument("--no-preview", action="store_true", help="no window; terminal only")
    p.add_argument(
        "--park", action="store_true",
        help="home and retreat the arm to camera park first (needs calibration)",
    )
    args = p.parse_args(argv)

    try:
        info = health(args.url)
    except QwenError as exc:
        print(f"qwen unavailable: {exc}", file=sys.stderr)
        return 1
    svc = f"{info.get('model', '?')}  {info.get('device', '?')}/{info.get('quantization', '?')}"
    if not info.get("loaded", False):
        print("note: model still loading; the first answer will be slow", file=sys.stderr)

    if args.prompt:
        return run_once(args, svc)

    client = None
    if args.park:
        # Optional and best-effort: the harness is about the camera and the
        # model, but an arm parked over the desk is a frame problem that
        # looks exactly like a model problem.
        try:
            from mt4_jog.client import Mt4Client
            from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration
            from mt4_vision.pickplace import ensure_homed, retreat_for_camera

            calib = load_calibration(DEFAULT_CALIB_PATH)
            client = Mt4Client()
            ensure_homed(client)
            retreat_for_camera(client, calib)
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001 -- arm is not required here
            print(f"warning: could not park the arm ({exc}); continuing", file=sys.stderr)
            if client is not None:
                client.close()
                client = None

    opts = Options(
        tokens=args.tokens, grid=args.grid, coord_mode=args.coords,
        send_mode=args.mode, frames=max(1, args.frames), frame_gap_s=max(0.05, args.gap),
        greedy=not args.sample,
    )
    stream: FrameStream | None = None
    view: HarnessPreview | None = None
    ui = BottomUI("ask")
    worker: QwenWorker | None = None
    watcher: MotionWatcher | None = None
    try:
        # Held open even with --no-preview: see QwenWorker.grab.
        stream = FrameStream(args.camera)
        worker = QwenWorker(stream=stream, url=args.url, opts=opts, ui=ui)
        watcher = MotionWatcher(stream, worker, ui, threshold=args.sens)
        if not args.no_preview:
            view = HarnessPreview(stream, worker, svc=svc, ui=ui, watcher=watcher)
        if not args.no_watch and args.watch:
            watcher.arm(args.watch)

        ui.emit(f"qwen3-vl probe -- {svc}")
        ui.emit("/help for commands, /preset for capability probes")
        if watcher.armed:
            ui.emit(
                f"  watching the desk (motion > {watcher.threshold:.5f}); default "
                "question boxes and describes whatever appears. What you type "
                "replaces it. /watch off to stop."
            )
            if opts.tokens < 500:
                ui.emit(f"  note: /tokens {opts.tokens} may truncate the list; 600+ is safer")
            ui.set_status("watching -- type a question, or wait for movement")
        else:
            ui.set_status("ready")
        while True:
            if view is not None and view.stopped_by_user():
                break
            line = ui.read_line()
            if line is None:
                break
            if not line:
                worker.clear()
                ui.set_status("live")
                continue
            if handle_command(
                line, worker=worker, ui=ui, view=view, url=args.url,
                outdir=Path(args.outdir), watcher=watcher,
            ):
                continue
            worker.ask(line)
            # A typed question answers now AND becomes what the watcher keeps
            # asking, so "watch this for me" needs no second command. Only
            # while already armed -- after an explicit /watch off, typing must
            # not quietly switch monitoring back on. Commands (/preset,
            # /again, /noimage) stay one-off and leave the standing question be.
            if watcher is not None and watcher.armed:
                # "latest", not "pair": a typed question is almost always about
                # the state of the scene ("where is the stapler", "how many
                # cubes"), and handing that a before/after pair reframes it as a
                # comparison -- which is how the pair framing kept answering
                # "it has not moved" instead of saying where things were. Only
                # the built-in "what changed" default and an explicit
                # /watch <q> want both frames.
                watcher.arm(line, send="latest")
                ui.set_status(f"answering this on every change: {line[:38]}")
        # Clean exit (EOF or window closed): let a question already with the
        # model finish, so a piped batch reports every answer.
        worker.drain()
        return 0
    except KeyboardInterrupt:
        return 0
    except CameraError as exc:
        ui.close()
        print(f"camera: {exc}\n{camera_hint()}", file=sys.stderr)
        return 1
    finally:
        if watcher is not None:
            watcher.close()
        if worker is not None:
            worker.close()
        ui.close()
        if view is not None:
            view.close()
        if stream is not None:
            stream.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    code = main()
    # Hard exit after main()'s cleanup has already released the camera and the
    # window. OpenCV highgui can keep a Windows process alive past the end of
    # main, and a lingering process holds the capture device open -- which
    # strands the camera for every other vision script in the repo until it is
    # killed by hand. Observed once; not worth risking again for a CLI whose
    # work is finished by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
