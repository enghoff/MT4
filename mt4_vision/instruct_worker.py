"""Runs English instructions on the arm, one at a time, off the prompt thread.

:class:`TaskWorker` owns the loop: park clear of the camera, capture one frame,
ask Qwen for one action, resolve it against that same snapshot, move. The prompt
thread never blocks on any of it, so ``/abort`` still lands mid-transfer.

Instructions queue FIFO because there is one arm. Arm chores (home, park, read
status) go through the same queue rather than running on the prompt thread:
``Mt4Client`` is individually thread-safe, but "home the arm" arriving in the
middle of a transfer is two callers disagreeing about where the arm should be.

Everything reported about a move is what was **sent**. There is no sensor in the
jaws, so outcome lines say "commanded, not checked". What is checked, before the
gripper opens, is reach, keep-out, ground, finger clearance, the desk polygon,
and both ends of a transfer.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision import instruct as I
from mt4_vision.camera import FrameStream
from mt4_vision.console import BottomUI
from mt4_vision.instruct_view import (
    CAPTION_BGR,
    CAPTION_DIM_BGR,
    MAX_STEPS,
    RunState,
    decision_view,
)
from mt4_vision.motion import pick_at, place_at, transfer
from mt4_vision.pickplace import retreat_for_camera
from mt4_vision.preview import QWEN_BOUND_BGR, QWEN_REFUSED_BGR


@dataclass
class Decision:
    """The last decision, kept for ``/save``."""

    instruction: str
    step: int
    action: I.Action
    summary: str
    view: np.ndarray
    grasp: str = ""


@dataclass
class Job:
    """One queued thing for the worker: an instruction, or an arm chore.

    Chores go through the same queue as instructions rather than running on
    the prompt thread, because both talk to one serial port and one arm.
    ``Mt4Client`` is individually thread-safe, but "home the arm" arriving in
    the middle of a transfer is not a locking problem -- it is two callers
    disagreeing about where the arm should be.
    """

    kind: str          # "task" | "chore"
    text: str
    fn: Callable[[], None] | None = None


class PlainUI:
    """``BottomUI``'s surface, printing straight to stdout.

    The one-shot form (``ask_qwen.py "..."``) is scripted, piped and
    quoted in the docs, so its output has to stay plain lines in order: no
    pinned footer, no escape sequences, nothing that assumes a terminal.
    Status is transient by definition and every durable thing is emitted, so
    dropping it here loses nothing.
    """

    def emit(self, text: str = "") -> None:
        print(text, flush=True)

    def set_status(self, msg: str) -> None:
        pass

    def close(self) -> None:
        pass


def _destination_words(obs: I.Observation, action: I.Action) -> str:
    """How to say where something is going, in a transcript line.

    The destination is a pixel, so the pixel is what gets said. A tag standing
    under that pixel is named as well, because that is the check worth making
    at a glance: told to place on marker 2, the line should read "onto
    marker_2", and told to place on a cube it should read as the bare pixel it
    is. The name is a label on the coordinate and never a substitute for it --
    ``instruct.tag_at`` is reporting only, and the arm goes to the pixel.
    """
    if not action.dest_point_px:
        return "down"
    px, py = action.dest_point_px
    x, y = obs.calib.pixel_to_robot(px, py, on_cube_top=False)
    on = I.tag_at(obs, x, y)
    if on is not None:
        return f"onto {on.id}"
    return f"down at image ({px:.0f}, {py:.0f})"


_DIGITS = re.compile(r"[-+]?\d*\.?\d+")


def _refusal_signature(reason: str) -> str:
    """A refusal reduced to which gate spoke, with every number removed.

    Two refusals of the same request are the same refusal even when their
    measurements differ, and they do differ: a box re-measured on a fresh frame
    moves a few millimetres, so ``r=367mm is beyond the 350mm max reach`` and
    ``r=373mm is beyond the 350mm max reach`` are one problem written twice.
    Comparing the prose with the numbers stripped catches that; comparing the
    strings does not.
    """
    return _DIGITS.sub("#", reason.strip().lower())


def _place_angle_words(dest_grasp) -> str:
    """How the object is set down, for a transcript line.

    Two outcomes and they must not read alike: the reply either pointed along
    an axis to lay the object on, or said nothing and the destination squares
    to the world axes. Both are decisions the arm carries out.
    """
    if dest_grasp.yaw_period_deg == 90.0:
        return "squared"
    return f"along {dest_grasp.yaw_deg:.0f}deg"


def describe(obs: I.Observation, action: I.Action) -> str:
    bits = [action.kind]
    if action.label:
        bits.append(action.label)
    if action.source is not None and action.source.box_px:
        x1, y1, x2, y2 = action.source.box_px
        bits.append(f"box ({x1:.0f}, {y1:.0f})-({x2:.0f}, {y2:.0f})")
    if action.kind in ("TRANSFER", "PLACE"):
        bits.append(f"-> {_destination_words(obs, action).removeprefix('down ')}")
    return "  ".join(bits)


class TaskWorker:
    """Runs instructions one at a time, off the prompt thread.

    The prompt thread must never block on the arm. A transfer is several
    seconds of motion and a decision is several more of GPU time, and during
    all of it the user has to be able to type ``/abort`` -- which is the whole
    reason this is a thread rather than a loop in ``main``.

    One at a time, FIFO, because two instructions cannot share an arm. The
    backlog is a real queue rather than a single slot because a silently
    dropped instruction and an instruction the model refused look identical
    afterwards.
    """

    def __init__(
        self,
        *,
        ui: BottomUI | PlainUI,
        stream: FrameStream,
        camera: int,
        client: Mt4Client | None,
        max_steps: int = MAX_STEPS,
        save_view: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self._ui = ui
        self._stream = stream
        self._camera = camera
        self._client = client
        self._save_view = save_view
        self._lock = threading.Lock()
        self._queue: list[Job] = []
        self._state = RunState(max_steps=max_steps, dry_run=dry_run)
        self._view: np.ndarray | None = None
        self._decision: Decision | None = None
        # Signature of the last refusal within the running instruction, so the
        # same refusal twice ends the task instead of spending the budget.
        # Reset per instruction in _run_task.
        self._last_refusal: str | None = None
        self._last_instruction = ""
        self._last_result: bool | None = None
        # Two levels, because they are two different requests. _stop ends the
        # run at the next step boundary, which leaves the arm somewhere it
        # chose to be. _abort additionally interrupts the move in flight,
        # which does not.
        self._stop = threading.Event()
        self._abort = threading.Event()
        self._closing = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="instruction", daemon=True,
        )
        self._thread.start()

    # -- called from the prompt thread -------------------------------------

    def submit(self, instruction: str) -> None:
        with self._lock:
            self._queue.append(Job("task", instruction))
            self._last_instruction = instruction
            waiting = len(self._queue) + (0 if self._state.phase == "idle" else 1)
            self._publish_state(queued=len(self._queue))
        self._wake.set()
        self._ui.set_status(
            "running..." if waiting <= 1 else f"queued ({waiting} waiting)"
        )

    def submit_chore(self, label: str, fn: Callable[[], None]) -> None:
        with self._lock:
            self._queue.append(Job("chore", label, fn))
            self._publish_state(queued=len(self._queue))
        self._wake.set()
        self._ui.set_status(f"{label}: queued")

    def cancel_queued(self) -> int:
        with self._lock:
            dropped = len(self._queue)
            self._queue.clear()
            self._publish_state(queued=0)
        return dropped

    def request_stop(self, *, abort: bool) -> bool:
        """Ask the run to end. Returns True if something was actually running.

        Dropping the queue and raising the flags happens in ONE critical
        section, the same one the worker clears the flags in when it starts a
        job. Split across two, a job popped in between would clear a stop that
        had already been asked for and run to completion anyway -- a rare loss,
        and the wrong command to lose rarely.
        """
        with self._lock:
            dropped = len(self._queue)
            self._queue.clear()
            running = self._state.phase != "idle"
            self._publish_state(
                queued=0,
                stopping="aborting -- halting the arm" if abort else
                         "stopping after this step",
            )
            self._stop.set()
            if abort:
                self._abort.set()
                if self._client is not None:
                    # Documented for exactly this: abort an in-flight move or
                    # gripper settle. The move returns not-ok, the motion call
                    # raises, and the step's own error path reports it.
                    self._client.request_interrupt()
        if dropped:
            self._ui.emit(f"  dropped {dropped} queued instruction(s)")
        return running or bool(dropped)

    def busy(self) -> bool:
        with self._lock:
            return self._state.phase != "idle" or bool(self._queue)

    def snapshot(self) -> tuple[RunState, np.ndarray | None]:
        """State and the frame to draw. The frame is shared, never mutated."""
        with self._lock:
            return self._state, self._view

    def last_decision(self) -> Decision | None:
        with self._lock:
            return self._decision

    def last_instruction(self) -> str:
        with self._lock:
            return self._last_instruction

    def last_result(self) -> bool | None:
        with self._lock:
            return self._last_result

    def held(self) -> str | None:
        with self._lock:
            return self._state.held

    def set_held(self, held: str | None) -> None:
        with self._lock:
            self._publish_state(held=held)

    def set_max_steps(self, steps: int) -> None:
        with self._lock:
            self._publish_state(max_steps=steps)

    def set_dry_run(self, dry: bool) -> None:
        with self._lock:
            self._publish_state(dry_run=dry)

    def drain(self, timeout: float = 900.0) -> None:
        """Block until nothing is queued or running.

        The one-shot path is the reason this exists: without it the process
        would exit while the arm was still carrying something.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.busy():
                return
            time.sleep(0.1)

    def close(self) -> None:
        self._closing.set()
        self._wake.set()
        self._thread.join(timeout=3.0)

    # -- state plumbing (call with the lock held) --------------------------

    def _publish_state(self, **changes: object) -> None:
        self._state = replace(self._state, **changes)

    def _set(self, **changes: object) -> None:
        with self._lock:
            self._publish_state(**changes)

    def _phase(self, phase: str) -> None:
        with self._lock:
            self._publish_state(phase=phase, phase_started=time.monotonic())

    def _show(
        self, obs: I.Observation, **kwargs: object,
    ) -> None:
        """Publish a decision frame for the window. Never mutated afterwards."""
        frame = decision_view(obs, **kwargs)   # type: ignore[arg-type]
        with self._lock:
            self._view = frame

    def _record(self, obs: I.Observation, action: I.Action, grasp: str = "") -> None:
        with self._lock:
            if self._view is None:
                return
            self._decision = Decision(
                instruction=self._state.instruction, step=self._state.step,
                action=action, summary=obs.snapshot.summary, view=self._view,
                grasp=grasp,
            )

    def _outcome(self, text: str) -> None:
        with self._lock:
            self._publish_state(outcomes=self._state.outcomes + (text,))

    # -- the loop ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._closing.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            with self._lock:
                job = self._queue.pop(0) if self._queue else None
                if job is not None:
                    # Phase leaves "idle" in the SAME critical section as the
                    # pop. Setting it later opens a window where the queue is
                    # already empty and the phase is still idle, which is
                    # exactly what busy() reads -- drain() saw that window and
                    # returned while an instruction was starting, and the
                    # one-shot path then tore the serial port down under a
                    # move that had just begun.
                    self._publish_state(
                        queued=len(self._queue), phase="starting",
                        phase_started=time.monotonic(),
                    )
                    # A fresh request starts from a clean slate: a stop asked
                    # for while the LAST one ran must not silently kill this
                    # one. Cleared inside the pop's critical section so it
                    # cannot cross with request_stop -- see the note there.
                    self._stop.clear()
                    self._abort.clear()
                    if self._client is not None:
                        self._client.clear_interrupt()
            if job is None:
                continue

            try:
                if job.kind == "chore":
                    self._phase(job.text)
                    self._ui.set_status(f"{job.text}...")
                    assert job.fn is not None
                    job.fn()
                else:
                    self._run_task(job.text)
            except Exception as exc:  # noqa: BLE001 -- a failure must not kill the worker
                # Anything uncaught here would end the thread, and a dead
                # worker looks exactly like a harness that ignores you: the
                # prompt still accepts instructions and nothing ever happens.
                message = f"{type(exc).__name__}: {exc}"
                self._ui.emit(f"  !! {message}")
                self._ui.set_status(message)
                self._set(error=message)
            finally:
                self._set(phase="idle", phase_started=0.0, stopping="")

    def _stopped(self) -> str:
        """The reason to stop, or "" to keep going."""
        if self._abort.is_set():
            return "aborted"
        if self._stop.is_set():
            return "stopped"
        return ""

    def frame(self) -> np.ndarray:
        """A frame whose exposure started after the arm finished parking.

        ``FrameStream.fresh`` is what makes that true -- it blocks for a frame
        captured entirely after the call, which a fixed flush count off a
        buffered device cannot promise. The settle before it is for the
        linkage, which is still ringing when the firmware says the move is
        done.
        """
        if CAMERA_SETTLE_S > 0:
            time.sleep(CAMERA_SETTLE_S)
        return self._stream.fresh(min_advance=2)

    def _declined(self, reason: str, held: str | None) -> None:
        """Report a STOP the model chose. Terminal, and never retried.

        A STOP is the model's answer to the request, so re-asking is asking the
        same question again. Live, "open the gripper" -- which names no object
        to see -- was answered STOP six times running, each time in different
        words, spending six park-capture-decide cycles to arrive where step 1
        already was.

        Sameness detection is not the fix. ``_refusal_signature`` strips
        numbers, which is enough for the fixed wording the gates emit and
        useless against model prose that rephrases the same judgement every
        time. The fix is not to re-ask at all.

        Full jaws do not buy a retry here either. The jaws stay as they are and
        the object stays in them, which is exactly where six more refusals
        would have left it -- so this says so on the first step instead of the
        sixth.
        """
        self._ui.emit("    -> STOP is the answer to this request, not a bad look at it.")
        if held is not None:
            self._ui.emit(
                f"       Still holding the {held} -- it stays in the jaws; "
                "/open releases it."
            )
        self._ui.set_status(f"stopped: {reason[:60]}")

    def _refused(
        self, reason: str, held: str | None, step: int, max_steps: int
    ) -> bool:
        """Report a refusal. True to retry this step, False to give up.

        For refusals the model did *not* choose: a malformed or missing reply,
        and the physical gates on a measurement or a pose. A STOP the model
        chose goes to :meth:`_declined` and never reaches here.

        A refusal caused by the frame rather than by the request goes away on
        its own: the next step is a fresh park, a fresh frame and a fresh
        decision. Measured on one run -- step 1 saw ``unknown=1``, a marker
        whose tag had not decoded, and step 2 saw ``unknown=0``; the desk had
        not changed, only the exposure. So such a refusal is worth one more
        look whatever the jaws hold, and a full gripper is called out because
        stopping there strands the object.

        **The same refusal twice running is not worth a third look.** Nothing
        has moved between tries, so a reason that recurs is about the request,
        and repeating it spends the whole step budget arriving where step 1
        already was. Sameness is judged with the numbers stripped
        (:func:`_refusal_signature`), because a re-measured object jitters --
        live, one out-of-reach blob refused at ``r=367mm`` and then ``r=373mm``,
        two strings for one problem.
        """
        signature = _refusal_signature(reason)
        repeat = signature == self._last_refusal
        self._last_refusal = signature
        if repeat:
            self._ui.emit(
                "    -> refused the same way twice, so the request is the "
                "problem rather than the photograph. Stopping."
            )
            if held is not None:
                self._ui.emit(
                    f"       Still holding the {held} -- it stays in the jaws; "
                    "/open releases it."
                )
            self._ui.set_status(f"refused twice: {reason[:52]}")
            return False
        if step < max_steps:
            if held is not None:
                self._ui.emit(
                    f"    -> refused, and still holding the {held} -- stopping "
                    "would strand it, so trying again on a fresh look"
                )
            else:
                self._ui.emit("    -> refused, trying again on a fresh look")
            self._ui.set_status(f"refused, retrying: {reason[:48]}")
            return True
        self._ui.emit(
            "    -> refused, stopping. The scene or the request needs to change."
        )
        self._ui.set_status(f"refused: {reason[:60]}")
        return False

    def _run_task(self, instruction: str) -> bool:
        """One instruction to completion. True when the model reported DONE."""
        with self._lock:
            self._publish_state(
                instruction=instruction, step=0, action="", reason="",
                outcomes=(), error="", stopping="",
            )
            max_steps = self._state.max_steps
            dry_run = self._state.dry_run
        client = None if dry_run else self._client
        history: list[str] = []
        # Per instruction, not per session: the previous task's last refusal
        # says nothing about this one, and carrying it over would stop a fresh
        # instruction on its first step.
        self._last_refusal: str | None = None
        # No object registry: a target is measured from the box the model
        # drew, on the frame it drew it on, and acted on in the same step, so
        # there is nothing to carry between steps and no id that can go stale
        # against a desk something has since moved.
        self._ui.emit("")
        self._ui.emit(f"> {instruction}")
        result = False

        for step in range(1, max_steps + 1):
            why = self._stopped()
            if why:
                self._ui.emit(f"    -> {why} before step {step}")
                self._ui.set_status(why)
                break
            # Re-read rather than carry a local across steps: /held corrects
            # what nothing on this rig can sense, and a correction typed
            # during step 2 has to reach the prompt built for step 3.
            with self._lock:
                held = self._state.held
            self._set(step=step)

            if client is not None:
                # The arm is in shot otherwise, and it shadows exactly the part
                # of the desk it just worked on.
                self._phase("parking")
                self._ui.set_status(f"step {step}: parking clear of the camera")
                retreat_for_camera(client, I.load_calibration())

            self._phase("looking")
            self._ui.set_status(f"step {step}: looking")
            obs = I.observe(
                self._camera, held=held, history=history,
                token=f"s{step}", frame=self.frame(),
            )
            if self._save_view:
                cv2.imwrite(self._save_view, obs.annotated)
            self._ui.emit("")
            # The summary is for the operator, not the model. It counts what the
            # cube detector saw, which is exactly the information the prompt
            # withholds -- printing it here is how a reader can tell whether a
            # refusal was the model missing something or the desk being empty.
            self._ui.emit(f"[{step}] {obs.snapshot.summary}")
            self._show(
                obs,
                caption=[
                    (f"[{step}] {instruction}", CAPTION_BGR),
                    (obs.snapshot.summary, CAPTION_DIM_BGR),
                ],
            )

            why = self._stopped()
            if why:
                self._ui.emit(f"    -> {why} before deciding")
                self._ui.set_status(why)
                break

            self._phase("deciding")
            self._ui.set_status(f"step {step}: asking the model what to do")
            action = I.decide(obs, instruction)
            said = describe(obs, action)
            self._ui.emit(f"    {said}")
            self._ui.emit(f"    reason: {action.reason}")
            self._set(action=said, reason=action.reason)
            self._show(
                obs, action=action,
                caption=[
                    (f"[{step}] {instruction}", CAPTION_BGR),
                    (said, QWEN_BOUND_BGR if action.ok else QWEN_REFUSED_BGR),
                    (action.reason, CAPTION_DIM_BGR),
                ],
            )
            self._record(obs, action)

            if action.kind == "DONE":
                self._ui.set_status(f"done in {step} step(s)")
                result = True
                break
            if action.declined:
                self._declined(action.reason, held)
                break
            if not action.ok:
                if self._refused(action.reason, held, step, max_steps):
                    continue
                break

            # ---- turn the reply into poses, before anything moves ---------
            #
            # Both ends of a transfer are resolved here, while the object is
            # still on the desk. Refusing after the grasp strands it in the
            # jaws.
            #
            # Every refusal past the reply's own shape lives here, and all of
            # them are physical. `decide` sees a box, not a target, so "is there
            # anything there, can it be measured, can the arm hold it" is
            # answered by segmenting and by asking the gates the rest of the
            # stack asks.
            grasp = None
            dest_grasp = None
            trouble = ""
            if action.kind in ("TRANSFER", "PICK"):
                self._phase("measuring")
                self._ui.set_status(f"step {step}: measuring the {action.label}")
                # GrabCut inside the model's box, on the exact frame it drew the
                # box on. No fresh capture: the arm is parked and nothing else
                # on this desk moves. Height is taken to be the configured cube
                # height rather than read off the silhouette -- see
                # instruct.measure_source.
                picked, why = I.measure_source(obs, action)
                if picked is None:
                    trouble = f"could not measure what it boxed: {why}"
                else:
                    entity = I.source_entity(obs, picked)
                    measured = (
                        f"measured the {action.label}: {picked.long_mm:.0f}x"
                        f"{picked.short_mm:.0f}mm at "
                        f"({picked.x:.0f}, {picked.y:.0f})"
                    )
                    self._ui.emit(f"    {measured}")
                    self._show(
                        obs, action=action, obj=picked,
                        caption=[
                            (f"[{step}] {instruction}", CAPTION_BGR),
                            (measured, CAPTION_DIM_BGR),
                        ],
                    )
                    if entity.pickable:
                        grasp = entity.as_grasp(obs.calib)
                    else:
                        # The physical gate, quoted verbatim. It names reach,
                        # the J1 keep-out, ground Z, jaw clearance or the desk
                        # polygon -- the useful answer, and the only thing here
                        # entitled to override the model.
                        trouble = f"cannot pick it up: {entity.reason}"
            if not trouble and action.kind in ("TRANSFER", "PLACE"):
                dest_grasp, why = I.destination_grasp(obs, action)
                if dest_grasp is None:
                    trouble = f"the destination will not do: {why}"
            if trouble:
                self._ui.emit(f"    {trouble}")
                self._set(reason=trouble)
                self._show(
                    obs, action=action,
                    caption=[
                        (f"[{step}] {instruction}", CAPTION_BGR),
                        (trouble, QWEN_REFUSED_BGR),
                    ],
                )
                if self._refused(trouble, held, step, max_steps):
                    continue
                break

            if client is None:
                self._ui.emit("    -> dry run, not moving")
                self._ui.set_status("dry run -- decided, did not move")
                break

            where = grasp if grasp is not None else dest_grasp
            yaw = "wrist kept" if where.yaw_deg is None else f"yaw {where.yaw_deg:.0f}"

            self._phase("moving")
            try:
                where_to = _destination_words(obs, action)
                if action.kind == "TRANSFER":
                    line = (
                        f"pick ({grasp.x:.0f}, {grasp.y:.0f}) {yaw}"
                        f"  ->  place ({dest_grasp.x:.0f}, {dest_grasp.y:.0f}) "
                        f"{_place_angle_words(dest_grasp)}"
                    )
                    self._ui.emit(f"    -> {line}")
                    self._record(obs, action, grasp=line)
                    self._ui.set_status(
                        f"step {step}: moving the {action.label} {where_to}"
                    )
                    # One planned operation: the lift out of the pick and the
                    # carry are the same leg, so the arm never stops between
                    # having the object and being on its way. Splitting this
                    # into pick, park, look, decide, place cost two captures and
                    # a second decision and could not answer the only question
                    # that mattered anyway.
                    transfer(client, obs.calib, grasp, dest_grasp)
                    retreat_for_camera(client, obs.calib)
                    moved = action.label or "object"
                    history.append(f"moved the {moved} {where_to}")
                    self._outcome(f"moved the {moved} {where_to}")
                    self._ui.emit(f"    -> moved the {moved} {where_to}{UNCHECKED}")
                elif action.kind == "PICK":
                    line = f"pick ({grasp.x:.0f}, {grasp.y:.0f}) {yaw}"
                    self._ui.emit(f"    -> {line}")
                    self._record(obs, action, grasp=line)
                    self._ui.set_status(f"step {step}: picking up the {action.label}")
                    pick_at(client, obs.calib, grasp)
                    held = action.label or "object"
                    self._set(held=held)
                    history.append(f"picked up the {held}")
                    self._outcome(f"picked up the {held}")
                    self._ui.emit(f"    -> holding the {held}{UNCHECKED}")
                else:
                    line = (
                        f"place ({dest_grasp.x:.0f}, {dest_grasp.y:.0f}) "
                        f"{_place_angle_words(dest_grasp)}"
                    )
                    self._ui.emit(f"    -> {line}")
                    self._record(obs, action, grasp=line)
                    self._ui.set_status(f"step {step}: putting it {where_to}")
                    place_at(client, obs.calib, dest_grasp)
                    retreat_for_camera(client, obs.calib)
                    history.append(f"put the {held} down {where_to}")
                    self._outcome(f"put the {held} down {where_to}")
                    self._ui.emit(f"    -> put the {held} down {where_to}{UNCHECKED}")
                    held = None
                    self._set(held=None)
            except Mt4ClientError as exc:
                # An abort lands here too: the interrupted move returns not-ok
                # and the motion call raises. Say which it was, because "the
                # arm stopped because you told it to" and "the arm could not do
                # that" need different next moves from whoever is standing
                # there. Either way the jaws are wherever the motion left them,
                # which is why held is not cleared.
                stopped = self._stopped()
                note = "aborted mid-move" if stopped == "aborted" else "motion failed"
                self._ui.emit(f"    -> {note}: {exc}")
                self._ui.set_status(f"{note}: {exc}")
                self._set(error=f"{note}: {exc}")
                break
        else:
            self._ui.emit(f"    -> gave up after {max_steps} steps")
            self._ui.set_status(f"gave up after {max_steps} steps")

        with self._lock:
            self._last_result = result
        return result


def save_record(outdir: Path, decision: Decision, canvas: np.ndarray | None,
                name: str = "") -> Path:
    """Write the frame the model saw, the whole window, and what was decided."""
    outdir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S")
    if name:
        stem = f"{stem}_{name}"
    cv2.imwrite(str(outdir / f"{stem}_seen.jpg"), decision.view)
    if canvas is not None:
        cv2.imwrite(str(outdir / f"{stem}_view.jpg"), canvas)
    record = {
        "instruction": decision.instruction,
        "step": decision.step,
        "snapshot": decision.summary,
        "grasp": decision.grasp,
        **decision.action.as_dict(),
    }
    path = outdir / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


MAX_STEPS = 6


# Seconds between the park finishing and the frame that gets decided on. The
# arm reports a move done when the firmware's step generator has finished, not
# when the linkage has stopped ringing, and the camera needs a moment more to
# clear the motion from its exposure. Cheap next to the 2-3s a camera reopen
# costs per step, which is why the loop holds one FrameStream open rather than
# calling capture_frame.
CAMERA_SETTLE_S = 0.4


# Every outcome line carries this, because the loop reports what it SENT. There
# is no sensor in the jaws and no vision test that survives contact with a real
# desk -- a template check matches the ArUco tag beside a stapler and calls a
# successful pick a failure. Printing the caveat is the honest version: report a
# command as an observation and a grasp that closed on air drives a confident
# DONE two steps later.
UNCHECKED = "  (commanded, not checked)"
