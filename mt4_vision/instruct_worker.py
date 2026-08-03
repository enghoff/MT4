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
from mt4_vision.locate import LocateError, measure
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

    The one-shot form (``run_instruction.py "..."``) is scripted, piped and
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


def describe(action: I.Action) -> str:
    bits = [action.kind]
    if action.entity_id:
        bits.append(f"{action.entity_id} ({action.label})")
    if action.model_entity_id and action.model_entity_id != action.entity_id:
        bits.append(f"[model said {action.model_entity_id}]")
    if action.dest_entity_id:
        bits.append(f"-> {action.dest_entity_id} ({action.dest_label})")
    if action.model_dest_entity_id and (
        action.model_dest_entity_id != action.dest_entity_id
    ):
        bits.append(f"[model said dest {action.model_dest_entity_id}]")
    return "  ".join(bits)


class TaskWorker:
    """Runs instructions one at a time, off the prompt thread.

    The prompt thread must never block on the arm. A transfer is several
    seconds of motion and a decision is several more of GPU time, and during
    all of it the user has to be able to type ``/abort`` -- which is the whole
    reason this is a thread rather than a loop in ``main``.

    One at a time, FIFO, because two instructions cannot share an arm. The
    backlog is a real queue rather than a single slot for the same reason
    ``ask_qwen``'s is: a silently dropped instruction and an instruction the
    model refused look identical afterwards.
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
        # Anything Qwen named that the cube detector cannot see, measured once
        # and then carried as obj_N. This is what makes the loop work on a
        # stapler and not only on coloured cubes: the entity layer treats a
        # registered object exactly like a cube from here on, same gates, same
        # motion. Rebuilt per instruction, because the positions in it describe
        # a desk that the last instruction has since rearranged.
        objects: dict[str, object] = {}
        obj_seq = 0
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
                self._camera, held=held, history=history, objects=objects,
                token=f"s{step}", frame=self.frame(),
            )
            if self._save_view:
                cv2.imwrite(self._save_view, obs.annotated)
            self._ui.emit("")
            self._ui.emit(f"[{step}] {obs.snapshot.summary}")
            self._show(
                obs,
                caption=[
                    (f"[{step}] {instruction}", CAPTION_BGR),
                    (obs.snapshot.summary, CAPTION_DIM_BGR),
                ],
            )

            # Anything the task names that the detector has no word for gets one
            # focused grounding call, before the decision, so the thing exists as
            # obj_N by the time an action is chosen. Left to the decision prompt
            # alone the model forces the task onto the nearest cube instead.
            # `held` is passed so the noun that started the task stops being
            # unmatched once the thing is in the jaws. Without it every later step
            # goes looking for it again and grounds onto the gripper.
            for noun in sorted(
                I.unmatched_nouns(instruction, obs.snapshot, held=held)
            ):
                if self._stopped():
                    break
                if any(noun in e.label.lower() for e in obs.snapshot.entities):
                    continue
                self._phase(f"locating {noun}")
                self._ui.set_status(f"step {step}: looking for a {noun}")
                grounding, why = I.locate_target(obs, noun)
                if grounding is None:
                    self._ui.emit(f"    could not find a {noun}: {why}")
                    self._show(
                        obs,
                        caption=[
                            (f"[{step}] {instruction}", CAPTION_BGR),
                            (f'locate "{noun}": {why}', QWEN_REFUSED_BGR),
                        ],
                    )
                    continue
                self._show(
                    obs, grounding=grounding,
                    caption=[
                        (f"[{step}] {instruction}", CAPTION_BGR),
                        (f'locate "{noun}" -- measuring it', CAPTION_DIM_BGR),
                    ],
                )
                # The box, not just its centre: GrabCut from a detector box finds
                # silhouettes the desk-deviation path cannot. Measured on one live
                # frame, point-only segmented 1 of 4 objects and the box segmented
                # 4 of 4 -- including the stapler this loop kept failing on.
                obj, why = I.measure_grounding(obs, grounding)
                if obj is None:
                    self._ui.emit(f"    found a {noun} but could not measure it: {why}")
                    self._show(
                        obs, grounding=grounding,
                        caption=[
                            (f"[{step}] {instruction}", CAPTION_BGR),
                            (f"found a {noun} but could not measure it: {why}",
                             QWEN_REFUSED_BGR),
                        ],
                    )
                    continue
                # On the frame the mask was actually cut from, before the
                # re-observe below replaces it. This is the one moment the
                # segmentation can be judged against the exact pixels it was
                # computed on.
                self._show(
                    obs, grounding=grounding, obj=obj,
                    caption=[
                        (f"[{step}] {instruction}", CAPTION_BGR),
                        (f'measured the {noun}: {obj.long_mm:.0f}x'
                         f'{obj.short_mm:.0f}mm at ({obj.x:.0f}, {obj.y:.0f})',
                         CAPTION_DIM_BGR),
                    ],
                )
                eid, is_new = I.register_object(objects, obj, seq=obj_seq + 1)
                if is_new:
                    obj_seq += 1
                    self._ui.emit(
                        f"    registered {eid}: {noun} at ({obj.x:.0f}, "
                        f"{obj.y:.0f}) {obj.long_mm:.0f}x{obj.short_mm:.0f}mm"
                    )
                else:
                    # A merge changes the entity's LABEL, and the label is what
                    # `unmatched_nouns` reads. Skipping the rebuild here left the
                    # second word of a two-word target still unmatched, so the task
                    # was refused for naming something already registered.
                    self._ui.emit(f"    the {noun} is {eid} ({objects[eid].label})")
                # Re-list, do not re-observe. The arm is parked and nothing on
                # the desk moved, so the frame in hand is still current -- and a
                # fresh capture costs another settle and another round of
                # detection, once per noun.
                obs = obs.relisted(objects=objects, token=f"s{step}b")

            why = self._stopped()
            if why:
                self._ui.emit(f"    -> {why} before deciding")
                self._ui.set_status(why)
                break

            self._phase("deciding")
            self._ui.set_status(f"step {step}: asking the model what to do")
            action = I.decide(obs, instruction)
            self._ui.emit(f"    {describe(action)}")
            self._ui.emit(f"    reason: {action.reason}")
            self._set(action=describe(action), reason=action.reason)
            self._show(
                obs, action=action, objects=objects,
                caption=[
                    (f"[{step}] {instruction}", CAPTION_BGR),
                    (describe(action), QWEN_BOUND_BGR if action.ok else QWEN_REFUSED_BGR),
                    (action.reason, CAPTION_DIM_BGR),
                ],
            )
            self._record(obs, action)

            if action.kind == "DONE":
                self._ui.set_status(f"done in {step} step(s)")
                result = True
                break
            if not action.ok:
                # Stopping here strands whatever is in the jaws. With the
                # gripper full the step is worth retrying instead: the next one
                # is a fresh park, a fresh frame and a fresh decision, and a
                # refusal caused by the frame rather than the request goes away
                # on its own. This run is the case in point -- step 1 saw
                # "unknown=1", a marker whose tag had not decoded, and step 2 saw
                # "unknown=0"; the desk had not changed, only the exposure.
                # Bounded by max_steps like everything else, and the arm is
                # parked between tries, so the cost of being wrong is a few
                # seconds rather than a dropped object.
                if held is not None and step < max_steps:
                    self._ui.emit(
                        f"    -> refused, but still holding the {held} -- "
                        "stopping would strand it, so trying again on a fresh look"
                    )
                    self._ui.set_status(f"refused, retrying: {action.reason[:48]}")
                    continue
                self._ui.emit(
                    "    -> refused, stopping. The scene or the request needs to change."
                )
                self._ui.set_status(f"refused: {action.reason[:60]}")
                break
            if client is None:
                self._ui.emit("    -> dry run, not moving")
                self._ui.set_status("dry run -- decided, did not move")
                break

            if action.kind == "LOCATE_AT_PIXEL":
                # Measured against the exact frame the model looked at, not a
                # fresh capture, so the pixel means what it saw. The hint only
                # has to land ON the object -- segmentation recovers the real
                # centre, long axis and size in millimetres.
                px, py = action.point_px
                try:
                    obj = measure(
                        obs.frame, px, py, obs.calib, action.label or "object",
                        marker_xy=[
                            (e.x, e.y)
                            for e in obs.snapshot.entities if e.kind == "marker"
                        ],
                    )
                except LocateError as exc:
                    self._ui.emit(f"    -> could not measure it: {exc}")
                    self._set(error=f"could not measure it: {exc}")
                    break
                obj_seq += 1
                objects[f"obj_{obj_seq}"] = obj
                history.append(
                    f"found the {action.label} on the desk and measured it "
                    f"({obj.long_mm:.0f}x{obj.short_mm:.0f}mm)"
                )
                self._ui.emit(
                    f"    -> registered obj_{obj_seq}: {action.label} at "
                    f"({obj.x:.0f}, {obj.y:.0f}) "
                    f"{obj.long_mm:.0f}x{obj.short_mm:.0f}mm"
                )
                continue

            entity = obs.snapshot.get(action.entity_id)
            # Straight off the snapshot the decision was made against -- see
            # instruct.grasp_for. The arm is parked and nothing else on this desk
            # moves, so a second capture before the grasp re-measured a scene that
            # had not changed, and the template re-match it did that with tracked
            # an ArUco tag instead of the object it was meant to find.
            grasp = I.grasp_for(entity, obs.calib)
            yaw = "wrist kept" if grasp.yaw_deg is None else f"yaw {grasp.yaw_deg:.0f}"

            self._phase("moving")
            try:
                if action.kind == "TRANSFER":
                    dest = obs.snapshot.get(action.dest_entity_id)
                    dest_grasp = I.grasp_for(dest, obs.calib)
                    line = (
                        f"pick ({grasp.x:.0f}, {grasp.y:.0f}) {yaw}"
                        f"  ->  place ({dest_grasp.x:.0f}, {dest_grasp.y:.0f}) squared"
                    )
                    self._ui.emit(f"    -> {line}")
                    self._record(obs, action, grasp=line)
                    self._ui.set_status(
                        f"step {step}: moving the {action.label} "
                        f"onto {action.dest_label}"
                    )
                    # One planned operation: the lift out of the pick and the
                    # carry are the same leg, so the arm never stops between
                    # having the object and being on its way. Splitting this
                    # into pick, park, look, decide, place cost two captures and
                    # a second decision and could not answer the only question
                    # that mattered anyway.
                    transfer(client, obs.calib, grasp, dest_grasp)
                    retreat_for_camera(client, obs.calib)
                    # The move invalidated this snapshot at both ends: the object
                    # left its position and the destination gained an occupant.
                    objects.pop(action.entity_id, None)
                    moved = action.label or "object"
                    # By label, not id. The destination keeps its printed number
                    # ("marker 2") across snapshots, while marker_N ids do not.
                    history.append(f"moved the {moved} onto {action.dest_label}")
                    self._outcome(f"moved the {moved} onto {action.dest_label}")
                    self._ui.emit(
                        f"    -> moved the {moved} onto {action.dest_label}{UNCHECKED}"
                    )
                elif action.kind == "PICK_ENTITY":
                    line = f"pick ({grasp.x:.0f}, {grasp.y:.0f}) {yaw}"
                    self._ui.emit(f"    -> {line}")
                    self._record(obs, action, grasp=line)
                    self._ui.set_status(f"step {step}: picking up the {action.label}")
                    pick_at(client, obs.calib, grasp)
                    # The label, never the id. The id described a position in the
                    # snapshot that has just been invalidated by this very move.
                    held = action.label or "object"
                    self._set(held=held)
                    # It is in the jaws, not on the desk, so it stops being an
                    # entity. Leaving it listed offered the model its own held
                    # object as a place destination -- measured: "put the stapler
                    # on marker 1" came back as PLACE_ENTITY obj_1, the stapler.
                    objects.pop(action.entity_id, None)
                    history.append(f"picked up the {held}")
                    self._outcome(f"picked up the {held}")
                    self._ui.emit(f"    -> holding the {held}{UNCHECKED}")
                else:
                    line = f"place ({grasp.x:.0f}, {grasp.y:.0f}) squared"
                    self._ui.emit(f"    -> {line}")
                    self._record(obs, action, grasp=line)
                    self._ui.set_status(f"step {step}: putting it on {action.label}")
                    place_at(client, obs.calib, grasp)
                    retreat_for_camera(client, obs.calib)
                    history.append(f"put the {held} down on {action.label}")
                    self._outcome(f"put the {held} down on {action.label}")
                    self._ui.emit(
                        f"    -> put the {held} down on {action.label}{UNCHECKED}"
                    )
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
# clear the motion from its exposure. Cheap next to the 2-3s the old reopen of
# the camera cost per step; see FrameStream vs capture_frame.
CAMERA_SETTLE_S = 0.4


# Every outcome line carries this, because the loop reports what it SENT. There
# is no sensor in the jaws and no vision test that survived contact with a real
# desk: the template check that used to run here matched the ArUco tag beside a
# stapler and declared a successful pick a failure. Printing the caveat is the
# honest version of that -- silently reporting commands as observations is how
# a grasp that closed on air used to drive a confident DONE two steps later.
UNCHECKED = "  (commanded, not checked)"
