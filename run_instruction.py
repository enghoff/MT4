#!/usr/bin/env python3
"""Type an instruction in English; the arm carries it out.

    python run_instruction.py                       # interactive
    python run_instruction.py "put the red cube on marker 3"
    python run_instruction.py --dry-run "pick up the green cube"
    python run_instruction.py --no-preview "..."    # no window (headless, CI)

A preview window opens by default showing the frame the model saw with its own
answer drawn on it -- see ``mt4_vision.preview.annotate_qwen``. q or Esc in that
window stops the run; the arm is still parked on the way out.

Each step is one capture, one decision, one motion:

    park clear of the camera
        -> one frame -> snapshot + pointing overlay (the same pixels)
        -> Qwen3-VL picks an action and names a target
        -> validated against the snapshot (mt4_vision.instruct)
        -> motion.transfer / pick_at / place_at, off that same snapshot
        -> park clear of the camera again

then round again, until the model says DONE or anything refuses. Ids belong to
the snapshot that produced them, so the loop never carries one across a move --
the only thing that persists is what the gripper is holding.

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

**One snapshot, one id space.** The script owns the snapshot it decides over
and executes against, rather than deciding over its own and then calling the
MCP server's ``mt4_pick``, which would resolve the id in a *different* capture.
Ids are assigned by spatial order, so those two usually agree -- and "usually"
is the wrong guarantee when the failure is picking the wrong cube in silence.

Nothing here computes a coordinate or a wrist angle. Target selection is the
only thing the model does; every millimetre and every safety gate is the
existing stack's.

Prereqs:
  * the Qwen3-VL service reachable (``.\\scripts\\start_qwen_tunnel.ps1``)
  * arm free on the serial port, vision calibration present
"""

from __future__ import annotations

import argparse
import sys

import cv2

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision import instruct as I
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, CameraError
from mt4_vision.locate import LocateError, measure
from mt4_vision.motion import pick_at, place_at, transfer
from mt4_vision.pickplace import ensure_homed, retreat_for_camera
from mt4_vision.preview import (
    QWEN_BOUND_BGR,
    QWEN_REFUSED_BGR,
    LivePreview,
    PreviewStopped,
    annotate_qwen,
    gui_available,
)
from mt4_vision.qwen import QwenError, health

MAX_STEPS = 6

CAPTION_BGR = (255, 255, 255)
CAPTION_DIM_BGR = (190, 190, 190)

# Every outcome line carries this, because the loop reports what it SENT. There
# is no sensor in the jaws and no vision test that survived contact with a real
# desk: the template check that used to run here matched the ArUco tag beside a
# stapler and declared a successful pick a failure. Printing the caveat is the
# honest version of that -- silently reporting commands as observations is how
# a grasp that closed on air used to drive a confident DONE two steps later.
UNCHECKED = "  (commanded, not checked)"


def _publish(
    preview: LivePreview | None,
    obs: I.Observation,
    *,
    grounding=None,
    obj=None,
    action: I.Action | None = None,
    objects: dict | None = None,
    caption: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> None:
    """Show what the model just answered, on the frame it answered about.

    ``obs.annotated`` and not a fresh capture: the grid and the entity circles
    are what the model was looking at when it produced these coordinates, so
    they are the only backdrop against which the answer means anything.

    The bound entity's pixel comes from the snapshot, which is why markers and
    slots can be drawn here at all -- they had no pixel until
    ``entities._desk_pixel_projector``.

    ``objects`` lets a decision that targets a registered object show the mask
    the object's geometry was measured from, which is what its size, yaw and
    ``grip_mm`` all came from. That mask is one capture old by then -- the loop
    re-observes after registering -- so it is drawn on a frame it was not cut
    from. With the arm parked between two captures seconds apart nothing has
    moved, and seeing which silhouette is about to be grasped is worth more than
    the strictness. It is also the last look anyone gets: nothing downstream
    re-measures, and nothing after the move checks the desk again.
    """
    if preview is None:
        return
    bound = dest_bound = None
    if action is not None and action.entity_id:
        entity = obs.snapshot.get(action.entity_id)
        if entity is not None:
            bound = entity.pixel
        if obj is None and objects:
            obj = objects.get(action.entity_id)
    if action is not None and action.dest_entity_id:
        dest = obs.snapshot.get(action.dest_entity_id)
        if dest is not None:
            dest_bound = dest.pixel
    preview.show(
        annotate_qwen(
            obs.annotated,
            grounding=grounding,
            obj=obj,
            action=action,
            bound_px=bound,
            dest_bound_px=dest_bound,
            accepted=None if action is None else action.ok,
            caption=caption,
        )
    )


def _describe(action: I.Action) -> str:
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


def run(
    instruction: str,
    client: Mt4Client | None,
    *,
    camera: int,
    max_steps: int = MAX_STEPS,
    save_view: str | None = None,
    preview: LivePreview | None = None,
) -> bool:
    """One instruction to completion. True when the model reported DONE."""
    held: str | None = None
    history: list[str] = []
    # Anything Qwen named that the cube detector cannot see, measured once and
    # then carried as obj_N. This is what makes the loop work on a stapler and
    # not only on coloured cubes: the entity layer treats a registered object
    # exactly like a cube from here on, same gates, same motion.
    objects: dict[str, object] = {}
    obj_seq = 0
    for step in range(1, max_steps + 1):
        if client is not None:
            # The arm is in shot otherwise, and it shadows exactly the part of
            # the desk it just worked on.
            retreat_for_camera(client, I.load_calibration())

        obs = I.observe(
            camera, held=held, history=history, objects=objects, token=f"s{step}"
        )
        if save_view:
            cv2.imwrite(save_view, obs.annotated)
        print(f"\n[{step}] {obs.snapshot.summary}")
        _publish(
            preview, obs,
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
            if any(noun in e.label.lower() for e in obs.snapshot.entities):
                continue
            grounding, why = I.locate_target(obs, noun)
            if grounding is None:
                print(f"    could not find a {noun}: {why}")
                _publish(
                    preview, obs,
                    caption=[
                        (f"[{step}] {instruction}", CAPTION_BGR),
                        (f'locate "{noun}": {why}', QWEN_REFUSED_BGR),
                    ],
                )
                continue
            _publish(
                preview, obs, grounding=grounding,
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
                print(f"    found a {noun} but could not measure it: {why}")
                _publish(
                    preview, obs, grounding=grounding,
                    caption=[
                        (f"[{step}] {instruction}", CAPTION_BGR),
                        (f"found a {noun} but could not measure it: {why}",
                         QWEN_REFUSED_BGR),
                    ],
                )
                continue
            # On the frame the mask was actually cut from, before the re-observe
            # below replaces it. This is the one moment the segmentation can be
            # judged against the exact pixels it was computed on.
            _publish(
                preview, obs, grounding=grounding, obj=obj,
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
                print(
                    f"    registered {eid}: {noun} at ({obj.x:.0f}, "
                    f"{obj.y:.0f}) {obj.long_mm:.0f}x{obj.short_mm:.0f}mm"
                )
            else:
                # A merge changes the entity's LABEL, and the label is what
                # `unmatched_nouns` reads. Skipping the rebuild here left the
                # second word of a two-word target still unmatched, so the task
                # was refused for naming something already registered.
                print(f"    the {noun} is {eid} ({objects[eid].label})")
            obs = I.observe(
                camera, held=held, history=history, objects=objects,
                token=f"s{step}b",
            )

        action = I.decide(obs, instruction)
        print(f"    {_describe(action)}")
        print(f"    reason: {action.reason}")
        _publish(
            preview, obs, action=action, objects=objects,
            caption=[
                (f"[{step}] {instruction}", CAPTION_BGR),
                (_describe(action), QWEN_BOUND_BGR if action.ok else QWEN_REFUSED_BGR),
                (action.reason, CAPTION_DIM_BGR),
            ],
        )

        if action.kind == "DONE":
            return True
        if not action.ok:
            print("    -> refused, stopping. The scene or the request needs to change.")
            return False
        if client is None:
            print("    -> --dry-run, not moving")
            return False

        if action.kind == "LOCATE_AT_PIXEL":
            # Measured against the exact frame the model looked at, not a fresh
            # capture, so the pixel means what it saw. The hint only has to
            # land ON the object -- segmentation recovers the real centre, long
            # axis and size in millimetres.
            px, py = action.point_px
            try:
                obj = measure(
                    obs.frame, px, py, obs.calib, action.label or "object",
                    marker_xy=[
                        (e.x, e.y) for e in obs.snapshot.entities if e.kind == "marker"
                    ],
                )
            except LocateError as exc:
                print(f"    -> could not measure it: {exc}")
                return False
            obj_seq += 1
            objects[f"obj_{obj_seq}"] = obj
            history.append(
                f"found the {action.label} on the desk and measured it "
                f"({obj.long_mm:.0f}x{obj.short_mm:.0f}mm)"
            )
            print(
                f"    -> registered obj_{obj_seq}: {action.label} at "
                f"({obj.x:.0f}, {obj.y:.0f}) {obj.long_mm:.0f}x{obj.short_mm:.0f}mm"
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

        try:
            if action.kind == "TRANSFER":
                dest = obs.snapshot.get(action.dest_entity_id)
                dest_grasp = I.grasp_for(dest, obs.calib)
                print(
                    f"    -> pick ({grasp.x:.0f}, {grasp.y:.0f}) {yaw}"
                    f"  ->  place ({dest_grasp.x:.0f}, {dest_grasp.y:.0f}) squared"
                )
                # One planned operation: the lift out of the pick and the carry
                # are the same leg, so the arm never stops between having the
                # object and being on its way. Splitting this into pick, park,
                # look, decide, place cost two captures and a second decision
                # and could not answer the only question that mattered anyway.
                transfer(client, obs.calib, grasp, dest_grasp)
                retreat_for_camera(client, obs.calib)
                # The move invalidated this snapshot at both ends: the object
                # left its position and the destination gained an occupant.
                objects.pop(action.entity_id, None)
                moved = action.label or "object"
                # By label, not id. The destination keeps its printed number
                # ("marker 2") across snapshots, while marker_N ids do not.
                history.append(f"moved the {moved} onto {action.dest_label}")
                print(f"    -> moved the {moved} onto {action.dest_label}{UNCHECKED}")
            elif action.kind == "PICK_ENTITY":
                print(f"    -> pick ({grasp.x:.0f}, {grasp.y:.0f}) {yaw}")
                pick_at(client, obs.calib, grasp)
                # The label, never the id. The id described a position in the
                # snapshot that has just been invalidated by this very move.
                held = action.label or "object"
                # It is in the jaws, not on the desk, so it stops being an
                # entity. Leaving it listed offered the model its own held
                # object as a place destination -- measured: "put the stapler
                # on marker 1" came back as PLACE_ENTITY obj_1, the stapler.
                objects.pop(action.entity_id, None)
                history.append(f"picked up the {held}")
                print(f"    -> holding the {held}{UNCHECKED}")
            else:
                print(f"    -> place ({grasp.x:.0f}, {grasp.y:.0f}) squared")
                place_at(client, obs.calib, grasp)
                retreat_for_camera(client, obs.calib)
                history.append(f"put the {held} down on {action.label}")
                print(f"    -> put the {held} down on {action.label}{UNCHECKED}")
                held = None
        except Mt4ClientError as exc:
            print(f"    -> motion failed: {exc}")
            return False

    print(f"    -> gave up after {max_steps} steps")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instruction", nargs="*", help="what to do; omit for interactive")
    ap.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    ap.add_argument("--dry-run", action="store_true", help="decide, never move")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--save-view", default=None, help="write the frame the model saw")
    ap.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True,
        help="live window: the frame the model saw, with its answer drawn on it, "
             "q or Esc in the window stops the run (default: on)",
    )
    args = ap.parse_args()

    try:
        health()
    except QwenError as exc:
        print(f"qwen service: {exc}", file=sys.stderr)
        return 1

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
    preview = None
    if args.preview:
        if gui_available():
            preview = LivePreview("mt4 -- what qwen said (q or Esc to stop)")
        else:
            print(
                "no OpenCV GUI here, so no preview window (is "
                "opencv-python-headless installed?). --save-view still writes "
                "the frame the model saw.",
                file=sys.stderr,
            )

    try:
        if args.instruction:
            ok = run(
                " ".join(args.instruction), client,
                camera=args.camera, max_steps=args.max_steps,
                save_view=args.save_view, preview=preview,
            )
            return 0 if ok else 2
        while True:
            try:
                text = input("\ninstruction> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text in ("quit", "exit"):
                return 0
            run(
                text, client, camera=args.camera,
                max_steps=args.max_steps, save_view=args.save_view,
                preview=preview,
            )
    except PreviewStopped:
        # Closing the window is a stop request, not a crash. The finally below
        # still parks the arm.
        print("\npreview closed -- stopping", file=sys.stderr)
        return 1
    except (CameraError, KeyboardInterrupt) as exc:
        print(f"\n{exc or 'interrupted'}", file=sys.stderr)
        return 1
    finally:
        if preview is not None:
            preview.close()
        if client is not None:
            try:
                retreat_for_camera(client, I.load_calibration())
            except Mt4ClientError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
