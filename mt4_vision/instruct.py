"""Natural-language instruction -> one validated action on one entity.

The policy layer. It decides *what to do next and to which thing*; every
millimetre, every safety gate and every motion still belongs to the existing
stack. Nothing here computes a robot coordinate.

Two deliberate choices, both from measurement rather than taste.

**The model must both name and point, and the two must agree.** Naming alone
leans on attribute-to-instance binding, which this build does badly -- it
labelled a red cube "green" while boxing it perfectly (docs/QWEN3-VL.md).
Pointing alone was the first design here, on the theory that localization is
the stronger skill. Measured on the live desk 2026-08-02, that theory is only
half right: asked for "the red cube" the model named ``cube_7`` correctly and
pointed 151px away from it, and asked for a stapler that was not on the desk at
all it pointed close enough to a blue cube for a nearest-entity rule to bind
it and report success.

So neither signal is trusted alone. The id must exist in the current snapshot,
the point must land on an entity, and they must be the *same* entity;
disagreement is treated as ambiguity and refused. Two independent channels
agreeing is a far stronger claim than either one, and the cost of the extra
strictness is a retry, which is the cheap failure.

**Coordinates are 0-1000, not pixels.** Measured on this build, 8 replies out
of 8: a box reported at x=807 sits at 1033px in a 1280-wide frame. Naively
treating the reply as pixels lands the point 206-285px away -- still on the
desk, so segmentation happily measures *something* and returns a confident
wrong answer. :func:`to_frame_pixels` is the only place that conversion
happens, and :func:`alternate_reading` is the only sanctioned way to second-
guess it.

**Grounding asks for a box, not a point.** ``bbox_2d`` unlocks GrabCut
segmentation, bounds the mask, and gives an extent that can be sanity-checked;
a point gives none of the three. See :class:`Grounding`.

One capture per decision. ``mt4_scene`` and ``mt4_camera_view`` each grab their
own frame, so the ids drawn on the overlay come from a different exposure than
the entity list; harmless for a still desk, wrong for point-and-bind.
:func:`observe` does it once.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from mt4_vision.calib import Calibration, load_calibration
from mt4_vision.camera import capture_frame
from mt4_vision.entities import Entity, Snapshot, build_snapshot
from mt4_vision.discover import MERGE_MM
from mt4_vision.preview import BIG_BOX_SHARE, annotate_for_pointing
from mt4_vision.qwen import DEFAULT_URL, QwenError, ask
from mt4_vision.scene import capture_scene
from mt4_vision.workspace import PICK_CLEARANCE_MM

# The model's coordinate space. Not pixels -- see the module docstring.
COORD_SCALE = 1000.0
# A point must land this close to an entity to name it, and no second entity
# may be this close. Expressed in millimetres and converted to pixels at the
# candidate's own position, because this camera is oblique and mm-per-pixel
# varies across the frame. PICK_CLEARANCE_MM is the right scale by
# construction: two objects nearer than the fingers need are not independently
# pickable anyway, so a point between them is a real ambiguity, not a
# resolution problem.
BIND_RADIUS_MM = PICK_CLEARANCE_MM
# Short replies only. A decision is action + point + a clause of reason; the
# 1664-token static cache holds one image plus this comfortably, and anything
# longer means the model started narrating.
MAX_NEW_TOKENS = 220
# Two grounded measurements this close describe one physical thing. THE same
# constant discover uses, not a copy of it: these are the two paths that can add
# an object to the world, and they have to agree about when two hits are one
# object or the same stone becomes obj_3 and obj_4.
DUPLICATE_MM = MERGE_MM

# TRANSFER leads because it is the shape of nearly every real task here: move
# a named thing to a named place. Splitting that into PICK then PLACE cost a
# park, a capture and a second decision in between, and bought nothing -- once
# the object is in the jaws there is nothing left to see that changes where it
# should go. PICK_ENTITY and PLACE_ENTITY remain for the halves that occur on
# their own: "pick up the stapler" with no destination, and putting down
# something the gripper was already holding when the run started.
ACTIONS = (
    "TRANSFER", "PICK_ENTITY", "PLACE_ENTITY", "LOCATE_AT_PIXEL", "DONE", "STOP",
)


@dataclass(frozen=True)
class Observation:
    """One capture: the frame, what is on it, and what the arm is holding."""

    frame: np.ndarray = field(repr=False)
    annotated: np.ndarray = field(repr=False)
    snapshot: Snapshot
    calib: Calibration
    held: str | None = None
    # What this task has already done, oldest first, in prose. Without it the
    # loop is memoryless and a completed task looks exactly like an untouched
    # one: measured on a live run, "pick up the green cube and place it on
    # marker 2" succeeded at step 2 and was then carried out twice more,
    # because every fresh observation showed an empty gripper, a green cube on
    # the desk, and that same instruction. The model was right every time.
    history: tuple[str, ...] = ()
    # The detections this snapshot was built from, kept so the entity list can
    # be rebuilt without going back to the camera -- see :meth:`relisted`.
    scene: Any = None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frame.shape[:2]
        return w, h

    def relisted(self, *, objects: Any = (), token: str = "s1") -> "Observation":
        """This same frame, re-listed with a newly registered object in it.

        Registering an object has to get it into the entity list before the
        decision, and the obvious way to do that is to observe again. But
        ``capture_frame`` reopens the camera and burns 20 warm-up reads --
        2-3 SECONDS -- and it does it once per unmatched noun, while the arm is
        parked and nothing on the desk has moved. The frame that would be
        thrown away is the one already in hand.

        Rebuilding the snapshot and the overlay from it instead costs about
        5ms. Same pixels, so the ids the model is about to answer with still
        describe the frame it is looking at, which is the property the whole
        one-snapshot-one-id-space rule depends on.
        """
        snapshot = build_snapshot(self.scene, token=token, objects=objects)
        return replace(
            self,
            snapshot=snapshot,
            annotated=annotate_for_pointing(self.frame, snapshot.entities),
        )


@dataclass(frozen=True)
class Action:
    """A validated decision, or a refusal.

    ``ok`` means the decision is safe to act on: an entity action that passed
    every check, or ``DONE``. Every ``STOP`` is ``ok=False``, whether the model
    chose it or validation forced it, so a caller has one thing to branch on.
    """

    kind: str
    ok: bool
    reason: str
    entity_id: str | None = None
    label: str | None = None
    point_px: tuple[float, float] | None = None
    # The same reply's point under the other coordinate convention, when there
    # is one. Never acted on -- it exists so the preview can show both and a
    # reader can see at a glance which one the stack believed.
    alt_point_px: tuple[float, float] | None = None
    # What the model itself claimed the id was, before binding. Kept to measure
    # how often naming and pointing agree; never used to choose.
    model_entity_id: str | None = None
    # TRANSFER only: the destination, resolved by the same rules and against the
    # same snapshot as the source. Both halves have to hold before the arm moves
    # at all, which is the point of naming them together -- a transfer that is
    # going to be refused for its destination should be refused before the
    # object is in the jaws, not after.
    dest_entity_id: str | None = None
    dest_label: str | None = None
    dest_point_px: tuple[float, float] | None = None
    dest_alt_point_px: tuple[float, float] | None = None
    model_dest_entity_id: str | None = None
    raw: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.kind, "ok": self.ok, "reason": self.reason}
        if self.entity_id:
            out["entity_id"] = self.entity_id
        if self.label:
            out["label"] = self.label
        if self.point_px:
            out["point_px"] = [round(self.point_px[0], 1), round(self.point_px[1], 1)]
        if self.alt_point_px:
            out["alt_point_px"] = [
                round(self.alt_point_px[0], 1), round(self.alt_point_px[1], 1)
            ]
        if self.model_entity_id:
            out["model_entity_id"] = self.model_entity_id
            out["agreed"] = self.model_entity_id == self.entity_id
        if self.dest_entity_id:
            out["dest_entity_id"] = self.dest_entity_id
        if self.dest_label:
            out["dest_label"] = self.dest_label
        if self.dest_point_px:
            out["dest_point_px"] = [
                round(self.dest_point_px[0], 1), round(self.dest_point_px[1], 1)
            ]
        if self.dest_alt_point_px:
            out["dest_alt_point_px"] = [
                round(self.dest_alt_point_px[0], 1),
                round(self.dest_alt_point_px[1], 1),
            ]
        if self.model_dest_entity_id:
            out["model_dest_entity_id"] = self.model_dest_entity_id
            out["dest_agreed"] = self.model_dest_entity_id == self.dest_entity_id
        return out


def observe(
    camera: int = 1,
    *,
    held: str | None = None,
    history: Sequence[str] = (),
    objects: Any = (),
    token: str = "s1",
) -> Observation:
    """One frame -> snapshot and pointing overlay built from the same pixels."""
    calib = load_calibration()
    frame = capture_frame(camera)
    scene = capture_scene(calib, frame)
    snapshot = build_snapshot(scene, token=token, objects=objects)
    annotated = annotate_for_pointing(frame, snapshot.entities)
    return Observation(
        frame=frame, annotated=annotated, snapshot=snapshot, calib=calib,
        held=held, history=tuple(history), scene=scene,
    )


# -- prompt ---------------------------------------------------------------


def _entity_lines(snapshot: Snapshot) -> str:
    rows = []
    for e in snapshot.entities:
        caps = []
        if e.pickable:
            caps.append("can be picked up")
        if e.placeable:
            caps.append("can be placed on")
        if not caps:
            caps.append(f"not available ({e.reason or 'no reason given'})")
        where = "" if e.pixel is None else f" at image point ({e.pixel[0]:.0f}, {e.pixel[1]:.0f})"
        rows.append(f"  {e.id}: {e.label}{where} -- {', '.join(caps)}")
    return "\n".join(rows) or "  (nothing detected)"


def build_prompt(obs: Observation, instruction: str) -> str:
    """The decision prompt.

    The schema is spelled out and prose is forbidden, because that is what was
    measured to work: "identify all objects" returned prose 3/3 and "reply in
    JSON" returned JSON of the wrong shape 3/3, while naming the keys returned
    the right shape 3/3 with no constrained decoding.
    """
    w, h = obs.size
    held = (
        "The gripper is empty."
        if not obs.held
        else (
            f"The gripper is already holding a {obs.held}, so the pick half of "
            "the task is done -- what remains is where to put it down. "
            "PLACE_ENTITY names the DESTINATION, which is always a marker_N or "
            "a slot_N, never the thing being held. TRANSFER is not available "
            "while the gripper is full."
        )
    )
    # No id for the held object, deliberately. Ids are assigned by spatial
    # order over what is on the desk, so they renumber the moment anything
    # moves -- and the held object has left the desk entirely. Naming it by its
    # previous id got that id reused against a different cube in the next
    # snapshot: measured on a live two-step run, "holding cube_3 (red cube)"
    # came back as PLACE_ENTITY cube_3, by then a blue cube.
    if obs.history:
        done = "\n".join(f"  {i}. {s}" for i, s in enumerate(obs.history, 1))
        progress = (
            f"Already carried out for this task, in order:\n{done}\n"
            "If those steps have completed the task, the answer is DONE. Do not "
            "repeat work that is already done -- check the desk against the "
            "task before choosing to act again.\n"
        )
    else:
        progress = "Nothing has been done for this task yet.\n"
    return (
        "A robot arm is working on this desk. Choose the single next action.\n\n"
        f'Task: "{instruction}"\n'
        f"{progress}"
        f"{held}\n\n"
        "Things the vision system has found in THIS image. These ids were "
        "assigned just now and mean nothing in any earlier step -- use only "
        "what is listed here. Each is circled in the image with its id "
        "written beside it:\n"
        f"{_entity_lines(obs.snapshot)}\n\n"
        "Choose one action:\n"
        "  TRANSFER        - move something onto somewhere, in one go: "
        "entity_id is the thing to pick up, dest_entity_id is the marker_N or "
        "slot_N it goes on. Use this whenever the task names BOTH a thing and "
        "a place for it, and the gripper is empty\n"
        "  PICK_ENTITY     - pick something up and hold it, when the task names "
        "no destination at all (only if the gripper is empty)\n"
        "  PLACE_ENTITY    - put down what is ALREADY held, onto a marker_N or "
        "a slot_N\n"
        "  LOCATE_AT_PIXEL - the task names something NOT in the list above; "
        "point at it so it can be measured\n"
        "  DONE            - the task is complete\n"
        "  STOP            - the task cannot be done, or the target is ambiguous\n\n"
        "Reply with ONLY a JSON object, no prose, no markdown fence:\n"
        '{"action": "...", "entity_id": "...", "point_2d": [x, y], '
        '"dest_entity_id": "...", "dest_2d": [x, y], '
        '"label": "...", "reason": "<one short clause>"}\n\n'
        "entity_id must be copied exactly from the list above, or null if the "
        "task names something that is not in it.\n"
        "dest_entity_id and dest_2d are for TRANSFER only -- the destination, "
        "copied from the list the same way. Use null for every other action.\n"
        # Both destination kinds print as "can be placed on" and nothing else
        # distinguishes them, so a task that rules one out has no way to be
        # obeyed. Measured live 2026-08-02: "place it on a non-marker location"
        # answered PLACE_ENTITY marker_0.
        "A marker_N is a printed paper tag on the desk. A slot_N is a bare spot "
        "on the desk with no marker on it. If the task rules one out -- 'not on "
        "a marker', 'somewhere clear', 'a non-marker location' -- use the other "
        "kind.\n"
        f"point_2d is the centre of that same thing, in PIXELS of this "
        f"{w}x{h} image -- read them off the numbered grid drawn on it. Do not "
        "normalize or rescale. Required for TRANSFER, PICK_ENTITY, "
        "PLACE_ENTITY and LOCATE_AT_PIXEL; use null for DONE and STOP. dest_2d "
        "is the centre of dest_entity_id, in that same pixel space.\n"
        "label is a short noun, only for LOCATE_AT_PIXEL.\n"
        "Begin your reply with { and end it with }."
    )


# -- parsing and binding --------------------------------------------------


def _first_json_object(text: str) -> dict[str, Any] | None:
    """The first balanced ``{...}`` that parses. Fenced or bare, prose or not."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    got = json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    start = -1
                    continue
                if isinstance(got, dict):
                    return got
    return None


def to_frame_pixels(
    coords: Sequence[float], size: tuple[int, int]
) -> tuple[float, ...]:
    """Model coordinates as frame pixels. The only place this convention lives.

    Takes a point ``(x, y)`` or a box ``(x1, y1, x2, y2)`` -- even indices are
    x, odd are y -- and returns the same shape.

    **Asked for a coordinate with nothing to anchor it, this build answers in
    0-1000 normalized space whatever the prompt says.** Measured 2026-08-02
    against known pixel positions: four objects (three cubes from the HSV
    detector plus a hand-read stapler) x two prompt forms (``point_2d`` and
    ``bbox_2d``), on both the raw frame and the grid-annotated one. Eight
    replies out of eight normalized, landing 1-10px from truth. Read as pixels,
    those same replies land 206-285px away.

    Asking for pixels does not change it. The prompt used to say "coordinates
    are pixels of this 1280x720 image; the grid drawn on it is labelled in
    pixels", and the model answered normalized anyway -- the grid overlay makes
    no measurable difference either.

    **Showing it a pixel does.** Once the entity list started printing "at image
    point (x, y)" for every entity, the decision reply switched to pixels --
    measured 2026-08-03, 5/5, within 0.4px of the named entity's own pixel.
    A worked example in the prompt beats an instruction about the format. So
    this function is the right default for an *unanchored* answer only;
    ``decide`` resolves its point through :func:`point_readings` instead.

    Nor does an in-frame test, which is what this function used to do, and
    which is the bug this replaces. A normalized coordinate is *almost always*
    in frame at this resolution: x <= 1000 < 1280 always, and y <= 720 over the
    lower 72% of the range. "Use as pixels when it fits, rescale otherwise"
    therefore fires the wrong way on nearly every reply. That is what sent
    "pick up the stapler" to (700, 687) -- bare desk near the bottom edge, 280px
    from the stapler -- where it failed as "could not segment an object". The
    quieter version of the same bug lands on a *different object* and measures
    it confidently.

    The one reading that can be ruled out from the numbers alone is normalized:
    no normalized coordinate exceeds ``COORD_SCALE``. So a coordinate above it
    means pixels, and everything else is scaled. That leaves a genuine blind
    spot -- a true pixel answer whose coordinates all happen to fall under 1000
    -- so callers that can afford a retry should offer
    :func:`alternate_reading` when the first reading fails to measure.
    """
    values = [float(c) for c in coords]
    if any(abs(v) > COORD_SCALE for v in values):
        return tuple(values)
    w, h = size
    sx, sy = w / COORD_SCALE, h / COORD_SCALE
    return tuple(v * (sx if i % 2 == 0 else sy) for i, v in enumerate(values))


def point_readings(
    point: Sequence[float], size: tuple[int, int]
) -> tuple[tuple[float, float], ...]:
    """Both in-frame readings of a **decision** reply's point, best first.

    The convention is a property of the prompt, not of the model, and the two
    prompts in this module differ:

    * :func:`locate_target` deliberately says nothing about coordinate space, so
      the model falls back to its trained default and answers 0-1000 normalized.
      Measured 8/8 (see :func:`to_frame_pixels`).
    * :func:`build_prompt` asks for pixels **and prints every entity's own pixel
      in the list**, so the model has a worked example of the space in front of
      it. Measured 2026-08-03, five consecutive decisions on one frame: the
      reply was the named entity's pixel to **0.4px**, every time. Read as
      normalized, those same five land **160px** away.

    That second measurement is why this exists. Pixels-first was wrong before
    the entity list carried pixels, and is right now -- so both readings are
    returned, in order, and the caller resolves them against the snapshot rather
    than trusting either.

    Readings outside the frame are dropped, and an exact duplicate is returned
    once: a coordinate above ``COORD_SCALE`` rules normalized out, so there is
    only one reading to have.
    """
    try:
        values = [float(point[0]), float(point[1])]
    except (TypeError, ValueError, IndexError):
        return ()
    w, h = size
    out: list[tuple[float, float]] = []

    def keep(candidate: tuple[float, float] | None) -> None:
        if candidate is None:
            return
        if not (0 <= candidate[0] <= w and 0 <= candidate[1] <= h):
            return
        if candidate not in out:
            out.append(candidate)

    keep((values[0], values[1]))
    scaled = to_frame_pixels(values, size)
    keep((float(scaled[0]), float(scaled[1])))
    return tuple(out)


def alternate_reading(
    coords: Sequence[float], size: tuple[int, int]
) -> tuple[float, ...] | None:
    """The other possible reading of the same numbers, or None if there is none.

    Only ever raw-as-pixels, and only when that lands inside the frame:
    :func:`to_frame_pixels` already returns raw pixels whenever a coordinate
    rules normalized out, so in that case there is nothing else to try.

    This exists so a failed measurement can be retried rather than reported,
    without ever *silently* choosing between the two -- the retry still has to
    survive segmentation, the two-window stability check and the work-region
    gate before it becomes a target.
    """
    values = [float(c) for c in coords]
    if any(abs(v) > COORD_SCALE for v in values):
        return None
    w, h = size
    if all(0 <= v <= w for v in values[0::2]) and all(
        0 <= v <= h for v in values[1::2]
    ):
        return tuple(values)
    return None


def _mm_to_px(calib: Calibration, x: float, y: float, mm: float) -> float:
    """How many pixels ``mm`` spans at robot (x, y). Position-dependent on an
    oblique mount, so it is asked per candidate rather than assumed once."""
    ax, ay = calib.robot_to_pixel(x, y)
    bx, by = calib.robot_to_pixel(x + mm, y)
    cx, cy = calib.robot_to_pixel(x, y + mm)
    return max(math.hypot(bx - ax, by - ay), math.hypot(cx - ax, cy - ay))


def bind(
    obs: Observation, point_px: tuple[float, float], *, kinds: Sequence[str] | None = None
) -> tuple[Entity | None, str]:
    """The entity this point names, or (None, why not).

    Refuses on ambiguity rather than taking the nearest, because taking the
    nearest is exactly the silent wrong-target failure the entity layer exists
    to prevent.
    """
    cands: list[tuple[float, float, Entity]] = []
    for e in obs.snapshot.entities:
        if kinds is not None and e.kind not in kinds:
            continue
        if e.pixel is None:
            # Slots are nominal positions with no detection pixel; project one.
            ex, ey = obs.calib.robot_to_pixel(e.x, e.y)
        else:
            ex, ey = e.pixel
        d_px = math.hypot(point_px[0] - ex, point_px[1] - ey)
        cands.append((d_px, _mm_to_px(obs.calib, e.x, e.y, BIND_RADIUS_MM), e))
    if not cands:
        return None, "no entities of the required kind in this snapshot"

    cands.sort(key=lambda c: c[0])
    d0, r0, best = cands[0]
    if d0 > r0:
        return None, (
            f"the point ({point_px[0]:.0f}, {point_px[1]:.0f}) is {d0:.0f}px from "
            f"the nearest entity {best.id} ({best.label}), past the "
            f"{r0:.0f}px that {BIND_RADIUS_MM:.0f}mm spans there -- it does not "
            "name anything in the list"
        )
    if len(cands) > 1:
        d1, _r1, second = cands[1]
        if d1 <= r0:
            return None, (
                f"ambiguous: the point is {d0:.0f}px from {best.id} "
                f"({best.label}) and {d1:.0f}px from {second.id} "
                f"({second.label}), both within {r0:.0f}px -- refusing rather "
                "than guessing which was meant"
            )
    return best, ""


def grasp_for(entity: Entity, calib: Calibration):
    """The motion-layer pose for acting on ``entity``, taken from the snapshot.

    No fresh capture. The loop parks the arm and captures once per step, the
    decision is made against that frame, and the arm then acts on the entity
    positions it produced. Nothing but the arm moves on this desk, and the arm
    is parked while the model thinks, so a second look before the grasp was
    re-measuring a scene that had not changed.

    What that second look used to buy was a template re-match for registered
    objects, and it was the wrong tool for the job. Measured 2026-08-03: a
    stapler lying against marker 1's paper stored a 59x59px template in which
    the ArUco tag covered 30% of the area and carried 87% of the pixel
    variance, the desk around it having a standard deviation of one grey level.
    The match therefore tracked the tag, not the stapler -- it "re-acquired" a
    lifted stapler onto the marker's own centre, 4mm out.

    Markers and slots are calibrated positions with nothing to detect, and a
    destination has no orientation of its own, so the landing yaw is squared to
    the world axes rather than left wherever the pick put the wrist. An object
    being picked does have an orientation, and keeps it.
    """
    from mt4_vision.motion import square_place

    if entity.kind in ("marker", "slot"):
        return square_place(entity.x, entity.y)
    return entity.as_grasp(calib)


def register_object(objects: dict, obj: Any, *, seq: int) -> tuple[str, bool]:
    """Add ``obj`` to the registry unless it is one already there. (id, is_new).

    A multi-word target grounds one word at a time and every word lands on the
    same thing. Measured 2026-08-02: "the grey rock next to the blue cube"
    registered ``obj_3`` ("grey") and ``obj_4`` ("rock") at the identical robot
    (109, 179). Two ids for one stone is not cosmetic -- the model then names
    one of them while the attribute check reads words that described the other,
    and the request is refused as self-contradictory.
    """
    for eid, known in objects.items():
        if (known.x - obj.x) ** 2 + (known.y - obj.y) ** 2 >= DUPLICATE_MM ** 2:
            continue
        # Fold the new word into the label rather than dropping it. Both words
        # described the same thing, so the entity should answer to both -- and
        # `unmatched_nouns` reads exactly these labels, so without this the
        # second word stays "unmatched" and the task is refused for naming
        # something that is in fact registered.
        extra = [
            w for w in obj.label.lower().split()
            if w not in known.label.lower().split()
        ]
        if extra:
            from dataclasses import replace as _replace

            merged = " ".join([*known.label.split(), *extra])
            try:
                objects[eid] = _replace(known, label=merged)
            except TypeError:
                pass
        return eid, False
    eid = f"obj_{seq}"
    objects[eid] = obj
    return eid, True


# An instruction that asks rather than orders. The action set has no way to
# answer one, so the model treats "is there anything on the desk that is not a
# cube" as a command and picks a cube -- measured 2026-08-02, and moving the
# arm in reply to a question is the wrong failure. Detected here rather than
# prompted away, because a deterministic guard cannot be talked out of it.
_INTERROGATIVE = frozenset(
    "is are was were do does did can could will would should "
    "what which where who whose how why when".split()
)


def is_question(instruction: str) -> bool:
    text = instruction.strip().lower()
    if text.endswith("?"):
        return True
    first = text.split()[0].strip(".,!?;:'\"") if text.split() else ""
    return first in _INTERROGATIVE


def named_destination_block_reason(
    instruction: str, snapshot: Snapshot, *, moving: str | None = None
) -> str | None:
    """Why the marker this task names cannot be placed on, or None.

    Checked *before* a pick, deterministically, from the printed tag number in
    the instruction. Without it the loop picks first and discovers at the
    place step that the destination was never available -- measured
    2026-08-02, "put the blue cube on marker 2" (occupied) returned
    ``PICK_ENTITY cube_2`` and would have stranded a cube in the jaws.

    ``moving`` is the entity about to be picked. A destination occupied by
    *that* entity is not blocked, it is already done: "put the green cube on
    marker 2" with the green cube sitting on marker 2 must not refuse itself.

    Silent when the task names no marker, which is most of them.
    """
    text = instruction.lower()
    for e in snapshot.entities:
        if e.kind != "marker":
            continue
        num = e.id.split("_")[-1]
        if f"marker {num}" not in text and f"marker{num}" not in text:
            continue
        if e.placeable or (moving is not None and e.holds == moving):
            continue
        return (
            f"{e.id} ({e.label}) cannot be placed on: "
            f"{e.reason or 'no reason given'}"
        )
    return None


# Kinds that exist to be placed *on*, never picked up. Naming one as the thing
# to pick is a category error, and the model resolves it by picking whatever it
# can instead -- measured 2026-08-02: "pick up slot 5" returned PICK_ENTITY
# cube_2, a silent substitution of exactly the sort this layer exists to stop.
_NEVER_PICKABLE_KINDS = ("marker", "slot")

# Ways a task rules markers out as a destination. Matched as substrings of the
# lowered instruction, so each entry covers its inflections.
_NO_MARKER_PHRASES = (
    "non-marker", "non marker", "nonmarker",
    "not a marker", "not on a marker", "not onto a marker", "not marker",
    "no marker", "without a marker", "other than a marker",
    "away from the marker", "off the marker", "off a marker",
    "avoid the marker", "avoiding the marker", "clear of the marker",
)


def excluded_destination_kind(instruction: str) -> str | None:
    """The destination kind this task forbids, or None.

    Both destination kinds reach the model as "can be placed on" and nothing
    else tells them apart, so a task that rules one out is asking for a
    distinction the entity list does not draw. Measured live 2026-08-02, "place
    it on a non-marker location" answered ``PLACE_ENTITY marker_0`` twice: once
    pointing at a second marker, and once -- after the prompt was taught what a
    slot is -- naming ``marker_0`` while pointing squarely at ``slot_4``.

    Reading the constraint here rather than trusting the prompt is the same
    discipline as everywhere else in this layer: the entity's kind is ground
    truth from the snapshot, so a task that names a kind can be checked against
    it instead of hoped for. Note this is *not* target substitution -- it never
    changes which physical thing is acted on beyond what the instruction itself
    states, and a destination that survives it still has to pass ``placeable``.
    """
    text = instruction.lower()
    if any(p in text for p in _NO_MARKER_PHRASES):
        return "marker"
    return None


def wrong_kind_block_reason(
    instruction: str, named: Entity | None, snapshot: Snapshot
) -> str | None:
    """Refuse a pick whose target the instruction never mentions.

    Fires only when both hold: the instruction names a never-pickable kind, and
    **no word of the chosen entity's own label appears in the instruction**.
    "pick up slot 5" choosing ``cube_2`` ("blue cube") matches neither "blue"
    nor "cube", so it is refused. "pick up the stapler and place it on marker 4"
    choosing ``obj_1`` ("stapler") matches "stapler", so it passes.

    The first version of this compared ``named.kind`` against the instruction
    instead of the label, and that was wrong in the one way that mattered: a
    registered object's kind is the word "object", which no real instruction
    ever contains. Every "pick up the <thing> and put it on marker N" was
    therefore refused -- the main non-cube use case, broken by the guard meant
    to protect it. Cubes hid the bug, because "cube" *is* both their kind and a
    word people type. Labels are what users actually name things by; kinds are
    an internal taxonomy.
    """
    if named is None:
        return None
    text = instruction.lower()
    asked = [k for k in _NEVER_PICKABLE_KINDS if k in text]
    if not asked:
        return None
    words = {
        w for w in named.label.lower().replace("(", " ").replace(")", " ").split()
        if w.isalpha() and len(w) > 2
    }
    if any(w in text for w in words):
        return None
    return (
        f"the task asks for a {asked[0]}, which is a place target and cannot "
        f"be picked up, and nothing in {named.id} ({named.label!r}) is named in "
        "the task -- name a pickable thing or say it cannot be done"
    )


@dataclass(frozen=True)
class Grounding:
    """Where the model says a named thing is, in frame pixels.

    ``box_px`` is the whole point of asking for a box rather than a point, and
    it buys three things a point cannot:

    * **GrabCut.** ``locate.measure_grabcut`` seeds a silhouette from the box.
      Measured on one live frame, the desk-deviation path that a bare point
      feeds segmented 1 of 4 objects; from the box, GrabCut segmented 4 of 4,
      and landed 6.3-12.4mm from where the HSV cube detector puts the same
      three cubes.
    * **A bound on the mask.** Desk-deviation floods into shadow and adjacent
      objects with nothing to stop it; the box says how far the object goes.
    * **A size check.** A box has an extent, so a reading that puts a stapler
      at 400mm long can be rejected before the arm moves. A point has no
      extent and cannot be sanity-checked at all.

    ``alt_point_px`` / ``alt_box_px`` are the same reply under the other
    coordinate convention -- see :func:`alternate_reading`. They are a retry,
    never a silent second choice.
    """

    label: str
    point_px: tuple[float, float]
    box_px: tuple[float, float, float, float] | None = None
    alt_point_px: tuple[float, float] | None = None
    alt_box_px: tuple[float, float, float, float] | None = None


def locate_target(
    obs: Observation, noun: str, *, url: str = DEFAULT_URL
) -> tuple[Grounding | None, str]:
    """Ask only "where is the <noun>", and return a :class:`Grounding` or why not.

    A separate, single-purpose call rather than another field on the decision
    prompt. Asked to choose an action *and* ground an unknown noun, this build
    reliably does neither -- it forces the task onto whatever cube it can see
    and explains in its reason why that is wrong. Asked nothing but "locate
    the stapler", with the keys spelled out, it is the grounding prompt that
    measured 10/10 (docs/QWEN3-VL.md).

    **The prompt deliberately says nothing about the coordinate space.** The
    model ignores that instruction either way (see :func:`to_frame_pixels`),
    and a stated convention it does not follow is worse than no statement at
    all -- it invites the reader to trust the wrong reading.
    """
    prompt = (
        f"Locate the {noun} in this image. If there is no {noun} visible, "
        'reply with an empty list [].\n'
        "Reply with ONLY JSON, no prose, no markdown fence:\n"
        f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{noun}"}}]\n'
        "One tight box around the whole object."
    )
    try:
        reply = ask(prompt, obs.annotated, url=url, max_new_tokens=120, do_sample=False)
    except QwenError as exc:
        return None, f"grounding call failed: {exc}"

    from mt4_vision.qwen import parse_regions

    regions = parse_regions(reply.text)
    # Boxes first: a point reply is still accepted, because the model
    # occasionally answers with one anyway, but it loses everything in the
    # Grounding docstring and is the weaker input.
    for r in sorted(regions, key=lambda g: 0 if g.kind == "box" else 1):
        prim = to_frame_pixels(r.coords, obs.size)
        alt = alternate_reading(r.coords, obs.size)
        g = _grounding(noun, r.kind, prim, alt, obs.size)
        if g is not None:
            return g, ""
    return None, f"no usable {noun} location in {reply.text.strip()[:120]!r}"


# A box larger than this share of the frame is the model declining to answer,
# not an object. Asked to locate something that is not there, this build
# sometimes returns the whole image rather than the empty list the prompt asks
# for -- measured on "location", box (0, 0, 1000, 1000), 100% of the frame.
#
# The plausibility band in ``locate.measure_box`` caught that one, but only
# after projecting it: a box spanning the frame includes pixels above the desk
# horizon, where this camera's homography diverges, so the refusal read "box
# measures 315037x312990mm". True, useless, and three layers from the cause.
#
# The threshold has room. The largest thing the stack will measure is
# ``MAX_PLAUSIBLE_LONG_MM`` = 200mm; at this mounting that is roughly a third of
# the frame's width, so well under a tenth of its area.
# THE threshold the overlay calls out at, not a copy: the picture and the
# refusal have to agree about what counts as "the model declined to answer".
MAX_BOX_FRAME_SHARE = BIG_BOX_SHARE


def _grounding(
    label: str,
    kind: str,
    prim: Sequence[float],
    alt: Sequence[float] | None,
    size: tuple[int, int],
) -> Grounding | None:
    """Assemble a Grounding, or None when the reading cannot be an object.

    Two rejections, both about the primary reading: a centre outside the frame,
    and a box covering most of it (see :data:`MAX_BOX_FRAME_SHARE`).
    """
    w, h = size

    def centre(c: Sequence[float]) -> tuple[float, float]:
        if len(c) == 2:
            return float(c[0]), float(c[1])
        return (float(c[0]) + float(c[2])) / 2, (float(c[1]) + float(c[3])) / 2

    def box(c: Sequence[float]) -> tuple[float, float, float, float] | None:
        if len(c) != 4:
            return None
        x1, y1, x2, y2 = (float(v) for v in c)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    cx, cy = centre(prim)
    if not (0 <= cx <= w and 0 <= cy <= h):
        return None
    prim_box = box(prim) if kind == "box" else None
    if prim_box is not None and w > 0 and h > 0:
        bx0, by0, bx1, by1 = prim_box
        if (bx1 - bx0) * (by1 - by0) > MAX_BOX_FRAME_SHARE * w * h:
            return None
    return Grounding(
        label=label,
        point_px=(cx, cy),
        box_px=prim_box,
        alt_point_px=None if alt is None else centre(alt),
        alt_box_px=None if alt is None or kind != "box" else box(alt),
    )


def measure_grounding(
    obs: Observation, g: Grounding, *, label: str | None = None
) -> tuple[Any | None, str]:
    """Turn a :class:`Grounding` into a measured object, or say why not.

    Prefers GrabCut from the box and falls back to the desk-deviation point
    path, which is what ``locate.measure_with_box_fallback`` already arranges.
    On failure it retries under the other coordinate reading rather than
    reporting -- the retry still has to survive segmentation, the two-window
    stability check and the work-region gate, so a wrong reading cannot buy
    itself a target by being tried twice.
    """
    from mt4_vision.locate import LocateError, measure_with_box_fallback

    name = label or g.label
    marker_xy = [(e.x, e.y) for e in obs.snapshot.entities if e.kind == "marker"]
    attempts: list[tuple[str, tuple[float, float], tuple[float, ...] | None]] = [
        ("", g.point_px, g.box_px)
    ]
    if g.alt_point_px is not None and g.alt_point_px != g.point_px:
        attempts.append(("alternate coordinate reading: ", g.alt_point_px, g.alt_box_px))

    why = ""
    for prefix, point, bx in attempts:
        try:
            obj = measure_with_box_fallback(
                obs.frame, point[0], point[1], obs.calib, name,
                box=None if bx is None else (bx[0], bx[1], bx[2], bx[3]),
                marker_xy=marker_xy,
            )
        except LocateError as exc:
            why = why or f"{prefix}{exc}"
            continue
        return obj, ""
    return None, why or f"nothing measurable at the {name} location"


def noun_phrase(instruction: str, head: str) -> str:
    """The run of words ending at ``head`` that could be describing it.

    A crude noun-phrase bracket, and crude is the point: it walks left from the
    head noun while the words are plausible modifiers (short, alphabetic, not
    a preposition or verb) and stops at the first that is not. "the grey rock
    next to the blue cube" with head "rock" gives "the grey rock", so "blue"
    stays out.

    Without this, attributes were harvested from the whole sentence, and a
    landmark mentioned only to disambiguate became a requirement of the target.
    Measured 2026-08-02: "pick up the grey rock next to the blue cube" was
    refused as ``the task says ['blue', 'grey', 'rock'] but obj_5 is 'rock'``
    -- a correct-looking sentence about a contradiction that was never in the
    instruction.

    Returns ``""`` when the head does not appear at all. That is an abstention,
    not a fallback to the whole sentence: if the instruction never mentions
    what this entity is called, no word in it can be shown to describe the
    entity, and harvesting the sentence anyway manufactures a contradiction.
    Measured 2026-08-02: the model named ``obj_2`` (label "stone") for "the
    grey rock", and the whole-sentence fallback refused it as ``the task says
    ['blue', 'grey', 'rock'] but obj_2 is 'stone'`` -- three words, none of
    which the instruction ever claimed about a stone.
    """
    words = instruction.lower().replace(",", " ").split()
    stripped = [w.strip(".!?;:'\"") for w in words]
    try:
        end = stripped.index(head)
    except ValueError:
        return ""
    start = end
    # Anything that ends a modifier run: prepositions, conjunctions and verbs
    # all live in _FILLER, but so do articles, which must NOT end it.
    articles = {"the", "a", "an", "its", "this", "that", "these", "those"}
    while start > 0:
        w = stripped[start - 1]
        if w in articles:
            start -= 1
            continue
        if not w.isalpha() or w in _FILLER:
            break
        start -= 1
    return " ".join(stripped[start : end + 1])


def instruction_attributes(
    instruction: str, snapshot: Snapshot, *, head: str | None = None
) -> set[str]:
    """Ground-truth-checkable words the instruction uses to name its target.

    Only words the *vision system* already knows the answer to: cube colours
    come from HSV, marker numbers from the printed ArUco tag. Those are facts,
    not opinions, so a model naming an entity whose own label contradicts them
    can be refused outright rather than trusted.

    Deliberately narrow. "the leftmost one" or "the one by the stapler" are not
    checkable this way and yield nothing, which is the honest answer -- the
    check abstains instead of guessing.

    ``head`` narrows the search to the noun phrase around that word (see
    :func:`noun_phrase`), so a landmark elsewhere in the sentence does not
    become a requirement of the target. Marker numbers are still read from the
    whole instruction, because "put the blue cube on marker 3" names the
    destination outside the target's phrase by construction.
    """
    text = instruction.lower() if head is None else noun_phrase(instruction, head)
    whole = instruction.lower()
    # Words naming the *kind* discriminate nothing -- every cube's label
    # contains "cube", so counting it made a blue cube satisfy "the red cube"
    # and turned the check into a no-op for exactly the mistake it exists to
    # catch. Only attributes that can tell two entities of a kind apart.
    generic = {e.kind for e in snapshot.entities} | {"open", "slot", "marker"}
    vocab = {
        w for e in snapshot.entities for w in e.label.lower().split()
    } - generic
    found = {w for w in vocab if w.isalpha() and len(w) > 2 and w in text}
    for e in snapshot.entities:
        if e.kind == "marker":
            num = e.id.split("_")[-1]
            if f"marker {num}" in whole:
                found.add(f"marker{num}")
    return found


# Words that cannot name a physical target, so they must never reach the
# grounding call. This list is load-bearing, not cosmetic. Measured over a
# 28-prompt sweep 2026-08-02, the previous shorter list let "somewhere",
# "clear", "tidy", "anything", "everything", "there" and "not" through -- and
# Qwen answers a grounding request for an abstract noun rather than declining
# it. Two of them were then *registered as physical objects*: "clear" at robot
# (24, -335) 83x50mm and "there" at (-293, 6) 55x33mm. Others came back as
# whole-frame boxes that projected to 315037x312990mm and were only caught by
# the plausibility band.
#
# Each bogus grounding also costs a model call (~1.5s) and produces a refusal
# whose stated reason is nonsense: "the task names ['tidy'], which nothing in
# this snapshot is -- register it with LOCATE_AT_PIXEL first".
#
# The largest single group is **place-holder nouns for a position** -- location,
# spot, position, space, site, square, gap. They name where something goes, and
# the loop already has a kind for that (slot_N). Measured live 2026-08-02, "pick
# up the green cube and place it on a non-marker location" grounded "location",
# got a whole-frame box back, projected it to 315037x312990mm, and refused the
# whole task as "the task names ['location'], which nothing in this snapshot is"
# -- on an instruction whose destination the snapshot listed 14 candidates for.
#
# Erring long is safe here. A real object noun wrongly listed here degrades to
# "not in the snapshot" -- the same refusal the loop already gives for anything
# it cannot see -- while a filler word wrongly omitted invents an entity.
_FILLER = frozenset(
    """
    pick place put set drop move take grab lift bring return leave keep
    carry fetch hold release give send push pull swap stack unstack tidy
    clean clear sort arrange organise organize rearrange shift transfer
    the a an it its them they this that those these one ones thing things
    something anything everything nothing anywhere somewhere everywhere
    nowhere there here where what which who how why when
    and then also next after before while during until once
    onto into over above upon under below beneath through across along
    down out off from with without for near beside between among around
    up top left right front back rear middle centre center side edge corner
    nearest closest furthest farthest leftmost rightmost topmost nearer
    further farther other another same different both either neither
    robot arm gripper jaws jaw wrist desk table surface board mat area
    workspace scene view camera frame image picture
    location locations spot spots position positions place places space
    spaces site sites square squares tile tiles cell cells zone zones
    region regions gap gaps patch patches point points
    free empty open occupied unoccupied vacant available spare taken busy
    used unused
    any all each every some none many few several more most less least
    number please now again still already just only even yet ever
    can could should would will shall may might must need needs want
    wants let make made does did done doing have has had are was were
    been being not don't dont cannot can't cant isn't isnt
    """.split()
)


# Other words for a KIND that is already in the snapshot. Not filler -- these
# name a real physical target, which is exactly why they must not reach the
# grounding call: the thing they name is already listed, with a calibrated
# position, and grounding it again invents a second entity on top of the first.
#
# Measured 2026-08-03, "move stapler to center aruco marker": "aruco" was not in
# any entity's label, so the pre-pass grounded it, measured the tag at 54x20mm
# and registered ``obj_1`` at (180, -142) -- 19.6mm from ``marker_2`` at
# (162, -150), the very marker meant. A destination of kind "object" is not a
# legal TRANSFER target, so the one-step move degraded into PICK then PLACE, and
# the extra step is where the run then failed.
#
# Deliberately only synonyms that are *definitional*, plus plurals. An ArUco tag
# IS a marker_N here -- the repo says so everywhere, from ``detect_markers`` to
# the "ArUco tag did not decode" refusal -- so the word cannot name anything
# else on this desk. "block" and "brick" are NOT in this table for cubes,
# tempting as they are: a wooden block that the HSV detector never saw is a
# thing the grounder should still be sent after, and crediting the word would
# turn that into a silent pick of the nearest coloured cube instead.
_KIND_SYNONYMS: dict[str, frozenset[str]] = {
    "marker": frozenset({
        "markers", "aruco", "arucos", "tag", "tags", "fiducial", "fiducials",
    }),
    "slot": frozenset({"slots"}),
    "cube": frozenset({"cubes"}),
    "object": frozenset({"objects"}),
}


def unmatched_nouns(
    instruction: str, snapshot: Snapshot, *, held: str | None = None
) -> set[str]:
    """Words naming a target that nothing in the snapshot could be.

    An unrecognised noun means the task is asking for something the cube
    detector cannot see, so the caller sends the grounder after it and registers
    what it finds as ``obj_N``. This is the productive use of the word list --
    nothing here refuses anything.

    ``held`` names what is in the jaws, and its words count as known vocabulary
    even though the object is deliberately no longer an entity. Without it the
    noun that started the task stays unmatched for the rest of the task and
    every later step re-grounds it, which finds the arm's own gripper.
    """
    vocab = {w for e in snapshot.entities for w in e.label.lower().split()}
    kinds = {e.kind for e in snapshot.entities}
    vocab |= kinds | {"cube", "marker", "slot", "object"}
    # Only for kinds actually present. With no marker in the snapshot there is
    # nothing for "the aruco tag" to already be, so the word stays unmatched and
    # reaches the grounder -- which is the honest answer rather than quietly
    # resolving to whichever kind the model picks. See :data:`_KIND_SYNONYMS`.
    for kind in kinds:
        vocab |= _KIND_SYNONYMS.get(kind, frozenset())
    if held:
        vocab |= set(held.lower().split())
    words = {
        w.strip(".,!?;:'\"")
        for w in instruction.lower().split()
        if w.strip(".,!?;:'\"").isalpha()
    }
    return {w for w in words if len(w) > 2 and w not in _FILLER and w not in vocab}


def _matches_attributes(entity: Entity, attrs: set[str]) -> bool:
    """Every named attribute must hold, not merely one of them.

    ``any`` was wrong and dangerously so: "the red cube" extracts more than one
    word, and a blue cube satisfying just one of them would be accepted.

    **Never require what nothing measured.** A colour word is checked against
    ``entity.color``, which the HSV detector fills for a cube and
    ``detect.classify_color`` for a registered object -- and which is None when
    the thing is not clearly any named colour (a grey stapler, a two-tone toy).
    None abstains: the word neither confirms nor contradicts, so it does not
    decide.

    Reading the colour off the label instead is what broke "pick up the green
    statue" on a desk holding a green statue. Absence from a label conflates
    "measured, and it is not green" with "nobody established a colour", and the
    second is not a contradiction. Worse, it was unfalsifiable: the attribute
    vocabulary is pooled from every entity's label, while a registered object's
    label only gained words *no* entity's label had -- so a shared adjective was
    required by construction and absent by construction.

    Nouns keep the label test. There the label is the whole of what is known
    (obj_2 answers to "stone", not "rock"), so absence really is contradiction.
    """
    label = entity.label.lower()
    known_colors = _color_vocabulary()
    for a in attrs:
        if a.startswith("marker") and a[6:].isdigit():
            if not (entity.kind == "marker" and entity.id.split("_")[-1] == a[6:]):
                return False
        elif a in known_colors:
            if entity.color is not None and entity.color.lower() != a:
                return False
        elif a not in label:
            return False
    return bool(attrs)


def _color_vocabulary() -> frozenset[str]:
    """Every colour name the detector can actually produce.

    Read from ``detect.COLOR_RANGES`` rather than written out here, so a colour
    added to the bands (the docstring there invites adding "orange") is checked
    the same way as the rest instead of quietly falling through to the noun
    test, where it would be matched as a substring of a label.
    """
    from mt4_vision.detect import COLOR_RANGES

    return frozenset(COLOR_RANGES)


def _said_id(got: dict[str, Any], key: str) -> str | None:
    """The id the model wrote under ``key``, or None.

    JSON null is None, but the model also writes the *string* "null" -- taking
    that literally would look up an entity called "null" and refuse for the
    wrong reason.
    """
    said = got.get(key)
    if isinstance(said, str) and said.strip().lower() in ("", "null", "none"):
        said = None
    return str(said) if isinstance(said, str) else None


def _resolve_target(
    obs: Observation,
    instruction: str,
    kind: str,
    *,
    said: str | None,
    point: Any,
    why: str,
    raw: str,
    said_label: str = "",
    point_key: str = "point_2d",
) -> Action:
    """Resolve ONE target -- the entity to act on, or a refusal saying why not.

    ``kind`` is the whole rule set, not a label: it selects which entity kinds
    are candidates, which of the task's attributes are allowed to describe this
    target, and whether an exclusion like "not on a marker" applies. Passing
    PICK_ENTITY resolves a thing to grasp; PLACE_ENTITY resolves a place to put
    one; LOCATE_AT_PIXEL resolves nothing and returns the point to measure at.

    A TRANSFER calls this twice against the same snapshot, once with each of the
    first two kinds. That is the entire reason it is a function: the two halves
    of a transfer answer to exactly the rules the two separate actions always
    did, so a transfer cannot become an easier way to get something wrong.

    Gripper state is deliberately NOT checked here. Whether the jaws may be full
    depends on the action -- a PLACE needs them full, a TRANSFER needs them
    empty and then fills them -- and that belongs to the caller.
    """
    # Existence before geometry: a task naming something absent should say so,
    # not complain that the accompanying point was out of frame.
    known = {e.id for e in obs.snapshot.entities}
    if said is not None and said not in known and kind != "LOCATE_AT_PIXEL":
        return Action(
            "STOP", False,
            f"{said!r} is not in this snapshot (ids: {', '.join(sorted(known))}) "
            "-- if the task names something the vision system has not found, "
            "the answer is LOCATE_AT_PIXEL or STOP, never a nearby substitute",
            model_entity_id=said, raw=raw,
        )

    # The point is a tie-breaker, not the target, so an unusable one degrades to
    # "no point" instead of failing the decision.
    #
    # Both readings are kept rather than one conversion applied. The decision
    # prompt prints each entity's pixel, and the model answers in that space --
    # so pixels lead. But "lead" is all: which reading is used gets settled by
    # binding against the snapshot below, not decided here.
    readings = (
        point_readings(point, obs.size)
        if isinstance(point, (list, tuple)) and len(point) == 2
        else ()
    )
    pt: tuple[float, float] | None = readings[0] if readings else None
    alt_pt: tuple[float, float] | None = readings[1] if len(readings) > 1 else None
    if pt is None and said is None:
        return Action(
            "STOP", False,
            f"neither a usable entity_id nor a usable {point_key} (got "
            f"{said!r} and {point!r}) -- nothing identifies a target",
            model_entity_id=said, raw=raw,
        )

    if kind == "LOCATE_AT_PIXEL":
        if pt is None:
            return Action(
                "STOP", False,
                "LOCATE_AT_PIXEL is only a point -- without a usable one there "
                "is nothing to measure",
                model_entity_id=said, raw=raw,
            )
        label = said_label.strip() or "object"
        hit, _err = bind(obs, pt)
        if hit is not None:
            return Action(
                "STOP", False,
                f"asked to register a new object, but that point is already "
                f"{hit.id} ({hit.label}) -- use it instead of registering a duplicate",
                entity_id=hit.id, point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
            )
        return Action(
            kind, True, why or f"register {label}", label=label,
            point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )

    if kind == "PICK_ENTITY":
        unknown = unmatched_nouns(instruction, obs.snapshot)
        if unknown:
            return Action(
                "STOP", False,
                f"the task names {sorted(unknown)}, which nothing in this "
                "snapshot is -- register it with LOCATE_AT_PIXEL first, or say "
                "it is not there. Picking the nearest cube instead is the one "
                "outcome nothing downstream can detect",
                point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
            )

    kinds = ("cube", "object") if kind == "PICK_ENTITY" else ("marker", "slot")
    # A forbidden kind is not a candidate for the point either. Without this the
    # nearest-entity search still ranks markers, and two free markers a few
    # dozen pixels apart made the point ambiguous on a task that had already
    # excluded both -- measured: "48px from marker_2 and 85px from marker_0,
    # both within 87px", on "place it on a non-marker location".
    banned = excluded_destination_kind(instruction) if kind == "PLACE_ENTITY" else None
    if banned is not None:
        kinds = tuple(k for k in kinds if k != banned)
    # Resolve the coordinate space against the snapshot. A reading only wins by
    # landing on something that is actually there, and if both readings land on
    # *different* entities the reply does not identify one -- refuse rather than
    # pick the convention that flatters the answer.
    pointed, point_err = None, "no usable point in the reply"
    landed = []
    for cand in readings:
        hit_e, err = bind(obs, cand, kinds=kinds)
        if hit_e is not None:
            landed.append((cand, hit_e))
        elif point_err == "no usable point in the reply":
            point_err = err
    if len({e.id for _c, e in landed}) > 1:
        both = ", ".join(f"{c[0]:.0f},{c[1]:.0f} -> {e.id}" for c, e in landed)
        return Action(
            "STOP", False,
            f"the point {point} reads as two different entities depending on the "
            f"coordinate space ({both}) -- the reply does not identify one",
            point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )
    if landed:
        pt, pointed = landed[0]
        alt_pt = next((c for c in readings if c != pt), None)
    named = obs.snapshot.get(said) if said else None

    # A target needs at least one channel that holds up, and no channel that
    # contradicts. Measured on this desk, neither is dependable alone: the
    # model pointed 149px off a cube it had named correctly, and pointed onto a
    # real cube for an object that was not present at all.
    # A two-part instruction names attributes of two different targets: "put
    # the blue cube on marker 3" yields {blue, marker3}, and demanding one
    # entity satisfy both refuses the blue cube for not being marker 3. Keep
    # only the attributes that can describe the kind being resolved, and read
    # them from the target's own noun phrase rather than the whole sentence --
    # otherwise a landmark ("next to the blue cube") becomes a requirement.
    if kind == "PICK_ENTITY":
        mismatch = wrong_kind_block_reason(instruction, named, obs.snapshot)
        if mismatch is not None:
            return Action(
                "STOP", False, mismatch,
                entity_id=None, model_entity_id=said, raw=raw,
            )

    head = None if named is None else named.label.lower().split()[-1]
    attrs = {
        a
        for a in instruction_attributes(instruction, obs.snapshot, head=head)
        if (a.startswith("marker") and a[6:].isdigit()) == (kind == "PLACE_ENTITY")
    }
    verified = named is not None and attrs and _matches_attributes(named, attrs)
    if named is not None and attrs and not verified:
        return Action(
            "STOP", False,
            f"the task says {sorted(attrs)} but {named.id} is {named.label!r} "
            "-- the detector's own label contradicts the choice",
            point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )

    if verified:
        # Attribute-verified names outrank a stray point, but must still be the
        # unique match: "the red cube" with two red cubes is ambiguous unless
        # the point picks one out.
        rivals = [
            e for e in obs.snapshot.entities
            if e.kind in kinds and e is not named and _matches_attributes(e, attrs)
            and (e.pickable if kind == "PICK_ENTITY" else e.placeable)
        ]
        if rivals and (pointed is None or pointed.id != named.id):
            ids = ", ".join(sorted([named.id] + [r.id for r in rivals]))
            return Action(
                "STOP", False,
                f"{sorted(attrs)} matches {ids} and the point does not single "
                "one out -- say which",
                point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
            )
        hit = named
    elif pointed is not None and (named is None or pointed.id == named.id):
        hit = pointed
    elif (
        named is not None and pointed is not None
        and banned is not None and named.kind == banned
    ):
        # There IS something in the task to settle it: the name is of the kind
        # the task excluded and the point is not. Measured live, this is the
        # common shape of the disagreement -- the model points at the right
        # place and writes the wrong id beside it.
        hit = pointed
    elif named is not None and pointed is not None:
        return Action(
            "STOP", False,
            f"named {named.id} ({named.label}) but pointed at {pointed.id} "
            f"({pointed.label}), with nothing in the task to settle which",
            point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )
    else:
        why_not = point_err or "could not resolve a target"
        if (
            kind == "PLACE_ENTITY" and named is not None
            and named.kind not in ("marker", "slot")
        ):
            # Measured live: step 2 of "move the green cube to an open slot"
            # answered PLACE_ENTITY cube_2 -- the id of the cube it had just
            # picked up in step 1, which by then named a different cube. The
            # refusal that came out was about the accompanying point being
            # 177px from a slot, which is true and tells the reader nothing
            # about what actually went wrong.
            why_not = (
                f"{named.id} is a {named.kind} ({named.label!r}), which is not "
                "somewhere to put things -- a destination is a marker_N or a "
                f"slot_N; the point did not name one either -- {why_not}"
            )
        elif named is not None and banned is not None and named.kind == banned:
            # Otherwise this reads "it does not name anything in the list" on a
            # reply that named marker_0 -- true of the point, and flatly wrong
            # about the reply, which is the sentence a user has to act on.
            why_not = (
                f"the task rules out placing on a {banned} and {named.id} "
                f"({named.label}) is one, so the name cannot stand; the point "
                f"could not stand in for it either -- {why_not}"
            )
        return Action(
            "STOP", False, why_not,
            point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )

    if banned is not None and hit.kind == banned:
        alts = [
            e.id for e in obs.snapshot.entities
            if e.kind != banned and e.kind in ("marker", "slot") and e.placeable
        ]
        return Action(
            "STOP", False,
            f"the task rules out placing on a {banned}, and {hit.id} "
            f"({hit.label}) is one"
            + (
                f" -- {', '.join(alts[:6])} are not"
                if alts else " -- and nothing else here can be placed on"
            ),
            entity_id=None, point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )

    capable = hit.pickable if kind == "PICK_ENTITY" else hit.placeable
    if not capable:
        verb = "picked up" if kind == "PICK_ENTITY" else "placed on"
        return Action(
            "STOP", False,
            f"{hit.id} ({hit.label}) cannot be {verb}: {hit.reason}",
            entity_id=hit.id, point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )

    return Action(
        kind, True, why or f"{kind.lower()} {hit.id}", entity_id=hit.id,
        label=hit.label, point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
    )


def decide(
    obs: Observation,
    instruction: str,
    *,
    url: str = DEFAULT_URL,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Action:
    """Ask for one action and return it validated, or a refusal explaining why.

    Every failure path produces ``ok=False`` with prose, never an exception and
    never a silently substituted target.
    """
    prompt = build_prompt(obs, instruction)
    try:
        reply = ask(
            prompt, obs.annotated, url=url,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
    except QwenError as exc:
        return Action("STOP", False, f"vision-language service failed: {exc}")

    raw = reply.text.strip()
    got = _first_json_object(raw)
    if got is None:
        return Action("STOP", False, f"no JSON object in the reply: {raw[:200]!r}", raw=raw)

    kind = str(got.get("action", "")).strip().upper()
    said = _said_id(got, "entity_id")
    why = str(got.get("reason", "") or "").strip()
    if kind not in ACTIONS:
        return Action(
            "STOP", False, f"{kind!r} is not one of {', '.join(ACTIONS)}",
            model_entity_id=said, raw=raw,
        )

    if kind in ("DONE", "STOP"):
        # ok means "there is something to act on". A model-chosen STOP is still
        # a refusal, and reads identically to a validation-forced one; DONE is
        # the loop's only successful exit.
        return Action(kind, kind == "DONE", why or kind.lower(),
                      model_entity_id=said, raw=raw)

    # A question is not an order. The action set cannot answer one, so the
    # model answers with whatever action fits -- measured: "is there anything
    # on the desk that is not a cube" returned PICK_ENTITY cube_2.
    if is_question(instruction):
        return Action(
            "STOP", False,
            f"that is a question, not an instruction, and the model answered "
            f"it with {kind} {said or '-'} -- nothing moves for a question",
            model_entity_id=said, raw=raw,
        )

    # If the task names a destination that cannot be placed on, refuse now
    # rather than after the pick. The alternative strands the object in the
    # jaws at a step that was never going to succeed.
    if kind in ("PICK_ENTITY", "TRANSFER"):
        blocked = named_destination_block_reason(
            instruction, obs.snapshot, moving=said
        )
        if blocked is not None:
            return Action(
                "STOP", False,
                f"the task's destination is not available, so the pick would "
                f"strand the object: {blocked}",
                model_entity_id=said, raw=raw,
            )

    if kind == "TRANSFER":
        if obs.held:
            return Action(
                "STOP", False,
                f"cannot start a transfer while holding {obs.held} -- that "
                "needs a PLACE_ENTITY first",
                model_entity_id=said, raw=raw,
            )
        # Both halves resolved against this one snapshot before the arm moves at
        # all. A transfer whose destination will not do must be refused while
        # the object is still on the desk -- refusing after the grasp strands it
        # in the jaws, which is the failure the old PICK-then-PLACE pair had.
        src = _resolve_target(
            obs, instruction, "PICK_ENTITY",
            said=said, point=got.get("point_2d"), why=why, raw=raw,
        )
        if not src.ok:
            return src
        dest_said = _said_id(got, "dest_entity_id")
        dst = _resolve_target(
            obs, instruction, "PLACE_ENTITY",
            said=dest_said, point=got.get("dest_2d"), why=why, raw=raw,
            point_key="dest_2d",
        )
        if not dst.ok:
            # Prefixed: the two halves produce identically worded refusals, and
            # "is not in this snapshot" tells a reader nothing about which end
            # of the move it means.
            return replace(dst, reason=f"the destination will not do: {dst.reason}")
        return Action(
            "TRANSFER", True, why or f"move {src.entity_id} onto {dst.entity_id}",
            entity_id=src.entity_id, label=src.label,
            point_px=src.point_px, alt_point_px=src.alt_point_px,
            model_entity_id=said,
            dest_entity_id=dst.entity_id, dest_label=dst.label,
            dest_point_px=dst.point_px, dest_alt_point_px=dst.alt_point_px,
            model_dest_entity_id=dest_said,
            raw=raw,
        )

    if kind == "PICK_ENTITY" and obs.held:
        return Action(
            "STOP", False,
            f"cannot pick while holding {obs.held} -- put it down first",
            model_entity_id=said, raw=raw,
        )
    if kind == "PLACE_ENTITY" and not obs.held:
        return Action(
            "STOP", False, "cannot place with an empty gripper",
            model_entity_id=said, raw=raw,
        )

    return _resolve_target(
        obs, instruction, kind,
        said=said, point=got.get("point_2d"), why=why, raw=raw,
        said_label=str(got.get("label") or ""),
    )
