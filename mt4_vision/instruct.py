"""Natural-language instruction -> one action on one entity.

The policy layer. It decides *what to do next and to which thing*; every
millimetre, every safety gate and every motion belongs to the existing stack.
Nothing here computes a robot coordinate.

**The model's target choice is taken at face value.** :func:`decide` asks for
one action plus an ``entity_id`` and a point, and :func:`_resolve_target` turns
those into an entity: the id when it names something in the snapshot, otherwise
the point bound to the nearest entity of a usable kind. It does not compare the
two against each other, against the instruction's wording, or against the
detector's own labels. The one thing that can still refuse a resolved target is
physical -- ``Entity.pickable`` / ``Entity.placeable`` carry reach, the J1
keep-out, ground Z, finger clearance and the desk polygon, and that gate is what
makes trusting the rest safe: nothing here can command a pose the envelope would
have rejected anyway.

**Coordinates.** A grounding reply (:func:`locate_target`, which says nothing
about coordinate space) comes back 0-1000 normalized; :func:`to_frame_pixels`
scales it. A decision reply comes back in pixels, because ``build_prompt``
prints every entity's own pixel as a worked example; :func:`point_readings`
returns both readings, pixels first, and binding settles which was meant.

**Grounding asks for a box, not a point.** ``bbox_2d`` unlocks GrabCut
segmentation, bounds the mask, and gives an extent that can be sanity-checked;
a point gives none of the three. See :class:`Grounding`.

One capture per decision, so the ids drawn on the overlay and the ids in the
entity list come from the same exposure -- :func:`observe` does it once.
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
# Re-exported via __all__ so `from mt4_vision.instruct import to_frame_pixels`
# keeps working for callers and tests; only point_readings is used here.
from mt4_vision.instruct_reply import (
    COORD_SCALE,
    MAX_BOX_FRAME_SHARE,
    Grounding,
    alternate_reading,
    locate_target,
    measure_grounding,
    point_readings,
    to_frame_pixels,
)
from mt4_vision.preview import annotate_for_pointing
from mt4_vision.qwen import DEFAULT_URL, QwenError, ask
from mt4_vision.scene import capture_scene
from mt4_vision.workspace import PICK_CLEARANCE_MM

__all__ = [
    "ACTIONS", "BIND_RADIUS_MM", "COORD_SCALE", "DUPLICATE_MM", "MAX_NEW_TOKENS",
    "MAX_BOX_FRAME_SHARE", "Action", "Grounding", "Observation",
    "alternate_reading", "bind", "build_prompt", "decide", "grasp_for",
    "is_question", "locate_target", "measure_grounding", "observe",
    "point_readings", "register_object", "to_frame_pixels", "unmatched_nouns",
    "load_calibration",
]

# A point must land this close to an entity to name it, and no second entity may
# be this close. In millimetres, converted to pixels at the candidate's own
# position, because this camera is oblique and mm-per-pixel varies across the
# frame. PICK_CLEARANCE_MM is the right scale: two objects closer together than
# the fingers need are not independently pickable, so a point between them is a
# real ambiguity rather than a resolution problem.
BIND_RADIUS_MM = PICK_CLEARANCE_MM
# Short replies only: a decision is action + point + a clause of reason, and the
# 1664-token static cache holds one image plus this comfortably.
MAX_NEW_TOKENS = 220
# Two grounded measurements this close describe one physical thing. THE same
# constant discover uses, not a copy: they are the two paths that can add an
# object to the world and must agree about when two hits are one object.
DUPLICATE_MM = MERGE_MM

# TRANSFER leads because it is the shape of nearly every real task here: move a
# named thing to a named place, in one step, with no park-look-decide in the
# middle that could not answer anything anyway. PICK_ENTITY and PLACE_ENTITY are
# for the halves that occur alone -- "pick up the stapler" with no destination,
# and putting down something the gripper was already holding.
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
    # loop is memoryless: a completed task looks exactly like an untouched one,
    # since both show an empty gripper and the same instruction.
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

        A newly registered object has to reach the entity list before the
        decision. Observing again would cost a fresh capture per noun; rebuilding
        the snapshot and overlay from the frame already in hand costs ~5ms and
        keeps the same pixels, so the ids the model answers with still describe
        the frame it is looking at.
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

    ``ok`` means there is something to act on: a resolved entity action, or
    ``DONE``. Every ``STOP`` is ``ok=False``, whether the model chose it or
    resolution forced it, so a caller has one thing to branch on.
    """

    kind: str
    ok: bool
    reason: str
    entity_id: str | None = None
    label: str | None = None
    point_px: tuple[float, float] | None = None
    # The same point under the other coordinate convention, when there is one.
    # Never acted on: the preview draws both so a reader can see which was used.
    alt_point_px: tuple[float, float] | None = None
    # The id string the model wrote, whatever it was. Reported, never used to
    # choose -- the transcript prints "[model said ...]" when it differs.
    model_entity_id: str | None = None
    # TRANSFER only: the destination, resolved by the same rules against the
    # same snapshot as the source, so both ends hold before the gripper opens.
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
    frame: np.ndarray | None = None,
) -> Observation:
    """One frame -> snapshot and pointing overlay built from the same pixels.

    ``frame`` lets a caller supply pixels it already holds instead of opening
    the camera here. That is not an optimisation but a requirement for any
    session that also shows a live feed: only one consumer can hold the
    capture device, so a caller running a ``FrameStream`` for its preview
    cannot let this function open a second one (on Windows DSHOW the second
    open simply fails). It is also faster -- ``capture_frame`` reopens the
    device and burns 20 exposure warm-up reads, 2-3 SECONDS, on every step.

    The caller then owns the freshness guarantee this used to provide. A frame
    off a continuously-drained stream is only as current as the moment it was
    pulled, so pull it *after* the arm has parked (``FrameStream.fresh``
    blocks for a frame whose capture started after the call, which is exactly
    that guarantee); handing in a frame captured before the last move gets a
    snapshot of a desk that no longer exists.
    """
    calib = load_calibration()
    if frame is None:
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
        # The guard that catches this is deterministic and lives outside the
        # prompt (missing_place_target_reason), so this sentence is not what
        # makes it safe -- it is what stops the loop burning a park, a capture
        # and a decision on a reply that was always going to be refused.
        # Measured live 2026-08-03, "place it on marker 0" on a desk of markers
        # 1-4: the model answered marker_3, then marker_1, then "task already
        # completed", never once saying the number was not there.
        "A number in the task is an identity, not a hint: 'marker 0' means the "
        "entity whose id is exactly marker_0. If that id is not in the list "
        "above, the task names something that is not on this desk -- answer "
        "STOP and say which number is missing. A different number is never a "
        "substitute for it.\n"
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
    decision is made against that frame and the arm acts on the entity positions
    it produced. Nothing but the arm moves on this desk, and the arm is parked
    while the model thinks, so there is nothing a second look would find.

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

    ``kind`` selects which entity kinds the point may bind to and which physical
    capability is required. PICK_ENTITY resolves a thing to grasp; PLACE_ENTITY
    resolves a place to put one; LOCATE_AT_PIXEL resolves nothing and returns the
    point to measure at.

    A TRANSFER calls this twice against the same snapshot, once with each of the
    first two kinds, so both ends are resolved before the gripper opens.

    Gripper state is deliberately NOT checked here. Whether the jaws may be full
    depends on the action -- a PLACE needs them full, a TRANSFER needs them
    empty and then fills them -- and that belongs to the caller.
    """
    # Both coordinate readings are kept: the primary (pixels of the submitted
    # frame) leads, because that is the space the prompt asks for and prints
    # every entity's own pixel in, and the alternate is carried only so the
    # preview can draw both and a reader can see at a glance which was believed.
    readings = (
        point_readings(point, obs.size)
        if isinstance(point, (list, tuple)) and len(point) == 2
        else ()
    )
    pt: tuple[float, float] | None = readings[0] if readings else None
    alt_pt: tuple[float, float] | None = readings[1] if len(readings) > 1 else None

    if kind == "LOCATE_AT_PIXEL":
        if pt is None:
            return Action(
                "STOP", False,
                "LOCATE_AT_PIXEL is only a point -- without a usable one there "
                "is nothing to measure",
                model_entity_id=said, raw=raw,
            )
        # No "that point is already an entity" refusal any more. Registering a
        # second id for something already listed is untidy, not dangerous: the
        # duplicate is measured at the same place, so acting on it acts on the
        # same object. Refusing the whole task over it was the expensive half.
        return Action(
            kind, True, why or f"register {said_label.strip() or 'object'}",
            label=said_label.strip() or "object",
            point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )

    # ---- resolution ------------------------------------------------------
    #
    # Two channels, taken in order of explicitness, and NOTHING cross-examines
    # them. Owner's decision 2026-08-03: once the model has named a target, that
    # is the target. Get the loop working; add a check back when a specific
    # failure demands it, rather than failing because the reply is not trusted.
    #
    # What used to sit here, and is gone (see §2u of docs/qwen3_vl_policy_status.md
    # for the full list and the reasoning): an unmatched-noun refusal, a
    # wrong-kind refusal, two attribute-contradiction refusals, a rival-ambiguity
    # refusal, a named-vs-pointed disagreement refusal, a banned-destination-kind
    # refusal, and a coordinate-space disagreement refusal. Every one of them
    # could turn a reply that identified the right thing into an abandoned task,
    # and between them they abandoned three consecutive real runs.
    #
    # The one gate left is not about the model at all: whether the arm can
    # physically do it. ``Entity.pickable`` / ``Entity.placeable`` carry reach,
    # the J1 keep-out, ground Z, finger clearance and the desk polygon, and
    # ``Entity.reason`` says which of them failed. That gate is the reason this
    # relaxation is safe to make: nothing here can command a pose the envelope
    # would not have allowed anyway.
    kinds = ("cube", "object") if kind == "PICK_ENTITY" else ("marker", "slot")

    # 1. The id, when it names something in this snapshot. The model's own
    #    answer, at face value -- no attribute check, no kind check, no
    #    comparison against where it pointed.
    hit = obs.snapshot.get(said) if said else None

    # 2. Failing that, the point. Restricted to kinds this action can use, which
    #    is a resolution aid rather than a verdict: it is what makes a
    #    destination point land on the marker under it instead of a cube beside
    #    it. Both coordinate readings are tried and the first that lands wins --
    #    where the two used to disagree the reply was refused, and now the
    #    primary (pixel) reading simply leads, because that is the space the
    #    prompt asks for and prints every entity's own pixel in.
    point_err = "no point in the reply"
    if hit is None:
        for cand in readings:
            pointed, err = bind(obs, cand, kinds=kinds)
            if pointed is not None:
                hit, pt = pointed, cand
                alt_pt = next((c for c in readings if c != cand), None)
                break
            if point_err == "no point in the reply":
                point_err = err

    if hit is None:
        # Nothing identified a target: not a judgement about the reply, an
        # absence of one. Both channels are reported because which was tried
        # is the whole of what a reader needs to know here.
        said_note = (
            f"{said!r} is not one of this snapshot's ids "
            f"({', '.join(sorted(e.id for e in obs.snapshot.entities))})"
            if said else "the reply named no entity_id"
        )
        return Action(
            "STOP", False,
            f"nothing in the reply identifies a target: {said_note}, and "
            f"{point_err}",
            point_px=pt, alt_point_px=alt_pt, model_entity_id=said, raw=raw,
        )

    # ---- the one gate: can the arm actually do this ----------------------
    capable = hit.pickable if kind == "PICK_ENTITY" else hit.placeable
    if not capable:
        verb = "picked up" if kind == "PICK_ENTITY" else "placed on"
        return Action(
            "STOP", False,
            f"{hit.id} ({hit.label}) cannot be {verb}: {hit.reason}",
            entity_id=hit.id, point_px=pt, alt_point_px=alt_pt,
            model_entity_id=said, raw=raw,
        )

    return Action(
        kind, True, why or f"{kind.lower()} {hit.id}", entity_id=hit.id,
        label=hit.label, point_px=pt, alt_point_px=alt_pt, model_entity_id=said,
        raw=raw,
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
        # DONE is taken at face value. It used to be audited against the
        # instruction's named destination -- "reported the task complete, but
        # marker_0 is not on this desk" -- and that audit is gone with the rest
        # (owner's decision 2026-08-03): it turned the model's own report of
        # success into a failure, which is the most expensive place to be wrong.
        # Nothing downstream re-examines DONE, and nothing here does either;
        # every outcome line already says "commanded, not checked".
        #
        # ok means "there is something to act on". A model-chosen STOP is still
        # a refusal; DONE is the loop's only successful exit.
        return Action(kind, kind == "DONE", why or kind.lower(),
                      model_entity_id=said, raw=raw)

    # A question is not an order. Kept when everything around it was removed,
    # because it is not a check on the model's judgement at all -- it reads the
    # INSTRUCTION, and no phrasing of a genuine pick-and-place trips it. Without
    # it "is there anything on the desk that is not a cube" moves the arm:
    # measured, that returned PICK_ENTITY cube_2 and the cube was picked up.
    if is_question(instruction):
        return Action(
            "STOP", False,
            f"that is a question, not an instruction, and the model answered "
            f"it with {kind} {said or '-'} -- nothing moves for a question",
            model_entity_id=said, raw=raw,
        )

    # Two things used to stand here and are gone: a refusal when the instruction
    # named a destination the snapshot has no id for ("marker 0" on a desk of
    # markers 1-4), and a refusal when that named destination existed but was
    # blocked. Both second-guessed the reply from the instruction's text, and the
    # first was measured refusing tasks whose destination the snapshot was
    # offering. What remains catches the same thing later and physically: an
    # unplaceable destination is refused by ``Entity.placeable`` in
    # _resolve_target, and for a TRANSFER that happens before the gripper opens.

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
