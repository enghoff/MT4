"""Natural-language instruction -> one action, on a box the model drew.

The policy layer. It decides *what to do next and to which thing*; every
millimetre, every safety gate and every motion belongs to the existing stack.
Nothing here computes a robot coordinate.

**Qwen is the visual and linguistic grounding, and nothing here second-guesses
it.** Three rules follow from that, and every one of them is load-bearing:

* **Nothing pickable is listed for the model.** The only thing the prompt names
  is the ArUco tags, and only because a tag's *number* is a printed code that
  cannot be read off an image -- the decoder supplies what the model physically
  cannot see, and nothing else. No cube list, no slot list, no object registry,
  no ``pickable``/``placeable`` advertisement. What is on the desk is the
  model's job to see. A list of detected things skews it hard toward that list:
  ids that name a class, labels that repeat it, and a short enumeration that
  reads as "these are the only things here".
* **The instruction reaches the model verbatim.** No noun extraction, no
  stopword list, no per-word grounding pre-pass, no question detection. Word
  lists that decide which parts of a sentence name objects get it wrong in the
  expensive direction: exact matching with no stemming treats ``holding`` as a
  thing to go and find.
* **A target is a box, not an id.** :func:`decide` asks for ``box_2d`` around
  the thing to pick up. With no id to resolve there is no binding, no ambiguity
  refusal and no named-vs-pointed disagreement -- three ways a correct reply
  can be turned into an abandoned task.

**What refuses, and why that is not second-guessing.** The measurement must
survive segmentation, the two-window stability check and the plausibility band,
and the resulting pose must clear reach, the J1 keep-out, ground Z, jaw
clearance and the desk polygon (``entities.object_entity``). Those gates read
geometry, never the model's judgement about *what* a thing is. They are what
makes trusting the rest safe: nothing here can command a pose the envelope
would reject.

**Why a box and not a point.** ``box_2d`` unlocks GrabCut, measured at 4 of 4
objects segmented on a frame where the bare-point desk-deviation path manages
1 of 4. It also bounds the mask so it cannot flood the desk, and gives an extent
that can be sanity-checked before the arm moves. See :class:`Grounding`.

**Grip geometry.** Every object is gripped at ``calib.table_z`` -- as low as the
jaws go -- at the table-plane projection of the GrabCut mask's centroid, with
the yaw of that mask's long axis. So ``object_height_mm=0`` on every
measurement, and no step of a pick depends on how tall the thing is. What that
trades is stated where it is paid: :data:`PICK_AT_TABLE_HEIGHT_MM`.

**Coordinates are 0-1000, everywhere.** That is the space this model answers
in, whatever a prompt asks for -- it is what Qwen document in
``cookbooks/spatial_understanding.ipynb`` and what every third-party integration
does -- so it is the space the prompt asks for and the space the tag positions
are printed in. :func:`to_frame_pixels` scales per axis against the original
frame, ``x * w / 1000`` and ``y * h / 1000``, two different factors on a
non-square frame.

Measured over 3 targets x 2 prompt styles on one 1280x720 frame: read as
0-1000, box centres land **2-13px** from truth, 6 of 6; read as raw pixels,
**264-363px** away. Asking for pixels does not change what comes back, so the
loop reads what the model actually emits.

A raw-pixel reading is still kept and retried when the first will not segment,
because a coordinate under 1000 is only *probably* normalized. That retry is
also the loop's one soft spot: both readings can land on real, separately
measurable objects, and the retry fires only when the leading one fails, so a
reading that is wrong *and* segments is not caught. It is a small target,
because the leading reading is the one the model means.

One capture per decision: :func:`observe` takes the frame, the overlay and the
marker positions from the same exposure.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mt4_vision.calib import Calibration, load_calibration
from mt4_vision.camera import capture_frame
from mt4_vision.entities import KIND_MARKER, Entity, Snapshot, build_snapshot
# Re-exported via __all__ so `from mt4_vision.instruct import to_frame_pixels`
# keeps working for callers and tests.
from mt4_vision.instruct_reply import (
    COORD_SCALE,
    MAX_BOX_FRAME_SHARE,
    Grounding,
    box_grounding,
    box_readings,
    measure_grounding,
    point_readings,
    to_frame_pixels,
)
from mt4_vision.preview import annotate_for_pointing
from mt4_vision.qwen import DEFAULT_URL, QwenError, ask
from mt4_vision.scene import capture_scene

__all__ = [
    "ACTIONS", "COORD_SCALE", "MAX_NEW_TOKENS", "MAX_BOX_FRAME_SHARE",
    "PICK_AT_TABLE_HEIGHT_MM", "Action", "Grounding", "Observation",
    "box_grounding", "box_readings", "build_prompt", "decide",
    "destination_grasp", "grasp_for", "measure_grounding", "observe",
    "point_readings", "to_frame_pixels", "load_calibration",
]

# Short replies only: a decision is an action, a box, a destination and a clause
# of reason, and the 1664-token static cache holds one image plus this
# comfortably.
MAX_NEW_TOKENS = 220

# Every pick measures at zero assumed object height: a target's XY is the plain
# table-plane projection of its mask centroid, and the jaws close at
# ``calib.table_z``. Grip as low as possible, at the point the model identified,
# oriented by the GrabCut mask.
#
# This keeps the height-from-silhouette estimator out of the pick path. That
# estimator infers 7.2-32.3mm for objects that are all 20mm, it assumes a cross
# section as tall as it is wide (false for anything flat), and its error lands
# as XY displacement of up to ~28mm against the ~10mm the jaws tolerate.
#
# The cost, and it is measured: a silhouette centroid sits *outward* of the real
# footprint on this oblique mount, by roughly the object's height times
# ``_parallax_gain`` (1.4-2.0 here). Against ``cube_top_homography`` on six cube
# detections this path lands 9.0-24.5mm out, mean ~17, always outward -- outside
# the ~10mm the jaws tolerate on 5 of the 6. The height-inferring path is better
# on 5 of 6 (2.7-14.1mm) and 43.2mm out on the sixth.
#
# So a 20mm cube gets shoved more often than gripped, and this stays because it
# is the right shape of error for something FLAT -- paper, a key, a card, where
# there is no height to mis-attribute and no independent ground truth on this
# desk to check it against. It is a choice of which error to carry, not a tuning
# knob. Revisit it by routing compact objects to the fitted cube homography
# after measurement, or by fixing the estimator's axis dependence.
PICK_AT_TABLE_HEIGHT_MM = 0.0

# TRANSFER leads because it is the shape of nearly every real task here: move a
# thing to a place, in one step, with no park-look-decide in the middle that
# could not answer anything anyway. PICK and PLACE are for the halves that occur
# alone -- "pick up the stapler" with no destination, and putting down something
# the gripper is already holding.
#
# There is no locate-and-register action, because there is no list to register
# into: every pick measures a box on the frame the box was drawn on, so every
# pick *is* a locate-at-pixel and costs no extra park-capture-decide round trip.
ACTIONS = ("TRANSFER", "PICK", "PLACE", "DONE", "STOP")


@dataclass(frozen=True)
class Observation:
    """One capture: the frame, the tags on it, and what the arm is holding.

    ``snapshot`` is the **full** detection -- cubes included -- because the
    internal safety gates need to know what is on the desk near a grasp point.
    It is never shown to the model. :attr:`markers` is the model-facing half,
    and the only thing :func:`build_prompt` and the overlay are allowed to read.
    Keeping the split here, on the type, is deliberate: the guarantee that the
    model sees nothing pickable should be one attribute a reader can check,
    rather than a discipline spread over two call sites.
    """

    frame: np.ndarray = field(repr=False)
    annotated: np.ndarray = field(repr=False)
    snapshot: Snapshot
    calib: Calibration
    held: str | None = None
    # What this task has already done, oldest first, in prose. Without it the
    # loop is memoryless: a completed task looks exactly like an untouched one,
    # since both show an empty gripper and the same instruction.
    history: tuple[str, ...] = ()
    # The detections the snapshot was built from. Kept for the internal gates
    # (``object_entity`` asks it how close the nearest other thing is), never
    # for the prompt.
    scene: Any = None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frame.shape[:2]
        return w, h

    @property
    def markers(self) -> list[Entity]:
        """The decoded ArUco tags -- everything the model is told about.

        A tag is listed for one reason: its number is a printed code that a
        vision-language model cannot read off the image, and the task refers to
        it by that number. Nothing else on this desk needs a name supplied.
        """
        return [e for e in self.snapshot.entities if e.kind == KIND_MARKER]

    def marker(self, entity_id: str) -> Entity | None:
        """The tag with this id, or None. Markers only, on purpose -- a reply
        naming ``cube_2`` must not resolve to something the model was never
        shown."""
        for e in self.markers:
            if e.id == entity_id:
                return e
        return None


@dataclass(frozen=True)
class Action:
    """A validated decision, or a refusal.

    ``ok`` means there is something to act on: a well-formed action, or
    ``DONE``. Every ``STOP`` is ``ok=False``, whether the model chose it or
    parsing forced it, so a caller has one thing to branch on.

    **This is the reply, not a resolved target.** ``ok`` says the JSON was
    usable and the action's required fields are present -- not that anything
    was measured, is reachable, or exists. The source box is measured and the
    destination projected by the caller, after this returns, because both cost
    real work and both can fail on their own terms. See :func:`measure_source`
    and :func:`destination_grasp`.
    """

    kind: str
    ok: bool
    reason: str
    # What the model called the thing it boxed. Carried for the transcript and
    # for `held`; nothing selects on it.
    label: str | None = None
    # The source box in frame pixels, and its centre. The box is what gets
    # measured; the centre is what the preview draws and what the desk-deviation
    # path falls back to when GrabCut cannot cut the box.
    source: Grounding | None = None
    # A destination is one of these two, never both. ``dest_entity_id`` is an
    # ArUco id -- the only name in this protocol, because it is the only thing
    # the model was given a name for. ``dest_point_px`` is a bare pixel on the
    # desk, for "anywhere else".
    dest_entity_id: str | None = None
    dest_point_px: tuple[float, float] | None = None
    dest_alt_point_px: tuple[float, float] | None = None
    raw: str = ""

    @property
    def point_px(self) -> tuple[float, float] | None:
        """Centre of the source box. Preview and transcript read this."""
        return None if self.source is None else self.source.point_px

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"action": self.kind, "ok": self.ok, "reason": self.reason}
        if self.label:
            out["label"] = self.label
        if self.source is not None:
            out["point_px"] = [
                round(self.source.point_px[0], 1), round(self.source.point_px[1], 1)
            ]
            if self.source.box_px:
                out["box_px"] = [round(v, 1) for v in self.source.box_px]
            if self.source.alt_point_px:
                out["alt_point_px"] = [
                    round(self.source.alt_point_px[0], 1),
                    round(self.source.alt_point_px[1], 1),
                ]
        if self.dest_entity_id:
            out["dest_entity_id"] = self.dest_entity_id
        if self.dest_point_px:
            out["dest_point_px"] = [
                round(self.dest_point_px[0], 1), round(self.dest_point_px[1], 1)
            ]
        if self.dest_alt_point_px:
            out["dest_alt_point_px"] = [
                round(self.dest_alt_point_px[0], 1),
                round(self.dest_alt_point_px[1], 1),
            ]
        return out


def observe(
    camera: int = 1,
    *,
    held: str | None = None,
    history: Sequence[str] = (),
    token: str = "s1",
    frame: np.ndarray | None = None,
) -> Observation:
    """One frame -> snapshot and pointing overlay built from the same pixels.

    **Only the ArUco tags are drawn on the overlay.** The frame the model is
    given carries the numbered grid it reads coordinates off, and a circled id
    beside each decoded tag, and nothing else. Circling the detected cubes would
    tell it where the stack believes the interesting things are, which is the
    picture-shaped version of handing it a cube list.

    ``frame`` lets a caller supply pixels it already holds instead of opening
    the camera here. That is not an optimisation but a requirement for any
    session that also shows a live feed: only one consumer can hold the
    capture device, so a caller running a ``FrameStream`` for its preview
    cannot let this function open a second one (on Windows DSHOW the second
    open simply fails). It is also faster -- ``capture_frame`` reopens the
    device and burns 20 exposure warm-up reads, 2-3 SECONDS, on every step.

    A caller passing ``frame`` owns the freshness guarantee. A frame off a
    continuously-drained stream is only as current as the moment it was pulled,
    so pull it *after* the arm has parked (``FrameStream.fresh`` blocks for a
    frame whose capture started after the call, which is exactly that
    guarantee); a frame captured before the last move describes a desk that has
    since changed.
    """
    calib = load_calibration()
    if frame is None:
        frame = capture_frame(camera)
    scene = capture_scene(calib, frame)
    snapshot = build_snapshot(scene, token=token)
    annotated = annotate_for_pointing(
        frame, [e for e in snapshot.entities if e.kind == KIND_MARKER]
    )
    return Observation(
        frame=frame, annotated=annotated, snapshot=snapshot, calib=calib,
        held=held, history=tuple(history), scene=scene,
    )


# -- prompt ---------------------------------------------------------------


def _marker_lines(obs: "Observation") -> str:
    """The ArUco tags, id and pixel. The whole of what the model is told.

    No capability flags. A flag like ``placeable`` invites the model to treat
    the list as the set of legal answers, and a tag that fails the reach test is
    still the tag the task named -- the honest place to say so is the refusal
    after the model has answered, in the words of the gate that failed.
    """
    rows = []
    for e in obs.markers:
        pixel = e.pixel
        if pixel is None:
            pixel = obs.calib.robot_to_pixel(e.x, e.y)
        w, h = obs.size
        rows.append(
            f"  {e.id} at ({pixel[0] * COORD_SCALE / w:.0f}, "
            f"{pixel[1] * COORD_SCALE / h:.0f})"
        )
    return "\n".join(rows) or "  (no tags decoded in this frame)"


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
            "the task is done -- what remains is where to put it down. PLACE "
            "says where it goes; there is no need to box the thing you are "
            "holding, and TRANSFER is not available while the gripper is full."
        )
    )
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
        # Why the tags are listed and nothing else is. Said out loud so the
        # model does not read a short list as "these are the only things here"
        # and force the task onto one of them -- which is what a list of
        # detected objects reliably provoked.
        "Printed square tags on the desk carry a number that cannot be read "
        "from the image, so the numbers are given to you here. Each is circled "
        "with its id beside it. This is NOT a list of what is on the desk -- "
        "every other object is yours to find by looking at the image:\n"
        f"{_marker_lines(obs)}\n\n"
        "Choose one action:\n"
        "  TRANSFER - pick something up and put it somewhere, in one go. "
        "box_2d is the thing to pick up; dest_marker or dest_2d is where it "
        "goes. Use this whenever the task names BOTH a thing and a place for "
        "it, and the gripper is empty\n"
        "  PICK     - pick something up and hold it, when the task names no "
        "destination at all (only if the gripper is empty). box_2d is the "
        "thing to pick up\n"
        "  PLACE    - put down what is ALREADY held. dest_marker or dest_2d "
        "says where\n"
        "  DONE     - the task is complete\n"
        "  STOP     - the task cannot be done, or you cannot tell which thing "
        "is meant\n\n"
        "Reply with ONLY a JSON object, no prose, no markdown fence:\n"
        '{"action": "...", "box_2d": [x1, y1, x2, y2], "label": "...", '
        '"dest_marker": "marker_N", "dest_2d": [x, y], '
        '"reason": "<one short clause>"}\n\n'
        "Every coordinate you give is on a 0-1000 scale across the image: 0 is "
        "the left edge, 1000 the right edge, and the same top to bottom. The "
        "tag positions above are written that way too.\n"
        "box_2d is a tight box around the WHOLE of the object to pick up. The "
        "jaws are aimed using that box, so it must contain that object and as "
        "little else as possible -- not the desk around it, and not a "
        "neighbouring object. Required for TRANSFER and PICK.\n"
        "label is a short noun for whatever you boxed.\n"
        "A destination is EITHER dest_marker or dest_2d, never both. Use "
        "dest_marker when the task names a tag by its number, copying the id "
        "exactly as listed above. Use dest_2d -- a pixel on bare desk -- for "
        "anywhere else, including 'somewhere clear', 'not on a marker' and "
        "'on the table'. One of the two is required for TRANSFER and PLACE.\n"
        # The one text-derived rule left, and it is about the one datum we
        # supply rather than about the model's own grounding. Measured live
        # 2026-08-03, "place it on marker 0" on a desk of markers 1-4: the model
        # answered marker_3, then marker_1, then "task already completed", never
        # once saying the number was not there.
        "A tag number in the task is an identity, not a hint: 'marker 0' means "
        "the tag whose id is exactly marker_0. If that id is not listed above "
        "then that tag is not on this desk -- answer STOP and say which number "
        "is missing. A different number is never a substitute for it.\n"
        "Use null for every field the chosen action does not need.\n"
        "Begin your reply with { and end it with }."
    )


# -- reading the reply ----------------------------------------------------


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


def _said_id(got: dict[str, Any], key: str) -> str | None:
    """The id the model wrote under ``key``, or None.

    JSON null is None, but the model also writes the *string* "null" -- taking
    that literally would look up a tag called "null" and refuse for the wrong
    reason.
    """
    said = got.get(key)
    if isinstance(said, str) and said.strip().lower() in ("", "null", "none"):
        said = None
    return str(said) if isinstance(said, str) else None


def grasp_for(entity: Entity, calib: Calibration):
    """The motion-layer pose for landing on a decoded tag.

    A tag is a calibrated position with no orientation of its own, so the
    landing yaw is squared to the world axes rather than left wherever the pick
    put the wrist.
    """
    from mt4_vision.motion import square_place

    return square_place(entity.x, entity.y)


def destination_grasp(
    obs: Observation, action: "Action"
) -> tuple[Any | None, str]:
    """Where to release, or (None, why not). Squared to the world axes.

    Two shapes, and both end in the same physical gate. A ``dest_entity_id`` is
    a decoded tag, so its position comes from the calibration. A
    ``dest_point_px`` is a pixel the model chose on bare desk, so its position
    is the plain table-plane projection of that pixel -- no height, because a
    destination is a place on the table rather than an object with a top.

    The pixel is **not** snapped to the nearest tag or slot. "Somewhere clear"
    means the pixel it pointed at; nudging that onto a calibrated position
    would be the loop overriding the only thing it asked the model to decide.
    What the pixel must still survive is ``work_region_block_reason``, which is
    geometry: reach, the J1 keep-out, ground Z, the desk polygon and the camera
    frame.
    """
    from mt4_vision.workspace import work_region_block_reason

    if action.dest_entity_id:
        entity = obs.marker(action.dest_entity_id)
        if entity is None:
            return None, (
                f"{action.dest_entity_id} is not a tag decoded in this frame"
            )
        why = work_region_block_reason(entity.x, entity.y, obs.calib)
        if why is not None:
            return None, f"{entity.id} cannot be reached: {why}"
        return grasp_for(entity, obs.calib), ""

    readings = [
        p for p in (action.dest_point_px, action.dest_alt_point_px) if p is not None
    ]
    if not readings:
        return None, "no destination in the reply"
    from mt4_vision.motion import square_place

    why = ""
    for px, py in readings:
        x, y = obs.calib.pixel_to_robot(px, py, on_cube_top=False)
        blocked = work_region_block_reason(x, y, obs.calib)
        if blocked is None:
            return square_place(x, y), ""
        why = why or (
            f"the destination pixel ({px:.0f}, {py:.0f}) is robot "
            f"({x:.0f}, {y:.0f}), where {blocked}"
        )
    return None, why


def measure_source(
    obs: Observation, action: "Action"
) -> tuple[Any | None, str]:
    """Measure the boxed object, or (None, why not).

    Three choices, all deliberate:

    * **GrabCut from the box first.** ``measure_with_box_fallback`` prefers it,
      falls back to desk-deviation at the box centre, then to the raw box.
      Measured on one live frame, GrabCut from a box segmented 4 of 4 objects
      where the point path managed 1 of 4.
    * **Zero assumed height** (:data:`PICK_AT_TABLE_HEIGHT_MM`), so the
      returned XY is the table-plane projection of the mask centroid and the
      jaws will close at ``calib.table_z``.
    * **Orientation from that same mask** -- ``axis_yaw_deg`` is the long axis
      of the mask's ``minAreaRect``, which is what ``object_entity`` turns into
      a wrist angle and a 90/180 periodicity.

    The alternate coordinate reading is tried only if the first fails to
    measure, inside ``measure_grounding``. It is a retry, never a preference.
    """
    if action.source is None:
        return None, "no box in the reply"
    return measure_grounding(
        obs, action.source, object_height_mm=PICK_AT_TABLE_HEIGHT_MM
    )


def source_entity(obs: Observation, obj: Any) -> Entity:
    """The measured object as an entity, so the pick gate is the shared one.

    ``object_entity`` is where reach, the J1 keep-out, ground Z, the desk
    polygon, the jaw-width plan and neighbour clearance all live, and it is the
    same function the MCP registration path uses. Routing through it rather
    than re-deriving a pick test here is the reason a box from the model cannot
    command a pose the envelope would have refused.

    ``obs.scene`` is passed for the clearance check, which asks how close the
    nearest *other* detected thing is to the planned grasp point. That is the
    one place the cube detector still earns its keep, and it is an internal
    safety input -- nothing about it reaches the model.
    """
    from mt4_vision.entities import object_entity

    return object_entity(obj, "obj_1", scene=obs.scene)


def decide(
    obs: Observation,
    instruction: str,
    *,
    url: str = DEFAULT_URL,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Action:
    """Ask for one action and return it parsed, or a refusal explaining why.

    Every failure path produces ``ok=False`` with prose, never an exception.

    What this checks is that the reply is *well formed for the action it
    chose*: a known action, a usable box where a box is required, a
    destination where one is required, and a gripper state that permits it.
    What it does not check is whether the model was right about what it boxed
    -- there is nothing here to check that against, which is the point.
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
    why = str(got.get("reason", "") or "").strip()
    if kind not in ACTIONS:
        return Action("STOP", False, f"{kind!r} is not one of {', '.join(ACTIONS)}", raw=raw)

    if kind in ("DONE", "STOP"):
        # DONE is taken at face value, and so is STOP. Auditing DONE against the
        # instruction turned the model's own report of success into a failure,
        # which is the most expensive place to be wrong. Nothing downstream
        # re-examines it; every outcome line already says "commanded, not
        # checked".
        #
        # ok means "there is something to act on". A model-chosen STOP is still
        # a refusal; DONE is the loop's only successful exit.
        return Action(kind, kind == "DONE", why or kind.lower(), raw=raw)

    # ---- gripper state ---------------------------------------------------
    #
    # Not a judgement about the reply, but about the arm: the jaws are either
    # full or they are not, and no reply can change that. `held` is session
    # state corrected by hand (`/held`), because nothing on this rig can sense
    # a grip -- so when this refuses, the operator's next move is usually to
    # correct it rather than to re-word the task.
    if kind in ("TRANSFER", "PICK") and obs.held:
        return Action(
            "STOP", False,
            f"cannot pick while holding {obs.held} -- put it down first, or "
            "correct the gripper state with /held if the jaws are actually empty",
            raw=raw,
        )
    if kind == "PLACE" and not obs.held:
        return Action(
            "STOP", False,
            "cannot place with an empty gripper -- if the jaws are actually "
            "holding something, say so with /held <thing> and repeat the task",
            raw=raw,
        )

    # ---- the source box --------------------------------------------------
    source: Grounding | None = None
    if kind in ("TRANSFER", "PICK"):
        label = str(got.get("label") or "").strip() or "object"
        source, bad = box_grounding(got.get("box_2d"), obs.size, label=label)
        if source is None:
            return Action("STOP", False, bad, raw=raw)

    # ---- the destination -------------------------------------------------
    dest_id: str | None = None
    dest_pt: tuple[float, float] | None = None
    dest_alt: tuple[float, float] | None = None
    if kind in ("TRANSFER", "PLACE"):
        dest_id = _said_id(got, "dest_marker")
        readings = point_readings(got.get("dest_2d"), obs.size)
        if dest_id and readings:
            # Both given. The tag wins, because it is the one thing in this
            # protocol the model was handed rather than asked to see, and a
            # pixel that disagrees with it is the weaker of the two.
            readings = ()
        if dest_id is None and not readings:
            return Action(
                "STOP", False,
                f"{kind} needs a destination: either dest_marker naming a tag "
                "listed in the prompt, or dest_2d as a pixel on the desk",
                raw=raw,
            )
        if dest_id is not None and obs.marker(dest_id) is None:
            listed = ", ".join(e.id for e in obs.markers) or "none"
            return Action(
                "STOP", False,
                f"the reply names {dest_id}, which did not decode in this "
                f"frame -- the tags in view are: {listed}",
                raw=raw,
            )
        dest_pt = readings[0] if readings else None
        dest_alt = readings[1] if len(readings) > 1 else None

    return Action(
        kind, True, why or kind.lower(),
        label=None if source is None else source.label,
        source=source,
        dest_entity_id=dest_id,
        dest_point_px=dest_pt,
        dest_alt_point_px=dest_alt,
        raw=raw,
    )
