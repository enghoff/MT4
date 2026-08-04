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
* **A destination is a point, not an id.** ``dest_2d`` is the only destination
  channel there is, on the same 0-1000 grid the tag positions are printed in.
  Landing on a tag means echoing that tag's listed coordinates back. An id would
  let a destination name something the reply never located: "on the red cube"
  answered as a tag id is exact by construction, agrees with nothing, and can be
  46mm from the red cube. A coordinate cannot hide that -- it is drawn on the
  overlay, and it is where the arm goes.

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
jaws go -- with the yaw of the GrabCut mask's long axis. Nothing about the grip
*height* depends on how tall the thing is: no object here is taller than the
jaws' vertical clearance, so the lowest grip is always available.

Where the jaws go in **XY** does depend on it. The camera looks at the desk from
a steep angle, so a tall object's image is smeared outward from the point the
camera sits above, and the middle of that smear is not above the object's
footprint. Projecting it as if it were flat aims the jaws past the object --
measured 18.1-22.4mm outward on 20mm cubes, against the ~10mm they tolerate.
So the measurement infers the object's height and unprojects accordingly; see
``locate._height_corrected`` for how, and for what it costs.

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
    "ON_TAG_MM", "Action", "Grounding", "Observation",
    "box_grounding", "box_readings", "build_prompt", "decide",
    "destination_grasp", "measure_grounding", "observe",
    "point_readings", "tag_at", "to_frame_pixels", "load_calibration",
]

# Short replies only: a decision is an action, a box, a destination and a clause
# of reason, and the 1664-token static cache holds one image plus this
# comfortably.
MAX_NEW_TOKENS = 220

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
    # Where to release, in frame pixels, and the same reply read as raw pixels
    # as a retry. This is the only destination form: landing on a tag means the
    # model echoed that tag's listed coordinates, so a destination is always a
    # place the reply itself pointed at rather than a name it looked up.
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
    given carries a circled id beside each decoded tag, and nothing else.
    Circling the detected cubes would tell it where the stack believes the
    interesting things are, which is the picture-shaped version of handing it a
    cube list.

    The grid over it is labelled in the **0-1000 scale the reply is read in**,
    not in pixels, so the drawn numbers and the requested space agree. That is
    the whole argument for it. Measured across five live cube layouts, a pixel
    grid, no grid and a 0-1000 grid are indistinguishable on action choice
    (34, 36 and 34 of 45), on box error and on picking the wrong cube (never,
    in any of them) -- so the choice rests on the labels naming a real space,
    not on a score. See ``preview.annotate_for_pointing``.

    The circles stay regardless: they are the only thing tying a printed tag
    number to the tag in the image, and on tasks that name a tag by what is
    sitting on it they resolve 9 of 14 against 7 of 14 on a bare frame.

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
        frame, [e for e in snapshot.entities if e.kind == KIND_MARKER], grid="norm"
    )
    return Observation(
        frame=frame, annotated=annotated, snapshot=snapshot, calib=calib,
        held=held, history=tuple(history), scene=scene,
    )


# -- prompt ---------------------------------------------------------------


def _marker_lines(obs: "Observation") -> str:
    """The ArUco tags, id and position. The whole of what the model is told.

    These coordinates are not just a hint about where a tag is -- they are the
    *answer* for a destination on one, because ``dest_2d`` is the only
    destination form. So the round trip has to be tight, and it is: a tag's
    listed pixel is its calibrated robot position run through
    ``robot_to_pixel``, and reading an exact echo back through
    ``pixel_to_robot`` inverts the same homography. All that survives is the
    rounding to whole 0-1000 units, measured across the five tags at
    **0.05-0.48mm**.

    What that rounding does *not* bound is the model mistyping a coordinate.
    One unit of the 0-1000 grid is 0.67-1.42mm on this desk depending where,
    so a ten-unit slip lands 6.6-14.0mm out -- past what a place tolerates.
    That failure is visible, which is the trade: it shows up as a drawn point
    in the wrong place instead of as a correct-looking tag id.

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
        "box_2d is the thing to pick up; dest_2d is where it goes. Use this "
        "whenever the task names BOTH a thing and a place for it, and the "
        "gripper is empty\n"
        "  PICK     - pick something up and hold it, when the task names no "
        "destination at all (only if the gripper is empty). box_2d is the "
        "thing to pick up\n"
        "  PLACE    - put down what is ALREADY held. dest_2d says where\n"
        "  DONE     - the task is complete\n"
        "  STOP     - the task cannot be done, or you cannot tell which thing "
        "is meant\n\n"
        "Reply with ONLY a JSON object, no prose, no markdown fence:\n"
        '{"action": "...", "box_2d": [x1, y1, x2, y2], "label": "...", '
        '"dest_2d": [x, y], "reason": "<one short clause>"}\n\n'
        "Every coordinate you give is on a 0-1000 scale across the image: 0 is "
        "the left edge, 1000 the right edge, and the same top to bottom. The "
        "tag positions above are written that way too, and the numbered "
        "gridlines drawn on the image are that same scale, every 100 units.\n"
        "box_2d is a tight box around the WHOLE of the object to pick up. The "
        "jaws are aimed using that box, so it must contain that object and as "
        "little else as possible -- not the desk around it, and not a "
        "neighbouring object. Required for TRANSFER and PICK.\n"
        "label is a short noun for whatever you boxed.\n"
        "dest_2d says where something goes, and it is always a point on the "
        "image -- there is no way to name a destination, only to point at one. "
        "Required for TRANSFER and PLACE. Three cases, all the same field:\n"
        "  - onto a tag: copy that tag's two numbers from the list above, "
        "exactly as they are written there\n"
        "  - onto or against another object: the point on THAT object. An "
        "object is a legal destination, not only bare desk -- when the task "
        "says to put something on top of something else, dest_2d is the thing "
        "it goes on, wherever in the image you can see that thing to be\n"
        "  - 'somewhere clear', 'on the table', 'not on a marker': any point "
        "on empty desk\n"
        # The one text-derived rule left, and it is about the one datum we
        # supply rather than about the model's own grounding. Measured live
        # 2026-08-03, "place it on marker 0" on a desk of markers 1-4: the model
        # answered marker_3, then marker_1, then "task already completed", never
        # once saying the number was not there.
        "A tag number in the task is an identity, not a hint: 'marker 0' means "
        "the tag whose id is exactly marker_0, so dest_2d must be the pair of "
        "numbers listed above for marker_0. If that id is not listed then the "
        "tag is not on this desk -- answer STOP and say which number is "
        "missing. Another tag's numbers are never a substitute for it, and "
        "neither is a point near where you expect it to be.\n"
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


def _named_a_tag(got: dict[str, Any]) -> str:
    """Refusal clause for a reply that named a tag instead of pointing at one.

    The one malformed destination worth spelling out, because the correction
    lives in the reply rather than on the desk: the tag's coordinates are
    already in the prompt and only have to be copied across. The model writes
    the *string* "null" as well as JSON null, and neither is a name.
    """
    said = got.get("dest_marker")
    if not isinstance(said, str) or said.strip().lower() in ("", "null", "none"):
        return ""
    return (
        f" -- the reply gives dest_marker {said.strip()!r}, and a destination "
        "is never a name; copy the coordinates listed for that tag into dest_2d"
    )


# Half a printed tag, which measures 41.2-44.1mm on a side across the five on
# this desk. A destination within this of a tag's listed position is standing on
# that tag, and the transcript says so by name.
ON_TAG_MM = 22.0


def tag_at(obs: Observation, x: float, y: float) -> Entity | None:
    """The tag a robot-frame destination is standing on, or None.

    Reporting only. Nothing routes a move through this: the coordinate the model
    gave is where the arm goes, whether or not a tag happens to be under it.
    What it buys is a transcript line an operator can check at a glance -- an
    echoed tag position reads as "onto marker_2", and a destination that is
    merely *near* one reads as the bare pixel it is.
    """
    best: Entity | None = None
    best_d = ON_TAG_MM
    for e in obs.markers:
        d = ((e.x - x) ** 2 + (e.y - y) ** 2) ** 0.5
        if d < best_d:
            best, best_d = e, d
    return best


def destination_grasp(
    obs: Observation, action: "Action"
) -> tuple[Any | None, str]:
    """Where to release, or (None, why not). Squared to the world axes.

    One shape: a point the model gave, projected onto the table plane. No
    height, because a destination is a place on the table rather than an object
    with a top.

    The point is **not** snapped to the nearest tag or slot, and that holds even
    when it lands squarely on one. A destination on marker 2 is marker 2's
    coordinates echoed back, so snapping could only ever move the arm somewhere
    the reply did not ask for -- and it would hide the case this protocol exists
    to expose, where the model points at a tag while the task named an object
    46mm away. What the point must survive is ``work_region_block_reason``,
    which is geometry: reach, the J1 keep-out, ground Z, the desk polygon and
    the camera frame.
    """
    from mt4_vision.workspace import work_region_block_reason

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
        on = tag_at(obs, x, y)
        why = why or (
            f"the destination pixel ({px:.0f}, {py:.0f}) is robot "
            f"({x:.0f}, {y:.0f})"
            f"{'' if on is None else f', on {on.id}'}, where {blocked}"
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
    * **Height inferred from the silhouette**, so the returned XY is where the
      object meets the table rather than where its top face images. The jaws
      still close at ``calib.table_z`` regardless -- height moves the aim
      point, never the grip height. See ``locate._height_corrected``.
    * **Orientation from that same mask** -- ``axis_yaw_deg`` is the long axis
      of the mask's ``minAreaRect``, which is what ``object_entity`` turns into
      a wrist angle and a 90/180 periodicity.

    The alternate coordinate reading is tried only if the first fails to
    measure, inside ``measure_grounding``. It is a retry, never a preference.
    """
    if action.source is None:
        return None, "no box in the reply"
    return measure_grounding(obs, action.source)


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
    dest_pt: tuple[float, float] | None = None
    dest_alt: tuple[float, float] | None = None
    if kind in ("TRANSFER", "PLACE"):
        readings = point_readings(got.get("dest_2d"), obs.size)
        if not readings:
            return Action(
                "STOP", False,
                f"{kind} needs dest_2d, a point [x, y] on the 0-1000 grid "
                f"inside the image{_named_a_tag(got)}",
                raw=raw,
            )
        dest_pt = readings[0]
        dest_alt = readings[1] if len(readings) > 1 else None

    return Action(
        kind, True, why or kind.lower(),
        label=None if source is None else source.label,
        source=source,
        dest_point_px=dest_pt,
        dest_alt_point_px=dest_alt,
        raw=raw,
    )
