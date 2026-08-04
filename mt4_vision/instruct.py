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
* **An answer about several things is several boxes.** ``REPORT`` carries an
  ``objects`` list, one box per thing found, and each box is measured and put
  to the pick gates exactly as a PICK's is (:func:`measure_report`). That is
  how "find all the pickable objects" gets a true answer: which things the arm
  can take is a fact about reach, the keep-out, the desk edge and the jaw
  width, none of which is visible in the photograph the model is looking at.
* **A destination is a point, not an id.** ``dest_2d`` is the only destination
  channel there is, on the same 0-1000 grid the tag positions are printed in.
  Landing on a tag means echoing that tag's listed coordinates back. An id would
  let a destination name something the reply never located: "on the red cube"
  answered as a tag id is exact by construction, agrees with nothing, and can be
  46mm from the red cube. A coordinate cannot hide that -- it is drawn on the
  overlay, and it is where the arm goes.

**What refuses, and why that is not second-guessing.** The measurement must
survive segmentation, the two-window stability check and the plausibility band,
and the resulting pose must clear reach, the J1 keep-out, ground Z, the
jaw-width plan and the desk edge (``entities.object_entity``). Those gates read
geometry, never the model's judgement about *what* a thing is. They are what
makes trusting the rest safe: nothing here can command a pose the envelope
would reject.

**What deliberately does not refuse.** Three checks the rest of the stack
applies are off here, because each substitutes a judgement for the operator's
rather than reporting a physical limit:

* **Camera coverage on a destination.** It says the result would leave the
  frame and could not be re-detected, not that the arm cannot place there. On
  this calibration it vetoes 316 cm2 of the 1541 cm2 the arm can hold a grasp
  pose over, 21%. Worth keeping where the stack picks a spot itself, which is
  why ``work_region_block_reason`` still defaults to enforcing it.
* **Neighbour clearance on a pick.** It counts ``scene.cubes`` and nothing
  else, so it refuses a grasp beside a cube and permits the same grasp beside
  a pen -- in a loop that never tells the model cubes are a category.
* **The gripper state.** ``held`` is typed in by hand, not sensed, so it is a
  belief and a stale one blocks a fine action.

None of the three can produce an illegal pose, and each failure they used to
prevent is visible on the desk and recoverable.

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
    report_groundings,
    to_frame_pixels,
)
from mt4_vision.preview import annotate_for_pointing
from mt4_vision.qwen import DEFAULT_URL, QwenError, ask
from mt4_vision.scene import capture_scene

__all__ = [
    "ACTIONS", "COORD_SCALE", "MAX_NEW_TOKENS", "MAX_BOX_FRAME_SHARE",
    "ON_TAG_MM", "Action", "Finding", "Grounding", "Observation",
    "box_grounding", "box_readings", "build_prompt", "decide",
    "destination_grasp", "measure_grounding", "measure_report", "observe",
    "point_readings", "report_groundings", "tag_at", "to_frame_pixels",
    "load_calibration",
]

# A decision is an action, a box, a destination and whatever the model has to
# say about it. The budget is well past what the JSON fields need so that
# ``reason`` can carry a real answer -- a question about the desk, or what a
# STOP actually ran into -- rather than the clause a tight cap forces. It is
# the only channel back from the model, and the arm-facing fields are only ever
# a few dozen tokens of it. A cap is a ceiling, not a target: a short reply
# still decodes short.
#
# A REPORT spends it on boxes rather than on prose -- one
# ``{"box_2d": [x, y, x, y], "label": "..."}`` entry is about 30 tokens of
# JSON, so this holds a list far longer than any desk this arm works over.
# Running out is not silent: a reply cut off mid-array has no balanced
# ``{...}`` for ``_first_json_object`` to find, so it refuses as "no JSON object
# in the reply" rather than reporting the objects that fit and losing the rest.
#
# It does not cost the service's fast path, because this prompt never had it.
# ``server.py`` takes the reused StaticCache when ``prompt_len + max_new_tokens
# <= CACHE_LEN`` (1664), and this prompt measures **2057 tokens** on a 1280x720
# frame with no tags listed -- over the cache on its own, so every decision
# already runs on the dynamic cache and eager decode whatever this number is.
# Restoring the fast path means raising ``QWEN_VL_CACHE_LEN`` past prompt +
# reply, which slows every other request the service serves; that is a
# service-level call, not one this constant can make.
MAX_NEW_TOKENS = 640

# The retry decode in `decide`, and the only sampled call in this loop. Low
# enough to stay on-task, non-zero because the point of the retry is to land
# somewhere the greedy path did not.
RETRY_TEMPERATURE = 0.6

# TRANSFER leads because it is the shape of nearly every real task here: move a
# thing to a place, in one step, with no park-look-decide in the middle that
# could not answer anything anyway. PICK and PLACE are for the halves that occur
# alone -- "pick up the stapler" with no destination, and putting down something
# the gripper is already holding.
#
# There is no locate-and-register action, because there is no list to register
# into: every pick measures a box on the frame the box was drawn on, so every
# pick *is* a locate-at-pixel and costs no extra park-capture-decide round trip.
#
# REPORT is the answer channel for a task whose answer is a list of things
# rather than a sentence -- "find all the pickable objects", "what is on the
# desk". It moves nothing. Prose in ``reason`` cannot answer it, for a reason
# that is not about writing: whether the arm can pick a thing up depends on
# reach, the J1 keep-out, ground Z, the desk polygon and the width of the jaws,
# and a model looking at a photograph can see none of those. Boxes can be
# measured and put to the real gates; a sentence cannot.
ACTIONS = ("TRANSFER", "PICK", "PLACE", "REPORT", "DONE", "STOP")


@dataclass(frozen=True)
class Observation:
    """One capture: the frame, the tags on it, and what the arm is holding.

    ``snapshot`` is the **full** detection -- cubes included -- because the
    operator summary line counts them and because it is the same object the
    rest of the stack passes around. It is never shown to the model, and no
    gate on this path reads the cube list. :attr:`markers` is the model-facing
    half, and the only thing :func:`build_prompt` and the overlay are allowed
    to read. Keeping the split here, on the type, is deliberate: the guarantee
    that the model sees nothing pickable should be one attribute a reader can
    check, rather than a discipline spread over two call sites.
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
    # The detections the snapshot was built from. Passed to ``object_entity``
    # for the calibration it carries, which the reach test and the jaw-width
    # plan both need; its cube list is not read on this path. Never reaches
    # the prompt.
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

    ``declined`` separates those two, because they deserve opposite handling.
    A forced ``STOP`` -- no JSON, an unknown action, a missing box, a service
    error -- says nothing about the request and is worth another look. A
    ``STOP`` the model chose *is* its answer to the request, and asking the
    same question again gets the same answer in different words.

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
    # True only when the model itself answered STOP. A STOP that no JSON, an
    # unknown action, a missing field or a service error forced leaves this
    # False, because none of those is a judgement about the request.
    declined: bool = False
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
    # A second point saying which way to lay the object down, in frame pixels.
    # Optional: absent means the reply had no orientation opinion and the
    # destination is squared to the world axes. See ``instruct._place_grasp``.
    dest_axis_px: tuple[float, float] | None = None
    # What a REPORT boxed: one grounding per object, in the order the reply
    # wrote them, and empty for every other action. An empty report with no
    # notes is a real answer -- the reply saying there is nothing to list.
    report: tuple[Grounding, ...] = ()
    # Entries of that list that would not read, one line each. Carried rather
    # than dropped because a report is a claim of completeness: "found 4" with a
    # fifth entry silently discarded is a wrong answer to "find all ...".
    report_notes: tuple[str, ...] = ()
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
        if self.dest_axis_px:
            out["dest_axis_px"] = [
                round(self.dest_axis_px[0], 1), round(self.dest_axis_px[1], 1)
            ]
        if self.report:
            out["objects"] = [
                {
                    "label": g.label,
                    "box_px": [round(v, 1) for v in g.box_px],
                    "point_px": [round(g.point_px[0], 1), round(g.point_px[1], 1)],
                }
                for g in self.report
            ]
        if self.report_notes:
            out["report_notes"] = list(self.report_notes)
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
            f"The gripper is holding a {obs.held}. That is a fact about the "
            "arm, not a step still owed: if the task wanted it put somewhere, "
            f"PLACE says where; if the task only asked for it to be picked up "
            "or held, the task is already done.\n"
            f"The {obs.held} is in the jaws, NOT lying on the desk, so there "
            "is nothing in the image to box for it and PLACE needs only "
            "dest_2d. Anything you can see on the desk is a DIFFERENT object, "
            f"even if it looks like a {obs.held} -- boxing one means picking "
            "that up and abandoning what you are carrying."
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
    # Descriptive, not an order: TRANSFER really is one continuous motion where
    # PICK then PLACE is two, with a park, a fresh photograph and a second
    # decision in between. Told that, the model can weigh it. Told "use
    # TRANSFER whenever the task names both", it cannot decompose a task at all
    # -- and some tasks need the look in the middle.
    #
    # The preference is stated only with empty jaws, because with full ones it
    # is false: TRANSFER there is not the same errand done in one go, it is
    # picking up a second object and dropping the first. Measured live, a
    # standing "prefer TRANSFER" against the task "place the held object on
    # marker 2" sent the arm to box a lookalike on the desk instead, twice.
    if obs.held:
        actions = (
            "  TRANSFER - pick something up and put it somewhere in one "
            "motion. box_2d is the thing to pick up; dest_2d is where it "
            "goes. It picks up a DIFFERENT object from the one in the jaws, "
            "so it abandons what you are carrying\n"
            "  PICK     - pick up a DIFFERENT object and hold that instead, "
            "abandoning what you are carrying. box_2d is the thing to pick "
            "up\n"
        )
    else:
        actions = (
            "  TRANSFER - pick something up and put it somewhere in one "
            "motion. box_2d is the thing to pick up; dest_2d is where it "
            "goes. Prefer it over PICK then PLACE, which stops for a second "
            "photograph in between\n"
            "  PICK     - pick something up and hold it. box_2d is the thing "
            "to pick up. The arm keeps holding it until something puts it "
            "down\n"
        )
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
        f"{actions}"
        "  PLACE    - put down what is held. dest_2d says where\n"
        "  REPORT   - the task asks what is on the desk rather than for "
        "something to be moved. Answer it by boxing the things it asks about: "
        "objects carries one box each. Nothing moves\n"
        "  DONE     - the task is complete\n"
        "  STOP     - the task cannot be done, or you cannot tell which thing "
        "is meant\n\n"
        "Reply with ONLY a JSON object, no prose, no markdown fence:\n"
        '{"action": "...", "box_2d": [x1, y1, x2, y2], "label": "...", '
        '"dest_2d": [x, y], "dest_axis_2d": [x, y], '
        '"objects": [{"box_2d": [x1, y1, x2, y2], "label": "..."}], '
        '"reason": "..."}\n\n'
        "Every coordinate you give is on a 0-1000 scale across the image: 0 is "
        "the left edge, 1000 the right edge, and the same top to bottom. The "
        "tag positions above are written that way too, and the numbered "
        "gridlines drawn on the image are that same scale, every 100 units.\n"
        "box_2d is a tight box around the WHOLE of the object to pick up. The "
        "jaws are aimed using that box, so it must contain that object and as "
        "little else as possible -- not the desk around it, and not a "
        "neighbouring object. Required for TRANSFER and PICK.\n"
        "label is a short noun for whatever you boxed.\n"
        "reason is the only thing you can say to the person who set the task. "
        "A sentence or two, more when there is more to say. If the task asked "
        "a question rather than for a movement, answer it there and pick "
        "DONE.\n"
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
        "dest_axis_2d is OPTIONAL: a second, well-separated point, such that "
        "the line from dest_2d to it is the direction the object should lie "
        "in. Give it only when the task says something about direction; null "
        "otherwise, and the object goes down square to the desk.\n"
        "objects is REPORT's answer, and the only way to answer about more "
        "than one thing: a list with one entry per thing you found, each "
        '{"box_2d": [x1, y1, x2, y2], "label": "..."}, every box drawn the '
        "same way box_2d is. Required for REPORT, and an empty list [] is a "
        "real answer -- it says there are none. Whether the arm can actually "
        "pick a thing up is not yours to judge and not something the image "
        "shows: how far it can reach, how close to its own base it can work, "
        "where the desk ends and how wide the jaws open decide that, and every "
        "box you list gets measured and put to those tests after you answer. "
        "So list every candidate object, and let the tests rule out the ones "
        "that fail. Leaving something out because it looks hard to grasp "
        "deletes it from the answer, and nobody can tell that it was there.\n"
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


def _axis_or_none(said: Any) -> Any:
    """``dest_axis_2d`` as given, or None when it is the model's way of null.

    An optional field gets an "unused" value rather than an absent one, and
    this build writes ``[0, 0]`` -- observed live alongside a real ``dest_2d``,
    which would otherwise read as "lay it along the line to the top-left corner
    of the frame" and turn the wrist to a direction nothing asked for. A
    genuine axis pointing at the exact frame origin is not a thing anyone can
    mean, so the reading is safe to spend.

    ``[null, null]`` needs no help; ``point_readings`` already drops it.
    """
    if isinstance(said, (list, tuple)) and len(said) >= 2:
        try:
            if all(float(v) == 0.0 for v in said[:2]):
                return None
        except (TypeError, ValueError):
            return said
    return said


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


# A destination axis shorter than this in robot millimetres is two points on
# top of each other, not a direction: the angle it implies is rounding noise.
MIN_DEST_AXIS_MM = 8.0


def _place_grasp(
    obs: Observation,
    action: "Action",
    x: float,
    y: float,
    px: float,
    py: float,
) -> Any | None:
    """The destination grasp implied by ``dest_axis_px``, or None for squared.

    The reply points at a second place on the image rather than naming an
    angle, for the reason every other destination is a point: an angle would
    need a frame to be measured in, and the model has no access to the robot's.
    Two pixels have one meaning. Both go through the flat table homography and
    the direction between them is read off in robot millimetres, so the answer
    is in the frame the wrist is commanded in.

    The period is 180°, because what the reply describes is an axis to lay the
    object along and the jaws close across it -- an axis and its reverse are
    the same instruction. When no axis is given, the caller squares instead.
    """
    from mt4_vision.motion import YAW_PERIOD_LONG_AXIS, Grasp

    if action.dest_axis_px is None:
        return None
    ax, ay = obs.calib.pixel_to_robot(
        action.dest_axis_px[0], action.dest_axis_px[1], on_cube_top=False
    )
    dx, dy = ax - x, ay - y
    if (dx * dx + dy * dy) ** 0.5 < MIN_DEST_AXIS_MM:
        return None
    import math

    return Grasp(
        x, y,
        yaw_deg=math.degrees(math.atan2(dy, dx)),
        yaw_period_deg=YAW_PERIOD_LONG_AXIS,
    )


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
    46mm away.

    What the point must survive is ``work_region_block_reason`` minus its
    camera-coverage test: reach, the J1 keep-out, ground Z and the desk edge.
    Those say the arm cannot do the thing. Coverage says only that the result
    would leave the frame and could not be re-detected, which is the operator's
    call to make about a spot they pointed at -- and it costs 316 cm2 of the
    1541 cm2 the arm can hold a grasp pose over, 21%.

    Orientation comes from ``dest_axis_px`` when the reply carried one, and is
    squared to the world axes when it did not. See :func:`build_prompt`.
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
        blocked = work_region_block_reason(x, y, obs.calib, require_camera=False)
        if blocked is None:
            return _place_grasp(obs, action, x, y, px, py) or square_place(x, y), ""
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
    * **Height assumed to be ``calib.cube_height_mm``**, not inferred from the
      silhouette. Height cannot be measured from one view, and on this steeply
      oblique mount every millimetre of error in it moves the aim point by
      1.4-1.9mm, so a bad guess is what a missed grip is made of. Inferring it
      from the silhouette guessed 0.4 to 16.8mm for 20mm cubes across nine
      readings of one frame (2026-08-04), because the silhouette it is derived
      from takes in shadow and marker paper and its size swings with the box
      the model drew -- the same red cube measured 38x11, 57x27 and 53x23mm
      under three box sizes. Assuming the configured height instead routes the
      centroid through the calibrated cube-top map and lands 1.9-4.8mm from the
      cube detector's reading on the same objects. The jaws still close at
      ``calib.table_z`` regardless -- height moves the aim point, never the
      grip height. See ``locate._height_corrected``.
    * **Orientation from that same mask** -- ``axis_yaw_deg`` is the long axis
      of the mask's ``minAreaRect``, which is what ``object_entity`` turns into
      a wrist angle and a 90/180 periodicity.

    The alternate coordinate reading is tried only if the first fails to
    measure, inside ``measure_grounding``. It is a retry, never a preference.
    """
    if action.source is None:
        return None, "no box in the reply"
    return measure_grounding(
        obs, action.source, object_height_mm=obs.calib.cube_height_mm
    )


@dataclass(frozen=True)
class Finding:
    """One object a REPORT boxed, measured and put to the pick gates.

    The model boxes; this says what the thing measures and whether the arm can
    take it. :attr:`pickable` is never the model's opinion -- it is
    ``entities.object_entity``'s, the same predicate a PICK has to satisfy
    before the jaws move. So a report cannot advertise a target that a pick
    would then refuse, and it cannot omit one a pick would accept.

    ``obj`` and ``entity`` are both None when the box would not segment;
    :attr:`trouble` says why, and the entry stays in the list. A box that named
    something the segmenter could not cut is still an object the model saw, and
    dropping it would answer "find all ..." with a shorter list than the reply
    gave.

    :attr:`index` numbers the row and the rectangle drawn on the frame with the
    same number, so a line of terminal text and a box on the desk can be tied
    together without counting from the top.
    """

    index: int
    label: str
    grounding: Grounding
    obj: Any | None = None
    entity: Entity | None = None
    trouble: str = ""

    @property
    def box_px(self) -> tuple[float, float, float, float] | None:
        """The box the model drew, in frame pixels. What the overlay draws."""
        return self.grounding.box_px

    @property
    def pickable(self) -> bool:
        return self.entity is not None and self.entity.pickable

    @property
    def note(self) -> str:
        """One clause: pickable, or the gate that says otherwise."""
        if self.entity is None:
            return self.trouble
        if self.entity.pickable:
            return "pickable"
        return self.entity.reason or "not pickable"

    def line(self) -> str:
        """The transcript row, in the column layout ``Snapshot.table`` uses."""
        head = f"{self.index}. {self.label:<14}"
        if self.entity is None or self.obj is None:
            return f"{head} could not be measured: {self.trouble}"
        return (
            f"{head} ({self.entity.x:>6.1f},{self.entity.y:>7.1f})  "
            f"{self.obj.long_mm:.0f}x{self.obj.short_mm:.0f}mm  {self.note}"
        )


def measure_report(obs: Observation, action: "Action") -> tuple[Finding, ...]:
    """Measure everything a REPORT boxed and ask the pick gates about each.

    Object for object, this is the same work :func:`measure_source` does for a
    single pick: GrabCut inside the box the model drew, on the frame it drew it
    on, at ``calib.cube_height_mm`` rather than a height read off the silhouette
    (see :func:`measure_source` for what inferring it costs). Then
    :func:`source_entity`, which is where reach, the J1 keep-out, ground Z, the
    desk polygon and the jaw-width plan live.

    Running the real gate is the whole point of answering with boxes instead of
    prose. "Can the arm pick this up" is geometry the model cannot see, so the
    only honest answer comes from the predicate the arm itself obeys.

    Nothing moves and nothing is registered. The ids are positional, for the
    report's own rows, and do not outlive it -- ``obs.snapshot`` is unchanged.

    Neighbour clearance stays off, as it is for a pick on this path: it counts
    only ``scene.cubes``, so it would call a grasp beside a cube blocked and the
    identical grasp beside a pen clear, in a loop that never tells the model
    cubes are a category.
    """
    out: list[Finding] = []
    for i, g in enumerate(action.report, start=1):
        obj, why = measure_grounding(
            obs, g, object_height_mm=obs.calib.cube_height_mm
        )
        if obj is None:
            out.append(Finding(i, g.label, g, trouble=why))
            continue
        out.append(
            Finding(
                i, g.label, g,
                obj=obj, entity=source_entity(obs, obj, eid=f"obj_{i}"),
            )
        )
    return tuple(out)


def source_entity(obs: Observation, obj: Any, *, eid: str = "obj_1") -> Entity:
    """The measured object as an entity, so the pick gate is the shared one.

    ``object_entity`` is where reach, the J1 keep-out, ground Z, the desk
    polygon, the jaw-width plan and neighbour clearance all live, and it is the
    same function the MCP registration path uses. Routing through it rather
    than re-deriving a pick test here is the reason a box from the model cannot
    command a pose the envelope would have refused.

    ``obs.scene`` is passed for the calibration it carries, which the reach
    test and the jaw-width plan both need. The neighbour check it could also
    feed is off here: it counts only ``scene.cubes``, so it refuses a grasp
    beside a cube and permits the identical grasp beside a pen, and this loop
    never tells the model that cubes are a category. A veto the model cannot
    see, cannot argue with and cannot route around is worse than a nudged
    neighbour, which is visible on the desk and recoverable.

    ``eid`` names the entity. A pick measures one thing and takes the default;
    a report measures several in one frame and numbers them, so its rows do not
    all claim to be ``obj_1``.
    """
    from mt4_vision.entities import object_entity

    return object_entity(obj, eid, scene=obs.scene, require_clearance=False)


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
    chose*: a known action, a usable box where a box is required, a destination
    where one is required, and a list where a REPORT's answer belongs. What it
    does not do is measure any of it -- a REPORT's boxes go to
    :func:`measure_report` after this returns, for the same reason a pick's
    single box goes to :func:`measure_source`. What it does not check is
    whether the model was right about what it boxed -- there is nothing here to
    check that against, which is the point. Nor does it check the gripper:
    ``held`` is a belief a person typed in, not a reading.

    A reply carrying no JSON at all is asked once more, **sampled**. Re-asking
    greedily would be pointless -- same prompt, same weights, same reply -- so
    the retry is the one place this loop does not decode greedily. It costs one
    GPU call and no motion, against losing the whole instruction to a stray
    token.
    """
    prompt = build_prompt(obs, instruction)
    raw = ""
    got = None
    for sampled in (False, True):
        try:
            reply = ask(
                prompt, obs.annotated, url=url,
                max_new_tokens=max_new_tokens,
                do_sample=sampled,
                temperature=RETRY_TEMPERATURE if sampled else None,
            )
        except QwenError as exc:
            return Action("STOP", False, f"vision-language service failed: {exc}")
        raw = reply.text.strip()
        got = _first_json_object(raw)
        if got is not None:
            break
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
        #
        # This is the one place `declined` is set, and it is what makes a STOP
        # here terminal while a forced one further down stays retryable.
        return Action(
            kind, kind == "DONE", why or kind.lower(),
            declined=kind == "STOP", raw=raw,
        )

    if kind == "REPORT":
        # ok=True whatever the list holds, including empty: a report moves
        # nothing, so there is no pose to be wrong about, and "there are none"
        # is an answer to the question that was asked. What the list is worth
        # is settled by ``measure_report``, object by object, against the gates.
        said = got.get("objects")
        if not isinstance(said, list):
            return Action(
                "STOP", False,
                "REPORT answers with objects, a list of "
                '{"box_2d": [x1, y1, x2, y2], "label": "..."} entries -- one '
                f"per thing found, [] when there are none -- and gave {said!r}",
                raw=raw,
            )
        found, notes = report_groundings(said, obs.size)
        return Action(
            kind, True, why or "report",
            report=found, report_notes=notes, raw=raw,
        )

    # No gate on the gripper. `held` is a belief typed in by hand (`/held`),
    # not a reading -- nothing on this rig can sense a grip -- so refusing on
    # it blocks a physically fine action whenever the belief is stale, which
    # it most often is after an abort mid-carry. Both directions are
    # recoverable and both are visible on the desk: a PLACE with empty jaws is
    # a move and an open command, and a PICK while genuinely holding drops the
    # carried object from grab height. The transcript records what was
    # commanded either way.

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
    dest_axis: tuple[float, float] | None = None
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
        # Optional, so an unreadable one is dropped rather than refused: the
        # destination still stands, and it simply gets squared.
        axis = point_readings(_axis_or_none(got.get("dest_axis_2d")), obs.size)
        dest_axis = axis[0] if axis else None

    return Action(
        kind, True, why or kind.lower(),
        label=None if source is None else source.label,
        source=source,
        dest_point_px=dest_pt,
        dest_alt_point_px=dest_alt,
        dest_axis_px=dest_axis,
        raw=raw,
    )
