"""Addressable scene entities: stable ids, and why a thing can't be picked.

The layer a language model actually talks to. A ``Scene``
(:mod:`mt4_vision.scene`) is detections and marker states; a ``Snapshot`` turns
that into a flat list of things with **ids**, so a caller can say "pick
``cube_2``" instead of "pick the red one" and have exactly one referent.

Two defects this replaces:

* ``mt4_pick_cube(color)`` resolved ambiguity silently via ``pick_largest_cube``
  -- no way to name *which* red cube, and no way for the request to fail.
* A blob the pick filters rejected simply wasn't reported, so the honest answer
  ("that is the arm's own paint, behind the desk's back edge") came back as
  "no red cube in view". Every detection that is a real object appears here,
  non-pickable ones included, each carrying the gate that stopped it.

Pure: no camera, no serial, and no ``Calibration`` required to build a snapshot
(only :meth:`Entity.as_grasp` optionally takes one), so this is testable
without hardware -- matching ``tests/test_scene.py``'s calib-free style.

Ids are per-snapshot and the snapshot carries a ``token``. That is deliberate:
``scene.py`` keeps no persistent tracks, so a vacated pose cannot linger, and an
id must not outlive the frame that justified it. Callers re-acquire against a
fresh frame before acting (see the MCP ``mt4_pick``), rather than trusting an id
to still mean the same object minutes later.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


from mt4_vision.calib import Calibration
from mt4_vision.detect import CubeDetection
from mt4_vision.motion import (
    YAW_PERIOD_LONG_AXIS,
    YAW_PERIOD_SQUARE,
    Grasp,
)
from mt4_vision.scene import (
    PICK_MAX_AREA,
    PICK_MIN_AREA,
    Scene,
)
from mt4_vision.workspace import (
    KEEPOUT_RADIUS_MM,
    MAX_REACH_MM,
    PICK_CLEARANCE_MM,
    MarkerSlot,
    dist_mm,
    is_mp_reachable_xy,
    work_region_block_reason,
)

if TYPE_CHECKING:  # only for the annotation; locate.py must not be a dependency
    from mt4_vision.locate import LocatedObject

KIND_CUBE = "cube"
KIND_MARKER = "marker"
KIND_SLOT = "slot"
KIND_OBJECT = "object"


@dataclass(frozen=True)
class Entity:
    """One addressable thing on the work surface."""

    id: str
    kind: str  # cube | marker | slot | object
    label: str  # "red cube", "marker 3", "open slot", "pen"
    x: float  # robot frame, mm
    y: float
    # Pixel position in the frame this snapshot was built from. For cubes and
    # objects it is where the detector saw them; for markers and slots it is
    # their calibrated robot position projected back onto the desk plane, which
    # is exact -- both are flat on the table, so there is no height to correct
    # for. None only when there is no calibration to project through.
    pixel: tuple[float, float] | None = None
    # The OBJECT's own robot-frame angle, not a wrist angle: a cube-face edge
    # for a square, the long axis for something elongated. yaw_period_deg says
    # which -- see mt4_vision.wrist.
    yaw_deg: float | None = None
    yaw_period_deg: float = YAW_PERIOD_SQUARE
    # Measured colour, or None when nothing measured one. Cubes get theirs from
    # the HSV detector that found them; registered objects from
    # ``detect.classify_color`` over the same named bands; markers and slots
    # have none and never will.
    #
    # It is in the label too, but the label cannot answer the question that
    # matters. "green" missing from a label means either "this thing is not
    # green" or "nobody looked", and a check that treats those alike refuses
    # correct answers -- which is exactly what happened to "the green statue".
    # None here means *unknown*, and unknown must abstain rather than refuse.
    color: str | None = None
    pickable: bool = False
    placeable: bool = False
    # Why not pickable/placeable, in prose. None when it is.
    reason: str | None = None
    source: str = "hsv"  # hsv | aruco | slot | vlm
    area_px: float | None = None
    extent_mm: tuple[float, float] | None = None  # (long, short)
    # Width the jaws will actually close across at the planned grasp point.
    # Distinct from extent_mm[1]: that is the narrowest the *whole outline*
    # gets, which for a stapler is the rail and for a mug is the handle --
    # neither of which is necessarily where the fingers end up.
    grip_mm: float | None = None
    confidence: float | None = None
    # Cross-references, e.g. an occupied marker naming its occupant.
    holds: str | None = None
    on: str | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "pickable": self.pickable,
            "placeable": self.placeable,
            "source": self.source,
        }
        if self.reason is not None:
            out["reason"] = self.reason
        if self.yaw_deg is not None:
            out["yaw_deg"] = round(self.yaw_deg, 1)
            out["yaw_period_deg"] = self.yaw_period_deg
        if self.pixel is not None:
            out["pixel"] = [round(self.pixel[0], 1), round(self.pixel[1], 1)]
        if self.area_px is not None:
            out["area_px"] = round(self.area_px)
        if self.extent_mm is not None:
            out["extent_mm"] = [
                round(self.extent_mm[0], 1),
                round(self.extent_mm[1], 1),
            ]
        if self.grip_mm is not None:
            out["grip_mm"] = round(self.grip_mm, 1)
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 2)
        if self.holds is not None:
            out["holds"] = self.holds
        if self.on is not None:
            out["on"] = self.on
        return out

    def as_grasp(self, calib: Calibration | None = None) -> Grasp:
        """The motion-layer pose for taking (or releasing at) this entity.

        Located objects open fully then command a full close -- the servo
        stops when the jaws meet resistance, so there is no need to size S
        from a measured width (or refuse when the jaw-span model is missing).
        Cubes keep ``None`` and inherit the calibrated cube ``grip_close_s``.
        """
        open_s: int | None = None
        close_s: int | None = None
        if self.kind == KIND_OBJECT:
            from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN

            open_s = GRIPPER_S_OPEN
            close_s = GRIPPER_S_CLOSED
        return Grasp(
            self.x,
            self.y,
            yaw_deg=self.yaw_deg,
            yaw_period_deg=self.yaw_period_deg,
            grip_open_s=open_s,
            grip_close_s=close_s,
        )


@dataclass(frozen=True)
class Snapshot:
    """One frame's worth of addressable entities."""

    token: str
    entities: list[Entity] = field(default_factory=list)
    summary: str = ""

    def get(self, entity_id: str) -> Entity | None:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None

    def of_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.entities if e.kind == kind]

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot": self.token,
            "summary": self.summary,
            "entities": [e.as_dict() for e in self.entities],
        }

    def table(self) -> list[str]:
        """Human-readable lines, for the CLI and logs."""
        lines = []
        for e in self.entities:
            flags = []
            if e.pickable:
                flags.append("pickable")
            if e.placeable:
                flags.append("placeable")
            tail = f"  [{', '.join(flags)}]" if flags else f"  -- {e.reason or 'n/a'}"
            yaw = "" if e.yaw_deg is None else f" yaw={e.yaw_deg:.0f}"
            lines.append(
                f"  {e.id:<10} {e.label:<14} ({e.x:>6.1f},{e.y:>7.1f}){yaw}{tail}"
            )
        return lines


# -- why a cube is not a pick target --------------------------------------


def pick_block_reason(cube: CubeDetection, scene: Scene) -> str | None:
    """The first gate that stops ``cube`` being picked, in prose, or None.

    Reports; it does not decide. Every threshold and predicate here is the one
    the pick path already uses (``scene.is_phantom_detection`` for the
    area/reach/hull gates, ``Scene.pickable`` for finger clearance), so this
    cannot drift into a second, disagreeing definition of "pickable" -- which
    is exactly what would make the explanations worse than no explanation.
    Ordered cheapest-and-most-fundamental first, so the reason returned is the
    one worth telling a caller about.
    """
    if cube.x is None or cube.y is None:
        return "no robot-frame position (detection predates calibration)"
    x, y = float(cube.x), float(cube.y)
    r = math.hypot(x, y)

    if cube.area < PICK_MIN_AREA:
        return (
            f"blob is {cube.area:.0f}px2, under the {PICK_MIN_AREA:.0f}px2 pick "
            f"floor -- glare or an arm-paint fleck, not a cube"
        )
    if cube.area > PICK_MAX_AREA:
        return (
            f"blob is {cube.area:.0f}px2, over the {PICK_MAX_AREA:.0f}px2 pick "
            f"ceiling -- the arm's own body or a smear, not a cube"
        )
    blocked = _reach_block_reason(x, y, scene.calib)
    if blocked is not None:
        return blocked

    nearest = None
    for other in scene.cubes:
        if other is cube or other.x is None or other.y is None:
            continue
        d = dist_mm(x, y, float(other.x), float(other.y))
        if nearest is None or d < nearest:
            nearest = d
    if nearest is not None and nearest < PICK_CLEARANCE_MM:
        return (
            f"neighbour cube {nearest:.0f}mm away, needs "
            f"{PICK_CLEARANCE_MM:.0f}mm for the fingers to straddle this one"
        )
    return None


def _reach_block_reason(x: float, y: float, calib) -> str | None:
    """Can the arm hold a pose here at all? The reason it cannot, or None.

    ``work_region_block_reason`` is the whole answer whenever there is a
    calibration -- it already runs IK, the soft limits, the keep-out, the reach
    circle, the desk polygon and camera coverage (see CLAUDE.md: one predicate,
    no fifth gate). The two ``elif`` arms below are the no-calibration fallback,
    which is the only case where reach has to be tested directly.

    One helper for all three call sites (cube pick, marker place, object pick),
    so the keep-out message names the firmware everywhere.
    """
    if calib is not None:
        return work_region_block_reason(x, y, calib)
    r = math.hypot(x, y)
    if not is_mp_reachable_xy(x, y):
        return (
            f"r={r:.0f}mm is inside the {KEEPOUT_RADIUS_MM:.0f}mm J1 keep-out "
            f"(firmware mp refuses any target there)"
        )
    if r > MAX_REACH_MM:
        return f"r={r:.0f}mm is beyond the {MAX_REACH_MM:.0f}mm max reach"
    return None


def _slot_admission(sx: float, sy: float, calib) -> dict[str, object]:
    """``placeable``/``reason`` for a free slot, from ONE region evaluation.

    ``in_work_region`` is defined as ``work_region_block_reason(...) is None``,
    so calling both ran the whole predicate twice -- two IK solves, the desk
    polygon and the camera-coverage test, per slot, and there are 15 of them.
    """
    blocked = None if calib is None else work_region_block_reason(sx, sy, calib)
    return {"placeable": blocked is None, "reason": blocked}


def _marker_place_block_reason(
    marker: MarkerSlot, scene: Scene, occupant: str | None
) -> str | None:
    if occupant is not None:
        return f"occupied by {occupant}"
    blocked = _reach_block_reason(marker.x, marker.y, scene.calib)
    if blocked is not None:
        return blocked
    if marker.marker_id not in scene.visible_marker_ids:
        return (
            "ArUco tag did not decode in this frame (arm, shadow, or something "
            "resting on the paper) -- never a place target while unproven"
        )
    return "a cube sits within place clearance of the marker centre"


# -- building a snapshot ---------------------------------------------------


def _spatial_order(cubes: Sequence[CubeDetection]) -> list[CubeDetection]:
    """Deterministic, spatially stable ordering for id assignment.

    Coarse 10mm buckets so a detection wobbling by a millimetre between frames
    does not swap ids with its neighbour, which would make an id read from one
    snapshot mean a different cube in the next.
    """
    return sorted(
        cubes,
        key=lambda c: (
            round(float(c.x or 0.0) / 10.0),
            round(float(c.y or 0.0) / 10.0),
            c.color,
        ),
    )


def _desk_pixel_projector(calib):
    """``(x_mm, y_mm) -> (px, py)`` for things lying flat on the desk, or None.

    Markers and slots had no pixel at all, and that was not a cosmetic gap. Two
    things consume ``Entity.pixel``, and both silently dropped them:

    * ``preview.annotate_for_pointing`` skips any entity without one, so the
      overlay handed to the model circled the cubes and nothing else;
    * ``instruct._entity_lines`` appends "at image point (x, y)" only when there
      is one, so the text list gave coordinates for cubes and nothing else.

    The decision prompt meanwhile asserts "Each is circled in the image with its
    id written beside it" and asks for the target's centre in pixels. For every
    marker and every slot that sentence was false, and the model was being asked
    to point at something it had never been shown. Measured live 2026-08-02 on
    "place it on a non-marker location", four runs: it named ``marker_0`` every
    time -- the first destination in the list -- and pointed at (1255, 402),
    (1261, 403), (1258, 402): the far right edge of the frame, nowhere near any
    destination, the same wrong answer each time.

    Projection is exact here rather than approximate. Both kinds are flat on the
    table, so the desk-plane homography is the whole story and none of the
    height correction that cubes need applies.

    Returns a function yielding None when there is no calibration (unit tests
    build scenes without one) or the point falls outside the frame.
    """
    if calib is None:
        return lambda _x, _y: None
    size = getattr(calib, "frame_size_px", None)

    def project(x: float, y: float) -> tuple[float, float] | None:
        try:
            px, py = calib.robot_to_pixel(x, y)
        except Exception:
            return None
        px, py = float(px), float(py)
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        if size and not (0 <= px < float(size[0]) and 0 <= py < float(size[1])):
            # Off-frame is not a pixel the model can point at, and quoting one
            # would invite it to.
            return None
        return px, py

    return project


def build_snapshot(
    scene: Scene,
    *,
    token: str,
    objects: Sequence["LocatedObject"] | Mapping[str, "LocatedObject"] = (),
) -> Snapshot:
    """Turn a ``Scene`` (plus any located non-cube objects) into addressable ids.

    Cubes come from ``scene.raw_cubes`` -- every robot-mapped detection that
    survived glare rejection, not just the pick-quality subset. A blob the pick
    filters demoted is still a real object the caller can see, and saying so
    with its reason beats omitting it and answering "no red cube in view".
    (Glare is already gone: ``capture_scene`` drops it before anything looks at
    it, because it is not an object at all.)

    ``objects`` are pre-measured by :mod:`mt4_vision.locate`; this only labels
    and numbers them. Pass a **mapping** of id -> object when the caller keeps a
    registry that outlives one snapshot (the MCP server does): a sequence is
    numbered by position, so the moment one object drops out -- which
    ``locate.relocate`` is deliberately strict about -- every id after it shifts
    onto a different physical thing. Ids that move are worse than no ids.
    """
    object_items: list[tuple[str, "LocatedObject"]] = (
        list(objects.items())
        if isinstance(objects, Mapping)
        else [(f"obj_{i}", o) for i, o in enumerate(objects, start=1)]
    )
    objects = [o for _eid, o in object_items]
    pickable_ids = {id(c) for c in scene.pickable(scene.cubes)}
    raw = scene.raw_cubes if scene.raw_cubes is not None else scene.cubes
    if objects:
        # One physical thing must not get two ids. A located object that
        # happens to be cube-coloured is also in the cube list (see
        # is_own_colour_blob), and that duplicate would be advertised as a
        # pickable cube_N -- picking it would grab the object with the wrong
        # (90°-periodic) wrist rule.
        raw = [
            c
            for c in raw
            if not any(is_own_colour_blob(obj, c) for obj in objects)
        ]
    occupant_of: dict[int, CubeDetection] = {
        m.marker_id: c for m, c in scene.occupied
    }
    placeable_marker_ids = {m.marker_id for m in scene.placeable_markers()}

    desk_pixel = _desk_pixel_projector(scene.calib)

    cube_entities: list[Entity] = []
    cube_id_of: dict[int, str] = {}
    for i, cube in enumerate(_spatial_order(raw), start=1):
        eid = f"cube_{i}"
        cube_id_of[id(cube)] = eid
        marker_id = scene.marker_for_cube(cube) if cube.x is not None else None
        reason = pick_block_reason(cube, scene)
        # A cube can clear every gate here yet still be outside the
        # pick-quality set (it was never a pick candidate this frame); trust
        # the set, and only explain when we have something to say.
        pickable = id(cube) in pickable_ids and reason is None
        if not pickable and reason is None:
            # Defensive: pick_block_reason mirrors every gate the pick path
            # applies, so clearing them all should mean membership. If the two
            # ever disagree, trust the pick path and say so plainly rather than
            # advertising a target it would refuse.
            reason = "the pick path did not accept this detection this frame"
        cube_entities.append(
            Entity(
                id=eid,
                kind=KIND_CUBE,
                label=f"{cube.color} cube",
                color=cube.color,
                x=float(cube.x or 0.0),
                y=float(cube.y or 0.0),
                pixel=(float(cube.px), float(cube.py)),
                yaw_deg=cube.yaw_deg,
                yaw_period_deg=YAW_PERIOD_SQUARE,
                pickable=pickable,
                placeable=False,
                reason=reason,
                source="hsv",
                area_px=float(cube.area),
                on=None if marker_id is None else f"marker_{marker_id}",
            )
        )

    free_marker_ids = {m.marker_id for m in scene.free_markers}
    marker_entities: list[Entity] = []
    for marker in sorted(scene.markers, key=lambda m: m.marker_id):
        occ = occupant_of.get(marker.marker_id)
        occ_id = None if occ is None else cube_id_of.get(id(occ))
        placeable = marker.marker_id in placeable_marker_ids
        state = (
            "occupied"
            if occ is not None
            else ("free" if marker.marker_id in free_marker_ids else "unknown")
        )
        marker_entities.append(
            Entity(
                id=f"marker_{marker.marker_id}",
                kind=KIND_MARKER,
                # The printed tag number, so "put it on marker 1" resolves
                # directly -- markers keep their own id rather than an ordinal.
                label=f"marker {marker.marker_id} ({state})",
                x=float(marker.x),
                y=float(marker.y),
                pixel=desk_pixel(float(marker.x), float(marker.y)),
                pickable=False,
                placeable=placeable,
                reason=(
                    None
                    if placeable
                    else _marker_place_block_reason(marker, scene, occ_id)
                ),
                source="aruco",
                holds=occ_id,
            )
        )

    # placeable is computed, not asserted. free_placement_slots gates on the
    # work region as well as keep-out and reach, which makes this
    # belt-and-braces; it is written out anyway because a slot list that lies
    # about placeable is the one failure mode with no symptom at the time.
    # Keep-out and reach alone are not enough: five of the eight fixed slots
    # sit where the detector discards anything put there.
    slot_entities = [
        Entity(
            id=f"slot_{i}",
            kind=KIND_SLOT,
            label="open slot",
            x=float(sx),
            y=float(sy),
            pixel=desk_pixel(float(sx), float(sy)),
            **_slot_admission(float(sx), float(sy), scene.calib),
            source="slot",
        )
        for i, (sx, sy) in enumerate(
            sorted(scene.free_slots, key=lambda s: (round(s[0] / 10.0), round(s[1] / 10.0))),
            start=1,
        )
    ]

    object_entities = [
        object_entity(obj, eid, scene=scene) for eid, obj in object_items
    ]

    entities = cube_entities + object_entities + marker_entities + slot_entities
    extra = f" objects={len(object_entities)}" if object_entities else ""
    return Snapshot(
        token=token,
        entities=entities,
        summary=scene.summary_line() + extra,
    )


def is_own_colour_blob(obj: "LocatedObject", cube: CubeDetection) -> bool:
    """True when ``cube`` is this object's own colour blob, detected again.

    The cube detector is colour+area based, so a coloured non-cube of plausible
    size reads as a cube: measured live 2026-07-30, a red clic eraser at
    (141, -180) came back as `red cube` area 1169px, 20mm from the eraser's own
    grasp point -- inside the finger clearance the object needs, so the eraser
    blocked its own pick. Two detections of one blob share one centroid, so
    identity is settled in pixels (the trick ``scene.is_held_cube_blob`` uses
    for the gripped cube): the cube's centroid falls inside the object's own
    mask footprint. A genuine neighbour sits outside it and still blocks.
    """
    mask_area_px = float(getattr(obj, "mask_area_px", 0.0) or 0.0)
    if mask_area_px <= 0.0:
        # Hand-built or duck-typed object (tests, or a measure() that kept no
        # mask): no footprint to compare against, so every cube is a neighbour.
        return False
    radius_px = math.sqrt(mask_area_px / math.pi)
    return math.hypot(cube.px - obj.px, cube.py - obj.py) <= radius_px


def object_entity(
    obj: "LocatedObject",
    ref: int | str,
    *,
    scene: Scene | None = None,
    require_clearance: bool = True,
) -> Entity:
    """Entity for a non-cube object measured by :mod:`mt4_vision.locate`.

    ``ref`` is either an ordinal (``1`` -> ``obj_1``, for a caller measuring one
    object in one frame) or the entity id itself, for a caller holding a
    registry whose keys must not be re-derived from list position -- see
    ``build_snapshot``.

    Elongated objects are 180°-periodic (the jaws close *across* the long
    axis). Compact / near-square extents use the 90° square period instead --
    grip orientation is not critical when there is no obvious shaft.

    ``require_clearance=False`` keeps every envelope test and drops only the
    neighbour check. ``scene`` must still be passed either way: it carries the
    calibration that ``_reach_block_reason`` and ``plan_object_grasp`` need,
    and without it the reach test degrades to keep-out and max reach alone.
    """
    from mt4_vision.locate import is_compact, plan_object_grasp

    reason: str | None = None
    calib = None if scene is None else scene.calib
    reason = _reach_block_reason(obj.x, obj.y, calib)
    # Where on it can the jaws close, and at what angle? This is the question
    # that decides a non-cube pick, and the outline's short axis is not an
    # answer to it -- that says nothing about the width at the point being
    # gripped. Measured live
    # 2026-08-02: a stapler picked cleanly when segmentation happened to return
    # only its 16mm rail, and closed on air when it returned the whole 73mm
    # body. Same object, same arm, no deliberate choice either time.
    plan = None
    if reason is None and calib is not None:
        plan, why = plan_object_grasp(obj, calib)
        if plan is None:
            reason = why
    # Clearance is judged at the planned grasp point, not the object's centre:
    # the fingers go where the plan says, and for anything elongated those can
    # be tens of millimetres apart.
    gx = obj.x if plan is None else plan.x
    gy = obj.y if plan is None else plan.y
    gw = obj.short_mm if plan is None else plan.width_mm
    # Only cubes are counted, because ``scene.cubes`` is the only list of other
    # things there is -- a pen or a key beside the grasp point is invisible
    # here and blocks nothing. That asymmetry is why the check is optional.
    if reason is None and scene is not None and require_clearance:
        need = max(gw, 20.0) * 0.5 + 12.0
        for other in scene.cubes:
            if other.x is None or other.y is None:
                continue
            if is_own_colour_blob(obj, other):
                continue
            d = dist_mm(gx, gy, float(other.x), float(other.y))
            if d < need:
                reason = (
                    f"a cube sits {d:.0f}mm from the grasp point, inside the "
                    f"{need:.0f}mm the jaws need across a {gw:.0f}mm grip"
                )
                break
    if reason is None and calib is not None and plan is not None:
        # The planned point can sit well off the centroid, so it needs the same
        # region test the centroid got -- an in-region object with an
        # out-of-region grasp is not pickable.
        moved = work_region_block_reason(gx, gy, calib)
        if moved is not None:
            reason = f"the best grip on it is at ({gx:.0f}, {gy:.0f}), where {moved}"
    period = (
        YAW_PERIOD_SQUARE
        if is_compact(obj.long_mm, obj.short_mm)
        else YAW_PERIOD_LONG_AXIS
    )
    # Measured colour first, then the noun -- the same shape a cube's label has
    # ("green cube"), and for the same reason: the first word is something the
    # detector established and the rest is what the thing is called. The label
    # is for transcripts and for `held`; nothing matches on it.
    label = f"{obj.color} {obj.label}" if obj.color else obj.label
    return Entity(
        id=ref if isinstance(ref, str) else f"obj_{ref}",
        kind=KIND_OBJECT,
        label=label,
        color=obj.color,
        # The grasp point, not the silhouette centroid. This is where the arm
        # goes, so it is what the entity must report -- otherwise the plan and
        # the motion disagree and only the arm finds out.
        x=float(gx),
        y=float(gy),
        pixel=(float(obj.px), float(obj.py)),
        yaw_deg=float(obj.axis_yaw_deg if plan is None else plan.yaw_deg),
        yaw_period_deg=period,
        pickable=reason is None,
        placeable=False,
        reason=reason,
        source="vlm",
        extent_mm=(float(obj.long_mm), float(obj.short_mm)),
        confidence=float(obj.confidence),
        grip_mm=None if plan is None else float(plan.width_mm),
    )
