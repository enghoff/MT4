"""Addressable scene entities: stable ids, and why a thing can't be picked.

The layer a language model actually talks to. A ``Scene``
(:mod:`mt4_vision.scene`) is detections and marker states; a ``Snapshot`` turns
that into a flat list of things with **ids**, so a caller can say "pick
``cube_2``" instead of "pick the red one" and have exactly one referent.

Two defects this replaces:

* ``mt4_pick_cube(color)`` resolved ambiguity silently via ``pick_largest_cube``
  -- no way to name *which* red cube, and no way for the request to fail.
* A blob the pick filters rejected simply wasn't reported, so the honest answer
  ("that is the arm's own paint, 78mm outside the marker hull") came back as
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
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import cv2

from mt4_vision.calib import Calibration
from mt4_vision.detect import CubeDetection
from mt4_vision.motion import (
    YAW_PERIOD_LONG_AXIS,
    YAW_PERIOD_SQUARE,
    Grasp,
)
from mt4_vision.scene import (
    HULL_OUTSIDE_MARGIN_MM,
    PICK_MAX_AREA,
    PICK_MIN_AREA,
    Scene,
    _marker_hull_robot,
    within_pick_hull,
)
from mt4_vision.workspace import (
    KEEPOUT_RADIUS_MM,
    MAX_REACH_MM,
    PICK_CLEARANCE_MM,
    MarkerSlot,
    dist_mm,
    is_mp_reachable_xy,
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
    # Pixel position in the frame this snapshot was built from; None for slots,
    # which are nominal coordinates rather than detections.
    pixel: tuple[float, float] | None = None
    # The OBJECT's own robot-frame angle, not a wrist angle: a cube-face edge
    # for a square, the long axis for something elongated. yaw_period_deg says
    # which -- see mt4_vision.wrist.
    yaw_deg: float | None = None
    yaw_period_deg: float = YAW_PERIOD_SQUARE
    pickable: bool = False
    placeable: bool = False
    # Why not pickable/placeable, in prose. None when it is.
    reason: str | None = None
    source: str = "hsv"  # hsv | aruco | slot | vlm
    area_px: float | None = None
    extent_mm: tuple[float, float] | None = None  # (long, short)
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
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 2)
        if self.holds is not None:
            out["holds"] = self.holds
        if self.on is not None:
            out["on"] = self.on
        return out

    def as_grasp(self, calib: Calibration | None = None) -> Grasp:
        """The motion-layer pose for taking (or releasing at) this entity.

        With a ``calib`` and a measured short axis, the gripper close value is
        sized to the object instead of inheriting the 20mm-cube default -- see
        ``calib.grip_s_for_span_mm``. Without that, closing on a ~10mm pen at
        the cube's value never contacts it, and with no grip-retention sensing
        the pick reports success having done nothing.
        """
        close_s: int | None = None
        if calib is not None and self.extent_mm is not None:
            from mt4_vision.calib import GRIP_SQUEEZE_MM, grip_s_for_span_mm

            # Close past the measured width, not to it. The measurement is a
            # silhouette from a steeply oblique camera, so it reads wide for
            # anything with height -- an error in the one direction that grips
            # nothing at all. Erring closed instead just grips harder.
            close_s = grip_s_for_span_mm(
                calib, max(2.0, self.extent_mm[1] - GRIP_SQUEEZE_MM)
            )
        return Grasp(
            self.x,
            self.y,
            yaw_deg=self.yaw_deg,
            yaw_period_deg=self.yaw_period_deg,
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
    if not is_mp_reachable_xy(x, y):
        return (
            f"r={r:.0f}mm is inside the {KEEPOUT_RADIUS_MM:.0f}mm J1 keep-out "
            f"(firmware mp refuses any target there)"
        )
    if r > MAX_REACH_MM:
        return f"r={r:.0f}mm is beyond the {MAX_REACH_MM:.0f}mm max reach"
    if not within_pick_hull(x, y, scene.markers):
        hull = _marker_hull_robot(scene.markers)
        outside = 0.0
        if hull is not None:
            outside = -float(cv2.pointPolygonTest(hull, (x, y), True))
        return (
            f"{outside:.0f}mm outside the marker hull (allowance "
            f"{HULL_OUTSIDE_MARGIN_MM:.0f}mm) -- arm paint or off-desk clutter"
        )

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


def _marker_place_block_reason(
    marker: MarkerSlot, scene: Scene, occupant: str | None
) -> str | None:
    if occupant is not None:
        return f"occupied by {occupant}"
    r = math.hypot(marker.x, marker.y)
    if not is_mp_reachable_xy(marker.x, marker.y):
        return f"r={r:.0f}mm is inside the {KEEPOUT_RADIUS_MM:.0f}mm J1 keep-out"
    if r > MAX_REACH_MM:
        return f"r={r:.0f}mm is beyond the {MAX_REACH_MM:.0f}mm max reach"
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


def build_snapshot(
    scene: Scene,
    *,
    token: str,
    objects: Sequence["LocatedObject"] = (),
) -> Snapshot:
    """Turn a ``Scene`` (plus any located non-cube objects) into addressable ids.

    Cubes come from ``scene.raw_cubes`` -- every robot-mapped detection that
    survived glare rejection, not just the pick-quality subset. A blob the pick
    filters demoted is still a real object the caller can see, and saying so
    with its reason beats omitting it and answering "no red cube in view".
    (Glare is already gone: ``capture_scene`` drops it before anything looks at
    it, because it is not an object at all.)

    ``objects`` are pre-measured by :mod:`mt4_vision.locate`; this only labels
    and numbers them.
    """
    pickable_ids = {id(c) for c in scene.pickable(scene.cubes)}
    raw = scene.raw_cubes if scene.raw_cubes is not None else scene.cubes
    occupant_of: dict[int, CubeDetection] = {
        m.marker_id: c for m, c in scene.occupied
    }
    placeable_marker_ids = {m.marker_id for m in scene.placeable_markers()}

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

    slot_entities = [
        Entity(
            id=f"slot_{i}",
            kind=KIND_SLOT,
            label="open slot",
            x=float(sx),
            y=float(sy),
            placeable=True,
            source="slot",
        )
        for i, (sx, sy) in enumerate(
            sorted(scene.free_slots, key=lambda s: (round(s[0] / 10.0), round(s[1] / 10.0))),
            start=1,
        )
    ]

    object_entities = [
        object_entity(obj, i, scene=scene)
        for i, obj in enumerate(objects, start=1)
    ]

    entities = cube_entities + object_entities + marker_entities + slot_entities
    extra = f" objects={len(object_entities)}" if object_entities else ""
    return Snapshot(
        token=token,
        entities=entities,
        summary=scene.summary_line() + extra,
    )


def object_entity(
    obj: "LocatedObject", index: int, *, scene: Scene | None = None
) -> Entity:
    """Entity for a non-cube object measured by :mod:`mt4_vision.locate`.

    Elongated objects are 180°-periodic (the jaws close *across* the long
    axis), which is the whole reason ``yaw_period_deg`` exists.

    Clearance is checked against the object's own SHORT axis, not the cube
    rule: a 138mm pen has neighbours inside ``PICK_CLEARANCE_MM`` constantly,
    and applying a 20mm cube's finger allowance to it would refuse nearly every
    real grasp. What matters is room beside the grasp point across the jaws.
    """
    reason: str | None = None
    r = math.hypot(obj.x, obj.y)
    if not is_mp_reachable_xy(obj.x, obj.y):
        reason = f"r={r:.0f}mm is inside the {KEEPOUT_RADIUS_MM:.0f}mm J1 keep-out"
    elif r > MAX_REACH_MM:
        reason = f"r={r:.0f}mm is beyond the {MAX_REACH_MM:.0f}mm max reach"
    elif scene is not None and not within_pick_hull(obj.x, obj.y, scene.markers):
        reason = "outside the marker hull -- off-desk clutter or a bad hint"
    elif scene is not None:
        need = max(obj.short_mm, 20.0) * 0.5 + 12.0
        for other in scene.cubes:
            if other.x is None or other.y is None:
                continue
            d = dist_mm(obj.x, obj.y, float(other.x), float(other.y))
            if d < need:
                reason = (
                    f"a cube sits {d:.0f}mm from the grasp point, inside the "
                    f"{need:.0f}mm the jaws need across a {obj.short_mm:.0f}mm object"
                )
                break
    return Entity(
        id=f"obj_{index}",
        kind=KIND_OBJECT,
        label=obj.label,
        x=float(obj.x),
        y=float(obj.y),
        pixel=(float(obj.px), float(obj.py)),
        yaw_deg=float(obj.axis_yaw_deg),
        yaw_period_deg=YAW_PERIOD_LONG_AXIS,
        pickable=reason is None,
        placeable=False,
        reason=reason,
        source="vlm",
        extent_mm=(float(obj.long_mm), float(obj.short_mm)),
        confidence=float(obj.confidence),
    )
