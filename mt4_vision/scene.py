"""Detection-as-state desk snapshot for the shuffle planner.

Each clear camera frame builds a fresh ``Scene`` from cube detections and
ArUco visibility. No persistent tracks -- vacated poses cannot linger.

Occupancy and place-clearance use every robot-mapped detection. Pick
candidates are a stricter subset (blob size, plus ``workspace.in_work_region``)
so arm paint is not grasped -- but a filtered blob still blocks placing on a
nearby marker.

No camera-park exclusion here: that guarded against the arm's own
silhouette being misread near its old capture-retreat pose, but the live
loop (shuffle.py) never retreats there between captures -- it was quietly
vetoing real desk locations (e.g. marker 4, ~42mm from that pose) that
have nothing to do with the arm's position during capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from mt4_vision.calib import Calibration
from mt4_vision.detect import (
    CubeDetection,
    detect_cubes,
    detect_markers,
    last_dropped_off_table,
)
from mt4_vision.workspace import (
    MARKER_DICT,
    MARKER_OCCUPY_RADIUS_MM,
    MAX_REACH_MM,
    PICK_CLEARANCE_MM,
    MarkerSlot,
    WorkspaceState,
    cubes_with_robot_coords,
    dist_mm,
    in_work_region,
    is_mp_reachable_xy,
    marker_slots_from_calibration,
    rebuild_workspace_state,
    work_region_block_reason,
)

# Real cube blobs under the closer overhead mount land ~2000-4000px^2
# (on-pad red measured 2790 then 3627 on 2026-07-20 as pose/lighting
# varied). Tighter than detect.py's floor so low-area glare/arm flecks
# are not pick targets while still counting toward marker occupancy via
# the raw detection list. Old far-mount pick band was 280-650.
PICK_MIN_AREA = 400.0
PICK_MAX_AREA = 5000.0
# A cube gripped at the capture pose hovers ~210mm over the table and
# registers as a normal-looking detection near a predictable pixel; from
# the raw list it leaks into marker occupancy / free-slot clearance and
# vetoes real park spots. Area does NOT separate it from table cubes:
# parallax scaling predicts ~2x (700-1070px^2) but measured 2026-07-19 a
# held red read 622px^2 -- the gripper fingers occlude part of the top
# face. Identity is color (the caller knows what it grips) + proximity to
# the parallax-predicted pixel (measured 54px off prediction; the radius
# covers that systematic error plus a few-mm grip offset at ~0.6mm/px).
HELD_CUBE_RADIUS_PX = 90.0
# Minimum saturation (0-255) at the top-face centroid for a blob to be a
# physical object at all. A laminated ArUco pad reflecting the room lights
# throws a specular highlight whose faintly-tinted rim clips the blue band
# (H 90-128) and survives every geometric gate -- the pad is on the desk,
# reachable, and the blob is cube-sized and square. Measured
# over 377 recorded episodes: 28 picks were sent at such a highlight, all
# blue, all ending grasp_failed or lost. Verified-real cubes bottom out at
# S=211 (blue), 201 (red), 100 (green); on-marker glare runs a median of 37.
# 90 leaves green a 10-point margin and rejects nothing that was ever
# successfully picked. See also is_on_visible_marker, which catches the
# minority of highlights bright enough to clear this floor.
MIN_CUBE_SATURATION = 90.0


def is_on_visible_marker(
    cube: CubeDetection, marker_quads: list[list[list[float]]]
) -> bool:
    """True when the blob's centroid sits inside a *detected* marker outline.

    A cube resting on a marker occludes its pattern, so the marker is not
    decodable -- that is exactly what ``unknown_markers`` means elsewhere in
    this module. The contrapositive is the useful direction: if the marker
    decoded this frame, nothing is standing on it, so any cube-coloured blob
    inside its outline is a reflection off the laminate. Measured across 360
    frames, red and green never once landed inside a visible marker quad
    while blue did 67 times, every one of them glare.
    """
    for quad in marker_quads:
        pts = np.asarray(quad, dtype=np.float32)
        if cv2.pointPolygonTest(pts, (float(cube.px), float(cube.py)), True) >= 0:
            return True
    return False


def is_glare_blob(
    cube: CubeDetection, marker_quads: list[list[list[float]]]
) -> bool:
    """True when a detection is a specular highlight rather than an object.

    Unlike ``is_phantom_detection`` (which only demotes a blob from being a
    *pick* target while still letting it occupy markers and slots), glare is
    not there at all -- so callers drop it from the raw list entirely.
    """
    return cube.sat < MIN_CUBE_SATURATION or is_on_visible_marker(cube, marker_quads)


def is_held_cube_blob(
    cube: CubeDetection,
    held_px: tuple[float, float],
    held_color: str | None = None,
) -> bool:
    """True when a raw detection is the gripped cube itself (held color at
    its predicted capture-pose pixel), not a cube on the desk."""
    if held_color is not None and cube.color != held_color:
        return False
    return (
        math.hypot(cube.px - held_px[0], cube.py - held_px[1])
        <= HELD_CUBE_RADIUS_PX
    )


@dataclass(frozen=True)
class Scene:
    """One clear-frame world: only what the camera sees right now."""

    cubes: list[CubeDetection]
    markers: list[MarkerSlot]
    occupied: list[tuple[MarkerSlot, CubeDetection]]
    free_markers: list[MarkerSlot]
    unknown_markers: list[MarkerSlot]
    free_slots: list[tuple[float, float]]
    visible_marker_ids: set[int]
    # Raw detections (occupancy source); cubes is the pick-quality subset.
    raw_cubes: list[CubeDetection] | None = None
    # The calibration this snapshot was built from. Carried because the work
    # region is defined by the desk polygon and the camera geometry, both of
    # which live on the calibration -- a Scene that could not answer "may the
    # arm work here?" would force every consumer to thread a second argument.
    calib: Calibration | None = None
    # Blobs detect_cubes discarded as not-on-the-desk for this frame.
    off_table_blobs: int = 0

    @classmethod
    def from_workspace(
        cls,
        state: WorkspaceState,
        *,
        pick_cubes: list[CubeDetection] | None = None,
        raw_cubes: list[CubeDetection] | None = None,
        calib: Calibration | None = None,
        off_table_blobs: int = 0,
    ) -> Scene:
        visible = state.visible_marker_ids or set()
        cubes = list(pick_cubes) if pick_cubes is not None else list(state.cubes)
        return cls(
            cubes=cubes,
            markers=list(state.markers),
            occupied=list(state.occupied),
            free_markers=list(state.free_markers),
            unknown_markers=list(state.unknown_markers),
            free_slots=list(state.free_slots),
            visible_marker_ids=set(visible),
            raw_cubes=list(raw_cubes) if raw_cubes is not None else None,
            calib=calib,
            off_table_blobs=int(off_table_blobs),
        )

    def summary_line(self) -> str:
        raw_n = len(self.raw_cubes) if self.raw_cubes is not None else len(self.cubes)
        dropped = max(0, raw_n - len(self.cubes))
        extra = f" phantoms_dropped={dropped}" if dropped else ""
        # Say so when the detector threw blobs away entirely. The gate this
        # replaced was silent, which is how three cubes sat on the desk
        # missing from the snapshot with nothing anywhere saying so.
        if self.off_table_blobs:
            extra += f" off_table_blobs={self.off_table_blobs}"
        return (
            f"cubes={len(self.cubes)} blockers={len(self.blockers())} "
            f"free_markers={len(self.placeable_markers())} "
            f"occupied={len(self.occupied)} "
            f"unknown={len(self.unknown_markers)} "
            f"free_slots={len(self.free_slots)}{extra}"
        )

    def cube_lines(self) -> list[str]:
        lines = []
        for c in sorted(self.cubes, key=lambda x: (-(x.area or 0), x.color)):
            on = self.marker_for_cube(c)
            tag = f" marker {on}" if on is not None else " open"
            lines.append(
                f"  {c.color} ({c.x:.0f},{c.y:.0f}) area={c.area:.0f}{tag}"
            )
        return lines

    def marker_for_cube(self, cube: CubeDetection) -> int | None:
        best_id: int | None = None
        best_d = MARKER_OCCUPY_RADIUS_MM
        for m in self.markers:
            d = dist_mm(float(cube.x), float(cube.y), m.x, m.y)
            if d < best_d:
                best_d = d
                best_id = m.marker_id
        return best_id

    def blockers(self) -> list[CubeDetection]:
        """Pick-quality cubes not sitting on any calibrated marker."""
        return [c for c in self.cubes if self.marker_for_cube(c) is None]

    def placeable_markers(self) -> list[MarkerSlot]:
        """Tag visible, place-clearance free, and inside the work region.

        One predicate with the pick path and the slot list, so a marker can
        never be offered as a place target somewhere a cube resting there
        could not be picked back up.
        """
        if self.calib is None:
            return [
                m
                for m in self.free_markers
                if is_mp_reachable_xy(m.x, m.y)
                and math.hypot(m.x, m.y) <= MAX_REACH_MM
            ]
        return [
            m for m in self.free_markers if in_work_region(m.x, m.y, self.calib)
        ]

    def occupied_pick_cubes(self) -> list[CubeDetection]:
        """Occupied-marker cubes that are also pick-quality (in Scene.cubes)."""
        pick_ids = {id(c) for c in self.cubes}
        return [c for _m, c in self.occupied if id(c) in pick_ids]

    def pickable(self, cubes: list[CubeDetection]) -> list[CubeDetection]:
        """Work-region cubes with finger clearance from every other pick cube."""
        out: list[CubeDetection] = []
        for c in cubes:
            if self.calib is not None:
                if not in_work_region(float(c.x), float(c.y), self.calib):
                    continue
            elif not is_mp_reachable_xy(float(c.x), float(c.y)) or math.hypot(
                float(c.x), float(c.y)
            ) > MAX_REACH_MM:
                continue
            if any(
                dist_mm(float(c.x), float(c.y), float(o.x), float(o.y))
                < PICK_CLEARANCE_MM
                for o in self.cubes
                if o is not c
            ):
                continue
            out.append(c)
        cx = sum(m.x for m in self.markers) / max(len(self.markers), 1)
        cy = sum(m.y for m in self.markers) / max(len(self.markers), 1)
        out.sort(
            key=lambda c: (
                abs(c.area - 2500.0),  # closer mount: real tops ~2–4k px²
                dist_mm(float(c.x), float(c.y), cx, cy),
            )
        )
        return out


def is_phantom_detection(cube: CubeDetection, calib: Calibration) -> bool:
    """True when a blob should not be treated as a pick target.

    Two independent reasons, and the split matters. A blob can be the wrong
    SIZE to be a cube (glare fleck, the arm's body, a smear) -- that is a
    statement about the blob. Or it can be somewhere the arm may not work --
    that is ``workspace.in_work_region``, and it is the same predicate a place
    target is held to, so a cube can never be pickable somewhere a place is
    refused or the other way round.

    This used to take ``markers`` and test the convex hull of their centres.
    See the note above ``workspace.in_work_region`` for what that cost.
    """
    if cube.x is None or cube.y is None:
        return True
    if cube.area < PICK_MIN_AREA or cube.area > PICK_MAX_AREA:
        return True
    return not in_work_region(float(cube.x), float(cube.y), calib)


def filter_phantoms(
    cubes: list[CubeDetection], calib: Calibration
) -> list[CubeDetection]:
    return [c for c in cubes if not is_phantom_detection(c, calib)]


def capture_scene(
    calib: Calibration,
    frame: np.ndarray,
    *,
    held_cube_px: tuple[float, float] | None = None,
    held_color: str | None = None,
) -> Scene:
    """Build a scene from one frame.

    Occupancy / free-marker / free-slot clearance uses *every* robot-mapped
    detection (raw). Phantom filtering only removes pick candidates.

    ``held_cube_px`` / ``held_color``: predicted pixel and color of a cube
    currently in the gripper (captures taken while holding). The matching
    detection is dropped from the raw list so the held cube cannot occupy
    markers or slots.

    Specular glare off the laminated marker pads is dropped from the raw
    list too, for the same reason and before anything else looks at it: it
    is not an object, so it must neither be picked nor count as occupancy.
    """
    markers = marker_slots_from_calibration(calib)
    raw = cubes_with_robot_coords(detect_cubes(frame, calib))
    off_table = last_dropped_off_table()
    marker_dets = detect_markers(frame, MARKER_DICT)
    quads = [m.corners for m in marker_dets if m.corners]
    raw = [c for c in raw if not is_glare_blob(c, quads)]
    if held_cube_px is not None:
        raw = [
            c for c in raw
            if not is_held_cube_blob(c, held_cube_px, held_color)
        ]
    visible = {m.marker_id for m in marker_dets}
    state = rebuild_workspace_state(
        calib, markers, raw, visible_marker_ids=visible
    )
    pick_cubes = filter_phantoms(raw, calib)
    return Scene.from_workspace(
        state,
        pick_cubes=pick_cubes,
        raw_cubes=raw,
        calib=calib,
        off_table_blobs=off_table,
    )


# Post-move verification -- implementation lives on motion so stack's
# pick_missed and shuffle's post-move check share one helper. Re-exported
# here so existing ``from mt4_vision.scene import verify_pick_place`` keeps
# working.
from mt4_vision.motion import (
    VERIFY_ORIGIN_RADIUS_MM,
    VERIFY_PLACED_RADIUS_MM,
    verify_pick_place as _verify_detections,
)


def verify_pick_place(
    scene: Scene,
    *,
    pick_x: float,
    pick_y: float,
    pick_color: str,
    place_x: float,
    place_y: float,
) -> str:
    """Judge an atomic pick+place from the post-move scene.

    Returns ``placed``, ``grasp_failed``, or ``lost``.
    """
    detections = [
        (c.color, float(c.x), float(c.y))
        for c in scene.cubes
        if c.x is not None and c.y is not None
    ]
    return _verify_detections(
        detections,
        pick_x=pick_x,
        pick_y=pick_y,
        pick_color=pick_color,
        place_x=place_x,
        place_y=place_y,
    )
