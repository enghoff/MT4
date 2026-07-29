"""Work-surface model: calibrated markers, cube detections, occupancy, slots."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from mt4_jog.joints import (
    GROUND_Z_MM,
    J2_J3_SUM_MAX_STEPS,
    J2_J3_SUM_MIN_STEPS,
    JOINT_SOFT_MAX_STEPS,
    JOINT_SOFT_MIN_STEPS,
)
from mt4_vision.calib import Calibration
from mt4_vision.detect import CubeDetection, detect_cubes

# Cube centroid within this of a marker center counts the marker occupied
# when the ArUco tag does not decode. A decoded tag alone is *not*
# enough to call the marker free -- see PLACE_CLEARANCE_MM below.
# With the cube-top calibration fitted, on-marker cubes read 5-15mm from
# center while beside-the-paper cubes read 20mm+; 40mm classified adjacent
# cubes as occupants. Measured live 2026-07-14: a cube resting on the tag
# read 23mm from center, 1mm outside the old 22mm radius -- missed
# "occupied" and (with the tag covered) landed in unknown instead, where
# the planner can neither place onto it nor pick it off.
MARKER_OCCUPY_RADIUS_MM = 26.0
# Min distance from any other cube for a *placement destination*: the
# fingers sweep outward when releasing, so they need more room than the
# cube footprint itself.
PLACE_CLEARANCE_MM = 45.0
# Open-table slots keep this far from marker papers (the printed sheet is
# wider than the occupancy radius).
MARKER_PAPER_CLEARANCE_MM = 40.0
# Min distance from any other cube for a *pick* target: the opened fingers
# straddle the cube, so a close neighbor gets clipped. Observed live: picks
# beside a ~35mm neighbor nudged both cubes instead of gripping.
PICK_CLEARANCE_MM = 45.0
# ArUco dictionary of the desk markers (same as calibrate_vision.py).
MARKER_DICT = "4x4_50"
# Measured operating envelope (envelope_samples.json, 2026-07-19): in-range
# max reach 352.1mm, out at 353.6mm. 350mm keeps a thin margin from the
# singularity edge while covering the measured workspace (marker 1 ~322mm).
MAX_REACH_MM = 350.0
# Firmware `mp` rejects TCP targets inside this cylinder (J1 axis, any Z).
KEEPOUT_RADIUS_MM = 140.0
KEEPOUT_TARGET_MARGIN_MM = 0.5  # mirrors start_absolute_move in motion.cpp

# Furthest radius the camera can still confirm a cube at. The arm reaches far
# beyond this; the camera does not, and anything put down past it is lost to
# vision until moved by hand. Two things go wrong at once out there:
#
#   1. The placement cannot be verified, so a perfect demonstration gets
#      stamped success=False and is dropped by the converter.
#   2. The cube is now invisible to every planner, which cannot pick it
#      again. Measured twice: four consecutive far-slot placements removed
#      4 of 7 cubes from the table and had to be retrieved open-loop, and
#      unstack scattering to r=300 drifted 5 of 9 cubes out of the
#      workspace over one long session.
#
# 240mm sits just above the furthest reliably-detected ArUco marker
# (r = 236mm) and leaves the near placement slots (r = 180-209mm) usable.
# This is a limit of the camera's coverage, not of the arm -- do not raise it
# to "reach more of the desk" without re-measuring detection out there.
MAX_VERIFIABLE_RADIUS_MM = 240.0

# Open-table placement candidates (robot frame, mm). Shared with
# calibrate_height.py probe grid.
PLACEMENT_SLOTS: list[tuple[float, float]] = [
    (200.0, -60.0),
    (200.0, 60.0),
    (150.0, 100.0),
    (240.0, -150.0),
    (240.0, 150.0),
    (150.0, -250.0),
    (150.0, 250.0),
    (280.0, 0.0),
]


@dataclass(frozen=True)
class MarkerSlot:
    marker_id: int
    x: float
    y: float


@dataclass
class WorkspaceState:
    cubes: list[CubeDetection]
    markers: list[MarkerSlot]
    occupied: list[tuple[MarkerSlot, CubeDetection]]
    # Placeable: tag decoded in the frame (provably empty) and clear of cubes.
    free_markers: list[MarkerSlot]
    # Neither provably empty nor occupied -- tag hidden (arm, shadow, a cube
    # on the paper's edge) with no cube inside the occupancy radius. Never a
    # placement target.
    unknown_markers: list[MarkerSlot]
    free_slots: list[tuple[float, float]]
    # Marker ids whose ArUco tag decoded in the source frame; None when the
    # state was built without decode information (legacy/test path).
    visible_marker_ids: set[int] | None = None


def dist_mm(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def is_mp_reachable_xy(x: float, y: float) -> bool:
    """True when firmware ``mp`` will accept (x, y) as a horizontal target."""
    return math.hypot(x, y) >= KEEPOUT_RADIUS_MM - KEEPOUT_TARGET_MARGIN_MM


def is_within_envelope(
    x: float,
    y: float,
    z: float,
    *,
    ground_z: float = GROUND_Z_MM,
    max_reach: float = MAX_REACH_MM,
) -> bool:
    """True when (x,y,z) clears keep-out, ground plane, and max reach."""
    r = math.hypot(x, y)
    if r < KEEPOUT_RADIUS_MM - KEEPOUT_TARGET_MARGIN_MM:
        return False
    if r > max_reach:
        return False
    if z < ground_z - 0.05:
        return False
    return True


def joints_within_soft_limits(
    steps: tuple[int, int, int, int] | list[int],
    *,
    lo: tuple[int, int, int, int] = JOINT_SOFT_MIN_STEPS,
    hi: tuple[int, int, int, int] = JOINT_SOFT_MAX_STEPS,
    sum23_lo: int = J2_J3_SUM_MIN_STEPS,
    sum23_hi: int = J2_J3_SUM_MAX_STEPS,
) -> bool:
    """True when joint step counters sit inside the soft envelope."""
    if len(steps) != 4:
        return False
    if not all(lo[i] <= int(steps[i]) <= hi[i] for i in range(4)):
        return False
    sum23 = int(steps[1]) + int(steps[2])
    return sum23_lo <= sum23 <= sum23_hi


def marker_slots_from_calibration(calib: Calibration) -> list[MarkerSlot]:
    obs = calib.raw_marker_observations
    if not obs:
        return []
    slots: list[MarkerSlot] = []
    for key, data in obs.items():
        rx, ry = data["robot"]
        slots.append(MarkerSlot(int(key), float(rx), float(ry)))
    return sorted(slots, key=lambda m: m.marker_id)


def cubes_with_robot_coords(cubes: list[CubeDetection]) -> list[CubeDetection]:
    return [c for c in cubes if c.x is not None and c.y is not None]


def partition_cubes_on_markers(
    cubes: list[CubeDetection], markers: list[MarkerSlot]
) -> tuple[list[tuple[MarkerSlot, CubeDetection]], list[CubeDetection]]:
    """Return (occupied marker pairs, cubes not on any marker).

    Globally greedy nearest-pair matching: when two cubes contend for one
    marker, the loser can still claim its own second-nearest marker. The old
    per-cube nearest-only rule dropped the loser entirely, leaving a
    physically occupied marker "free" -- an invitation to stack.
    """
    pairs: list[tuple[float, int, MarkerSlot]] = []
    for index, cube in enumerate(cubes):
        for marker in markers:
            d = dist_mm(cube.x, cube.y, marker.x, marker.y)
            if d < MARKER_OCCUPY_RADIUS_MM:
                pairs.append((d, index, marker))
    pairs.sort(key=lambda p: p[0])

    on_marker: dict[int, CubeDetection] = {}
    assigned: set[int] = set()
    for _d, index, marker in pairs:
        if index in assigned or marker.marker_id in on_marker:
            continue
        on_marker[marker.marker_id] = cubes[index]
        assigned.add(index)

    occupied = [
        (m, on_marker[m.marker_id])
        for m in markers
        if m.marker_id in on_marker
    ]
    off_marker = [c for i, c in enumerate(cubes) if i not in assigned]
    return occupied, off_marker


def free_placement_slots(
    calib: Calibration,
    markers: list[MarkerSlot],
    cubes: list[CubeDetection],
    *,
    slots: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    candidates = slots if slots is not None else PLACEMENT_SLOTS
    free: list[tuple[float, float]] = []
    for sx, sy in candidates:
        if math.hypot(sx, sy) > MAX_REACH_MM:
            continue
        if not is_mp_reachable_xy(sx, sy):
            continue
        if any(dist_mm(sx, sy, m.x, m.y) < MARKER_PAPER_CLEARANCE_MM for m in markers):
            continue
        if any(dist_mm(sx, sy, c.x, c.y) < PLACE_CLEARANCE_MM for c in cubes):
            continue
        free.append((sx, sy))
    return free


def analyze_workspace(
    calib: Calibration,
    frame: np.ndarray,
) -> WorkspaceState:
    from mt4_vision.detect import detect_markers

    markers = marker_slots_from_calibration(calib)
    cubes = cubes_with_robot_coords(detect_cubes(frame, calib))
    visible = {m.marker_id for m in detect_markers(frame, MARKER_DICT)}
    return rebuild_workspace_state(calib, markers, cubes, visible_marker_ids=visible)


def rebuild_workspace_state(
    calib: Calibration,
    markers: list[MarkerSlot],
    cubes: list[CubeDetection],
    visible_marker_ids: set[int] | None = None,
) -> WorkspaceState:
    """Classify markers from detections plus tag visibility.

    Free requires *both* a decoded ArUco tag *and* no cube within
    PLACE_CLEARANCE_MM of the marker center. A visible tag alone is not
    enough -- a cube can sit beside or partly on the paper while the tag
    still decodes, and placing onto that marker would stack.

    Occupied: a cube sits inside MARKER_OCCUPY_RADIUS_MM (tag visibility
    does not override this -- we still check for blocks in the marker area).
    Unknown: neither free nor occupied (tag hidden with no nearby cube, or
    tag visible but a cube still inside place clearance). Never a place target.

    With visible_marker_ids=None (no frame available), falls back to
    proximity-only classification, still requiring place clearance for free.
    """
    if visible_marker_ids is None:
        occupied, _off = partition_cubes_on_markers(cubes, markers)
        occupied_ids = {m.marker_id for m, _ in occupied}
        free_markers = [
            m
            for m in markers
            if m.marker_id not in occupied_ids
            and all(dist_mm(m.x, m.y, c.x, c.y) >= PLACE_CLEARANCE_MM for c in cubes)
        ]
        free_ids = {m.marker_id for m in free_markers}
        unknown_markers = [
            m
            for m in markers
            if m.marker_id not in occupied_ids and m.marker_id not in free_ids
        ]
    else:
        # Cubes inside occupy radius claim their marker even if the tag still
        # decodes (partial occlusion / noisy decode). Visible tags without a
        # nearby cube still need place-clearance before counting as free.
        occupied, _off = partition_cubes_on_markers(cubes, markers)
        occupied_ids = {m.marker_id for m, _ in occupied}
        free_markers = [
            m
            for m in markers
            if m.marker_id in visible_marker_ids
            and m.marker_id not in occupied_ids
            and all(dist_mm(m.x, m.y, c.x, c.y) >= PLACE_CLEARANCE_MM for c in cubes)
        ]
        free_ids = {m.marker_id for m in free_markers}
        unknown_markers = [
            m
            for m in markers
            if m.marker_id not in occupied_ids and m.marker_id not in free_ids
        ]
    free_slots = free_placement_slots(calib, markers, cubes)
    return WorkspaceState(
        cubes=cubes,
        markers=markers,
        occupied=occupied,
        free_markers=free_markers,
        unknown_markers=unknown_markers,
        free_slots=free_slots,
        visible_marker_ids=visible_marker_ids,
    )


def cubes_of_color(cubes: list[CubeDetection], color: str) -> list[CubeDetection]:
    return [c for c in cubes if c.color == color]


def pick_largest_cube(cubes: list[CubeDetection]) -> CubeDetection | None:
    if not cubes:
        return None
    return max(cubes, key=lambda c: c.area)

