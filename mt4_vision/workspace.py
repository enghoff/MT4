"""Work-surface model: calibrated markers, cube detections, occupancy, slots."""

from __future__ import annotations

import math
from dataclasses import dataclass
import cv2
import numpy as np

from mt4_jog.joints import (
    GROUND_Z_MM,
    J2_J3_SUM_MAX_STEPS,
    J2_J3_SUM_MIN_STEPS,
    JOINT_SOFT_MAX_STEPS,
    JOINT_SOFT_MIN_STEPS,
)
from mt4_jog.kinematics import JointAnglesDeg, ik_position, steps_from_angles
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

# Elbow-up IK branch seed (the tape-fit park posture; see kinematics.py).
NEAR_J2_DEG = 107.0
NEAR_J3_DEG = -9.3

# Vertical clearance a pick or place must have at its own XY: the TCP has to
# rise this far straight up off the grasp without leaving the joint envelope.
# 50mm covers a 20mm cube plus fingertip length with room to spare, and is
# comfortably above the 27.8mm the travel height (safe_z 155 over table_z
# 127.2) actually uses today.
#
# Measured 2026-08-02 over the whole table on a 5mm grid: this check excludes
# NOTHING. Every XY where the TCP can grip at table height can also lift 50mm
# from there. At the outer edge the binding pose is the LOW one, not the
# lifted one -- at r=342 the arm reaches z=155 and z=177 but not z=127.2,
# because the coupled J2+J3 extension cap stops it stretching flat that far
# out. So the check can only ever enlarge the region, never shrink it. It is
# kept anyway because it is nearly free and it is the thing we actually mean:
# if safe_z, table_z, or MT4_JOINT_SOFT_* ever move, this notices.
PICK_LIFT_MM = 50.0

# Keep-out from the frame border when asking "can the camera confirm a cube
# here". A cube top runs ~45-65px across near the camera, so a centroid this
# far in still has its whole blob inside the frame -- a clipped contour moves
# the centroid, which is worse than not seeing it at all.
CAMERA_FRAME_MARGIN_PX = 50.0

# Frame size assumed when the calibration does not record one (older files).
# Must match what calibrate_vision.py captured at -- the homography is only
# valid for the resolution it was fitted at.
DEFAULT_FRAME_SIZE_PX = (1280, 720)

# Open-table placement candidates (robot frame, mm). Shared with
# calibrate_height.py's probe grid. These are *candidates* only --
# free_placement_slots gates every one of them through in_work_region, so a
# slot that the camera cannot confirm never reaches a caller. Before that gate
# existed, five of these eight sat in the detector's discard zone and a cube
# placed on one vanished from every later scan.
PLACEMENT_SLOTS: list[tuple[float, float]] = [
    (150.0, 100.0),
    (150.0, -100.0),
    (200.0, -60.0),
    (200.0, 60.0),
    (240.0, -150.0),
    (240.0, 150.0),
    (150.0, -250.0),
    (150.0, 250.0),
    (270.0, 0.0),
    (250.0, -100.0),
    (250.0, 100.0),
    (100.0, -280.0),
    (100.0, 280.0),
    (200.0, -220.0),
    (200.0, 220.0),
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


# -- the work region ------------------------------------------------------
#
# "Where may a pick or a place happen?" has exactly one answer, and it is
# ``in_work_region``. Four independent things have to be true, and each is
# separately checkable so a refusal can name the one that failed:
#
#   1. the arm can hold the grasp pose            (joint_reachable at z)
#   2. it can lift PICK_LIFT_MM straight off it   (joint_reachable at z+lift)
#   3. the desk is physically there               (on_table)
#   4. the camera can confirm what landed         (camera_covers)
#
# What this replaced, and why: the old gate was the convex hull of the
# calibrated ArUco marker centres, applied twice with different allowances --
# -80px in detect.py (a hard drop, the blob never became a detection) and
# -55mm here (a demotion to non-pick). Marker positions are where the printed
# paper happens to lie; they are not a statement about the desk, the arm, or
# the camera. Measured 2026-08-02, that hull admitted 828cm^2 of a table where
# the arm can safely work 2278cm^2, cut off at r=255-292mm depending on
# bearing, and rejected everything past +-105deg. Three cubes physically on
# the desk and well inside the arm's envelope were absent from the snapshot
# entirely -- not listed-with-a-reason, just gone, because the drop happened
# below the layer that writes reasons.
#
# The angular span of that hull turned out to be about right: past ~100deg the
# desk itself runs out. It was the RADIUS that was badly under-called, by
# 40-90mm in nearly every direction.


def _table_polygon(calib: Calibration) -> np.ndarray | None:
    poly = getattr(calib, "table_polygon_robot", None)
    if not poly:
        return None
    return np.array(poly, dtype=np.float32)


def on_table(x: float, y: float, calib: Calibration) -> bool:
    """True when (x, y) is desk surface rather than past its edge.

    The polygon is stored in the calibration by ``calibrate_table_edge.py``
    and already carries its safety margin, so this is a plain containment
    test. A calibration with no polygon accepts everything -- the same
    fallback ``within_pick_hull`` used with fewer than three markers, and the
    reason ``calibrate_table_edge.py`` prints a loud warning when the fit is
    thin.
    """
    poly = _table_polygon(calib)
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def camera_covers(
    x: float,
    y: float,
    calib: Calibration,
    *,
    margin_px: float = CAMERA_FRAME_MARGIN_PX,
    z: float | None = None,
) -> bool:
    """True when a cube resting at (x, y) images inside the frame.

    Asks the question the old ``MAX_VERIFIABLE_RADIUS_MM`` circle was reaching
    for, but asks it of the actual camera. This mount is steeply oblique with
    its nadir off to one side at (518, -35), so coverage is nothing like a
    circle about the base: the frame's near edge cuts in at x~284mm on the
    centreline while both sides reach past 380mm. A 240mm circle threw away
    everything the arm could reach at +-90deg and still did not describe the
    +x limit correctly.

    Projected at cube-top height, because that is what the detector sees.
    """
    top = calib.table_z + calib.cube_height_mm if z is None else float(z)
    w, h = getattr(calib, "frame_size_px", None) or DEFAULT_FRAME_SIZE_PX
    px, py = calib.robot_to_pixel(float(x), float(y), top)
    return (
        margin_px <= px <= float(w) - margin_px
        and margin_px <= py <= float(h) - margin_px
    )


def joint_reachable(x: float, y: float, z: float) -> bool:
    """True when (x, y, z) passes envelope, closed-form IK, and soft limits.

    ``ik_position`` alone only checks link-length geometry; the real ceiling
    over the desk (~315mm at the marker radii) comes from the J3 soft max,
    which ``joints_within_soft_limits`` enforces.

    Lives here rather than in stackpath so the work-region predicate can use
    it without workspace importing stackpath (which imports workspace).
    ``stackpath`` re-exports it.
    """
    r = math.hypot(x, y)
    if r < KEEPOUT_RADIUS_MM - KEEPOUT_TARGET_MARGIN_MM or r > MAX_REACH_MM:
        return False
    if z < GROUND_Z_MM - 0.05:
        return False
    near = JointAnglesDeg(
        math.degrees(math.atan2(y, x)), NEAR_J2_DEG, NEAR_J3_DEG, 0.0
    )
    q = ik_position(x, y, z, near=near)
    if q is None:
        return False
    return joints_within_soft_limits(steps_from_angles(q))


def max_z_at(x: float, y: float, *, hi: float = 340.0) -> float | None:
    """Highest joint-reachable TCP z at (x, y), or None when nothing is."""
    lo = GROUND_Z_MM
    if not joint_reachable(x, y, lo):
        return None
    if joint_reachable(x, y, hi):
        return hi
    while hi - lo > 0.5:
        mid = 0.5 * (lo + hi)
        if joint_reachable(x, y, mid):
            lo = mid
        else:
            hi = mid
    return lo


def work_region_block_reason(
    x: float,
    y: float,
    calib: Calibration,
    *,
    z: float | None = None,
    lift_mm: float = PICK_LIFT_MM,
) -> str | None:
    """The first work-region gate (x, y) fails, in prose, or None.

    Reports; it does not decide. ``in_work_region`` is this function asking
    whether the answer is None, so the two cannot drift into disagreeing
    definitions of the region -- the failure the marker-hull gate had when it
    lived in two files with two different allowances.
    """
    gz = float(calib.table_z) if z is None else float(z)
    r = math.hypot(x, y)
    if not is_mp_reachable_xy(x, y):
        return (
            f"r={r:.0f}mm is inside the {KEEPOUT_RADIUS_MM:.0f}mm J1 keep-out "
            f"(firmware mp refuses any target there)"
        )
    if r > MAX_REACH_MM:
        return f"r={r:.0f}mm is beyond the {MAX_REACH_MM:.0f}mm max reach"
    if not joint_reachable(x, y, gz):
        return (
            f"no arm pose holds the TCP at ({x:.0f},{y:.0f},{gz:.0f}) inside "
            f"the joint soft limits"
        )
    if not joint_reachable(x, y, gz + lift_mm):
        return (
            f"the {lift_mm:.0f}mm lift off ({x:.0f},{y:.0f}) leaves the joint "
            f"soft limits"
        )
    if not on_table(x, y, calib):
        return "past the edge of the desk -- nothing there to set an object on"
    if not camera_covers(x, y, calib):
        return (
            "outside the camera frame -- an object placed here could not be "
            "seen again, so nothing could pick it up"
        )
    return None


def in_work_region(
    x: float,
    y: float,
    calib: Calibration,
    *,
    z: float | None = None,
    lift_mm: float = PICK_LIFT_MM,
) -> bool:
    """True when a pick or place may happen at (x, y). See the block above."""
    return work_region_block_reason(x, y, calib, z=z, lift_mm=lift_mm) is None


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
        # The work-region gate is the whole point of this call. Without it
        # the reach/keep-out pair alone let through slots the camera could
        # not confirm: measured 2026-08-02, five of the eight original
        # PLACEMENT_SLOTS sat in the detector's discard zone and were still
        # being offered as placeable=True, so a cube placed on one dropped
        # out of every later scan and had to be retrieved by hand.
        if not in_work_region(sx, sy, calib):
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

