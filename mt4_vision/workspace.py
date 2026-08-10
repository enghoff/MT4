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
from mt4_vision.detect import (
    MAX_BLOB_AREA,
    MIN_BLOB_AREA,
    CubeDetection,
    detect_cubes,
)

# Cube centroid within this of a marker center counts the marker occupied
# when the ArUco tag does not decode. A decoded tag alone is *not*
# enough to call the marker free -- see PLACE_CLEARANCE_MM below.
# With the cube-top calibration fitted, on-marker cubes read 5-15mm from
# center while beside-the-paper cubes read 20mm+; 40mm classified adjacent
# cubes as occupants. Measured live 2026-07-14: a cube resting on the tag
# read 23mm from center, so a 22mm radius misses "occupied" and (with the
# tag covered) lands the marker in unknown instead, where the planner can
# neither place onto it nor pick it off.
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
# comfortably above the 35mm the travel height (safe_z 155 over table_z 120)
# actually uses today.
#
# Measured 2026-08-04 over the whole table on a 2mm grid: this check excludes
# NOTHING. Every XY where the TCP can grip at table height can also lift 50mm
# from there. At the outer edge the binding pose is the LOW one, not the lifted
# one -- at r=342 the arm reaches z=155 and z=177 but not z=120, because the
# coupled J2+J3 extension cap stops it stretching flat that far out. So the
# check can only ever enlarge the region, never shrink it. It is
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
# placed on one vanished from every later scan. The gate was in-frame only until
# 2026-08-10, which let the same failure back in by a different door: five of
# the fifteen slots below are near enough the camera that a cube on one imaged
# over the detector's size cap, so it was in frame, reachable, and still
# discarded. work_region_block_reason now checks the size a cube would image at
# as well as where it would land.
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
    # state was built without decode information (tests, and any caller that
    # assembles a state by hand).
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
# Nothing else may gate a pick or a place, and in particular the convex hull
# of the calibrated ArUco marker centres is not a work region. Marker
# positions are where the printed paper happens to lie; they are not a
# statement about the desk, the arm, or the camera. Measured 2026-08-02, that
# hull covers 828cm^2 of a table where the arm can safely work 2237cm^2: it
# stops at r=255-292mm depending on bearing, 40-90mm short in nearly every
# direction, and rejects everything past +-105deg (only the angular span is
# about right -- past ~100deg the desk itself runs out). A gate applied below
# the layer that writes reasons is worse still: a cube dropped there is absent
# from the snapshot entirely rather than listed with a reason.


def _table_polygon(calib: Calibration) -> np.ndarray | None:
    poly = getattr(calib, "table_polygon_robot", None)
    if not poly:
        return None
    return np.array(poly, dtype=np.float32)


def on_table(x: float, y: float, calib: Calibration) -> bool:
    """True when (x, y) is desk surface rather than past its edge.

    The polygon is stored in the calibration by ``calibrate_table_edge.py``
    and already carries its safety margin, so this is a plain containment
    test.

    A calibration with no polygon accepts everything, which makes this gate
    inert rather than strict -- there is no desk edge to be past. Run
    ``calibrate_table_edge.py`` to fit one, and note that doing so turns a
    gate *on*: places and picks beyond the fitted edge start being refused.
    ``calibrate_table_edge.py`` prints a loud warning when the fit is thin,
    because a thin fit is the version of this that refuses real desk.
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

    Asks that of the actual camera rather than of a radius about the base.
    This mount is steeply oblique with
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


def expected_cube_area_px2(x: float, y: float, calib: Calibration) -> float:
    """Blob area a cube resting at (x, y) should image at, in px^2.

    The detector's size gate used to be two fixed numbers, which only works if
    a cube looks the same size everywhere. On this mount it does not: the
    camera is low and steeply oblique, so a 20mm cube images ~1900px^2 at the
    far corners of the reach and ~5800px^2 at the near slots, a factor of 3.
    Fixed thresholds tuned in the middle of that range therefore throw away
    real cubes at one end while passing smears at the other.

    So predict instead of assume. The cube's eight corners are projected with
    the same height-corrected camera model ``camera_covers`` uses, and the
    silhouette is their convex hull -- which is what a solid-coloured cube
    presents to the detector. Yaw is not known at predict time and is taken
    axis-aligned; turning a cube swings the true silhouette by about +-10%,
    which is small next to the 3x positional swing and is absorbed by the
    tolerance band the callers apply (PICK_AREA_LO/HI below).
    """
    side = float(calib.cube_height_mm)
    half = 0.5 * side
    z0 = float(calib.table_z)
    corners = [
        calib.robot_to_pixel(x + dx, y + dy, z0 + dz)
        for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))
        for dz in (0.0, side)
    ]
    hull = cv2.convexHull(np.array(corners, dtype=np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull))


# How far a real cube's blob may sit from expected_cube_area_px2 and still be a
# pick target. Measured over 229 logged detections spanning the whole table
# (two shuffle runs, 2026-08-10): measured/predicted ran 0.78-1.04, median 0.92,
# with no drift across the 3x positional range -- so the band has to reach down
# to ~0.78 and does not have to be wide. 0.60 leaves a real cube at the far
# corners (predicted ~2100px^2, floor ~1260) clear of detect.py's own floor;
# 1.45 still refuses a blob half again cube-sized -- the arm's body, a smear, or
# two cubes merged into one contour. Mid-table that ceiling works out ~3600px^2,
# where the old fixed 5000 passed a smear as a cube.
PICK_AREA_LO = 0.60
PICK_AREA_HI = 1.45

# Yaw is unknown when predicting, and turning a cube swings its silhouette by
# about +-10% (measured 5030-5788px^2 over six yaws at (250,100)). A prediction
# used to decide whether a *future* cube will be detectable has to allow for the
# worst yaw, so it carries this factor; a prediction judged against a blob that
# has already been measured does not.
CUBE_YAW_AREA_SPREAD = 1.15

# A cube is not read where it stands. Its blob centroid sits at an unknown point
# on the top face and the cube-top map corrects for a nominal one, so the
# reported position drifts a few mm with yaw. Measured over 90 placements -- one
# cube teleported onto each of the 15 slots at six yaws, live detector,
# 2026-08-10 -- median 6.8mm, 95th 13.1mm, worst 17.8mm.
#
# On top of that the placement itself misses: measured 0.7-7.9mm over the ten
# to_slot placements in those runs, from the cube poses on the stage.
#
# So a spot chosen for a *future* cube has to stay inside camera coverage over a
# disc this wide, not just at its centre. (270,0) passes coverage at its centre,
# and a cube placed there was read at (272,-7), which does not -- in frame by the
# slot's reckoning, out of frame by the cube's.
#
# 20mm is the 95th-percentile read offset (13.1) plus the worst placement miss
# (7.9). Stacking the two worst cases instead would give 26mm, and that is not a
# free choice: at 20mm this rule costs 2 of the 15 slots, at 25mm it costs 5 --
# back to the 10 that were usable before any of this was fixed. The two errors
# have unrelated causes and need not align, so the 95th is the honest input.
CUBE_READ_TOLERANCE_MM = 20.0


def cube_area_block_reason(
    area: float, x: float, y: float, calib: Calibration
) -> str | None:
    """Why a blob of ``area`` at (x, y) is the wrong size to be a cube, or None.

    One predicate for the pick path and the place path both, so a cube can
    never be unpickable somewhere a place was offered.
    """
    expected = expected_cube_area_px2(x, y, calib)
    if expected <= 0.0:
        return None
    ratio = area / expected
    if ratio < PICK_AREA_LO:
        return (
            f"blob is {area:.0f}px2, under the {expected * PICK_AREA_LO:.0f}px2 "
            f"pick floor at ({x:.0f},{y:.0f}) where a cube images "
            f"{expected:.0f}px2 -- glare or an arm-paint fleck, not a cube"
        )
    if ratio > PICK_AREA_HI:
        return (
            f"blob is {area:.0f}px2, over the {expected * PICK_AREA_HI:.0f}px2 "
            f"pick ceiling at ({x:.0f},{y:.0f}) where a cube images "
            f"{expected:.0f}px2 -- the arm's own body, a smear, or two cubes "
            f"run together, not a cube"
        )
    return None


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
    require_camera: bool = True,
    read_tolerance_mm: float = 0.0,
) -> str | None:
    """The first work-region gate (x, y) fails, in prose, or None.

    Reports; it does not decide. ``in_work_region`` is this function asking
    whether the answer is None, so the two cannot drift into disagreeing
    definitions of the region -- the failure the marker-hull gate had when it
    lived in two files with two different allowances.

    ``read_tolerance_mm`` widens the coverage test into a disc, for a caller
    choosing a spot for a cube that is not there yet -- the cube will be read a
    few mm off wherever it lands, and all of that has to stay in frame. It
    defaults to 0 because the pick path asks about a cube it has *already*
    detected, and holding that cube to where it might have been read instead
    would refuse perfectly visible ones near the edge.

    ``require_camera=False`` drops the camera-coverage test and keeps every
    other one. The distinction is what the gate is *for*: reach, the keep-out,
    the joint limits and the desk edge say the arm cannot do the thing, while
    coverage says the arm could do it and would not see the result again.
    That is worth refusing when the stack is choosing a spot on its own
    initiative -- a free slot nothing can re-detect is a bad autonomous choice
    -- and not worth refusing when a person has pointed at the spot. Coverage
    vetoes 316 cm2 of the 1541 cm2 where the arm can hold a grasp pose at table
    height, 21% -- measured with a desk polygon fitted, which narrows the 2237
    cm2 the arm can reach down to that 1541. A calibration carrying no
    `table_polygon_robot` has no such narrowing, so the share differs; re-measure
    after `calibrate_table_edge.py`.
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
    if require_camera and not camera_covers(x, y, calib):
        return (
            "outside the camera frame -- an object placed here could not be "
            "seen again, so nothing could pick it up"
        )
    if require_camera and read_tolerance_mm > 0.0:
        # Coverage has to hold wherever the cube will be *read*, not only at the
        # spot it is set down on. See CUBE_READ_TOLERANCE_MM.
        for i in range(8):
            a = 2.0 * math.pi * i / 8.0
            if not camera_covers(
                x + read_tolerance_mm * math.cos(a),
                y + read_tolerance_mm * math.sin(a),
                calib,
            ):
                return (
                    f"within {read_tolerance_mm:.0f}mm of the edge of the camera "
                    f"frame -- a cube set down here lands and reads far enough "
                    f"off to fall outside it, so nothing could pick it up"
                )
    if require_camera:
        # In frame is not the same as detected. detect_cubes discards a contour
        # outside its absolute size limits before any pick logic sees it, and a
        # cube's silhouette varies 3x across this table, so a slot can be fully
        # in frame and still sit where the cube images off the end of that
        # range. That is what stranded three of nine cubes per shuffle run on
        # 2026-08-10 (measured: cubes on the five near-camera slots sat idle
        # 18 carries on average against 4 elsewhere, p=0.002).
        expected = expected_cube_area_px2(x, y, calib)
        if expected * CUBE_YAW_AREA_SPREAD > MAX_BLOB_AREA:
            return (
                f"a cube here images ~{expected:.0f}px2, over the "
                f"{MAX_BLOB_AREA:.0f}px2 the detector will keep -- it would be "
                f"discarded as a smear and never picked up again"
            )
        if expected / CUBE_YAW_AREA_SPREAD < MIN_BLOB_AREA:
            return (
                f"a cube here images ~{expected:.0f}px2, under the "
                f"{MIN_BLOB_AREA:.0f}px2 the detector will keep -- it would be "
                f"discarded as glare and never picked up again"
            )
    return None


def in_work_region(
    x: float,
    y: float,
    calib: Calibration,
    *,
    z: float | None = None,
    lift_mm: float = PICK_LIFT_MM,
    require_camera: bool = True,
    read_tolerance_mm: float = 0.0,
) -> bool:
    """True when a pick or place may happen at (x, y). See the block above."""
    return work_region_block_reason(
        x,
        y,
        calib,
        z=z,
        lift_mm=lift_mm,
        require_camera=require_camera,
        read_tolerance_mm=read_tolerance_mm,
    ) is None


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
    marker, the loser can still claim its own second-nearest marker. A
    per-cube nearest-only rule drops the loser entirely, leaving a physically
    occupied marker "free" -- an invitation to stack.
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
        # read_tolerance_mm: this is the autonomous-choice path, picking a spot
        # for a cube that is not there yet, so coverage has to hold over the disc
        # the cube will actually be read in.
        if not in_work_region(
            sx, sy, calib, read_tolerance_mm=CUBE_READ_TOLERANCE_MM
        ):
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

