"""ArUco marker and colored-cube detection in the (oblique) scene camera."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from mt4_vision.calib import Calibration

# HSV ranges (OpenCV scale: H 0-179, S/V 0-255) for solid-colored cubes under
# typical indoor lighting. Lighting-sensitive by nature -- override per-setup
# via Calibration.color_ranges rather than editing these. Red wraps the hue
# axis, hence two bands.
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    # No "orange" by default: the wood table and red cubes' shaded side faces
    # both land in the orange hue band. Add it via Calibration.color_ranges
    # if the cube set actually has orange.
    "red": [((0, 100, 70), (9, 255, 255)), ((170, 100, 70), (179, 255, 255))],
    "yellow": [((23, 100, 100), (34, 255, 255))],
    # Green cubes sit darker than the rest under this lighting (V median ~84,
    # shadow side lower still), hence the low V floor.
    "green": [((36, 70, 45), (88, 255, 255))],
    "blue": [((90, 100, 60), (128, 255, 255))],
}
# Reject blobs smaller than this (px^2) -- noise, shadows, cable ties.
# At the measured mount distance real cube blobs are ~1600-3600px^2 (measured
# 2026-07-20). Keep a 2x margin below the smallest real cube: sub-cube blobs
# are never cubes here, and a floor near 120 lets a 263px^2 specular glare on
# a laminated marker (reflecting the wall) pass as a blue cube -- which
# calibrate_height then "picks" and "places", recording the phantom's pixels
# as probe data.
MIN_BLOB_AREA = 800.0
# Reject blobs larger than this -- the arm's own orange/red body still reads
# as "red", but a real on-pad cube is ~2800-3600px^2 (measured 2026-07-20), so
# a cap anywhere near cube size drops the cube as "too big" and leaves only
# arm-paint flecks. Arm-scale blobs that survive are still rejected downstream
# by keep-out / reach / work region (scene.is_phantom_detection). detect_cubes
# sorts largest-first, so this cap still matters when an uncapped wall/arm
# smear would otherwise outrank every cube.
MAX_BLOB_AREA = 6000.0
# Reject blobs whose bounding-box aspect is far from square (cubes are square
# from above; this drops elongated glare streaks and desk-edge artifacts).
MAX_ASPECT = 2.0
# Top-face centroid: the whole-blob centroid includes however much side face
# the oblique camera sees, which pulls it 4-6px (~4-7mm) off the top-face
# center by an amount that varies across the desk -- measured 2026-07-18 as
# the dominant residual in the cube-top map (marker-3 region ~10mm). The top
# face cannot be identified by brightness alone (red's side faces are darker
# than its top, but green's lit front face is BRIGHTER than its top --
# observed live), so use geometry: the top face is the blob's topmost pixels
# for THIS mount. That is nadir-direction-dependent, not a universal top-down
# fact: the camera is oblique with its nadir toward +x (robot), so raising a
# point shifts its image toward -x, i.e. upward, putting the top face above
# the visible side faces. A camera whose nadir sat on the opposite side would
# put the top face at the BOTTOM of the blob and this seed would grab a side
# face -- re-check the seed edge if the camera is ever remounted. Seed there,
# then keep the connected region of similar V. A blob that is all top face
# (viewed near-vertically) is V-uniform, so the whole blob is kept and the
# centroid is unchanged.
TOP_FACE_SEED_FRACTION = 0.2  # top rows of the blob used as the seed
TOP_FACE_V_TOL = 35.0  # faces differ by >=40-80 V; within-face spread ~15-25
TOP_FACE_MIN_PIXELS = 20
# Half-width (px) of the square sampled at the top-face centroid for
# CubeDetection.sat. Small enough to stay on a ~20px cube face, big enough
# that JPEG chroma noise on a single pixel cannot swing the median.
SATURATION_PATCH_PX = 4

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass
class MarkerDetection:
    marker_id: int
    px: float  # center, pixels
    py: float
    # The 4 outline corners (pixels, detector order: TL, TR, BR, BL relative
    # to the marker's own printed orientation). Subpixel-refined. These carry
    # far more geometric information than the center alone -- 20 corners
    # across 5 identical printed squares are what make the table-plane
    # perspective actually observable (5 centers alone are not enough; see
    # table_fit.py).
    corners: list[list[float]] | None = None


@dataclass
class CubeDetection:
    color: str
    px: float  # centroid, pixels
    py: float
    area: float  # px^2
    x: float | None = None  # robot frame (mm), None when uncalibrated
    y: float | None = None
    # Robot-frame yaw (deg) of one top-face edge: atan2 of an edge mapped
    # through the cube-top (or table) homography. Squares are 90°-periodic;
    # None when the blob is too small / uncalibrated. Used for J4 face-align.
    yaw_deg: float | None = None
    # Median HSV saturation over the blob (0-255). A painted cube is deeply
    # saturated; a specular highlight is near-white and only clips a colour
    # band at its fringe, so this separates the two by an order of magnitude
    # (see scene.is_glare_blob). Defaults to a saturated value so a
    # hand-built detection in a test is never mistaken for glare.
    sat: float = 255.0

    def as_dict(self) -> dict[str, object]:
        return {
            "color": self.color,
            "pixel": [round(self.px, 1), round(self.py, 1)],
            "area_px": round(self.area),
            "x": None if self.x is None else round(self.x, 1),
            "y": None if self.y is None else round(self.y, 1),
            "yaw_deg": None if self.yaw_deg is None else round(self.yaw_deg, 1),
            "sat": round(self.sat),
        }


def detect_markers(
    frame: np.ndarray, dict_name: str = "4x4_50"
) -> list[MarkerDetection]:
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f"unknown ArUco dict {dict_name!r}, one of {sorted(ARUCO_DICTS)}")
    params = cv2.aruco.DetectorParameters()
    # Subpixel corner refinement: the corners feed the table-plane fit, where
    # a half-pixel error is ~1mm on the table.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name]), params
    )
    corners, ids, _rejected = detector.detectMarkers(frame)
    if ids is None:
        return []
    out = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        quad = marker_corners[0]
        center = quad.mean(axis=0)
        out.append(
            MarkerDetection(
                int(marker_id),
                float(center[0]),
                float(center[1]),
                corners=[[float(cx), float(cy)] for cx, cy in quad],
            )
        )
    return sorted(out, key=lambda m: m.marker_id)


def scan_marker_dicts(frame: np.ndarray) -> dict[str, int]:
    """Try every known dictionary; return {dict_name: markers_found} for hits."""
    hits = {}
    for name in ARUCO_DICTS:
        found = detect_markers(frame, name)
        if found:
            hits[name] = len(found)
    return hits


def _top_face_centroid(
    hsv: np.ndarray,
    contour: np.ndarray,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """Centroid of the cube's top face within the blob.

    Seeds from the blob's topmost rows (always top face), keeps the
    connected region whose V matches the seed. Falls back to the whole-blob
    centroid when the blob is too small to segment meaningfully.
    """
    bx, by, bw, bh = cv2.boundingRect(contour)
    local = np.zeros((bh, bw), dtype=np.uint8)
    cv2.drawContours(local, [contour - [bx, by]], -1, 255, -1)
    blob = local > 0
    if blob.sum() < TOP_FACE_MIN_PIXELS:
        return fallback
    v = hsv[by : by + bh, bx : bx + bw, 2].astype(np.float64)

    rows = np.nonzero(blob.any(axis=1))[0]
    seed_limit = rows[0] + max(2, int(round(bh * TOP_FACE_SEED_FRACTION)))
    seed = blob & (np.arange(bh)[:, None] < seed_limit)
    if seed.sum() < 4:
        return fallback
    med = float(np.median(v[seed]))

    similar = blob & (np.abs(v - med) <= TOP_FACE_V_TOL)
    n, labels = cv2.connectedComponents(similar.astype(np.uint8))
    seed_labels = np.unique(labels[seed & similar])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size == 0:
        return fallback
    sel = np.isin(labels, seed_labels)
    ys, xs = np.nonzero(sel)
    if xs.size < TOP_FACE_MIN_PIXELS:
        return fallback
    return bx + float(xs.mean()), by + float(ys.mean())


def _blob_saturation(hsv: np.ndarray, px: float, py: float) -> float:
    """Median S (0-255) in a small disc at the blob's top-face centroid.

    Sampled at the centroid, NOT over the filled contour: the contour only
    contains pixels that already passed the colour band's S floor, so its
    median is bounded below by that floor and cannot distinguish anything
    (measured -- on-marker glare contours median S 107, real green cubes
    134, overlapping). A specular highlight on a laminated marker is a
    near-white core with a faintly tinted rim; only the rim clips the band,
    so the blob is a ring and its centre is the part that gives it away
    (measured S 5-88 for glare vs 215-244 for real cubes).

    This is also the pixel the gripper is sent to descend on, which is the
    right place to ask "is the cube's colour actually there?".
    """
    r = SATURATION_PATCH_PX
    h, w = hsv.shape[:2]
    y0, y1 = max(0, int(py) - r), min(h, int(py) + r + 1)
    x0, x1 = max(0, int(px) - r), min(w, int(px) + r + 1)
    patch = hsv[y0:y1, x0:x1, 1]
    return float(np.median(patch)) if patch.size else 255.0


def _robot_edge_yaw_deg(
    contour: np.ndarray,
    calibration: Calibration,
) -> float | None:
    """Yaw (deg) of a cube face edge in robot XY from minAreaRect corners.

    Maps two adjacent box corners through the cube-top (preferred) or table
    map and returns atan2(dy, dx). Squares are 90°-periodic -- either edge
    is a valid face direction. Returns None when the contour is degenerate.
    """
    if len(contour) < 3:
        return None
    rect = cv2.minAreaRect(contour.astype(np.float32))
    box = cv2.boxPoints(rect)
    e01 = box[1] - box[0]
    e12 = box[2] - box[1]
    if float(np.linalg.norm(e01)) >= float(np.linalg.norm(e12)):
        p0, p1 = box[0], box[1]
    else:
        p0, p1 = box[1], box[2]
    if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) < 2.0:
        return None
    x0, y0 = calibration.pixel_to_robot(float(p0[0]), float(p0[1]), on_cube_top=True)
    x1, y1 = calibration.pixel_to_robot(float(p1[0]), float(p1[1]), on_cube_top=True)
    dx, dy = x1 - x0, y1 - y0
    if math.hypot(dx, dy) < 1e-3:
        return None
    return math.degrees(math.atan2(dy, dx))


def color_ranges_for(calibration: Calibration | None) -> dict[str, list]:
    """The named HSV bands in force, defaults overlaid with the calibration's.

    One place, because ``classify_color`` and ``detect_cubes`` must agree about
    what "green" is: ``instruct`` refuses a task when a named colour
    contradicts an entity's, and a colour that meant one thing for a cube and
    another for a statue would refuse correct answers.
    """
    ranges: dict[str, list] = dict(COLOR_RANGES)
    if calibration is not None:
        ranges.update(calibration.color_ranges)
    return ranges


def band_mask(hsv: np.ndarray, bands: list) -> np.ndarray:
    """Union of one colour's HSV bands. Red needs two -- it wraps the hue axis."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in bands:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


# A colour must own this much of the sampled pixels to be claimed, and beat the
# runner-up by this margin. Both are deliberately blunt: the answer this
# function must get right is "I don't know", not "which of two greens". A
# stapler is grey plastic and chrome and matches nothing; a two-tone toy
# matches two things weakly; wood desk leaking into a loose mask matches
# nothing. All three have to come back None, because the caller turns a colour
# into a word that a task is then *refused* against.
COLOR_MIN_SHARE = 0.35
COLOR_MARGIN = 0.15


def classify_color(
    frame: np.ndarray,
    mask: np.ndarray | None = None,
    origin: tuple[int, int] = (0, 0),
    calibration: Calibration | None = None,
) -> str | None:
    """Which named colour is this region, or None when it is not clearly one.

    The **same** named HSV bands ``detect_cubes`` uses, including any override
    in ``Calibration.color_ranges``, so "green" means one measured thing across
    the whole stack rather than one thing for a cube and another for a statue.
    That shared meaning is the entire point: ``instruct`` refuses a task when a
    named colour contradicts an entity's own, and a colour word that meant
    something slightly different for registered objects would refuse correct
    answers instead.

    ``mask`` is the object's silhouette (as ``locate`` stores it) with its
    top-left at ``origin`` in frame pixels. Without one, the middle of ``frame``
    is sampled instead -- a detector box's centre is overwhelmingly object,
    and anything the desk leaks in matches no band and pushes the answer
    toward None, which is the safe direction.
    """
    ranges = color_ranges_for(calibration)

    if mask is not None and mask.size:
        h, w = mask.shape[:2]
        x0, y0 = int(origin[0]), int(origin[1])
        crop = frame[y0 : y0 + h, x0 : x0 + w]
        if crop.shape[:2] != (h, w):  # mask ran off the frame edge
            mask = mask[: crop.shape[0], : crop.shape[1]]
        sel = mask.astype(bool)
    else:
        h, w = frame.shape[:2]
        # Central half by each side, i.e. the middle quarter by area.
        crop = frame[h // 4 : h - h // 4, w // 4 : w - w // 4]
        sel = np.ones(crop.shape[:2], dtype=bool)
    if crop.size == 0 or not sel.any():
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = int(sel.sum())
    shares = {
        color: float((band_mask(hsv, bands).astype(bool) & sel).sum()) / total
        for color, bands in ranges.items()
    }

    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] < COLOR_MIN_SHARE:
        return None
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if ranked[0][1] - runner_up < COLOR_MARGIN:
        return None
    return ranked[0][0]


def detect_cubes(
    frame: np.ndarray,
    calibration: Calibration | None = None,
    colors: list[str] | None = None,
) -> list[CubeDetection]:
    """Detect colored cubes; robot XY filled in when a calibration is given."""
    ranges = color_ranges_for(calibration)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Close first to heal ragged threshold edges, then a small open for
    # speckle -- a bigger open kernel eats the ~15-20px cube blobs whole.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Deferred: workspace imports this module for CubeDetection.
    from mt4_vision.workspace import on_table

    detections: list[CubeDetection] = []
    dropped_off_table = 0
    for color, bands in ranges.items():
        if colors is not None and color not in colors:
            continue
        mask = band_mask(hsv, bands)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
                continue
            _x, _y, w, h = cv2.boundingRect(contour)
            if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            px, py = _top_face_centroid(
                hsv, contour, (m["m10"] / m["m00"], m["m01"] / m["m00"])
            )
            det = CubeDetection(color, px, py, area, sat=_blob_saturation(hsv, px, py))
            if calibration is not None:
                det.x, det.y = calibration.pixel_to_robot(px, py, on_cube_top=True)
                off = calibration.color_xy_offset_mm.get(color)
                if off:
                    det.x += float(off[0])
                    det.y += float(off[1])
                # The ONLY thing dropped here is what is not on the desk at
                # all: the arm's own links and the wall behind it. On this
                # steeply oblique mount anything with height images ABOVE the
                # desk-edge line, so the arm's body maps behind the edge and
                # falls out here for free, while its base maps inside the
                # keep-out and is rejected later.
                #
                # Everything else stays, including cubes the arm cannot reach
                # and cubes the camera can barely confirm. Those are real
                # objects: they must still block a placement beside them, and
                # a caller asking about one deserves the reason rather than
                # silence. Deciding they are not PICK targets is
                # scene.is_phantom_detection's job, one layer up, where a
                # reason can be attached.
                #
                # This replaced a -80px test against the marker-centre hull,
                # which was a hard drop with no reason attached. Measured
                # 2026-08-02 on a live frame, it was deleting three cubes that
                # were plainly on the desk and plainly in reach.
                if not on_table(det.x, det.y, calibration):
                    dropped_off_table += 1
                    continue
                det.yaw_deg = _robot_edge_yaw_deg(contour, calibration)
            detections.append(det)
    if dropped_off_table:
        _LAST_DROPPED[0] = dropped_off_table
    else:
        _LAST_DROPPED[0] = 0
    return sorted(detections, key=lambda d: -d.area)


# Blobs the last detect_cubes() call discarded as not-on-the-desk. Surfaced in
# the scene summary so "the cube vanished" is diagnosable instead of silent --
# the failure mode of the gate this replaced was that it left no trace at all.
_LAST_DROPPED = [0]


def last_dropped_off_table() -> int:
    """Count of off-desk blobs discarded by the most recent detect_cubes()."""
    return _LAST_DROPPED[0]
