"""The desk-edge scan must not care what the desk or the wall are made of.

Every frame here is synthetic, and deliberately so. What is under test is that
the scan carries to a surface and a background the code has never seen, which
a photograph of one particular desk cannot demonstrate -- and the field failure
these guard against was precisely a colour that used to work and stopped
(see calibrate_table_edge.__doc__).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibrate_table_edge import (  # noqa: E402
    MIN_SEPARATION_SIGMA,
    SEARCH_ROWS,
    acceptance_failure,
    find_edge_pixels,
    fit_edge,
    learn_desk_reference,
    polygon_from_edge,
)
from rig import CALIB  # noqa: E402

WIDTH, HEIGHT = 1280, 720
BOUNDARY_ROW = 310  # inside SEARCH_ROWS, and below every reference sample

# BGR pairs, none of them this rig's wood-against-teal. Each is (surface,
# background) and each has to work with no change to the source.
SURFACES = {
    "pale pink laminate under a dark teal wall": ((186, 196, 205), (74, 61, 30)),
    "warm orange wood under a dark teal wall": ((120, 160, 200), (74, 61, 30)),
    "dark grey slate under a cream wall": ((70, 68, 66), (205, 225, 240)),
    "green cutting mat under a wood-panelled wall": ((90, 130, 70), (120, 160, 200)),
    "white bench under a black curtain": ((225, 228, 230), (25, 25, 28)),
}


def frame_with_edge(
    surface_bgr, background_bgr, *, boundary_row: int = BOUNDARY_ROW
) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:boundary_row] = background_bgr
    frame[boundary_row:] = surface_bgr
    # Sensor noise, so nothing passes by virtue of being mathematically exact.
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 2.0, frame.shape)
    return np.clip(frame.astype(float) + noise, 0, 255).astype(np.uint8)


def measure(frame: np.ndarray):
    """Fitted edge and the acceptance verdict, the way main() computes them."""
    scan = find_edge_pixels(frame, CALIB)
    if len(scan.pixels) < 4:
        return scan, None, acceptance_failure(scan)
    robot = np.array(
        [CALIB.pixel_to_robot(px, py, on_cube_top=True) for px, py in scan.pixels]
    )
    a, b, keep, _ = fit_edge(robot[:, 0], robot[:, 1])
    rms = float((robot[keep, 0] - (a * robot[keep, 1] + b)).std())
    return scan, (a, b, keep, rms), acceptance_failure(scan, keep, rms)


def expected_edge_x(boundary_row: int = BOUNDARY_ROW) -> float:
    """Robot x the boundary row maps to at mid-frame."""
    return CALIB.pixel_to_robot(WIDTH / 2, boundary_row, on_cube_top=True)[0]


@pytest.mark.parametrize("name", sorted(SURFACES))
def test_edge_is_found_whatever_the_two_surfaces_are(name):
    surface, background = SURFACES[name]
    scan, fit, problem = measure(frame_with_edge(surface, background))
    assert problem is None, f"{name}: {problem}"
    a, b, keep, rms = fit
    # A synthetic edge is a straight row, so the row it recovers is the test --
    # not the millimetre value, which is the homography's business.
    found = np.median(scan.pixels[keep, 1])
    assert abs(found - BOUNDARY_ROW) <= 2, f"{name}: found row {found}"


def test_lightness_gradient_across_both_surfaces_does_not_split_them():
    """A 7x lightness range on one wall is what this rig actually presents
    (L 109 daylit to 16 shaded, measured 2026-08-04), so neither surface may be
    required to be uniformly lit."""
    surface, background = SURFACES["pale pink laminate under a dark teal wall"]
    frame = frame_with_edge(surface, background).astype(float)
    ramp = np.linspace(0.25, 1.6, WIDTH)[None, :, None]
    frame = np.clip(frame * ramp, 0, 255).astype(np.uint8)

    scan, fit, problem = measure(frame)
    assert problem is None, problem
    found = np.median(scan.pixels[fit[2], 1])
    assert abs(found - BOUNDARY_ROW) <= 2


def test_clutter_near_the_edge_does_not_move_it():
    """Cables along the back of the desk are why the region is closed before its
    outline is read: on the real rig they push a per-column run rule 100px down
    the desk across half the frame.

    They lie just below the edge here, as they do on the rig, leaving a thin
    strip of surface above them. Clutter that BURIES the edge over most of the
    frame is not recoverable by this or any method -- the edge is not in the
    image -- and the honest outcome there is the outlier trim reporting a short
    span, not a rescued number.
    """
    surface, background = SURFACES["warm orange wood under a dark teal wall"]
    frame = frame_with_edge(surface, background)
    cv2.line(frame, (200, BOUNDARY_ROW + 25), (900, BOUNDARY_ROW + 25), (30, 30, 30), 13)
    cv2.rectangle(
        frame, (400, BOUNDARY_ROW + 50), (520, BOUNDARY_ROW + 110), (20, 20, 20), -1
    )

    scan, fit, problem = measure(frame)
    assert problem is None, problem
    found = np.median(scan.pixels[fit[2], 1])
    assert abs(found - BOUNDARY_ROW) <= 2


def test_chroma_carries_it_when_there_is_chroma_and_lightness_when_there_is_not():
    """The choice is measured, not configured. Comparing on lightness where
    chroma would do is what put this rig's edge at +88mm instead of -72mm."""
    coloured, wall = SURFACES["green cutting mat under a wood-panelled wall"]
    assert find_edge_pixels(frame_with_edge(coloured, wall), CALIB).desk.channels == (1, 2)

    grey, curtain = SURFACES["white bench under a black curtain"]
    assert find_edge_pixels(frame_with_edge(grey, curtain), CALIB).desk.channels == (
        0,
        1,
        2,
    )


def test_a_background_that_looks_like_the_desk_is_refused_not_guessed():
    """Two surfaces this close have no findable boundary. Saying so is the
    whole point -- a polygon invented here is silent and moves the desk."""
    surface = (186, 196, 205)
    scan, _, problem = measure(frame_with_edge(surface, (184, 194, 203)))
    assert scan.separation < MIN_SEPARATION_SIGMA
    assert problem is not None and "spreads" in problem


def test_a_frame_with_no_edge_at_all_is_refused():
    """One flat surface everywhere: whatever the scan reports is noise."""
    _, _, problem = measure(frame_with_edge((186, 196, 205), (186, 196, 205)))
    assert problem is not None


def test_desk_reference_ignores_a_minority_of_non_desk_samples():
    """The sampled grid lands on markers, cubes and cables as well as desk, so
    the reference has to be the majority material rather than their average."""
    surface, background = SURFACES["warm orange wood under a dark teal wall"]
    clean = learn_desk_reference(frame_with_edge(surface, background), CALIB)

    littered = frame_with_edge(surface, background)
    for x0 in range(300, 1100, 90):  # black pads over part of the sampled grid
        cv2.rectangle(littered, (x0, 430), (x0 + 45, 620), (12, 12, 12), -1)
    dirty = learn_desk_reference(littered, CALIB)

    assert np.allclose(clean.centre, dirty.centre, atol=2.0)


def test_the_scan_never_reports_a_boundary_outside_the_search_rows():
    surface, background = SURFACES["dark grey slate under a cream wall"]
    scan = find_edge_pixels(frame_with_edge(surface, background), CALIB)
    assert len(scan.pixels)
    assert scan.pixels[:, 1].min() >= SEARCH_ROWS[0]
    assert scan.pixels[:, 1].max() <= SEARCH_ROWS[1]


def test_polygon_margin_moves_the_edge_toward_the_arm():
    """+x is toward the arm, so the margin must raise the measured edge x."""
    poly = polygon_from_edge(0.0, -72.2, margin_mm=25.0)
    assert poly[0][0] == pytest.approx(-47.2)
    assert poly[3][0] == pytest.approx(-47.2)
    # The three nominal sides stay far outside any reach.
    assert poly[1][0] == poly[2][0] == 500.0
