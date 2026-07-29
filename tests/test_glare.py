"""Specular-glare rejection (no hardware, no camera).

The failure this guards against: a laminated ArUco pad reflecting the room
lights throws a highlight whose faintly-tinted rim clips the blue band, and
the resulting blob passes every geometric gate there is -- it is cube-sized,
square, reachable, and sits inside the marker hull by definition. Measured
across 377 recorded episodes, 28 picks were dispatched at one; all were blue
and all ended ``grasp_failed`` or ``lost``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_vision.detect import CubeDetection
from mt4_vision.scene import (
    MIN_CUBE_SATURATION,
    is_glare_blob,
    is_on_visible_marker,
)

# A marker pad's outline in pixels, detector corner order.
QUAD = [[400.0, 300.0], [460.0, 300.0], [460.0, 360.0], [400.0, 360.0]]


def blob(px: float, py: float, *, sat: float = 240.0, color: str = "blue") -> CubeDetection:
    return CubeDetection(color=color, px=px, py=py, area=2200.0, x=180.0, y=90.0, sat=sat)


# ---------------------------------------------------------------------------
# geometric gate
# ---------------------------------------------------------------------------


def test_blob_inside_a_decoded_marker_outline_is_glare():
    """A cube standing on a marker occludes its pattern, so a marker that
    decoded this frame provably has nothing on it."""
    assert is_on_visible_marker(blob(430.0, 330.0), [QUAD])
    assert is_glare_blob(blob(430.0, 330.0), [QUAD])


def test_blob_beside_the_marker_is_not_glare():
    assert not is_on_visible_marker(blob(500.0, 330.0), [QUAD])
    assert not is_glare_blob(blob(500.0, 330.0), [QUAD])


def test_marker_edge_is_inside_but_just_outside_is_not():
    assert is_on_visible_marker(blob(460.0, 330.0), [QUAD])
    assert not is_on_visible_marker(blob(460.5, 330.0), [QUAD])


def test_no_visible_markers_means_the_geometric_gate_abstains():
    """When no marker decodes, every pad may have a cube on it -- the gate
    must not then reject blobs for lack of evidence."""
    assert not is_on_visible_marker(blob(430.0, 330.0), [])
    assert not is_glare_blob(blob(430.0, 330.0), [])


# ---------------------------------------------------------------------------
# photometric gate
# ---------------------------------------------------------------------------


def test_desaturated_core_is_glare_even_away_from_any_marker():
    """Highlights also land on the white paper beside the pads, where the
    quad test cannot see them."""
    assert is_glare_blob(blob(900.0, 500.0, sat=37.0), [QUAD])


def test_saturated_blob_survives_both_gates():
    assert not is_glare_blob(blob(900.0, 500.0, sat=230.0), [QUAD])


def test_threshold_clears_the_dimmest_colour_actually_picked():
    """Green is the least saturated cube in this set; verified-placed green
    picks bottom out at S=100 while on-marker glare medians 37. The floor
    must sit between, with margin under green."""
    assert MIN_CUBE_SATURATION < 100.0
    assert not is_glare_blob(blob(900.0, 500.0, sat=100.0, color="green"), [])
    assert is_glare_blob(blob(900.0, 500.0, sat=37.0, color="green"), [])


def test_hand_built_detections_default_to_saturated():
    """Fixtures across the suite construct CubeDetection without a sat, and
    must not silently become glare."""
    c = CubeDetection(color="red", px=10.0, py=10.0, area=2000.0, x=200.0, y=0.0)
    assert not is_glare_blob(c, [])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
