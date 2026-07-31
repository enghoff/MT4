"""Unit tests for the motion gate in ask_qwen.py (no camera, no service).

The gate decides when the model is asked at all, so its threshold has to sit
between two measured numbers: the desk camera's noise floor (max 0.00016 over
60 static frames) and the smallest event worth catching (a 25x25px object
moving, ~0.0014). These tests pin that gap.
"""

from __future__ import annotations

import numpy as np

from ask_qwen import (
    MOTION_THRESHOLD,
    _prep,
    changed_boxes,
    montage,
    motion_score,
)

H, W = 720, 1280


def blank() -> np.ndarray:
    return np.full((H, W, 3), 60, dtype=np.uint8)


def with_square(x: int, y: int, side: int = 40, value: int = 240) -> np.ndarray:
    img = blank()
    img[y:y + side, x:x + side] = value
    return img


# -- scoring --------------------------------------------------------------- #


def test_identical_frames_score_zero() -> None:
    a = _prep(blank())
    assert motion_score(a, a) == 0.0


def test_threshold_sits_above_sensor_noise() -> None:
    """Simulated noise at the measured amplitude must not trip the gate."""
    rng = np.random.default_rng(0)
    base = blank()
    noisy = np.clip(
        base.astype(np.int16) + rng.integers(-6, 7, base.shape), 0, 255
    ).astype(np.uint8)
    assert motion_score(_prep(base), _prep(noisy)) < MOTION_THRESHOLD


def test_threshold_sits_below_a_small_object_moving() -> None:
    """A 40px object relocating is ~11x the noise floor and must trip it."""
    score = motion_score(_prep(with_square(100, 100)), _prep(with_square(400, 400)))
    assert score > MOTION_THRESHOLD
    # Both the vacated and the occupied patch change, so ~2x one footprint.
    assert score > 2 * (40 * 40) / (H * W) * 0.5


def test_score_grows_with_the_size_of_the_change() -> None:
    small = motion_score(_prep(blank()), _prep(with_square(200, 200, side=40)))
    large = motion_score(_prep(blank()), _prep(with_square(200, 200, side=200)))
    assert large > small * 10


def test_a_uniform_brightness_shift_below_delta_is_ignored() -> None:
    """Gentle auto-exposure drift must not read as motion."""
    base = blank()
    lifted = np.clip(base.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    assert motion_score(_prep(base), _prep(lifted)) == 0.0


# -- changed regions ------------------------------------------------------- #


def test_changed_boxes_locates_a_single_change() -> None:
    (box,) = changed_boxes(blank(), with_square(300, 200, side=80))
    x1, y1, x2, y2 = box
    assert x1 <= 300 <= x2 and y1 <= 200 <= y2
    assert x1 <= 380 <= x2 + 40  # dilation widens the box a little


def test_changed_boxes_finds_both_ends_of_a_move() -> None:
    boxes = changed_boxes(with_square(80, 80), with_square(700, 500))
    assert len(boxes) == 2


def test_changed_boxes_empty_when_nothing_changed() -> None:
    assert changed_boxes(blank(), blank()) == []


def test_changed_boxes_are_largest_first_and_capped() -> None:
    before = blank()
    after = blank()
    for i, side in enumerate((20, 200, 60, 90, 30)):
        after[50 + i * 120:50 + i * 120 + side, 50 + i * 200:50 + i * 200 + side] = 240
    boxes = changed_boxes(before, after, max_boxes=3)
    assert len(boxes) == 3
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    assert areas == sorted(areas, reverse=True)


# -- montage (display tiling, also used by montage send mode) -------------- #


def test_montage_of_one_frame_is_that_frame() -> None:
    f = with_square(10, 10)
    assert montage([f]) is f


def test_montage_tiles_without_losing_frames() -> None:
    frames = [with_square(10 + 30 * i, 10) for i in range(4)]
    out = montage(frames)
    # 4 frames -> 2x2 of half-size tiles, so the canvas keeps the frame size.
    assert out.shape[0] == (H // 2) * 2
    assert out.shape[1] == (W // 2) * 2
