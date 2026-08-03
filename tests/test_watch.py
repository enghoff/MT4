"""Unit tests for the motion gate behind ask_qwen.py (no camera, no service).

The gate decides when the model is asked at all, so its threshold has to sit
between two measured numbers: the desk camera's noise floor (max 0.00016 over
60 static frames) and the smallest event worth catching (a 25x25px object
moving, ~0.0014). These tests pin that gap.
"""

from __future__ import annotations

import threading

import numpy as np

from mt4_vision.qwen_panel import Options, WatchState, compose, montage
from mt4_vision.qwen_watch import (
    MOTION_THRESHOLD,
    MotionWatcher,
    _prep,
    changed_boxes,
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


# -- the panel draws what the watcher reports ------------------------------- #
#
# compose() runs on the preview thread behind a catch-all draw guard, so
# anything it raises costs the whole window rather than one frame. Armed is the
# default, so these run the default path -- a fixed-arity unpack of the watcher
# snapshot broke exactly this and the window silently never opened.


def armed_snapshot(**over: object) -> WatchState:
    """A real MotionWatcher snapshot, not a hand-written stand-in.

    Building the tuple by hand here would have kept passing while the watcher
    grew a field, which is the drift that caused the outage.
    """
    watcher = MotionWatcher.__new__(MotionWatcher)  # no camera, no threads
    watcher._lock = threading.Lock()
    watcher._state = "quiet"
    watcher._score = 0.0
    watcher._events = 0
    watcher._skipped = 0
    watcher._boxes = []
    watcher._send = "pair"
    for name, value in over.items():
        setattr(watcher, f"_{name}", value)
    return watcher.snapshot()


def compose_with(watch: WatchState | None) -> np.ndarray:
    return compose(
        blank(), answer=None, pending=None, elapsed=0.0,
        opts=Options(), svc="probe", watch=watch,
    )


def test_panel_draws_while_watching() -> None:
    canvas = compose_with(armed_snapshot())
    assert canvas.shape[0] == H
    assert canvas.shape[1] > W  # frame + panel


def test_panel_draws_for_every_watcher_state() -> None:
    for state in ("off", "quiet", "moving"):
        for send in ("pair", "latest"):
            assert compose_with(armed_snapshot(state=state, send=send)) is not None


def test_panel_draws_with_events_and_boxes() -> None:
    snap = armed_snapshot(
        state="moving", score=0.0123, events=3, skipped=1,
        boxes=[(10, 10, 90, 90), (200, 100, 260, 180)],
    )
    assert compose_with(snap) is not None


def test_panel_draws_with_no_watcher_at_all() -> None:
    assert compose_with(None) is not None


def test_watch_snapshot_is_read_by_name() -> None:
    """Guards the drift directly: fields are addressable, order is not load-bearing."""
    snap = armed_snapshot(state="moving", send="latest")
    assert snap.state == "moving"
    assert snap.send == "latest"
    assert snap.boxes == []
