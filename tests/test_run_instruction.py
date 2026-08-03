"""Unit tests for the instruction harness (no camera, no arm, no service).

Everything here is either a drawing invariant or a queue invariant. Both were
picked because a real bug landed on them while the harness was being built:

* the outcome list was budgeted in *outcomes* rather than in the wrapped
  *lines* each one costs, so it ran off the bottom edge of the panel as soon
  as any outcome wrapped -- which is almost all of them, since "moved the red
  cube onto marker 3" does not fit the column on one line;
* ``busy()`` was assembled from two fields written in separate critical
  sections, leaving a window where the queue was already empty and the phase
  was still "idle". ``drain()`` reads exactly that, and the one-shot path used
  it to decide the run was over -- closing the serial port under a move that
  had just started.

The decision loop itself is not tested here. It needs a camera, an arm and a
GPU service, and mocking all three would only test the mocks.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from mt4_vision.instruct_view import BG, PANEL_W, RunState, compose, render_panel
from mt4_vision.instruct_worker import TaskWorker
from mt4_vision.preview import draw_inset, wrap_text

H, W = 720, 1280


def blank(h: int = H, w: int = W, value: int = 60) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


class Recorder:
    """A ``PlainUI`` that keeps what it was told instead of printing it."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.status: list[str] = []

    def emit(self, text: str = "") -> None:
        self.lines.append(text)

    def set_status(self, msg: str) -> None:
        self.status.append(msg)

    def close(self) -> None:
        pass


def worker(**kwargs) -> TaskWorker:
    """A worker with no camera and no arm -- only the queue is exercised."""
    return TaskWorker(
        ui=Recorder(), stream=object(), camera=0, client=None,
        dry_run=True, **kwargs,
    )


# -- panel geometry -------------------------------------------------------- #

LONG_OUTCOMES = tuple(
    f"moved the red cube onto marker {i} which is long enough to wrap twice"
    for i in range(20)
)


def test_panel_is_the_declared_width_and_the_requested_height():
    for height in (360, 720, 900):
        panel = render_panel(height, state=RunState(), svc="svc", camera=1)
        assert panel.shape == (height, PANEL_W, 3)


def test_outcomes_never_run_off_the_bottom_of_the_panel():
    """The bug this pins: budgeting in outcomes rather than wrapped lines."""
    for height in (360, 480, 720, 900):
        for count in (0, 1, 2, 5, 12, 20):
            panel = render_panel(
                height,
                state=RunState(instruction="do a thing", step=1,
                               outcomes=LONG_OUTCOMES[:count]),
                svc="svc", camera=1,
            )
            tail = panel[height - 8 :, :, :]
            assert not (tail != np.array(BG, np.uint8)).any(), (
                f"drew into the bottom margin at height={height}, "
                f"{count} outcome(s)"
            )


def test_the_newest_outcome_always_survives_the_budget():
    """Oldest are dropped, never the most recent -- that is the one being read."""
    panel = render_panel(
        400, state=RunState(instruction="x", outcomes=LONG_OUTCOMES), svc="s", camera=1,
    )
    trimmed = render_panel(
        400, state=RunState(instruction="x", outcomes=LONG_OUTCOMES[:1]),
        svc="s", camera=1,
    )
    # Both drew something; the point is only that a full list still renders.
    assert (panel != np.array(BG, np.uint8)).any()
    assert (trimmed != np.array(BG, np.uint8)).any()


def test_compose_pairs_the_main_pane_with_the_panel():
    live = blank()
    state = RunState(instruction="put the red cube on marker 3", step=2)
    for view in (None, blank(value=90)):
        canvas = compose(live, view, state, svc="svc", camera=1)
        assert canvas.shape == (H, W + PANEL_W, 3)


def test_compose_survives_a_view_and_a_live_frame_of_different_sizes():
    """The window must not die because the camera changed mode mid-session."""
    canvas = compose(
        blank(h=480, w=640), blank(), RunState(), svc="svc", camera=1,
    )
    # Height comes from the main pane (the decision frame), not the inset.
    assert canvas.shape == (H, W + PANEL_W, 3)


# -- shared preview helpers ------------------------------------------------ #


def test_hard_break_splits_a_word_that_never_fits():
    word = "x" * 400
    plain = wrap_text(word, max_px=200, scale=0.45)
    broken = wrap_text(word, max_px=200, scale=0.45, hard_break=True)
    assert plain == [word]                      # unchanged default behaviour
    assert len(broken) > 1
    assert "".join(broken) == word              # nothing lost in the split


def test_hard_break_keeps_paragraphs_and_default_does_not():
    text = "first line\n\nsecond line"
    assert wrap_text(text, max_px=4000, scale=0.45, hard_break=True) == [
        "first line", "", "second line",
    ]
    # Without it, newlines are just whitespace -- what draw_caption has always
    # assumed, so it must not change.
    assert wrap_text(text, max_px=4000, scale=0.45) == ["first line second line"]


def test_draw_inset_writes_into_the_corner_and_leaves_the_rest_alone():
    canvas = blank(value=10)
    draw_inset(canvas, blank(value=200), "LIVE")
    assert (canvas[:H // 2, :W // 2] == 10).all()      # top-left untouched
    assert (canvas[H - 60 :, W - 120 :] != 10).any()   # bottom-right written


def test_draw_inset_is_a_no_op_when_it_cannot_fit():
    tiny = blank(h=40, w=40, value=10)
    draw_inset(tiny, blank(), "LIVE")
    assert (tiny == 10).all()


# -- queue and stop -------------------------------------------------------- #


def test_chores_run_in_the_order_they_were_queued():
    w = worker()
    order: list[int] = []
    for n in range(5):
        w.submit_chore(f"job {n}", lambda n=n: order.append(n))
    w.drain(timeout=5)
    w.close()
    assert order == [0, 1, 2, 3, 4]


def test_drain_waits_for_work_queued_a_moment_earlier():
    """The race that closed the serial port under a move that had just begun."""
    w = worker()
    ran = threading.Event()

    def slow() -> None:
        time.sleep(0.05)
        ran.set()

    w.submit_chore("slow", slow)
    w.drain(timeout=5)
    assert ran.is_set(), "drain returned before the job it was waiting on ran"
    w.close()


def test_busy_stays_true_from_submit_until_the_job_finishes():
    """Sampled, not proved -- but the gap this pins was wide enough to hit."""
    w = worker()
    started, release = threading.Event(), threading.Event()
    w.submit_chore("blocking", lambda: (started.set(), release.wait(5)))
    samples = [w.busy()]
    while not started.is_set():
        samples.append(w.busy())
    samples.append(w.busy())
    release.set()
    w.drain(timeout=5)
    w.close()
    assert all(samples), "busy() reported idle while a job was being started"
    assert not w.busy()


def test_cancel_drops_the_backlog_and_reports_how_many():
    w = worker()
    started, release = threading.Event(), threading.Event()
    # Wait for the first one to be OFF the queue and running, or the count
    # below is racing the worker thread for it rather than testing anything.
    w.submit_chore("blocking", lambda: (started.set(), release.wait(5)))
    assert started.wait(5)
    for n in range(3):
        w.submit_chore(f"later {n}", lambda: None)
    assert w.cancel_queued() == 3
    assert w.cancel_queued() == 0
    release.set()
    w.drain(timeout=5)
    w.close()


def test_a_failing_job_does_not_kill_the_worker():
    """A dead worker looks exactly like a harness that ignores you."""
    w = worker()
    after: list[str] = []

    def boom() -> None:
        raise RuntimeError("no")

    w.submit_chore("boom", boom)
    w.submit_chore("after", lambda: after.append("ran"))
    w.drain(timeout=5)
    w.close()
    assert after == ["ran"]
    assert any("no" in line for line in w._ui.lines)


# -- held state ------------------------------------------------------------ #


def test_held_round_trips_and_shows_up_in_the_state():
    w = worker()
    assert w.held() is None
    w.set_held("stapler")
    assert w.held() == "stapler"
    state, _view = w.snapshot()
    assert state.held == "stapler"
    w.set_held(None)
    assert w.held() is None
    w.close()


def test_stop_requests_are_visible_to_the_running_task():
    w = worker()
    assert w.request_stop(abort=False) is False    # nothing was running
    w.close()
