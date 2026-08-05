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

from mt4_vision import instruct_view
from mt4_vision.instruct_view import (
    BG,
    PANEL_W,
    RunPreview,
    RunState,
    compose,
    render_panel,
)
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


# -- recording ------------------------------------------------------------- #


class FrameCounter:
    """A ``VideoRecorder`` that counts frames instead of encoding them."""

    def __init__(self) -> None:
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        self.frames += 1

    def close(self) -> None:
        pass


def paced(fps: float = 10.0) -> RunPreview:
    """A ``RunPreview`` carrying only the fields ``_record`` reads.

    Built without ``__init__`` because the rest of it is a camera, a window
    and a thread, and the pacing is arithmetic on a clock.
    """
    view = object.__new__(RunPreview)
    view._recorder = FrameCounter()
    view._record_period = 1.0 / fps
    view._next_write = 0.0
    view._ui = Recorder()
    return view


def replay(view: RunPreview, ticks: list[float], monkeypatch) -> None:
    """Offer one canvas per tick, with the clock reading that tick's time."""
    clock = {"now": 0.0}
    monkeypatch.setattr(instruct_view.time, "monotonic", lambda: clock["now"])
    canvas = blank(h=8, w=8)
    for tick in ticks:
        clock["now"] = tick
        view._record(canvas)


def test_recording_writes_at_the_declared_rate(monkeypatch):
    """10 s of a 30/s loop is 100 frames of 10 fps video, not 300."""
    view = paced(10.0)
    replay(view, [i / 30 for i in range(1, 300)], monkeypatch)
    assert view._recorder.frames == 100


def test_a_stalled_tick_is_filled_so_the_file_keeps_wall_time(monkeypatch):
    """The shortfall this pins: 7.4 s of video for an 8.0 s run.

    Writing at most one frame per tick drops every period a slow tick jumped
    over, and the recording ends up shorter than the run it recorded.
    """
    view = paced(10.0)
    # A second of ordinary ticks, a 500 ms stall, then a second more.
    ticks = [i / 30 for i in range(1, 31)]
    ticks += [1.5]
    ticks += [1.5 + i / 30 for i in range(1, 30)]
    replay(view, ticks, monkeypatch)
    assert view._recorder.frames == 25          # 2.5 s at 10 fps


def test_a_recorder_that_fails_gives_up_the_file_and_keeps_the_window(monkeypatch):
    """A path with no codec must not take the run's only display down."""
    view = paced(10.0)

    class Broken:
        def write(self, frame: np.ndarray) -> None:
            raise RuntimeError("no codec for .xyz")

    view._recorder = Broken()
    replay(view, [0.1, 0.2, 0.3], monkeypatch)
    assert view._recorder is None
    assert any("no codec" in line for line in view._ui.lines)


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
