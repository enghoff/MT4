"""The instruction loop's window: decision frame, state panel, live inset.

Three panes composited by :class:`RunPreview` on its own thread, so the window
keeps moving through a multi-second arm move:

* the frame the model last decided on, with its own answer drawn over it by
  ``preview.annotate_qwen`` -- deliberately not live, because those are the
  exact pixels a decision was made from;
* a panel of :class:`RunState`: instruction, step, phase, what the jaws are
  believed to hold, what has been done;
* a live inset, so the desk stays visible while the stale main pane is up.

The first and third trade places while the hardware moves (see
:data:`MOVING_PHASES`): a carry is the one stretch where the desk is changing
and the frame with the overlay is describing a moment that has gone.

The same composited canvas is what ``video_path`` records and what ``/save``
writes, so a recording and a window show the identical picture.

``worker`` is duck-typed (it needs ``snapshot()``) and ``ui`` needs
``emit``/``set_status``, so nothing here imports the modules that import it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from mt4_vision import instruct as I
from mt4_vision.camera import CameraError, FrameStream
if TYPE_CHECKING:  # duck-typed at runtime -- see the module docstring
    from mt4_vision.console import BottomUI
    from mt4_vision.instruct_worker import PlainUI, TaskWorker

from mt4_vision.preview import (
    LivePreview,
    PreviewStopped,
    VideoRecorder,
    annotate_qwen,
    draw_inset,
    draw_move,
    draw_outlined_text,
    wrap_text,
)

MAX_STEPS = 6


# Frames per second written by ``--record``, and the rate the file declares.
# The compositing loop turns over at camera rate -- 30.7/s measured on a
# 1280x720 stream, 26.3 ms of it waiting for the next frame and 6.0 ms drawing
# -- so recording every canvas would triple the file for motion nobody is
# watching at that resolution. RunPreview writes on a clock instead of every
# tick, which is also what keeps the moving stretches at real speed.
RECORD_FPS = 10.0


# How much sparser the recording gets while the arm stands still. Waiting on
# the model is seconds of a picture that does not change, so those stretches
# are written at a tenth of the rate and play back ten times faster; motion is
# written in full. A recording is therefore real time through every move and
# fast-forward between them, which is what makes a whole session watchable.
WAIT_SPEEDUP = 10


CAPTION_BGR = (255, 255, 255)


CAPTION_DIM_BGR = (190, 190, 190)


# Panel palette and metrics.
PANEL_W = 460


PANEL_PAD = 14


BODY_SCALE = 0.45


BODY_LINE_PX = 18


HEAD_SCALE = 0.52


BG = (28, 28, 30)


RULE = (72, 72, 78)


DIM = (150, 150, 156)


TEXT = (238, 238, 240)


TASK_BGR = (0, 200, 255)      # amber -- the instruction being carried out


OK_BGR = (130, 230, 150)      # green -- what has been done


ERR_BGR = (90, 90, 250)       # red


WARN_BGR = (0, 200, 255)


# Phases during which the desk itself is changing: the worker's own labels for
# arm motion, plus the chore labels ``ask_qwen`` submits for the commands that
# drive the hardware. ``/open`` is here because the jaws letting go changes the
# desk as much as a carry does, even though the arm holds still for it. A phase
# missing from this set costs nothing but the swap.
MOVING_PHASES = frozenset({"moving", "parking", "homing", "opening the jaws"})


@dataclass(frozen=True)
class RunState:
    """What the panel draws. Replaced wholesale, never mutated in place."""

    instruction: str = ""
    step: int = 0
    max_steps: int = MAX_STEPS
    # What the worker is doing right now, in the words the panel shows:
    # idle / parking / looking / locating <noun> / deciding / moving / <chore>.
    phase: str = "idle"
    phase_started: float = 0.0
    held: str | None = None
    action: str = ""
    reason: str = ""
    # The move in flight, as pixels of the desk the camera is looking at now:
    # where the jaws close and where they open again, projected from the poses
    # the arm was actually given. Either can be absent -- a pick has no
    # destination, a place no source. Drawn on the live pane while it has the
    # space, and cleared when the motion call returns.
    move_from_px: tuple[float, float] | None = None
    move_to_px: tuple[float, float] | None = None
    # Oldest first, the same sentences the model is told about on the next
    # step -- so the panel and the prompt cannot drift apart.
    outcomes: tuple[str, ...] = ()
    queued: int = 0
    dry_run: bool = False
    stopping: str = ""
    error: str = ""

    @property
    def elapsed(self) -> float:
        return 0.0 if not self.phase_started else time.monotonic() - self.phase_started

    @property
    def arm_moving(self) -> bool:
        """Whether the hardware is in motion, and the window should show it."""
        return self.phase in MOVING_PHASES


def decision_view(
    obs: I.Observation,
    *,
    grounding=None,
    obj=None,
    action: I.Action | None = None,
    report=None,
    caption: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """What the model just answered, drawn on the frame it answered about.

    ``obs.annotated`` and not a fresh capture: the tag circles are
    what the model was looking at when it produced these coordinates, so they
    are the only backdrop against which the answer means anything.

    ``obj`` is the measurement taken from the model's box, when there is one.
    Its mask is the last look anyone gets at what is about to be grasped:
    nothing downstream re-measures, and nothing after the move checks the desk
    again. Its centroid is drawn as the *bound* point, joined to the box centre
    the model gave, so the pixel gap between "where it pointed" and "where the
    jaws will actually go" is a glance rather than an investigation.

    The destination dot comes from the reply itself, always. There is no entity
    behind it to draw instead: landing on a tag means the model echoed that
    tag's coordinates, so the dot sits on the circled tag exactly when the echo
    was right, and visibly off it when the model pointed somewhere else.

    ``report`` is a REPORT's measured findings, drawn as one numbered box per
    object in the same green/red the terminal rows read in. It is the only case
    where the answer is several things at once, and the picture is where the
    model's list and the arm's verdict on each item can be compared.
    """
    grounding = grounding if grounding is not None else (
        None if action is None else action.source
    )
    bound = None
    if obj is not None:
        px, py = getattr(obj, "px", None), getattr(obj, "py", None)
        if px is not None and py is not None:
            bound = (float(px), float(py))
    dest_bound = None if action is None else action.dest_point_px
    return annotate_qwen(
        obs.annotated,
        grounding=grounding,
        obj=obj,
        action=action,
        report=report,
        bound_px=bound,
        dest_bound_px=dest_bound,
        accepted=None if action is None else action.ok,
        caption=caption,
    )


def render_panel(
    height: int, *, state: RunState, svc: str, camera: int,
) -> np.ndarray:
    """The right-hand column: what is being done, and what has been done."""
    panel = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    x = PANEL_PAD
    body_px = PANEL_W - 2 * PANEL_PAD
    y = 26

    def rule(yy: int) -> int:
        cv2.line(panel, (x, yy), (PANEL_W - PANEL_PAD, yy), RULE, 1)
        return yy + 16

    def block(
        label: str, text: str, color: tuple[int, int, int], yy: int,
        *, limit: int | None = None,
    ) -> int:
        if label:
            draw_outlined_text(panel, label, (x, yy), scale=0.42, color=DIM)
            yy += BODY_LINE_PX
        lines = wrap_text(text, max_px=body_px, scale=BODY_SCALE, hard_break=True)
        clipped = 0
        if limit is not None and len(lines) > limit:
            clipped = len(lines) - limit
            lines = lines[:limit]
        for line in lines:
            draw_outlined_text(panel, line, (x, yy), scale=BODY_SCALE, color=color)
            yy += BODY_LINE_PX
        if clipped:
            draw_outlined_text(
                panel, f"... +{clipped} more (full text in the terminal)",
                (x, yy), scale=0.4, color=DIM,
            )
            yy += BODY_LINE_PX
        return yy + 6

    draw_outlined_text(panel, "MT4 INSTRUCTION", (x, y), scale=HEAD_SCALE, color=TEXT)
    y += BODY_LINE_PX
    draw_outlined_text(panel, svc, (x, y), scale=0.4, color=DIM)
    y += BODY_LINE_PX
    flags = f"camera {camera}  max {state.max_steps} steps"
    if state.dry_run:
        flags += "  DRY-RUN"
    draw_outlined_text(
        panel, flags, (x, y), scale=0.4, color=WARN_BGR if state.dry_run else DIM,
    )
    y += BODY_LINE_PX
    # Always drawn, empty or not: "what does the arm think it is carrying" is
    # the one piece of state nothing on this rig can sense, and a line that
    # appears only when it is non-empty is a line nobody learns to look for.
    held = f"holding: {state.held}" if state.held else "holding: nothing"
    draw_outlined_text(
        panel, held, (x, y), scale=0.4, color=OK_BGR if state.held else DIM,
    )
    y = rule(y + 10)

    if state.instruction:
        y = block("INSTRUCTION", state.instruction, TASK_BGR, y, limit=4)
        line = f"step {state.step}/{state.max_steps}  {state.phase}"
        if state.phase != "idle":
            line += f"  {state.elapsed:4.1f}s"
        draw_outlined_text(
            panel, line, (x, y), scale=BODY_SCALE,
            color=WARN_BGR if state.phase != "idle" else DIM,
        )
        y += BODY_LINE_PX + 6
    else:
        for line in wrap_text(
            "Type an instruction to carry out. /help lists the commands.",
            max_px=body_px, scale=BODY_SCALE,
        ):
            draw_outlined_text(panel, line, (x, y), scale=BODY_SCALE, color=DIM)
            y += BODY_LINE_PX
        y += 6

    if state.queued:
        draw_outlined_text(
            panel, f"{state.queued} waiting", (x, y), scale=0.4, color=DIM,
        )
        y += BODY_LINE_PX
    if state.stopping:
        draw_outlined_text(
            panel, state.stopping, (x, y), scale=BODY_SCALE, color=WARN_BGR,
        )
        y += BODY_LINE_PX

    if state.action:
        y = rule(y + 4)
        y = block("DECIDED", state.action, TASK_BGR, y, limit=4)
        if state.reason:
            y = block("", state.reason, DIM, y, limit=5)

    if state.error:
        y = rule(y)
        y = block("ERROR", state.error, ERR_BGR, y, limit=6)

    if state.outcomes:
        y = rule(y)
        draw_outlined_text(panel, "DONE SO FAR", (x, y), scale=0.42, color=DIM)
        y += BODY_LINE_PX
        # Filled newest-first and drawn oldest-first, budgeted in the WRAPPED
        # lines each outcome actually costs rather than one line each. Counting
        # outcomes instead ran the list off the bottom edge of the panel the
        # moment any of them wrapped, which is most of them -- "moved the red
        # cube onto marker 3" does not fit this column on one line.
        # One line held back for the "(+N earlier)" note, which is itself a row.
        budget = max(1, (height - y - 24) // BODY_LINE_PX - 1)
        chosen: list[list[str]] = []
        for i in range(len(state.outcomes), 0, -1):
            lines = wrap_text(
                f"{i}. {state.outcomes[i - 1]}",
                max_px=body_px, scale=BODY_SCALE, hard_break=True,
            )[:2]
            # Always show one, even if it is the only thing that fits.
            if chosen and len(lines) > budget:
                break
            chosen.append(lines)
            budget -= len(lines)
        hidden = len(state.outcomes) - len(chosen)
        if hidden > 0:
            draw_outlined_text(
                panel, f"(+{hidden} earlier)", (x, y), scale=0.4, color=DIM,
            )
            y += BODY_LINE_PX
        for lines in reversed(chosen):
            for piece in lines:
                draw_outlined_text(panel, piece, (x, y), scale=BODY_SCALE, color=OK_BGR)
                y += BODY_LINE_PX
    return panel


def compose(
    live: np.ndarray,
    view: np.ndarray | None,
    state: RunState,
    *,
    svc: str,
    camera: int,
) -> np.ndarray:
    """The main pane, whichever of the two frames it is, plus the panel.

    The two pictures trade places on ``state.arm_moving``. Standing still, the
    decision frame is large and the live feed is the corner inset: the question
    then is what the model pointed at, and that is a question about pixels a
    box was drawn on. Moving, they swap: the desk is changing, the frame with
    the overlay describes a moment that has passed, and the thing worth the
    large pane is the arm actually crossing the desk. Neither picture ever
    leaves the window, so the comparison is always available -- what swaps is
    which one is worth the space.

    The live pane carries the move itself while it is the large one -- rings on
    where the jaws close and where they open again, an arrow between. Those are
    the poses the arm was given, projected into pixels, so the arm can be
    watched against its own plan rather than against a memory of the overlay
    that is now two inches wide in the corner.
    """
    if view is None or state.arm_moving:
        main = live.copy()
        draw_outlined_text(main, "LIVE", (14, 28), scale=0.55, color=(220, 220, 220))
        draw_move(main, state.move_from_px, state.move_to_px)
        if view is not None:
            draw_inset(main, view, "DECIDED")
    else:
        main = view.copy()
        draw_inset(main, live, "LIVE")
    panel = render_panel(main.shape[0], state=state, svc=svc, camera=camera)
    return np.hstack([main, panel])


class RunPreview:
    """Composites the decision frame, the live feed and the state panel.

    Its own thread against the shared ``FrameStream``, so the window keeps
    moving through a multi-second transfer. That is the difference from
    publishing frames straight from the worker: a preview that only updates
    when the worker has something to say is frozen for exactly the seconds
    when watching the desk matters most.

    ``window`` and ``video_path`` are independent outputs of the same canvas.
    Recording without a window is the headless case -- a machine whose OpenCV
    has no GUI can still write the file, and that is where a recording is the
    only way to see what happened.
    """

    def __init__(
        self,
        stream: FrameStream,
        worker: TaskWorker,
        *,
        svc: str,
        camera: int,
        ui: BottomUI | PlainUI,
        title: str = "mt4 -- what qwen said (q or Esc to stop)",
        window: bool = True,
        video_path: str | None = None,
        fps: float = RECORD_FPS,
    ) -> None:
        self._stream = stream
        self._worker = worker
        self._svc = svc
        self._camera = camera
        self._ui = ui
        self._draw_errors: set[str] = set()
        self._preview = LivePreview(title) if window else None
        self._recorder = (
            VideoRecorder(video_path=video_path, fps=fps) if video_path else None
        )
        # The catch-up in _record divides the elapsed time by this, so a
        # non-positive rate has no meaning here; callers validate it.
        self._record_period = 1.0 / fps
        self._next_write = time.monotonic()
        self._last_period = self._record_period
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._user_quit = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="run-view", daemon=True)
        self._thread.start()

    def stopped_by_user(self) -> bool:
        return self._user_quit.is_set()

    def latest_canvas(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                live = self._stream.fresh(min_advance=1)
            except CameraError:
                time.sleep(0.2)
                continue
            state, view = self._worker.snapshot()
            try:
                canvas = compose(
                    live, view, state, svc=self._svc, camera=self._camera,
                )
            except Exception as exc:  # noqa: BLE001 -- a draw bug must not kill the feed
                # ...but it must not be invisible either. Swallowing this
                # silently is how a compose() crash presented as "the preview
                # feature just doesn't work": no window, no error, nothing to
                # search for. Report the first of each kind and keep going.
                sig = f"{type(exc).__name__}: {exc}"
                if sig not in self._draw_errors:
                    self._draw_errors.add(sig)
                    self._ui.emit(f"  preview draw failed -- {sig}")
                time.sleep(0.2)
                continue
            with self._lock:
                self._latest = canvas
            self._record(canvas, state.arm_moving)
            if self._preview is None:
                continue
            try:
                self._preview.show(canvas)
            except PreviewStopped:
                self._user_quit.set()
                # Closing the window is a stop request for the arm too. The
                # prompt thread is blocked reading a key and cannot notice on
                # its own, so say so on the status line rather than leaving a
                # transfer running behind a window that is no longer there.
                self._worker.request_stop(abort=False)
                self._ui.set_status("window closed -- stopping; press Enter to quit")
                break

    def _record(self, canvas: np.ndarray, moving: bool) -> None:
        """Write this canvas as many times as the recording clock owes it.

        Paced against wall time, not against the loop, so a move plays back at
        the speed it happened -- a move timed off the video is the move that
        happened. The loop turns over faster than the recording rate (30.7/s
        measured, against 10 written), so an ordinary tick writes one frame or
        none; a late tick -- window-redraw jitter, or the 200 ms sleep after a
        camera error -- repeats the canvas to fill the gap it left. Writing at
        most one frame per tick instead drops those periods and shortens the
        video: 7.4 s of file for an 8.0 s run, measured with the window open.

        Standing still, the rate drops by ``WAIT_SPEEDUP``, so the seconds
        spent waiting on the model pass ten times faster on playback. The new
        pace starts at the tick that changed it rather than after the deadline
        already pending, because the switch into motion is exactly where a
        whole slow period would otherwise swallow the start of a move.

        ``now`` is read once, so a write slower than the period cannot chase
        its own tail.
        """
        if self._recorder is None:
            return
        period = self._record_period * (1 if moving else WAIT_SPEEDUP)
        now = time.monotonic()
        if period != self._last_period:
            self._last_period = period
            self._next_write = min(self._next_write, now)
        while now >= self._next_write:
            try:
                self._recorder.write(canvas)
            except Exception as exc:  # noqa: BLE001 -- see the compose handler above
                # A path OpenCV will not open -- a directory that does not
                # exist, an extension with no codec -- must not take the window
                # down with it. Say so once, give up on the file, keep showing
                # the run.
                self._recorder = None
                self._ui.emit(f"  recording stopped -- {type(exc).__name__}: {exc}")
                return
            self._next_write += period

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._preview is not None:
            self._preview.close()
        if self._recorder is not None:
            self._recorder.close()
