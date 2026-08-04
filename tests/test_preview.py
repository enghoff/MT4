"""Annotated preview + recording (no hardware, no camera)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from mt4_vision.detect import CubeDetection, MarkerDetection
from mt4_vision.preview import (
    QWEN_BOUND_BGR,
    QWEN_MASK_BGR,
    QWEN_REFUSED_BGR,
    VideoRecorder,
    annotate_qwen,
    annotate_scene,
    clipped_text,
    draw_mask,
    wrap_text,
)
from mt4_vision.scene import Scene
from mt4_vision.workspace import MarkerSlot, rebuild_workspace_state
from rig import CALIB as RIG_CALIB


def cube(color: str, px: float, py: float, x: float, y: float) -> CubeDetection:
    return CubeDetection(color=color, px=px, py=py, area=450.0, x=x, y=y)


def make_scene() -> Scene:
    markers = [MarkerSlot(0, 100.0, 0.0), MarkerSlot(1, 200.0, 100.0)]
    cubes = [cube("red", 300.0, 200.0, 100.0, 0.0)]
    state = rebuild_workspace_state(
        RIG_CALIB, markers, cubes, visible_marker_ids={0, 1}
    )
    return Scene.from_workspace(state, raw_cubes=cubes, calib=RIG_CALIB)


def test_annotate_scene_does_not_mutate_input_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    scene = make_scene()
    markers_px = [MarkerDetection(0, 10.0, 10.0), MarkerDetection(1, 50.0, 50.0)]

    out = annotate_scene(
        frame, scene, markers_px, status_lines=["hello"],
    )

    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    assert np.array_equal(frame, np.zeros((240, 320, 3), dtype=np.uint8))


def test_annotate_scene_highlights_target():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    scene = make_scene()
    target = scene.cubes[0]

    with_target = annotate_scene(frame, scene, [], target=target)
    without_target = annotate_scene(frame, scene, [])

    assert not np.array_equal(with_target, without_target)


# ------------------------------------------------------------- annotate_qwen


class _Grounding:
    def __init__(self, **kw):
        self.label = kw.get("label", "stapler")
        self.point_px = kw.get("point_px")
        self.box_px = kw.get("box_px")
        self.alt_point_px = kw.get("alt_point_px")
        self.alt_box_px = kw.get("alt_box_px")


class _Action:
    def __init__(self, point_px=None, ok=True, dest_point_px=None):
        self.point_px = point_px
        self.ok = ok
        self.dest_point_px = dest_point_px


def _blank(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_annotate_qwen_never_touches_the_frame_it_was_given():
    frame = _blank()
    out = annotate_qwen(
        frame,
        grounding=_Grounding(point_px=(160.0, 120.0), box_px=(140, 100, 180, 140)),
        action=_Action((150.0, 110.0)),
        bound_px=(200.0, 130.0),
        caption=[("hello", (255, 255, 255))],
    )
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    assert np.array_equal(frame, _blank())


def test_annotate_qwen_with_nothing_to_draw_is_a_plain_copy():
    """A step before any model call still has to publish a frame."""
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    assert np.array_equal(annotate_qwen(frame), frame)


def test_the_alternate_coordinate_reading_is_drawn_separately():
    """The one diagnostic that made the coordinate-space bug obvious.

    Both readings of the same reply have to appear, in different colours, or
    the picture cannot show which one the stack believed.
    """
    frame = _blank()
    primary = _Grounding(point_px=(160.0, 120.0), box_px=(140, 100, 180, 140))
    with_alt = annotate_qwen(
        frame,
        grounding=_Grounding(
            point_px=(160.0, 120.0), box_px=(140, 100, 180, 140),
            alt_point_px=(40.0, 30.0), alt_box_px=(20, 20, 60, 40),
        ),
    )
    assert not np.array_equal(with_alt, annotate_qwen(frame, grounding=primary))


def test_a_refused_action_rings_the_bound_entity_in_red():
    frame = _blank()
    ok = annotate_qwen(frame, action=_Action((100.0, 100.0), ok=True), bound_px=(200.0, 150.0), accepted=True)
    no = annotate_qwen(frame, action=_Action((100.0, 100.0), ok=False), bound_px=(200.0, 150.0), accepted=False)
    assert not np.array_equal(ok, no)
    # The ring is where bound_px says, so sample the colour actually laid down.
    assert tuple(int(v) for v in no[150, 180]) == QWEN_REFUSED_BGR
    assert tuple(int(v) for v in ok[150, 180]) == QWEN_BOUND_BGR


def test_a_transfer_draws_both_ends_and_an_arrow_between_them():
    """A transfer is carried out without another look at the desk, so this
    picture is the only warning that the arrow points at the wrong marker."""
    frame = _blank()
    act = _Action((100.0, 100.0), ok=True, dest_point_px=(250.0, 200.0))
    one_end = annotate_qwen(
        frame, action=act, bound_px=(110.0, 105.0), accepted=True
    )
    both = annotate_qwen(
        frame, action=act, bound_px=(110.0, 105.0),
        dest_bound_px=(240.0, 195.0), accepted=True,
    )
    assert not np.array_equal(both, one_end)
    # The destination ring, in the same colour the accepted source ring gets.
    assert tuple(int(v) for v in both[195, 220]) == QWEN_BOUND_BGR
    # And the arrow crosses the space between the two rings, which the
    # single-ended drawing leaves untouched.
    mid = (int((105 + 195) / 2), int((110 + 240) / 2))
    assert both[mid].any() and not one_end[mid].any()


def test_a_refused_transfer_rings_both_ends_in_red():
    frame = _blank()
    act = _Action((100.0, 100.0), ok=False, dest_point_px=(250.0, 200.0))
    out = annotate_qwen(
        frame, action=act, bound_px=(110.0, 105.0),
        dest_bound_px=(240.0, 195.0), accepted=False,
    )
    assert tuple(int(v) for v in out[195, 220]) == QWEN_REFUSED_BGR


def test_a_grounding_box_bigger_than_the_frame_still_draws():
    """The whole-frame non-answer must be visible, not crash the overlay.

    Measured live: asked to locate a "location", the model returned the entire
    image as a box. That is exactly the case someone opens the preview to see.
    """
    frame = _blank()
    out = annotate_qwen(
        frame,
        grounding=_Grounding(label="location", point_px=(160.0, 120.0),
                             box_px=(0.0, 0.0, 320.0, 240.0)),
    )
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_a_caption_longer_than_the_frame_wraps_instead_of_running_off():
    reason = (
        "the task rules out placing on a marker and marker_0 (marker 0 (free)) "
        "is one, so the name cannot stand; the point could not stand in for it "
        "either -- the point (936, 420) is 91px from the nearest entity slot_5"
    )
    lines = wrap_text(reason, max_px=304, scale=0.5)
    assert len(lines) > 1
    assert " ".join(lines) == reason
    # And the band it needs must fit inside the frame it is drawn on.
    out = annotate_qwen(_blank(), caption=[(reason, (255, 255, 255))])
    assert out.shape == (240, 320, 3)


def test_wrap_text_keeps_a_single_unbreakable_word():
    assert wrap_text("supercalifragilistic", max_px=10, scale=0.5) == [
        "supercalifragilistic"
    ]


def test_clipped_text_ends_where_the_room_does_and_says_it_was_cut():
    """One line with a hard edge -- a label beside a box has the room between
    that box and the frame's edge and no more. Silently truncating a gate's
    reason would read as the whole reason."""
    reason = "r=123mm is inside the 140mm J1 keep-out (firmware mp refuses any target there)"
    assert clipped_text(reason, max_px=4000, scale=0.42) == reason
    cut = clipped_text(reason, max_px=200, scale=0.42)
    assert cut.endswith("...")
    assert reason.startswith(cut[:-3])
    assert cv2.getTextSize(cut, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] <= 200


# ---------------------------------------------------------------- draw_report


class _Found:
    """An instruct.Finding as far as the overlay is concerned."""

    def __init__(self, index, label, box_px, pickable, note):
        self.index, self.label = index, label
        self.box_px, self.pickable, self.note = box_px, pickable, note


def test_a_report_draws_every_object_and_splits_them_by_the_pick_verdict():
    """The split inside the list is the answer. Which things the arm can take
    comes from reach, the keep-out, the desk edge and the jaw width -- none of
    which is in the photograph -- so the picture is the only place the model's
    list and the arm's verdict can be compared."""
    frame = _blank()
    out = annotate_qwen(
        frame,
        report=[
            _Found(1, "red cube", (40.0, 40.0, 80.0, 80.0), True, "pickable"),
            _Found(2, "stapler", (160.0, 40.0, 200.0, 80.0), False, "r=367mm is beyond reach"),
        ],
    )
    assert np.array_equal(frame, _blank())
    # Sampled on each rectangle's own top edge, so this reads the colour laid
    # down rather than a colour that happens to be somewhere in the frame.
    assert tuple(int(v) for v in out[40, 60]) == QWEN_BOUND_BGR
    assert tuple(int(v) for v in out[40, 180]) == QWEN_REFUSED_BGR


def test_a_report_row_that_could_not_be_measured_is_still_drawn():
    """A box that named something the segmenter could not cut is still an
    object the model saw. Dropping it would shorten the answer."""
    out = annotate_qwen(
        _blank(),
        report=[_Found(1, "smudge", (40.0, 40.0, 80.0, 80.0), False, "no foreground in that box")],
    )
    assert tuple(int(v) for v in out[40, 60]) == QWEN_REFUSED_BGR


def test_a_report_label_at_the_right_edge_stays_inside_the_frame():
    """A box near the edge has almost no room for its reason, and the reason is
    a whole sentence. It is clipped there, not run off the side."""
    out = annotate_qwen(
        _blank(),
        report=[
            _Found(
                1, "chip", (300.0, 100.0, 316.0, 116.0), False,
                "r=123mm is inside the 140mm J1 keep-out (firmware mp refuses any "
                "target there)",
            )
        ],
    )
    assert out.shape == (240, 320, 3)


# ----------------------------------------------------------------- draw_mask


class _Obj:
    """A LocatedObject as far as the overlay is concerned."""

    def __init__(self, mask, origin, *, px=50.0, py=50.0, long_mm=90.0, short_mm=41.0):
        self.mask = mask
        self.mask_origin_px = origin
        self.px, self.py = px, py
        self.long_mm, self.short_mm = long_mm, short_mm


def _square_mask(n=20, fill=None):
    m = np.zeros((n, n), dtype=np.uint8)
    m[: fill or n // 2] = 255
    return m


def test_draw_mask_tints_the_masked_pixels_and_reports_its_fill_share():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    share = draw_mask(img, _square_mask(20), (100, 100), QWEN_MASK_BGR)
    assert share == 0.5
    # Inside the mask is tinted, outside is untouched.
    assert img[105, 105].any()
    assert not img[125, 105].any()


def test_draw_mask_clips_a_window_that_hangs_off_the_frame():
    """A detector box near the edge gets padded past it, mask and all."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    for origin in [(-10, -10), (310, 230), (0, 235), (-30, 100)]:
        share = draw_mask(img, _square_mask(20, fill=20), origin, QWEN_MASK_BGR)
        assert 0.0 <= share <= 1.0


def test_draw_mask_entirely_outside_the_frame_draws_nothing():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    assert draw_mask(img, _square_mask(20, fill=20), (500, 500), QWEN_MASK_BGR) == 0.0
    assert not img.any()


def test_draw_mask_tolerates_an_absent_or_empty_mask():
    """``LocatedObject.mask`` defaults to a (0, 0) array on the fallback paths."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    assert draw_mask(img, None, (0, 0), QWEN_MASK_BGR) == 0.0
    assert draw_mask(img, np.zeros((0, 0), np.uint8), (0, 0), QWEN_MASK_BGR) == 0.0
    assert draw_mask(img, np.zeros((20, 20), np.uint8), (0, 0), QWEN_MASK_BGR) == 0.0
    assert not img.any()


def test_annotate_qwen_draws_the_mask_under_the_grounding_box():
    """The box edge must survive the fill it produced."""
    frame = _blank()
    box = (100, 100, 120, 120)
    obj = _Obj(_square_mask(20, fill=20), (100, 100), px=110.0, py=110.0)
    out = annotate_qwen(
        frame, grounding=_Grounding(point_px=(110.0, 110.0), box_px=box), obj=obj
    )
    from mt4_vision.preview import QWEN_BOX_BGR

    # Top edge of the box, mid-span: still box yellow, not mask green.
    assert tuple(int(v) for v in out[100, 110]) == QWEN_BOX_BGR


def test_video_recorder_writes_video(tmp_path):
    video_path = tmp_path / "run.avi"
    recorder = VideoRecorder(video_path=str(video_path), fps=2.0)
    frame = np.full((64, 64, 3), 200, dtype=np.uint8)

    recorder.write(frame)
    recorder.write(frame)
    recorder.close()

    assert video_path.exists()
    assert video_path.stat().st_size > 0
