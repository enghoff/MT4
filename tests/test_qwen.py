"""Unit tests for the Qwen3-VL client and its coordinate parsing (no network)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mt4_vision.qwen import (
    QwenError,
    Region,
    Reply,
    ask,
    generate,
    health,
    parse_regions,
)


def _resp(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# -- transport ------------------------------------------------------------- #


def test_health_ok() -> None:
    payload = {"ok": True, "model": "qwen", "loaded": True}
    with patch("urllib.request.urlopen", return_value=_resp(payload)) as open_:
        assert health("http://example") == payload
        assert open_.call_args[0][0] == "http://example/health"


def test_health_unreachable_names_the_tunnel_script() -> None:
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(QwenError, match="start_tunnel"):
            health("http://example")


def test_generate_with_image_posts_multipart() -> None:
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    with patch(
        "urllib.request.urlopen", return_value=_resp({"ok": True, "response": "a pen"})
    ) as open_:
        assert generate("what is this?", frame, max_new_tokens=32) == "a pen"
    req = open_.call_args[0][0]
    assert req.method == "POST"
    assert "multipart/form-data" in req.headers["Content-type"]
    body = req.data
    assert b'name="prompt"' in body
    assert b"what is this?" in body
    assert b'name="max_new_tokens"' in body and b"32" in body
    assert b'name="image"' in body
    assert body.rstrip().endswith(b"------mt4qwen--")


def test_generate_without_image_omits_the_part() -> None:
    with patch(
        "urllib.request.urlopen", return_value=_resp({"ok": True, "response": "hi"})
    ) as open_:
        assert generate("hello") == "hi"
    assert b'name="image"' not in open_.call_args[0][0].data


def test_generate_rejects_empty_prompt() -> None:
    with pytest.raises(QwenError, match="empty prompt"):
        generate("   ")


# -- multi-frame and video framing ---------------------------------------- #


def _frames(n: int) -> list[np.ndarray]:
    return [np.full((16, 24, 3), i * 10, dtype=np.uint8) for i in range(n)]


def test_several_frames_post_repeated_image_fields() -> None:
    with patch(
        "urllib.request.urlopen",
        return_value=_resp({"ok": True, "response": "x", "mode": "images",
                            "frames_sent": 3, "images_encoded": 3}),
    ) as open_:
        r = ask("q", _frames(3))
    body = open_.call_args[0][0].data
    assert body.count(b'name="image"') == 3
    assert b'name="video"' not in body
    assert b'name="fps"' not in body  # fps is meaningless for images
    assert r.mode == "images" and r.frames_sent == 3


def test_video_mode_posts_video_fields_and_fps() -> None:
    with patch(
        "urllib.request.urlopen",
        return_value=_resp({"ok": True, "response": "x", "mode": "video",
                            "frames_sent": 4, "temporal_groups": 2,
                            "timestamps_s": [0.2, 1.2]}),
    ) as open_:
        r = ask("q", _frames(4), mode="video", fps=2.0)
    body = open_.call_args[0][0].data
    assert body.count(b'name="video"') == 4
    assert b'name="image"' not in body
    assert b'name="fps"' in body and b"2.0" in body
    assert r.temporal_groups == 2 and r.timestamps_s == (0.2, 1.2)


def test_one_frame_in_video_mode_falls_back_to_image() -> None:
    """A single frame is not a sequence; the service would reject video."""
    with patch(
        "urllib.request.urlopen",
        return_value=_resp({"ok": True, "response": "x", "mode": "image"}),
    ) as open_:
        ask("q", _frames(1), mode="video")
    body = open_.call_args[0][0].data
    assert b'name="image"' in body and b'name="video"' not in body


def test_bare_single_frame_is_accepted() -> None:
    with patch(
        "urllib.request.urlopen", return_value=_resp({"ok": True, "response": "x"})
    ) as open_:
        ask("q", np.zeros((8, 8, 3), dtype=np.uint8))
    assert open_.call_args[0][0].data.count(b'name="image"') == 1


def test_greedy_and_temperature_are_only_sent_when_asked() -> None:
    with patch(
        "urllib.request.urlopen", return_value=_resp({"ok": True, "response": "x"})
    ) as open_:
        ask("q", None)
    assert b'name="do_sample"' not in open_.call_args[0][0].data

    with patch(
        "urllib.request.urlopen", return_value=_resp({"ok": True, "response": "x"})
    ) as open_:
        ask("q", None, do_sample=False)
    body = open_.call_args[0][0].data
    assert b'name="do_sample"' in body and b"false" in body


def test_ask_rejects_unknown_mode() -> None:
    with pytest.raises(QwenError, match="mode must be"):
        ask("q", _frames(2), mode="film")


# -- the frame-drop guard -------------------------------------------------- #


def test_frame_warning_catches_dropped_video_frames() -> None:
    """The failure this catches: 6 frames in, 2 temporal groups out."""
    dropped = Reply(text="fluent but wrong", mode="video", frames_sent=6,
                    temporal_groups=2)
    warning = dropped.frame_warning()
    assert warning is not None and "dropped" in warning
    assert "expected 3" in warning


def test_frame_warning_silent_when_every_frame_landed() -> None:
    assert Reply(text="x", mode="video", frames_sent=6, temporal_groups=3).frame_warning() is None
    assert Reply(text="x", mode="video", frames_sent=5, temporal_groups=3).frame_warning() is None
    assert Reply(text="x", mode="images", frames_sent=4, images_encoded=4).frame_warning() is None


def test_frame_warning_catches_dropped_images() -> None:
    warning = Reply(text="x", mode="images", frames_sent=6, images_encoded=4).frame_warning()
    assert warning is not None and "4 of 6" in warning


def test_frame_warning_ignores_text_only_replies() -> None:
    assert Reply(text="x", mode="text").frame_warning() is None


# -- region parsing -------------------------------------------------------- #


def test_parse_bbox_json_array() -> None:
    text = '[{"bbox_2d": [10, 20, 30, 40], "label": "pen"}]'
    (r,) = parse_regions(text)
    assert (r.kind, r.label, r.coords) == ("box", "pen", (10.0, 20.0, 30.0, 40.0))
    assert (r.cx, r.cy) == (20.0, 30.0)


def test_parse_fenced_json_with_prose_around_it() -> None:
    text = (
        "Sure, here are the objects I found:\n"
        '```json\n[{"bbox_2d": [1, 2, 3, 4], "label": "cube"},\n'
        ' {"bbox_2d": [5, 6, 7, 8], "label": "key"}]\n```\n'
        "Let me know if you need more."
    )
    regions = parse_regions(text)
    assert [r.label for r in regions] == ["cube", "key"]


def test_parse_point_and_alternate_keys() -> None:
    text = '{"point_2d": [100, 200], "name": "screwdriver"}'
    (r,) = parse_regions(text)
    assert (r.kind, r.label, r.cx, r.cy) == ("point", "screwdriver", 100.0, 200.0)

    (r2,) = parse_regions('[{"box": [1, 2, 3, 4], "label": "x"}]')
    assert r2.kind == "box"


def test_parse_multiple_boxes_under_one_label() -> None:
    text = '[{"label": "cube", "bbox_2d": [[1, 2, 3, 4], [5, 6, 7, 8]]}]'
    regions = parse_regions(text)
    assert len(regions) == 2
    assert all(r.label == "cube" and r.kind == "box" for r in regions)


def test_parse_bare_bracket_fallback() -> None:
    (r,) = parse_regions("The pen is at [312, 195, 401, 289] in the image.")
    assert r.kind == "box" and r.label == ""

    (p,) = parse_regions("Look at [640, 360].")
    assert p.kind == "point"


def test_parse_bare_fallback_is_skipped_when_keyed_form_present() -> None:
    text = '[{"bbox_2d": [1, 2, 3, 4], "label": "a"}] and also [9, 9]'
    regions = parse_regions(text)
    assert len(regions) == 1 and regions[0].label == "a"


def test_parse_ignores_prose_without_coordinates() -> None:
    assert parse_regions("The pen is on the left side of the desk.") == []


def test_parse_dedups_identical_regions() -> None:
    text = '[{"bbox_2d": [1, 2, 3, 4], "label": "a"}, {"bbox_2d": [1, 2, 3, 4], "label": "a"}]'
    assert len(parse_regions(text)) == 1


def test_parse_survives_malformed_json() -> None:
    assert parse_regions('[{"bbox_2d": [1, 2,, "label"}]') == []


def test_parse_recovers_complete_entries_from_a_truncated_array() -> None:
    """A reply cut off by max_new_tokens must not lose the entries it finished.

    The outer array never closes, so the whole-span parse fails; without
    per-object recovery this fell through to the bare-coordinate fallback,
    which drops every label and then dedups distinct objects sharing a box.
    """
    text = (
        '[{"bbox_2d": [0, 40, 104, 317], "label": "lamp", "description": "a lamp."},\n'
        '{"bbox_2d": [499, 0, 999, 534], "label": "balcony", "description": "a balcony."},\n'
        '{"bbox_2d": [499, 0, 999, 534], "label": "rail", "description": "a rail."},\n'
        '{"bbox_2d": [499,'
    )
    regions = parse_regions(text)
    assert [r.label for r in regions] == ["lamp", "balcony", "rail"]
    assert regions[0].description == "a lamp."
    # Same box, different labels -- these must survive the dedup.
    assert regions[1].coords == regions[2].coords


# -- coordinate-space handling -------------------------------------------- #


def test_in_bounds_flags_the_normalized_space_mismatch() -> None:
    # A 0-1000-space box read as pixels lands far outside a 640x360 frame,
    # which is the signal that the reply needs scaling rather than taking raw.
    r = Region("pen", "box", (500.0, 500.0, 800.0, 900.0))
    assert not r.in_bounds(640, 360)
    assert r.scaled(640 / 1000, 360 / 1000).in_bounds(640, 360)


def test_in_bounds_allows_slight_overshoot() -> None:
    assert Region("a", "box", (0.0, 0.0, 645.0, 361.0)).in_bounds(640, 360)
    assert not Region("a", "box", (0.0, 0.0, 900.0, 361.0)).in_bounds(640, 360)


def test_scaled_preserves_kind_and_label() -> None:
    r = Region("pen", "point", (100.0, 200.0)).scaled(0.5, 0.25)
    assert (r.kind, r.label, r.coords) == ("point", "pen", (50.0, 50.0))
