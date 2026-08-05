"""Unit tests for the SAM 2.1 HTTP client (no network)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mt4_vision.sam import Mask, SamError, best_per_object, decode_rle, health, segment


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_decode_rle_round_trip() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1:4] = True
    mask[3, 0] = True
    flat = mask.reshape(-1)
    runs = np.diff(np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))).tolist()
    assert np.array_equal(decode_rle(runs, 4, 5), mask)


def test_decode_rle_starts_inside_mask() -> None:
    """A mask covering pixel (0, 0) is sent with a leading zero-length run."""
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0:2] = True
    assert np.array_equal(decode_rle([0, 2, 4], 2, 3), mask)


def test_mask_centroid_is_area_weighted() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 6:10] = True
    m = Mask(object_index=0, score=0.9, area=int(mask.sum()), bbox=(6, 2, 9, 3), mask=mask)
    assert m.cx == 7.5
    assert m.cy == 2.5


def test_health_unreachable_names_the_tunnel() -> None:
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(SamError, match="start_tunnel"):
            health("http://example")


def test_segment_parses_masks() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    payload = {
        "ok": True,
        "width": 3,
        "height": 2,
        "objects": 1,
        "masks": [
            {"object_index": 0, "score": 0.9, "area": 2, "bbox": [0, 0, 1, 0], "rle": [0, 2, 4]},
            {"object_index": 0, "score": 0.4, "area": 1, "bbox": [2, 1, 2, 1], "rle": [5, 1]},
        ],
    }
    with patch("urllib.request.urlopen", return_value=_response(payload)) as open_:
        masks = segment(frame, points=[[1, 1]])
    assert [m.score for m in masks] == [0.9, 0.4]
    assert masks[0].mask.shape == (2, 3)
    assert masks[0].mask[0, 0]
    assert open_.call_args[0][0].full_url == "http://127.0.0.1:8767/segment"


def test_segment_sends_prompts_as_json() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    with patch("urllib.request.urlopen", return_value=_response({"ok": True, "width": 3, "height": 2, "masks": []})) as open_:
        segment(frame, points=[[1, 1]], labels=[1], boxes=[[0, 0, 2, 2]], multimask=False)
    body = open_.call_args[0][0].data.decode("utf-8", errors="replace")
    assert '"points"\r\n\r\n[[1.0, 1.0]]' in body
    assert '"labels"\r\n\r\n[1]' in body
    assert '"boxes"\r\n\r\n[[0.0, 0.0, 2.0, 2.0]]' in body
    assert '"multimask"\r\n\r\nfalse' in body


def test_segment_by_image_id_sends_no_frame() -> None:
    with patch("urllib.request.urlopen", return_value=_response({"ok": True, "width": 3, "height": 2, "masks": []})) as open_:
        segment(None, points=[[1, 1]], image_id="abc123")
    body = open_.call_args[0][0].data.decode("utf-8", errors="replace")
    assert "abc123" in body
    assert "filename=" not in body


def test_segment_needs_a_prompt() -> None:
    with pytest.raises(SamError, match="points, boxes"):
        segment(np.zeros((2, 3, 3), dtype=np.uint8))


def test_segment_needs_a_frame_or_id() -> None:
    with pytest.raises(SamError, match="frame or an image_id"):
        segment(None, points=[[1, 1]])


def test_best_per_object_keeps_prompt_order() -> None:
    m = np.zeros((2, 2), dtype=bool)
    masks = [
        Mask(1, 0.5, 1, None, m),
        Mask(0, 0.2, 1, None, m),
        Mask(0, 0.8, 1, None, m),
        Mask(1, 0.1, 1, None, m),
    ]
    best = best_per_object(masks)
    assert [(b.object_index, b.score) for b in best] == [(0, 0.8), (1, 0.5)]
