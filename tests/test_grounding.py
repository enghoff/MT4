"""Unit tests for the Grounding DINO HTTP client (no network)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mt4_vision.grounding import Detection, GroundingError, detect, health


def test_detection_center() -> None:
    d = Detection("pen", 0.5, 10, 20, 30, 40)
    assert d.cx == 20
    assert d.cy == 30


def test_health_ok() -> None:
    payload = {"ok": True, "model": "tiny", "device": "cuda"}
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=resp) as open_:
        assert health("http://example") == payload
        assert open_.call_args[0][0] == "http://example/health"


def test_health_unreachable() -> None:
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("down"),
    ):
        with pytest.raises(GroundingError, match="unreachable"):
            health("http://example")


def test_detect_parses_multipart_response() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    payload = {
        "ok": True,
        "detections": [
            {"label": "pen", "score": 0.4, "box": [1, 2, 11, 22]},
            {"label": "cube", "score": 0.9, "box": [5, 5, 15, 15]},
        ],
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=resp):
        dets = detect(frame, "pen. cube.")
    assert [d.label for d in dets] == ["cube", "pen"]  # score desc
    assert dets[0].score == 0.9
    assert dets[0].cx == 10
