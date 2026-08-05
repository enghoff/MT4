"""Unit tests for in-process SAM 2.1 (no model download)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mt4_vision import sam as sam_mod
from mt4_vision.sam import (
    Mask,
    SamError,
    best_per_object,
    decode_rle,
    embed,
    health,
    segment,
)


def test_decode_rle_round_trip() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 1:4] = True
    mask[3, 0] = True
    flat = mask.reshape(-1)
    runs = np.diff(np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))).tolist()
    assert np.array_equal(decode_rle(runs, 4, 5), mask)


def test_decode_rle_starts_inside_mask() -> None:
    """A mask covering pixel (0, 0) is encoded with a leading zero-length run."""
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0:2] = True
    assert np.array_equal(decode_rle([0, 2, 4], 2, 3), mask)


def test_mask_centroid_is_area_weighted() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 6:10] = True
    m = Mask(object_index=0, score=0.9, area=int(mask.sum()), bbox=(6, 2, 9, 3), mask=mask)
    assert m.cx == 7.5
    assert m.cy == 2.5


def test_health_does_not_load_the_model() -> None:
    info = health()
    assert info["ok"] is True
    assert info["loaded"] is False
    assert "model" in info
    assert "cache" in info


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


def test_missing_deps_raise_sam_error() -> None:
    engine = sam_mod._SamEngine()
    real_import = __import__

    def boom(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("torch", "PIL", "PIL.Image", "transformers") or (
            fromlist and any(x in ("Sam2Model", "Sam2Processor", "Image") for x in fromlist)
        ):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=boom):
        with pytest.raises(SamError, match="requirements-sam"):
            engine.ensure_loaded()


class _FakeEngine:
    """Stand-in that returns a fixed mask so segment/embed stay offline."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._ids: dict[str, np.ndarray] = {}

    def health(self) -> dict:
        return {
            "ok": True,
            "model": "fake",
            "device": "cpu",
            "cuda": False,
            "loaded": True,
            "dtype": "fp32",
            "compile": "off",
            "cache": {"frames": len(self._ids), "capacity": 8, "hits": 0, "misses": 0},
        }

    def embed(self, frame: np.ndarray) -> str:
        image_id = sam_mod._frame_id(frame)
        self._ids[image_id] = frame
        return image_id

    def segment(self, frame, *, points=None, labels=None, boxes=None, multimask=True, image_id=None):
        self.calls.append(
            {
                "points": points,
                "labels": labels,
                "boxes": boxes,
                "multimask": multimask,
                "image_id": image_id,
                "has_frame": frame is not None,
            }
        )
        if frame is None:
            if not image_id or image_id not in self._ids:
                raise SamError(f"image_id {image_id} is not cached")
            h, w = self._ids[image_id].shape[:2]
        else:
            h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[0, 0:2] = True
        return [
            Mask(object_index=0, score=0.9, area=2, bbox=(0, 0, 1, 0), mask=mask),
            Mask(object_index=0, score=0.4, area=1, bbox=(2, 1, 2, 1), mask=mask.copy()),
        ]


def test_segment_returns_masks_from_engine() -> None:
    fake = _FakeEngine()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    with patch.object(sam_mod, "_ENGINE", fake):
        masks = segment(frame, points=[[1, 1]])
    assert [m.score for m in masks] == [0.9, 0.4]
    assert masks[0].mask.shape == (2, 3)
    assert fake.calls[0]["points"] == [[1, 1]]


def test_segment_forwards_prompts() -> None:
    fake = _FakeEngine()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    with patch.object(sam_mod, "_ENGINE", fake):
        segment(frame, points=[[1, 1]], labels=[1], boxes=[[0, 0, 2, 2]], multimask=False)
    assert fake.calls[0]["labels"] == [1]
    assert fake.calls[0]["boxes"] == [[0, 0, 2, 2]]
    assert fake.calls[0]["multimask"] is False


def test_segment_by_image_id_skips_frame() -> None:
    fake = _FakeEngine()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    with patch.object(sam_mod, "_ENGINE", fake):
        image_id = embed(frame)
        masks = segment(None, points=[[1, 1]], image_id=image_id)
    assert masks
    assert fake.calls[0]["has_frame"] is False
    assert fake.calls[0]["image_id"] == image_id
