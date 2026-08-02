"""Annotated preview + recording (no hardware, no camera)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from mt4_vision.detect import CubeDetection, MarkerDetection
from mt4_vision.preview import VideoRecorder, annotate_scene
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


def test_video_recorder_writes_video(tmp_path):
    video_path = tmp_path / "run.avi"
    recorder = VideoRecorder(video_path=str(video_path), fps=2.0)
    frame = np.full((64, 64, 3), 200, dtype=np.uint8)

    recorder.write(frame)
    recorder.write(frame)
    recorder.close()

    assert video_path.exists()
    assert video_path.stat().st_size > 0
