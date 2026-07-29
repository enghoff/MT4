"""Schema/export/pack tests for the ACT pipeline (no hardware, no lerobot).

The three modules under test split along dependency lines on purpose, and
these tests hold that line: `schema` is numpy-only, `export_episodes` adds
kinematics, and `pack_lerobot` keeps `cv2`/`lerobot` behind function-local
imports so its pure array logic stays testable on a machine with neither.

The properties worth pinning are the ones whose violation would train fine
and only fail on the arm: the absolute-target invariant, the task one-hot
layout, and the fact that dropping static frames re-points the labels it
leaves behind.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_pi.act import export_episodes, pack_lerobot, schema

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_state_layout_is_self_consistent():
    assert schema.STATE_DIM == len(schema.STATE_COLUMNS) == 14
    assert schema.ACTION_DIM == len(schema.ACTION_COLUMNS) == 5
    # The first five state columns must mirror the action columns exactly --
    # `retarget_actions` slices `state[:, :5]` as if it were an action.
    assert schema.STATE_COLUMNS[: schema.ACTION_DIM] == schema.ACTION_COLUMNS


def test_build_state_places_joints_gripper_and_task_encoding():
    row = schema.build_state(
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        gripper=0.75,
        color="green",
        place_kind="unstack",
        goal_xy_mm=(150.0, -75.0),
    )
    assert row.dtype == np.float32 and row.shape == (schema.STATE_DIM,)
    np.testing.assert_allclose(row[:4], [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
    assert row[4] == pytest.approx(0.75)

    col = {name: i for i, name in enumerate(schema.STATE_COLUMNS)}
    assert row[col["color_green"]] == 1.0
    assert row[col["color_red"]] == row[col["color_blue"]] == 0.0
    assert row[col["kind_unstack"]] == 1.0
    assert row[col["kind_to_marker"]] == 0.0
    assert row[col["goal_x_norm"]] == pytest.approx(150.0 / schema.GOAL_XY_SCALE_MM)
    assert row[col["goal_y_norm"]] == pytest.approx(-75.0 / schema.GOAL_XY_SCALE_MM)


@pytest.mark.parametrize(
    "color,kind", [("purple", "to_marker"), ("red", "to_bin"), ("", "stack")]
)
def test_unknown_task_labels_raise_rather_than_encoding_all_zeros(color, kind):
    """A silent all-zero one-hot would train without complaint and only
    misbehave on the arm, so a typo has to fail here instead."""
    with pytest.raises(ValueError):
        schema.task_features(color, kind, (0.0, 0.0))


def test_prune_constant_columns_drops_zero_variance_and_reports_names():
    # A shuffle-only subset: the stack/unstack one-hots never vary.
    rows = np.stack(
        [
            schema.build_state(np.array([0.0, 1.0, 0.1, 0.0], np.float32), 0.2, "red", "to_marker", (10.0, 20.0)),
            schema.build_state(np.array([0.5, 1.2, 0.3, 0.1], np.float32), 0.8, "blue", "to_marker", (90.0, 20.0)),
        ]
    )
    pruned, kept, dropped = schema.prune_constant_columns(rows)
    assert pruned.shape == (2, len(kept))
    assert "kind_stack" in dropped and "kind_unstack" in dropped
    assert "color_green" in dropped  # absent from this two-row subset
    assert "kind_to_marker" in dropped  # constant at 1.0, so carries nothing
    assert "j1_rad" in kept and "color_red" in kept and "goal_x_norm" in kept
    assert set(kept).isdisjoint(dropped)
    assert len(kept) + len(dropped) == schema.STATE_DIM


def test_select_columns_reproduces_the_pruned_layout_at_serving_time():
    rows = np.stack(
        [
            schema.build_state(np.array([0.0, 1.0, 0.1, 0.0], np.float32), 0.2, "red", "to_marker", (10.0, 20.0)),
            schema.build_state(np.array([0.5, 1.2, 0.3, 0.1], np.float32), 0.8, "blue", "to_marker", (90.0, 20.0)),
        ]
    )
    pruned, kept, _ = schema.prune_constant_columns(rows)
    for i in range(len(rows)):
        np.testing.assert_allclose(schema.select_columns(rows[i], kept), pruned[i], rtol=1e-6)


def test_select_columns_rejects_a_layout_the_schema_no_longer_has():
    row = schema.build_state(np.zeros(4, np.float32), 0.0, "red", "stack", (0.0, 0.0))
    with pytest.raises(ValueError, match="not present in the current schema"):
        schema.select_columns(row, ["j1_rad", "color_chartreuse"])


# ---------------------------------------------------------------------------
# export_episodes.build_arrays
# ---------------------------------------------------------------------------


def _tick(x, y, z, j4=0.0, grip=0.5):
    return {"state": {"x": x, "y": y, "z": z, "j4": j4, "grip": grip}}


def _meta(color="red", kind="to_marker", goal=(150.0, 100.0)):
    return {"pick": {"color": color}, "place": {"kind": kind, "x": goal[0], "y": goal[1]}}


def _straight_line_episode(n=6):
    """A reachable descending path, so IK actually solves for every tick."""
    ticks = [_tick(190.0, 0.0, 226.0 - 6.0 * i, grip=0.5 + 0.02 * i) for i in range(n)]
    joints = export_episodes.solve_joint_track(ticks)
    assert joints is not None, "fixture path must be IK-reachable"
    return _meta(), ticks, joints


def test_build_arrays_drops_the_final_tick():
    meta, ticks, joints = _straight_line_episode(n=6)
    arrays = export_episodes.build_arrays(meta, ticks, joints)
    # The last tick has no successor, so no action can be labelled for it.
    assert len(arrays["state"]) == len(arrays["action"]) == len(ticks) - 1


def test_action_is_the_next_absolute_state_not_a_delta():
    """The invariant the whole pipeline rests on. If this ever becomes a
    delta again, every downstream assumption (no integration, no rad/s clip,
    rate independence, static-frame retargeting) silently breaks."""
    meta, ticks, joints = _straight_line_episode(n=8)
    arrays = export_episodes.build_arrays(meta, ticks, joints)
    state, action = arrays["state"], arrays["action"]
    np.testing.assert_allclose(action[:-1], state[1:, : schema.ACTION_DIM], rtol=1e-6)

    # And it is genuinely absolute: joint targets sit at the pose's own
    # magnitude, nowhere near a per-tick delta.
    assert abs(action[0, 1]) > 0.5  # j2 near ~1.8 rad, not ~0.01 rad/s


def test_state_carries_the_current_pose_and_action_the_next_one():
    meta, ticks, joints = _straight_line_episode(n=5)
    arrays = export_episodes.build_arrays(meta, ticks, joints)
    np.testing.assert_allclose(
        arrays["state"][0, :4],
        [math.radians(v) for v in (joints[0].j1, joints[0].j2, joints[0].j3, joints[0].j4)],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        arrays["action"][0, :4],
        [math.radians(v) for v in (joints[1].j1, joints[1].j2, joints[1].j3, joints[1].j4)],
        rtol=1e-6,
    )


def test_static_flag_marks_frames_with_neither_joint_nor_gripper_motion():
    ticks = [
        _tick(190.0, 0.0, 226.0, grip=0.5),
        _tick(190.0, 0.0, 226.0, grip=0.5),  # frame 0 -> 1: nothing moves
        _tick(190.0, 0.0, 226.0, grip=0.9),  # frame 1 -> 2: gripper only
        _tick(190.0, 0.0, 200.0, grip=0.9),  # frame 2 -> 3: arm moves
        _tick(190.0, 0.0, 200.0, grip=0.9),
    ]
    joints = export_episodes.solve_joint_track(ticks)
    assert joints is not None
    static = export_episodes.build_arrays(_meta(), ticks, joints)["static"]
    assert static.tolist() == [True, False, False, True]


def test_task_columns_are_constant_across_an_episode():
    meta, ticks, joints = _straight_line_episode(n=6)
    state = export_episodes.build_arrays(meta, ticks, joints)["state"]
    task_tail = state[:, schema.ACTION_DIM :]
    assert np.all(task_tail == task_tail[0]), "task conditioning must not vary within an episode"


# ---------------------------------------------------------------------------
# pack_lerobot: static trimming
# ---------------------------------------------------------------------------


def test_static_keep_mask_disabled_keeps_everything():
    static = np.array([True] * 5)
    assert pack_lerobot.static_keep_mask(static, None).all()


def test_static_keep_mask_trims_each_run_independently():
    static = np.array([False, True, True, True, True, False, True, True, False])
    keep = pack_lerobot.static_keep_mask(static, max_run=2)
    #                 F     T     T      d      d     F     T     T     F
    assert keep.tolist() == [True, True, True, False, False, True, True, True, True]


def test_static_keep_mask_zero_drops_every_static_frame():
    static = np.array([False, True, True, False, True])
    keep = pack_lerobot.static_keep_mask(static, max_run=0)
    assert keep.tolist() == [True, False, False, True, False]


def test_retarget_actions_points_survivors_at_the_next_surviving_state():
    """Deleting a frame must not leave the previous frame commanding a pose
    that is no longer in the dataset -- only legal because targets are
    absolute."""
    state = np.arange(40, dtype=np.float32).reshape(8, 5)
    action = np.concatenate([state[1:], state[-1:]])  # action[t] = state[t+1]
    keep = np.array([True, False, False, True, True, False, True, True])

    out = pack_lerobot.retarget_actions(state, action, keep)
    idx = np.flatnonzero(keep)
    assert len(out) == len(idx)
    # Every kept frame but the last now targets the next kept frame's state.
    np.testing.assert_allclose(out[:-1], state[idx[1:]], rtol=1e-6)
    # The last survivor keeps its original label; there is no successor.
    np.testing.assert_allclose(out[-1], action[idx[-1]], rtol=1e-6)


def test_retarget_actions_is_a_no_op_when_nothing_is_dropped():
    state = np.arange(25, dtype=np.float32).reshape(5, 5)
    action = np.concatenate([state[1:], state[-1:]])
    out = pack_lerobot.retarget_actions(state, action, np.ones(5, dtype=bool))
    np.testing.assert_allclose(out, action, rtol=1e-6)


def test_retarget_actions_handles_a_single_surviving_frame():
    state = np.arange(15, dtype=np.float32).reshape(3, 5)
    action = np.concatenate([state[1:], state[-1:]])
    keep = np.array([False, True, False])
    out = pack_lerobot.retarget_actions(state, action, keep)
    np.testing.assert_allclose(out, action[1:2], rtol=1e-6)
