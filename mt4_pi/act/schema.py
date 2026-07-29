"""The observation/action contract shared by ACT export, packing, and serving.

Deliberately dependency-light (numpy only) so the packing step can run on a
RunPod box that has `lerobot` but not `mt4_jog`/`cv2`/serial, and so the
serving side can import it without dragging in the training stack.

Why this looks nothing like the pi0.5 path
------------------------------------------
The pi05 converter (`mt4_pi/collect/convert_to_lerobot.py`) had to disguise
the MT4 as a Franka to reuse a DROID-pretrained checkpoint: pad 4 joints to 7
(then to 32), invent a zeroed wrist-camera channel, and inherit DROID's 15 Hz
assumption. ACT trains from scratch, so every one of those is dropped. What
remains is the smallest honest description of this robot.

Actions are ABSOLUTE joint targets, not velocities
--------------------------------------------------
The pi05 labels were per-tick deltas divided by `DT_S`, i.e. rad/s. That
choice cost us three separate bugs, and all three disappear with absolute
targets:

  * **Rate coupling.** `DT_S` was 1/15, but the recorder actually ticks at
    ~10 Hz (measured mean 0.1003 s, median 0.094 s over 44,475 ticks), so
    every velocity label was ~1.5x too large. Worse, we *execute* at ~4.5 Hz
    because each `queue_move` costs ~204 ms on the serial link -- so even a
    correctly-scaled velocity would be integrated over the wrong interval. An
    absolute target is a target; it does not care how long the trip takes.
  * **Integration drift.** Velocities have to be integrated against a live
    state estimate, so any per-step error accumulates. Absolute targets are
    self-correcting: a wrong one is wrong once.
  * **The saturation clip.** `adapter.integrate_joint_target` clips to
    +-1.0 rad/s, which silently throttled predictions (the model's outputs
    ran 2-5x larger than truth, so the clip was active on nearly every
    chunk). There is nothing to clip here beyond the joint limits themselves.

Task conditioning rides in the state vector
-------------------------------------------
ACT has no language input -- it is a per-task behavioural-cloning method. Our
corpus is not one task: it spans 4 place kinds and 3 cube colours, and given
the same scene "put the red cube on the marker" and "take the top block off
the stack" demand different behaviour. Trained unconditioned, ACT can only
learn the marginal over tasks, which is a guaranteed failure.

Rather than split the 225 episodes into per-task datasets of ~50 (or modify
the architecture), the task is encoded as one-hot columns appended to
`observation.state`. ACT consumes state as an opaque vector, so this is task
conditioning at zero model-code cost, and it keeps all 225 episodes in one
training run. The goal placement is included for the same reason it is
available at inference time -- `mt4_vision` computes the free slot, so
withholding it would be pretending to a harder problem than we actually
have. The PICK location is deliberately NOT included: finding the cube to
grasp is the part we want the policy to learn from pixels.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------

#: Recorder tick rate, measured over the whole corpus rather than assumed.
#: `mt4_pi.collect.recorder` nominally targets 15 Hz but does not hit it; the
#: empirical distribution is mean 0.1003 s / median 0.0940 s per tick
#: (p05 0.0930, p95 0.1100). LeRobot derives frame timestamps as `i / fps`, so
#: this only needs to be right on average -- and with absolute action targets
#: the residual +-10% jitter is harmless (see module docstring).
FPS = 10

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

#: Single steeply-oblique exterior camera. Source frames are 1280x720; halving
#: each dimension preserves the 16:9 aspect (no letterbox padding, unlike the
#: pi05 path's square 224x224 `resize_with_pad`) and still leaves a ~30 px cube.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

#: LeRobot feature key. Named for what it is rather than borrowing DROID's
#: `exterior_image_1_left`, and there is no wrist entry at all -- an absent
#: camera is better represented by absence than by a zero-filled tensor the
#: model must learn to ignore.
CAMERA_KEY = "observation.images.overhead"

# ---------------------------------------------------------------------------
# State / action layout
# ---------------------------------------------------------------------------

MT4_NUM_JOINTS = 4

#: Cube colours the collector uses, in a fixed order (one-hot index).
COLORS = ("red", "green", "blue")

#: `meta["place"]["kind"]` values, in a fixed order (one-hot index).
#: to_marker/to_slot come from `collect_shuffle.py`, stack/unstack from
#: `collect_stack.py`.
PLACE_KINDS = ("to_marker", "to_slot", "stack", "unstack")

#: Divisor applied to the goal placement (mm) before it enters the state
#: vector. LeRobot normalises state with dataset statistics anyway, so this
#: only keeps the raw column on the same order of magnitude as its
#: neighbours -- it makes the un-normalised vector readable in logs and stops
#: a 275 mm entry from dominating any accidental unnormalised path.
GOAL_XY_SCALE_MM = 300.0

#: Column names of `observation.state`, in order. Kept as data rather than
#: hard-coded offsets so the packing step can prune degenerate columns (see
#: `prune_constant_columns`) and record exactly what survived.
STATE_COLUMNS: tuple[str, ...] = (
    "j1_rad",
    "j2_rad",
    "j3_rad",
    "j4_rad",
    "gripper",
    *(f"color_{c}" for c in COLORS),
    *(f"kind_{k}" for k in PLACE_KINDS),
    "goal_x_norm",
    "goal_y_norm",
)

#: Column names of `action`, in order. Absolute joint targets at t+1 plus the
#: gripper command at t+1 -- the same quantity the firmware consumes, so
#: serving is a direct hand-off with no integration step.
ACTION_COLUMNS: tuple[str, ...] = (
    "j1_rad",
    "j2_rad",
    "j3_rad",
    "j4_rad",
    "gripper",
)

STATE_DIM = len(STATE_COLUMNS)
ACTION_DIM = len(ACTION_COLUMNS)


def task_features(color: str, place_kind: str, goal_xy_mm: tuple[float, float]) -> np.ndarray:
    """Build the task-conditioning tail of the state vector.

    Raises on an unknown colour or place kind rather than emitting an all-zero
    one-hot: a silent all-zero task encoding would train fine and then fail
    only on the arm, which is the worst possible place to discover a typo.
    """
    if color not in COLORS:
        raise ValueError(f"unknown color {color!r}; expected one of {COLORS}")
    if place_kind not in PLACE_KINDS:
        raise ValueError(f"unknown place kind {place_kind!r}; expected one of {PLACE_KINDS}")

    out = np.zeros(len(COLORS) + len(PLACE_KINDS) + 2, dtype=np.float32)
    out[COLORS.index(color)] = 1.0
    out[len(COLORS) + PLACE_KINDS.index(place_kind)] = 1.0
    out[-2] = goal_xy_mm[0] / GOAL_XY_SCALE_MM
    out[-1] = goal_xy_mm[1] / GOAL_XY_SCALE_MM
    return out


def build_state(
    joints_rad: np.ndarray,
    gripper: float,
    color: str,
    place_kind: str,
    goal_xy_mm: tuple[float, float],
) -> np.ndarray:
    """Assemble a full `observation.state` row. Used by both the exporter and
    the serving client so the two cannot drift apart.
    """
    joints_rad = np.asarray(joints_rad, dtype=np.float32)
    if joints_rad.shape != (MT4_NUM_JOINTS,):
        raise ValueError(f"expected {MT4_NUM_JOINTS} joint angles, got shape {joints_rad.shape}")
    return np.concatenate(
        [joints_rad, np.float32([gripper]), task_features(color, place_kind, goal_xy_mm)]
    ).astype(np.float32)


def prune_constant_columns(
    state: np.ndarray, columns: tuple[str, ...] = STATE_COLUMNS, *, tol: float = 1e-3
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Drop state columns with no variance across the dataset.

    A one-hot column is constant whenever the corpus subset happens to contain
    only one value for it -- e.g. filtering to shuffle-only episodes leaves
    `kind_stack` and `kind_unstack` at zero for every frame. LeRobot's
    MEAN_STD normalisation divides by `std + eps`, so a constant column
    becomes `0 / 1e-8` at best and an inf-generating landmine at worst. Since
    a constant column carries no information anyway, removing it is strictly
    better than trying to normalise it.

    Returns `(pruned_state, kept_columns, dropped_columns)`. Callers must
    persist `kept_columns` alongside the dataset -- serving has to build the
    identical vector, and the layout is no longer inferable from the schema
    once anything has been pruned.
    """
    if state.ndim != 2 or state.shape[1] != len(columns):
        raise ValueError(f"expected state of shape (N, {len(columns)}), got {state.shape}")
    keep = state.std(axis=0) > tol
    kept = tuple(c for c, k in zip(columns, keep) if k)
    dropped = tuple(c for c, k in zip(columns, keep) if not k)
    return state[:, keep], kept, dropped


def select_columns(row: np.ndarray, kept: tuple[str, ...] | list[str]) -> np.ndarray:
    """Reduce a full-layout state row to the columns a dataset actually kept.

    The serving-side counterpart of `prune_constant_columns`: build the full
    `STATE_COLUMNS` vector, then project it onto whatever the trained model
    expects.
    """
    index = {name: i for i, name in enumerate(STATE_COLUMNS)}
    missing = [c for c in kept if c not in index]
    if missing:
        raise ValueError(f"kept columns not present in the current schema: {missing}")
    return np.asarray(row, dtype=np.float32)[[index[c] for c in kept]]
