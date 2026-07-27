"""Builds pi0.5 observation dicts from the MT4 camera and joint state.

pi05_droid expects a 7-DoF Franka. The MT4 has 4 joints + gripper, so state
is padded to 7 and the (nonexistent) wrist camera is zeroed out until the MT4
has one -- a stand-in mapping, not a real correspondence. See
https://github.com/enghoff/pi0.5-server/blob/master/docs/mt4-client-integration.md,
"The wire contract".
"""

from __future__ import annotations

import numpy as np
from openpi_client import image_tools

from mt4_jog.joints import GRIPPER_S_CLOSED, GRIPPER_S_OPEN
from mt4_jog.kinematics import JointAnglesDeg
from mt4_jog.status import Mt4Status

MT4_NUM_JOINTS = 4
DROID_NUM_JOINTS = 7
IMAGE_SIZE = 224


def mt4_state_to_droid(q_deg: JointAnglesDeg, gripper_position: float) -> tuple[np.ndarray, np.ndarray]:
    """Pad MT4's 4 joints into pi05_droid's 7-DoF Franka layout. Columns 4:7
    have no MT4 counterpart and are dropped on the way back out in
    adapter.py.
    """
    joints = np.zeros(DROID_NUM_JOINTS, dtype=np.float32)
    joints[:MT4_NUM_JOINTS] = [q_deg.j1, q_deg.j2, q_deg.j3, q_deg.j4]
    return joints, np.asarray([gripper_position], dtype=np.float32)


def build_observation(
    frame_bgr: np.ndarray,
    status: Mt4Status,
    prompt: str,
    *,
    wrist_frame_bgr: np.ndarray | None = None,
) -> dict[str, object]:
    """Assemble a pi05_droid observation dict from a camera frame + MT4 status.

    `status.joints` holds firmware step counters, not degrees -- converted
    via JointAnglesDeg.from_steps() to match what fk_tcp() expects elsewhere
    in the adapter chain.
    """
    if status.joints.keys() != {"j1", "j2", "j3", "j4"}:
        raise ValueError(f"incomplete joint status: {status.joints}")
    steps = (status.joints["j1"], status.joints["j2"], status.joints["j3"], status.joints["j4"])
    q_deg = JointAnglesDeg.from_steps(steps)

    gripper_position = _gripper_fraction(status.tcp.grip if status.tcp else GRIPPER_S_OPEN)
    joints, gripper = mt4_state_to_droid(q_deg, gripper_position)

    exterior = _prep_image(frame_bgr)
    wrist = (
        _prep_image(wrist_frame_bgr)
        if wrist_frame_bgr is not None
        else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    )

    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joints,
        "observation/gripper_position": gripper,
        "prompt": prompt,
    }


def _prep_image(frame_bgr: np.ndarray) -> np.ndarray:
    # capture_frame() returns OpenCV BGR; the DROID checkpoint was trained on RGB.
    frame_rgb = frame_bgr[:, :, ::-1]
    resized = image_tools.resize_with_pad(frame_rgb, IMAGE_SIZE, IMAGE_SIZE)
    return image_tools.convert_to_uint8(resized)


def _gripper_fraction(servo_position: float) -> float:
    """Firmware gripper servo units (GRIPPER_S_OPEN..GRIPPER_S_CLOSED) -> [0, 1]."""
    span = GRIPPER_S_CLOSED - GRIPPER_S_OPEN
    return max(0.0, min(1.0, (servo_position - GRIPPER_S_OPEN) / span))
