"""The MT4's joint-angle convention, shared by every policy path.

Extracted from `observation.py` so it can be used without pulling in
`openpi_client`: this is a statement about how the arm reports itself, not
about pi0.5, and both the pi0.5 and ACT runtimes need it. `observation.py`
re-exports it, so existing imports are unaffected.
"""

from __future__ import annotations

from mt4_jog.kinematics import JointAnglesDeg
from mt4_jog.status import Mt4Status


def joint_state_from_status(status: Mt4Status) -> JointAnglesDeg:
    """MT4 status -> the joint convention the policies are TRAINED on.

    j1..j3 come from the firmware step counters, but **j4 must come from
    `status.tcp.j4`, not `status.joints["j4"]`** -- they are different
    quantities and mixing them is a silent, compounding error:

    - The firmware runs `ORIENT=hold`, so the wrist counter-rotates 1:1 with
      the base to hold an absolute heading. Measured: commanding j4=0 while
      sweeping the base to azimuth 40 deg leaves `tcp.j4 == 0.0` but drives
      `joints["j4"]` to -1800 steps == -40.00 deg. The relation is
      `raw_j4 = tcp_j4 - j1`.
    - Both training paths label the j4 channel with the recorded *world-frame*
      wrist angle (`convert_to_lerobot._solve_joint_track` for pi0.5,
      `act.export_episodes.solve_joint_track` for ACT), and `queue_move()`
      likewise takes a TCP orientation. So TCP j4 is the convention on both
      the training and the command side; raw joint j4 is the odd one out.

    Reading raw j4 here fed the model an out-of-distribution state off by
    -j1, and integrating an action onto it then re-commanding the result as
    a TCP angle drove the wrist a further -j1 each tick. That compounded to
    a -134 deg runaway over 7 ticks in the first round-2 hardware test; see
    docs/PI05_FINETUNING_PIPELINE.md 7b.
    """
    if status.joints.keys() != {"j1", "j2", "j3", "j4"}:
        raise ValueError(f"incomplete joint status: {status.joints}")
    if status.tcp is None:
        raise ValueError("status has no TCP pose; cannot resolve world-frame j4")
    steps = (status.joints["j1"], status.joints["j2"], status.joints["j3"], status.joints["j4"])
    q_raw = JointAnglesDeg.from_steps(steps)
    return JointAnglesDeg(q_raw.j1, q_raw.j2, q_raw.j3, status.tcp.j4)
