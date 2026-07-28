"""Autonomous demonstration collection for the MT4 -> pi0.5 fine-tune.

Wraps mt4_vision's existing vision-guided pick/place (the same primitives
shuffle_blocks.py runs unattended) with per-episode recording, so cube
shuffling doubles as self-supervised demonstration collection instead of
requiring Xbox teleop for every episode. See
https://github.com/enghoff/pi0.5-server/blob/master/docs/mt4-client-integration.md
("First Integration Milestone", steps 11-12) for where this fits.

Output is raw per-episode traces, not a LeRobot dataset -- a separate
conversion step (mirroring openpi's
examples/libero/convert_libero_data_to_lerobot.py) does that on the GPU/
Linux side. See recorder.py for the on-disk layout: per-tick state comes
from real commanded waypoints (mt4_vision.pickplace's ``on_waypoint``
hook), linearly interpolated within each leg by that leg's own measured
duration and held flat across the dead time between legs.
"""
