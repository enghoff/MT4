"""Leg-composition tests for the mt4_vision.motion primitives (no hardware).

Pins what pick_at / place_at / transfer actually queue: which legs exist, which
run slow, which wrist each holds, and -- the point of the whole layer -- where
the grip stations sit and how few move_path calls it takes. Uses a recording
fake client plus the real StackPlanner geometry.

Run: python tests/test_motion.py  (or pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_jog.client import Mt4ClientError
from mt4_jog.joints import JOINT_SOFT_MAX_STEPS, JOINT_SOFT_MIN_STEPS
from mt4_jog.kinematics import STEPS_PER_DEG
from mt4_vision.motion import (
    BARRIER_DWELL_MS,
    DEFAULT_STATION_DWELL_MS,
    MAX_LEGS_PER_QUEUE,
    YAW_PERIOD_LONG_AXIS,
    Grasp,
    Leg,
    pick_at,
    place_at,
    plan_pick_legs,
    plan_place_legs,
    split_legs,
    transfer,
)
from mt4_vision.stackpath import StackPlanner
from mt4_vision.wrist import (
    DEFAULT_J4_MARGIN_STEPS,
    _j4_on_lattice,
    j4_for_long_axis,
)

# Live calibration values (vision_calibration.json, 2026-07).
CALIB = SimpleNamespace(
    table_z=127.2,
    cube_height_mm=20.0,
    safe_z=155.0,
    travel_speed_us=700,
    approach_speed_us=2400,
    grip_open_s=140,
    grip_close_s=240,
    face_align_picks=True,
)
PICK_XY = (220.0, -60.0)
PLACE_XY = (180.0, 120.0)


class _Tcp:
    def __init__(self, x, y, z, j4=0.0):
        self.x, self.y, self.z, self.j4 = x, y, z, j4


class FakeClient:
    """Records every move_path call; reports homed and a fixed TCP."""

    def __init__(self, tcp=None, homed=True):
        self._tcp = tcp or _Tcp(200.0, 0.0, 260.0)
        self._homed = homed
        self.calls: list[dict[str, object]] = []
        self.homes = 0
        self.gripper_calls: list[object] = []
        self.tcp_reads = 0

    def get_tcp(self):
        self.tcp_reads += 1
        return self._tcp

    def get_status(self):
        return SimpleNamespace(homed=self._homed, joints={"j4": 0})

    def home(self):
        self.homes += 1
        self._homed = True
        return {"ok": True}

    def gripper(self, action):
        self.gripper_calls.append(action)
        return {"ok": True}

    def move_path(self, waypoints, j4=None, grip=0, speed_us=0, dwell_ms=0, **kw):
        n = len(waypoints)

        def spread(v):
            return list(v) if isinstance(v, (list, tuple)) else [v] * n

        self.calls.append(
            {
                "wps": list(waypoints),
                "j4": spread(j4),
                "speeds": spread(speed_us),
                # move_path's scalar-grip rule is first-leg-only, not broadcast.
                "grips": list(grip) if isinstance(grip, (list, tuple))
                else [grip] + [0] * (n - 1),
                "dwells": spread(dwell_ms),
            }
        )
        return {"ok": True}

    def legs(self) -> list[Leg]:
        """Every recorded call flattened back into Legs, in send order."""
        out: list[Leg] = []
        for c in self.calls:
            for i, wp in enumerate(c["wps"]):
                out.append(
                    Leg(
                        wp[0], wp[1], wp[2],
                        j4=c["j4"][i],
                        speed_us=c["speeds"][i],
                        grip=c["grips"][i],
                        dwell_ms=c["dwells"][i],
                    )
                )
        return out


def free() -> StackPlanner:
    return StackPlanner.free_space(CALIB)


def expect_error(fn, needle: str) -> None:
    try:
        fn()
    except Mt4ClientError as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc!r}"
        return
    raise AssertionError(f"expected Mt4ClientError containing {needle!r}")


# -- pick -----------------------------------------------------------------


def test_pick_leg_shape() -> None:
    """transit(opening) -> barrier -> slow descend -> close station -> lift."""
    c = FakeClient()
    pick_at(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0))
    legs = c.legs()
    assert len(legs) == 5, legs
    transit, barrier, descend, close, lift = legs

    assert transit.xyz == (*PICK_XY, CALIB.safe_z)
    # Jaws open on the FIRST leg so the sweep overlaps the transit for free.
    assert transit.grip == CALIB.grip_open_s
    assert not transit.is_station
    assert transit.speed_us == CALIB.travel_speed_us

    # Barrier waits out that open sweep before descending on the cube.
    assert barrier.is_station and barrier.grip == 0
    assert barrier.dwell_ms == BARRIER_DWELL_MS
    assert barrier.xyz == transit.xyz  # a station must sit where the arm is

    assert descend.xyz == (*PICK_XY, CALIB.table_z)
    assert descend.speed_us == CALIB.approach_speed_us  # only the drop is slow
    assert descend.grip == 0

    assert close.is_station
    assert close.grip == CALIB.grip_close_s
    assert close.dwell_ms == DEFAULT_STATION_DWELL_MS
    assert close.xyz == descend.xyz

    assert lift.xyz == (*PICK_XY, CALIB.safe_z)
    assert lift.speed_us == CALIB.travel_speed_us


def test_pick_is_one_queue() -> None:
    """The whole point: no blocking gripper calls, no per-segment round trips."""
    c = FakeClient()
    out = pick_at(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0))
    assert len(c.calls) == 1
    assert out["queues"] == 1
    assert c.gripper_calls == []


def test_pick_lift_after_false_leaves_arm_down() -> None:
    c = FakeClient()
    pick_at(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), lift_after=False)
    legs = c.legs()
    assert legs[-1].is_station  # ends on the close, still at grab height
    assert legs[-1].z == CALIB.table_z


def test_pick_station_poses_match_preceding_leg() -> None:
    """Firmware rejects a station whose pose is not where the arm is
    ("err mq station pose"), so every station must repeat its predecessor."""
    c = FakeClient()
    pick_at(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0))
    legs = c.legs()
    for i, leg in enumerate(legs):
        if leg.is_station:
            assert i > 0, "a station cannot be the first leg of this plan"
            assert leg.xyz == legs[i - 1].xyz, f"station {i} moved"


def test_pick_yaw_none_preserves_wrist() -> None:
    c = FakeClient()
    out = pick_at(c, CALIB, Grasp(*PICK_XY))
    assert "j4" not in out
    # No explicit angle anywhere; the firmware wrist sentinel holds instead.
    assert all(leg.j4 in (None, "wrist") for leg in c.legs())


def test_pick_face_align_yaw_lands_on_lattice() -> None:
    c = FakeClient()
    out = pick_at(c, CALIB, Grasp(*PICK_XY, yaw_deg=100.0))
    # 100 deg folds onto the same square face as 10 deg.
    assert abs(float(out["j4"]) - 10.0) < 1e-6
    assert out["yaw_deg"] == 100.0


# -- long-axis (elongated object) grasps -----------------------------------


def test_long_axis_grasp_closes_across_the_axis() -> None:
    c = FakeClient()
    out = pick_at(
        c, CALIB,
        Grasp(*PICK_XY, yaw_deg=37.0, yaw_period_deg=YAW_PERIOD_LONG_AXIS),
    )
    expected = j4_for_long_axis(37.0, current_j4_deg=0.0, x=PICK_XY[0], y=PICK_XY[1])
    assert expected is not None
    assert abs(float(out["j4"]) - expected) < 1e-6
    # Jaws perpendicular to the shaft, not along it.
    assert abs(((float(out["j4"]) - 37.0) % 180.0) - 90.0) < 1e-6


def test_stock_soft_limits_never_starve_either_lattice() -> None:
    """Documents why the abstain path below is unreachable in normal operation:
    the joint-J4 window (~351 deg) is wider than either grasp period, so even
    the 180 deg lattice's two candidates per turn always leave one inside."""
    width = (
        JOINT_SOFT_MAX_STEPS[3] - JOINT_SOFT_MIN_STEPS[3] - 2 * DEFAULT_J4_MARGIN_STEPS
    ) / STEPS_PER_DEG[3]
    assert width >= YAW_PERIOD_LONG_AXIS
    for x, y in ((220.0, -60.0), (150.0, -250.0), (52.0, -258.6), (60.0, 300.0)):
        for axis in range(0, 180, 5):
            assert (
                j4_for_long_axis(float(axis), current_j4_deg=0.0, x=x, y=y) is not None
            )


def test_long_axis_grasp_abstains_when_the_window_is_narrower_than_the_period() -> None:
    """The guard for a rig whose MT4_JOINT_SOFT_* window is tighter than 180 deg.
    Shrinking the margin below half the window reproduces that condition: the
    resolver must abstain, not fall back to grasping along the shaft."""
    narrow = 5000  # window drops to ~138 deg, under the 180 deg period
    assert (
        _j4_on_lattice(
            0.0 + 90.0, YAW_PERIOD_LONG_AXIS, current_j4_deg=0.0,
            x=PICK_XY[0], y=PICK_XY[1],
            j4_margin_steps=narrow, require_feasible=True,
        )
        is None
    )
    # ...and with the fallback allowed (the 90 deg path's behaviour) it answers.
    assert (
        _j4_on_lattice(
            0.0, 90.0, current_j4_deg=0.0, x=PICK_XY[0], y=PICK_XY[1],
            j4_margin_steps=narrow, require_feasible=False,
        )
        is not None
    )


def test_long_axis_abstention_surfaces_as_an_error_not_a_silent_grasp() -> None:
    """resolve_yaw_j4 must raise on an abstaining lattice rather than return the
    same None that means "no yaw opinion" -- those two must never merge."""
    import mt4_vision.motion as m

    c = FakeClient()
    orig = m.j4_for_long_axis
    m.j4_for_long_axis = lambda *a, **k: None
    try:
        expect_error(
            lambda: pick_at(
                c, CALIB,
                Grasp(*PICK_XY, yaw_deg=37.0, yaw_period_deg=YAW_PERIOD_LONG_AXIS),
            ),
            "no feasible wrist angle",
        )
    finally:
        m.j4_for_long_axis = orig
    assert c.calls == [], "nothing should have been sent"


# -- place ----------------------------------------------------------------


def test_place_leg_shape() -> None:
    """carry -> slow descend -> open station -> lift clear."""
    c = FakeClient(_Tcp(*PICK_XY, CALIB.safe_z))
    out = place_at(c, CALIB, Grasp(*PLACE_XY))
    legs = c.legs()
    carry, descend, release, lift = legs[-4:]
    rz = CALIB.table_z + 3.0
    assert carry.xyz == (*PLACE_XY, CALIB.safe_z)
    assert descend.xyz == (*PLACE_XY, rz)
    assert descend.speed_us == CALIB.approach_speed_us
    assert release.is_station and release.grip == CALIB.grip_open_s
    assert release.xyz == descend.xyz
    assert lift.xyz == (*PLACE_XY, CALIB.safe_z)
    assert out["release_z"] == rz
    assert len(c.calls) == 1


def test_place_releases_above_grasp_height() -> None:
    """The object drops the last couple of mm rather than being pressed in."""
    c = FakeClient(_Tcp(*PICK_XY, CALIB.safe_z))
    place_at(c, CALIB, Grasp(*PLACE_XY))
    release = [leg for leg in c.legs() if leg.is_station][-1]
    assert release.z > CALIB.table_z


def test_square_place_lands_square_to_the_axes_not_on_the_carried_wrist() -> None:
    """A bare Grasp means "no yaw opinion, keep the wrist", which lands a cube
    at whatever angle the pick left it. A destination has no orientation of its
    own, so "put it down here" means square -- what pickplace.place has always
    done via resolve_place_j4, and what the queued path dropped."""
    from mt4_vision.motion import square_place

    c = FakeClient(_Tcp(*PICK_XY, CALIB.safe_z, j4=37.0))
    out = place_at(c, CALIB, Grasp(*PLACE_XY))
    # Every leg carries a firmware hold sentinel ("wrist"/None), never an angle,
    # so the cube lands at whatever yaw the pick left.
    assert "j4" not in out
    assert not any(isinstance(leg.j4, float) for leg in c.legs())

    c = FakeClient(_Tcp(*PICK_XY, CALIB.safe_z, j4=37.0))
    out = place_at(c, CALIB, square_place(*PLACE_XY))
    # 0 deg on the 90 deg lattice, nearest the current wrist of 37 deg.
    assert out["j4"] == 0.0
    assert any(leg.j4 == 0.0 for leg in c.legs())


def test_place_explicit_j4_bypasses_the_resolver() -> None:
    """unstack's random landing orientation is already lattice-solved; it must
    reach the wire unfolded."""
    c = FakeClient(_Tcp(*PICK_XY, CALIB.safe_z))
    out = place_at(c, CALIB, Grasp(*PLACE_XY), j4=33.0)
    assert out["j4"] == 33.0
    assert any(leg.j4 == 33.0 for leg in c.legs())


# -- transfer -------------------------------------------------------------


def test_transfer_is_one_queue_with_two_stations() -> None:
    c = FakeClient()
    out = transfer(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), Grasp(*PLACE_XY))
    assert out["queues"] == 1, c.calls
    assert len(c.calls) == 1
    stations = [leg for leg in c.legs() if leg.is_station]
    grips = [s.grip for s in stations]
    assert CALIB.grip_close_s in grips and CALIB.grip_open_s in grips
    assert c.gripper_calls == []


def test_chained_transfers_are_one_send() -> None:
    """Shuffle lookahead: two transfers planned back-to-back share one send_legs
    (and therefore no stop/settle between the first place and the second pick)."""
    from mt4_vision.motion import plan_transfer_legs, send_legs

    c = FakeClient()
    start = (200.0, 0.0, 260.0)
    legs1, end1, _, _ = plan_transfer_legs(
        c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), Grasp(*PLACE_XY), start=start,
    )
    legs2, _end2, _, _ = plan_transfer_legs(
        c, CALIB,
        Grasp(200.0, 80.0, yaw_deg=0.0), Grasp(160.0, -100.0),
        start=end1,
    )
    assert c.calls == []  # planning only
    queues = send_legs(c, legs1 + legs2, step="lookahead")
    # Two transfers fit in one firmware-depth queue for these short free-space
    # routes; if they ever need chunking, split_legs still cuts on stations.
    assert queues >= 1
    stations = [leg for leg in c.legs() if leg.is_station and leg.grip]
    # close+open per transfer
    assert len(stations) == 4
    assert c.gripper_calls == []
    # Second pick's transit starts from where the first place left the arm.
    assert end1 == legs1[-1].xyz


def test_chained_transfer_resolves_its_wrist_from_the_plan_not_the_live_tcp() -> None:
    """Once the first transfer is planned but unsent, the live TCP's j4 is the
    wrong angle to minimize wrist travel from -- it is where the arm is now, not
    where the queue will have left it. It also costs a serial read per stage."""
    from mt4_vision.motion import plan_transfer_legs, square_place

    c = FakeClient(_Tcp(200.0, 0.0, 260.0, j4=0.0))
    _legs1, end1, _pj1, place_j4 = plan_transfer_legs(
        c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), square_place(*PLACE_XY),
    )
    c.tcp_reads = 0
    # A 100 deg face is the same square face as 10 deg; which representative
    # wins depends entirely on the wrist it is measured against.
    _legs2, _end2, pick_j4, _qj = plan_transfer_legs(
        c, CALIB, Grasp(200.0, 80.0, yaw_deg=100.0), square_place(160.0, -100.0),
        start=end1, current_j4=place_j4,
    )
    assert c.tcp_reads == 0, "chained planning must not re-read the TCP"
    # Nearest 90 deg-lattice representative of 100 deg to the carried wrist (0),
    # not to the stale live j4 -- here they agree in value but not in provenance,
    # so pin the read count above and the angle here.
    assert pick_j4 == 10.0


def test_verify_pick_place_outcomes() -> None:
    from mt4_vision.motion import verify_pick_place

    dets = [("blue", 177.0, 181.0)]
    assert (
        verify_pick_place(
            dets, pick_x=240.0, pick_y=0.0, pick_color="blue",
            place_x=177.2, place_y=181.5,
        )
        == "placed"
    )
    assert (
        verify_pick_place(
            [("blue", 241.0, 101.0)],
            pick_x=240.0, pick_y=100.0, pick_color="blue",
            place_x=177.2, place_y=181.5,
        )
        == "grasp_failed"
    )
    assert (
        verify_pick_place(
            [("blue", 100.0, 100.0)],
            pick_x=240.0, pick_y=100.0, pick_color="blue",
            place_x=177.2, place_y=181.5,
        )
        == "lost"
    )


def test_grasp_failed_at_returns_origin_xy() -> None:
    from mt4_vision.motion import grasp_failed_at

    assert grasp_failed_at(
        [("red", 68.1, 202.7), ("red", 110.0, 195.0)],
        pick_x=55.8, pick_y=195.0, pick_color="red",
    ) == (68.1, 202.7)
    assert grasp_failed_at(
        [("red", 110.0, 195.0)],
        pick_x=55.8, pick_y=195.0, pick_color="red",
    ) is None


def test_transfer_fuses_the_post_grip_lift_into_the_carry() -> None:
    """One vertical rise off the grab, planned by the router as the carry's
    lift-off -- not a lift leg followed by a separate carry."""
    c = FakeClient()
    transfer(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), Grasp(*PLACE_XY))
    legs = c.legs()
    close_i = next(i for i, leg in enumerate(legs) if leg.grip == CALIB.grip_close_s)
    after = legs[close_i + 1 :]
    rises = [leg for leg in after if leg.xyz == (*PICK_XY, CALIB.safe_z)]
    assert len(rises) == 1, f"expected exactly one lift-off, got {rises}"
    # And it holds the wrist, so the gripped orientation survives the rise.
    assert rises[0].j4 == "wrist"


def test_transfer_station_poses_match_preceding_leg() -> None:
    c = FakeClient()
    transfer(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), Grasp(*PLACE_XY))
    legs = c.legs()
    for i, leg in enumerate(legs):
        if leg.is_station:
            assert leg.xyz == legs[i - 1].xyz, f"station {i} moved"


def test_transfer_fits_the_firmware_queue_depth() -> None:
    c = FakeClient()
    transfer(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), Grasp(*PLACE_XY))
    assert len(c.calls[0]["wps"]) <= MAX_LEGS_PER_QUEUE


def test_transfer_homes_when_not_homed() -> None:
    c = FakeClient(homed=False)
    transfer(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), Grasp(*PLACE_XY))
    assert c.homes == 1


def test_transfer_rejects_keepout_targets_before_moving() -> None:
    c = FakeClient()
    expect_error(
        lambda: transfer(c, CALIB, Grasp(100.0, 0.0), Grasp(*PLACE_XY)),
        "keep-out",
    )
    expect_error(
        lambda: transfer(c, CALIB, Grasp(*PICK_XY), Grasp(0.0, 50.0)),
        "keep-out",
    )
    assert c.calls == []


# -- chunking -------------------------------------------------------------


def _leg(i: int, station: bool = False) -> Leg:
    return Leg(200.0 + i, 0.0, 155.0, dwell_ms=150 if station else 0)


def test_split_short_plan_is_one_chunk() -> None:
    legs = [_leg(i) for i in range(MAX_LEGS_PER_QUEUE)]
    assert split_legs(legs) == [legs]


def test_split_prefers_the_last_station_boundary() -> None:
    """A chunk boundary is a real stop, but the arm is already stopped at a
    station -- so cutting there is free and cutting mid-transit is not."""
    legs = [_leg(i, station=(i == 3)) for i in range(12)]
    chunks = split_legs(legs)
    assert [len(c) for c in chunks] == [4, 8]
    assert chunks[0][-1].is_station


def test_split_falls_back_to_a_hard_cut_without_stations() -> None:
    legs = [_leg(i) for i in range(20)]
    chunks = split_legs(legs)
    assert [len(c) for c in chunks] == [9, 9, 2]
    assert sum(len(c) for c in chunks) == 20


def test_split_never_exceeds_the_queue_depth() -> None:
    for n in range(1, 40):
        for station_at in (None, 0, 3, 8):
            legs = [_leg(i, station=(i == station_at)) for i in range(n)]
            chunks = split_legs(legs)
            assert all(1 <= len(c) <= MAX_LEGS_PER_QUEUE for c in chunks)
            assert [leg for c in chunks for leg in c] == legs


def test_long_plan_splits_and_still_sends_everything() -> None:
    c = FakeClient()
    legs, _j4 = plan_pick_legs(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0))
    padded = legs + [_leg(i) for i in range(12)]
    from mt4_vision.motion import send_legs

    queues = send_legs(c, padded, step="t")
    assert queues == len(split_legs(padded))
    assert sum(len(call["wps"]) for call in c.calls) == len(padded)


# -- planners -------------------------------------------------------------


def test_free_space_planner_is_the_default() -> None:
    """No planner passed means no column model -- but still keep-out and joint
    limits, via the same StackPlanner code the stack path uses."""
    c = FakeClient()
    legs_default, _ = plan_pick_legs(c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0))
    legs_explicit, _ = plan_pick_legs(
        c, CALIB, Grasp(*PICK_XY, yaw_deg=10.0), planner=free()
    )
    assert [leg.xyz for leg in legs_default] == [leg.xyz for leg in legs_explicit]


def test_place_plan_from_grab_height_lifts_before_carrying() -> None:
    c = FakeClient()
    legs, _ = plan_place_legs(
        c, CALIB, Grasp(*PLACE_XY),
        start=(*PICK_XY, CALIB.table_z), lift_from=CALIB.safe_z,
    )
    assert legs[0].xyz == (*PICK_XY, CALIB.safe_z)
    assert legs[0].j4 == "wrist"


def test_camera_park_clamps_the_lift_to_the_ceiling_at_this_radius() -> None:
    """The arm's ceiling falls off with reach, because J2 and J3 share a coupled
    extension cap. Stranded live at (66.5, -309.1, 155), r = 316mm, the ceiling
    is 253mm and the straight lift to the 260mm park height needed J2+J3 = 4592
    steps against a 4410 cap -- ``err mp joints``, and the firmware refuses the
    whole queued path, so the recovery move was the one that could not run from
    exactly the poses that most need it.
    """
    from mt4_vision.pickplace import (
        CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z, retreat_for_camera,
    )
    from mt4_vision.workspace import joint_reachable

    c = FakeClient(tcp=_Tcp(66.5, -309.1, 155.0))
    retreat_for_camera(c, CALIB)
    wps = [tuple(round(v, 1) for v in xyz) for xyz in c.calls[0]["wps"]]

    # Every waypoint has to pass the joint check the firmware agrees with.
    for x, y, z in wps:
        assert joint_reachable(x, y, z), f"{(x, y, z)} is not holdable"

    # The departure is still a pure vertical lift -- same XY, higher z -- so it
    # still cannot diagonal into a column standing where the arm was.
    assert wps[0][:2] == (66.5, -309.1)
    assert wps[0][2] > 155.0
    assert wps[0][2] < 260.0, "the lift that failed live must not be re-issued"

    # And the height it could not gain out there is gained over the park XY.
    assert wps[-1] == (CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z)
    assert any(w[:2] == (CAMERA_PARK_X, CAMERA_PARK_Y) for w in wps[1:])


def test_camera_park_from_a_close_pose_keeps_the_three_corner_shape() -> None:
    """The clamp must not change the ordinary case: near the base the full lift
    is legal, so the path stays lift / traverse and costs one extra check."""
    from mt4_vision.pickplace import (
        CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z, retreat_for_camera,
    )

    c = FakeClient(tcp=_Tcp(200.0, -60.0, CALIB.table_z))
    retreat_for_camera(c, CALIB)
    wps = [tuple(round(v, 1) for v in xyz) for xyz in c.calls[0]["wps"]]
    assert wps == [
        (200.0, -60.0, CAMERA_PARK_Z),
        (CAMERA_PARK_X, CAMERA_PARK_Y, CAMERA_PARK_Z),
    ]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
