"""Quota-driven collection planner tests (no hardware).

The properties that matter are the two the shuffle planner gets structurally
wrong, and which produced the skewed corpus: `to_slot` must stay reachable
while markers are free, and a starved colour must win over an abundant one.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_pi.collect import balance
from mt4_vision.scene import CubeDetection, MarkerSlot, Scene


# Real reachable positions -- Scene.pickable() rejects anything inside the
# 140 mm J1 keep-out or beyond MAX_REACH_MM, and cubes closer together than
# PICK_CLEARANCE_MM, so synthetic coordinates near the origin silently
# produce an empty candidate list. These are taken from a live scene read.
MARKER_XY = (211.3, 7.3)
A_XY = (222.8, 1.8)
B_XY = (195.7, 85.7)
C_XY = (151.0, -132.0)


def cube(color, x, y):
    return CubeDetection(color=color, px=0.0, py=0.0, area=2500.0, x=x, y=y, yaw_deg=0.0)


def scene(cubes, markers=(), free_slots=(), occupied=()):
    return Scene(
        cubes=list(cubes),
        markers=list(markers),
        occupied=list(occupied),
        free_markers=list(markers),
        unknown_markers=[],
        free_slots=[tuple(s) for s in free_slots],
        visible_marker_ids={m.marker_id for m in markers},
    )


# ---------------------------------------------------------------------------
# deficits / counting
# ---------------------------------------------------------------------------


def test_deficits_cover_every_cell_and_clamp_at_zero():
    cells = Counter({("red", "to_marker"): 58, ("blue", "to_slot"): 2})
    need = balance.deficits(cells, target=40)
    assert len(need) == len(balance.COLORS) * len(balance.KINDS) == 6
    assert need[("red", "to_marker")] == 0  # already over target, never negative
    assert need[("blue", "to_slot")] == 38
    assert need[("red", "to_slot")] == 40  # a cell with no episodes at all


def test_count_cells_reads_only_successful_episodes(tmp_path):
    root = tmp_path / "demos"
    eps = root / "episodes"
    eps.mkdir(parents=True)
    def write(name, color, kind, success):
        d = eps / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps(
            {"success": success, "pick": {"color": color}, "place": {"kind": kind}}
        ), encoding="utf-8")
    write("ep_000000", "blue", "to_slot", True)
    write("ep_000001", "blue", "to_slot", False)   # failed -- conversion drops it
    write("ep_000002", "red", "to_marker", True)

    cells = balance.count_cells(root)
    assert cells[("blue", "to_slot")] == 1
    assert cells[("red", "to_marker")] == 1
    assert sum(cells.values()) == 2


def test_count_cells_tolerates_partial_and_missing_roots(tmp_path):
    root = tmp_path / "demos"
    eps = root / "episodes"
    eps.mkdir(parents=True)
    (eps / "ep_000000").mkdir()                       # no meta.json (partial)
    (eps / "ep_000001").mkdir()
    (eps / "ep_000001" / "meta.json").write_text("{not json", encoding="utf-8")
    assert balance.count_cells(root, tmp_path / "absent") == Counter()


# ---------------------------------------------------------------------------
# candidate enumeration -- the structural fix
# ---------------------------------------------------------------------------


def test_slot_moves_stay_available_while_markers_are_free():
    """plan_shuffle only reaches to_slot when NO marker is free, which is why
    the corpus has 7 to_slot episodes against 104 to_marker."""
    s = scene([cube("blue", *A_XY)], markers=[MarkerSlot(0, *MARKER_XY)], free_slots=[C_XY])
    kinds = {c.place_kind for c in balance.candidates(s)}
    assert kinds == {"to_marker", "to_slot"}


def test_candidates_reject_slots_the_camera_cannot_verify():
    """The far PLACEMENT_SLOTS (r = 280-292 mm) are reachable but outside the
    detection hull. Placing there strands the cube AND mislabels a good
    demonstration as failed -- measured, it removed 4 of 7 cubes from the
    table in four consecutive episodes."""
    far = [(240.0, -150.0), (280.0, 0.0), (240.0, 150.0), (150.0, 250.0)]
    s = scene([cube("blue", *A_XY)], free_slots=[*far, (200.0, 60.0)])
    slots = {(c.place_x, c.place_y) for c in balance.candidates(s)}
    assert slots == {(200.0, 60.0)}
    for f in far:
        assert f not in slots


def test_near_slots_stay_inside_the_verifiable_radius():
    """Guards the constant against being tightened past the usable slots."""
    from mt4_vision.workspace import PLACEMENT_SLOTS

    usable = [s for s in PLACEMENT_SLOTS
              if (s[0] ** 2 + s[1] ** 2) ** 0.5 <= balance.MAX_VERIFIABLE_RADIUS_MM]
    assert len(usable) >= 3, f"too few verifiable slots left: {usable}"


def test_candidates_skip_placing_a_cube_on_the_marker_it_occupies():
    m = MarkerSlot(0, *MARKER_XY)
    s = scene([cube("red", *MARKER_XY)], markers=[m], free_slots=[C_XY])
    assert not any(
        c.place_kind == "to_marker" and c.place_marker_id == 0 for c in balance.candidates(s)
    )


# ---------------------------------------------------------------------------
# plan_balanced
# ---------------------------------------------------------------------------


def test_starved_colour_beats_abundant_colour():
    s = scene(
        [cube("red", *A_XY), cube("blue", *B_XY)],
        markers=[MarkerSlot(0, *MARKER_XY)],
    )
    cells = Counter({("red", "to_marker"): 58, ("blue", "to_marker"): 3})
    action = balance.plan_balanced(s, cells, target=40)
    assert action.kind == "pick"
    assert action.cube.color == "blue"


def test_starved_kind_beats_abundant_kind_even_with_a_free_marker():
    s = scene(
        [cube("red", *A_XY)],
        markers=[MarkerSlot(0, *MARKER_XY)],
        free_slots=[C_XY],
    )
    cells = Counter({("red", "to_marker"): 58, ("red", "to_slot"): 0})
    action = balance.plan_balanced(s, cells, target=40)
    assert action.place_kind == "to_slot"
    assert (action.place_x, action.place_y) == C_XY


def test_waits_once_every_cell_is_at_target():
    s = scene([cube("red", *A_XY)], markers=[MarkerSlot(0, *MARKER_XY)], free_slots=[C_XY])
    cells = Counter({(c, k): 40 for c in balance.COLORS for k in balance.KINDS})
    assert balance.plan_balanced(s, cells, target=40).kind == "wait"


def test_waits_when_the_scene_offers_no_move():
    assert balance.plan_balanced(scene([]), Counter(), target=40).kind == "wait"
    # cube present but nowhere to put it
    assert balance.plan_balanced(scene([cube("red", *A_XY)]), Counter(), target=40).kind == "wait"


def test_falls_back_to_a_filler_move_when_nothing_serves_a_needy_cell():
    """Blue is starved but no blue cube is reachable; red is saturated.
    Waiting cannot help -- the scene only changes when the arm moves
    something -- so make a space-freeing move instead of stalling."""
    s = scene([cube("red", *A_XY)], markers=[MarkerSlot(0, *MARKER_XY)], free_slots=[C_XY])
    cells = Counter({("red", "to_marker"): 40, ("red", "to_slot"): 40, ("blue", "to_marker"): 0})
    action = balance.plan_balanced(s, cells, target=40)
    assert action.kind == "pick"
    assert action.place_kind == "to_marker", "filler should free a slot, not fill one"
    assert "filler" in action.reason


def test_filler_can_be_disabled_for_callers_that_prefer_to_wait():
    s = scene([cube("red", *A_XY)], markers=[MarkerSlot(0, *MARKER_XY)], free_slots=[C_XY])
    cells = Counter({("red", "to_marker"): 40, ("red", "to_slot"): 40, ("blue", "to_marker"): 0})
    action = balance.plan_balanced(s, cells, target=40, allow_filler=False)
    assert action.kind == "wait"


def test_just_placed_cube_is_demoted_but_not_excluded():
    """Two cubes serve the same cell; the one just moved should lose."""
    s = scene(
        [cube("blue", *A_XY), cube("blue", *C_XY)],
        markers=[MarkerSlot(0, *MARKER_XY)],
    )
    cells = Counter({("blue", "to_marker"): 0})
    action = balance.plan_balanced(s, cells, target=40, avoid_xy=A_XY)
    assert (action.cube.x, action.cube.y) == C_XY


def test_demotion_never_stalls_a_cell_only_one_cube_can_fill():
    """If the only cube serving the neediest cell is the one just placed,
    move it again rather than returning 'wait' -- a stalled collector
    gathers nothing."""
    s = scene([cube("blue", *A_XY)], markers=[MarkerSlot(0, *MARKER_XY)])
    cells = Counter({("blue", "to_marker"): 0})
    action = balance.plan_balanced(s, cells, target=40, avoid_xy=A_XY)
    assert action.kind == "pick"
    assert action.cube.color == "blue"


def test_action_reason_names_the_cell_and_its_deficit():
    s = scene([cube("blue", *A_XY)], free_slots=[C_XY])
    action = balance.plan_balanced(s, Counter(), target=40)
    assert "to_slot" in action.reason and "blue" in action.reason and "needs 40" in action.reason


# ---------------------------------------------------------------------------
# slot geometry -- the invariant behind occupied-target drops
# ---------------------------------------------------------------------------


def test_collection_slots_are_separated_by_more_than_the_clearance_radius():
    """`free_placement_slots` only marks a slot occupied when a *detected*
    cube is within PLACE_CLEARANCE_MM. Slots packed closer than that make
    correctness depend on never missing a detection -- one occluded or
    sub-threshold cube and the neighbouring slot reads free, so the arm
    drops onto an occupied spot. The first grid had 20 mm spacing and 51
    violating pairs, which is what produced occupied-target drops."""
    import itertools
    import math

    from mt4_vision.workspace import PLACE_CLEARANCE_MM

    pairs = list(itertools.combinations(balance.COLLECTION_SLOTS, 2))
    assert pairs, "need at least two slots"
    closest = min(math.dist(a, b) for a, b in pairs)
    assert closest >= PLACE_CLEARANCE_MM, (
        f"closest slot pair is {closest:.1f}mm apart, inside the "
        f"{PLACE_CLEARANCE_MM}mm clearance radius the free-check uses"
    )
    assert closest >= balance.MIN_SLOT_SEPARATION_MM


def test_every_scatter_destination_stays_verifiable():
    """`unstack_cubes` scatters cubes to a random radius, and it did not
    share this limit -- it drew up to r=300 while the camera tops out at
    240. Each such drop is a one-way trip: the placement reads "lost"
    despite being perfect, and the cube is then invisible to every planner.
    Over one session that drifted 5 of 9 cubes out of the workspace."""
    from unstack_cubes import SCATTER_MAX_RADIUS_MM, SCATTER_MIN_RADIUS_MM

    assert SCATTER_MAX_RADIUS_MM <= balance.MAX_VERIFIABLE_RADIUS_MM
    assert SCATTER_MIN_RADIUS_MM < SCATTER_MAX_RADIUS_MM


def test_the_verifiable_radius_has_one_definition():
    """balance re-exports it; a second literal would let the two drift."""
    from mt4_vision.workspace import MAX_VERIFIABLE_RADIUS_MM

    assert balance.MAX_VERIFIABLE_RADIUS_MM is MAX_VERIFIABLE_RADIUS_MM


def test_collection_slots_lie_inside_the_verifiable_radius():
    import math

    for s in balance.COLLECTION_SLOTS:
        r = math.hypot(*s)
        assert r <= balance.MAX_VERIFIABLE_RADIUS_MM, f"{s} at r={r:.0f} is not verifiable"


def test_collection_slots_are_reachable_and_clear_of_the_keepout():
    import math

    from mt4_vision.workspace import KEEPOUT_RADIUS_MM, is_mp_reachable_xy

    for s in balance.COLLECTION_SLOTS:
        assert math.hypot(*s) > KEEPOUT_RADIUS_MM, f"{s} inside the J1 keep-out"
        assert is_mp_reachable_xy(*s), f"{s} not mp-reachable"
