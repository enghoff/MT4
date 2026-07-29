"""Quota-driven action selection, for collecting the episodes we are missing.

`mt4_vision.policy.plan_shuffle` is the right planner for *shuffling cubes*:
it prefers markers, falls back to slots, and picks uniformly among whatever
happens to be reachable. As a *data collector* it produced a badly skewed
corpus, and for structural reasons rather than bad luck:

  * `to_slot` is only ever reached when there are **no free markers**. With
    5 markers and 7 cubes there almost always is one, so of 110 usable
    shuffle episodes only 7 were `to_slot`.
  * Colour is `random.choice` over pickable cubes, so a colour that spends
    the session sitting on a marker (or off the table) barely appears. Blue
    ended up with 5 episodes against red's 58.

The resulting cells -- red/to_marker 58, green/to_marker 42, green/to_slot 5,
blue/to_marker 3, blue/to_slot 2, red/to_slot 0 -- mean the colour and kind
one-hots in `mt4_pi.act.schema` have almost no support outside two cells. A
policy cannot learn "pick the blue one" from 3 demonstrations.

This planner instead scores every feasible move by how far its (colour, kind)
cell is below target and takes the largest deficit. It deliberately chooses
`to_slot` even when markers are free, which the shuffle planner never does.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mt4_vision.policy import Action
from mt4_vision.scene import CubeDetection, Scene
from mt4_vision.workspace import MAX_VERIFIABLE_RADIUS_MM as _MAX_VERIFIABLE_RADIUS_MM

#: Placement targets beyond this radius are reachable but NOT verifiable.
#: Re-exported from `mt4_vision.workspace`, where it now lives so that every
#: planner that puts a cube down shares one limit -- `unstack_cubes` did not,
#: and scattered to r = 300 until it had drifted most of the cubes out of the
#: camera's reach. `workspace.PLACEMENT_SLOTS` also contains five slots at
#: r = 280-292 mm, which is what motivated the constant here first.
MAX_VERIFIABLE_RADIUS_MM = _MAX_VERIFIABLE_RADIUS_MM

#: Minimum centre-to-centre spacing between collection slots.
#:
#: `free_placement_slots` marks a slot occupied when a *detected* cube is
#: within PLACE_CLEARANCE_MM (45 mm) of it. Slots packed closer than that
#: make the check depend on every neighbouring cube being detected in every
#: frame -- one miss (occlusion, shadow, hull edge, sub-threshold blob) and
#: the adjacent slot reads free, so the arm drops onto an occupied spot.
#:
#: The first COLLECTION_SLOTS grid got this wrong: 20 mm minimum spacing and
#: 51 pairs closer than the clearance radius, against 64 mm / zero for the
#: stock `workspace.PLACEMENT_SLOTS`. Separation above the clearance radius
#: also means an *undetected* cube is physically further from the target, so
#: a miss degrades to a near miss instead of a collision.
MIN_SLOT_SEPARATION_MM = 55.0

#: Placement targets for COLLECTION, replacing `workspace.PLACEMENT_SLOTS`.
#:
#: Only three stock slots survive the verifiable-radius filter, which is too
#: few: the planner burns a filler move freeing a slot for nearly every
#: `to_slot` episode, and every such episode lands on one of three
#: coordinates, so `goal_x/goal_y` gets three values to memorise.
#:
#: Laid out on two polar rings inside the verifiable annulus rather than a
#: Cartesian grid -- the reachable band is only ~90 mm wide radially, so a
#: square grid cannot hold MIN_SLOT_SEPARATION_MM without collapsing to a
#: handful of points. Pre-filtered clear of the current marker layout;
#: `free_placement_slots` still re-checks reachability and live cube
#: clearance every cycle.
#:
#: Kept separate from `workspace.PLACEMENT_SLOTS`, which is shared with
#: calibrate_height.py's probe grid.
COLLECTION_SLOTS: list[tuple[float, float]] = [
    (42.7, -159.4),
    (84.2, 141.9),
    (93.6, 204.6),
    (109.2, -196.7),
    (132.4, -98.4),
    (154.1, 59.0),
    (157.4, -49.4),
    (206.7, 88.9),
    (212.9, -72.6),
]

#: The cells we want filled. Kinds are limited to the two the shuffle
#: collector can produce; stack/unstack come from a different script and
#: currently have a recorder defect (see docs/ACT_PIPELINE.md).
KINDS = ("to_marker", "to_slot")
COLORS = ("red", "green", "blue")


@dataclass(frozen=True)
class Candidate:
    cube: CubeDetection
    place_x: float
    place_y: float
    place_kind: str
    place_marker_id: int | None

    @property
    def cell(self) -> tuple[str, str]:
        return (self.cube.color, self.place_kind)


def count_cells(*roots: Path) -> Counter:
    """Tally successful episodes per (colour, kind) across collection roots.

    Read from disk rather than tracked in memory so a resumed or restarted
    run continues filling the same gaps instead of re-balancing from zero.
    Only successful episodes count -- a failed grasp is not a demonstration,
    and `convert`/`export` drop them, so counting them would leave the real
    corpus short.
    """
    cells: Counter = Counter()
    for root in roots:
        episodes = root / "episodes"
        if not episodes.is_dir():
            continue
        for ep in episodes.glob("ep_*"):
            meta_path = ep / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not meta.get("success"):
                continue
            color = (meta.get("pick") or {}).get("color")
            kind = (meta.get("place") or {}).get("kind")
            if color in COLORS and kind in KINDS:
                cells[(color, kind)] += 1
    return cells


def deficits(cells: Counter, target: int) -> dict[tuple[str, str], int]:
    return {
        (c, k): max(0, target - cells.get((c, k), 0)) for c in COLORS for k in KINDS
    }


def candidates(
    scene: Scene,
    *,
    exclude_cube: CubeDetection | None = None,
    slots: list[tuple[float, float]] | None = None,
) -> list[Candidate]:
    """Every (cube, destination) pair this scene actually supports.

    Unlike `plan_shuffle`, marker and slot destinations are enumerated
    together rather than as a priority chain, so a `to_slot` move stays
    available while markers are free.

    `slots` overrides the scene's own free slots -- the collector passes the
    COLLECTION_SLOTS grid, re-checked for clearance against the live scene.
    Defaults to `scene.free_slots` so the function stays usable (and
    testable) without a Calibration.
    """
    free_slots = scene.free_slots if slots is None else slots
    pickable = [c for c in scene.pickable(scene.cubes) if c is not exclude_cube]
    out: list[Candidate] = []
    for cube in pickable:
        for marker in scene.placeable_markers():
            # Don't "move" a cube onto the marker it already sits on.
            if abs(marker.x - cube.x) < 1e-6 and abs(marker.y - cube.y) < 1e-6:
                continue
            out.append(Candidate(cube, marker.x, marker.y, "to_marker", marker.marker_id))
        for sx, sy in free_slots:
            if math.hypot(float(sx), float(sy)) > MAX_VERIFIABLE_RADIUS_MM:
                continue  # placeable but unverifiable -- see the constant
            out.append(Candidate(cube, float(sx), float(sy), "to_slot", None))
    return out


def plan_balanced(
    scene: Scene,
    cells: Counter,
    *,
    target: int,
    avoid_xy: tuple[float, float] | None = None,
    allow_filler: bool = True,
    slots: list[tuple[float, float]] | None = None,
    rng: random.Random | None = None,
) -> Action:
    """Choose the feasible move whose (colour, kind) cell is furthest behind.

    `avoid_xy` demotes the cube just placed there, so the collector does not
    shuttle one cube back and forth and call it 30 episodes of data -- the
    visual variety across episodes is the point.
    """
    rng = rng or random
    options = candidates(scene, slots=slots)
    if not options:
        return Action("wait", f"no feasible pick/place pair ({scene.summary_line()})")

    need = deficits(cells, target)
    if not any(need.values()):
        return Action("wait", f"all cells at target {target}")

    def score(cand: Candidate) -> tuple[int, float]:
        d = need[cand.cell]
        # Demote, don't exclude: if the only cube that fills the neediest
        # cell is the one we just moved, moving it again beats stalling.
        penalty = 0.0
        if avoid_xy is not None:
            dx, dy = cand.cube.x - avoid_xy[0], cand.cube.y - avoid_xy[1]
            if (dx * dx + dy * dy) ** 0.5 < 25.0:
                penalty = 0.5
        return (d, -penalty)

    best = max(score(c) for c in options)
    if best[0] == 0:
        # Nothing on offer serves a needy cell. Waiting cannot fix that --
        # the scene only changes when the arm moves something -- and with
        # just three verifiable slots this is reachable: fill all three and
        # the remaining `to_slot` deficits become momentarily infeasible.
        #
        # So make a move that changes availability instead of stalling. A
        # `to_marker` move is the useful one: it lifts a cube off a slot (or
        # off the open table) and frees a placement target for the next
        # cycle. The episode lands in an already-satisfied cell, which costs
        # one recording and unsticks the run.
        if not allow_filler:
            return Action(
                "wait",
                f"no move serves an unfilled cell (needs {sorted(k for k, v in need.items() if v)})",
            )
        freeing = [c for c in options if c.place_kind == "to_marker"] or options
        pick = rng.choice(freeing)
        return Action(
            "pick",
            f"filler {pick.place_kind}: {pick.cube.color} "
            f"({pick.cube.x:.0f},{pick.cube.y:.0f}) -> ({pick.place_x:.0f},{pick.place_y:.0f}) "
            f"[no feasible move for {sorted(k for k, v in need.items() if v)}; freeing space]",
            cube=pick.cube,
            place_x=pick.place_x,
            place_y=pick.place_y,
            place_marker_id=pick.place_marker_id,
            place_kind=pick.place_kind,
        )
    tied = [c for c in options if score(c) == best]
    pick = rng.choice(tied)

    return Action(
        "pick",
        f"{pick.place_kind}: {pick.cube.color} ({pick.cube.x:.0f},{pick.cube.y:.0f}) "
        f"-> ({pick.place_x:.0f},{pick.place_y:.0f}) "
        f"[cell {pick.cell} needs {need[pick.cell]}]",
        cube=pick.cube,
        place_x=pick.place_x,
        place_y=pick.place_y,
        place_marker_id=pick.place_marker_id,
        place_kind=pick.place_kind,
    )


def format_cells(cells: Counter, target: int) -> str:
    rows = []
    for c in COLORS:
        for k in KINDS:
            n = cells.get((c, k), 0)
            rows.append(f"{c}/{k}={n}{'*' if n < target else ''}")
    return "  ".join(rows)
