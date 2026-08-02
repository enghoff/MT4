"""Stack-aware path planning: joint-limit ceilings and column keep-away.

Pure geometry -- no serial I/O. ``stack_cubes.py`` turns these plans into
firmware ``mp`` moves.

Two obstacle models guard a stack of ``levels`` cubes at (sx, sy):

* **Column cylinder** -- fingertips reach roughly table level when the TCP
  is at ``table_z``, so below ``grip-top + clearance`` the TCP must stay
  ``COLUMN_AVOID_MM`` from the stack axis in XY.
* **Forearm wedge** -- every arm link sits on the base side of the TCP
  (HEAD_OFFSET points the head away from the wrist), so the forearm can
  only cross the column when the TCP is at or beyond the stack's base
  distance near the stack's bearing. It clears once the TCP is high
  enough; ``FOREARM_RISE_PER_MM`` is a conservative slope taken from FK
  poses (TCP at r=260, z=155 puts the forearm at ~179mm over r=211).

Heights are TCP-frame like the calibration: ``table_z`` grips a cube on the
table, so a TCP at ``table_z + k*cube_height`` grips cube ``k`` of a stack,
and fingertips clear an ``L``-cube stack once the TCP is above
``table_z + L*cube_height`` plus margin.
"""

from __future__ import annotations

import heapq
import math

from mt4_jog.joints import GROUND_Z_MM
from mt4_vision.workspace import joint_reachable, max_z_at

# ``joint_reachable`` and ``max_z_at`` moved to workspace so the work-region
# predicate could use them without workspace importing this module (which
# imports workspace). Re-exported here because that is where they have always
# been imported from.
__all__ = ["joint_reachable", "max_z_at", "StackPlanner"]

# Heights (mm, TCP frame -- relative to release_z / stack grip-top).
RELEASE_ABOVE_MM = 4.0        # drop height above the seated stack top
HOVER_ABOVE_MIN_MM = 8.0      # min carry clearance over the stack top
HOVER_ABOVE_MAX_MM = 25.0     # preferred carry clearance when reachable
FREE_RETREAT_ABOVE_MM = 24.0  # floor: fingertips clear the placed cube by ~8mm
FREE_RETREAT_PREFER_MM = 36.0  # preferred: ~20mm fingertip margin (lean-proof)
SLIDE_LIFT_ABOVE_MM = 12.0    # slide-out with fingers straddling the top cube
Z_LIMIT_MARGIN_MM = 8.0       # stay this far under the joint-limit z ceiling

# XY keep-away around the stack axis.
COLUMN_AVOID_MM = 55.0
STAGE_OFFSET_MM = 60.0        # hover entry / exit points sit here
ROUTE_RING_MM = COLUMN_AVOID_MM + 15.0
CARRY_CLEAR_ABOVE_MM = 4.0    # inside the cylinder, z must beat grip-top by this
# Detour waypoints around the J1 keep-out cylinder: straight chords between
# picks and stages on opposite bearings dip inside the keep-out (field case
# 2026-07-24, marker 2: (59,179)->(160,-90) bottoms out at r~121). The
# firmware would arc around it on its own, but host-side planning must
# produce chords that stay clear so the checked path is the executed path.
# 30-degree spacing keeps chords at r >= 160*cos(15) ~ 154.5.
BASE_RING_MM = 160.0
BASE_RING_STEP_DEG = 30.0

# Forearm-over-column model (conservative slope, see module docstring).
FOREARM_RADIAL_SLACK_MM = 15.0
FOREARM_LATERAL_MM = 40.0
FOREARM_RISE_PER_MM = 0.4
FOREARM_MARGIN_MM = 5.0

SEGMENT_SAMPLE_MM = 10.0

XYZ = tuple[float, float, float]


class StackPlanner:
    """Height and route planning around a growing stack at (sx, sy).

    The site is optional: ``StackPlanner.free_space(calib)`` (or sx/sy None)
    gives a planner with no column to avoid, whose ``route()`` still detours
    around the J1 keep-out and still checks every sampled pose against IK and
    the joint soft limits. That is what a plain point-to-point move wants, and
    routing it through this same class -- rather than a second, parallel
    free-space router -- is deliberate: ``pickplace.routed_travel`` requires
    stack and unstack to share one safety model, and a duplicate implementation
    is how that guarantee rots. With a site, behaviour is unchanged.

    The stack-specific height and staging helpers (``grip_top_z``,
    ``hover_z``, ``stage_point``, ``slide_exits``, ...) require a site and say
    so rather than returning a plausible-looking number.
    """

    def __init__(
        self, calib, sx: float | None = None, sy: float | None = None
    ) -> None:
        self.sx = None if sx is None else float(sx)
        self.sy = None if sy is None else float(sy)
        self.table_z = float(calib.table_z)
        self.cube_h = float(calib.cube_height_mm)
        self.safe_z = float(calib.safe_z)
        self.site_max_z = (
            (max_z_at(self.sx, self.sy) or GROUND_Z_MM)
            if self.has_site
            else GROUND_Z_MM
        )

    @classmethod
    def free_space(cls, calib) -> StackPlanner:
        """A planner with no stack column -- keep-out and joint limits only."""
        return cls(calib)

    @property
    def has_site(self) -> bool:
        return self.sx is not None and self.sy is not None

    def _require_site(self, what: str) -> None:
        if not self.has_site:
            raise ValueError(
                f"StackPlanner.{what} needs a stack site; this planner was "
                "built for free space (StackPlanner.free_space)"
            )

    # -- heights -----------------------------------------------------------

    def grip_top_z(self, levels: int) -> float:
        """TCP height whose fingertips sit level with a ``levels``-cube top."""
        return self.table_z + levels * self.cube_h

    def release_z(self, level: int) -> float:
        return self.grip_top_z(level - 1) + RELEASE_ABOVE_MM

    def hover_z(self, level: int) -> float | None:
        """Carry height over the stack for placing ``level``, or None.

        As much clearance as fits under the joint-limit ceiling, up to
        HOVER_ABOVE_MAX_MM; a level is buildable only if at least
        HOVER_ABOVE_MIN_MM fits.
        """
        self._require_site("hover_z")
        rz = self.release_z(level)
        hz = min(rz + HOVER_ABOVE_MAX_MM, self.site_max_z - Z_LIMIT_MARGIN_MM)
        return hz if hz >= rz + HOVER_ABOVE_MIN_MM else None

    def free_retreat_z(self, level: int) -> float | None:
        """Height lifting the fingertips clear of the placed cube, or None
        when the ceiling is too low to lift free (slide out instead).

        Prefers FREE_RETREAT_PREFER_MM (~20mm fingertip margin, so a
        leaning column cannot be clipped by the exit hop) and accepts down
        to FREE_RETREAT_ABOVE_MM before switching to the slide retreat.
        """
        self._require_site("free_retreat_z")
        rz = self.release_z(level)
        fz = min(
            rz + FREE_RETREAT_PREFER_MM, self.site_max_z - Z_LIMIT_MARGIN_MM
        )
        if fz >= rz + FREE_RETREAT_ABOVE_MM:
            return fz
        return None

    def slide_z(self, level: int) -> float:
        """Slide-out height: fingers straddle the upper half of the placed
        cube, so a move perpendicular to the jaw axis sweeps them off the
        cube faces without touching it."""
        self._require_site("slide_z")
        rz = self.release_z(level)
        return min(rz + SLIDE_LIFT_ABOVE_MM, self.site_max_z - Z_LIMIT_MARGIN_MM)

    # -- safety model ------------------------------------------------------

    def pose_safe(self, x: float, y: float, z: float, levels: int) -> bool:
        """True when a TCP pose cannot touch a ``levels``-cube stack.

        With no site (free space) this is exactly the ``levels <= 0`` case that
        already existed: reachability, IK branch, and joint soft limits, with
        no column or forearm model to apply.
        """
        if not joint_reachable(x, y, z):
            return False
        if not self.has_site or levels <= 0:
            return True
        if (
            math.hypot(x - self.sx, y - self.sy) < COLUMN_AVOID_MM
            and z < self.grip_top_z(levels) + CARRY_CLEAR_ABOVE_MM
        ):
            return False
        r = math.hypot(x, y)
        rs = math.hypot(self.sx, self.sy)
        if r > rs - FOREARM_RADIAL_SLACK_MM:
            # Lateral distance from the stack center to the arm's vertical
            # plane (through the origin at the TCP bearing).
            lateral = abs(x * self.sy - y * self.sx) / max(r, 1e-6)
            if lateral < FOREARM_LATERAL_MM:
                forearm_z = (
                    z
                    + FOREARM_RISE_PER_MM * max(r - rs, 0.0)
                    - FOREARM_MARGIN_MM
                )
                if forearm_z < self.grip_top_z(levels) + FOREARM_MARGIN_MM:
                    return False
        return True

    def segment_safe(self, a: XYZ, b: XYZ, levels: int) -> bool:
        """Sampled pose_safe along a->b (start excluded, end included).

        The start is the arm's current pose and may be formally "unsafe"
        (e.g. hovering over the stack itself); only where the move goes
        matters.
        """
        length = math.dist(a, b)
        n = max(1, math.ceil(length / SEGMENT_SAMPLE_MM))
        for i in range(1, n + 1):
            t = i / n
            if not self.pose_safe(
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                levels,
            ):
                return False
        return True

    # -- stages and routing ------------------------------------------------

    def _ring(self, radius: float) -> list[tuple[float, float]]:
        """Eight points around the stack, index 0 toward the base.

        Empty with no site: there is no column to ring, so ``_route_via``
        falls through to the J1 keep-out base ring alone.
        """
        if not self.has_site:
            return []
        rs = math.hypot(self.sx, self.sy)
        ux, uy = -self.sx / rs, -self.sy / rs
        out: list[tuple[float, float]] = []
        for k in range(8):
            ang = math.radians(45.0 * k)
            ca, sa = math.cos(ang), math.sin(ang)
            out.append(
                (
                    self.sx + (ux * ca - uy * sa) * radius,
                    self.sy + (ux * sa + uy * ca) * radius,
                )
            )
        return out

    def stage_point(
        self,
        z: float,
        levels: int,
        *,
        prefer_xy: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Hover entry/exit point STAGE_OFFSET_MM from the axis, nearest to
        ``prefer_xy``. Requires a little z headroom beyond ``z`` so the
        firmware's own per-segment IK cannot reject it by a hair."""
        self._require_site("stage_point")
        cands = [
            p
            for p in self._ring(STAGE_OFFSET_MM)
            if self.pose_safe(p[0], p[1], z, levels)
            and joint_reachable(p[0], p[1], z + 3.0)
        ]
        if not cands:
            return None
        return min(
            cands,
            key=lambda p: math.hypot(p[0] - prefer_xy[0], p[1] - prefer_xy[1]),
        )

    def slide_exits(
        self,
        jaw_world_deg: float,
        level: int,
        *,
        prefer_xy: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Slide-out endpoints perpendicular to the jaw axis, best first.

        After release the open fingers still straddle the placed cube; the
        only safe low retreat is a straight slide perpendicular to the jaw
        axis so both fingers sweep off the cube faces without pushing it.
        """
        self._require_site("slide_exits")
        sz = self.slide_z(level)
        ranked: list[tuple[bool, float, tuple[float, float]]] = []
        for ang in (jaw_world_deg + 90.0, jaw_world_deg - 90.0):
            rad = math.radians(ang)
            ex = (
                self.sx + math.cos(rad) * STAGE_OFFSET_MM,
                self.sy + math.sin(rad) * STAGE_OFFSET_MM,
            )
            if not joint_reachable(ex[0], ex[1], sz):
                continue
            unsafe = not self.pose_safe(ex[0], ex[1], sz, level)
            d = math.hypot(ex[0] - prefer_xy[0], ex[1] - prefer_xy[1])
            ranked.append((unsafe, d, ex))
        ranked.sort(key=lambda t: (t[0], t[1]))
        return [t[2] for t in ranked]

    def _base_ring(self) -> list[tuple[float, float]]:
        """Waypoints around the J1 keep-out (unreachable bearings drop out
        via pose_safe's joint check)."""
        out: list[tuple[float, float]] = []
        ang = 0.0
        while ang < 360.0:
            rad = math.radians(ang)
            out.append((BASE_RING_MM * math.cos(rad), BASE_RING_MM * math.sin(rad)))
            ang += BASE_RING_STEP_DEG
        return out

    def route(self, a: XYZ, b: XYZ, levels: int) -> list[XYZ] | None:
        """Waypoints from a to b (excluding a, ending with b), or None.

        Straight when safe (one diagonal ``mp``, the common case);
        otherwise a shortest path over detour waypoints around the stack
        column and the J1 keep-out at a transit height, trying the higher
        endpoint height first, then ``safe_z`` (closer to the base the z
        ceiling drops, so low transits have more joint headroom there).
        """
        if self.segment_safe(a, b, levels):
            return [b]
        z_options: list[float] = []
        for z in (max(a[2], b[2]), self.safe_z):
            if z not in z_options:
                z_options.append(z)
        for via_z in z_options:
            path = self._route_via(a, b, via_z, levels)
            if path is not None:
                return path
        return None

    def _route_via(
        self, a: XYZ, b: XYZ, via_z: float, levels: int
    ) -> list[XYZ] | None:
        """Dijkstra over {start, stack ring, base ring, goal} at ``via_z``."""
        start: XYZ = (a[0], a[1], via_z)
        pre: list[XYZ] = []
        if abs(via_z - a[2]) > 0.5:
            if not self.segment_safe(a, start, levels):
                return None
            pre = [start]
        mids: list[XYZ] = [
            (x, y, via_z)
            for x, y in self._ring(ROUTE_RING_MM) + self._base_ring()
            if self.pose_safe(x, y, via_z, levels)
        ]
        nodes: list[XYZ] = [start] + mids + [b]
        goal = len(nodes) - 1
        best = {0: 0.0}
        prev: dict[int, int] = {}
        heap: list[tuple[float, int]] = [(0.0, 0)]
        done: set[int] = set()
        while heap:
            d, i = heapq.heappop(heap)
            if i in done:
                continue
            done.add(i)
            if i == goal:
                order: list[int] = []
                k = goal
                while k != 0:
                    order.append(k)
                    k = prev[k]
                order.reverse()
                return pre + [nodes[k] for k in order]
            for j in range(1, len(nodes)):
                if j in done or j == i:
                    continue
                if not self.segment_safe(nodes[i], nodes[j], levels):
                    continue
                nd = d + math.dist(nodes[i], nodes[j])
                if nd < best.get(j, math.inf):
                    best[j] = nd
                    prev[j] = i
                    heapq.heappush(heap, (nd, j))
        return None
