"""Reading Qwen's replies: coordinate conventions and grounding.

Split from :mod:`mt4_vision.instruct`, which imports everything here and
re-exports it, so ``from mt4_vision.instruct import to_frame_pixels`` still
works. Two jobs live here:

* **Which space a coordinate is in.** A grounding reply is 0-1000 normalized; a
  decision reply is pixels, because the decision prompt prints every entity's
  own pixel. :func:`to_frame_pixels` and :func:`point_readings` are the only
  places either convention is applied.
* **Turning "where is the X" into something measurable.** :func:`locate_target`
  asks for one box, :func:`measure_grounding` segments it into a
  ``LocatedObject`` with real millimetres.

``obs`` parameters are ``instruct.Observation``, kept untyped at runtime so this
module does not import back into the one that imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mt4_vision.preview import BIG_BOX_SHARE
from mt4_vision.qwen import DEFAULT_URL, QwenError, ask

if TYPE_CHECKING:
    from mt4_vision.instruct import Observation


# The model's coordinate space. Not pixels -- see the module docstring.
COORD_SCALE = 1000.0


def to_frame_pixels(
    coords: Sequence[float], size: tuple[int, int]
) -> tuple[float, ...]:
    """Model coordinates as frame pixels. The only place this convention lives.

    Takes a point ``(x, y)`` or a box ``(x1, y1, x2, y2)`` -- even indices are
    x, odd are y -- and returns the same shape.

    **An unanchored coordinate from this build is 0-1000 normalized**, whatever
    the prompt asks for, so that is what this scales. Use it for
    :func:`locate_target`'s grounding replies. A *decision* reply is different:
    ``build_prompt`` prints every entity's own pixel, and with that example in
    front of it the model answers in pixels, so ``decide`` reads its point
    through :func:`point_readings` instead.

    The only reading the numbers alone can rule out is normalized, since nothing
    normalized exceeds ``COORD_SCALE``. A coordinate above it is therefore
    pixels and is passed through; everything else is scaled. That leaves a blind
    spot -- a true pixel answer whose coordinates all fall under 1000 -- so
    callers that can afford a retry should offer :func:`alternate_reading` when
    the first reading fails to measure.
    """
    values = [float(c) for c in coords]
    if any(abs(v) > COORD_SCALE for v in values):
        return tuple(values)
    w, h = size
    sx, sy = w / COORD_SCALE, h / COORD_SCALE
    return tuple(v * (sx if i % 2 == 0 else sy) for i, v in enumerate(values))


def point_readings(
    point: Sequence[float], size: tuple[int, int]
) -> tuple[tuple[float, float], ...]:
    """Both in-frame readings of a **decision** reply's point, pixels first.

    The convention is a property of the prompt, and this module's two prompts
    differ: :func:`locate_target` says nothing about coordinate space and gets
    0-1000 normalized back, while :func:`build_prompt` prints every entity's own
    pixel and gets pixels back. Pixels therefore lead here, but only lead -- both
    readings are returned and the caller resolves them against the snapshot.

    Readings outside the frame are dropped, and a coordinate above
    ``COORD_SCALE`` rules normalized out, leaving a single reading.
    """
    try:
        values = [float(point[0]), float(point[1])]
    except (TypeError, ValueError, IndexError):
        return ()
    w, h = size
    out: list[tuple[float, float]] = []

    def keep(candidate: tuple[float, float] | None) -> None:
        if candidate is None:
            return
        if not (0 <= candidate[0] <= w and 0 <= candidate[1] <= h):
            return
        if candidate not in out:
            out.append(candidate)

    keep((values[0], values[1]))
    scaled = to_frame_pixels(values, size)
    keep((float(scaled[0]), float(scaled[1])))
    return tuple(out)


def alternate_reading(
    coords: Sequence[float], size: tuple[int, int]
) -> tuple[float, ...] | None:
    """The other possible reading of the same numbers, or None if there is none.

    Only ever raw-as-pixels, and only when that lands inside the frame:
    :func:`to_frame_pixels` already returns raw pixels whenever a coordinate
    rules normalized out, so in that case there is nothing else to try.

    This exists so a failed measurement can be retried rather than reported,
    without ever *silently* choosing between the two -- the retry still has to
    survive segmentation, the two-window stability check and the work-region
    gate before it becomes a target.
    """
    values = [float(c) for c in coords]
    if any(abs(v) > COORD_SCALE for v in values):
        return None
    w, h = size
    if all(0 <= v <= w for v in values[0::2]) and all(
        0 <= v <= h for v in values[1::2]
    ):
        return tuple(values)
    return None


@dataclass(frozen=True)
class Grounding:
    """Where the model says a named thing is, in frame pixels.

    ``box_px`` is the whole point of asking for a box rather than a point, and
    it buys three things a point cannot:

    * **GrabCut.** ``locate.measure_grabcut`` seeds a silhouette from the box.
      Measured on one live frame, the desk-deviation path that a bare point
      feeds segmented 1 of 4 objects; from the box, GrabCut segmented 4 of 4,
      and landed 6.3-12.4mm from where the HSV cube detector puts the same
      three cubes.
    * **A bound on the mask.** Desk-deviation floods into shadow and adjacent
      objects with nothing to stop it; the box says how far the object goes.
    * **A size check.** A box has an extent, so a reading that puts a stapler
      at 400mm long can be rejected before the arm moves. A point has no
      extent and cannot be sanity-checked at all.

    ``alt_point_px`` / ``alt_box_px`` are the same reply under the other
    coordinate convention -- see :func:`alternate_reading`. They are a retry,
    never a silent second choice.
    """

    label: str
    point_px: tuple[float, float]
    box_px: tuple[float, float, float, float] | None = None
    alt_point_px: tuple[float, float] | None = None
    alt_box_px: tuple[float, float, float, float] | None = None


def locate_target(
    obs: Observation, noun: str, *, url: str = DEFAULT_URL
) -> tuple[Grounding | None, str]:
    """Ask only "where is the <noun>", and return a :class:`Grounding` or why not.

    A separate, single-purpose call rather than another field on the decision
    prompt. Asked to choose an action *and* ground an unknown noun, this build
    reliably does neither -- it forces the task onto whatever cube it can see
    and explains in its reason why that is wrong. Asked nothing but "locate
    the stapler", with the keys spelled out, it is the grounding prompt that
    measured 10/10 (docs/QWEN3-VL.md).

    **The prompt deliberately says nothing about the coordinate space.** The
    model ignores that instruction either way (see :func:`to_frame_pixels`),
    and a stated convention it does not follow is worse than no statement at
    all -- it invites the reader to trust the wrong reading.
    """
    prompt = (
        f"Locate the {noun} in this image. If there is no {noun} visible, "
        'reply with an empty list [].\n'
        "Reply with ONLY JSON, no prose, no markdown fence:\n"
        f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{noun}"}}]\n'
        "One tight box around the whole object."
    )
    try:
        reply = ask(prompt, obs.annotated, url=url, max_new_tokens=120, do_sample=False)
    except QwenError as exc:
        return None, f"grounding call failed: {exc}"

    from mt4_vision.qwen import parse_regions

    regions = parse_regions(reply.text)
    # Boxes first: a point reply is still accepted, because the model
    # occasionally answers with one anyway, but it loses everything in the
    # Grounding docstring and is the weaker input.
    for r in sorted(regions, key=lambda g: 0 if g.kind == "box" else 1):
        prim = to_frame_pixels(r.coords, obs.size)
        alt = alternate_reading(r.coords, obs.size)
        g = _grounding(noun, r.kind, prim, alt, obs.size)
        if g is not None:
            return g, ""
    return None, f"no usable {noun} location in {reply.text.strip()[:120]!r}"


# A box larger than this share of the frame is the model declining to answer,
# not an object. Asked to locate something that is not there, this build
# sometimes returns the whole image rather than the empty list the prompt asks
# for -- measured on "location", box (0, 0, 1000, 1000), 100% of the frame.
#
# The plausibility band in ``locate.measure_box`` caught that one, but only
# after projecting it: a box spanning the frame includes pixels above the desk
# horizon, where this camera's homography diverges, so the refusal read "box
# measures 315037x312990mm". True, useless, and three layers from the cause.
#
# The threshold has room. The largest thing the stack will measure is
# ``MAX_PLAUSIBLE_LONG_MM`` = 200mm; at this mounting that is roughly a third of
# the frame's width, so well under a tenth of its area.
# THE threshold the overlay calls out at, not a copy: the picture and the
# refusal have to agree about what counts as "the model declined to answer".
MAX_BOX_FRAME_SHARE = BIG_BOX_SHARE


def _grounding(
    label: str,
    kind: str,
    prim: Sequence[float],
    alt: Sequence[float] | None,
    size: tuple[int, int],
) -> Grounding | None:
    """Assemble a Grounding, or None when the reading cannot be an object.

    Two rejections, both about the primary reading: a centre outside the frame,
    and a box covering most of it (see :data:`MAX_BOX_FRAME_SHARE`).
    """
    w, h = size

    def centre(c: Sequence[float]) -> tuple[float, float]:
        if len(c) == 2:
            return float(c[0]), float(c[1])
        return (float(c[0]) + float(c[2])) / 2, (float(c[1]) + float(c[3])) / 2

    def box(c: Sequence[float]) -> tuple[float, float, float, float] | None:
        if len(c) != 4:
            return None
        x1, y1, x2, y2 = (float(v) for v in c)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    cx, cy = centre(prim)
    if not (0 <= cx <= w and 0 <= cy <= h):
        return None
    prim_box = box(prim) if kind == "box" else None
    if prim_box is not None and w > 0 and h > 0:
        bx0, by0, bx1, by1 = prim_box
        if (bx1 - bx0) * (by1 - by0) > MAX_BOX_FRAME_SHARE * w * h:
            return None
    return Grounding(
        label=label,
        point_px=(cx, cy),
        box_px=prim_box,
        alt_point_px=None if alt is None else centre(alt),
        alt_box_px=None if alt is None or kind != "box" else box(alt),
    )


def measure_grounding(
    obs: Observation, g: Grounding, *, label: str | None = None
) -> tuple[Any | None, str]:
    """Turn a :class:`Grounding` into a measured object, or say why not.

    Prefers GrabCut from the box and falls back to the desk-deviation point
    path, which is what ``locate.measure_with_box_fallback`` already arranges.
    On failure it retries under the other coordinate reading rather than
    reporting -- the retry still has to survive segmentation, the two-window
    stability check and the work-region gate, so a wrong reading cannot buy
    itself a target by being tried twice.
    """
    from mt4_vision.locate import LocateError, measure_with_box_fallback

    name = label or g.label
    marker_xy = [(e.x, e.y) for e in obs.snapshot.entities if e.kind == "marker"]
    attempts: list[tuple[str, tuple[float, float], tuple[float, ...] | None]] = [
        ("", g.point_px, g.box_px)
    ]
    if g.alt_point_px is not None and g.alt_point_px != g.point_px:
        attempts.append(("alternate coordinate reading: ", g.alt_point_px, g.alt_box_px))

    why = ""
    for prefix, point, bx in attempts:
        try:
            obj = measure_with_box_fallback(
                obs.frame, point[0], point[1], obs.calib, name,
                box=None if bx is None else (bx[0], bx[1], bx[2], bx[3]),
                marker_xy=marker_xy,
            )
        except LocateError as exc:
            why = why or f"{prefix}{exc}"
            continue
        return obj, ""
    return None, why or f"nothing measurable at the {name} location"
