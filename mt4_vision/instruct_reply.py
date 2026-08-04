"""Reading Qwen's replies: coordinate conventions and measurement.

Split from :mod:`mt4_vision.instruct`, which imports everything here and
re-exports it, so ``from mt4_vision.instruct import to_frame_pixels`` still
works. Two jobs live here:

* **Which space a coordinate is in.** The decision prompt asks for pixels and
  draws the grid to read them off, but this build answers 0-1000 normalized
  often enough that both readings have to be kept. :func:`to_frame_pixels`,
  :func:`point_readings` and :func:`box_readings` are the only places either
  convention is applied, and they order their two readings identically.
* **Turning the box the model drew into something measurable.**
  :func:`box_grounding` validates it, :func:`measure_grounding` segments it
  into a ``LocatedObject`` with real millimetres. :func:`report_groundings`
  does the first of those for a REPORT's whole list at once.

There is **one** grounding call per decision, and it is the decision itself:
the reply carries the boxes, and those boxes are what get measured. No separate
"where is the X" pass runs first, because there is no entity list for an
unlisted noun to be grounded into. See the :mod:`mt4_vision.instruct` module
docstring.

``obs`` parameters are ``instruct.Observation``, kept untyped at runtime so this
module does not import back into the one that imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mt4_vision.preview import BIG_BOX_SHARE

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

    **A coordinate from this model is 0-1000 normalized**, scaled per axis
    against the ORIGINAL frame size -- ``x * w / 1000``, ``y * h / 1000``, which
    for a non-square frame is two different factors. That is what Qwen document
    (``cookbooks/spatial_understanding.ipynb``) and what every third-party
    write-up does, and it is what this measures: over 3 targets x 2 prompt
    styles on one 1280x720 frame, the normalized reading lands 2-13px from truth
    and the raw-pixel reading 264-363px away, 6 times out of 6.

    The only reading the numbers alone can rule out is normalized, since nothing
    normalized exceeds ``COORD_SCALE``. A coordinate above it is therefore
    pixels and is passed through; everything else is scaled. Raw pixels survive
    as a retry, not a default -- see :func:`point_readings`.
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

    Used for ``dest_2d``, the one bare point left in the protocol. The
    normalized reading leads because that is the space this model answers in --
    see :func:`to_frame_pixels` for the measurement. Raw pixels are kept as a
    retry rather than dropped: a coordinate is only *probably* normalized, and
    the retry costs a segmentation attempt where being wrong costs a grasp.

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

    scaled = to_frame_pixels(values, size)
    keep((float(scaled[0]), float(scaled[1])))
    keep((values[0], values[1]))
    return tuple(out)


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
    coordinate convention -- see :func:`box_readings`. They are a retry, never
    a silent second choice.
    """

    label: str
    point_px: tuple[float, float]
    box_px: tuple[float, float, float, float] | None = None
    alt_point_px: tuple[float, float] | None = None
    alt_box_px: tuple[float, float, float, float] | None = None


def box_readings(
    coords: Sequence[float], size: tuple[int, int]
) -> tuple[tuple[float, ...], ...]:
    """Both in-frame readings of a **decision** reply's box, normalized first.

    The box counterpart of :func:`point_readings`, ordered the same way on
    purpose: one prompt asks for both a box and a point, so one convention has
    to answer for both, and a reader should not have to remember which field
    follows which rule.

    The 0-1000 reading leads because that is the space this model answers in
    whatever the prompt asks -- measured at 2-13px from truth against 264-363px
    for the raw-pixel reading, 6 of 6. Raw pixels stay as the retry that
    ``measure_grounding`` tries when the first reading will not segment.

    A reading whose centre falls outside the frame is dropped, and a coordinate
    above ``COORD_SCALE`` rules normalized out, leaving a single reading.
    """
    w, h = size
    out: list[tuple[float, ...]] = []

    def keep(candidate: Sequence[float] | None) -> None:
        if candidate is None:
            return
        vals = tuple(float(v) for v in candidate)
        cx = (vals[0] + vals[2]) / 2.0
        cy = (vals[1] + vals[3]) / 2.0
        if not (0 <= cx <= w and 0 <= cy <= h):
            return
        if vals not in out:
            out.append(vals)

    values = [float(c) for c in coords]
    keep(to_frame_pixels(values, size))
    if not any(abs(v) > COORD_SCALE for v in values):
        # Above COORD_SCALE the raw reading is the only possible one, and
        # to_frame_pixels has already passed it through unscaled.
        keep(values)
    return tuple(out)


def box_grounding(
    box: Any,
    size: tuple[int, int],
    *,
    label: str = "object",
    around: str = "the thing to pick up",
) -> tuple[Grounding | None, str]:
    """The decision reply's ``box_2d`` as a :class:`Grounding`, or (None, why).

    Two rejections, and neither is about whether the model boxed the *right*
    thing. A box must be four numbers, and it must not cover most of the frame
    -- asked to locate something absent, this build returns the whole image
    instead of declining (see :data:`MAX_BOX_FRAME_SHARE`).

    ``around`` names what the box was asked to enclose, for the refusal text.
    A PICK's box goes round the thing to pick up; a REPORT's entries go round
    each thing found, and the message has to say which was wanted.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None, (
            f"the reply needs box_2d as four pixel numbers [x1, y1, x2, y2] "
            f"around {around}, and gave {box!r}"
        )
    try:
        coords = [float(v) for v in box]
    except (TypeError, ValueError):
        return None, f"box_2d is not four numbers: {box!r}"

    readings = box_readings(coords, size)
    w, h = size
    if not readings:
        cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
        return None, (
            f"box_2d centres at ({cx:.0f}, {cy:.0f}) read as pixels, and "
            f"nowhere inside this {w}x{h} image under either reading"
        )

    g = _grounding(
        label, "box", readings[0], readings[1] if len(readings) > 1 else None, size
    )
    if g is not None:
        return g, ""

    x1, y1, x2, y2 = readings[0]
    area = abs(x2 - x1) * abs(y2 - y1)
    return None, (
        f"box_2d covers {100.0 * area / max(1, w * h):.0f}% of the frame, "
        "which is the model declining to answer rather than an object -- "
        "nothing on this desk is that big"
    )


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


def report_groundings(
    entries: Sequence[Any], size: tuple[int, int]
) -> tuple[tuple[Grounding, ...], tuple[str, ...]]:
    """A REPORT reply's ``objects`` list as groundings, and what would not read.

    One entry is ``{"box_2d": [x1, y1, x2, y2], "label": "..."}`` and goes
    through :func:`box_grounding`, so a box inside a report is validated exactly
    as a PICK's single box is: the same two coordinate readings, the same
    whole-frame rejection.

    An entry that will not read comes back as a note rather than as a refusal of
    the whole reply. A report answers "what is on this desk", and four objects
    plus a named gap is a better answer than none at all. The gap has to be
    *named*, though, which is why the notes are returned alongside: a report
    that quietly dropped an entry would under-count while looking complete, and
    a question phrased "find all ..." is answered wrongly by exactly that.
    """
    found: list[Grounding] = []
    notes: list[str] = []
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            notes.append(
                f"entry {i} is {entry!r}, not an object with box_2d and label"
            )
            continue
        label = str(entry.get("label") or "").strip() or "object"
        g, bad = box_grounding(
            entry.get("box_2d"), size,
            label=label, around="each thing you found",
        )
        if g is None:
            notes.append(f"entry {i} ({label}): {bad}")
            continue
        found.append(g)
    return tuple(found), tuple(notes)


def measure_grounding(
    obs: Observation,
    g: Grounding,
    *,
    label: str | None = None,
    object_height_mm: float | None = None,
) -> tuple[Any | None, str]:
    """Turn a :class:`Grounding` into a measured object, or say why not.

    Prefers GrabCut from the box and falls back to the desk-deviation point
    path, which is what ``locate.measure_with_box_fallback`` already arranges.
    On failure it retries under the other coordinate reading rather than
    reporting -- the retry still has to survive segmentation, the two-window
    stability check and the work-region gate, so a wrong reading cannot buy
    itself a target by being tried twice.

    ``object_height_mm`` is passed straight through. ``None`` -- what the
    instruction loop uses -- infers the height from the silhouette and
    unprojects the centroid to the table plane. A number overrides that, and
    ``0`` measures the object as if it were flat, with no parallax
    de-inflation at all.
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
                object_height_mm=object_height_mm,
            )
        except LocateError as exc:
            why = why or f"{prefix}{exc}"
            continue
        return obj, ""
    return None, why or f"nothing measurable at the {name} location"
