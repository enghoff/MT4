#!/usr/bin/env python3
"""Open-vocab sweep: move every handleable desk object onto a free marker.

For each clear camera frame::

    Grounding DINO(--object prompt)  -> candidate boxes
    locate.measure each              -> robot XY + long-axis yaw (skip failures)
    filter                           -> reachable / pickable / not already on a marker
    assign 1:1 to placeable free markers
    motion.transfer                  -> pick with 180° long-axis wrist, place on marker

Objects that cannot be segmented (e.g. sitting on marker paper), are out of
reach, fail grasp feasibility, or have no free marker left are **skipped** --
never substituted onto a different target than planned for that cycle.

Re-parks and re-scans after every successful transfer so occupancy stays true.
Stops when a scan finds no more handleable objects or no free markers.

Prereqs:
  * ``.\\scripts\\start_grounding_tunnel.ps1``
  * Arm free on the serial port; vision calibration present

Example::

    python move_object_to_marker.py --object "pen. eraser. drill bits." --camera 1
    python move_object_to_marker.py --object "red eraser" --dry-run --no-park
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import DEFAULT_CALIB_PATH, Calibration, load_calibration
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, capture_frame
from mt4_vision.entities import object_entity
from mt4_vision.grounding import Detection, GroundingError, detect, health
from mt4_vision.locate import LocateError, LocatedObject, grasp_feasibility, measure
from mt4_vision.motion import Grasp, YAW_PERIOD_LONG_AXIS, transfer
from mt4_vision.pickplace import ensure_homed, retreat_for_camera
from mt4_vision.scene import Scene, capture_scene
from mt4_vision.workspace import (
    MARKER_OCCUPY_RADIUS_MM,
    MarkerSlot,
    dist_mm,
)

CAMERA_SETTLE_S = 0.5
# Two DINO boxes that measure within this distance are the same physical thing.
DEDUP_MM = 25.0
# After a failed transfer, ignore future candidates this close (mm).
DENY_MM = 30.0


@dataclass(frozen=True)
class Candidate:
    det: Detection
    obj: LocatedObject


def _on_marker(obj: LocatedObject, markers: list[MarkerSlot]) -> MarkerSlot | None:
    for m in markers:
        if dist_mm(obj.x, obj.y, m.x, m.y) < MARKER_OCCUPY_RADIUS_MM:
            return m
    return None


def _denied(obj: LocatedObject, deny: list[tuple[float, float]]) -> bool:
    return any(dist_mm(obj.x, obj.y, dx, dy) < DENY_MM for dx, dy in deny)


def _gather_candidates(
    frame,
    calib: Calibration,
    scene: Scene,
    prompt: str,
    *,
    box_threshold: float,
    text_threshold: float,
    deny: list[tuple[float, float]],
) -> tuple[list[Candidate], list[str]]:
    """Return (handleable candidates highest-score first, skip notes)."""
    skips: list[str] = []
    dets = detect(
        frame, prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if not dets:
        skips.append(f"dino: no detections for {prompt!r}")
        return [], skips

    marker_xy = [(m.x, m.y) for m in scene.markers]
    measured: list[Candidate] = []
    for i, det in enumerate(dets):
        label = det.label.strip() or prompt.strip().rstrip(".")
        tag = (
            f"[{i}] {label!r} score={det.score:.3f} "
            f"hint=({det.cx:.0f},{det.cy:.0f})"
        )
        try:
            obj = measure(
                frame, det.cx, det.cy, calib, label, marker_xy=marker_xy,
            )
        except LocateError as exc:
            skips.append(f"{tag}: skip measure -- {exc}")
            continue

        if _denied(obj, deny):
            skips.append(
                f"{tag}: skip denied after earlier transfer fail "
                f"near ({obj.x:.0f},{obj.y:.0f})"
            )
            continue

        home = _on_marker(obj, scene.markers)
        if home is not None:
            skips.append(
                f"{tag}: skip already on marker {home.marker_id} "
                f"at ({obj.x:.0f},{obj.y:.0f})"
            )
            continue

        ok, reason = grasp_feasibility(obj, calib)
        entity = object_entity(obj, len(measured) + 1, scene=scene)
        if not ok:
            skips.append(f"{tag}: skip grasp -- {reason}")
            continue
        if not entity.pickable:
            skips.append(f"{tag}: skip pickable -- {entity.reason}")
            continue

        if any(dist_mm(obj.x, obj.y, c.obj.x, c.obj.y) < DEDUP_MM for c in measured):
            skips.append(
                f"{tag}: skip duplicate of earlier hit "
                f"near ({obj.x:.0f},{obj.y:.0f})"
            )
            continue

        measured.append(Candidate(det=det, obj=obj))

    return measured, skips


def _annotate_plan(
    frame,
    pairs: list[tuple[Candidate, MarkerSlot]],
    skips: list[str],
    path: Path,
) -> None:
    out = frame.copy()
    for cand, marker in pairs:
        d, obj = cand.det, cand.obj
        cv2.rectangle(
            out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 255), 2,
        )
        cv2.circle(out, (int(obj.px), int(obj.py)), 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"{obj.label}->{marker.marker_id}",
            (int(d.x1), max(20, int(d.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
    cv2.putText(
        out,
        f"plan {len(pairs)} move(s), {len(skips)} skip(s)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(path), out)
    print(f"annotated frame saved to {path}")


def _park(client: Mt4Client, calib: Calibration) -> None:
    retreat_for_camera(client, calib)
    time.sleep(CAMERA_SETTLE_S)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Detect open-vocab objects with Grounding DINO and move each "
            "handleable one onto a free marker; skip the rest"
        ),
    )
    p.add_argument(
        "--object",
        required=True,
        help=(
            'DINO vocabulary, period-separated phrases '
            'e.g. "pen. eraser. drill bits."'
        ),
    )
    p.add_argument(
        "--marker",
        type=int,
        action="append",
        default=None,
        help=(
            "optional: only use these marker ids as destinations "
            "(repeatable). Default: every placeable free marker"
        ),
    )
    p.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    p.add_argument("--port", default="")
    p.add_argument("--box-threshold", type=float, default=0.25)
    p.add_argument("--text-threshold", type=float, default=0.20)
    p.add_argument(
        "--align-place",
        action="store_true",
        help="re-apply long-axis wrist at the marker (default: preserve pick yaw)",
    )
    p.add_argument(
        "--max-moves",
        type=int,
        default=20,
        help="safety cap on transfers this run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="one scan + plan only; do not move the arm",
    )
    p.add_argument(
        "--no-park",
        action="store_true",
        help="skip retreat to camera park before captures",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=Path("move_object_frame.jpg"),
        help="annotated plan frame (empty string to skip)",
    )
    args = p.parse_args(argv)

    try:
        info = health()
        print(f"grounding: {info}")
    except GroundingError as exc:
        print(exc, file=sys.stderr)
        return 1

    calib = load_calibration(args.calib)
    allow_markers = set(args.marker) if args.marker else None

    client: Mt4Client | None = None
    if not args.dry_run or not args.no_park:
        try:
            client = Mt4Client() if not args.port else Mt4Client(port=args.port)
            ensure_homed(client)
            if not args.no_park:
                _park(client, calib)
        except Mt4ClientError as exc:
            if args.dry_run:
                print(f"arm unavailable ({exc}); capturing anyway")
                client = None
            else:
                print(f"arm error: {exc}", file=sys.stderr)
                return 1

    moved = 0
    deny: list[tuple[float, float]] = []
    try:
        while moved < args.max_moves:
            if client is not None and not args.no_park:
                try:
                    _park(client, calib)
                except Mt4ClientError as exc:
                    print(f"park failed: {exc}", file=sys.stderr)
                    return 1

            frame = capture_frame(args.camera)
            scene = capture_scene(calib, frame)
            free = scene.placeable_markers()
            if allow_markers is not None:
                free = [m for m in free if m.marker_id in allow_markers]
            free = sorted(free, key=lambda m: m.marker_id)
            print(
                f"scan: free placeable markers="
                f"{[m.marker_id for m in free]} "
                f"occupied={[m.marker_id for m, _ in scene.occupied]}"
            )
            if not free:
                print("no free markers left -- done")
                break

            try:
                candidates, skips = _gather_candidates(
                    frame, calib, scene, args.object,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    deny=deny,
                )
            except GroundingError as exc:
                print(exc, file=sys.stderr)
                return 1

            for note in skips:
                print(f"  {note}")

            if not candidates:
                print("no handleable objects this scan -- done")
                break

            for c, m in zip(candidates, free):
                print(
                    f"  plan: {c.obj.label!r} ({c.obj.x:.0f},{c.obj.y:.0f}) "
                    f"yaw={c.obj.axis_yaw_deg:.1f} "
                    f"-> marker {m.marker_id} ({m.x:.0f},{m.y:.0f})"
                )

            if args.save and str(args.save):
                _annotate_plan(
                    frame, list(zip(candidates, free)), skips, args.save,
                )

            if args.dry_run:
                print("dry-run: skipping transfers")
                break

            # One transfer per scan so the next free-marker set is truthful.
            cand = candidates[0]
            marker = free[0]
            assert client is not None
            entity = object_entity(cand.obj, 1, scene=scene)
            src = entity.as_grasp(calib)
            assert src.yaw_period_deg == YAW_PERIOD_LONG_AXIS
            if args.align_place:
                dst = Grasp(
                    marker.x, marker.y,
                    yaw_deg=cand.obj.axis_yaw_deg,
                    yaw_period_deg=YAW_PERIOD_LONG_AXIS,
                    grip_close_s=src.grip_close_s,
                )
            else:
                dst = Grasp(marker.x, marker.y)

            print(
                f"transfer {cand.obj.label!r} -> marker {marker.marker_id} ..."
            )
            try:
                result = transfer(client, calib, src, dst)
                print(result)
                moved += 1
            except Mt4ClientError as exc:
                print(
                    f"transfer failed ({exc}); denying this pose and continuing",
                    file=sys.stderr,
                )
                deny.append((cand.obj.x, cand.obj.y))

        print(f"finished: {moved} object(s) moved")
        return 0
    finally:
        if client is not None:
            try:
                if not args.no_park:
                    retreat_for_camera(client, calib)
            except Mt4ClientError:
                pass
            client.close()


if __name__ == "__main__":
    sys.exit(main())
