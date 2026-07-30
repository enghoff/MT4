#!/usr/bin/env python3
"""Interactive open-vocab mover: prompt for an object, place it on a free marker.

Repeatedly asks for an object description, then::

    Grounding DINO(prompt)  -> candidate boxes
    locate.measure each     -> robot XY + long-axis yaw (skip failures)
    filter                  -> reachable / pickable / not already on a marker
    motion.transfer         -> pick with 180° long-axis wrist, place on free marker

Re-parks after every transfer so the next prompt can capture immediately.
Objects that cannot be segmented, are out of reach, fail grasp feasibility, or
have no free marker are skipped -- never substituted onto a different target.
Loops until Ctrl+C / EOF.

Prereqs:
  * ``.\\scripts\\start_grounding_tunnel.ps1``
  * Arm free on the serial port; vision calibration present

Example::

    python move_object_to_marker.py --camera 1
    python move_object_to_marker.py --dry-run --no-park
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
from mt4_vision.preview import LivePreview, PreviewStopped
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
        skips.append(f"didn't find {prompt}")
        return [], skips

    marker_xy = [(m.x, m.y) for m in scene.markers]
    measured: list[Candidate] = []
    for det in dets:
        label = det.label.strip() or prompt.strip().rstrip(".")
        try:
            obj = measure(
                frame, det.cx, det.cy, calib, label, marker_xy=marker_xy,
            )
        except LocateError:
            skips.append(f"couldn't measure {label}")
            continue

        if _denied(obj, deny):
            skips.append(f"skipping {label} (earlier fail)")
            continue

        home = _on_marker(obj, scene.markers)
        if home is not None:
            skips.append(f"{label} already on marker {home.marker_id}")
            continue

        ok, reason = grasp_feasibility(obj, calib)
        entity = object_entity(obj, len(measured) + 1, scene=scene)
        if not ok:
            skips.append(f"can't pick {label}: {reason}")
            continue
        if not entity.pickable:
            skips.append(f"can't pick {label}: {entity.reason}")
            continue

        if any(dist_mm(obj.x, obj.y, c.obj.x, c.obj.y) < DEDUP_MM for c in measured):
            continue

        measured.append(Candidate(det=det, obj=obj))

    return measured, skips


def _annotate_plan(
    frame,
    pairs: list[tuple[Candidate, MarkerSlot]],
    skips: list[str],
):
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
    return out


def _park(client: Mt4Client, calib: Calibration) -> None:
    retreat_for_camera(client, calib)
    time.sleep(CAMERA_SETTLE_S)


def _prompt_object() -> str | None:
    """Ask for the next object description. None means exit (EOF)."""
    try:
        raw = input("\nobject: ")
    except EOFError:
        print()
        return None
    return raw.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Prompt for an open-vocab object, detect it with Grounding DINO, "
            "and move it onto a free marker; repeat until terminated"
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
        "--dry-run",
        action="store_true",
        help="detect + plan only; do not move the arm",
    )
    p.add_argument(
        "--no-park",
        action="store_true",
        help="skip retreat to camera park before captures",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="skip the annotated OpenCV preview window",
    )
    args = p.parse_args(argv)

    try:
        health()
    except GroundingError as exc:
        print(f"grounding unavailable: {exc}", file=sys.stderr)
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
                client = None
            else:
                print(f"arm error: {exc}", file=sys.stderr)
                return 1

    moved = 0
    deny: list[tuple[float, float]] = []
    preview = None if args.no_preview else LivePreview("move object")
    print("ready")
    try:
        while True:
            prompt = _prompt_object()
            if prompt is None:
                break
            if not prompt:
                continue

            frame = capture_frame(args.camera)
            scene = capture_scene(calib, frame)
            free = scene.placeable_markers()
            if allow_markers is not None:
                free = [m for m in free if m.marker_id in allow_markers]
            free = sorted(free, key=lambda m: m.marker_id)
            if not free:
                print("no free markers")
                continue

            try:
                candidates, skips = _gather_candidates(
                    frame, calib, scene, prompt,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    deny=deny,
                )
            except GroundingError as exc:
                print(f"couldn't detect: {exc}")
                continue

            if not candidates:
                print(skips[0] if skips else f"didn't find {prompt}")
                continue

            cand = candidates[0]
            marker = free[0]
            annotated = _annotate_plan(frame, [(cand, marker)], skips)
            if preview is not None:
                try:
                    preview.show(annotated)
                except PreviewStopped:
                    break

            if args.dry_run:
                print(f"would move {cand.obj.label} -> marker {marker.marker_id}")
                continue

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

            try:
                transfer(client, calib, src, dst)
                print(f"moved {cand.obj.label} -> marker {marker.marker_id}")
                moved += 1
            except Mt4ClientError as exc:
                print(f"couldn't move: {exc}")
                deny.append((cand.obj.x, cand.obj.y))
            if not args.no_park:
                try:
                    _park(client, calib)
                except Mt4ClientError as exc:
                    print(f"couldn't park: {exc}")
                    return 1

        return 0
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        if preview is not None:
            preview.close()
        if client is not None:
            try:
                if not args.no_park:
                    retreat_for_camera(client, calib)
            except Mt4ClientError:
                pass
            client.close()


if __name__ == "__main__":
    sys.exit(main())
