"""CLI for vision diagnostics and vision-driven pick/place.

Workflow:
  1. python -m mt4_vision markers      -- verify the ArUco markers are seen
  2. python calibrate_vision.py        -- jog-to-marker interactive calibration
  3. python -m mt4_vision scene        -- sanity-check cube detections
  4. python -m mt4_vision pick red     -- hardware test of one pick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, capture_frame
from mt4_vision.detect import detect_cubes, detect_markers, scan_marker_dicts
from mt4_vision.scene import filter_phantoms, is_phantom_detection
from mt4_vision.workspace import (
    analyze_workspace,
    cubes_of_color,
    cubes_with_robot_coords,
    marker_slots_from_calibration,
    pick_largest_cube,
)


def _save_annotated(frame, path: str) -> None:
    cv2.imwrite(path, frame)
    print(f"Annotated frame saved to {path}")


def cmd_markers(args: argparse.Namespace) -> int:
    frame = capture_frame(args.camera)
    if args.dict == "scan":
        hits = scan_marker_dicts(frame)
        if not hits:
            print("No ArUco markers found with any known dictionary")
            _save_annotated(frame, "markers_frame.jpg")
            return 1
        for name, count in sorted(hits.items(), key=lambda kv: -kv[1]):
            print(f"{name}: {count} markers")
        best = max(hits, key=hits.get)  # type: ignore[arg-type]
        print(f"\nUsing --dict {best} for detail:")
        args.dict = best
    markers = detect_markers(frame, args.dict)
    for m in markers:
        print(f"  id {m.marker_id}: pixel ({m.px:.1f}, {m.py:.1f})")
        cv2.circle(frame, (int(m.px), int(m.py)), 6, (0, 0, 255), 2)
        cv2.putText(
            frame, str(m.marker_id), (int(m.px) + 8, int(m.py) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
    _save_annotated(frame, "markers_frame.jpg")
    return 0 if markers else 1


def cmd_scene(args: argparse.Namespace) -> int:
    try:
        calib = load_calibration(Path(args.calib))
    except Exception as exc:  # noqa: BLE001 -- scene is usable pre-calibration
        print(f"(No calibration: {exc})")
        calib = None
    frame = capture_frame(args.camera)
    if calib is not None:
        state = analyze_workspace(calib, frame)
        cubes = state.cubes
        print(
            f"markers: {len(state.free_markers)} free, "
            f"{len(state.occupied)} occupied, "
            f"{len(state.free_slots)} open slots"
        )
        for marker, cube in state.occupied:
            print(
                f"  marker {marker.marker_id} ({marker.x:.1f}, {marker.y:.1f}): "
                f"{cube.color}"
            )
        for marker in state.free_markers:
            print(f"  Marker {marker.marker_id} ({marker.x:.1f}, {marker.y:.1f}): empty")
    else:
        cubes = detect_cubes(frame, calib)
    if not cubes:
        print("No cubes detected")
    slots = marker_slots_from_calibration(calib) if calib is not None else []
    for c in cubes:
        robot = f" robot ({c.x:.1f}, {c.y:.1f})" if c.x is not None else ""
        yaw = f" yaw {c.yaw_deg:.0f}°" if c.yaw_deg is not None else ""
        phantom = (
            " [phantom -- not a pick target]"
            if calib is not None and is_phantom_detection(c, calib)
            else ""
        )
        print(f"  {c.color}: pixel ({c.px:.0f}, {c.py:.0f}) area {c.area:.0f}px^2{robot}{yaw}{phantom}")
        cv2.circle(frame, (int(c.px), int(c.py)), 8, (255, 255, 255), 2)
        cv2.putText(
            frame, c.color, (int(c.px) + 10, int(c.py)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
    _save_annotated(frame, "scene_frame.jpg")
    return 0


def cmd_entities(args: argparse.Namespace) -> int:
    """Print the entity table the MCP layer and any model would see."""
    from mt4_vision.entities import build_snapshot
    from mt4_vision.scene import capture_scene

    calib = load_calibration(Path(args.calib))
    frame = capture_frame(args.camera)
    scene = capture_scene(calib, frame)
    snapshot = build_snapshot(scene, token="cli")
    print(snapshot.summary)
    for line in snapshot.table():
        print(line)
    from mt4_vision.preview import annotate_for_pointing

    _save_annotated(annotate_for_pointing(frame, snapshot.entities), "entities_frame.jpg")
    return 0


def cmd_locate(args: argparse.Namespace) -> int:
    """Measure the object at a pixel hint -- the grounding path, without MCP.

    Lets the segmentation window and match threshold be tuned against the real
    object on the real desk with no model in the loop.
    """
    from mt4_vision.entities import object_entity
    from mt4_vision.locate import LocateError, grasp_feasibility, measure
    from mt4_vision.preview import annotate_for_pointing
    from mt4_vision.scene import capture_scene

    calib = load_calibration(Path(args.calib))
    frame = capture_frame(args.camera)
    scene = capture_scene(calib, frame)
    try:
        obj = measure(
            frame, args.pixel[0], args.pixel[1], calib, args.label,
            win=args.window, marker_xy=[(m.x, m.y) for m in scene.markers],
        )
    except LocateError as exc:
        print(f"locate failed: {exc}")
        return 1
    print(f"{obj.label}: {obj.as_dict()}")
    ok, reason = grasp_feasibility(obj, calib)
    print(f"graspable: {ok}" + ("" if ok else f" -- {reason}"))

    entity = object_entity(obj, 1, scene=scene)
    _save_annotated(
        annotate_for_pointing(frame, [entity]), "locate_frame.jpg"
    )
    if not args.pick:
        return 0
    if not ok:
        print("refusing to pick: not graspable")
        return 1

    from mt4_jog.client import Mt4ClientError
    from mt4_vision.motion import pick_at

    client = _pick_place_client(args)
    try:
        print(pick_at(client, calib, entity.as_grasp(calib)))
    except Mt4ClientError as exc:
        print(f"pick failed: {exc}")
        return 1
    finally:
        client.close()
    return 0


def cmd_sam(args: argparse.Namespace) -> int:
    """Segment at a pixel or a box via the SAM 2.1 service (MT4_SAM_URL).

    The mask is what a box or a click is not: an actual silhouette, with a
    centre of area rather than a box centre. Saved tinted onto the frame so
    the edge can be judged against the real object.
    """
    import numpy as np

    from mt4_vision.sam import SamError, best_per_object, health, segment

    try:
        print(f"service: {health()}")
    except SamError as exc:
        print(exc)
        return 1

    frame = capture_frame(args.camera)
    try:
        masks = segment(
            frame,
            points=[args.pixel] if args.pixel else None,
            boxes=[args.box] if args.box else None,
        )
    except SamError as exc:
        print(exc)
        return 1

    overlay = frame.copy()
    for m in masks if args.candidates else best_per_object(masks):
        if m.bbox is None:
            print(f"[{m.object_index}] score={m.score:.3f} empty mask")
            continue
        print(
            f"[{m.object_index}] score={m.score:.3f} area={m.area} "
            f"bbox={m.bbox} centroid=({m.cx:.0f},{m.cy:.0f})"
        )
        overlay[m.mask] = (0.55 * overlay[m.mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(
            m.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.circle(overlay, (int(m.cx), int(m.cy)), 5, (0, 0, 255), -1)
    _save_annotated(overlay, "sam_frame.jpg")
    return 0 if masks else 1


def cmd_grounding(args: argparse.Namespace) -> int:
    """Open-vocab detect via the Grounding DINO service (MT4_GROUNDING_URL)."""
    from mt4_vision.grounding import GroundingError, detect, health

    try:
        info = health()
        print(f"service: {info}")
    except GroundingError as exc:
        print(exc)
        return 1

    frame = capture_frame(args.camera)
    try:
        dets = detect(
            frame, args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    except GroundingError as exc:
        print(exc)
        return 1

    if not dets:
        print(f"no detections for prompt={args.prompt!r}")
        _save_annotated(frame, "grounding_frame.jpg")
        return 1

    # Draw on a COPY. cv2 draws in place, and the hint dot lands exactly on the
    # pixel measure() segments around -- a saturated disc on top of the object
    # fuses into its own mask and biases the centroid and width, which the
    # two-window stability check cannot catch (the disc is in both windows).
    overlay = frame.copy()
    for i, d in enumerate(dets):
        print(f"[{i}] {d.label} score={d.score:.3f} box=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f}) cx={d.cx:.0f} cy={d.cy:.0f}")
        cv2.rectangle(
            overlay, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2,
        )
        cv2.circle(overlay, (int(d.cx), int(d.cy)), 6, (0, 0, 255), -1)
        cv2.putText(
            overlay, f"{d.label} {d.score:.2f}", (int(d.x1), max(20, int(d.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    _save_annotated(overlay, "grounding_frame.jpg")

    if not args.locate and not args.pick:
        return 0

    from mt4_vision.entities import object_entity
    from mt4_vision.locate import LocateError, grasp_feasibility, measure_with_box_fallback
    from mt4_vision.preview import annotate_for_pointing
    from mt4_vision.scene import capture_scene

    best = dets[0]
    label = args.label or best.label
    calib = load_calibration(Path(args.calib))
    scene = capture_scene(calib, frame)
    try:
        obj = measure_with_box_fallback(
            frame, best.cx, best.cy, calib, label,
            box=(best.x1, best.y1, best.x2, best.y2),
            win=args.window, marker_xy=[(m.x, m.y) for m in scene.markers],
            confidence=best.score,
        )
    except LocateError as exc:
        print(f"locate failed: {exc}")
        return 1
    print(f"{obj.label}: {obj.as_dict()}")
    ok, reason = grasp_feasibility(obj, calib)
    print(f"graspable: {ok}" + ("" if ok else f" -- {reason}"))
    entity = object_entity(obj, 1, scene=scene)
    _save_annotated(annotate_for_pointing(frame, [entity]), "grounding_locate.jpg")
    if not args.pick:
        return 0
    if not ok:
        print("refusing to pick: not graspable")
        return 1

    from mt4_jog.client import Mt4ClientError
    from mt4_vision.motion import pick_at

    client = _pick_place_client(args)
    try:
        print(pick_at(client, calib, entity.as_grasp(calib)))
    except Mt4ClientError as exc:
        print(f"pick failed: {exc}")
        return 1
    finally:
        client.close()
    return 0


def cmd_transfer(args: argparse.Namespace) -> int:
    """Exercise the queued transfer primitive directly (moves the arm)."""
    from mt4_jog.client import Mt4ClientError
    from mt4_vision.motion import Grasp, transfer

    calib = load_calibration(Path(args.calib))
    client = _pick_place_client(args)
    try:
        result = transfer(
            client,
            calib,
            Grasp(args.src[0], args.src[1], yaw_deg=args.src_yaw, center=args.center),
            Grasp(args.dst[0], args.dst[1], yaw_deg=args.dst_yaw),
        )
        print(result)
    except Mt4ClientError as exc:
        print(f"transfer failed: {exc}")
        return 1
    finally:
        client.close()
    return 0


def _pick_place_client(args: argparse.Namespace):
    from mt4_jog.client import Mt4Client

    return Mt4Client() if not args.port else Mt4Client(port=args.port)


def cmd_pick(args: argparse.Namespace) -> int:
    from mt4_jog.client import Mt4ClientError
    from mt4_vision.pickplace import pick_cube

    calib = load_calibration(Path(args.calib))
    frame = capture_frame(args.camera)
    candidates = filter_phantoms(
        cubes_with_robot_coords(detect_cubes(frame, calib)), calib
    )
    target = pick_largest_cube(cubes_of_color(candidates, args.color))
    if target is None:
        print(f"No {args.color} cube in view")
        return 1
    yaw = (
        f" yaw={target.yaw_deg:.0f}°"
        if target.yaw_deg is not None
        else ""
    )
    print(f"Picking {args.color} at robot ({target.x:.1f}, {target.y:.1f}){yaw}")
    client = _pick_place_client(args)
    try:
        result = pick_cube(client, calib, target)
        if "j4" in result:
            print(f"  face-align j4={result['j4']:.1f}°")
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()
    print("Done")
    return 0


def cmd_place(args: argparse.Namespace) -> int:
    from mt4_jog.client import Mt4ClientError
    from mt4_vision.pickplace import place

    calib = load_calibration(Path(args.calib))
    client = _pick_place_client(args)
    try:
        place(client, calib, args.x, args.y)
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()
    print("Done")
    return 0


def cmd_place_here(args: argparse.Namespace) -> int:
    from mt4_jog.client import Mt4ClientError
    from mt4_vision.pickplace import place_here

    calib = load_calibration(Path(args.calib))
    client = _pick_place_client(args)
    try:
        tcp = client.get_tcp()
        print(f"Placing at current position ({tcp.x:.1f}, {tcp.y:.1f})")
        place_here(client, calib)
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()
    print("Done")
    return 0


def cmd_shuffle(args: argparse.Namespace) -> int:
    import time

    from mt4_jog.client import Mt4ClientError
    from mt4_vision.shuffle import run_shuffle_loop

    calib = load_calibration(Path(args.calib))
    client = _pick_place_client(args)
    try:
        time.sleep(1.0)
        print("Shuffle loop started (Ctrl+C to stop, H in this terminal to re-home)")
        run_shuffle_loop(
            client,
            calib,
            camera=args.camera,
            pause_s=args.pause,
            retry_s=args.retry,
        )
    except KeyboardInterrupt:
        print("\nStopped")
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


def cmd_goto_marker(args: argparse.Namespace) -> int:
    from mt4_jog.client import Mt4ClientError
    from mt4_vision.pickplace import goto_marker

    calib = load_calibration(Path(args.calib))
    frame = capture_frame(args.camera)
    markers = detect_markers(frame, args.dict)
    match = next((m for m in markers if m.marker_id == args.marker_id), None)
    if match is None:
        print(f"Marker {args.marker_id} not in view (visible: "
              f"{sorted(m.marker_id for m in markers)})")
        return 1
    x, y = calib.pixel_to_robot(match.px, match.py)
    print(f"Marker {args.marker_id} at pixel ({match.px:.0f}, {match.py:.0f}) "
          f"-> robot ({x:.1f}, {y:.1f}){' -- touching table' if args.touch else ' -- hovering at safe_z'}")
    client = _pick_place_client(args)
    try:
        goto_marker(client, calib, x, y, touch=args.touch)
    except Mt4ClientError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        client.close()
    print("Done")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mt4_vision",
        epilog="Put --camera / --calib before the subcommand, e.g. "
               "python -m mt4_vision --camera 1 scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("markers", help="detect ArUco markers, save annotated frame")
    p.add_argument("--dict", default="scan", help="ArUco dict name, or 'scan' to try all")
    p.set_defaults(func=cmd_markers)

    p = sub.add_parser("scene", help="detect cubes, print robot coords")
    p.set_defaults(func=cmd_scene)

    p = sub.add_parser("pick", help="pick a cube by color (moves the arm)")
    p.add_argument("color")
    p.add_argument("--port", default="")
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("place", help="place held cube at robot X Y (moves the arm)")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("--port", default="")
    p.set_defaults(func=cmd_place)

    p = sub.add_parser(
        "place-here",
        help="place held cube at the current TCP xy (moves the arm)",
    )
    p.add_argument("--port", default="")
    p.set_defaults(func=cmd_place_here)

    p = sub.add_parser(
        "goto-marker",
        help="move the arm to a detected marker -- calibration accuracy check",
    )
    p.add_argument("marker_id", type=int)
    p.add_argument("--dict", default="4x4_50")
    p.add_argument(
        "--touch", action="store_true",
        help="descend to table_z instead of hovering at safe_z",
    )
    p.add_argument("--port", default="")
    p.set_defaults(func=cmd_goto_marker)

    p = sub.add_parser(
        "entities",
        help="print the addressable entity table (ids, pickable, why not)",
    )
    p.set_defaults(func=cmd_entities)

    p = sub.add_parser(
        "locate",
        help="measure a non-cube object at a pixel hint (the grounding path)",
    )
    p.add_argument("--pixel", type=float, nargs=2, required=True, metavar=("PX", "PY"))
    p.add_argument("--label", default="object")
    p.add_argument("--window", type=int, default=280, help="segmentation crop side (px)")
    p.add_argument("--pick", action="store_true", help="then pick it (moves the arm)")
    p.add_argument("--port", default="")
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser(
        "grounding",
        help="open-vocab detect via the Grounding DINO service (MT4_GROUNDING_URL)",
    )
    p.add_argument("--prompt", required=True, help='e.g. "pen" or "pen. red cube."')
    p.add_argument("--label", default="", help="override label when --locate/--pick")
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--locate", action="store_true", help="measure top hit via locate.measure")
    p.add_argument("--window", type=int, default=280)
    p.add_argument("--pick", action="store_true", help="then pick top hit (moves the arm)")
    p.add_argument("--port", default="")
    p.set_defaults(func=cmd_grounding)

    p = sub.add_parser(
        "sam",
        help="segment at a pixel or box via the SAM 2.1 service (MT4_SAM_URL)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pixel", type=float, nargs=2, metavar=("PX", "PY"))
    g.add_argument(
        "--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
        help="e.g. a Grounding DINO box",
    )
    p.add_argument(
        "--candidates", action="store_true",
        help="draw all three candidate masks, not just the best-scoring one",
    )
    p.set_defaults(func=cmd_sam)

    p = sub.add_parser(
        "transfer",
        help="queued pick+place between two robot-frame XYs (moves the arm)",
    )
    p.add_argument("--from", dest="src", type=float, nargs=2, required=True, metavar=("X", "Y"))
    p.add_argument("--to", dest="dst", type=float, nargs=2, required=True, metavar=("X", "Y"))
    p.add_argument("--from-yaw", dest="src_yaw", type=float, default=None)
    p.add_argument("--to-yaw", dest="dst_yaw", type=float, default=None)
    p.add_argument("--center", action="store_true", help="+/-90 deg re-grip after gripping")
    p.add_argument("--port", default="")
    p.set_defaults(func=cmd_transfer)

    p = sub.add_parser(
        "shuffle",
        help="home then shuffle cubes between markers and open table (Ctrl+C to stop)",
    )
    p.add_argument("--port", default="")
    p.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="settle time before post-release capture (overlaps with lift)",
    )
    p.add_argument("--retry", type=float, default=5.0)
    p.set_defaults(func=cmd_shuffle)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
