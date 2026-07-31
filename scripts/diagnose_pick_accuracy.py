#!/usr/bin/env python3
"""Isolate pick-position error sources without relying on a successful grasp.

Reports (mm) for each chain segment:

  1. table-plane calibration   -- marker reprojection residuals
  2. camera align / drift      -- live marker pixels vs calibration snapshot
  3. cube-top parallax gap     -- table-plane map vs cube_top map at cube pixels
  4. visual detection noise    -- multi-frame cube centroid RMS
  5. MT4 control consistency   -- commanded vs reported TCP after move_to

Usage:
  python scripts/diagnose_pick_accuracy.py --port COM9
  python scripts/diagnose_pick_accuracy.py --port COM9 --no-arm   # camera-only
  python scripts/diagnose_pick_accuracy.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.ports import resolve_port
from mt4_vision.calib import DEFAULT_CALIB_PATH, load_calibration, reprojection_errors
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, capture_frame
from mt4_vision.detect import detect_cubes, detect_markers


CUBETOP_BACKUP = ROOT / "backups" / "vision_calibration_cubetop_20260714.json"
CONTROL_TARGETS = [
    (180.0, 0.0),
    (150.0, -120.0),
    (150.0, 120.0),
    (220.0, 40.0),
]
NOISE_FRAMES = 8
NOISE_PAUSE_S = 0.15


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean_mm": round(statistics.mean(values), 2),
        "max_mm": round(max(values), 2),
        "rms_mm": round(math.sqrt(sum(v * v for v in values) / len(values)), 2),
    }


def _pixel_jacobian_mm_per_px(calib, px: float, py: float, eps: float = 1.0) -> float:
    """Local mm/pixel scale from the table-plane homography."""
    x0, y0 = calib.pixel_to_robot(px, py)
    x1, y1 = calib.pixel_to_robot(px + eps, py)
    x2, y2 = calib.pixel_to_robot(px, py + eps)
    return 0.5 * (math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x0, y2 - y0)) / eps


def table_plane_residuals(calib) -> dict:
    obs = calib.raw_marker_observations or {}
    if not obs:
        return {"error": "no raw_marker_observations in calibration"}
    px = [(v["pixel"][0], v["pixel"][1]) for v in obs.values()]
    rb = [(v["robot"][0], v["robot"][1]) for v in obs.values()]
    errs = reprojection_errors(calib.homography, px, rb)
    per = {mid: round(e, 2) for mid, e in zip(obs.keys(), errs)}
    return {"per_marker_mm": per, **_stats(errs)}


def camera_align_drift(calib, frame) -> dict:
    obs = calib.raw_marker_observations or {}
    live = {m.marker_id: m for m in detect_markers(frame)}
    rows = []
    mm_errs = []
    for mid_s, v in obs.items():
        mid = int(mid_s)
        if mid not in live:
            rows.append({"marker_id": mid, "status": "not_seen"})
            continue
        m = live[mid]
        dpx = math.hypot(m.px - v["pixel"][0], m.py - v["pixel"][1])
        scale = _pixel_jacobian_mm_per_px(calib, m.px, m.py)
        d_mm = dpx * scale
        # Also: live pixel through current calib vs stored robot touch
        rx, ry = calib.pixel_to_robot(m.px, m.py)
        map_err = math.hypot(rx - v["robot"][0], ry - v["robot"][1])
        mm_errs.append(d_mm)
        rows.append(
            {
                "marker_id": mid,
                "status": "ok",
                "pixel_drift_px": round(dpx, 2),
                "pixel_drift_mm": round(d_mm, 2),
                "live_map_vs_touch_mm": round(map_err, 2),
                "live_robot": [round(rx, 1), round(ry, 1)],
                "calib_robot": list(v["robot"]),
            }
        )
    return {
        "markers_calib": len(obs),
        "markers_live": len(live),
        "per_marker": rows,
        "pixel_drift": _stats(mm_errs),
        "live_map_vs_touch": _stats(
            [r["live_map_vs_touch_mm"] for r in rows if r.get("status") == "ok"]
        ),
    }


def cube_top_gap(calib, frame, backup_path: Path | None) -> dict:
    cubes = detect_cubes(frame, calibration=calib)
    backup = None
    if backup_path and backup_path.exists():
        backup = load_calibration(backup_path)

    has_live_top = calib.cube_top_homography is not None
    has_backup_top = backup is not None and backup.cube_top_homography is not None

    rows = []
    live_deltas = []
    missing_deltas = []
    for c in cubes:
        row: dict = {
            "color": c.color,
            "pixel": [round(c.px, 1), round(c.py, 1)],
            "area_px": round(c.area),
            "table_xy": None,
            "cube_top_xy": None,
            "delta_mm": None,
        }
        xt, yt = calib.pixel_to_robot(c.px, c.py, on_cube_top=False)
        row["table_xy"] = [round(xt, 1), round(yt, 1)]
        if has_live_top:
            xc, yc = calib.pixel_to_robot(c.px, c.py, on_cube_top=True)
            d = math.hypot(xc - xt, yc - yt)
            row["cube_top_xy"] = [round(xc, 1), round(yc, 1)]
            row["delta_mm"] = round(d, 2)
            live_deltas.append(d)
        elif has_backup_top:
            # Live file missing cube_top: estimate pick error if we keep
            # using the table plane at this cube's pixels, using the last
            # good cube_top fit as a proxy for truth.
            xc, yc = backup.pixel_to_robot(c.px, c.py, on_cube_top=True)
            d = math.hypot(xc - xt, yc - yt)
            row["cube_top_xy_proxy"] = [round(xc, 1), round(yc, 1)]
            row["delta_mm"] = round(d, 2)
            row["proxy"] = "backup_cubetop"
            missing_deltas.append(d)
        rows.append(row)

    # Workspace sample even without cubes: marker pixels
    marker_proxy = []
    if has_backup_top and not has_live_top:
        for mid, v in (calib.raw_marker_observations or {}).items():
            px, py = v["pixel"]
            xt, yt = calib.pixel_to_robot(px, py, on_cube_top=False)
            xc, yc = backup.pixel_to_robot(px, py, on_cube_top=True)
            d = math.hypot(xc - xt, yc - yt)
            marker_proxy.append({"marker_id": int(mid), "delta_mm": round(d, 2)})

    return {
        "live_cube_top_homography": "set" if has_live_top else "NULL",
        "backup_cube_top_homography": "set" if has_backup_top else "missing",
        "cubes": rows,
        "live_table_vs_cubetop": _stats(live_deltas),
        "estimated_pick_error_without_cubetop": _stats(missing_deltas),
        "marker_pixel_proxy_deltas": marker_proxy,
        "marker_pixel_proxy": _stats([r["delta_mm"] for r in marker_proxy]),
        "note": (
            "Live calibration has no cube_top_homography; cubes are mapped "
            "through the table-plane homography. Deltas use the 2026-07-14 "
            "backup cube_top fit as a truth proxy."
            if not has_live_top and has_backup_top
            else None
        ),
    }


def detection_noise(calib, camera: int = DEFAULT_CAMERA_INDEX, n_frames: int = NOISE_FRAMES) -> dict:
    """Multi-frame centroid RMS per color (nearest-neighbor track)."""
    # Per-frame detections as list of (color, px, py)
    frames: list[list[tuple[str, float, float]]] = []
    for i in range(n_frames):
        frame = capture_frame(camera)
        cubes = detect_cubes(frame, calibration=calib)
        frames.append([(c.color, c.px, c.py) for c in cubes])
        if i + 1 < n_frames:
            time.sleep(NOISE_PAUSE_S)

    # Seed tracks from first frame with detections
    seed_idx = next((i for i, f in enumerate(frames) if f), None)
    if seed_idx is None:
        return {"frames": n_frames, "tracks": [], "centroid_rms": {"n": 0}}

    tracks: list[dict] = [
        {"color": color, "hist": [(px, py)]} for color, px, py in frames[seed_idx]
    ]
    for dets in frames[seed_idx + 1 :]:
        claimed: set[int] = set()
        for tr in tracks:
            last = tr["hist"][-1]
            best_j, best_d = None, 1e9
            for j, (color, px, py) in enumerate(dets):
                if j in claimed or color != tr["color"]:
                    continue
                d = math.hypot(px - last[0], py - last[1])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d < 50:
                claimed.add(best_j)
                _, px, py = dets[best_j]
                tr["hist"].append((px, py))

    per_track = []
    rms_mm_all = []
    for i, tr in enumerate(tracks):
        hist = tr["hist"]
        if len(hist) < 3:
            continue
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        rms_px = math.sqrt(
            sum((x - mx) ** 2 + (y - my) ** 2 for x, y in hist) / len(hist)
        )
        scale = _pixel_jacobian_mm_per_px(calib, mx, my)
        rms_mm = rms_px * scale
        rms_mm_all.append(rms_mm)
        per_track.append(
            {
                "track": f"{tr['color']}#{i}",
                "frames": len(hist),
                "rms_px": round(rms_px, 2),
                "rms_mm": round(rms_mm, 2),
                "mean_pixel": [round(mx, 1), round(my, 1)],
            }
        )
    return {
        "frames": n_frames,
        "tracks": per_track,
        "centroid_rms": _stats(rms_mm_all),
    }


def control_consistency(client: Mt4Client, safe_z: float) -> dict:
    client.ensure_connected()
    st = client.get_status()
    if not st.homed:
        print("Homing arm...")
        client.home()
        time.sleep(0.5)

    rows = []
    errs = []
    for x, y in CONTROL_TARGETS:
        print(f"  move_to ({x:.0f},{y:.0f},{safe_z:.0f})...")
        try:
            client.move_to(x, y, safe_z)
        except Mt4ClientError as exc:
            rows.append({"cmd": [x, y, safe_z], "error": str(exc)})
            continue
        tcp = client.get_tcp()
        e = math.hypot(tcp.x - x, tcp.y - y)
        ez = abs(tcp.z - safe_z)
        errs.append(e)
        rows.append(
            {
                "cmd": [x, y, safe_z],
                "reported": [round(tcp.x, 2), round(tcp.y, 2), round(tcp.z, 2)],
                "xy_err_mm": round(e, 3),
                "z_err_mm": round(ez, 3),
            }
        )
    # Return toward a clear pose
    try:
        client.move_to(200.0, 0.0, safe_z)
    except Mt4ClientError:
        pass
    return {
        "note": (
            "Open-loop consistency only: firmware reports pose from step "
            "counters, not an external truth. Large values would mean "
            "command truncation / kinematics bugs; small values do not "
            "prove absolute world accuracy."
        ),
        "targets": rows,
        "xy": _stats(errs),
    }


def marker_touch_consistency(calib, frame) -> dict:
    """Vision+calib path vs stored jog-to-touch robot XY (no arm motion)."""
    return camera_align_drift(calib, frame)["live_map_vs_touch"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose pick accuracy error budget")
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--calib", default=str(DEFAULT_CALIB_PATH))
    parser.add_argument("--cubetop-backup", default=str(CUBETOP_BACKUP))
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--json", default=None, help="write full report JSON here")
    parser.add_argument("--noise-frames", type=int, default=NOISE_FRAMES)
    args = parser.parse_args()

    calib = load_calibration(Path(args.calib))
    cam = args.camera if args.camera is not None else DEFAULT_CAMERA_INDEX

    print("Capturing scene...")
    frame = capture_frame(cam)

    report: dict = {
        "calib_path": str(Path(args.calib).resolve()),
        "cube_top_homography": "set" if calib.cube_top_homography else "NULL",
        "table_z": calib.table_z,
        "cube_height_mm": calib.cube_height_mm,
    }

    print("1/5 Table-plane calibration residuals...")
    report["1_table_plane_calibration"] = table_plane_residuals(calib)
    print("   ", report["1_table_plane_calibration"])

    print("2/5 Camera align / marker drift...")
    report["2_camera_align"] = camera_align_drift(calib, frame)
    print("   Drift:", report["2_camera_align"]["pixel_drift"])
    print("   Live map vs touch:", report["2_camera_align"]["live_map_vs_touch"])

    print("3/5 Cube-top parallax gap...")
    report["3_cube_top_parallax"] = cube_top_gap(
        calib, frame, Path(args.cubetop_backup)
    )
    print("   Live cube_top:", report["3_cube_top_parallax"]["live_cube_top_homography"])
    print(
        "   Est pick error:",
        report["3_cube_top_parallax"].get("estimated_pick_error_without_cubetop")
        or report["3_cube_top_parallax"].get("live_table_vs_cubetop"),
    )

    print(f"4/5 Visual detection noise ({args.noise_frames} frames)...")
    report["4_visual_detection_noise"] = detection_noise(calib, cam, args.noise_frames)
    print("   ", report["4_visual_detection_noise"]["centroid_rms"])

    if args.no_arm:
        report["5_mt4_control"] = {"skipped": True}
    else:
        port = resolve_port(args.port)
        print(f"5/5 MT4 control consistency on {port}...")
        client = Mt4Client(port=port)
        try:
            report["5_mt4_control"] = control_consistency(client, calib.safe_z)
            print("   ", report["5_mt4_control"]["xy"])
        finally:
            client.close()

    # Error budget summary (heuristic dominant terms)
    budget = []
    t = report["1_table_plane_calibration"]
    if "rms_mm" in t:
        budget.append({"source": "table_plane_calibration", "rms_mm": t["rms_mm"], "max_mm": t["max_mm"]})
    a = report["2_camera_align"]["pixel_drift"]
    if "rms_mm" in a:
        budget.append({"source": "camera_align_drift", "rms_mm": a["rms_mm"], "max_mm": a["max_mm"]})
    p = report["3_cube_top_parallax"]
    gap = p.get("estimated_pick_error_without_cubetop") or p.get("live_table_vs_cubetop") or {}
    if "rms_mm" in gap:
        budget.append(
            {
                "source": "cube_top_parallax_or_missing_correction",
                "rms_mm": gap["rms_mm"],
                "max_mm": gap["max_mm"],
            }
        )
    elif "rms_mm" in (p.get("marker_pixel_proxy") or {}):
        mp = p["marker_pixel_proxy"]
        budget.append(
            {
                "source": "cube_top_parallax_proxy_at_markers",
                "rms_mm": mp["rms_mm"],
                "max_mm": mp["max_mm"],
            }
        )
    n = report["4_visual_detection_noise"]["centroid_rms"]
    if "rms_mm" in n:
        budget.append({"source": "visual_detection_noise", "rms_mm": n["rms_mm"], "max_mm": n["max_mm"]})
    c = report.get("5_mt4_control") or {}
    if "xy" in c and "rms_mm" in c["xy"]:
        budget.append({"source": "mt4_control_consistency", "rms_mm": c["xy"]["rms_mm"], "max_mm": c["xy"]["max_mm"]})
    budget.sort(key=lambda r: r.get("rms_mm", 0), reverse=True)
    report["error_budget"] = budget

    print("\n=== ERROR BUDGET (largest first) ===")
    for row in budget:
        print(f"  {row['source']:40s}  rms={row['rms_mm']:6.2f} mm  max={row['max_mm']:6.2f} mm")

    if p.get("live_cube_top_homography") == "NULL":
        print(
            "\nWARNING: live vision_calibration.json has cube_top_homography=null. "
            "Cube picks currently use the table-plane map (typical 5–27mm parallax error)"
        )

    out = {
        "report": report,
        "budget": budget,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    else:
        # Always dump a compact json blob at the end for machine use
        print("\n--- JSON ---")
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Diagnose failed: {exc}", file=sys.stderr)
        raise
