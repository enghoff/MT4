#!/usr/bin/env python3
"""Interactive open-vocab mover: prompt for an object, place it on a free marker.

Repeatedly asks for an object description, then::

    Grounding DINO(prompt)  -> candidate boxes
    locate.measure each     -> robot XY + long-axis yaw (skip failures)
    filter                  -> reachable / pickable / not already on a marker
    motion.transfer         -> pick with 180° long-axis wrist, place on free marker

While a prompt is active, a background thread re-runs DINO on fresh camera
frames and updates the preview -- paced by detection latency (~0.3s against a
CUDA service; slower on CPU).
Re-parks after every transfer so the next planning capture is clear. Objects
that cannot be segmented, are out of reach, fail grasp feasibility, or have no
free marker are skipped. Loops until Ctrl+C / EOF.

Prereqs:
  * ``.\\scripts\\start_grounding_tunnel.ps1``
  * Arm free on the serial port; vision calibration present

Example::

    python ask_dino.py --camera 1
    python ask_dino.py --dry-run --no-park
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_vision.calib import DEFAULT_CALIB_PATH, Calibration, load_calibration
from mt4_vision.camera import DEFAULT_CAMERA_INDEX, FrameStream, CameraError, capture_frame
from mt4_vision.console import BottomUI
from mt4_vision.entities import object_entity
from mt4_vision.grounding import Detection, GroundingError, detect, health
from mt4_vision.locate import (
    LocateError,
    LocatedObject,
    grasp_feasibility,
    measure_with_box_fallback,
    refine_at_box,
    refine_at_hint,
)
from mt4_vision.motion import Grasp, transfer
from mt4_vision.sam import SamError
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
            obj = measure_with_box_fallback(
                frame, det.cx, det.cy, calib, label,
                box=(det.x1, det.y1, det.x2, det.y2),
                marker_xy=marker_xy,
                confidence=det.score,
            )
        except LocateError as exc:
            skips.append(f"couldn't measure {label}: {exc}")
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


MASK_OVERLAY_BGR = (0, 220, 255)  # amber
MASK_OVERLAY_ALPHA = 0.40


def _overlay_mask(
    out: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = MASK_OVERLAY_BGR,
    alpha: float = MASK_OVERLAY_ALPHA,
) -> None:
    """Blend ``mask`` (crop-local 0/1) onto ``out`` in place."""
    x0, y0, x1, y1 = origin
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return
    # refine_at_hint's origin is the crop box; trust mask shape over x1/y1.
    y1 = min(out.shape[0], y0 + h)
    x1 = min(out.shape[1], x0 + w)
    region = out[y0:y1, x0:x1]
    m = mask[: y1 - y0, : x1 - x0].astype(bool)
    if not m.any():
        return
    tinted = region.copy()
    tinted[m] = color
    blended = cv2.addWeighted(tinted, alpha, region, 1.0 - alpha, 0)
    region[m] = blended[m]


def _annotate_track(
    frame: np.ndarray,
    dets: list[Detection],
    prompt: str,
    status: str,
) -> np.ndarray:
    """Draw DINO boxes plus locate masks for each hit."""
    out = frame.copy()
    for i, d in enumerate(dets):
        color = (0, 255, 255) if i == 0 else (180, 180, 80)
        try:
            mask, origin = refine_at_box(frame, d.x1, d.y1, d.x2, d.y2)
            _overlay_mask(out, mask, origin, color=color)
        except (LocateError, SamError):
            # The preview is decoration -- the measurement path says plainly
            # when the segmentation service is down, so falling to the
            # desk-deviation outline here loses nothing anyone is waiting on.
            try:
                mask, origin = refine_at_hint(frame, d.cx, d.cy)
                _overlay_mask(out, mask, origin, color=color)
            except LocateError:
                pass
        cv2.rectangle(
            out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2,
        )
        cv2.putText(
            out,
            f"{d.label} {d.score:.2f}",
            (int(d.x1), max(20, int(d.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    title = status or f"tracking {prompt}"
    cv2.putText(
        out, title, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return out


def _park(client: Mt4Client, calib: Calibration) -> None:
    retreat_for_camera(client, calib)
    time.sleep(CAMERA_SETTLE_S)


class PickSearch:
    """Background scan/pick so the main thread can keep taking ``object:`` input.

    ``look_for(prompt)`` starts (or replaces) a search. A later prompt cancels
    the in-flight one via a generation counter. Arm motion is serialized.
    """

    def __init__(
        self,
        *,
        ui: BottomUI,
        calib: Calibration,
        camera: int,
        stream: FrameStream | None,
        tracker: DinoTrackPreview | None,
        client: Mt4Client | None,
        allow_markers: set[int] | None,
        box_threshold: float,
        text_threshold: float,
        dry_run: bool,
        no_park: bool,
        align_place: bool,
    ) -> None:
        self._ui = ui
        self._calib = calib
        self._camera = camera
        self._stream = stream
        self._tracker = tracker
        self._client = client
        self._allow_markers = allow_markers
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._dry_run = dry_run
        self._no_park = no_park
        self._align_place = align_place
        self._lock = threading.Lock()
        self._arm_lock = threading.Lock()
        self._prompt: str | None = None
        self._gen = 0
        self._stop = threading.Event()
        self.deny: list[tuple[float, float]] = []
        self.moved = 0
        self._thread = threading.Thread(
            target=self._loop, name="pick-search", daemon=True,
        )
        self._thread.start()

    def look_for(self, prompt: str) -> None:
        with self._lock:
            self._prompt = prompt
            self._gen += 1
        self._ui.set_status(f"looking for {prompt}…")
        if self._tracker is not None:
            self._tracker.track(prompt, status=f"finding {prompt}")

    def _target(self) -> tuple[str | None, int]:
        with self._lock:
            return self._prompt, self._gen

    def _clear_if_current(self, gen: int) -> None:
        with self._lock:
            if self._gen == gen:
                self._prompt = None

    def _loop(self) -> None:
        skip_gen = -1
        markers_gen = -1
        while not self._stop.is_set():
            if self._tracker is not None and self._tracker.stopped_by_user():
                break
            prompt, gen = self._target()
            if not prompt:
                time.sleep(0.05)
                continue

            try:
                frame = (
                    self._stream.fresh(min_advance=1) if self._stream is not None
                    else capture_frame(self._camera)
                )
            except CameraError as exc:
                self._ui.set_status(f"camera: {exc}")
                time.sleep(0.5)
                continue

            scene = capture_scene(self._calib, frame)
            free = scene.placeable_markers()
            if self._allow_markers is not None:
                free = [m for m in free if m.marker_id in self._allow_markers]
            free = sorted(free, key=lambda m: m.marker_id)
            if not free:
                if markers_gen != gen:
                    self._ui.set_status("no free markers — still looking…")
                    markers_gen = gen
                if self._tracker is not None:
                    self._tracker.set_status("waiting for free marker")
                continue

            try:
                candidates, skips = _gather_candidates(
                    frame, self._calib, scene, prompt,
                    box_threshold=self._box_threshold,
                    text_threshold=self._text_threshold,
                    deny=self.deny,
                )
            except GroundingError as exc:
                self._ui.set_status(f"couldn't detect: {exc}")
                time.sleep(0.5)
                continue

            if not candidates:
                if skips and skip_gen != gen:
                    note = skips[0]
                    if note.startswith("can't pick") or "measure" in note:
                        self._ui.set_status(f"{note} — still looking…")
                        skip_gen = gen
                    else:
                        self._ui.set_status(f"looking for {prompt}…")
                        skip_gen = gen
                if self._tracker is not None:
                    self._tracker.set_status(f"finding {prompt}")
                continue

            prompt_now, gen_now = self._target()
            if gen_now != gen or prompt_now != prompt:
                continue

            cand = candidates[0]
            marker = free[0]
            self._ui.set_status(f"found {cand.obj.label}")
            if self._tracker is not None:
                self._tracker.set_status(
                    f"moving {cand.obj.label} -> marker {marker.marker_id}"
                )

            with self._arm_lock:
                prompt_now, gen_now = self._target()
                if gen_now != gen or prompt_now != prompt:
                    continue
                if self._dry_run or self._client is None:
                    self._ui.set_status(
                        f"would move {cand.obj.label} -> marker {marker.marker_id}"
                    )
                    self._clear_if_current(gen)
                    if self._tracker is not None:
                        self._tracker.set_status(f"tracking {prompt}")
                    continue

                entity = object_entity(cand.obj, 1, scene=scene)
                src = entity.as_grasp(self._calib)
                if self._align_place:
                    dst = Grasp(
                        marker.x, marker.y,
                        yaw_deg=cand.obj.axis_yaw_deg,
                        yaw_period_deg=src.yaw_period_deg,
                        grip_close_s=src.grip_close_s,
                    )
                else:
                    dst = Grasp(marker.x, marker.y)
                self._ui.set_status(
                    f"moving {cand.obj.label} -> marker {marker.marker_id}"
                )
                try:
                    transfer(self._client, self._calib, src, dst)
                    self._ui.set_status(
                        f"moved {cand.obj.label} -> marker {marker.marker_id}"
                    )
                    self.moved += 1
                except Mt4ClientError as exc:
                    self._ui.set_status(f"couldn't move: {exc}")
                    self.deny.append((cand.obj.x, cand.obj.y))
                if self._tracker is not None:
                    self._tracker.set_status(f"tracking {prompt}")
                if not self._no_park:
                    try:
                        _park(self._client, self._calib)
                    except Mt4ClientError as exc:
                        self._ui.set_status(f"couldn't park: {exc}")
                self._clear_if_current(gen)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

class DinoTrackPreview:
    """Background DINO → segment → annotate → preview loop for the active prompt.

    Pace is whatever ``detect`` takes (plus a fresh camera frame and a cheap
    locate mask per hit). Runs through pick/place and the next prompt wait so
    the last requested object stays visually tracked.
    """

    def __init__(
        self,
        stream: FrameStream,
        preview: LivePreview,
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        self._stream = stream
        self._preview = preview
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._lock = threading.Lock()
        self._prompt: str | None = None
        self._status = ""
        self._stop = threading.Event()
        self._user_quit = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="dino-track", daemon=True,
        )
        self._thread.start()

    def track(self, prompt: str, status: str = "") -> None:
        with self._lock:
            self._prompt = prompt
            self._status = status

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def stopped_by_user(self) -> bool:
        return self._user_quit.is_set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                prompt = self._prompt
                status = self._status
            if not prompt:
                time.sleep(0.05)
                continue
            try:
                frame = self._stream.fresh(min_advance=1)
                dets = detect(
                    frame, prompt,
                    box_threshold=self._box_threshold,
                    text_threshold=self._text_threshold,
                )
            except (CameraError, GroundingError):
                time.sleep(0.2)
                continue
            except Exception:  # noqa: BLE001 -- keep the feed alive
                time.sleep(0.2)
                continue
            try:
                self._preview.show(_annotate_track(frame, dets, prompt, status))
            except PreviewStopped:
                self._user_quit.set()
                break

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._preview.close()


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

    stream: FrameStream | None = None
    tracker: DinoTrackPreview | None = None
    if not args.no_preview:
        stream = FrameStream(args.camera)
        tracker = DinoTrackPreview(
            stream,
            LivePreview("move object"),
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

    ui = BottomUI()
    search = PickSearch(
        ui=ui,
        calib=calib,
        camera=args.camera,
        stream=stream,
        tracker=tracker,
        client=client,
        allow_markers=allow_markers,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        dry_run=args.dry_run,
        no_park=args.no_park,
        align_place=args.align_place,
    )
    ui.set_status("ready")
    try:
        while True:
            if tracker is not None and tracker.stopped_by_user():
                break
            prompt = ui.read_line()
            if prompt is None:
                break
            if not prompt:
                continue
            if tracker is not None and tracker.stopped_by_user():
                break
            search.look_for(prompt)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        search.close()
        ui.close()
        if tracker is not None:
            tracker.close()
        if stream is not None:
            stream.close()
        if client is not None:
            try:
                if not args.no_park:
                    retreat_for_camera(client, calib)
            except Mt4ClientError:
                pass
            client.close()


if __name__ == "__main__":
    sys.exit(main())
