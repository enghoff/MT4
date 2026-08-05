"""Local HTTP MCP server for MT4 Cartesian control and status."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from mt4_jog.client import Mt4Client, Mt4ClientError
from mt4_jog.joints import DEFAULT_BAUD
from mt4_jog.joints import DEFAULT_PORT as DEFAULT_SERIAL_PORT
from mt4_mcp.auth import build_auth_provider, oauth_enabled

load_dotenv()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8787

_client: Mt4Client | None = None


def _preimport_vision() -> None:
    """Load the OpenCV/numpy C extensions on the main thread, before serving.

    FastMCP runs sync tools in worker threads, and the vision tools import cv2
    lazily inside them (see the entity layer below). On Windows that first
    `import cv2` from a worker thread of a stdio child deadlocks in numpy's
    extension loader -- faulthandler showed mt4_scene parked forever in
    numpy/_core/overrides.py create_module, so the tool never returned and the
    client timed out with no error. Importing it here makes the tool-level
    imports cache hits. Hosts without the camera stack are unaffected: the
    motion tools keep working and the vision tools raise as they did before.
    """
    try:
        import cv2  # noqa: F401
    except Exception:  # noqa: BLE001 -- no camera stack is a supported setup
        pass


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def get_client() -> Mt4Client:
    if _client is None:
        raise RuntimeError("MT4 client is not initialized")
    _client.ensure_connected()
    return _client


@asynccontextmanager
async def lifespan(_app: FastMCP):
    global _client
    serial_port = os.environ.get("MT4_SERIAL_PORT", DEFAULT_SERIAL_PORT)
    baud = _env_int("MT4_BAUD", DEFAULT_BAUD)
    _client = Mt4Client(port=serial_port, baud=baud)
    try:
        yield
    finally:
        if _client is not None:
            _client.close()
            _client = None


def create_mcp(*, auth: Any | None = None) -> FastMCP:
    # Per-server state behind the entity layer. Deliberately not module-level:
    # each create_mcp() gets its own, so a test server cannot inherit another's
    # snapshot ids or registered objects.
    #   _snapshot -- the latest Snapshot, the only source of valid entity ids
    #   _view     -- (token, frame) from the last mt4_camera_view, so a reported
    #                pixel is measured against the exact frame it was read from
    #   _objects  -- id -> LocatedObject registry surviving across snapshots,
    #                since a VLM-named object has no colour rule to re-detect it
    #   _obj_seq  -- monotonic, NEVER reused. An obj_N id must name one physical
    #                thing for the life of the server: _build_snapshot drops
    #                objects that fail to re-acquire, so numbering from
    #                len(_objects) would both re-issue a live key and (via a
    #                positional build_snapshot) slide surviving ids onto their
    #                neighbours -- the exact substitution the entity layer exists
    #                to prevent.
    _snapshot: Any = None
    _view: tuple[str, Any] | None = None
    _objects: dict[str, Any] = {}
    _token_seq = 0
    _obj_seq = 0

    server = FastMCP(
        name="MT4 Robot",
        instructions=(
            "Control and read status from a WLKATA MT4 arm over serial. "
            "TCP x/y/z are in mm with origin at the base under J1's pivot. "
            "j4 is world-frame gripper yaw in degrees. "
            "Execute motion commands directly when asked -- never ask the "
            "user to confirm before calling a tool. mt4_move_to and "
            "mt4_move_relative move the arm immediately. Check mt4_status "
            "first when you need the current pose or homed flag. "
            "mt4_move_to requires homing this session (mt4_home) first; "
            "mt4_home returns homed and tcp on success.\n\n"
            "To move things on the work surface, work by ENTITY ID. Call "
            "mt4_scene to list what is there -- cube_N, marker_N (the printed "
            "tag number), slot_N, obj_N -- then mt4_pick(id) and "
            "mt4_place(id). Ids belong to the snapshot that produced them, so "
            "re-call mt4_scene after anything moves.\n\n"
            "If the object asked for is not in mt4_scene's list, it may be "
            "something the cube detector cannot see (a pen, a key). Call "
            "mt4_camera_view, read its pixel off the drawn grid, and register "
            "it with mt4_locate_at_pixel; then it has an id like the rest. "
            "Alternatively mt4_locate_by_prompt runs Grounding DINO and "
            "registers the top hit the same way.\n\n"
            "Do not substitute a target. When a referent cannot be resolved, "
            "or an entity reports pickable/placeable false, say so and quote "
            "the reason field -- the physical constraint it names (keep-out, "
            "reach, finger clearance, an undecoded tag) is the useful answer. "
            "Picking a different object than the one asked for is a worse "
            "outcome than doing nothing, because nothing in this stack can "
            "detect that it happened.\n\n"
            "mt4_pick_at / mt4_place_at / mt4_transfer take raw robot-frame "
            "coordinates and skip all of the above. Use them for probing and "
            "calibration, not for acting on something the camera can see."
        ),
        auth=auth,
        lifespan=lifespan,
    )

    @server.tool
    def mt4_status() -> dict[str, Any]:
        """Full arm status: homed flag, mode, joints, TCP pose, drivers, jog."""
        try:
            return get_client().get_status().as_dict()
        except Mt4ClientError as exc:
            return {"error": str(exc)}

    @server.tool
    def mt4_tcp() -> dict[str, Any]:
        """Current Cartesian TCP pose (x/y/z mm, world-frame j4 deg, grip, speed)."""
        try:
            return get_client().get_tcp().as_dict()
        except Mt4ClientError as exc:
            return {"error": str(exc)}

    @server.tool
    def mt4_stop() -> dict[str, Any]:
        """Stop jog and cancel any in-progress coordinated move."""
        try:
            lines = get_client().stop()
            return {"ok": True, "lines": lines}
        except Mt4ClientError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_home() -> dict[str, Any]:
        """Home J1 and J2 by driving them into their limit switches, then
        reference J3 indirectly through J2's switch (J3 has no switch of its
        own). Required once per power cycle/session before mt4_move_to will
        accept absolute moves -- check mt4_status's `homed` field first to
        see if this is even necessary.

        Runs immediately, no confirmation or workspace check required --
        call directly. The arm moves on its own, and both J1 and J2 travel
        to their hard limit switches during the seek. Takes up to ~30s;
        can take longer (up to 180s) if a limit switch isn't found on the
        first pass. On success, returns `homed` and `tcp` from a fresh
        status query so callers don't need a separate mt4_status round-trip.
        """
        try:
            return get_client().home()
        except Mt4ClientError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_move_to(
        x: float,
        y: float,
        z: float,
        j4: float | None = None,
        grip: int = 0,
        speed_us: int = 0,
    ) -> dict[str, Any]:
        """Move the TCP to an absolute Cartesian position in a straight
        world-frame line (firmware `mp`). Requires the arm to have homed
        this session (mt4_status's `homed` field) -- call mt4_home first if
        not. Blocks until the move completes or times out (~30s), then
        returns the arm's final pose.

        Args:
            x: Target TCP X in mm, origin at the base under J1's pivot.
            y: Target TCP Y in mm.
            z: Target TCP Z in mm.
            j4: Target gripper yaw in world-frame degrees. If omitted, the
                current yaw is reused, which makes the firmware hold gripper
                orientation fixed in world space during the move (like
                `orient on`) rather than rotating it.
            grip: Absolute gripper position, 120 (open) to 285 (closed).
                0 (default) leaves the gripper wherever it currently is.
            speed_us: Step period in microseconds, 700 (fast) to 4000
                (slow). 0 (default) leaves the current speed unchanged.
        """
        try:
            return get_client().move_to(
                x, y, z, j4=j4, grip=grip, speed_us=speed_us
            )
        except Mt4ClientError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_move_relative(
        dj1: int,
        dj2: int,
        dj3: int,
        dj4: int,
        dgrip: int = 0,
    ) -> dict[str, Any]:
        """Nudge each joint by a relative step count, all axes finishing
        together (firmware `m`). Does not require homing -- deltas are
        relative to whatever the current step counters are. Prefer
        mt4_move_to for absolute Cartesian targets once homed; use this for
        small joint-space nudges or before homing has run. Blocks until the
        move completes or times out (~30s).

        Args:
            dj1: J1 (base) step delta, signed.
            dj2: J2 (shoulder) step delta, signed.
            dj3: J3 (elbow) step delta, signed.
            dj4: J4 (wrist) step delta, signed.
            dgrip: Gripper S-value delta, signed (gripper spans 120-285).
                0 (default) leaves the gripper unchanged.
        """
        try:
            return get_client().move_relative(dj1, dj2, dj3, dj4, dgrip=dgrip)
        except Mt4ClientError as exc:
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_gripper(action: str | int) -> dict[str, Any]:
        """Open, close, stop, or set the gripper (firmware `g`).
        Args:
            action: One of the strings "open", "close", "stop" (start/stop
                a sweep between the gripper's travel limits), or an absolute
                integer S-value from 120 (fully open) to 285 (fully closed).
        """
        try:
            return get_client().gripper(action)
        except Mt4ClientError as exc:
            return {"ok": False, "error": str(exc)}

    # Vision tools import cv2 lazily so the motion tools keep working on
    # hosts without the camera stack installed.
    # -- entity layer -----------------------------------------------------
    #
    # The model addresses things by id, not by colour or coordinate. This
    # replaces mt4_pick_cube(color), whose implicit pick_largest_cube meant
    # "pick the red one" silently chose among several and could never fail.
    #
    # Ids live only as long as the snapshot that produced them (mt4_vision.scene
    # keeps no persistent tracks, so a vacated pose cannot linger). Every acting
    # tool re-captures and re-acquires before it moves; a target that has moved
    # since the snapshot is refused, not approximated.

    def _next_token(prefix: str) -> str:
        nonlocal _token_seq
        _token_seq += 1
        return f"{prefix}{_token_seq}"

    def _register_object(obj: Any, scene: Any) -> tuple[str, Any]:
        """Give a measured object a permanent id and return (id, Entity).

        The one place obj_N ids are minted, so every registration path agrees
        on the invariant: the id is the registry key, the registry key is what
        build_snapshot labels the entity with, and neither is ever re-derived
        from a position in a list.
        """
        from mt4_vision.entities import object_entity

        nonlocal _objects, _obj_seq
        _obj_seq += 1
        key = f"obj_{_obj_seq}"
        _objects = {**_objects, key: obj}
        return key, object_entity(obj, key, scene=scene)

    def _build_snapshot(camera: int = 1) -> tuple[Any, Any]:
        """Fresh frame -> (Scene, Snapshot), re-acquiring registered objects."""
        from mt4_vision.calib import load_calibration
        from mt4_vision.camera import capture_frame
        from mt4_vision.entities import build_snapshot
        from mt4_vision.locate import relocate
        from mt4_vision.scene import capture_scene

        nonlocal _snapshot, _objects
        calib = load_calibration()
        frame = capture_frame(camera)
        scene = capture_scene(calib, frame)

        # A located object is not re-detectable from scratch (there is no colour
        # rule for "pen"), so each is template-matched in this frame. Strict:
        # anything that fails to re-acquire is dropped rather than reported at a
        # stale position the arm would then act on.
        marker_xy = [(m.x, m.y) for m in scene.markers]
        refreshed: dict[str, Any] = {}
        for key, obj in _objects.items():
            again = relocate(frame, obj, calib, marker_xy=marker_xy)
            if again is not None:
                refreshed[key] = again
        _objects = refreshed

        # The registry itself, not its values: ids must survive a drop above.
        snapshot = build_snapshot(
            scene, token=_next_token("s"), objects=_objects
        )
        _snapshot = snapshot
        return scene, snapshot

    @server.tool
    def mt4_scene(camera: int = 1) -> dict[str, Any]:
        """List everything on the work surface as addressable entities.

        Returns {snapshot, summary, entities}. Each entity has an `id` to pass
        to mt4_pick / mt4_place, robot-frame x/y in mm, and `pickable` /
        `placeable` flags. When one of those is false, `reason` says which
        physical constraint stopped it (keep-out, reach, finger clearance,
        off the desk, outside the camera frame, undecoded tag) -- report that
    reason rather
        than trying a different target.

        Entity kinds: `cube_N` (colour-detected cubes), `marker_N` (the printed
        ArUco tag number, so "marker 1" is `marker_1`), `slot_N` (open table
        spots), `obj_N` (anything registered via mt4_locate_at_pixel).

        Detections that are real objects but not valid pick targets ARE listed,
        with their reason -- "that blob is past the desk's back edge, most
        likely the arm's own paint" is a better answer than omitting it.

        Uses a fresh camera frame, so ids change between calls. Re-call after
        anything moves. Requires the calibration from `python calibrate_vision.py`.

        Args:
            camera: USB camera index (default 1 -- the overhead work camera).
        """
        try:
            _scene, snapshot = _build_snapshot(camera)
            return {"ok": True, **snapshot.as_dict()}
        except Exception as exc:  # noqa: BLE001 -- surface camera/calib errors
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_camera_view(camera: int = 1) -> Any:
        """Return the current camera frame as an image, for finding something
        the cube detector cannot see (a pen, a key, a screwdriver).

        The frame carries a numbered 100px pixel grid and the ids of everything
        already detected. Read the target's pixel coordinates off that grid and
        pass them to mt4_locate_at_pixel together with the `view` token in the
        accompanying text. Point anywhere ON the object -- its centre is
        recovered by segmentation, so being off by a few tens of pixels along a
        pen is fine; landing beside it on the desk is not.

        Only for objects mt4_scene does not already list. If the thing you want
        is already a `cube_N` or `marker_N`, use that id instead.

        Args:
            camera: USB camera index (default 1 -- the overhead work camera).
        """
        try:
            import cv2
            from fastmcp.utilities.types import Image

            from mt4_vision.camera import capture_frame
            from mt4_vision.preview import annotate_for_pointing

            nonlocal _view
            frame = capture_frame(camera)
            entities = [] if _snapshot is None else _snapshot.entities
            annotated = annotate_for_pointing(frame, entities)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return {"ok": False, "error": "JPEG encode failed"}
            token = _next_token("v")
            _view = (token, frame)
            h, w = frame.shape[:2]
            return [
                f"view={token} size={w}x{h} grid=100px -- report pixels in this "
                f"space and pass view={token} to mt4_locate_at_pixel",
                Image(data=bytes(buf), format="jpeg"),
            ]
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_locate_at_pixel(
        px: float, py: float, label: str, view: str
    ) -> dict[str, Any]:
        """Register the object at pixel (px, py) of a mt4_camera_view frame as a
        new `obj_N` entity, and return it.

        Measured against the exact frame that view returned -- not a fresh
        capture -- so the coordinates mean what you saw. A stale or unknown
        `view` token is refused rather than silently re-measured against a
        different frame.

        Segmentation recovers the object's true centre, long axis and size in
        millimetres, so only the pixel needs to come from you. Thin objects
        lying flat on the table only: height is assumed to be table level, and
        an implausible measurement is refused instead of picked at.

        Args:
            px: X pixel in the mt4_camera_view frame.
            py: Y pixel in the mt4_camera_view frame.
            label: What to call it, e.g. "pen". Used as the entity's label.
            view: The `view=` token from the mt4_camera_view call you read.
        """
        try:
            from mt4_vision.calib import load_calibration
            from mt4_vision.locate import LocateError, grasp_feasibility, measure
            from mt4_vision.scene import capture_scene

            if _view is None:
                return {
                    "ok": False,
                    "error": "no camera view to point at -- call mt4_camera_view first",
                }
            token, frame = _view
            if view != token:
                return {
                    "ok": False,
                    "error": (
                        f"view {view!r} is stale (current view is {token!r}); the "
                        "pixel coordinates would refer to a different frame -- "
                        "call mt4_camera_view again and re-read them"
                    ),
                }
            calib = load_calibration()
            scene = capture_scene(calib, frame)
            try:
                obj = measure(
                    frame, px, py, calib, label,
                    marker_xy=[(m.x, m.y) for m in scene.markers],
                )
            except LocateError as exc:
                return {"ok": False, "error": str(exc)}
            ok, reason = grasp_feasibility(obj, calib)

            _key, entity = _register_object(obj, scene)
            out = {"ok": True, "entity": entity.as_dict()}
            if not ok and entity.reason is None:
                out["entity"]["pickable"] = False
                out["entity"]["reason"] = reason
            return out
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_locate_by_prompt(
        prompt: str,
        camera: int = 1,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Find an object by open-vocab text via Grounding DINO, then register
        the top hit as `obj_N` the same way mt4_locate_at_pixel does.

        Requires the Grounding DINO service reachable at `MT4_GROUNDING_URL`
        (default `http://127.0.0.1:8765`); if it runs on another host behind an
        SSH forward, `scripts/start_grounding_tunnel.ps1` opens it. Prefer
        mt4_scene ids for coloured cubes; use this for things HSV cannot name
        (pen, key, ...).

        Args:
            prompt: What to look for, e.g. "pen" or "screwdriver".
            camera: USB camera index (default 1).
            box_threshold: Grounding DINO box score floor.
            text_threshold: Grounding DINO text score floor.
        """
        try:
            from mt4_vision.calib import load_calibration
            from mt4_vision.camera import capture_frame
            from mt4_vision.grounding import GroundingError, detect
            from mt4_vision.locate import LocateError, grasp_feasibility, measure_with_box_fallback
            from mt4_vision.scene import capture_scene

            nonlocal _view
            frame = capture_frame(camera)
            try:
                dets = detect(
                    frame, prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            except GroundingError as exc:
                return {"ok": False, "error": str(exc)}
            if not dets:
                return {
                    "ok": False,
                    "error": f"no detections for prompt={prompt!r}",
                    "detections": [],
                }
            best = dets[0]
            calib = load_calibration()
            scene = capture_scene(calib, frame)
            label = best.label.strip() or prompt.strip().rstrip(".")
            try:
                obj = measure_with_box_fallback(
                    frame, best.cx, best.cy, calib, label,
                    box=(best.x1, best.y1, best.x2, best.y2),
                    marker_xy=[(m.x, m.y) for m in scene.markers],
                    confidence=best.score,
                )
            except LocateError as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "detections": [d.as_dict() for d in dets],
                }
            ok, reason = grasp_feasibility(obj, calib)
            _key, entity = _register_object(obj, scene)
            token = _next_token("v")
            _view = (token, frame)
            out = {
                "ok": True,
                "entity": entity.as_dict(),
                "detection": best.as_dict(),
                "detections": [d.as_dict() for d in dets],
                "view": token,
            }
            if not ok and entity.reason is None:
                out["entity"]["pickable"] = False
                out["entity"]["reason"] = reason
            return out
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _reacquire(entity: Any, camera: int = 1) -> tuple[Any, Any, str | None]:
        """(calib, Grasp, error) for acting on ``entity`` against a fresh frame.

        Cubes are re-detected by colour near the remembered position; located
        objects are template-matched. Either way the arm acts on where the thing
        is NOW, not on a snapshot that may be seconds old -- and if it is no
        longer there, that is an error, never a nearest-match guess.
        """
        from mt4_vision.calib import load_calibration
        from mt4_vision.camera import capture_frame
        from mt4_vision.entities import KIND_CUBE, KIND_MARKER, KIND_OBJECT, KIND_SLOT
        from mt4_vision.locate import relocate
        from mt4_vision.scene import capture_scene
        from mt4_vision.workspace import dist_mm

        MOVED_TOLERANCE_MM = 15.0
        calib = load_calibration()

        if entity.kind in (KIND_MARKER, KIND_SLOT):
            # Fixed calibrated positions; nothing to re-detect. A destination
            # has no orientation of its own, so the landing yaw is squared to
            # the world axes (square_place) rather than left at whatever the
            # pick happened to leave the wrist at -- what pickplace.place has
            # always done via resolve_place_j4.
            from mt4_vision.motion import square_place

            return calib, square_place(entity.x, entity.y), None

        frame = capture_frame(camera)
        if entity.kind == KIND_OBJECT:
            obj = _objects.get(entity.id)
            if obj is None:
                return calib, None, f"{entity.id} is no longer registered"
            scene = capture_scene(calib, frame)
            again = relocate(
                frame, obj, calib, marker_xy=[(m.x, m.y) for m in scene.markers]
            )
            if again is None:
                return calib, None, (
                    f"{entity.id} ({entity.label}) could not be re-acquired in a "
                    "fresh frame -- call mt4_scene again"
                )
            from mt4_vision.entities import object_entity

            fresh = object_entity(again, entity.id, scene=scene)
            if not fresh.pickable:
                return calib, None, f"{entity.id} is not pickable: {fresh.reason}"
            return calib, fresh.as_grasp(calib), None

        if entity.kind == KIND_CUBE:
            scene = capture_scene(calib, frame)
            colour = entity.label.split()[0]
            candidates = [
                c
                for c in scene.pickable(scene.cubes)
                if c.color == colour
                and dist_mm(float(c.x), float(c.y), entity.x, entity.y)
                <= MOVED_TOLERANCE_MM
            ]
            if not candidates:
                return calib, None, (
                    f"{entity.id} ({entity.label}) is no longer a pickable "
                    f"detection within {MOVED_TOLERANCE_MM:.0f}mm of "
                    f"({entity.x:.0f}, {entity.y:.0f}) -- the scene changed; "
                    "call mt4_scene again"
                )
            cube = min(
                candidates,
                key=lambda c: dist_mm(float(c.x), float(c.y), entity.x, entity.y),
            )
            from mt4_vision.motion import YAW_PERIOD_SQUARE, Grasp

            return calib, Grasp(
                float(cube.x), float(cube.y),
                yaw_deg=(
                    cube.yaw_deg
                    if getattr(calib, "face_align_picks", True)
                    else None
                ),
                yaw_period_deg=YAW_PERIOD_SQUARE,
            ), None

        return calib, None, f"{entity.id} is a {entity.kind}, which cannot be picked"

    def _lookup(entity_id: str) -> tuple[Any, str | None]:
        if _snapshot is None:
            return None, "no scene snapshot yet -- call mt4_scene first"
        entity = _snapshot.get(entity_id)
        if entity is None:
            known = ", ".join(e.id for e in _snapshot.entities) or "(none)"
            return None, (
                f"no entity {entity_id!r} in snapshot {_snapshot.token} -- "
                f"known ids: {known}. Do not substitute a similar one; re-scan "
                "or report that the thing asked for is not there."
            )
        return entity, None

    @server.tool
    def mt4_pick(entity_id: str, camera: int = 1) -> dict[str, Any]:
        """Pick up the entity with this id (from mt4_scene / mt4_locate_at_pixel).

        Re-detects the target on a fresh frame first and refuses if it is no
        longer there, is not pickable, or the id is unknown -- there is no
        nearest-match fallback, because picking the wrong object silently is
        worse than reporting that the referent could not be resolved.

        The whole grasp is one queued firmware path: the jaws open during the
        transit, wait, descend slowly, close with the arm stopped, and lift.
        Wrist angle comes from the entity's own geometry -- square cube faces are
        met face-on, elongated objects are gripped across their long axis.

        Args:
            entity_id: e.g. "cube_2" or "obj_1".
            camera: USB camera index for re-detection (default 1).
        """
        try:
            from mt4_vision.motion import pick_at

            from mt4_vision.entities import KIND_CUBE, KIND_OBJECT

            entity, err = _lookup(entity_id)
            if err is not None:
                return {"ok": False, "error": err}
            # Kind first: `reason` explains the capability an entity is *for*,
            # so a marker's reason is about placing and would read as a bare
            # "None" here. Say what is actually wrong instead.
            if entity.kind not in (KIND_CUBE, KIND_OBJECT):
                return {
                    "ok": False,
                    "error": (
                        f"{entity_id} is a {entity.kind}, not something to pick "
                        f"up -- pick a cube_N or obj_N. To move the arm there "
                        f"instead, use mt4_move_to with x={entity.x:.1f} "
                        f"y={entity.y:.1f}."
                    ),
                }
            if not entity.pickable:
                return {
                    "ok": False,
                    "error": f"{entity_id} is not pickable: {entity.reason}",
                }
            calib, grasp, err = _reacquire(entity, camera)
            if err is not None:
                return {"ok": False, "error": err}
            result = pick_at(get_client(), calib, grasp)
            result["entity_id"] = entity_id
            result["label"] = entity.label
            return result
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_place(entity_id: str) -> dict[str, Any]:
        """Put the currently-held object down at this entity's position.

        Destinations are `marker_N` (a calibrated ArUco marker) or `slot_N` (an
        open table spot) from mt4_scene. Refuses a destination whose `placeable`
        is false -- an occupied marker, or one whose tag did not decode this
        frame (which cannot be proven empty, so placing there risks stacking).

        Args:
            entity_id: e.g. "marker_1" or "slot_2".
        """
        try:
            from mt4_vision.motion import place_at

            from mt4_vision.entities import KIND_MARKER, KIND_SLOT

            entity, err = _lookup(entity_id)
            if err is not None:
                return {"ok": False, "error": err}
            # Kind first, for the same reason as mt4_pick: a cube's `reason`
            # is about picking, so quoting it here would say nothing useful.
            if entity.kind not in (KIND_MARKER, KIND_SLOT):
                return {
                    "ok": False,
                    "error": (
                        f"{entity_id} is a {entity.kind}, not a place destination "
                        f"-- place onto a marker_N or slot_N, or give explicit "
                        f"coordinates with mt4_place_at. (Nothing here stacks one "
                        f"object on another.)"
                    ),
                }
            if not entity.placeable:
                return {
                    "ok": False,
                    "error": f"{entity_id} is not a valid destination: {entity.reason}",
                }
            calib, grasp, err = _reacquire(entity)
            if err is not None:
                return {"ok": False, "error": err}
            result = place_at(get_client(), calib, grasp)
            result["entity_id"] = entity_id
            return result
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # -- raw-pose primitives ----------------------------------------------

    @server.tool
    def mt4_pick_at(
        x: float, y: float, yaw_deg: float | None = None, center: bool = False
    ) -> dict[str, Any]:
        """Pick at explicit robot-frame coordinates, bypassing the entity layer.

        For probing and calibration work. Prefer mt4_pick with an id for
        anything the camera can see -- coordinates carry no identity, so nothing
        checks that the thing you meant is still there.

        Args:
            x: Target X in mm (robot frame).
            y: Target Y in mm (robot frame).
            yaw_deg: The OBJECT's cube-face edge angle in the robot frame (not a
                wrist angle). Omit to keep the current wrist.
            center: Run the extra +/-90 deg re-grip that squares a cube under the
                gripper. Costs several extra moves; it cannot be queued.
        """
        try:
            from mt4_vision.calib import load_calibration
            from mt4_vision.motion import Grasp, pick_at

            calib = load_calibration()
            return pick_at(
                get_client(), calib, Grasp(x, y, yaw_deg=yaw_deg, center=center)
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_place_at(
        x: float, y: float, yaw_deg: float | None = None
    ) -> dict[str, Any]:
        """Release the held object at explicit robot-frame (x, y).

        Moves there at the calibrated safe height, descends slowly, releases
        just above table height so the object drops rather than being pressed
        in, and lifts clear. Get coordinates from mt4_scene, and keep at least
        one object width (~35mm) from anything already down.

        Args:
            x: Target X in mm (robot frame).
            y: Target Y in mm (robot frame).
            yaw_deg: Desired landing angle of the object in the robot frame.
                Omit to square the held object to the X/Y axes.
        """
        try:
            from mt4_vision.calib import load_calibration
            from mt4_vision.motion import Grasp, place_at, square_place

            calib = load_calibration()
            # Omitted yaw squares to the world axes; a bare Grasp would instead
            # preserve whatever wrist the pick left, which is not what "put it
            # down at (x, y)" means.
            dst = square_place(x, y) if yaw_deg is None else Grasp(x, y, yaw_deg=yaw_deg)
            return place_at(get_client(), calib, dst)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_transfer(
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        from_yaw_deg: float | None = None,
        to_yaw_deg: float | None = None,
        center: bool = False,
    ) -> dict[str, Any]:
        """Move an object from one robot-frame position to another in one go.

        Faster and smoother than mt4_pick_at followed by mt4_place_at: the
        post-grip lift folds into the carry and both gripper actions are queued
        firmware stations, so it is a single continuous path with no stop
        between grip and transit.

        Args:
            from_x: Pick X in mm (robot frame).
            from_y: Pick Y in mm (robot frame).
            to_x: Place X in mm (robot frame).
            to_y: Place Y in mm (robot frame).
            from_yaw_deg: The object's edge/axis angle at the pick, in the robot
                frame. Omit to keep the current wrist.
            to_yaw_deg: Desired landing angle. Omit to square to the X/Y axes.
            center: Run the +/-90 deg re-grip that squares a cube after gripping.
                Forces the slower split form; it cannot be queued.
        """
        try:
            from mt4_vision.calib import load_calibration
            from mt4_vision.motion import Grasp, square_place, transfer

            calib = load_calibration()
            # Omitted landing yaw squares to the world axes -- see mt4_place_at.
            dst = (
                square_place(to_x, to_y)
                if to_yaw_deg is None
                else Grasp(to_x, to_y, yaw_deg=to_yaw_deg)
            )
            return transfer(
                get_client(),
                calib,
                Grasp(from_x, from_y, yaw_deg=from_yaw_deg, center=center),
                dst,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @server.tool
    def mt4_goto_marker(
        marker_id: int, touch: bool = False, camera: int = 1
    ) -> dict[str, Any]:
        """Move the TCP to a calibration ArUco marker's position -- a
        calibration accuracy check, not a normal operation. Re-detects the
        marker on a fresh frame and converts its pixel position through the
        calibration. Hovers at the calibrated safe height by default (won't
        crash into the table even if the calibration is off); pass
        touch=true to descend and physically touch the table at that spot.

        Args:
            marker_id: ArUco marker id to move to (see mt4_scene's calibration
                or the physical markers on the work surface).
            touch: If true, descend to table height instead of hovering.
            camera: USB camera index (default 1 -- the overhead work camera).
        """
        try:
            from mt4_vision.calib import load_calibration
            from mt4_vision.camera import capture_frame
            from mt4_vision.detect import detect_markers
            from mt4_vision.pickplace import goto_marker

            calib = load_calibration()
            markers = detect_markers(capture_frame(camera))
            match = next((m for m in markers if m.marker_id == marker_id), None)
            if match is None:
                return {
                    "ok": False,
                    "error": f"marker {marker_id} not in view "
                    f"(visible: {sorted(m.marker_id for m in markers)})",
                }
            x, y = calib.pixel_to_robot(match.px, match.py)
            result = goto_marker(get_client(), calib, x, y, touch=touch)
            result["marker_id"] = marker_id
            return result
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    return server


# Default module-level server for imports/tests (no OAuth).
mcp = create_mcp()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MT4 MCP server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="stdio transport for Cursor/Claude Desktop (default: HTTP)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MT4_MCP_HOST", DEFAULT_HOST),
        help="HTTP bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("MT4_MCP_PORT", DEFAULT_MCP_PORT),
        help="HTTP port (default: 8787)",
    )
    args = parser.parse_args()
    _preimport_vision()

    if args.stdio:
        create_mcp().run(transport="stdio")
        return

    auth = build_auth_provider() if oauth_enabled() else None
    public = os.environ.get("MT4_MCP_PUBLIC", "").lower() in ("1", "true", "yes")
    http_kwargs: dict[str, object] = {
        "transport": "http",
        "host": args.host,
        "port": args.port,
        "path": os.environ.get("MT4_MCP_PATH", "/mcp"),
    }
    if public or auth is not None:
        # Allow ngrok / reverse-proxy Host headers through to the MCP endpoint.
        http_kwargs["host_origin_protection"] = False

    create_mcp(auth=auth).run(**http_kwargs)


if __name__ == "__main__":
    main()
