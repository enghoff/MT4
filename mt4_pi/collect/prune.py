"""Quarantine unusable episodes out of a raw collection corpus.

    python -m mt4_pi.collect.prune --dry-run
    python -m mt4_pi.collect.prune --root data/pi_demos --root data/pi_stack_demos

Rejected episodes are MOVED to ``<root>/rejected/<reason>/`` by default, not
deleted: a demonstration costs minutes of arm time to replace, several of the
reasons below are recorder defects rather than bad demonstrations, and a
rename inside one volume is instant and reversible. ``--delete`` removes them
outright once you have read the report.

This deliberately duplicates the criteria that `mt4_pi.act.export_episodes`
and `pack_lerobot` already apply at conversion time. Those filters keep bad
data out of one dataset; this one takes it out of the corpus, so the counts
you plan collection against ("do I have enough to_slot blue?") describe usable
episodes rather than recorded ones.

Reasons, and what each really means
-----------------------------------
``no_meta`` / ``short`` / ``ik_failed``
    Structurally unusable: a directory abandoned mid-cycle by a hard kill,
    an episode with no transition to label, or a recorded pose that no IK
    branch reaches. Never recoverable.

``label_jump``
    ``success=True``, the arm did the right thing, but the *labels* claim a
    single-tick joint move larger than `mt4_pi.safety` would ever execute.
    See `pack_lerobot.episode_max_jump_deg` for the mechanism. This is a
    recorder defect, not a demonstration defect -- which is exactly why the
    default is to move rather than delete: if the recorder is fixed and the
    raw waypoint log is ever re-derivable, these frames come back.

``failed``
    ``success=False``. Not corrupt, just not a demonstration of the
    behaviour we want imitated. Off by default (``--drop-failed``) because
    these are the only negative examples in the corpus.

``stale_recorder``
    Recorded before ``--recorded-after``, i.e. by a recorder version now
    known to have mis-timed its labels. Distinct from ``label_jump``, which
    catches only the *gross* violations: after the 2026-07-29 leg-weighting
    fix, 26 stack episodes still passed a 15 deg jump filter despite having
    been produced by the same broken code -- their routes simply happened
    not to put a big wrist swing on a short leg. "Survived the tripwire" is
    not the same as "correct", and there is no way to tell the two apart
    per-episode, because the raw waypoint log is not stored. Use this when
    fresh data exists to replace them.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mt4_jog.kinematics import (
    HOME_J1_DEG,
    HOME_J2_DEG,
    HOME_J3_DEG,
    HOME_J4_DEG,
    JointAnglesDeg,
    ik_position,
)

DEFAULT_ROOTS = (Path("data/pi_demos"), Path("data/pi_stack_demos"))

#: Largest single-tick joint delta (deg) a label may claim. `mt4_pi.safety`
#: caps an executed step at 5 deg; 15 leaves room for the coarsest legitimate
#: transit tick (the clean shuffle corpus tops out at 17.1 deg, so this is
#: deliberately just under that -- see --max-jump-deg to loosen it).
DEFAULT_MAX_JUMP_DEG = 15.0

REJECTED_DIRNAME = "rejected"
REPORT_NAME = "prune_report.json"


@dataclass
class Verdict:
    ep_dir: Path
    reason: str | None
    detail: dict = field(default_factory=dict)
    bytes: int = 0
    #: Survives the prune AND would reach a dataset. False for a retained
    #: success=False episode, which stays on disk but is dropped at export.
    usable: bool = True

    @property
    def bucket(self) -> str:
        if self.reason is not None:
            return self.reason
        return "keep" if self.usable else "keep_failed"


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def worst_joint_jump_deg(ticks: list[dict]) -> float | None:
    """Largest single-tick joint discontinuity, or None when IK fails.

    Computed from the raw ticks rather than an export so the corpus can be
    cleaned before anything has been converted. Mirrors
    `mt4_pi.act.export_episodes.solve_joint_track`: seed each solve with the
    previous result so the arm stays on one IK branch, and take j4 straight
    from the log (it is already the commanded world-frame wrist angle).
    """
    near = JointAnglesDeg(HOME_J1_DEG, HOME_J2_DEG, HOME_J3_DEG, HOME_J4_DEG)
    worst = 0.0
    prev: JointAnglesDeg | None = None
    for tick in ticks:
        s = tick["state"]
        sol = ik_position(s["x"], s["y"], s["z"], near=near, hold_orientation=False)
        if sol is None:
            return None
        q = JointAnglesDeg(sol.j1, sol.j2, sol.j3, s["j4"])
        if prev is not None:
            worst = max(
                worst,
                abs(q.j1 - prev.j1), abs(q.j2 - prev.j2),
                abs(q.j3 - prev.j3), abs(q.j4 - prev.j4),
            )
        prev = q
        near = q
    return worst


def judge(
    ep_dir: Path,
    *,
    max_jump_deg: float,
    drop_failed: bool,
    measure_bytes: bool,
    recorded_after: float | None = None,
) -> Verdict:
    size = dir_size(ep_dir) if measure_bytes else 0
    meta_path = ep_dir / "meta.json"
    ticks_path = ep_dir / "ticks.jsonl"
    if not meta_path.is_file() or not ticks_path.is_file():
        return Verdict(ep_dir, "no_meta", {}, size)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ticks = [
            json.loads(line)
            for line in ticks_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (json.JSONDecodeError, OSError) as exc:
        return Verdict(ep_dir, "no_meta", {"error": str(exc)}, size)

    kind = str(meta.get("place", {}).get("kind", "?"))
    color = str(meta.get("pick", {}).get("color", "?"))
    base = {"kind": kind, "color": color, "verdict": meta.get("verdict")}

    # Before the structural checks: an episode from a recorder known to
    # mis-time its labels is untrustworthy whatever its jump metric says,
    # and reporting it as "stale_recorder" rather than "keep" is the point.
    started = meta.get("started_at")
    if recorded_after is not None and (started is None or float(started) < recorded_after):
        return Verdict(ep_dir, "stale_recorder",
                       {**base, "started_at": started}, size)

    if len(ticks) < 2:
        return Verdict(ep_dir, "short", {**base, "n_ticks": len(ticks)}, size)
    if not meta.get("success"):
        # Kept-but-unused when --drop-failed is off. Bucketed separately from
        # a clean keep so the headline count is not read as "usable": these
        # never reach a dataset, because export drops success=False.
        return Verdict(ep_dir, "failed" if drop_failed else None, base, size,
                       usable=False)

    jump = worst_joint_jump_deg(ticks)
    if jump is None:
        return Verdict(ep_dir, "ik_failed", base, size)
    if jump > max_jump_deg:
        return Verdict(ep_dir, "label_jump", {**base, "worst_jump_deg": round(jump, 1)}, size)
    return Verdict(ep_dir, None, {**base, "worst_jump_deg": round(jump, 1)}, size)


def prune(
    roots: list[Path],
    *,
    max_jump_deg: float = DEFAULT_MAX_JUMP_DEG,
    drop_failed: bool = False,
    dry_run: bool = True,
    delete: bool = False,
    recorded_after: float | None = None,
) -> dict:
    report: dict = {
        "max_jump_deg": max_jump_deg,
        "drop_failed": drop_failed,
        "recorded_after": recorded_after,
        "action": "report" if dry_run else ("delete" if delete else "quarantine"),
        "roots": {},
    }
    for root in roots:
        episodes_dir = root / "episodes"
        if not episodes_dir.is_dir():
            print(f"{root}: no episodes/ directory, skipping")
            continue
        eps = sorted(p for p in episodes_dir.glob("ep_*") if p.is_dir())
        verdicts = [
            judge(p, max_jump_deg=max_jump_deg, drop_failed=drop_failed,
                  measure_bytes=True, recorded_after=recorded_after)
            for p in eps
        ]
        by_reason: Counter[str] = Counter()
        bytes_by_reason: Counter[str] = Counter()
        moved = []
        for v in verdicts:
            by_reason[v.bucket] += 1
            bytes_by_reason[v.bucket] += v.bytes
            if v.reason is None:
                continue
            moved.append({"episode": v.ep_dir.name, "reason": v.reason, **v.detail})
            if dry_run:
                continue
            if delete:
                shutil.rmtree(v.ep_dir, ignore_errors=True)
            else:
                dest = root / REJECTED_DIRNAME / v.reason / v.ep_dir.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(str(v.ep_dir), str(dest))

        usable = by_reason["keep"]
        print(f"\n{root}: {len(verdicts)} episodes -> {usable} usable, {len(moved)} rejected")
        labels = {
            "keep": "usable (reaches a dataset)",
            "keep_failed": "kept, success=False (never exported)",
        }
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            mb = bytes_by_reason[reason] / 1e6
            print(f"    {reason:<12} {n:>4}   {mb:>8.0f} MB   {labels.get(reason, 'rejected')}")
        report["roots"][str(root)] = {
            "n_episodes": len(verdicts),
            "n_usable": usable,
            "counts": dict(by_reason),
            "bytes": dict(bytes_by_reason),
            "rejected": moved,
        }
        if not dry_run:
            (root / REPORT_NAME).write_text(json.dumps(report["roots"][str(root)], indent=2),
                                            encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", action="append", dest="roots", default=None,
                    help="collection root; repeat (default: data/pi_demos, data/pi_stack_demos)")
    ap.add_argument("--max-jump-deg", type=float, default=DEFAULT_MAX_JUMP_DEG,
                    help=f"reject labels claiming a bigger single-tick joint move "
                         f"(default {DEFAULT_MAX_JUMP_DEG})")
    ap.add_argument("--drop-failed", action="store_true",
                    help="also reject success=False episodes (the only negative examples)")
    ap.add_argument("--recorded-after", default=None, metavar="ISO8601",
                    help="reject episodes recorded before this local time (e.g. "
                         "2026-07-29T17:00), i.e. by a recorder version known to "
                         "mis-time labels; only meaningful once replacement data exists")
    ap.add_argument("--dry-run", action="store_true", help="report only, touch nothing")
    ap.add_argument("--delete", action="store_true",
                    help="delete instead of moving to <root>/rejected/ -- irreversible")
    args = ap.parse_args(argv)

    roots = [Path(r) for r in args.roots] if args.roots else [p for p in DEFAULT_ROOTS if p.exists()]
    if not roots:
        raise SystemExit("no collection roots found; pass --root")
    if args.delete and args.dry_run:
        raise SystemExit("--delete and --dry-run are contradictory")
    cutoff = None
    if args.recorded_after:
        from datetime import datetime

        try:
            cutoff = datetime.fromisoformat(args.recorded_after).timestamp()
        except ValueError as exc:
            raise SystemExit(f"--recorded-after must be ISO8601: {exc}") from exc
    prune(roots, max_jump_deg=args.max_jump_deg, drop_failed=args.drop_failed,
          dry_run=args.dry_run, delete=args.delete, recorded_after=cutoff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
