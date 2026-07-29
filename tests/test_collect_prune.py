"""Corpus pruning (no hardware).

The property that matters most is that pruning is *reversible by default*:
several of the reasons are recorder defects rather than bad demonstrations,
and a demonstration costs minutes of arm time to replace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt4_pi.collect import prune

# A pose the arm actually reaches, so IK succeeds and the episode is judged
# on its labels rather than rejected as unreachable.
BASE = {"x": 200.0, "y": 0.0, "z": 200.0, "j4": 0.0, "grip": 0.1}


def write_episode(root: Path, name: str, states: list[dict], *, success: bool = True,
                  started_at: float | None = 1_000_000.0):
    d = root / "episodes" / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "frames" / "000000.jpg").write_bytes(b"x" * 32)
    meta = {
        "success": success, "verdict": "placed" if success else "lost",
        "pick": {"color": "red"}, "place": {"kind": "to_marker"},
    }
    if started_at is not None:
        meta["started_at"] = started_at
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "ticks.jsonl").write_text(
        "".join(json.dumps({"t": i * 0.1, "frame": "frames/000000.jpg", "state": s}) + "\n"
                for i, s in enumerate(states)),
        encoding="utf-8")
    return d


def smooth(n: int = 6) -> list[dict]:
    return [{**BASE, "x": 200.0 + 2.0 * i} for i in range(n)]


def teleport() -> list[dict]:
    """Two poses on opposite sides of the base: a huge J1 step in one tick."""
    return [{**BASE, "x": 60.0, "y": -230.0}, {**BASE, "x": 60.0, "y": 230.0},
            {**BASE, "x": 60.0, "y": 232.0}]


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_smooth_successful_episode_is_usable(tmp_path):
    d = write_episode(tmp_path, "ep_000000", smooth())
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False)
    assert v.reason is None and v.usable and v.bucket == "keep"


def test_single_tick_teleport_is_rejected(tmp_path):
    d = write_episode(tmp_path, "ep_000000", teleport())
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False)
    assert v.reason == "label_jump"
    assert v.detail["worst_jump_deg"] > 15.0


def test_failed_episode_is_kept_but_not_counted_usable(tmp_path):
    """success=False never reaches a dataset, so it must not inflate the
    'usable' figure that collection targets are planned against."""
    d = write_episode(tmp_path, "ep_000000", smooth(), success=False)
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False)
    assert v.reason is None and not v.usable and v.bucket == "keep_failed"


def test_failed_episode_is_rejected_only_when_asked(tmp_path):
    d = write_episode(tmp_path, "ep_000000", smooth(), success=False)
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=True, measure_bytes=False)
    assert v.reason == "failed"


def test_a_failed_episode_is_never_judged_on_its_labels(tmp_path):
    """It is dropped for being failed, not for a jump -- the reason string
    is what a human reads when deciding whether to restore it."""
    d = write_episode(tmp_path, "ep_000000", teleport(), success=False)
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=True, measure_bytes=False)
    assert v.reason == "failed"


def test_partial_directory_and_stub_episode(tmp_path):
    (tmp_path / "episodes" / "ep_000000" / "frames").mkdir(parents=True)
    v = prune.judge(tmp_path / "episodes" / "ep_000000", max_jump_deg=15.0,
                    drop_failed=False, measure_bytes=False)
    assert v.reason == "no_meta"

    d = write_episode(tmp_path, "ep_000001", [dict(BASE)])
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False)
    assert v.reason == "short"


def test_episodes_from_a_known_bad_recorder_are_rejected(tmp_path):
    """Smooth labels are not proof of correct ones: after the leg-weighting
    fix, 26 stack episodes still passed a 15 deg jump filter despite coming
    from the same broken recorder."""
    d = write_episode(tmp_path, "ep_000000", smooth(), started_at=1_000_000.0)
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False,
                    recorded_after=2_000_000.0)
    assert v.reason == "stale_recorder"


def test_episodes_after_the_cutoff_survive(tmp_path):
    d = write_episode(tmp_path, "ep_000000", smooth(), started_at=3_000_000.0)
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False,
                    recorded_after=2_000_000.0)
    assert v.reason is None and v.usable


def test_an_undated_episode_counts_as_stale(tmp_path):
    """No started_at means it predates the field; do not give it the
    benefit of the doubt when the caller asked for a cutoff."""
    d = write_episode(tmp_path, "ep_000000", smooth(), started_at=None)
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False,
                    recorded_after=2_000_000.0)
    assert v.reason == "stale_recorder"


def test_no_cutoff_means_the_age_check_is_inert(tmp_path):
    d = write_episode(tmp_path, "ep_000000", smooth(), started_at=None)
    v = prune.judge(d, max_jump_deg=15.0, drop_failed=False, measure_bytes=False)
    assert v.reason is None


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------


def test_dry_run_touches_nothing(tmp_path):
    d = write_episode(tmp_path, "ep_000000", teleport())
    prune.prune([tmp_path], dry_run=True)
    assert d.is_dir()
    assert not (tmp_path / prune.REJECTED_DIRNAME).exists()
    assert not (tmp_path / prune.REPORT_NAME).exists()


def test_quarantine_moves_rather_than_deletes(tmp_path):
    write_episode(tmp_path, "ep_000000", smooth())
    write_episode(tmp_path, "ep_000001", teleport())
    prune.prune([tmp_path], dry_run=False)

    assert (tmp_path / "episodes" / "ep_000000").is_dir(), "kept episode moved"
    assert not (tmp_path / "episodes" / "ep_000001").exists()
    moved = tmp_path / prune.REJECTED_DIRNAME / "label_jump" / "ep_000001"
    assert moved.is_dir()
    assert (moved / "ticks.jsonl").is_file(), "quarantine must preserve contents"


def test_report_records_a_reason_per_rejected_episode(tmp_path):
    write_episode(tmp_path, "ep_000000", teleport())
    prune.prune([tmp_path], dry_run=False)
    report = json.loads((tmp_path / prune.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["n_usable"] == 0
    assert report["rejected"][0]["episode"] == "ep_000000"
    assert report["rejected"][0]["reason"] == "label_jump"


def test_delete_removes_only_the_rejected(tmp_path):
    write_episode(tmp_path, "ep_000000", smooth())
    write_episode(tmp_path, "ep_000001", teleport())
    prune.prune([tmp_path], dry_run=False, delete=True)
    assert (tmp_path / "episodes" / "ep_000000").is_dir()
    assert not (tmp_path / "episodes" / "ep_000001").exists()
    assert not (tmp_path / prune.REJECTED_DIRNAME).exists()


def test_threshold_is_honoured(tmp_path):
    write_episode(tmp_path, "ep_000000", teleport())
    prune.prune([tmp_path], max_jump_deg=1000.0, dry_run=False)
    assert (tmp_path / "episodes" / "ep_000000").is_dir()


def test_missing_episodes_dir_is_not_an_error(tmp_path):
    prune.prune([tmp_path / "absent"], dry_run=True)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
