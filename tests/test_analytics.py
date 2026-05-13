from __future__ import annotations

import json
from pathlib import Path

from nalu.dashboard.analytics import (
    RunRecord,
    RunsSummary,
    coarse_failure_kind,
    failure_kind_breakdown,
    iter_runs,
    summarize_runs,
)


def _write_run(parent: Path, name: str, **meta) -> Path:
    run = parent / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps(meta))
    return run


def test_iter_runs_skips_dir_without_meta(tmp_path: Path):
    (tmp_path / "20260505_010000").mkdir()
    _write_run(tmp_path, "20260505_020000", goal="g", status="completed", steps=3)
    names = [r.name for r in iter_runs(tmp_path)]
    assert names == ["20260505_020000"]


def test_iter_runs_yields_in_sorted_order(tmp_path: Path):
    _write_run(tmp_path, "20260505_030000", goal="b", status="completed", steps=1)
    _write_run(tmp_path, "20260505_010000", goal="a", status="completed", steps=2)
    names = [r.name for r in iter_runs(tmp_path)]
    assert names == ["20260505_010000", "20260505_030000"]


def test_iter_runs_skips_corrupted_meta(tmp_path: Path):
    bad = tmp_path / "20260505_010000"
    bad.mkdir()
    (bad / "meta.json").write_text("{not valid json")
    _write_run(tmp_path, "20260505_020000", goal="g", status="completed", steps=1)
    names = [r.name for r in iter_runs(tmp_path)]
    assert names == ["20260505_020000"]


def test_summary_counts_completed_and_failed(tmp_path: Path):
    _write_run(tmp_path, "a", status="completed", steps=4, started_ts=1.0)
    _write_run(tmp_path, "b", status="completed", steps=8, started_ts=2.0)
    _write_run(tmp_path, "c", status="failed", reason="timeout", steps=10, started_ts=3.0)
    s = summarize_runs(tmp_path)
    assert s.total == 3
    assert s.completed == 2
    assert s.failed == 1
    assert s.completion_rate == 2 / 3


def test_summary_failure_breakdown_keyed_by_raw_reason(tmp_path: Path):
    _write_run(tmp_path, "a", status="failed", reason="timeout", steps=10)
    _write_run(tmp_path, "b", status="failed", reason="stuck:repeat", steps=5)
    _write_run(tmp_path, "c", status="failed", reason="stuck:repeat", steps=6)
    s = summarize_runs(tmp_path)
    assert s.failure_breakdown == {"timeout": 1, "stuck:repeat": 2}


def test_summary_top_failure(tmp_path: Path):
    _write_run(tmp_path, "a", status="failed", reason="timeout", steps=1)
    _write_run(tmp_path, "b", status="failed", reason="stuck:repeat", steps=1)
    _write_run(tmp_path, "c", status="failed", reason="stuck:repeat", steps=1)
    s = summarize_runs(tmp_path)
    assert s.top_failure == ("stuck:repeat", 2)


def test_summary_top_failure_none_when_no_failures(tmp_path: Path):
    _write_run(tmp_path, "a", status="completed", steps=1)
    s = summarize_runs(tmp_path)
    assert s.top_failure is None


def test_summary_avg_steps_split_by_outcome(tmp_path: Path):
    _write_run(tmp_path, "a", status="completed", steps=4)
    _write_run(tmp_path, "b", status="completed", steps=8)
    _write_run(tmp_path, "c", status="failed", reason="x", steps=12)
    s = summarize_runs(tmp_path)
    assert s.avg_steps_completed == 6.0
    assert s.avg_steps_failed == 12.0


def test_summary_other_status(tmp_path: Path):
    _write_run(tmp_path, "a", status="unknown", steps=0)
    s = summarize_runs(tmp_path)
    assert s.total == 1
    assert s.other == 1
    assert s.completed == 0 and s.failed == 0


def test_summary_filters_by_since_ts(tmp_path: Path):
    _write_run(tmp_path, "old", status="completed", steps=1, started_ts=10.0)
    _write_run(tmp_path, "new", status="completed", steps=1, started_ts=20.0)
    s = summarize_runs(tmp_path, since_ts=15.0)
    assert s.total == 1
    assert s.records[0].name == "new"


def test_summary_completion_rate_zero_when_nothing_finished(tmp_path: Path):
    s = summarize_runs(tmp_path)
    assert s.total == 0
    assert s.completion_rate == 0.0


def test_coarse_failure_kind_collapses_namespaced_reasons():
    assert coarse_failure_kind("stuck:repeat") == "stuck"
    assert coarse_failure_kind("stuck:alternation") == "stuck"
    assert coarse_failure_kind("dispatch: oops") == "dispatch"
    assert coarse_failure_kind("vision: model crashed") == "vision"
    assert coarse_failure_kind("timeout") == "timeout"
    assert coarse_failure_kind("max_steps_exceeded") == "max_steps_exceeded"


def test_coarse_failure_kind_handles_empty():
    assert coarse_failure_kind("") == "unspecified"


def test_failure_kind_breakdown_aggregates_subtypes():
    s = RunsSummary(
        failure_breakdown={
            "stuck:repeat": 3,
            "stuck:alternation": 2,
            "timeout": 1,
            "dispatch: actuator denied": 1,
        }
    )
    out = failure_kind_breakdown(s)
    assert out == {"stuck": 5, "timeout": 1, "dispatch": 1}


def test_run_record_from_meta_handles_missing_fields(tmp_path: Path):
    rec = RunRecord.from_meta(tmp_path / "x", {})
    assert rec.status == "unknown"
    assert rec.steps == 0
    assert rec.started_ts is None


def test_iter_runs_returns_nothing_when_dir_missing(tmp_path: Path):
    out = list(iter_runs(tmp_path / "does_not_exist"))
    assert out == []
