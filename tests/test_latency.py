from __future__ import annotations

import json
from pathlib import Path

import pytest

from nalu.dashboard.latency import build_run_latency


def _write_run(
    run_dir: Path,
    *,
    actions: list[dict] | None = None,
    meta: dict | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if actions is not None:
        (run_dir / "actions.jsonl").write_text(
            "\n".join(json.dumps(a) for a in actions)
        )
    if meta is not None:
        (run_dir / "meta.json").write_text(json.dumps(meta))
    return run_dir


def test_empty_run_returns_empty_latency(tmp_path):
    out = build_run_latency(tmp_path)
    assert out.n_steps == 0
    assert out.total_wall_ms is None
    assert out.median_step_ms is None


def test_single_step_with_ended_ts_uses_meta_for_duration(tmp_path):
    _write_run(
        tmp_path,
        actions=[{"step": 0, "action": "click", "ts": 1000.0}],
        meta={"started_ts": 999.5, "ended_ts": 1001.0},
    )
    out = build_run_latency(tmp_path)
    assert out.n_steps == 1
    assert out.steps[0].duration_ms == pytest.approx(1000.0)  # ended_ts - decided_ts
    assert out.total_wall_ms == pytest.approx(1500.0)  # ended - started
    assert out.median_step_ms == pytest.approx(1000.0)


def test_multi_step_durations_use_consecutive_gaps(tmp_path):
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click", "ts": 100.0},
            {"step": 1, "action": "type", "ts": 102.0},
            {"step": 2, "action": "done", "ts": 105.0},
        ],
        meta={"started_ts": 99.0, "ended_ts": 106.0},
    )
    out = build_run_latency(tmp_path)
    assert [s.duration_ms for s in out.steps] == pytest.approx([2000.0, 3000.0, 1000.0])
    assert out.total_wall_ms == pytest.approx(7000.0)


def test_last_step_duration_is_none_when_ended_ts_missing(tmp_path):
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click", "ts": 100.0},
            {"step": 1, "action": "type", "ts": 102.0},
        ],
        meta={"started_ts": 99.0},
    )
    out = build_run_latency(tmp_path)
    assert out.steps[0].duration_ms == pytest.approx(2000.0)
    assert out.steps[1].duration_ms is None


def test_total_wall_falls_back_to_sum_of_durations_when_meta_missing(tmp_path):
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click", "ts": 100.0},
            {"step": 1, "action": "click", "ts": 101.0},
            {"step": 2, "action": "done", "ts": 105.0},
        ],
        meta={"ended_ts": 106.0},  # missing started_ts → no meta-based wall time
    )
    out = build_run_latency(tmp_path)
    # 1000 + 4000 + 1000 = 6000
    assert out.total_wall_ms == pytest.approx(6000.0)


def test_records_without_ts_are_skipped(tmp_path):
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click"},  # no ts
            {"step": 1, "action": "type", "ts": 102.0},
            {"step": 2, "action": "done", "ts": 103.0},
        ],
        meta={"started_ts": 100.0, "ended_ts": 104.0},
    )
    out = build_run_latency(tmp_path)
    assert out.n_steps == 2
    assert [s.kind for s in out.steps] == ["type", "done"]


def test_corrupted_jsonl_lines_are_skipped(tmp_path):
    (tmp_path / "actions.jsonl").write_text(
        '{"step": 0, "action": "click", "ts": 100.0}\n'
        "not json at all\n"
        '{"step": 1, "action": "done", "ts": 101.0}\n'
    )
    (tmp_path / "meta.json").write_text(json.dumps({"started_ts": 99.5, "ended_ts": 102.0}))
    out = build_run_latency(tmp_path)
    assert out.n_steps == 2


def test_corrupted_meta_json_falls_back_to_durations(tmp_path):
    (tmp_path / "actions.jsonl").write_text(
        '{"step": 0, "action": "click", "ts": 100.0}\n'
        '{"step": 1, "action": "done", "ts": 101.0}\n'
    )
    (tmp_path / "meta.json").write_text("{not valid")
    out = build_run_latency(tmp_path)
    # last step has no successor and no ended_ts → its duration is None,
    # so total_wall falls back to sum of known durations (1000ms).
    assert out.total_wall_ms == pytest.approx(1000.0)


def test_longest_step_picks_max_duration(tmp_path):
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click", "ts": 100.0},
            {"step": 1, "action": "type", "ts": 100.5},  # 9.5s gap before step 2 → step 1 dur=9500
            {"step": 2, "action": "click", "ts": 110.0},
            {"step": 3, "action": "done", "ts": 110.5},
        ],
        meta={"started_ts": 99.5, "ended_ts": 111.0},
    )
    out = build_run_latency(tmp_path)
    assert out.longest_step is not None
    assert out.longest_step.step == 1
    assert out.longest_step.duration_ms == pytest.approx(9500.0)


def test_p95_with_few_samples_returns_max(tmp_path):
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click", "ts": 100.0},
            {"step": 1, "action": "done", "ts": 105.0},
        ],
        meta={"started_ts": 99.0, "ended_ts": 106.0},
    )
    out = build_run_latency(tmp_path)
    # nearest-rank p95 of [1000, 5000] picks index ceil(0.95*2)-1 = 1 → 5000
    assert out.p95_step_ms == pytest.approx(5000.0)


def test_median_with_even_sample_count_averages_middle_two(tmp_path):
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click", "ts": 0.0},
            {"step": 1, "action": "click", "ts": 1.0},  # 1000ms
            {"step": 2, "action": "click", "ts": 4.0},  # 3000ms
            {"step": 3, "action": "done", "ts": 9.0},  # 5000ms (last)
        ],
        meta={"started_ts": -0.5, "ended_ts": 14.0},
    )
    out = build_run_latency(tmp_path)
    # durations: [1000, 3000, 5000, 5000] → median (3000+5000)/2 = 4000
    assert out.median_step_ms == pytest.approx(4000.0)


def test_negative_total_wall_clamped_to_zero_via_meta_fallback(tmp_path):
    # If ended_ts < started_ts (clock skew on serialization), the meta-based
    # total is rejected and we fall back to summed durations.
    _write_run(
        tmp_path,
        actions=[
            {"step": 0, "action": "click", "ts": 100.0},
            {"step": 1, "action": "done", "ts": 101.0},
        ],
        meta={"started_ts": 105.0, "ended_ts": 100.0},
    )
    out = build_run_latency(tmp_path)
    # ended - started is negative → ignore, sum durations: only first step has a successor (1000ms),
    # last step has ended_ts but ended < decided so its duration is clamped to 0 → total 1000.
    assert out.total_wall_ms == pytest.approx(1000.0)
