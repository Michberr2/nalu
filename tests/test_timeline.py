from __future__ import annotations

import json
from pathlib import Path

from nalu.dashboard.timeline import (
    ACTION_TOPIC,
    TimelineEntry,
    build_run_timeline,
)


def _write_meta(run: Path, **fields) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps(fields))


def _write_actions(run: Path, records: list[dict]) -> None:
    (run / "actions.jsonl").write_text("\n".join(json.dumps(r) for r in records))


def _write_events(events_log: Path, events: list[dict]) -> None:
    events_log.parent.mkdir(parents=True, exist_ok=True)
    events_log.write_text("\n".join(json.dumps(e) for e in events))


def test_returns_empty_when_meta_missing(tmp_path: Path):
    assert build_run_timeline(tmp_path / "missing", events_log=tmp_path / "ev") == []


def test_returns_empty_when_meta_lacks_started_ts(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", status="completed")
    assert build_run_timeline(run, events_log=tmp_path / "ev") == []


def test_includes_actions_from_actions_jsonl(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=100.0, ended_ts=200.0, status="completed")
    _write_actions(run, [
        {"step": 0, "action": "click", "args": {"x": 10, "y": 20}, "reason": "r", "ts": 110.0},
        {"step": 1, "action": "type", "args": {"text": "hi"}, "reason": "r", "ts": 120.0},
    ])
    out = build_run_timeline(run, events_log=tmp_path / "ev")
    assert [e.kind for e in out] == ["action", "action"]
    assert "click" in out[0].summary
    assert "'hi'" in out[1].summary


def test_includes_events_in_run_window(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=100.0, ended_ts=200.0, status="completed")
    events = tmp_path / "ev.jsonl"
    _write_events(events, [
        {"topic": "task_started", "payload": {"goal": "g"}, "ts": 100.0, "source": "planner"},
        {"topic": "stuck_detected", "payload": {"reason": "repeat", "step": 3}, "ts": 130.0, "source": "planner"},
        {"topic": "task_completed", "payload": {"answer": "ok", "steps": 5}, "ts": 195.0, "source": "planner"},
    ])
    out = build_run_timeline(run, events_log=events)
    topics = [e.topic for e in out]
    assert "task_started" in topics
    assert "stuck_detected" in topics
    assert "task_completed" in topics


def test_excludes_events_outside_window(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=100.0, ended_ts=200.0, status="completed")
    events = tmp_path / "ev.jsonl"
    _write_events(events, [
        {"topic": "task_started", "payload": {}, "ts": 50.0, "source": "p"},
        {"topic": "stuck_detected", "payload": {"reason": "repeat"}, "ts": 150.0, "source": "p"},
        {"topic": "task_completed", "payload": {"steps": 5}, "ts": 9999.0, "source": "p"},
    ])
    out = build_run_timeline(run, events_log=events)
    topics = [e.topic for e in out]
    assert "task_started" not in topics  # before started_ts
    assert "task_completed" not in topics  # after ended_ts
    assert "stuck_detected" in topics


def test_includes_events_when_ended_ts_missing(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=100.0, status="unknown")
    events = tmp_path / "ev.jsonl"
    _write_events(events, [
        {"topic": "stuck_detected", "payload": {"reason": "x"}, "ts": 99999.0, "source": "p"},
    ])
    out = build_run_timeline(run, events_log=events)
    assert any(e.topic == "stuck_detected" for e in out)


def test_dedups_action_decided_already_in_actions_jsonl(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=100.0, ended_ts=200.0, status="completed")
    _write_actions(run, [
        {"step": 0, "action": "click", "args": {"x": 1, "y": 2}, "reason": "r", "ts": 150.0},
    ])
    events = tmp_path / "ev.jsonl"
    _write_events(events, [
        {"topic": ACTION_TOPIC, "payload": {"step": 0, "action": "click"}, "ts": 150.0, "source": "p"},
    ])
    out = build_run_timeline(run, events_log=events)
    action_count = sum(1 for e in out if e.topic == ACTION_TOPIC)
    assert action_count == 1


def test_entries_sorted_by_ts(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=100.0, ended_ts=200.0, status="completed")
    _write_actions(run, [
        {"step": 1, "action": "type", "args": {"text": "b"}, "reason": "", "ts": 130.0},
        {"step": 0, "action": "click", "args": {"x": 1, "y": 2}, "reason": "", "ts": 110.0},
    ])
    events = tmp_path / "ev.jsonl"
    _write_events(events, [
        {"topic": "task_started", "payload": {"goal": "g"}, "ts": 100.5, "source": "p"},
        {"topic": "task_completed", "payload": {"steps": 2}, "ts": 195.0, "source": "p"},
    ])
    out = build_run_timeline(run, events_log=events)
    timestamps = [e.ts for e in out]
    assert timestamps == sorted(timestamps)


def test_severity_classification():
    run_meta = {"goal": "g", "started_ts": 0.0, "ended_ts": 100.0, "status": "completed"}
    # we'll just exercise build with synthetic data and check severity per topic
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        run = td_path / "run"
        _write_meta(run, **run_meta)
        events = td_path / "ev.jsonl"
        _write_events(events, [
            {"topic": "task_started", "payload": {}, "ts": 1.0, "source": "p"},
            {"topic": "stuck_detected", "payload": {"reason": "x"}, "ts": 2.0, "source": "p"},
            {"topic": "completion_verified", "payload": {"reasoning": "yes"}, "ts": 3.0, "source": "p"},
            {"topic": "task_failed", "payload": {"reason": "boom"}, "ts": 4.0, "source": "p"},
        ])
        out = build_run_timeline(run, events_log=events)
        sev = {e.topic: e.severity for e in out}
        assert sev["task_started"] == "info"
        assert sev["stuck_detected"] == "warning"
        assert sev["completion_verified"] == "success"
        assert sev["task_failed"] == "failure"


def test_handles_corrupted_events_log_gracefully(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=100.0, ended_ts=200.0, status="completed")
    events = tmp_path / "ev.jsonl"
    events.write_text(
        '{"topic": "task_started", "payload": {}, "ts": 110.0, "source": "p"}\n'
        '{not valid json\n'
        '{"topic": "task_completed", "payload": {"steps": 1}, "ts": 190.0, "source": "p"}\n'
    )
    out = build_run_timeline(run, events_log=events)
    topics = [e.topic for e in out]
    assert "task_started" in topics
    assert "task_completed" in topics


def test_summarize_truncates_long_typed_text(tmp_path: Path):
    run = tmp_path / "run"
    _write_meta(run, goal="g", started_ts=0.0, ended_ts=100.0, status="completed")
    long_text = "x" * 200
    _write_actions(run, [
        {"step": 0, "action": "type", "args": {"text": long_text}, "reason": "", "ts": 1.0},
    ])
    out = build_run_timeline(run, events_log=tmp_path / "ev")
    assert out[0].summary.endswith("…'") or "…" in out[0].summary


def test_meta_corruption_returns_empty(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text("{not valid")
    assert build_run_timeline(run, events_log=tmp_path / "ev") == []
