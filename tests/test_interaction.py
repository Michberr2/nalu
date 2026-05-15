from __future__ import annotations

import json
from pathlib import Path

import pytest

from nalu.dashboard.interaction import (
    InteractionMetrics,
    compute_interaction_metrics,
)


def _approx_list(values, expected, *, rel=1e-9):
    assert len(values) == len(expected)
    for v, e in zip(values, expected):
        assert v == pytest.approx(e, rel=rel, abs=1e-6), f"{v} != {e}"


def _ev(topic: str, ts: float, **payload):
    return {"topic": topic, "ts": ts, "payload": payload}


def test_empty_events_returns_zero_metrics():
    m = compute_interaction_metrics([])
    assert m.intents_observed == 0
    assert m.queries_observed == 0
    assert m.median_ttfa_ms is None
    assert m.median_ttfr_ms is None
    assert m.median_settle_ms is None
    assert m.proactive_utterances == 0


def test_ttfa_measured_from_user_intent_to_first_action():
    events = [
        _ev("user_intent", 100.0, text="open mail"),
        _ev("action_decided", 100.5),  # 500ms
        _ev("action_decided", 101.0),  # already-consumed intent → only first counts
        _ev("user_intent", 200.0, text="next task"),
        _ev("action_decided", 200.2),  # 200ms
    ]
    m = compute_interaction_metrics(events)
    assert m.intents_observed == 2
    _approx_list(m.ttfa_samples_ms, [500.0, 200.0])
    assert m.median_ttfa_ms == pytest.approx(350.0)


def test_ttfr_measured_from_user_query_to_responder_reply():
    events = [
        _ev("user_query", 50.0, text="what time?"),
        _ev("responder_reply", 50.3, reply="Five."),  # 300ms
        _ev("user_query", 100.0, text="and you?"),
        _ev("responder_reply", 100.8, reply="Fine, thanks."),  # 800ms
    ]
    m = compute_interaction_metrics(events)
    assert m.queries_observed == 2
    _approx_list(m.ttfr_samples_ms, [300.0, 800.0])
    assert m.median_ttfr_ms == pytest.approx(550.0)


def test_settle_metrics_capture_distribution_and_stability():
    events = [
        _ev("screen_settled", 1.0, elapsed_ms=120, stable=True),
        _ev("screen_settled", 2.0, elapsed_ms=80, stable=True),
        _ev("screen_settled", 3.0, elapsed_ms=1500, stable=False),
        _ev("screen_settled", 4.0, elapsed_ms=200, stable=True),
    ]
    m = compute_interaction_metrics(events)
    assert m.settle_samples_ms == [120.0, 80.0, 1500.0, 200.0]
    assert m.median_settle_ms == 160.0
    assert m.p95_settle_ms is not None
    assert m.settle_stable_count == 3
    assert m.settle_capped_count == 1
    assert m.settle_stability_rate == 0.75


def test_inter_step_gap_excludes_first_action_after_intent():
    events = [
        _ev("user_intent", 0.0),
        _ev("action_decided", 1.0),  # TTFA, not a gap
        _ev("action_decided", 1.5),  # gap = 500ms
        _ev("action_decided", 2.7),  # gap = 1200ms
        _ev("user_intent", 10.0),  # resets the chain
        _ev("action_decided", 10.2),  # TTFA, not a gap
        _ev("action_decided", 10.4),  # gap = 200ms
    ]
    m = compute_interaction_metrics(events)
    _approx_list(m.inter_step_gaps_ms, [500.0, 1200.0, 200.0])
    assert m.median_inter_step_gap_ms == pytest.approx(500.0)


def test_proactive_count_uses_default_topic_set():
    events = [
        _ev("task_started", 0.0),
        _ev("subgoal_started", 1.0),
        _ev("action_no_effect", 2.0),
        _ev("task_recovering", 3.0),
        _ev("stuck_detected", 4.0),
        _ev("task_paused", 5.0),
        _ev("task_failed", 6.0),
        _ev("action_decided", 7.0),  # not a proactive trigger
        _ev("planner_ready", 8.0),  # not a proactive trigger
    ]
    m = compute_interaction_metrics(events)
    assert m.proactive_utterances == 7


def test_user_intent_resets_inter_step_chain_between_runs():
    # If we don't reset, the gap between the last action of run A and the first
    # action of run B would be counted as an "inter-step gap" — that's wrong;
    # it's just idle time between tasks.
    events = [
        _ev("user_intent", 0.0),
        _ev("action_decided", 0.1),
        _ev("action_decided", 0.2),
        _ev("user_intent", 60.0),  # 1 minute idle, then a new task
        _ev("action_decided", 60.1),
        _ev("action_decided", 60.2),
    ]
    m = compute_interaction_metrics(events)
    # 2 gaps total, both 100ms — no spurious 59.9s gap.
    _approx_list(m.inter_step_gaps_ms, [100.0, 100.0])


def test_compute_handles_missing_payload_keys():
    events = [
        _ev("screen_settled", 1.0),  # no elapsed_ms
        _ev("screen_settled", 2.0, elapsed_ms="not a number"),  # bad type
        _ev("screen_settled", 3.0, elapsed_ms=100, stable=True),
    ]
    m = compute_interaction_metrics(events)
    assert m.settle_samples_ms == [100.0]
    assert m.settle_stable_count == 1


def test_compute_skips_malformed_events():
    events = [
        {"topic": None, "ts": 1.0},
        {"topic": "user_intent", "ts": "not a number"},
        {"ts": 2.0},  # no topic
        _ev("user_intent", 3.0),
        _ev("action_decided", 3.5),
    ]
    m = compute_interaction_metrics(events)
    assert m.intents_observed == 1
    assert m.ttfa_samples_ms == [500.0]


def test_compute_reads_from_events_log_path(tmp_path: Path):
    log = tmp_path / "events.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(_ev("user_intent", 1.0)),
                json.dumps(_ev("action_decided", 1.25)),
                "",  # blank line
                "not valid json",
                json.dumps(_ev("screen_settled", 1.5, elapsed_ms=42, stable=True)),
            ]
        )
    )
    m = compute_interaction_metrics(events_log=log)
    assert m.ttfa_samples_ms == [250.0]
    assert m.settle_samples_ms == [42.0]


def test_compute_returns_empty_metrics_when_log_missing(tmp_path: Path):
    m = compute_interaction_metrics(events_log=tmp_path / "nope.jsonl")
    assert isinstance(m, InteractionMetrics)
    assert m.intents_observed == 0
