from __future__ import annotations

from collections import deque

from nalu.agents.planner.planner import (
    CONVERSATION_TURNS_FOR_PLANNER,
    conversation_snapshot,
    format_conversation,
)


def test_snapshot_returns_empty_when_conversation_none():
    assert conversation_snapshot(None, before_ts=10.0) == []


def test_snapshot_returns_empty_for_empty_conversation():
    assert conversation_snapshot(deque(), before_ts=10.0) == []


def test_snapshot_filters_strictly_before_ts():
    convo = deque([
        {"role": "user", "text": "a", "ts": 1.0},
        {"role": "assistant", "text": "b", "ts": 2.0},
        {"role": "user", "text": "c", "ts": 3.0},
    ])
    out = conversation_snapshot(convo, before_ts=3.0)
    assert [t["text"] for t in out] == ["a", "b"]


def test_snapshot_excludes_concurrent_event_at_exact_ts():
    convo = deque([
        {"role": "user", "text": "older", "ts": 4.99},
        {"role": "user", "text": "current", "ts": 5.0},
    ])
    out = conversation_snapshot(convo, before_ts=5.0)
    assert [t["text"] for t in out] == ["older"]


def test_snapshot_caps_at_max_turns():
    convo = deque([{"role": "user", "text": str(i), "ts": float(i)} for i in range(20)])
    out = conversation_snapshot(convo, before_ts=100.0, max_turns=4)
    assert [t["text"] for t in out] == ["16", "17", "18", "19"]


def test_snapshot_default_cap_matches_constant():
    convo = deque([{"role": "user", "text": str(i), "ts": float(i)} for i in range(20)])
    out = conversation_snapshot(convo, before_ts=100.0)
    assert len(out) == CONVERSATION_TURNS_FOR_PLANNER


def test_snapshot_handles_missing_ts_field():
    convo = deque([
        {"role": "user", "text": "no-ts"},
        {"role": "user", "text": "has-ts", "ts": 1.0},
    ])
    out = conversation_snapshot(convo, before_ts=2.0)
    assert [t["text"] for t in out] == ["no-ts", "has-ts"]


def test_format_empty_conversation_returns_empty_string():
    assert format_conversation([]) == ""


def test_format_labels_user_and_assistant():
    turns = [
        {"role": "user", "text": "open settings"},
        {"role": "assistant", "text": "done"},
    ]
    assert format_conversation(turns) == "User: open settings\nNalu: done"


def test_format_skips_blank_text():
    turns = [
        {"role": "user", "text": "  "},
        {"role": "assistant", "text": "hello"},
        {"role": "user", "text": ""},
    ]
    assert format_conversation(turns) == "Nalu: hello"


def test_format_strips_whitespace():
    turns = [{"role": "user", "text": "  hi  "}]
    assert format_conversation(turns) == "User: hi"


def test_format_treats_unknown_role_as_assistant():
    turns = [{"role": "system", "text": "x"}]
    assert format_conversation(turns) == "Nalu: x"
