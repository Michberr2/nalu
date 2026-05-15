"""Interaction-quality metrics for the Nalu dashboard.

These complement `latency.py` (per-step wall clock) with metrics specific to
the interactive surface — how fast Nalu *feels*, not just how fast it runs:

  * **TTFA** — Time to First Action: gap from `user_intent` to the first
    `action_decided`. Captures "did the agent start working right away?"
  * **TTFR** — Time to First Response: gap from `user_query` to the matching
    `responder_reply`. Captures conversational latency, which is judged on a
    different scale than task completion.
  * **median_settle_ms / p95_settle_ms** — distribution of `screen_settled`
    elapsed times. Tells you how long the screen takes to stabilize after
    actions, the cost we replaced the 400ms fixed sleep with.
  * **inter_step_gap_ms** — median + p95 gap between consecutive
    `action_decided` events. The "perceive→act tick rate" — lower is tighter.
  * **proactive_utterances** — count of `proactive_ready`-gated speech events
    (TTS playback is a side effect; we use `screen_settled`'s counter as a
    proxy by counting `proactive_*` topics if/when emitted; today we count
    distinct proactive-speak triggers).

Pure-Python, reads `EVENTS_LOG` only — no Streamlit dependency here so the
metrics can be unit-tested and reused in CLI / programmatic eval scripts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config


@dataclass
class InteractionMetrics:
    intents_observed: int = 0
    queries_observed: int = 0
    ttfa_samples_ms: list[float] = field(default_factory=list)
    ttfr_samples_ms: list[float] = field(default_factory=list)
    settle_samples_ms: list[float] = field(default_factory=list)
    inter_step_gaps_ms: list[float] = field(default_factory=list)
    proactive_utterances: int = 0
    settle_stable_count: int = 0
    settle_capped_count: int = 0

    @property
    def median_ttfa_ms(self) -> float | None:
        return _median(self.ttfa_samples_ms) if self.ttfa_samples_ms else None

    @property
    def median_ttfr_ms(self) -> float | None:
        return _median(self.ttfr_samples_ms) if self.ttfr_samples_ms else None

    @property
    def median_settle_ms(self) -> float | None:
        return _median(self.settle_samples_ms) if self.settle_samples_ms else None

    @property
    def p95_settle_ms(self) -> float | None:
        return _percentile(sorted(self.settle_samples_ms), 95.0) if self.settle_samples_ms else None

    @property
    def median_inter_step_gap_ms(self) -> float | None:
        return _median(self.inter_step_gaps_ms) if self.inter_step_gaps_ms else None

    @property
    def p95_inter_step_gap_ms(self) -> float | None:
        return (
            _percentile(sorted(self.inter_step_gaps_ms), 95.0)
            if self.inter_step_gaps_ms
            else None
        )

    @property
    def settle_stability_rate(self) -> float | None:
        total = self.settle_stable_count + self.settle_capped_count
        if total == 0:
            return None
        return self.settle_stable_count / total


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = max(0, min(len(sorted_values) - 1, int(round((pct / 100.0) * len(sorted_values))) - 1))
    return sorted_values[k]


def _load_events(events_log: Path) -> list[dict]:
    if not events_log.exists():
        return []
    out: list[dict] = []
    for line in events_log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def compute_interaction_metrics(
    events: list[dict] | None = None,
    *,
    events_log: Path | None = None,
) -> InteractionMetrics:
    """Compute TTFA / TTFR / settle / proactive metrics from the events log.

    Either pass `events` directly (tests, in-memory replay) or let the loader
    read `config.EVENTS_LOG`. Each event is a dict with `topic`, `ts`, and
    `payload` — matching the on-disk JSONL format.
    """
    if events is None:
        events = _load_events(events_log or config.EVENTS_LOG)

    m = InteractionMetrics()
    pending_intent_ts: float | None = None
    pending_query_ts: float | None = None
    last_action_ts: float | None = None

    for ev in events:
        topic = ev.get("topic")
        ts = ev.get("ts")
        if topic is None or not isinstance(ts, (int, float)):
            continue
        payload = ev.get("payload") or {}

        if topic == "user_intent":
            m.intents_observed += 1
            pending_intent_ts = ts
            last_action_ts = None  # new task starts a fresh tick chain
        elif topic == "user_query":
            m.queries_observed += 1
            pending_query_ts = ts
        elif topic == "action_decided":
            if pending_intent_ts is not None:
                m.ttfa_samples_ms.append((ts - pending_intent_ts) * 1000.0)
                pending_intent_ts = None
            if last_action_ts is not None:
                m.inter_step_gaps_ms.append((ts - last_action_ts) * 1000.0)
            last_action_ts = ts
        elif topic == "responder_reply":
            if pending_query_ts is not None:
                m.ttfr_samples_ms.append((ts - pending_query_ts) * 1000.0)
                pending_query_ts = None
        elif topic == "screen_settled":
            elapsed_ms = payload.get("elapsed_ms")
            if isinstance(elapsed_ms, (int, float)):
                m.settle_samples_ms.append(float(elapsed_ms))
            if payload.get("stable") is True:
                m.settle_stable_count += 1
            elif payload.get("stable") is False:
                m.settle_capped_count += 1
        elif topic == "proactive_ready":
            # `proactive_ready` fires once at startup. Speech events themselves
            # aren't on the bus — they're a side effect of subscribed topics.
            # We count those topics as a proxy below.
            pass

    # Proactive utterances: events the ProactiveSpeaker subscribes to, scoped to
    # this run window. We use the same default topic set as ProactiveSpeaker.
    m.proactive_utterances = _count_proactive_triggers(events)
    return m


_PROACTIVE_TOPICS = (
    "task_started",
    "subgoal_started",
    "action_no_effect",
    "task_recovering",
    "stuck_detected",
    "task_paused",
    "task_failed",
)


def _count_proactive_triggers(events: list[dict]) -> int:
    """Count events that would have triggered a proactive utterance.

    This is a *would-have* count, not a definitive "did the user hear it" —
    cooldown silencing and the env gate are runtime decisions we don't replay
    here. The dashboard label clarifies the framing.
    """
    return sum(1 for ev in events if ev.get("topic") in _PROACTIVE_TOPICS)


__all__ = [
    "InteractionMetrics",
    "compute_interaction_metrics",
]
