"""Per-run event timeline.

The Runs tab can render screenshots step-by-step, but the *interesting* stuff
happens between steps: a `stuck_detected` warning before step 7 means the agent
got hinted; a `completion_denied` after step 12 means the model claimed done
and the verifier pushed back.

`build_run_timeline` joins `runs/<run>/actions.jsonl` with the slice of
`events.jsonl` that falls inside the run's wall-clock window, sorts by ts, and
returns a list of `TimelineEntry`s the dashboard can render. Pure-Python — the
dashboard layer is just rendering.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config


SUCCESS_TOPICS = {"task_completed", "completion_verified"}
WARNING_TOPICS = {
    "stuck_detected",
    "action_no_effect",
    "completion_denied",
    "task_paused",
    "action_refused",
    "action_jittered",
    "task_recovering",
}
FAILURE_TOPICS = {"task_failed"}
ACTION_TOPIC = "action_decided"


@dataclass
class TimelineEntry:
    ts: float
    kind: str  # "event" | "action" | "boundary"
    topic: str
    summary: str
    severity: str  # "info" | "success" | "warning" | "failure"
    payload: dict = field(default_factory=dict)
    step: int | None = None


def _severity(topic: str) -> str:
    if topic in SUCCESS_TOPICS:
        return "success"
    if topic in WARNING_TOPICS:
        return "warning"
    if topic in FAILURE_TOPICS:
        return "failure"
    return "info"


def _summarize_action(rec: dict) -> str:
    kind = rec.get("action") or "?"
    args = rec.get("args") or {}
    if kind == "click":
        return f"click ({args.get('x')}, {args.get('y')})"
    if kind == "double_click":
        return f"double_click ({args.get('x')}, {args.get('y')})"
    if kind == "type":
        text = str(args.get("text", ""))
        if len(text) > 60:
            text = text[:59] + "…"
        return f"type {text!r}"
    if kind == "key":
        mods = "+".join(args.get("modifiers", []) or [])
        name = args.get("name", "")
        return f"key {mods + '+' + name if mods else name}"
    if kind == "scroll":
        return f"scroll dx={args.get('dx', 0)} dy={args.get('dy', 0)}"
    if kind == "wait":
        return f"wait {args.get('ms', 0)}ms"
    if kind == "done":
        return f"done answer={args.get('answer', '')!r}"
    return f"{kind}"


def _summarize_event(topic: str, payload: dict) -> str:
    if topic == "task_started":
        goal = payload.get("goal") or ""
        return f"task started — {goal[:80]}"
    if topic == "task_completed":
        return f"task completed (steps={payload.get('steps')}, answer={payload.get('answer', '')!r})"
    if topic == "task_failed":
        return f"task failed: {payload.get('reason', '')}"
    if topic == "task_paused":
        return f"paused: {payload.get('reason', '')}"
    if topic == "stuck_detected":
        return f"stuck:{payload.get('reason', '')} — agent was hinted"
    if topic == "action_no_effect":
        diff = payload.get("diff", 0)
        return f"prior {payload.get('kind', '')} had no effect (diff={diff:.4f})"
    if topic == "completion_denied":
        return f"completion denied — {payload.get('reasoning', '')}"
    if topic == "completion_verified":
        return f"completion verified — {payload.get('reasoning', '')}"
    return topic


def _read_events_in_window(events_log: Path, started_ts: float, ended_ts: float | None) -> list[dict]:
    if not events_log.exists():
        return []
    out: list[dict] = []
    end = ended_ts if ended_ts is not None else float("inf")
    for line in events_log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = ev.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        if started_ts <= ts <= end + 0.5:
            out.append(ev)
    return out


def build_run_timeline(
    run_dir: Path,
    *,
    events_log: Path | None = None,
) -> list[TimelineEntry]:
    """Return chronological timeline entries for a single run."""
    events_log = events_log or config.EVENTS_LOG
    meta_path = run_dir / "meta.json"
    actions_path = run_dir / "actions.jsonl"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    started_ts = meta.get("started_ts")
    ended_ts = meta.get("ended_ts")
    if not isinstance(started_ts, (int, float)):
        return []

    entries: list[TimelineEntry] = []

    if actions_path.exists():
        for line in actions_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts", started_ts)
            entries.append(
                TimelineEntry(
                    ts=float(ts),
                    kind="action",
                    topic=ACTION_TOPIC,
                    summary=_summarize_action(rec),
                    severity="info",
                    payload=rec,
                    step=rec.get("step"),
                )
            )

    seen_action_ts: set[tuple[float, int | None]] = {(e.ts, e.step) for e in entries}
    for ev in _read_events_in_window(events_log, started_ts, ended_ts):
        topic = ev.get("topic", "")
        ts = float(ev.get("ts", started_ts))
        payload = ev.get("payload") or {}
        if topic == ACTION_TOPIC and (ts, payload.get("step")) in seen_action_ts:
            continue  # already rendered from actions.jsonl
        entries.append(
            TimelineEntry(
                ts=ts,
                kind="event",
                topic=topic,
                summary=_summarize_event(topic, payload),
                severity=_severity(topic),
                payload=payload,
                step=payload.get("step"),
            )
        )

    entries.sort(key=lambda e: e.ts)
    return entries
