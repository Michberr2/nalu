"""Per-run latency profile from `actions.jsonl` + `meta.json`.

Each `action_decided` record carries a `ts`. Step duration = gap to the next
record's ts. The last step's duration is `meta.ended_ts - last_decided_ts`
when `ended_ts` is available.

We deliberately don't try to split "vision time" vs "dispatch time" — the
planner doesn't emit those boundaries cleanly enough. What you do get is
per-step wall-clock + aggregate stats (median, p95, total), which is the
question users actually ask: "where is the time going across this run?"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StepTiming:
    step: int
    kind: str
    decided_ts: float
    duration_ms: float | None  # None for the last step when meta.ended_ts is missing


@dataclass
class RunLatency:
    steps: list[StepTiming] = field(default_factory=list)
    total_wall_ms: float | None = None
    median_step_ms: float | None = None
    p95_step_ms: float | None = None
    longest_step: StepTiming | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Caller passes already-sorted values."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = max(0, min(len(sorted_values) - 1, int(round((pct / 100.0) * len(sorted_values))) - 1))
    return sorted_values[k]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def build_run_latency(run_dir: Path) -> RunLatency:
    """Compute per-step + aggregate timings for a single planner run."""
    actions_path = run_dir / "actions.jsonl"
    meta_path = run_dir / "meta.json"
    if not actions_path.exists():
        return RunLatency()

    records: list[dict] = []
    for line in actions_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        return RunLatency()

    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    ended_ts = meta.get("ended_ts")
    started_ts = meta.get("started_ts")

    steps: list[StepTiming] = []
    for i, rec in enumerate(records):
        decided_ts = rec.get("ts")
        if decided_ts is None:
            continue
        if i + 1 < len(records):
            next_ts = records[i + 1].get("ts")
            duration_ms = (next_ts - decided_ts) * 1000.0 if next_ts is not None else None
        elif ended_ts is not None:
            duration_ms = max(0.0, (ended_ts - decided_ts) * 1000.0)
        else:
            duration_ms = None
        steps.append(
            StepTiming(
                step=int(rec.get("step", i)),
                kind=str(rec.get("action", "")),
                decided_ts=float(decided_ts),
                duration_ms=duration_ms,
            )
        )

    durations = [s.duration_ms for s in steps if s.duration_ms is not None]
    total_wall_ms = None
    if started_ts is not None and ended_ts is not None and ended_ts >= started_ts:
        total_wall_ms = (ended_ts - started_ts) * 1000.0
    elif durations:
        total_wall_ms = sum(durations)

    longest = None
    if durations:
        longest = max(
            (s for s in steps if s.duration_ms is not None),
            key=lambda s: s.duration_ms,
        )

    return RunLatency(
        steps=steps,
        total_wall_ms=total_wall_ms,
        median_step_ms=_median(durations) if durations else None,
        p95_step_ms=_percentile(sorted(durations), 95.0) if durations else None,
        longest_step=longest,
    )
