"""Aggregate run outcomes for the dashboard.

Each completed planner run writes its verdict back to `runs/<ts>/meta.json`:
`status` (completed/failed/unknown), `reason` (timeout / max_steps_exceeded /
stuck:repeat / stuck:alternation / parse:... / dispatch:... / vision:...),
`steps`, optional `answer`. This module reads those and produces a small
summary that powers the dashboard's failure-mode panel.

Pure-Python and side-effect-free — `summarize_runs` takes a directory path and
a clock, so tests can exercise the cutoff logic without touching real time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .. import config


VALID_STATUSES = ("completed", "failed", "paused", "unknown")


@dataclass
class RunRecord:
    name: str
    goal: str
    status: str
    reason: str
    steps: int
    started_ts: float | None
    ended_ts: float | None
    answer: str = ""

    @classmethod
    def from_meta(cls, run_dir: Path, meta: dict) -> "RunRecord":
        return cls(
            name=run_dir.name,
            goal=str(meta.get("goal", "")),
            status=str(meta.get("status", "unknown")),
            reason=str(meta.get("reason", "")),
            steps=int(meta.get("steps", 0) or 0),
            started_ts=meta.get("started_ts"),
            ended_ts=meta.get("ended_ts"),
            answer=str(meta.get("answer", "") or ""),
        )


@dataclass
class RunsSummary:
    total: int = 0
    completed: int = 0
    failed: int = 0
    other: int = 0
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    avg_steps_completed: float = 0.0
    avg_steps_failed: float = 0.0
    records: list[RunRecord] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        finished = self.completed + self.failed
        if finished == 0:
            return 0.0
        return self.completed / finished

    @property
    def top_failure(self) -> tuple[str, int] | None:
        if not self.failure_breakdown:
            return None
        return max(self.failure_breakdown.items(), key=lambda kv: kv[1])


def _read_meta(run_dir: Path) -> dict | None:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def iter_runs(runs_dir: Path) -> Iterable[RunRecord]:
    if not runs_dir.exists():
        return
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        meta = _read_meta(run)
        if meta is None:
            continue
        yield RunRecord.from_meta(run, meta)


def summarize_runs(
    runs_dir: Path | None = None,
    *,
    since_ts: float | None = None,
) -> RunsSummary:
    """Aggregate run verdicts under `runs_dir` (defaults to config.RUNS_DIR).

    `since_ts` filters by `started_ts >= since_ts` when set; useful for
    "last 24h" panels.
    """
    runs_dir = runs_dir or config.RUNS_DIR
    summary = RunsSummary()
    completed_steps: list[int] = []
    failed_steps: list[int] = []

    for rec in iter_runs(runs_dir):
        if since_ts is not None and (rec.started_ts is None or rec.started_ts < since_ts):
            continue
        summary.records.append(rec)
        summary.total += 1
        if rec.status == "completed":
            summary.completed += 1
            completed_steps.append(rec.steps)
        elif rec.status == "failed":
            summary.failed += 1
            failed_steps.append(rec.steps)
            bucket = rec.reason.strip() or "unspecified"
            summary.failure_breakdown[bucket] = summary.failure_breakdown.get(bucket, 0) + 1
        else:
            summary.other += 1

    if completed_steps:
        summary.avg_steps_completed = sum(completed_steps) / len(completed_steps)
    if failed_steps:
        summary.avg_steps_failed = sum(failed_steps) / len(failed_steps)
    return summary


def coarse_failure_kind(reason: str) -> str:
    """Bucket a raw reason string into a stable category for charting.

    `stuck:repeat` and `stuck:alternation` fold into `stuck`. `dispatch: ...`
    folds into `dispatch`. Unrecognized reasons return as-is.
    """
    if not reason:
        return "unspecified"
    if ":" in reason:
        head = reason.split(":", 1)[0].strip()
        if head:
            return head
    return reason


def failure_kind_breakdown(summary: RunsSummary) -> dict[str, int]:
    out: dict[str, int] = {}
    for reason, count in summary.failure_breakdown.items():
        kind = coarse_failure_kind(reason)
        out[kind] = out.get(kind, 0) + count
    return out


def now_minus(seconds: float, *, clock: Callable[[], float] | None = None) -> float:
    import time

    return (clock or time.time)() - seconds
