from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ... import config


@dataclass
class TrainingRecommendation:
    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    suggested_data: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class TrainerAgent:
    """Reads real run logs and emits recommendations. No actual training step yet —
    the training pipeline lands in a later phase. This is the eval-feedback half."""

    def __init__(self, runs_dir: Path = config.RUNS_DIR):
        self.runs_dir = runs_dir

    def collect_metrics(self, last_n: int = 20) -> dict:
        runs = sorted([p for p in self.runs_dir.glob("*") if p.is_dir()], reverse=True)[:last_n]
        if not runs:
            return {"runs": 0}

        completed = 0
        failed = 0
        action_counts: dict[str, int] = {}
        steps = []
        failures: list[str] = []

        for run in runs:
            log_path = run / "actions.jsonl"
            if not log_path.exists():
                continue
            saw_done = False
            run_steps = 0
            with log_path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    run_steps += 1
                    kind = rec.get("action", "unknown")
                    action_counts[kind] = action_counts.get(kind, 0) + 1
                    if kind == "done":
                        saw_done = True
                    if kind == "error":
                        failures.append(rec.get("reason", ""))
            steps.append(run_steps)
            if saw_done:
                completed += 1
            else:
                failed += 1

        n = completed + failed
        return {
            "runs": n,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / n if n else 0.0,
            "avg_steps": sum(steps) / len(steps) if steps else 0,
            "action_counts": action_counts,
            "recent_failures": failures[-5:],
        }

    def recommend(self) -> TrainingRecommendation:
        m = self.collect_metrics()
        rec = TrainingRecommendation(should_retrain=False, metrics=m)

        if m.get("runs", 0) < 5:
            rec.reasons.append(f"only {m.get('runs', 0)} runs logged — collect more before evaluating retraining")
            return rec

        sr = m.get("success_rate", 0.0)
        if sr < 0.6:
            rec.should_retrain = True
            rec.reasons.append(f"success rate {sr:.0%} below 60% threshold over last {m['runs']} runs")
            rec.suggested_data.append("collect 200+ failed-task screenshots labeled with the correct action")

        err_count = m.get("action_counts", {}).get("error", 0)
        if err_count > 0.2 * sum(m.get("action_counts", {}).values() or [1]):
            rec.should_retrain = True
            rec.reasons.append(f"{err_count} parser/output errors — model is producing malformed JSON too often")
            rec.suggested_data.append("fine-tune on schema-conformant action JSON examples")

        if m.get("avg_steps", 0) > 15:
            rec.reasons.append(f"avg {m['avg_steps']:.1f} steps per task is high — consider planning data")
            rec.suggested_data.append("collect multi-step trajectories with clear sub-goals")

        return rec
