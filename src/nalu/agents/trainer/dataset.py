from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ... import config


@dataclass
class DatasetSummary:
    out_path: Path
    runs_total: int
    runs_with_done: int
    examples: int
    actions: dict[str, int]


def _run_completed(records: list[dict]) -> bool:
    return any(r.get("action") == "done" for r in records)


def collect(
    runs_dir: Path = config.RUNS_DIR,
    out_dir: Path | None = None,
    only_completed: bool = True,
) -> DatasetSummary:
    """Walk past runs and emit a JSONL dataset of (screenshot, goal, action) triples.

    Only emits examples whose action is dispatchable (click/type/key/scroll/done).
    Skips parser errors. By default skips runs that never reached a done action.
    """
    out_dir = out_dir or (config.ROOT / "training" / "datasets" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.jsonl"

    runs = sorted([p for p in runs_dir.glob("*") if p.is_dir()])
    runs_total = 0
    runs_with_done = 0
    examples = 0
    actions: dict[str, int] = {}

    keep_kinds = {"click", "double_click", "type", "key", "scroll", "done"}

    with out_path.open("w") as out:
        for run in runs:
            log_path = run / "actions.jsonl"
            if not log_path.exists():
                continue
            runs_total += 1
            records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            if not records:
                continue
            completed = _run_completed(records)
            if completed:
                runs_with_done += 1
            if only_completed and not completed:
                continue

            meta_path = run / "meta.json"
            goal = ""
            if meta_path.exists():
                try:
                    goal = json.loads(meta_path.read_text()).get("goal", "")
                except json.JSONDecodeError:
                    pass
            for rec in records:
                kind = rec.get("action", "")
                if kind not in keep_kinds:
                    continue
                step = rec.get("step", 0)
                shot = run / f"step_{step:03d}.jpg"
                if not shot.exists():
                    continue
                example = {
                    "run": run.name,
                    "step": step,
                    "image": str(shot.relative_to(config.ROOT)),
                    "goal": goal,
                    "action": kind,
                    "args": rec.get("args", {}),
                    "thought": rec.get("reason", ""),
                }
                out.write(json.dumps(example) + "\n")
                examples += 1
                actions[kind] = actions.get(kind, 0) + 1

    summary_path = out_dir / "summary.json"
    summary = {
        "out": str(out_path),
        "runs_total": runs_total,
        "runs_with_done": runs_with_done,
        "examples": examples,
        "actions": actions,
        "only_completed": only_completed,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return DatasetSummary(
        out_path=out_path,
        runs_total=runs_total,
        runs_with_done=runs_with_done,
        examples=examples,
        actions=actions,
    )


def list_datasets(root: Path | None = None) -> list[dict]:
    root = root or (config.ROOT / "training" / "datasets")
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        s = d / "summary.json"
        if s.exists():
            out.append({"name": d.name, "path": str(d), **json.loads(s.read_text())})
    return out
