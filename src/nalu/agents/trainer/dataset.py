from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
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
    train_path: Path | None = None
    eval_path: Path | None = None
    train_examples: int = 0
    eval_examples: int = 0
    train_runs: list[str] = field(default_factory=list)
    eval_runs: list[str] = field(default_factory=list)


def _read_meta(run: Path) -> dict:
    meta_path = run / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def _run_completed(records: list[dict], meta: dict) -> bool:
    """Decide whether a run is high-quality enough to train on.

    Prefers the planner's stamped `meta.status` (set by the run-outcomes work):
    only `completed` survives. Legacy runs without a status field fall back to
    the older "has a `done` action" heuristic so pre-Phase-5 data still flows.
    """
    status = meta.get("status")
    if status:
        return status == "completed"
    return any(r.get("action") == "done" for r in records)


_KEEP_KINDS = {"click", "double_click", "type", "key", "scroll", "done"}


def _example_from_record(run: Path, goal: str, rec: dict) -> dict | None:
    kind = rec.get("action", "")
    if kind not in _KEEP_KINDS:
        return None
    step = rec.get("step", 0)
    shot = run / f"step_{step:03d}.jpg"
    if not shot.exists():
        return None
    return {
        "run": run.name,
        "step": step,
        "image": str(shot.relative_to(config.ROOT)),
        "goal": goal,
        "action": kind,
        "args": rec.get("args", {}),
        "thought": rec.get("reason", ""),
    }


def collect(
    runs_dir: Path = config.RUNS_DIR,
    out_dir: Path | None = None,
    only_completed: bool = True,
    eval_ratio: float = 0.0,
    seed: int = 1337,
) -> DatasetSummary:
    """Walk past runs and emit a JSONL dataset of (screenshot, goal, action) triples.

    Only emits examples whose action is dispatchable (click/type/key/scroll/done).
    Skips parser errors. By default skips runs that never reached a done action.

    When `eval_ratio > 0`, runs are randomly partitioned (deterministic via `seed`)
    so an entire run's frames go to either train.jsonl or eval.jsonl — never both.
    `dataset.jsonl` always contains the full set.
    """
    out_dir = out_dir or (
        config.ROOT / "training" / "datasets" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.jsonl"

    runs = sorted([p for p in runs_dir.glob("*") if p.is_dir()])
    runs_total = 0
    runs_with_done = 0

    # Pass 1: discover eligible runs + their goal.
    eligible: list[tuple[Path, str, list[dict]]] = []
    for run in runs:
        log_path = run / "actions.jsonl"
        if not log_path.exists():
            continue
        runs_total += 1
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        meta = _read_meta(run)
        completed = _run_completed(records, meta)
        if completed:
            runs_with_done += 1
        if only_completed and not completed:
            continue
        eligible.append((run, meta.get("goal", ""), records))

    # Run-level split (not example-level — frames from the same run share goal,
    # screen, and intent, so example-level shuffling leaks information).
    eval_run_names: set[str] = set()
    if eval_ratio > 0 and len(eligible) >= 2:
        rng = random.Random(seed)
        names = [r.name for r, _, _ in eligible]
        rng.shuffle(names)
        n_eval = max(1, int(round(len(names) * eval_ratio)))
        n_eval = min(n_eval, len(names) - 1)  # keep at least one train run
        eval_run_names = set(names[:n_eval])

    train_path = out_dir / "train.jsonl" if eval_run_names else None
    eval_path = out_dir / "eval.jsonl" if eval_run_names else None

    examples = 0
    actions: dict[str, int] = {}
    train_examples = 0
    eval_examples = 0
    train_runs: list[str] = []
    eval_runs: list[str] = []

    full = out_path.open("w")
    train_f = train_path.open("w") if train_path else None
    eval_f = eval_path.open("w") if eval_path else None
    try:
        for run, goal, records in eligible:
            in_eval = run.name in eval_run_names
            (eval_runs if in_eval else train_runs).append(run.name)
            target_split = eval_f if in_eval else train_f
            for rec in records:
                ex = _example_from_record(run, goal, rec)
                if ex is None:
                    continue
                line = json.dumps(ex) + "\n"
                full.write(line)
                examples += 1
                actions[ex["action"]] = actions.get(ex["action"], 0) + 1
                if target_split is not None:
                    target_split.write(line)
                    if in_eval:
                        eval_examples += 1
                    else:
                        train_examples += 1
    finally:
        full.close()
        if train_f is not None:
            train_f.close()
        if eval_f is not None:
            eval_f.close()

    summary = {
        "out": str(out_path),
        "runs_total": runs_total,
        "runs_with_done": runs_with_done,
        "examples": examples,
        "actions": actions,
        "only_completed": only_completed,
        "eval_ratio": eval_ratio,
        "seed": seed,
        "train_path": str(train_path) if train_path else None,
        "eval_path": str(eval_path) if eval_path else None,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_runs": sorted(train_runs),
        "eval_runs": sorted(eval_runs),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return DatasetSummary(
        out_path=out_path,
        runs_total=runs_total,
        runs_with_done=runs_with_done,
        examples=examples,
        actions=actions,
        train_path=train_path,
        eval_path=eval_path,
        train_examples=train_examples,
        eval_examples=eval_examples,
        train_runs=sorted(train_runs),
        eval_runs=sorted(eval_runs),
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
