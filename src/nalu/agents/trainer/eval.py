from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from ... import config


@dataclass
class EvalSummary:
    out_dir: Path
    total: int
    action_correct: int
    click_examples: int
    click_hit_64: int
    click_mae: float
    text_examples: int
    text_correct: int
    adapter_dir: Path | None
    elapsed_s: float


def _action_correct(pred_kind: str, truth_kind: str) -> bool:
    if pred_kind == truth_kind:
        return True
    # double_click is a strict subset; either side counts as "correct kind".
    if {pred_kind, truth_kind} == {"click", "double_click"}:
        return True
    return False


def _click_distance(pred_args: dict, truth_args: dict) -> float | None:
    if "x" not in pred_args or "y" not in pred_args:
        return None
    if "x" not in truth_args or "y" not in truth_args:
        return None
    dx = pred_args["x"] - truth_args["x"]
    dy = pred_args["y"] - truth_args["y"]
    return math.hypot(dx, dy)


def _text_match(pred_args: dict, truth_args: dict) -> bool | None:
    pt = pred_args.get("text") or pred_args.get("answer")
    tt = truth_args.get("text") or truth_args.get("answer")
    if pt is None or tt is None:
        return None
    return str(pt).strip().lower() == str(tt).strip().lower()


def evaluate(
    dataset_path: Path,
    out_dir: Path | None = None,
    limit: int | None = 25,
    hit_threshold_px: int = 64,
) -> EvalSummary:
    """Run the active VisionAgent over `dataset_path` and write per-example results.

    Loads the same VisionAgent the daemon uses, so the active adapter (if any)
    is applied automatically.
    """
    from ..vision import VisionAgent
    from .runner import active_adapter_dir

    out_dir = out_dir or (
        config.ROOT / "training" / "evals" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    dataset_path = Path(dataset_path)
    examples: list[dict] = []
    for line in dataset_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"no examples in {dataset_path}")

    adapter = active_adapter_dir()
    vision = VisionAgent()
    vision.load()

    started = time.time()
    total = 0
    action_correct = 0
    click_examples = 0
    click_hits = 0
    click_distances: list[float] = []
    text_examples = 0
    text_correct = 0

    with results_path.open("w") as out:
        for ex in examples:
            img_path = config.ROOT / ex["image"]
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                im.load()
                action = vision.decide(im, ex.get("goal", ""), history=None)

            truth_kind = ex["action"]
            truth_args = ex.get("args", {}) or {}
            kind_ok = _action_correct(action.kind, truth_kind)

            distance = None
            if truth_kind in ("click", "double_click"):
                distance = _click_distance(action.args, truth_args)
                if distance is not None:
                    click_examples += 1
                    click_distances.append(distance)
                    if distance <= hit_threshold_px:
                        click_hits += 1

            text_ok = None
            if truth_kind in ("type", "done"):
                text_ok = _text_match(action.args, truth_args)
                if text_ok is not None:
                    text_examples += 1
                    if text_ok:
                        text_correct += 1

            total += 1
            if kind_ok:
                action_correct += 1

            out.write(
                json.dumps(
                    {
                        "run": ex.get("run"),
                        "step": ex.get("step"),
                        "goal": ex.get("goal", ""),
                        "truth_kind": truth_kind,
                        "truth_args": truth_args,
                        "pred_kind": action.kind,
                        "pred_args": action.args,
                        "pred_reason": action.reason,
                        "kind_correct": kind_ok,
                        "click_distance_px": distance,
                        "text_correct": text_ok,
                    }
                )
                + "\n"
            )

    elapsed = time.time() - started
    click_mae = sum(click_distances) / len(click_distances) if click_distances else 0.0

    summary = {
        "dataset": str(dataset_path),
        "adapter": str(adapter) if adapter else None,
        "total": total,
        "action_kind_accuracy": action_correct / total if total else 0.0,
        "click_examples": click_examples,
        "click_hit_rate_64px": click_hits / click_examples if click_examples else 0.0,
        "click_mae_px": click_mae,
        "text_examples": text_examples,
        "text_accuracy": text_correct / text_examples if text_examples else 0.0,
        "elapsed_s": elapsed,
        "ts": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return EvalSummary(
        out_dir=out_dir,
        total=total,
        action_correct=action_correct,
        click_examples=click_examples,
        click_hit_64=click_hits,
        click_mae=click_mae,
        text_examples=text_examples,
        text_correct=text_correct,
        adapter_dir=adapter,
        elapsed_s=elapsed,
    )


def _load_results(path: Path) -> dict[tuple[str, int], dict]:
    """Index a results.jsonl by (run, step) for joining."""
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[(rec.get("run") or "", int(rec.get("step", 0)))] = rec
    return out


def compare_evals(baseline_dir: Path, candidate_dir: Path) -> dict:
    """Join two eval runs by (run, step) and report metric deltas + per-action breakdown.

    Returns a dict with:
      - `metrics`: dict of {metric_name: {"baseline": v, "candidate": v, "delta": v}}
      - `per_action`: list of {action, total, baseline_correct, candidate_correct,
        flipped_to_correct, flipped_to_wrong}
      - `examples`: list of joined per-example records (subset of fields)
    """
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    bsum = json.loads((baseline_dir / "summary.json").read_text())
    csum = json.loads((candidate_dir / "summary.json").read_text())
    bres = _load_results(baseline_dir / "results.jsonl")
    cres = _load_results(candidate_dir / "results.jsonl")

    metrics = {}
    for k in (
        "action_kind_accuracy",
        "click_hit_rate_64px",
        "click_mae_px",
        "text_accuracy",
        "elapsed_s",
    ):
        bv = bsum.get(k)
        cv = csum.get(k)
        delta = (cv - bv) if (isinstance(bv, (int, float)) and isinstance(cv, (int, float))) else None
        metrics[k] = {"baseline": bv, "candidate": cv, "delta": delta}

    shared = sorted(bres.keys() & cres.keys())
    per_action_acc: dict[str, dict[str, int]] = {}
    examples = []
    flipped_correct = 0
    flipped_wrong = 0
    both_correct = 0
    both_wrong = 0

    for key in shared:
        b = bres[key]
        c = cres[key]
        truth = b.get("truth_kind") or c.get("truth_kind") or "?"
        slot = per_action_acc.setdefault(
            truth,
            {"total": 0, "baseline_correct": 0, "candidate_correct": 0,
             "flipped_to_correct": 0, "flipped_to_wrong": 0},
        )
        slot["total"] += 1
        b_ok = bool(b.get("kind_correct"))
        c_ok = bool(c.get("kind_correct"))
        if b_ok:
            slot["baseline_correct"] += 1
        if c_ok:
            slot["candidate_correct"] += 1
        if b_ok and not c_ok:
            slot["flipped_to_wrong"] += 1
            flipped_wrong += 1
        elif not b_ok and c_ok:
            slot["flipped_to_correct"] += 1
            flipped_correct += 1
        elif b_ok and c_ok:
            both_correct += 1
        else:
            both_wrong += 1

        examples.append(
            {
                "run": key[0],
                "step": key[1],
                "truth": truth,
                "baseline_pred": b.get("pred_kind"),
                "candidate_pred": c.get("pred_kind"),
                "baseline_correct": b_ok,
                "candidate_correct": c_ok,
                "baseline_dist": b.get("click_distance_px"),
                "candidate_dist": c.get("click_distance_px"),
            }
        )

    return {
        "baseline": {"name": baseline_dir.name, "adapter": bsum.get("adapter")},
        "candidate": {"name": candidate_dir.name, "adapter": csum.get("adapter")},
        "shared_examples": len(shared),
        "metrics": metrics,
        "per_action": [
            {"action": k, **v} for k, v in sorted(per_action_acc.items())
        ],
        "tally": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "flipped_to_correct": flipped_correct,
            "flipped_to_wrong": flipped_wrong,
        },
        "examples": examples,
    }


def list_evals(root: Path | None = None) -> list[dict]:
    root = root or (config.ROOT / "training" / "evals")
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        s = d / "summary.json"
        if s.exists():
            try:
                out.append({"name": d.name, "path": str(d), **json.loads(s.read_text())})
            except json.JSONDecodeError:
                continue
    return out
