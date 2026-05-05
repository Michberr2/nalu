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
