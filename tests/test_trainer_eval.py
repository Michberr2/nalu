from __future__ import annotations

import json
from pathlib import Path

from nalu.agents.trainer.eval import (
    _action_correct,
    _click_distance,
    _text_match,
    compare_evals,
)


def test_action_kind_match():
    assert _action_correct("click", "click")
    assert _action_correct("type", "type")


def test_action_click_double_click_equivalence():
    assert _action_correct("click", "double_click")
    assert _action_correct("double_click", "click")


def test_action_kind_mismatch():
    assert not _action_correct("click", "type")
    assert not _action_correct("scroll", "key")


def test_click_distance_basic():
    assert _click_distance({"x": 0, "y": 0}, {"x": 3, "y": 4}) == 5.0


def test_click_distance_missing_pred():
    assert _click_distance({}, {"x": 1, "y": 2}) is None


def test_click_distance_missing_truth():
    assert _click_distance({"x": 1, "y": 2}, {}) is None


def test_text_match_exact():
    assert _text_match({"text": "Hello"}, {"text": "hello"}) is True


def test_text_match_strips_whitespace():
    assert _text_match({"text": "  hi  "}, {"text": "hi"}) is True


def test_text_match_done_answer_field():
    assert _text_match({"answer": "42"}, {"answer": "42"}) is True


def test_text_match_mismatch():
    assert _text_match({"text": "foo"}, {"text": "bar"}) is False


def test_text_match_none_when_field_absent():
    assert _text_match({}, {"text": "foo"}) is None


def _write_eval(dir_: Path, summary: dict, results: list[dict]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "summary.json").write_text(json.dumps(summary))
    with (dir_ / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return dir_


def test_compare_evals_reports_metric_deltas(tmp_path: Path):
    base = _write_eval(
        tmp_path / "base",
        {
            "action_kind_accuracy": 0.5,
            "click_hit_rate_64px": 0.4,
            "click_mae_px": 80.0,
            "text_accuracy": 0.6,
            "elapsed_s": 12.0,
            "adapter": None,
        },
        [
            {"run": "a", "step": 0, "truth_kind": "click", "kind_correct": True, "pred_kind": "click", "click_distance_px": 30},
            {"run": "a", "step": 1, "truth_kind": "click", "kind_correct": False, "pred_kind": "type", "click_distance_px": None},
        ],
    )
    cand = _write_eval(
        tmp_path / "cand",
        {
            "action_kind_accuracy": 0.75,
            "click_hit_rate_64px": 0.6,
            "click_mae_px": 50.0,
            "text_accuracy": 0.8,
            "elapsed_s": 11.0,
            "adapter": "/path/to/adapter",
        },
        [
            {"run": "a", "step": 0, "truth_kind": "click", "kind_correct": True, "pred_kind": "click", "click_distance_px": 20},
            {"run": "a", "step": 1, "truth_kind": "click", "kind_correct": True, "pred_kind": "click", "click_distance_px": 40},
        ],
    )

    cmp = compare_evals(base, cand)
    assert cmp["baseline"]["adapter"] is None
    assert cmp["candidate"]["adapter"] == "/path/to/adapter"
    assert cmp["metrics"]["action_kind_accuracy"]["delta"] == 0.25
    assert cmp["metrics"]["click_mae_px"]["delta"] == -30.0
    assert cmp["shared_examples"] == 2


def test_compare_evals_per_action_breakdown_and_flips(tmp_path: Path):
    base = _write_eval(
        tmp_path / "base",
        {"action_kind_accuracy": 0.5},
        [
            {"run": "a", "step": 0, "truth_kind": "click", "kind_correct": False, "pred_kind": "type"},
            {"run": "a", "step": 1, "truth_kind": "click", "kind_correct": True, "pred_kind": "click"},
            {"run": "a", "step": 2, "truth_kind": "type", "kind_correct": True, "pred_kind": "type"},
            {"run": "a", "step": 3, "truth_kind": "type", "kind_correct": False, "pred_kind": "click"},
        ],
    )
    cand = _write_eval(
        tmp_path / "cand",
        {"action_kind_accuracy": 0.5},
        [
            {"run": "a", "step": 0, "truth_kind": "click", "kind_correct": True, "pred_kind": "click"},
            {"run": "a", "step": 1, "truth_kind": "click", "kind_correct": False, "pred_kind": "type"},
            {"run": "a", "step": 2, "truth_kind": "type", "kind_correct": True, "pred_kind": "type"},
            {"run": "a", "step": 3, "truth_kind": "type", "kind_correct": True, "pred_kind": "type"},
        ],
    )

    cmp = compare_evals(base, cand)
    assert cmp["tally"]["flipped_to_correct"] == 2  # step 0 click, step 3 type
    assert cmp["tally"]["flipped_to_wrong"] == 1   # step 1 click
    assert cmp["tally"]["both_correct"] == 1       # step 2 type

    by_action = {row["action"]: row for row in cmp["per_action"]}
    assert by_action["click"]["flipped_to_correct"] == 1
    assert by_action["click"]["flipped_to_wrong"] == 1
    assert by_action["type"]["flipped_to_correct"] == 1
    assert by_action["type"]["flipped_to_wrong"] == 0


def test_compare_evals_handles_disjoint_examples(tmp_path: Path):
    base = _write_eval(
        tmp_path / "base",
        {"action_kind_accuracy": 1.0},
        [{"run": "a", "step": 0, "truth_kind": "click", "kind_correct": True, "pred_kind": "click"}],
    )
    cand = _write_eval(
        tmp_path / "cand",
        {"action_kind_accuracy": 1.0},
        [{"run": "b", "step": 0, "truth_kind": "click", "kind_correct": True, "pred_kind": "click"}],
    )
    cmp = compare_evals(base, cand)
    assert cmp["shared_examples"] == 0
    assert cmp["per_action"] == []
