from __future__ import annotations

import json
from pathlib import Path

import pytest

from nalu import config as nconfig
from nalu.agents.trainer.dataset import collect


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(nconfig, "ROOT", tmp_path)
    monkeypatch.setattr(nconfig, "RUNS_DIR", runs_dir)
    from nalu.agents.trainer import dataset as ds

    monkeypatch.setattr(ds.config, "ROOT", tmp_path)
    monkeypatch.setattr(ds.config, "RUNS_DIR", runs_dir)
    return tmp_path, runs_dir


def _make_run(
    runs_dir: Path,
    name: str,
    n_steps: int = 3,
    with_done: bool = True,
    status: str | None = None,
) -> Path:
    run = runs_dir / name
    run.mkdir()
    meta: dict = {"goal": f"goal-{name}"}
    if status is not None:
        meta["status"] = status
    (run / "meta.json").write_text(json.dumps(meta))
    actions: list[dict] = []
    for i in range(n_steps):
        (run / f"step_{i:03d}.jpg").write_bytes(b"x")
        actions.append({"step": i, "action": "click", "args": {"x": i, "y": i}, "reason": "go"})
    if with_done:
        last = n_steps
        (run / f"step_{last:03d}.jpg").write_bytes(b"x")
        actions.append({"step": last, "action": "done", "args": {"answer": name}})
    (run / "actions.jsonl").write_text("\n".join(json.dumps(a) for a in actions))
    return run


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_collect_no_split(fake_root):
    _, runs_dir = fake_root
    for n in ["a", "b", "c"]:
        _make_run(runs_dir, n)

    summary = collect(runs_dir=runs_dir)
    assert summary.train_path is None
    assert summary.eval_path is None
    assert summary.examples > 0
    assert summary.train_examples == 0
    assert summary.eval_examples == 0


def test_collect_split_partitions_by_run(fake_root):
    _, runs_dir = fake_root
    for n in ["a", "b", "c", "d"]:
        _make_run(runs_dir, n)

    summary = collect(runs_dir=runs_dir, eval_ratio=0.25, seed=42)
    assert summary.train_path is not None
    assert summary.eval_path is not None

    train_runs = {r["run"] for r in _read_jsonl(summary.train_path)}
    eval_runs = {r["run"] for r in _read_jsonl(summary.eval_path)}
    # Run-level split: no overlap between train and eval runs.
    assert not (train_runs & eval_runs)
    assert train_runs | eval_runs == {"a", "b", "c", "d"}


def test_collect_split_is_deterministic(fake_root):
    _, runs_dir = fake_root
    for n in ["a", "b", "c", "d", "e"]:
        _make_run(runs_dir, n)

    s1 = collect(runs_dir=runs_dir, eval_ratio=0.4, seed=7)
    s2 = collect(runs_dir=runs_dir, eval_ratio=0.4, seed=7)
    assert s1.eval_runs == s2.eval_runs
    assert s1.train_runs == s2.train_runs


def test_collect_split_keeps_at_least_one_train_run(fake_root):
    _, runs_dir = fake_root
    _make_run(runs_dir, "only-train")
    _make_run(runs_dir, "only-eval")

    summary = collect(runs_dir=runs_dir, eval_ratio=0.99, seed=0)
    assert len(summary.train_runs) >= 1
    assert len(summary.eval_runs) >= 1


def test_collect_split_skipped_with_one_run(fake_root):
    _, runs_dir = fake_root
    _make_run(runs_dir, "lonely")

    summary = collect(runs_dir=runs_dir, eval_ratio=0.5)
    # Can't split a single run — fall back to no split.
    assert summary.train_path is None
    assert summary.eval_path is None


def test_collect_skips_failed_runs_by_default(fake_root):
    _, runs_dir = fake_root
    _make_run(runs_dir, "good", with_done=True)
    _make_run(runs_dir, "bad", with_done=False)

    summary = collect(runs_dir=runs_dir)
    runs_in_dataset = {r["run"] for r in _read_jsonl(summary.out_path)}
    assert runs_in_dataset == {"good"}


def test_collect_skips_runs_with_failed_status_even_if_done_emitted(fake_root):
    _, runs_dir = fake_root
    # `done` was emitted but verifier denied → planner stamped status="failed".
    # Old heuristic would keep it; the meta-aware check must drop it.
    _make_run(runs_dir, "denied", with_done=True, status="failed")
    _make_run(runs_dir, "good", with_done=True, status="completed")

    summary = collect(runs_dir=runs_dir)
    runs_in_dataset = {r["run"] for r in _read_jsonl(summary.out_path)}
    assert runs_in_dataset == {"good"}


def test_collect_prefers_meta_status_over_done_heuristic(fake_root):
    _, runs_dir = fake_root
    # No done action, but planner stamped status="completed" anyway (e.g. answer
    # came back via verify_completion path). Trust the stamped status.
    _make_run(runs_dir, "stamped", with_done=False, status="completed")

    summary = collect(runs_dir=runs_dir)
    runs_in_dataset = {r["run"] for r in _read_jsonl(summary.out_path)}
    assert runs_in_dataset == {"stamped"}


def test_collect_legacy_runs_without_status_use_done_heuristic(fake_root):
    _, runs_dir = fake_root
    # Legacy run (pre-Phase-5): no status field. Falls back to has-done check.
    _make_run(runs_dir, "legacy_good", with_done=True)
    _make_run(runs_dir, "legacy_bad", with_done=False)

    summary = collect(runs_dir=runs_dir)
    runs_in_dataset = {r["run"] for r in _read_jsonl(summary.out_path)}
    assert runs_in_dataset == {"legacy_good"}


def test_collect_only_completed_false_keeps_failed_runs(fake_root):
    _, runs_dir = fake_root
    _make_run(runs_dir, "denied", with_done=True, status="failed")
    _make_run(runs_dir, "good", with_done=True, status="completed")

    summary = collect(runs_dir=runs_dir, only_completed=False)
    runs_in_dataset = {r["run"] for r in _read_jsonl(summary.out_path)}
    assert runs_in_dataset == {"denied", "good"}


def test_collect_full_dataset_jsonl_unaffected_by_split(fake_root):
    _, runs_dir = fake_root
    for n in ["a", "b", "c"]:
        _make_run(runs_dir, n, n_steps=2)

    summary = collect(runs_dir=runs_dir, eval_ratio=0.34, seed=1)
    full = _read_jsonl(summary.out_path)
    train = _read_jsonl(summary.train_path)
    eval_ = _read_jsonl(summary.eval_path)
    assert len(full) == len(train) + len(eval_)
