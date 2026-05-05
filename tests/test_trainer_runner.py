from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nalu.agents.trainer.runner import (
    _METRIC_RE,
    _MetricsTee,
    _build_completion,
    _serialize_action,
    activate_adapter,
    active_adapter_dir,
    deactivate_adapter,
)


def test_serialize_click():
    assert _serialize_action("click", {"x": 100, "y": 200}) == "click(x=100, y=200)"


def test_serialize_double_click():
    assert _serialize_action("double_click", {"x": 5, "y": 10}) == "double_click(x=5, y=10)"


def test_serialize_type_escapes_quotes():
    out = _serialize_action("type", {"text": 'hello "world"'})
    assert out == 'type(text="hello \\"world\\"")'


def test_serialize_key_with_modifiers():
    assert _serialize_action("key", {"name": "space", "modifiers": ["cmd"]}) == 'hotkey(keys="cmd+space")'


def test_serialize_key_bare():
    assert _serialize_action("key", {"name": "enter"}) == 'press(key="enter")'


def test_serialize_scroll():
    assert _serialize_action("scroll", {"dx": 0, "dy": -3}) == "scroll(dx=0, dy=-3)"


def test_serialize_done():
    assert _serialize_action("done", {"answer": "42"}) == 'finished(content="42")'


def test_build_completion_with_thought():
    out = _build_completion("look at menu", "click", {"x": 1, "y": 2})
    assert out == "Thought: look at menu\nAction: click(x=1, y=2)"


def test_build_completion_no_thought():
    out = _build_completion("", "done", {"answer": "ok"})
    assert out == 'Action: finished(content="ok")'


def test_metrics_regex_matches_real_format():
    line = (
        "Iter 25: Train loss \x1b[92m1.23456789\x1b[0m, "
        "Learning Rate 2.000e-05, "
        "It/sec 0.500, "
        "Tokens/sec 123.456, "
        "Trained Tokens 1024, "
        "Peak mem 8.500 GB"
    )
    import re
    clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = _METRIC_RE.search(clean)
    assert m
    assert int(m.group(1)) == 25
    assert float(m.group(2)) == 1.23456789
    assert float(m.group(3)) == 2e-5
    assert float(m.group(4)) == 0.5
    assert float(m.group(7)) == 8.5


def test_metrics_tee_writes_jsonl(tmp_path: Path):
    metrics_path = tmp_path / "metrics.jsonl"
    sink = io.StringIO()
    tee = _MetricsTee(sink, metrics_path)
    tee.write(
        "Iter 10: Train loss \x1b[92m0.50000000\x1b[0m, "
        "Learning Rate 1.000e-05, "
        "It/sec 1.000, "
        "Tokens/sec 50.000, "
        "Trained Tokens 200, "
        "Peak mem 4.000 GB\n"
    )
    assert metrics_path.exists()
    rec = json.loads(metrics_path.read_text().strip())
    assert rec["step"] == 10
    assert rec["train_loss"] == 0.5
    assert rec["peak_mem_gb"] == 4.0
    assert "ts" in rec


@pytest.fixture
def isolated_root(tmp_path: Path, monkeypatch):
    from nalu import config as nconfig
    from nalu.agents.trainer import runner as runner_mod

    monkeypatch.setattr(nconfig, "ROOT", tmp_path)
    monkeypatch.setattr(runner_mod.config, "ROOT", tmp_path)
    return tmp_path


def _make_run_dir(root: Path, name: str = "run-1") -> Path:
    d = root / "training" / "runs" / name
    d.mkdir(parents=True)
    (d / "adapters.safetensors").write_bytes(b"\x00")
    (d / "adapter_config.json").write_text('{"rank": 8}')
    return d


def test_activate_adapter_writes_pointer(isolated_root: Path):
    run = _make_run_dir(isolated_root)
    result = activate_adapter(run)
    assert result == run.resolve()
    pointer = isolated_root / "training" / "active_adapter"
    assert pointer.exists()
    assert pointer.read_text().strip() == str(run.resolve())


def test_active_adapter_dir_returns_target(isolated_root: Path):
    run = _make_run_dir(isolated_root)
    activate_adapter(run)
    assert active_adapter_dir() == run.resolve()


def test_active_adapter_dir_none_when_unset(isolated_root: Path):
    assert active_adapter_dir() is None


def test_active_adapter_dir_none_when_files_missing(isolated_root: Path):
    run = _make_run_dir(isolated_root)
    activate_adapter(run)
    (run / "adapters.safetensors").unlink()
    assert active_adapter_dir() is None


def test_deactivate_adapter(isolated_root: Path):
    run = _make_run_dir(isolated_root)
    activate_adapter(run)
    assert deactivate_adapter() is True
    assert active_adapter_dir() is None
    assert deactivate_adapter() is False


def test_activate_rejects_missing_files(isolated_root: Path):
    bad = isolated_root / "empty"
    bad.mkdir()
    with pytest.raises(FileNotFoundError):
        activate_adapter(bad)


def test_metrics_tee_buffers_partial_lines(tmp_path: Path):
    metrics_path = tmp_path / "metrics.jsonl"
    tee = _MetricsTee(io.StringIO(), metrics_path)
    tee.write("Iter 1: Train loss 0.10000000, Learning Rate 1.000e-05, ")
    assert not metrics_path.exists()
    tee.write("It/sec 1.000, Tokens/sec 10.000, Trained Tokens 5, Peak mem 1.000 GB\n")
    assert metrics_path.exists()
    rec = json.loads(metrics_path.read_text().strip())
    assert rec["step"] == 1
