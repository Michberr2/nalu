from __future__ import annotations

import io
import json
from pathlib import Path

from nalu.agents.trainer.runner import (
    _METRIC_RE,
    _MetricsTee,
    _build_completion,
    _serialize_action,
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


def test_metrics_tee_buffers_partial_lines(tmp_path: Path):
    metrics_path = tmp_path / "metrics.jsonl"
    tee = _MetricsTee(io.StringIO(), metrics_path)
    tee.write("Iter 1: Train loss 0.10000000, Learning Rate 1.000e-05, ")
    assert not metrics_path.exists()
    tee.write("It/sec 1.000, Tokens/sec 10.000, Trained Tokens 5, Peak mem 1.000 GB\n")
    assert metrics_path.exists()
    rec = json.loads(metrics_path.read_text().strip())
    assert rec["step"] == 1
