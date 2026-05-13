from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nalu.agents.trainer.merger import (
    MergeConfig,
    MergeRunner,
    MergeSource,
    list_merges,
    parse_sources,
    write_config,
)


def test_parse_sources_plain():
    out = parse_sources(["repo/a", "repo/b"])
    assert [(s.model, s.weight, s.density) for s in out] == [
        ("repo/a", 0.5, None),
        ("repo/b", 0.5, None),
    ]


def test_parse_sources_with_weight():
    out = parse_sources(["repo/a@0.7", "repo/b@0.3"])
    assert [(s.model, s.weight) for s in out] == [("repo/a", 0.7), ("repo/b", 0.3)]


def test_parse_sources_with_density():
    out = parse_sources(["repo/a@0.6:0.5"])
    s = out[0]
    assert s.weight == 0.6 and s.density == 0.5


def test_validate_requires_two_sources():
    cfg = MergeConfig(sources=[MergeSource(model="repo/a")])
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_unknown_method():
    cfg = MergeConfig(
        sources=[MergeSource(model="a"), MergeSource(model="b")],
        merge_method="bogus",
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_ties_requires_base():
    cfg = MergeConfig(
        sources=[MergeSource(model="a"), MergeSource(model="b")],
        merge_method="ties",
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_linear_requires_positive_total_weight():
    cfg = MergeConfig(
        sources=[MergeSource(model="a", weight=0.0), MergeSource(model="b", weight=0.0)],
        merge_method="linear",
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_to_yaml_includes_density_only_when_set():
    cfg = MergeConfig(
        sources=[
            MergeSource(model="a", weight=0.5),
            MergeSource(model="b", weight=0.5, density=0.7),
        ],
    )
    out = cfg.to_yaml()
    assert "parameters" in out["models"][0]
    assert "density" not in out["models"][0]["parameters"]
    assert out["models"][1]["parameters"]["density"] == 0.7


def test_write_config_round_trip(tmp_path: Path):
    cfg = MergeConfig(
        sources=[MergeSource(model="repo/a"), MergeSource(model="repo/b")],
        merge_method="slerp",
        dtype="float16",
    )
    p = write_config(cfg, tmp_path / "merge.yaml")
    parsed = yaml.safe_load(p.read_text())
    assert parsed["merge_method"] == "slerp"
    assert parsed["dtype"] == "float16"
    assert [m["model"] for m in parsed["models"]] == ["repo/a", "repo/b"]


def test_runner_invokes_mergekit_then_mlx_convert(tmp_path: Path, monkeypatch):
    from nalu.agents.trainer import merger

    calls: list[list[str]] = []

    def fake_check(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_run(cmd, log_path):
        calls.append(cmd)
        log_path.write_text("ok\n")

    monkeypatch.setattr(merger, "_check_tool", fake_check)

    cfg = MergeConfig(
        sources=[MergeSource(model="repo/a"), MergeSource(model="repo/b")],
        merge_method="linear",
    )
    runner = MergeRunner(cfg, out_dir=tmp_path / "merge")
    summary = runner.run(runner=fake_run)

    assert calls[0][0] == "mergekit-yaml"
    assert calls[1][0] == "mlx_vlm.convert"
    assert summary.mlx_dir is not None
    assert (summary.out_dir / "summary.json").exists()
    written = json.loads((summary.out_dir / "summary.json").read_text())
    assert written["method"] == "linear"
    assert written["sources"] == ["repo/a", "repo/b"]


def test_runner_skips_quantize_when_disabled(tmp_path: Path, monkeypatch):
    from nalu.agents.trainer import merger

    calls: list[list[str]] = []
    monkeypatch.setattr(merger, "_check_tool", lambda name: f"/usr/bin/{name}")
    cfg = MergeConfig(
        sources=[MergeSource(model="a"), MergeSource(model="b")],
        merge_method="linear",
    )
    runner = MergeRunner(cfg, out_dir=tmp_path / "merge", quantize=False)
    summary = runner.run(
        runner=lambda cmd, log_path: (calls.append(cmd), log_path.write_text("ok\n"))[1]
    )
    assert summary.mlx_dir is None
    assert all(c[0] != "mlx_vlm.convert" for c in calls)


def test_runner_registers_when_id_provided(tmp_path: Path, monkeypatch):
    from nalu import config
    from nalu.agents.trainer import merger
    from nalu.agents.vision import registry

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    monkeypatch.setattr(merger, "_check_tool", lambda name: f"/usr/bin/{name}")

    cfg = MergeConfig(
        sources=[MergeSource(model="a"), MergeSource(model="b")],
        merge_method="linear",
    )
    runner = MergeRunner(
        cfg,
        out_dir=tmp_path / "merge",
        register_as="nalu-merged-test",
        register_label="Test Merge",
    )
    summary = runner.run(runner=lambda cmd, log_path: log_path.write_text("ok\n"))

    assert summary.registered_id == "nalu-merged-test"
    entry = registry.get_model("nalu-merged-test")
    assert entry is not None
    assert entry.kind == "merged"
    assert entry.label == "Test Merge"
    assert Path(entry.path) == summary.mlx_dir


def test_list_merges_returns_recent_first(tmp_path: Path, monkeypatch):
    from nalu.agents.trainer import merger

    monkeypatch.setattr(merger, "MERGES_ROOT", tmp_path)

    for name, method in [("20260101-000001", "linear"), ("20260101-000002", "slerp")]:
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({
            "out_dir": str(d), "method": method, "sources": ["a", "b"],
        }))

    rows = list_merges()
    assert [r["method"] for r in rows] == ["slerp", "linear"]


def test_runner_propagates_subprocess_failure(tmp_path: Path, monkeypatch):
    from nalu.agents.trainer import merger

    monkeypatch.setattr(merger, "_check_tool", lambda name: f"/usr/bin/{name}")

    def boom(cmd, log_path):
        log_path.write_text("BANG\n")
        raise RuntimeError("command failed (exit 1): " + " ".join(cmd))

    cfg = MergeConfig(
        sources=[MergeSource(model="a"), MergeSource(model="b")],
        merge_method="linear",
    )
    runner = MergeRunner(cfg, out_dir=tmp_path / "merge")
    with pytest.raises(RuntimeError, match="command failed"):
        runner.run(runner=boom)
