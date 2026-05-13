from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def reg(tmp_path: Path, monkeypatch):
    """Force the registry to live under a per-test tmp dir."""
    from nalu import config
    from nalu.agents.vision import registry

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.json")
    return registry


def test_first_load_seeds_default(reg):
    models = reg.list_models()
    assert len(models) == 1
    assert models[0].id == reg.DEFAULT_MODEL_ID
    assert reg.active_model_id() == reg.DEFAULT_MODEL_ID
    assert reg.REGISTRY_PATH.exists()


def test_register_and_get(reg):
    reg.register_model("os-atlas-7b", "mlx-community/OS-Atlas-Base-7B-4bit", label="OS-Atlas 7B")
    entry = reg.get_model("os-atlas-7b")
    assert entry is not None
    assert entry.path == "mlx-community/OS-Atlas-Base-7B-4bit"
    assert entry.label == "OS-Atlas 7B"


def test_register_replaces_existing(reg):
    reg.register_model("foo", "old/path")
    reg.register_model("foo", "new/path", label="updated")
    entries = [m for m in reg.list_models() if m.id == "foo"]
    assert len(entries) == 1
    assert entries[0].path == "new/path"
    assert entries[0].label == "updated"


def test_register_rejects_invalid_id(reg):
    with pytest.raises(ValueError):
        reg.register_model("Bad ID", "x")
    with pytest.raises(ValueError):
        reg.register_model("UPPER", "x")
    with pytest.raises(ValueError):
        reg.register_model("", "x")


def test_register_rejects_empty_path(reg):
    with pytest.raises(ValueError):
        reg.register_model("ok-id", "")


def test_set_active_changes_pointer(reg):
    reg.register_model("alt", "alt/path")
    reg.set_active("alt")
    assert reg.active_model_id() == "alt"
    assert reg.active_model().path == "alt/path"


def test_set_active_unknown_raises(reg):
    with pytest.raises(FileNotFoundError):
        reg.set_active("nope")


def test_unregister_drops_entry(reg):
    reg.register_model("alt", "alt/path")
    assert reg.unregister_model("alt") is True
    assert reg.get_model("alt") is None


def test_unregister_returns_false_when_missing(reg):
    assert reg.unregister_model("nope") is False


def test_unregister_refuses_active(reg):
    with pytest.raises(ValueError):
        reg.unregister_model(reg.DEFAULT_MODEL_ID)


def test_resolve_returns_active_path_when_no_id(reg):
    reg.register_model("alt", "alt/path")
    reg.set_active("alt")
    assert reg.resolve_model_path() == "alt/path"


def test_resolve_unknown_raises(reg):
    with pytest.raises(FileNotFoundError):
        reg.resolve_model_path("ghost")


def test_persists_across_instances(reg, tmp_path):
    reg.register_model("alt", "alt/path", label="Alt")
    raw = json.loads((tmp_path / "registry.json").read_text())
    assert any(m["id"] == "alt" and m["label"] == "Alt" for m in raw["models"])


def test_inconsistent_active_falls_back_to_default(reg):
    state = {"active": "ghost", "models": []}
    reg.REGISTRY_PATH.write_text(json.dumps(state))
    a = reg.active_model()
    assert a.id == reg.DEFAULT_MODEL_ID
