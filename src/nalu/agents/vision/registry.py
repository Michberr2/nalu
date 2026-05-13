"""Per-model registry — maps short ids to MLX-VLM model paths and tracks the active one.

Lives at `MODELS_DIR/registry.json`. The first read seeds it with the built-in default
(`UI-TARS-1.5-7B-4bit`) so a fresh install behaves exactly as before.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ... import config


REGISTRY_PATH = config.MODELS_DIR / "registry.json"
DEFAULT_MODEL_ID = "ui-tars-1.5-7b-4bit"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass
class ModelEntry:
    id: str
    path: str
    kind: str = "base"
    label: str = ""
    added_ts: float = 0.0


def _default_state() -> dict:
    return {
        "active": DEFAULT_MODEL_ID,
        "models": [
            asdict(
                ModelEntry(
                    id=DEFAULT_MODEL_ID,
                    path=config.VISION_MODEL,
                    kind="base",
                    label="UI-TARS 1.5 7B (4-bit)",
                    added_ts=time.time(),
                )
            )
        ],
    }


def _load() -> dict:
    if not REGISTRY_PATH.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        state = _default_state()
        REGISTRY_PATH.write_text(json.dumps(state, indent=2))
        return state
    return json.loads(REGISTRY_PATH.read_text())


def _save(state: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(state, indent=2))


def list_models() -> list[ModelEntry]:
    state = _load()
    return [ModelEntry(**m) for m in state.get("models", [])]


def get_model(model_id: str) -> ModelEntry | None:
    for m in list_models():
        if m.id == model_id:
            return m
    return None


def register_model(model_id: str, path: str, kind: str = "base", label: str = "") -> ModelEntry:
    """Add or update a registry entry. Idempotent — re-registering an id replaces it."""
    if not _ID_RE.match(model_id):
        raise ValueError(
            f"invalid model id {model_id!r}: must be lowercase letters/digits/._- (max 64 chars)"
        )
    if not path:
        raise ValueError("path is required")
    state = _load()
    entry = ModelEntry(
        id=model_id,
        path=path,
        kind=kind,
        label=label,
        added_ts=time.time(),
    )
    models = [m for m in state.get("models", []) if m.get("id") != model_id]
    models.append(asdict(entry))
    state["models"] = models
    _save(state)
    return entry


def unregister_model(model_id: str) -> bool:
    """Drop an entry. Refuses to remove the active model — caller must `use` another first."""
    state = _load()
    if state.get("active") == model_id:
        raise ValueError(f"cannot unregister {model_id!r}: it is the active model")
    before = len(state.get("models", []))
    state["models"] = [m for m in state.get("models", []) if m.get("id") != model_id]
    if len(state["models"]) == before:
        return False
    _save(state)
    return True


def active_model_id() -> str:
    state = _load()
    return state.get("active") or DEFAULT_MODEL_ID


def active_model() -> ModelEntry:
    """Return the active entry. If the registry is somehow inconsistent, fall back to default."""
    aid = active_model_id()
    entry = get_model(aid)
    if entry is None:
        return ModelEntry(id=DEFAULT_MODEL_ID, path=config.VISION_MODEL, kind="base")
    return entry


def set_active(model_id: str) -> ModelEntry:
    entry = get_model(model_id)
    if entry is None:
        raise FileNotFoundError(f"unknown model id: {model_id!r}")
    state = _load()
    state["active"] = model_id
    _save(state)
    return entry


def resolve_model_path(model_id: str | None = None) -> str:
    """If `model_id` is given, look it up; else return the active model's path."""
    if model_id is None:
        return active_model().path
    entry = get_model(model_id)
    if entry is None:
        raise FileNotFoundError(f"unknown model id: {model_id!r}")
    return entry.path
