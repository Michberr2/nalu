from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

APP_NAME = "Nalu"

ROOT = Path(os.environ.get("NALU_HOME") or Path.home() / "Library" / "Application Support" / APP_NAME)
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
RUN_DIR = ROOT / "run"
BUS_SOCKET = RUN_DIR / "bus.sock"
DAEMON_PID = RUN_DIR / "daemon.pid"
LOG_DIR = ROOT / "logs"
EVENTS_LOG = LOG_DIR / "events.jsonl"
PLANNER_CONFIG_FILE = MODELS_DIR / "planner.json"


def _load_planner_config() -> dict:
    """Read MODELS_DIR/planner.json if present. Returns {} on any failure — never raises."""
    try:
        if PLANNER_CONFIG_FILE.exists():
            return json.loads(PLANNER_CONFIG_FILE.read_text())
    except Exception:
        pass
    return {}


_planner_cfg = _load_planner_config()


def _truthy(v) -> bool:
    return str(v).strip().lower() not in ("0", "", "false", "no", "off")


VISION_MODEL = os.environ.get("NALU_VISION_MODEL", "mlx-community/UI-TARS-1.5-7B-4bit")
PLANNER_LLM_MODEL = (
    os.environ.get("NALU_PLANNER_LLM")
    or _planner_cfg.get("model_id")
    or "mlx-community/Qwen2.5-7B-Instruct-4bit"
)
USE_LLM_PLANNER = (
    _truthy(os.environ.get("NALU_USE_LLM_PLANNER", "0"))
    or bool(_planner_cfg.get("enabled", False))
)
PLANNER_SUBGOAL_MAX_STEPS = int(os.environ.get("NALU_PLANNER_SUBGOAL_STEPS", "8"))
PLANNER_MAX_REPLANS = int(os.environ.get("NALU_PLANNER_MAX_REPLANS", "1"))


def write_planner_config(*, enabled: bool | None = None, model_id: str | None = None) -> dict:
    """Persist planner state to MODELS_DIR/planner.json. Returns the new state.

    Either field may be left as None to keep its prior value. Env vars still
    override at runtime — this file is the "no env vars set" default.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    current = _load_planner_config()
    if enabled is not None:
        current["enabled"] = bool(enabled)
    if model_id is not None:
        current["model_id"] = str(model_id)
    PLANNER_CONFIG_FILE.write_text(json.dumps(current, indent=2))
    return current


def read_planner_config() -> dict:
    """Snapshot of MODELS_DIR/planner.json (re-read at call time, not cached)."""
    return _load_planner_config()
TTS_VOICE = os.environ.get("NALU_TTS_VOICE", "en_GB-alan-medium")
TTS_LENGTH_SCALE = float(os.environ.get("NALU_TTS_LENGTH_SCALE", "0.85"))
STT_MODEL = os.environ.get("NALU_STT_MODEL", "base.en")
CAPTURE_FPS = float(os.environ.get("NALU_CAPTURE_FPS", "2.0"))

PAUSE_HOTKEY = os.environ.get("NALU_PAUSE_HOTKEY", "<ctrl>+<alt>+<cmd>+.")
PUSH_TO_TALK_HOTKEY = os.environ.get("NALU_PTT_HOTKEY", "<ctrl>+<alt>+<cmd>+<space>")
PTT_RECORD_SECONDS = float(os.environ.get("NALU_PTT_SECONDS", "6.0"))

WAKEWORD_ENABLED = os.environ.get("NALU_WAKEWORD", "0") not in ("0", "", "false", "False")
WAKEWORD_KEYWORD = os.environ.get("NALU_WAKEWORD_KEYWORD", "hey_jarvis")
WAKEWORD_THRESHOLD = float(os.environ.get("NALU_WAKEWORD_THRESHOLD", "0.5"))
WAKEWORD_COOLDOWN_S = float(os.environ.get("NALU_WAKEWORD_COOLDOWN_S", "2.0"))

CAPTURE_MAX_WIDTH = 1280
CAPTURE_JPEG_QUALITY = 70

PLANNER_MAX_STEPS = 25
PLANNER_STEP_TIMEOUT_S = 30
PLANNER_TASK_TIMEOUT_S = 300


def ensure_dirs() -> None:
    for p in (ROOT, RUNS_DIR, MODELS_DIR, RUN_DIR, LOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def new_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RUNS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    return d
