from __future__ import annotations

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

VISION_MODEL = os.environ.get("NALU_VISION_MODEL", "mlx-community/UI-TARS-1.5-7B-4bit")
TTS_VOICE = os.environ.get("NALU_TTS_VOICE", "en_GB-alan-medium")
TTS_LENGTH_SCALE = float(os.environ.get("NALU_TTS_LENGTH_SCALE", "0.85"))
STT_MODEL = os.environ.get("NALU_STT_MODEL", "base.en")
CAPTURE_FPS = float(os.environ.get("NALU_CAPTURE_FPS", "2.0"))

PAUSE_HOTKEY = os.environ.get("NALU_PAUSE_HOTKEY", "<ctrl>+<alt>+<cmd>+.")
PUSH_TO_TALK_HOTKEY = os.environ.get("NALU_PTT_HOTKEY", "<ctrl>+<alt>+<cmd>+<space>")
PTT_RECORD_SECONDS = float(os.environ.get("NALU_PTT_SECONDS", "6.0"))

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
