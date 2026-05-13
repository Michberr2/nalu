"""First-run wizard.

Pure-Python orchestration over discrete `OnboardingStep`s. Each step is independently
testable and idempotent — re-running the wizard skips steps that already pass. The
CLI shell (`nalu onboard`) renders progress, prompts the user between blocking steps,
and surfaces remediation hints.

Steps in order:
  1. disk          — confirm enough free space for first-time model downloads
  2. permissions   — Screen Recording + Accessibility (required), Microphone (optional)
  3. tts           — Piper voice pack present and synthesizes a sentence (~30 MB)
  4. stt           — faster-whisper model loads (~150 MB)
  5. wake          — openwakeword keyword loads (skipped if wake-word disabled)
  6. vision        — vision model warm-up (~5 GB first run, ~16s cold load thereafter)
  7. screenshot    — capture current display + decode to a single Action
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from . import config

DISK_REQUIRED_GB = 8  # 5 GB vision + 0.2 GB STT + tts/cache/headroom
APPROX_DOWNLOAD_BYTES = {
    "tts": 30 * 1024 * 1024,
    "stt": 150 * 1024 * 1024,
    "vision": 5 * 1024 * 1024 * 1024,
}


class StepStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class StepResult:
    name: str
    status: StepStatus
    detail: str = ""
    fix_hint: str = ""
    elapsed_s: float = 0.0


@dataclass
class OnboardingStep:
    name: str
    summary: str
    run: Callable[[], StepResult]
    required: bool = True

    def execute(self) -> StepResult:
        t0 = time.monotonic()
        try:
            result = self.run()
        except Exception as e:
            result = StepResult(name=self.name, status=StepStatus.FAIL, detail=str(e))
        result.elapsed_s = time.monotonic() - t0
        return result


def _step_disk() -> StepResult:
    """Pre-flight free-space check so first-run downloads don't die mid-stream."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(config.MODELS_DIR)
    free_gb = usage.free / (1024**3)
    if free_gb < DISK_REQUIRED_GB:
        return StepResult(
            name="disk",
            status=StepStatus.FAIL,
            detail=f"only {free_gb:.1f} GB free at {config.MODELS_DIR} — Nalu needs ~{DISK_REQUIRED_GB} GB",
            fix_hint="free up space or set NALU_HOME to a directory on a larger volume",
        )
    return StepResult(
        name="disk",
        status=StepStatus.PASS,
        detail=f"{free_gb:.1f} GB free at {config.MODELS_DIR}",
    )


def _step_permissions() -> StepResult:
    from . import permissions

    statuses = permissions.check_all()
    blockers = [s for s in statuses if not s.granted and s.name in ("Screen Recording", "Accessibility")]
    if blockers:
        names = ", ".join(s.name for s in blockers)
        return StepResult(
            name="permissions",
            status=StepStatus.FAIL,
            detail=f"missing: {names}",
            fix_hint="run `nalu setup` to open the Settings panes",
        )
    missing_optional = [s.name for s in statuses if not s.granted]
    detail = "all required granted"
    if missing_optional:
        detail += f"; optional missing: {', '.join(missing_optional)}"
    return StepResult(name="permissions", status=StepStatus.PASS, detail=detail)


def _step_tts() -> StepResult:
    from .agents.voice import TTS

    tts = TTS()
    tts.load()
    return StepResult(
        name="tts",
        status=StepStatus.PASS,
        detail=f"voice {config.TTS_VOICE} loaded (length_scale={config.TTS_LENGTH_SCALE})",
    )


def _step_stt() -> StepResult:
    from .agents.voice import STT

    stt = STT()
    stt.load()
    return StepResult(
        name="stt",
        status=StepStatus.PASS,
        detail=f"faster-whisper {config.STT_MODEL} loaded",
    )


def _step_wake() -> StepResult:
    if not config.WAKEWORD_ENABLED:
        return StepResult(
            name="wake",
            status=StepStatus.SKIP,
            detail="NALU_WAKEWORD not set",
        )
    from .agents.voice import OpenWakeWordSpotter

    spotter = OpenWakeWordSpotter(models=[config.WAKEWORD_KEYWORD])
    spotter.load()
    return StepResult(
        name="wake",
        status=StepStatus.PASS,
        detail=f"wake-word {config.WAKEWORD_KEYWORD!r} loaded",
    )


def _step_vision() -> StepResult:
    from .agents.vision import VisionAgent

    vision = VisionAgent()
    vision.load()
    return StepResult(
        name="vision",
        status=StepStatus.PASS,
        detail=f"model {vision.model_id} loaded",
    )


def _step_screenshot() -> StepResult:
    from .agents.vision import VisionAgent
    from .capture import capture_main_display

    shot = capture_main_display()
    vision = VisionAgent()
    # Use judge() (raw text) rather than decide() (forces action parsing) — for a
    # smoke test we just want to verify the capture → vision round-trip works
    # without coupling success to whether the prompt happened to be action-shaped.
    response = vision.judge(shot.image, "Describe what is visible on this screen in one short sentence.")
    if not response or not response.strip():
        return StepResult(
            name="screenshot",
            status=StepStatus.FAIL,
            detail=f"captured {shot.captured_width}x{shot.captured_height}; vision returned no text",
            fix_hint="vision model loaded but produced empty output — check NALU_VISION_MODEL",
        )
    return StepResult(
        name="screenshot",
        status=StepStatus.PASS,
        detail=f"captured {shot.captured_width}x{shot.captured_height}; vision responded ({len(response)} chars)",
    )


def _step_planner_llm() -> StepResult:
    if not config.USE_LLM_PLANNER:
        return StepResult(
            name="planner_llm",
            status=StepStatus.SKIP,
            detail="NALU_USE_LLM_PLANNER not set and no planner.json — single-shot vision mode",
        )
    from .agents.planner_llm import LLMDecomposer

    decomposer = LLMDecomposer()
    decomposer.load()
    return StepResult(
        name="planner_llm",
        status=StepStatus.PASS,
        detail=f"planner LLM {decomposer.model_id} loaded",
    )


def default_steps() -> list[OnboardingStep]:
    return [
        OnboardingStep("disk", "Free disk space", _step_disk),
        OnboardingStep("permissions", "macOS permissions", _step_permissions),
        OnboardingStep("tts", "Piper TTS voice (~30 MB on first run)", _step_tts, required=False),
        OnboardingStep("stt", "faster-whisper STT (~150 MB on first run)", _step_stt, required=False),
        OnboardingStep("wake", "openwakeword keyword", _step_wake, required=False),
        OnboardingStep("vision", "Vision model warm-up (~5 GB on first run)", _step_vision),
        OnboardingStep("planner_llm", "Planner LLM (~5 GB; opt-in)", _step_planner_llm, required=False),
        OnboardingStep("screenshot", "Screenshot decode round-trip", _step_screenshot),
    ]


@dataclass
class OnboardingReport:
    results: list[StepResult] = field(default_factory=list)

    @property
    def required_failures(self) -> list[StepResult]:
        return [r for r in self.results if r.status == StepStatus.FAIL]

    @property
    def is_ready(self) -> bool:
        # Required failures are surfaced via OnboardingWizard.run, which already
        # respects the per-step `required` flag. By the time a report exists,
        # required steps that failed have been recorded as FAIL — readiness means
        # no FAILs from required steps remain. Optional steps (tts/stt/wake) are
        # excluded so a user without a microphone can still run nalu ask.
        optional = {"tts", "stt", "wake", "planner_llm"}
        return not any(r.status == StepStatus.FAIL for r in self.results if r.name not in optional)


class OnboardingWizard:
    """Runs steps in order, optionally re-running selected ones, gathers a report.

    The CLI uses `step_callback` to prompt the user between steps; tests can supply
    a callable that returns deterministic decisions.
    """

    def __init__(
        self,
        steps: list[OnboardingStep] | None = None,
        before_step: Callable[[OnboardingStep], None] | None = None,
        after_step: Callable[[OnboardingStep, StepResult], bool] | None = None,
    ):
        self._steps = steps if steps is not None else default_steps()
        self._before = before_step
        self._after = after_step

    def run(self) -> OnboardingReport:
        report = OnboardingReport()
        for step in self._steps:
            if self._before:
                self._before(step)
            result = step.execute()
            report.results.append(result)
            keep_going = True
            if self._after:
                keep_going = self._after(step, result)
            if step.required and result.status == StepStatus.FAIL and not keep_going:
                break
        return report
