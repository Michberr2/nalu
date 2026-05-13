from __future__ import annotations

from nalu.onboarding import (
    OnboardingStep,
    OnboardingWizard,
    StepResult,
    StepStatus,
)


def _passing(name: str) -> OnboardingStep:
    return OnboardingStep(
        name=name,
        summary=name,
        run=lambda: StepResult(name=name, status=StepStatus.PASS, detail="ok"),
    )


def _failing(name: str, *, required: bool = True) -> OnboardingStep:
    return OnboardingStep(
        name=name,
        summary=name,
        run=lambda: StepResult(name=name, status=StepStatus.FAIL, detail="boom", fix_hint="try again"),
        required=required,
    )


def _raising(name: str) -> OnboardingStep:
    def _go() -> StepResult:
        raise RuntimeError("kaboom")

    return OnboardingStep(name=name, summary=name, run=_go)


def test_wizard_runs_all_steps_when_passing():
    seen: list[str] = []

    def before(step):
        seen.append(f"before:{step.name}")

    def after(step, result):
        seen.append(f"after:{step.name}:{result.status.value}")
        return True

    wizard = OnboardingWizard(steps=[_passing("a"), _passing("b")], before_step=before, after_step=after)
    report = wizard.run()
    assert [r.name for r in report.results] == ["a", "b"]
    assert seen == [
        "before:a", "after:a:pass",
        "before:b", "after:b:pass",
    ]
    assert report.is_ready


def test_wizard_records_step_failure_and_continues_when_callback_says_so():
    after = lambda step, result: True
    wizard = OnboardingWizard(
        steps=[_failing("a"), _passing("b")],
        after_step=after,
    )
    report = wizard.run()
    assert [r.status for r in report.results] == [StepStatus.FAIL, StepStatus.PASS]
    assert not report.is_ready


def test_wizard_aborts_when_callback_returns_false_for_required_failure():
    wizard = OnboardingWizard(
        steps=[_failing("a", required=True), _passing("b")],
        after_step=lambda step, result: False,
    )
    report = wizard.run()
    assert [r.name for r in report.results] == ["a"]
    assert not report.is_ready


def test_wizard_does_not_abort_on_optional_failure():
    wizard = OnboardingWizard(
        steps=[_failing("a", required=False), _passing("b")],
        after_step=lambda step, result: False,
    )
    report = wizard.run()
    assert [r.name for r in report.results] == ["a", "b"]


def test_wizard_catches_step_exceptions():
    wizard = OnboardingWizard(steps=[_raising("a")])
    report = wizard.run()
    assert report.results[0].status == StepStatus.FAIL
    assert "kaboom" in report.results[0].detail


def test_step_records_elapsed_time():
    s = _passing("a")
    r = s.execute()
    assert r.elapsed_s >= 0.0


def test_skip_status_does_not_count_as_failure():
    skipped = OnboardingStep(
        name="wake",
        summary="wake",
        run=lambda: StepResult(name="wake", status=StepStatus.SKIP, detail="not enabled"),
    )
    wizard = OnboardingWizard(steps=[_passing("a"), skipped])
    report = wizard.run()
    assert report.is_ready
    assert any(r.status == StepStatus.SKIP for r in report.results)


def test_wake_failure_does_not_block_readiness():
    """Wake-word is optional even when its step exists; readiness ignores it."""
    wake_fail = OnboardingStep(
        name="wake",
        summary="wake",
        run=lambda: StepResult(name="wake", status=StepStatus.FAIL, detail="no model"),
        required=False,
    )
    wizard = OnboardingWizard(steps=[_passing("permissions"), wake_fail, _passing("vision")])
    report = wizard.run()
    assert report.is_ready


def test_required_failures_listed():
    wizard = OnboardingWizard(steps=[_failing("a"), _passing("b"), _failing("c", required=False)])
    report = wizard.run()
    fail_names = {r.name for r in report.required_failures}
    assert fail_names == {"a", "c"}  # both flagged regardless of `required`; required_failures = all FAIL


def test_default_steps_includes_expected_names():
    from nalu.onboarding import default_steps

    names = [s.name for s in default_steps()]
    assert names == ["disk", "permissions", "tts", "stt", "wake", "vision", "screenshot"]


def test_disk_step_passes_when_volume_has_room(tmp_path, monkeypatch):
    """Disk pre-flight should report PASS on a volume with plenty of space."""
    from nalu import config, onboarding

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    result = onboarding._step_disk()
    assert result.status == onboarding.StepStatus.PASS
    assert "free" in result.detail


def test_disk_step_fails_when_required_threshold_jumped(monkeypatch, tmp_path):
    """Mock a 1 GB free volume to verify the failure path surfaces a fix hint."""
    import shutil
    from nalu import config, onboarding

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(onboarding, "DISK_REQUIRED_GB", 10**9)  # bigger than any plausible volume
    result = onboarding._step_disk()
    assert result.status == onboarding.StepStatus.FAIL
    assert "GB free" in result.detail
    assert "NALU_HOME" in result.fix_hint
