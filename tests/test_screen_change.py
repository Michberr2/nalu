from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from nalu.agents.planner.screen_change import (
    DEFAULT_CHANGE_THRESHOLD,
    EFFECT_BEARING_KINDS,
    NoEffectSignal,
    evaluate_action_effect,
    is_changed,
    is_effect_bearing,
    perceptual_diff,
)


def _solid(width: int, height: int, value: int) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8), mode="RGB")


def _half_split(width: int, height: int, left: int, right: int) -> Image.Image:
    arr = np.full((height, width, 3), left, dtype=np.uint8)
    arr[:, width // 2:, :] = right
    return Image.fromarray(arr, mode="RGB")


def test_perceptual_diff_zero_for_identical_frames():
    a = _solid(128, 96, 128)
    assert perceptual_diff(a, a) == 0.0


def test_perceptual_diff_high_for_inverted_frames():
    a = _solid(128, 96, 0)
    b = _solid(128, 96, 255)
    assert perceptual_diff(a, b) == pytest.approx(1.0, abs=1e-3)


def test_perceptual_diff_partial_change():
    a = _solid(128, 96, 0)
    b = _half_split(128, 96, 0, 255)
    diff = perceptual_diff(a, b)
    assert 0.4 < diff < 0.55


def test_perceptual_diff_invalid_downsample():
    a = _solid(64, 64, 100)
    with pytest.raises(ValueError):
        perceptual_diff(a, a, downsample=0)


def test_is_changed_false_for_identical():
    a = _solid(64, 64, 100)
    assert is_changed(a, a) is False


def test_is_changed_true_for_different():
    a = _solid(64, 64, 0)
    b = _solid(64, 64, 255)
    assert is_changed(a, b) is True


def test_is_changed_respects_custom_threshold():
    a = _solid(64, 64, 100)
    b = _solid(64, 64, 102)
    diff = perceptual_diff(a, b)
    assert is_changed(a, b, threshold=diff + 0.001) is False
    assert is_changed(a, b, threshold=diff - 0.001) is True


def test_is_effect_bearing_lists_expected_kinds():
    assert set(EFFECT_BEARING_KINDS) == {"click", "double_click", "type", "key", "scroll", "drag"}
    assert is_effect_bearing("click")
    assert not is_effect_bearing("wait")
    assert not is_effect_bearing("done")


def test_evaluate_returns_none_for_non_effect_bearing():
    a = _solid(64, 64, 100)
    assert evaluate_action_effect("wait", a, a) is None
    assert evaluate_action_effect("done", a, a) is None


def test_evaluate_returns_signal_when_no_change():
    a = _solid(64, 64, 100)
    sig = evaluate_action_effect("click", a, a)
    assert isinstance(sig, NoEffectSignal)
    assert sig.action_kind == "click"
    assert sig.diff == 0.0
    assert sig.threshold == DEFAULT_CHANGE_THRESHOLD
    assert "click" in sig.hint


def test_evaluate_returns_none_when_screen_changed():
    a = _solid(64, 64, 0)
    b = _solid(64, 64, 255)
    assert evaluate_action_effect("click", a, b) is None


def test_evaluate_uses_custom_threshold():
    a = _solid(64, 64, 100)
    b = _solid(64, 64, 102)
    diff = perceptual_diff(a, b)
    assert evaluate_action_effect("click", a, b, threshold=diff + 0.001) is not None
    assert evaluate_action_effect("click", a, b, threshold=diff - 0.001) is None


def test_evaluate_handles_drag_and_scroll():
    a = _solid(64, 64, 100)
    sig = evaluate_action_effect("scroll", a, a)
    assert sig is not None and sig.action_kind == "scroll"
    sig = evaluate_action_effect("drag", a, a)
    assert sig is not None and sig.action_kind == "drag"
