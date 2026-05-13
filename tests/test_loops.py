from __future__ import annotations

from nalu.agents.planner.loops import (
    LoopDetector,
    StuckSignal,
    action_signature,
)


def test_signature_collapses_clicks_within_grid_bucket():
    a = action_signature("click", {"x": 100, "y": 200})
    b = action_signature("click", {"x": 110, "y": 215})
    c = action_signature("click", {"x": 200, "y": 200})
    assert a == b
    assert a != c


def test_signature_distinguishes_button_and_clicks():
    base = action_signature("click", {"x": 0, "y": 0})
    right = action_signature("click", {"x": 0, "y": 0, "button": "right"})
    double = action_signature("click", {"x": 0, "y": 0, "clicks": 2})
    assert base != right
    assert base != double


def test_signature_truncates_long_type_text():
    a = action_signature("type", {"text": "x" * 200})
    b = action_signature("type", {"text": "x" * 50})
    assert a == b


def test_signature_handles_unknown_kind_gracefully():
    sig = action_signature("weird", {"foo": 1, "bar": "two"})
    assert sig[0] == "weird"


def test_repeat_does_not_fire_below_threshold():
    d = LoopDetector(repeat_threshold=3)
    assert d.observe("click", {"x": 100, "y": 100}) is None
    assert d.observe("click", {"x": 100, "y": 100}) is None


def test_repeat_fires_at_threshold():
    d = LoopDetector(repeat_threshold=3)
    d.observe("click", {"x": 100, "y": 100})
    d.observe("click", {"x": 100, "y": 100})
    sig = d.observe("click", {"x": 100, "y": 100})
    assert sig is not None
    assert sig.reason == "repeat"
    assert "click" in sig.hint


def test_repeat_uses_grid_bucketing():
    d = LoopDetector(repeat_threshold=3, click_grid_px=32)
    d.observe("click", {"x": 100, "y": 100})
    d.observe("click", {"x": 110, "y": 105})
    sig = d.observe("click", {"x": 120, "y": 115})
    assert sig is not None
    assert sig.reason == "repeat"


def test_repeat_resets_when_action_changes():
    d = LoopDetector(repeat_threshold=3)
    d.observe("click", {"x": 100, "y": 100})
    d.observe("click", {"x": 100, "y": 100})
    assert d.observe("scroll", {"dx": 0, "dy": -1}) is None
    assert d.observe("click", {"x": 100, "y": 100}) is None


def test_alternation_fires_after_three_cycles():
    d = LoopDetector(repeat_threshold=99, alternation_cycles=3)
    for _ in range(3):
        d.observe("click", {"x": 0, "y": 0})
        sig = d.observe("click", {"x": 200, "y": 200})
    assert sig is not None
    assert sig.reason == "alternation"


def test_alternation_does_not_fire_with_three_distinct_actions():
    d = LoopDetector(repeat_threshold=99, alternation_cycles=3)
    sigs = []
    for _ in range(3):
        sigs.append(d.observe("click", {"x": 0, "y": 0}))
        sigs.append(d.observe("click", {"x": 200, "y": 0}))
        sigs.append(d.observe("click", {"x": 0, "y": 200}))
    assert all(s is None or s.reason != "alternation" for s in sigs)


def test_consecutive_signals_increments_when_same_signature_re_fires():
    d = LoopDetector(repeat_threshold=3)
    d.observe("click", {"x": 100, "y": 100})
    d.observe("click", {"x": 100, "y": 100})
    first = d.observe("click", {"x": 100, "y": 100})
    second = d.observe("click", {"x": 100, "y": 100})
    assert first.count == 1
    assert second.count == 2
    assert d.consecutive_signals == 2


def test_consecutive_resets_when_signature_changes():
    d = LoopDetector(repeat_threshold=3)
    d.observe("click", {"x": 100, "y": 100})
    d.observe("click", {"x": 100, "y": 100})
    d.observe("click", {"x": 100, "y": 100})
    d.observe("scroll", {"dx": 0, "dy": -1})
    assert d.consecutive_signals == 0


def test_reset_clears_state():
    d = LoopDetector(repeat_threshold=3)
    for _ in range(5):
        d.observe("click", {"x": 100, "y": 100})
    d.reset()
    assert d.consecutive_signals == 0
    assert d.observe("click", {"x": 100, "y": 100}) is None


def test_stuck_signal_carries_signature_for_repeat():
    d = LoopDetector(repeat_threshold=3)
    d.observe("click", {"x": 100, "y": 100})
    d.observe("click", {"x": 100, "y": 100})
    sig = d.observe("click", {"x": 100, "y": 100})
    assert isinstance(sig, StuckSignal)
    assert sig.signature[0] == "click"
