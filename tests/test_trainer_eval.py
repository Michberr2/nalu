from __future__ import annotations

from nalu.agents.trainer.eval import _action_correct, _click_distance, _text_match


def test_action_kind_match():
    assert _action_correct("click", "click")
    assert _action_correct("type", "type")


def test_action_click_double_click_equivalence():
    assert _action_correct("click", "double_click")
    assert _action_correct("double_click", "click")


def test_action_kind_mismatch():
    assert not _action_correct("click", "type")
    assert not _action_correct("scroll", "key")


def test_click_distance_basic():
    assert _click_distance({"x": 0, "y": 0}, {"x": 3, "y": 4}) == 5.0


def test_click_distance_missing_pred():
    assert _click_distance({}, {"x": 1, "y": 2}) is None


def test_click_distance_missing_truth():
    assert _click_distance({"x": 1, "y": 2}, {}) is None


def test_text_match_exact():
    assert _text_match({"text": "Hello"}, {"text": "hello"}) is True


def test_text_match_strips_whitespace():
    assert _text_match({"text": "  hi  "}, {"text": "hi"}) is True


def test_text_match_done_answer_field():
    assert _text_match({"answer": "42"}, {"answer": "42"}) is True


def test_text_match_mismatch():
    assert _text_match({"text": "foo"}, {"text": "bar"}) is False


def test_text_match_none_when_field_absent():
    assert _text_match({}, {"text": "foo"}) is None
