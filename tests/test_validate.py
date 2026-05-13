from __future__ import annotations

from nalu.agents.planner.validate import RefusalSignal, validate_action


W, H = 1200, 800


def test_click_inside_bounds_passes():
    assert validate_action("click", {"x": 100, "y": 100}, W, H) is None


def test_click_at_edge_passes():
    assert validate_action("click", {"x": 0, "y": 0}, W, H) is None
    assert validate_action("click", {"x": W - 1, "y": H - 1}, W, H) is None


def test_click_one_past_edge_refused():
    sig = validate_action("click", {"x": W, "y": 0}, W, H)
    assert isinstance(sig, RefusalSignal)
    assert sig.reason == "out_of_bounds"
    assert "click" in sig.hint
    assert f"{W}×{H}" in sig.hint


def test_click_negative_coords_refused():
    sig = validate_action("click", {"x": -1, "y": 50}, W, H)
    assert sig is not None and sig.reason == "out_of_bounds"


def test_click_far_out_of_bounds_refused_with_actual_value_in_hint():
    sig = validate_action("click", {"x": 4096, "y": 50}, W, H)
    assert sig is not None and "4096" in sig.hint


def test_double_click_validates_same_as_click():
    assert validate_action("double_click", {"x": 50, "y": 50}, W, H) is None
    sig = validate_action("double_click", {"x": -1, "y": 50}, W, H)
    assert sig is not None and sig.reason == "out_of_bounds"
    assert "double_click" in sig.hint


def test_missing_coords_refused():
    sig = validate_action("click", {"y": 50}, W, H)
    assert sig is not None and sig.reason == "missing_coords"
    assert "x=None" in sig.hint


def test_non_numeric_coords_refused():
    sig = validate_action("click", {"x": "fifty", "y": 50}, W, H)
    assert sig is not None and sig.reason == "missing_coords"
    assert "fifty" in sig.hint


def test_bool_coords_refused_not_treated_as_int():
    # bool is a subclass of int in Python; we should not silently accept True as x=1.
    sig = validate_action("click", {"x": True, "y": 50}, W, H)
    assert sig is not None and sig.reason == "missing_coords"


def test_float_coords_accepted_and_truncated():
    # Vision sometimes emits floats. Accept them.
    assert validate_action("click", {"x": 100.7, "y": 50.2}, W, H) is None


def test_drag_with_both_endpoints_in_bounds_passes():
    args = {"x1": 10, "y1": 10, "x2": 100, "y2": 100}
    assert validate_action("drag", args, W, H) is None


def test_drag_with_start_out_of_bounds_refused():
    args = {"x1": -5, "y1": 10, "x2": 100, "y2": 100}
    sig = validate_action("drag", args, W, H)
    assert sig is not None and sig.reason == "out_of_bounds"
    assert "drag start" in sig.hint


def test_drag_with_end_out_of_bounds_refused():
    args = {"x1": 10, "y1": 10, "x2": W + 5, "y2": 100}
    sig = validate_action("drag", args, W, H)
    assert sig is not None and sig.reason == "out_of_bounds"
    assert "drag end" in sig.hint


def test_non_coord_kinds_pass_through():
    # type, key, scroll, wait, done — none touch screen coords.
    assert validate_action("type", {"text": "hello"}, W, H) is None
    assert validate_action("key", {"name": "enter"}, W, H) is None
    assert validate_action("scroll", {"dx": 0, "dy": -100}, W, H) is None
    assert validate_action("wait", {"ms": 500}, W, H) is None
    assert validate_action("done", {"answer": "yes"}, W, H) is None


def test_non_dict_args_refused_for_coord_kinds():
    sig = validate_action("click", None, W, H)
    assert sig is not None and sig.reason == "bad_args"


def test_non_dict_args_refused_for_type():
    sig = validate_action("type", None, W, H)
    assert sig is not None and sig.reason == "bad_args"


def test_type_without_text_refused():
    sig = validate_action("type", {}, W, H)
    assert isinstance(sig, RefusalSignal)
    assert sig.reason == "missing_text"
    assert "text" in sig.hint


def test_type_with_empty_text_refused():
    sig = validate_action("type", {"text": ""}, W, H)
    assert sig is not None and sig.reason == "missing_text"


def test_type_with_non_string_text_refused():
    sig = validate_action("type", {"text": 42}, W, H)
    assert sig is not None and sig.reason == "missing_text"


def test_key_without_name_refused():
    sig = validate_action("key", {}, W, H)
    assert isinstance(sig, RefusalSignal)
    assert sig.reason == "missing_key_name"


def test_key_with_empty_name_refused():
    sig = validate_action("key", {"name": ""}, W, H)
    assert sig is not None and sig.reason == "missing_key_name"


def test_key_with_modifiers_only_refused():
    sig = validate_action("key", {"modifiers": ["cmd"]}, W, H)
    assert sig is not None and sig.reason == "missing_key_name"


def test_unknown_kind_passes_through():
    # Validation is best-effort. Unknown kinds are someone else's problem.
    assert validate_action("teleport", {"x": 50, "y": 50}, W, H) is None
