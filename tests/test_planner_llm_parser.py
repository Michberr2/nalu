from __future__ import annotations

import pytest

from nalu.agents.planner_llm.subgoal import Plan, Subgoal, parse_plan_response


GOAL = "research the weather in Paris and save to a note"


def test_subgoal_rejects_empty_string():
    with pytest.raises(ValueError):
        Subgoal(goal="")


def test_subgoal_rejects_whitespace():
    with pytest.raises(ValueError):
        Subgoal(goal="   ")


def test_plan_iteration_and_length():
    p = Plan(subgoals=[Subgoal(goal="a"), Subgoal(goal="b")])
    assert len(p) == 2
    assert [s.goal for s in p] == ["a", "b"]
    assert not p.is_empty()


def test_parse_clean_json_array():
    raw = '[{"goal": "Open Safari", "success_criteria": "Safari frontmost"}, {"goal": "Type weather.com"}]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert len(plan) == 2
    assert plan.subgoals[0].goal == "Open Safari"
    assert plan.subgoals[0].success_criteria == "Safari frontmost"
    assert plan.subgoals[1].success_criteria == ""


def test_parse_fenced_json_block():
    raw = "Here's the plan:\n```json\n[{\"goal\": \"Open Notes\"}]\n```\nLet me know."
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert len(plan) == 1
    assert plan.subgoals[0].goal == "Open Notes"


def test_parse_fenced_block_without_lang_tag():
    raw = "```\n[{\"goal\": \"Click Apply\"}]\n```"
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert plan.subgoals[0].goal == "Click Apply"


def test_parse_object_with_plan_key():
    raw = '{"plan": [{"goal": "Open Notes"}, {"goal": "Type the note"}]}'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert len(plan) == 2


def test_parse_object_with_subgoals_key():
    raw = '{"subgoals": [{"goal": "Step one"}]}'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert plan.subgoals[0].goal == "Step one"


def test_parse_object_with_steps_key():
    raw = '{"steps": [{"step": "Open it"}, {"step": "Close it"}]}'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert plan.subgoals[0].goal == "Open it"


def test_parse_strings_in_array():
    raw = '["Open Safari", "Search for Paris", "Read temperature"]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert len(plan) == 3
    assert plan.subgoals[1].goal == "Search for Paris"


def test_parse_strips_whitespace():
    raw = '[{"goal": "  Open Safari  ", "success_criteria": "  frontmost  "}]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert plan.subgoals[0].goal == "Open Safari"
    assert plan.subgoals[0].success_criteria == "frontmost"


def test_parse_skips_invalid_items_in_array():
    raw = '[{"goal": "Valid"}, {}, {"goal": ""}, 42, "Another valid"]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert [s.goal for s in plan] == ["Valid", "Another valid"]


def test_parse_array_of_all_invalid_falls_back():
    raw = '[{}, {"goal": ""}, 42]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert plan.fallback
    assert len(plan) == 1
    assert plan.subgoals[0].goal == GOAL


def test_parse_empty_string_falls_back():
    plan = parse_plan_response("", fallback_goal=GOAL)
    assert plan.fallback
    assert plan.subgoals[0].goal == GOAL


def test_parse_pure_prose_falls_back():
    plan = parse_plan_response("Sure, I can help with that!", fallback_goal=GOAL)
    assert plan.fallback


def test_parse_malformed_json_falls_back():
    plan = parse_plan_response('[{"goal": "missing brace"', fallback_goal=GOAL)
    assert plan.fallback


def test_parse_bullet_list_recovered():
    raw = """Here's what I'd do:
1. Open Safari
2. Navigate to weather.com
3. Read the Paris forecast
"""
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert len(plan) == 3
    assert plan.subgoals[0].goal == "Open Safari"


def test_parse_dash_bullets_recovered():
    raw = "- Step one\n- Step two\n- Step three"
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert len(plan) == 3


def test_parse_single_bullet_does_not_recover():
    # one bullet is indistinguishable from a heading — fall back.
    raw = "- Step one"
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert plan.fallback


def test_parse_alternate_goal_field_names():
    raw = '[{"task": "Open Mail"}, {"description": "Compose"}]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert not plan.fallback
    assert plan.subgoals[0].goal == "Open Mail"
    assert plan.subgoals[1].goal == "Compose"


def test_parse_alternate_criteria_field_names():
    raw = '[{"goal": "Open", "done_when": "visible"}]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert plan.subgoals[0].success_criteria == "visible"


def test_non_string_criteria_coerced():
    raw = '[{"goal": "Open", "success_criteria": 42}]'
    plan = parse_plan_response(raw, fallback_goal=GOAL)
    assert plan.subgoals[0].success_criteria == "42"


def test_none_raw_is_safe():
    plan = parse_plan_response(None, fallback_goal=GOAL)  # type: ignore[arg-type]
    assert plan.fallback


def test_empty_array_falls_back():
    plan = parse_plan_response("[]", fallback_goal=GOAL)
    assert plan.fallback
    assert len(plan) == 1
