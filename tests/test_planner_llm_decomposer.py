from __future__ import annotations

import pytest

from nalu.agents.planner_llm.decomposer import LLMDecomposer
from nalu.agents.planner_llm.prompts import (
    build_decompose_user_prompt,
    build_replan_user_prompt,
    format_plan_for_log,
)
from nalu.agents.planner_llm.subgoal import Plan, Subgoal


class _FakeDecomposer(LLMDecomposer):
    """Replaces _generate with a canned response so tests don't need MLX."""

    def __init__(self, canned: str | Exception):
        super().__init__(model_id="fake")
        self.canned = canned
        self.calls: list[tuple[str, str]] = []

    def _generate(self, system: str, user: str) -> str:  # type: ignore[override]
        self.calls.append((system, user))
        if isinstance(self.canned, Exception):
            raise self.canned
        return self.canned


def test_decompose_returns_plan_for_clean_json():
    d = _FakeDecomposer('[{"goal": "Open Safari"}, {"goal": "Type weather.com"}]')
    plan = d.decompose("research the weather in Paris")
    assert not plan.fallback
    assert len(plan) == 2
    assert plan.subgoals[0].goal == "Open Safari"


def test_decompose_invokes_generate_with_user_goal():
    d = _FakeDecomposer('[{"goal": "x"}]')
    d.decompose("look up the time")
    system, user = d.calls[0]
    assert "look up the time" in user
    assert "planner" in system.lower()


def test_decompose_empty_goal_returns_empty_plan_without_call():
    d = _FakeDecomposer('[{"goal": "should not appear"}]')
    plan = d.decompose("")
    assert plan.is_empty()
    assert plan.fallback
    assert not d.calls


def test_decompose_whitespace_goal_returns_empty_plan_without_call():
    d = _FakeDecomposer('[{"goal": "x"}]')
    plan = d.decompose("   \n\t  ")
    assert plan.is_empty()
    assert not d.calls


def test_decompose_falls_back_on_generate_exception():
    d = _FakeDecomposer(RuntimeError("oom"))
    plan = d.decompose("research the weather")
    assert plan.fallback
    assert plan.subgoals[0].goal == "research the weather"


def test_decompose_falls_back_on_unparseable_output():
    d = _FakeDecomposer("I'm a chatbot and I refuse to do JSON today.")
    plan = d.decompose("research the weather")
    assert plan.fallback
    assert plan.subgoals[0].goal == "research the weather"


def test_decompose_passes_conversation_when_provided():
    d = _FakeDecomposer('[{"goal": "x"}]')
    d.decompose("follow up", conversation="User: hi\nNalu: hello")
    _, user = d.calls[0]
    assert "Recent conversation" in user
    assert "Nalu: hello" in user


def test_decompose_omits_conversation_block_when_empty():
    d = _FakeDecomposer('[{"goal": "x"}]')
    d.decompose("solo goal")
    _, user = d.calls[0]
    assert "Recent conversation" not in user


def test_replan_returns_plan_for_clean_json():
    d = _FakeDecomposer('[{"goal": "Close modal"}, {"goal": "Retry the search"}]')
    plan = d.replan(
        original_goal="search weather",
        completed=["Open Safari"],
        failed_subgoal="Search for Paris",
        failure_reason="stuck:repeat",
    )
    assert not plan.fallback
    assert len(plan) == 2
    assert plan.subgoals[0].goal == "Close modal"


def test_replan_invokes_with_failure_context():
    d = _FakeDecomposer('[{"goal": "x"}]')
    d.replan(
        original_goal="find Paris weather",
        completed=["Open Safari", "Type weather.com"],
        failed_subgoal="Search Paris",
        failure_reason="stuck:alternation",
    )
    system, user = d.calls[0]
    assert "replanner" in system.lower()
    assert "find Paris weather" in user
    assert "Open Safari" in user
    assert "Type weather.com" in user
    assert "stuck:alternation" in user


def test_replan_falls_back_on_exception_with_failed_subgoal_as_goal():
    d = _FakeDecomposer(RuntimeError("model crash"))
    plan = d.replan(
        original_goal="x",
        completed=[],
        failed_subgoal="Search Paris",
        failure_reason="timeout",
    )
    assert plan.fallback
    assert plan.subgoals[0].goal == "Search Paris"


def test_replan_includes_screen_summary_when_provided():
    d = _FakeDecomposer('[{"goal": "x"}]')
    d.replan(
        original_goal="g",
        completed=[],
        failed_subgoal="s",
        failure_reason="r",
        screen_summary="A modal dialog covers the search box.",
    )
    _, user = d.calls[0]
    assert "Current screen" in user
    assert "modal dialog" in user


def test_build_decompose_prompt_strips_user_goal_whitespace():
    p = build_decompose_user_prompt("  research  \n")
    assert "User goal\nresearch" in p


def test_build_replan_prompt_renders_empty_completed_as_none():
    p = build_replan_user_prompt(
        original_goal="g", completed=[], failed_subgoal="s", failure_reason="r"
    )
    assert "(none)" in p


def test_format_plan_for_log_is_numbered():
    plan = Plan(subgoals=[Subgoal(goal="alpha"), Subgoal(goal="beta")])
    out = format_plan_for_log(plan)
    assert "1. alpha" in out
    assert "2. beta" in out
