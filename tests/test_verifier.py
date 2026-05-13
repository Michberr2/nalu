from __future__ import annotations

import numpy as np
from PIL import Image

from nalu.agents.planner.verifier import (
    VerifyResult,
    build_verify_prompt,
    parse_verify_response,
    verify_completion,
)


def _img() -> Image.Image:
    return Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")


def test_build_prompt_includes_goal_and_answer():
    p = build_verify_prompt("open Safari", "I clicked the Safari icon")
    assert "open Safari" in p
    assert "I clicked the Safari icon" in p
    assert "YES" in p and "NO" in p


def test_build_prompt_handles_empty_answer():
    p = build_verify_prompt("do thing", "")
    assert "(no answer text)" in p


def test_parse_yes_with_reason():
    r = parse_verify_response("YES the form is submitted, redirect visible")
    assert r.confirmed is True
    assert "form is submitted" in r.reasoning


def test_parse_no_with_reason():
    r = parse_verify_response("NO the submit button is still on screen")
    assert r.confirmed is False
    assert "submit button" in r.reasoning


def test_parse_lowercase_yes():
    r = parse_verify_response("yes done")
    assert r.confirmed is True


def test_parse_handles_punctuation_after_lead():
    r = parse_verify_response("YES, the dialog closed.")
    assert r.confirmed is True
    assert "dialog closed" in r.reasoning


def test_parse_empty_string_is_not_confirmed():
    r = parse_verify_response("")
    assert r.confirmed is False
    assert "empty" in r.reasoning.lower()


def test_parse_ambiguous_lead_is_not_confirmed():
    r = parse_verify_response("Maybe — hard to tell.")
    assert r.confirmed is False
    assert "ambiguous" in r.reasoning


def test_parse_uses_only_first_line():
    r = parse_verify_response("YES looks good\nbut also some extra musing")
    assert r.confirmed is True
    assert "looks good" in r.reasoning
    assert "extra musing" not in r.reasoning


def test_verify_completion_confirms_via_judge():
    def judge(image, prompt):
        assert "open Safari" in prompt
        return "YES Safari is in the foreground"

    out = verify_completion(judge, _img(), "open Safari", "clicked Safari")
    assert isinstance(out, VerifyResult)
    assert out.confirmed is True


def test_verify_completion_denies_via_judge():
    def judge(image, prompt):
        return "NO the dock didn't change"

    out = verify_completion(judge, _img(), "open Safari", "clicked Safari")
    assert out.confirmed is False
    assert "dock" in out.reasoning


def test_verify_completion_swallows_judge_exceptions():
    def boom(image, prompt):
        raise RuntimeError("model crashed")

    out = verify_completion(boom, _img(), "g", "a")
    assert out.confirmed is False
    assert "verifier error" in out.reasoning


def test_verify_completion_swallows_empty_judge_output():
    out = verify_completion(lambda i, p: "", _img(), "g", "a")
    assert out.confirmed is False
