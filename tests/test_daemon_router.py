from __future__ import annotations

import pytest

from nalu.daemon import classify_user_text


@pytest.mark.parametrize(
    "text",
    [
        "what time is it?",
        "what's the weather",
        "who wrote this",
        "how does this work",
        "why is the sky blue",
        "hi there",
        "hello",
        "thanks",
        "tell me a joke",
        "good morning",
        "is python a snake",
    ],
)
def test_classify_conversational_returns_query(text):
    assert classify_user_text(text) == "query"


@pytest.mark.parametrize(
    "text",
    [
        "open Safari and navigate to weather.com",
        "click the submit button",
        "type my password",
        "search for Paris weather and save it to a note",
        "find Paris weather",
        "save the file",
        "create a new note in Notes",
        "switch to the Mail app",
    ],
)
def test_classify_actions_return_intent(text):
    assert classify_user_text(text) == "intent"


def test_empty_text_returns_intent():
    assert classify_user_text("") == "intent"
    assert classify_user_text("   ") == "intent"


def test_long_question_is_treated_as_intent():
    # A long, multi-clause question is more likely a research goal for the planner
    # than a chatty turn for the responder.
    text = "what is the current temperature in Paris and please also save it to a new note in the Notes app"
    assert classify_user_text(text) == "intent"


def test_action_verb_beats_question_prefix():
    # "find" is an action hint — must route to planner even though it could be conversational.
    assert classify_user_text("find Paris weather") == "intent"
