"""Completion verification — re-ask the model whether a `done` action is real.

The vision model occasionally hallucinates completion: it emits `finished()` while
the form on screen is still empty, or "task complete" when an error dialog is open.
Without a check, the planner publishes `task_completed` and the user is told the
job is done when it isn't.

This module asks the same model a follow-up question against the *current*
screenshot — "you said it's done; is it really?" — and parses YES/NO + a short
reason. We default to `confirmed=False` on ambiguous output: a false-deny just
keeps the agent working another turn, but a false-confirm misleads the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from PIL import Image


VERIFY_PROMPT_TEMPLATE = (
    "You previously decided this task is complete with this answer:\n"
    "  {answer}\n\n"
    "Re-examine the current screenshot. Is the user's original goal really achieved?\n"
    "  Goal: {goal}\n\n"
    "Reply with exactly one line, starting with YES or NO:\n"
    "  YES <one short sentence of evidence>\n"
    "  NO <one short sentence of what's still missing>"
)

_LEAD_RE = re.compile(r"^\s*([A-Za-z]+)\b[\s,:.\-]*", re.IGNORECASE)


@dataclass
class VerifyResult:
    confirmed: bool
    reasoning: str
    raw: str = ""


def build_verify_prompt(goal: str, claimed_answer: str) -> str:
    answer = (claimed_answer or "").strip() or "(no answer text)"
    return VERIFY_PROMPT_TEMPLATE.format(goal=goal.strip(), answer=answer)


def parse_verify_response(text: str) -> VerifyResult:
    """Extract YES/NO + a short reason from the model's reply.

    Defaults to `confirmed=False` for empty or ambiguous output.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return VerifyResult(confirmed=False, reasoning="empty response", raw=text or "")

    first_line = cleaned.splitlines()[0].strip()
    m = _LEAD_RE.match(first_line)
    if not m:
        return VerifyResult(confirmed=False, reasoning=f"ambiguous: {first_line[:120]}", raw=cleaned)

    head = m.group(1).lower()
    tail = first_line[m.end():].strip()
    if head == "yes":
        return VerifyResult(confirmed=True, reasoning=tail or "confirmed", raw=cleaned)
    if head == "no":
        return VerifyResult(confirmed=False, reasoning=tail or "denied", raw=cleaned)
    return VerifyResult(confirmed=False, reasoning=f"ambiguous: {first_line[:120]}", raw=cleaned)


JudgeCallable = Callable[[Image.Image, str], str]


def verify_completion(
    judge: JudgeCallable, image: Image.Image, goal: str, claimed_answer: str,
) -> VerifyResult:
    """Run the verifier prompt through `judge` and parse the reply.

    `judge(image, prompt) -> str` is anything that returns raw model text. In
    production this is `VisionAgent.judge`; tests can pass a stub function.
    Exceptions in the judge collapse into `confirmed=False` so a flaky verifier
    never silently confirms.
    """
    prompt = build_verify_prompt(goal, claimed_answer)
    try:
        raw = judge(image, prompt)
    except Exception as e:
        return VerifyResult(confirmed=False, reasoning=f"verifier error: {e}", raw="")
    return parse_verify_response(raw)
