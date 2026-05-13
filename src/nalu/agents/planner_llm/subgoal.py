"""Plan / Subgoal dataclasses + tolerant LLM-output parser.

The decomposer prompt asks for a JSON list of objects with `goal` and optional
`success_criteria`. Real LLM outputs vary: fenced code blocks, leading prose
("Sure! Here's the plan:"), JSON arrays embedded inside larger objects, single
strings instead of objects, missing optional fields.

`parse_plan_response` returns a `Plan` for *any* shape, falling back to a
single-subgoal plan with the raw user goal when output is unsalvageable.
Failure is never fatal — the planner can always execute the unmodified goal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Subgoal:
    goal: str
    success_criteria: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("Subgoal.goal must be a non-empty string")


@dataclass
class Plan:
    subgoals: list[Subgoal] = field(default_factory=list)
    raw: str = ""
    fallback: bool = False

    def __len__(self) -> int:
        return len(self.subgoals)

    def __iter__(self):
        return iter(self.subgoals)

    def is_empty(self) -> bool:
        return not self.subgoals


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _extract_json_blob(text: str) -> str | None:
    """Pull the first plausible JSON value out of `text`.

    Tries: fenced ```json block → first balanced `[...]` → first balanced `{...}`.
    Returns the substring (still un-parsed) or None.
    """
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start = text.find(open_c)
        if start < 0:
            continue
        depth = 0
        in_str: str | None = None
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ('"', "'"):
                in_str = ch
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _coerce_subgoal(item: Any) -> Subgoal | None:
    if isinstance(item, str):
        s = item.strip()
        return Subgoal(goal=s) if s else None
    if isinstance(item, dict):
        goal = item.get("goal") or item.get("task") or item.get("step") or item.get("description")
        if not isinstance(goal, str) or not goal.strip():
            return None
        crit = item.get("success_criteria") or item.get("done_when") or item.get("success") or ""
        if not isinstance(crit, str):
            crit = str(crit)
        return Subgoal(goal=goal.strip(), success_criteria=crit.strip())
    return None


def parse_plan_response(raw: str, fallback_goal: str) -> Plan:
    """Best-effort parse of a planner-LLM response into a Plan.

    Always returns a Plan with at least one subgoal — falls back to
    `[Subgoal(goal=fallback_goal)]` when output can't be salvaged. The
    `fallback` flag lets the orchestrator know to log the degradation.
    """
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    blob = _extract_json_blob(raw)
    parsed: Any = None
    if blob is not None:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict):
        for key in ("plan", "subgoals", "steps", "tasks"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
    if isinstance(parsed, list):
        subs: list[Subgoal] = []
        for item in parsed:
            sg = _coerce_subgoal(item)
            if sg is not None:
                subs.append(sg)
        if subs:
            return Plan(subgoals=subs, raw=raw)
    # Last resort: split on newlines for a numbered/bulleted prose list.
    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$")
    bullets = [bullet_re.match(line) for line in raw.splitlines()]
    bullet_subs = [Subgoal(goal=m.group(1)) for m in bullets if m and m.group(1).strip()]
    if len(bullet_subs) >= 2:
        return Plan(subgoals=bullet_subs, raw=raw)
    return Plan(subgoals=[Subgoal(goal=fallback_goal.strip() or "complete the user goal")], raw=raw, fallback=True)
