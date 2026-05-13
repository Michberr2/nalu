"""System + user prompts for the planner LLM.

Kept separate from the decomposer so the prompts can be unit-tested without
loading MLX. The decomposer prompt asks for a short JSON array of subgoals;
the replanner prompt is fed the original goal, the failure reason, and what's
already been done so it can produce a corrected continuation.

We deliberately constrain output: 1–6 subgoals, single-screen each. UI-TARS's
planning horizon is ~4 steps per subgoal; longer subgoals reliably loop.
"""
from __future__ import annotations

from .subgoal import Plan


DECOMPOSE_SYSTEM = """You are Nalu's task planner. You break a user's high-level instruction into a sequence of small subgoals that a separate vision-grounding agent (UI-TARS) will execute one at a time.

## Rules
- Return ONLY a JSON array. No prose before or after.
- Each item is an object with `goal` (string, imperative) and optional `success_criteria` (string).
- Each subgoal must be a single visible-screen action sequence (≤4 UI steps): one app to focus, one panel to navigate, one value to read or enter.
- Use 1–6 subgoals total. Prefer fewer.
- Reference apps and elements by their visible name; do not invent UI affordances.
- If the user's goal is already a single small step, return a one-item array with the goal verbatim.
- If the user wants a value computed or retrieved, the LAST subgoal should be: "Report the value <description> as the final answer."

## Output format
[
  {"goal": "Open Safari", "success_criteria": "Safari window is frontmost"},
  {"goal": "Navigate to weather.com and search for Paris", "success_criteria": "Paris current conditions page is visible"},
  {"goal": "Report the current Paris temperature and conditions as the final answer."}
]
"""


REPLAN_SYSTEM = """You are Nalu's task replanner. A previous plan partially executed and one subgoal failed. Produce a corrected continuation that achieves the user's original goal from the current state.

## Rules
- Return ONLY a JSON array of subgoals (same schema as planning).
- The first subgoal should address the failure (e.g. close a modal, switch apps, scroll into view).
- Do not repeat subgoals that already succeeded.
- ≤4 subgoals.
"""


def build_decompose_user_prompt(goal: str, conversation: str = "") -> str:
    conv_block = f"## Recent conversation\n{conversation.strip()}\n\n" if conversation and conversation.strip() else ""
    return f"{conv_block}## User goal\n{goal.strip()}\n\nProduce the JSON plan."


def build_replan_user_prompt(
    *,
    original_goal: str,
    completed: list[str],
    failed_subgoal: str,
    failure_reason: str,
    screen_summary: str = "",
) -> str:
    completed_block = "\n".join(f"- {c}" for c in completed) if completed else "(none)"
    screen_block = f"\n## Current screen\n{screen_summary.strip()}\n" if screen_summary.strip() else ""
    return (
        f"## Original user goal\n{original_goal.strip()}\n\n"
        f"## Already completed\n{completed_block}\n\n"
        f"## Failed subgoal\n{failed_subgoal.strip()}\n\n"
        f"## Failure reason\n{failure_reason.strip()}\n"
        f"{screen_block}\n"
        f"Produce a corrected JSON plan continuation."
    )


def format_plan_for_log(plan: Plan) -> str:
    """Compact one-line-per-subgoal form for log output / debugging."""
    return "\n".join(f"  {i+1}. {sg.goal}" for i, sg in enumerate(plan.subgoals))
