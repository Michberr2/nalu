"""LLM planner layer: decomposes a user goal into a sequence of vision-actionable subgoals.

UI-TARS-1.5-7B is a GUI-grounding specialist — given a clear subgoal that fits in
its planning horizon (~4 steps), it's fast and accurate. It is *not* a general
reasoning model. Goals like "research the Paris weather and save it to a note"
exceed that horizon and the vision agent loops until step-cap.

This module sits in front: a separate text-only LLM (default
`mlx-community/Qwen2.5-7B-Instruct-4bit`, Apache 2.0) decomposes the goal into
discrete vision subgoals, the existing planner runs each one to completion, and
results stream forward as state.

Pure-Python parsing + dataclasses live here so the test path doesn't need MLX-LM.
"""

from .decomposer import LLMDecomposer
from .subgoal import Plan, Subgoal, parse_plan_response

__all__ = ["LLMDecomposer", "Plan", "Subgoal", "parse_plan_response"]
