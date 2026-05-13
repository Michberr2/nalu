"""LLMDecomposer — text-only MLX-LM wrapper for goal decomposition + replanning.

Lazy MLX import (test code never triggers a model download) and a threading
lock so concurrent calls from the daemon (decompose + replan racing on the
same task) queue safely. Generation is intentionally tight: low max_tokens,
deterministic decoding, short stop set — we want a JSON array out, not a
chatbot turn.
"""
from __future__ import annotations

import threading
from typing import Callable

import structlog

from ... import config
from .prompts import (
    DECOMPOSE_SYSTEM,
    REPLAN_SYSTEM,
    build_decompose_user_prompt,
    build_replan_user_prompt,
)
from .subgoal import Plan, parse_plan_response


log = structlog.get_logger("planner_llm")


GenerateFn = Callable[[str, str], str]


class LLMDecomposer:
    """Wraps a local MLX-LM chat model for decompose/replan calls.

    Loads on first use (model download / quantize happens once via mlx-lm,
    cached on disk like the vision model). The `_generate` hook is exposed
    so tests can monkeypatch the entire MLX path away.
    """

    def __init__(self, model_id: str | None = None, max_tokens: int = 512) -> None:
        self.model_id = model_id or config.PLANNER_LLM_MODEL
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_id)

    def _generate(self, system: str, user: str) -> str:
        """Run a single chat turn. Override-able for tests."""
        self.load()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        with self._lock:
            chat = self._tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            sampler = make_sampler(temp=0.0)
            return generate(
                self._model,
                self._tokenizer,
                prompt=chat,
                max_tokens=self.max_tokens,
                sampler=sampler,
                verbose=False,
            )

    def decompose(self, goal: str, conversation: str = "") -> Plan:
        """Decompose `goal` into a Plan. Always returns a non-empty Plan."""
        goal = (goal or "").strip()
        if not goal:
            return Plan(subgoals=[], raw="", fallback=True)
        user = build_decompose_user_prompt(goal, conversation)
        try:
            raw = self._generate(DECOMPOSE_SYSTEM, user)
        except Exception as e:
            log.exception("decompose_generate_failed", error=str(e))
            return parse_plan_response("", fallback_goal=goal)
        plan = parse_plan_response(raw, fallback_goal=goal)
        if plan.fallback:
            log.info("decompose_fallback", goal=goal, raw_head=raw[:200])
        else:
            log.info("decompose_ok", goal=goal, n=len(plan))
        return plan

    def replan(
        self,
        *,
        original_goal: str,
        completed: list[str],
        failed_subgoal: str,
        failure_reason: str,
        screen_summary: str = "",
    ) -> Plan:
        """Generate a corrected continuation after a subgoal failure."""
        user = build_replan_user_prompt(
            original_goal=original_goal,
            completed=completed,
            failed_subgoal=failed_subgoal,
            failure_reason=failure_reason,
            screen_summary=screen_summary,
        )
        try:
            raw = self._generate(REPLAN_SYSTEM, user)
        except Exception as e:
            log.exception("replan_generate_failed", error=str(e))
            return parse_plan_response("", fallback_goal=failed_subgoal)
        plan = parse_plan_response(raw, fallback_goal=failed_subgoal)
        if plan.fallback:
            log.info("replan_fallback", failed_subgoal=failed_subgoal, raw_head=raw[:200])
        else:
            log.info("replan_ok", failed_subgoal=failed_subgoal, n=len(plan))
        return plan
