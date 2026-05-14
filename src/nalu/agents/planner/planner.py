from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import structlog

from ... import config
from ...actuator import Actuator, ActionRefused, PauseController
from ...bus import BusClient, Event
from ...capture import ContinuousCapture, capture_main_display
from ..planner_llm import LLMDecomposer, Plan, Subgoal
from ..vision import Action, VisionAgent
from .annotate import draw_action_marker
from .history import compact_history
from .jitter import jitter_click_args
from .loops import LoopDetector
from .screen_change import evaluate_action_effect
from .settle import wait_for_screen_settle
from .validate import validate_action
from .verifier import JudgeCallable, verify_completion


POST_ACTION_FALLBACK_SLEEP_S = 0.4


CONVERSATION_TURNS_FOR_PLANNER = 6
STUCK_GIVE_UP_AFTER = 2  # consecutive signals for the same signature
MAX_RECOVERIES_PER_TASK = 1  # one auto-retry on stuck/dispatch before failing for real
MAX_PARSE_RETRIES_PER_TASK = 2  # small models often emit a description on the first vision turn — retry with a self-correction nudge


def conversation_snapshot(
    conversation: Iterable[dict] | None, before_ts: float, max_turns: int = CONVERSATION_TURNS_FOR_PLANNER,
) -> list[dict]:
    """Return up to `max_turns` most-recent conversation turns strictly older than `before_ts`."""
    if conversation is None:
        return []
    prior = [t for t in conversation if t.get("ts", 0) < before_ts]
    return prior[-max_turns:]


def format_conversation(turns: list[dict]) -> str:
    if not turns:
        return ""
    lines = []
    for t in turns:
        role = t.get("role", "user")
        text = (t.get("text") or "").strip()
        if not text:
            continue
        prefix = "User" if role == "user" else "Nalu"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


log = structlog.get_logger("planner")


@dataclass
class SubgoalOutcome:
    """Result of running one subgoal through the perceive-reason-act loop."""

    status: str = "unknown"  # "completed" | "failed"
    reason: str = ""
    steps_used: int = 0
    answer: str = ""
    recoveries_used: int = 0


@dataclass
class _LoopContext:
    """Mutable state threaded through the per-subgoal loop. Keeps the signature manageable."""

    run_dir: Path
    actions_log: Any
    history: list[str]
    deadline: float
    step_offset: int = 0  # global step counter across multi-subgoal plans
    prev_shot: Any = None
    prev_action_kind: str | None = None
    prev_action: Action | None = None
    jittered_for_step: int | None = None
    parse_retries_used: int = 0


class Planner:
    def __init__(
        self,
        bus: BusClient,
        actuator: Actuator,
        vision: VisionAgent,
        pause: PauseController,
        capture: ContinuousCapture | None = None,
        conversation: deque | None = None,
        loop_detector: LoopDetector | None = None,
        judge: JudgeCallable | None = None,
        decomposer: LLMDecomposer | None = None,
    ):
        self.bus = bus
        self.actuator = actuator
        self.vision = vision
        self.pause = pause
        self.capture = capture
        self.conversation = conversation
        self.loop_detector = loop_detector if loop_detector is not None else LoopDetector()
        self.judge = judge if judge is not None else getattr(vision, "judge", None)
        self.decomposer = decomposer

    async def run(self) -> None:
        await self.bus.subscribe("user_intent", self._on_intent)
        await self.bus.publish("planner_ready", {"ts": time.time()})

    async def _on_intent(self, ev: Event) -> None:
        goal = ev.payload.get("text", "").strip()
        if not goal:
            return
        prior_turns = conversation_snapshot(self.conversation, before_ts=ev.ts)
        conv_text = format_conversation(prior_turns)
        run_dir = config.new_run_dir()
        started_ts = time.time()
        meta: dict[str, Any] = {
            "goal": goal,
            "started_ts": started_ts,
            "via": ev.payload.get("via", ""),
            "conversation": prior_turns,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta))
        await self.bus.publish("task_started", {"goal": goal, "run_dir": str(run_dir)})

        plan = await self._make_plan(goal, conv_text, run_dir)
        await self.bus.publish(
            "plan_decomposed",
            {
                "goal": goal,
                "subgoals": [{"goal": sg.goal, "success_criteria": sg.success_criteria} for sg in plan],
                "fallback": plan.fallback,
            },
        )

        actions_log = (run_dir / "actions.jsonl").open("a")
        history: list[str] = []
        task_deadline = time.time() + config.PLANNER_TASK_TIMEOUT_S
        per_subgoal_cap = (
            config.PLANNER_SUBGOAL_MAX_STEPS if len(plan) > 1 else config.PLANNER_MAX_STEPS
        )
        subgoal_outcomes: list[dict[str, Any]] = []
        ctx = _LoopContext(
            run_dir=run_dir,
            actions_log=actions_log,
            history=history,
            deadline=task_deadline,
        )
        final_answer = ""
        final_status = "unknown"
        final_reason = ""
        # Queue of remaining (original_subgoal, source) pairs. Replan appends to this.
        queue: list[tuple[Subgoal, str]] = [(sg, "plan") for sg in plan]
        replans_used = 0
        completed_goals: list[str] = []
        try:
            i = 0
            while queue:
                subgoal, _source = queue.pop(0)
                effective_goal = self._compose_effective_goal(subgoal, subgoal_outcomes)
                await self.bus.publish(
                    "subgoal_started",
                    {"index": i, "remaining": len(queue), "goal": effective_goal},
                )
                self.loop_detector.reset()
                outcome = await self._run_subgoal_loop(
                    effective_goal, conv_text, per_subgoal_cap, ctx
                )
                subgoal_outcomes.append(
                    {
                        "index": i,
                        "goal": subgoal.goal,
                        "effective_goal": effective_goal,
                        "status": outcome.status,
                        "reason": outcome.reason,
                        "steps": outcome.steps_used,
                        "answer": outcome.answer,
                    }
                )
                if outcome.status == "completed":
                    await self.bus.publish(
                        "subgoal_completed",
                        {"index": i, "answer": outcome.answer, "steps": outcome.steps_used},
                    )
                    final_answer = outcome.answer or final_answer
                    completed_goals.append(subgoal.goal)
                    i += 1
                    continue
                await self.bus.publish(
                    "subgoal_failed",
                    {"index": i, "reason": outcome.reason, "steps": outcome.steps_used},
                )
                if (
                    self.decomposer is not None
                    and config.USE_LLM_PLANNER
                    and replans_used < config.PLANNER_MAX_REPLANS
                    and outcome.reason != "timeout"
                ):
                    replans_used += 1
                    new_plan = await self._do_replan(
                        original_goal=goal,
                        completed=completed_goals,
                        failed_subgoal=subgoal.goal,
                        failure_reason=outcome.reason,
                    )
                    await self.bus.publish(
                        "plan_replanned",
                        {
                            "replans_used": replans_used,
                            "failed_subgoal": subgoal.goal,
                            "subgoals": [
                                {"goal": sg.goal, "success_criteria": sg.success_criteria}
                                for sg in new_plan
                            ],
                            "fallback": new_plan.fallback,
                        },
                    )
                    queue = [(sg, "replan") for sg in new_plan] + queue
                    i += 1
                    continue
                final_status = "failed"
                final_reason = f"subgoal[{i}]:{outcome.reason}"
                break
            else:
                final_status = "completed"
            if final_status == "unknown":
                final_status = "completed"
        finally:
            actions_log.close()
            if final_status == "unknown":
                final_status = "failed"
                final_reason = final_reason or "incomplete"
            if final_status == "completed":
                await self.bus.publish("task_completed", {"answer": final_answer, "subgoals": len(plan)})
            else:
                await self.bus.publish("task_failed", {"reason": final_reason})
            try:
                meta_path = run_dir / "meta.json"
                meta = json.loads(meta_path.read_text())
                meta.update(
                    {
                        "status": final_status,
                        "reason": final_reason,
                        "answer": final_answer,
                        "steps": ctx.step_offset,
                        "ended_ts": time.time(),
                        "subgoals": subgoal_outcomes,
                        "plan_fallback": plan.fallback,
                        "replans_used": replans_used,
                    }
                )
                meta_path.write_text(json.dumps(meta))
            except Exception:
                log.exception("meta_finalize_failed")

    async def _make_plan(self, goal: str, conversation_text: str, run_dir: Path) -> Plan:
        """Decompose `goal` into subgoals if a decomposer is wired; otherwise return a one-item plan."""
        if self.decomposer is None or not config.USE_LLM_PLANNER:
            return Plan(subgoals=[Subgoal(goal=goal)])
        try:
            plan = await asyncio.to_thread(self.decomposer.decompose, goal, conversation_text)
        except Exception:
            log.exception("decompose_failed")
            return Plan(subgoals=[Subgoal(goal=goal)], fallback=True)
        if plan.is_empty():
            return Plan(subgoals=[Subgoal(goal=goal)], fallback=True)
        try:
            (run_dir / "plan.json").write_text(
                json.dumps(
                    {
                        "goal": goal,
                        "fallback": plan.fallback,
                        "raw": plan.raw,
                        "subgoals": [
                            {"goal": sg.goal, "success_criteria": sg.success_criteria} for sg in plan
                        ],
                    },
                    indent=2,
                )
            )
        except Exception:
            log.exception("plan_persist_failed")
        return plan

    async def _do_replan(
        self,
        *,
        original_goal: str,
        completed: list[str],
        failed_subgoal: str,
        failure_reason: str,
    ) -> Plan:
        try:
            new_plan = await asyncio.to_thread(
                self.decomposer.replan,
                original_goal=original_goal,
                completed=completed,
                failed_subgoal=failed_subgoal,
                failure_reason=failure_reason,
            )
        except Exception:
            log.exception("replan_failed")
            return Plan(subgoals=[Subgoal(goal=failed_subgoal)], fallback=True)
        if new_plan.is_empty():
            return Plan(subgoals=[Subgoal(goal=failed_subgoal)], fallback=True)
        return new_plan

    def _compose_effective_goal(self, subgoal: Subgoal, prior_outcomes: list[dict]) -> str:
        """Prepend any answers carried from earlier subgoals to give vision the context it needs."""
        last_answer = next(
            (o.get("answer") for o in reversed(prior_outcomes) if o.get("answer")),
            "",
        )
        if last_answer:
            return f"{subgoal.goal}\n\n[Carried context from previous subgoal: {last_answer}]"
        return subgoal.goal

    async def _run_subgoal_loop(
        self,
        goal: str,
        conv_text: str,
        step_cap: int,
        ctx: _LoopContext,
    ) -> SubgoalOutcome:
        """Execute one subgoal through the perceive→reason→act loop. Returns the outcome."""
        outcome = SubgoalOutcome()
        recoveries_used = 0
        for local_step in range(step_cap):
            step = ctx.step_offset + local_step
            if time.time() > ctx.deadline:
                outcome.status = "failed"
                outcome.reason = "timeout"
                outcome.steps_used = local_step
                outcome.recoveries_used = recoveries_used
                break

            shot = self.capture.latest_frame() if self.capture else None
            if shot is None:
                shot = capture_main_display()
            shot.image.save(ctx.run_dir / f"step_{step:03d}.jpg", quality=70)

            if ctx.prev_shot is not None and ctx.prev_action_kind is not None:
                no_effect = evaluate_action_effect(
                    ctx.prev_action_kind, ctx.prev_shot.image, shot.image
                )
                if no_effect is not None:
                    log.info("action_no_effect", kind=ctx.prev_action_kind, diff=no_effect.diff)
                    await self.bus.publish(
                        "action_no_effect",
                        {"kind": ctx.prev_action_kind, "diff": no_effect.diff, "step": step},
                    )
                    ctx.history.append(f"step {step - 1}: NO EFFECT -- {no_effect.hint}")

                    if (
                        ctx.prev_action is not None
                        and ctx.prev_action_kind in ("click", "double_click")
                        and ctx.jittered_for_step != (step - 1)
                        and "x" in ctx.prev_action.args
                        and "y" in ctx.prev_action.args
                    ):
                        jittered_args = jitter_click_args(
                            ctx.prev_action.args, shot.image.width, shot.image.height
                        )
                        jit = Action(
                            kind=ctx.prev_action_kind,
                            args=jittered_args,
                            reason=(
                                f"jitter retry: ({ctx.prev_action.args.get('x')},"
                                f"{ctx.prev_action.args.get('y')}) had no effect, retrying nearby"
                            ),
                        )
                        rec = {
                            "step": step,
                            "action": jit.kind,
                            "args": jit.args,
                            "reason": jit.reason,
                            "ts": time.time(),
                            "synthetic": "jitter",
                        }
                        ctx.actions_log.write(json.dumps(rec) + "\n")
                        ctx.actions_log.flush()
                        await self.bus.publish("action_decided", rec)
                        await self.bus.publish(
                            "action_jittered",
                            {
                                "step": step,
                                "from": {
                                    "x": ctx.prev_action.args.get("x"),
                                    "y": ctx.prev_action.args.get("y"),
                                },
                                "to": {"x": jittered_args.get("x"), "y": jittered_args.get("y")},
                            },
                        )
                        try:
                            dispatch_action(jit, shot, self.actuator)
                        except ActionRefused as e:
                            await self.bus.publish("task_paused", {"reason": str(e), "step": step})
                            while self.pause.paused:
                                await asyncio.sleep(0.2)
                            continue
                        except Exception as e:
                            log.exception("jitter_dispatch_failed")
                            ctx.history.append(f"step {step}: JITTERED RETRY FAILED -- {e}")
                        else:
                            ctx.history.append(
                                f"step {step}: JITTERED RETRY -- "
                                f"({ctx.prev_action.args['x']},{ctx.prev_action.args['y']}) had no effect; "
                                f"retried at ({jittered_args['x']},{jittered_args['y']})"
                            )
                        ctx.jittered_for_step = step
                        ctx.prev_shot = shot
                        ctx.prev_action = jit
                        ctx.prev_action_kind = jit.kind
                        await self._settle_after_action(step)
                        continue

            try:
                action: Action = await asyncio.to_thread(
                    self.vision.decide, shot.image, goal, compact_history(ctx.history), conv_text
                )
            except Exception as e:
                outcome.status = "failed"
                outcome.reason = f"vision: {e}"
                outcome.steps_used = local_step
                outcome.recoveries_used = recoveries_used
                log.exception("vision_failed")
                break

            rec = {
                "step": step,
                "action": action.kind,
                "args": action.args,
                "reason": action.reason,
                "ts": time.time(),
            }
            ctx.actions_log.write(json.dumps(rec) + "\n")
            ctx.actions_log.flush()
            await self.bus.publish("action_decided", rec)

            try:
                annotated = draw_action_marker(shot.image, action.kind, action.args)
                annotated.convert("RGB").save(ctx.run_dir / f"step_{step:03d}_decided.jpg", quality=70)
            except Exception:
                log.exception("annotation_failed")

            if action.kind == "done":
                answer = action.args.get("answer", "")
                if self.judge is not None:
                    verify = await asyncio.to_thread(
                        verify_completion, self.judge, shot.image, goal, answer
                    )
                    await self.bus.publish(
                        "completion_verified" if verify.confirmed else "completion_denied",
                        {"step": step, "answer": answer, "reasoning": verify.reasoning},
                    )
                    if not verify.confirmed:
                        log.warning("completion_denied", reasoning=verify.reasoning)
                        ctx.history.append(
                            f"step {step}: VERIFICATION DENIED -- claimed answer '{answer}' but the verifier said: {verify.reasoning}. Continue working on the goal."
                        )
                        await asyncio.sleep(0.4)
                        continue
                outcome.status = "completed"
                outcome.steps_used = local_step + 1
                outcome.answer = answer
                outcome.recoveries_used = recoveries_used
                break

            if action.kind == "error":
                is_descriptive = isinstance(action.reason, str) and action.reason.startswith(
                    "unparseable model output"
                )
                if is_descriptive and ctx.parse_retries_used < MAX_PARSE_RETRIES_PER_TASK:
                    ctx.parse_retries_used += 1
                    log.info("parse_retry", attempt=ctx.parse_retries_used, reason=action.reason)
                    await self.bus.publish(
                        "parse_retried",
                        {"step": step, "attempt": ctx.parse_retries_used, "reason": action.reason},
                    )
                    ctx.history.append(
                        f"step {step}: PARSE RETRY -- previous response was prose, not an action. "
                        f"Emit a single action like `click(x=…, y=…)`, `type(text=\"…\")`, "
                        f"`key(name=\"…\", modifiers=[…])`, `scroll(dx=…, dy=…)`, or `done(answer=\"…\")`."
                    )
                    await asyncio.sleep(0.2)
                    continue
                outcome.status = "failed"
                outcome.reason = f"parse: {action.reason}"
                outcome.steps_used = local_step
                outcome.recoveries_used = recoveries_used
                log.warning("parse_error", reason=action.reason)
                break

            refusal = validate_action(action.kind, action.args, shot.image.width, shot.image.height)
            if refusal is not None:
                log.warning("action_refused", reason=refusal.reason, hint=refusal.hint)
                await self.bus.publish(
                    "action_refused",
                    {"reason": refusal.reason, "hint": refusal.hint, "step": step},
                )
                ctx.history.append(f"step {step}: REFUSED -- {refusal.hint}")
                await asyncio.sleep(0.4)
                continue

            signal = self.loop_detector.observe(action.kind, action.args)
            if signal is not None:
                if signal.count >= STUCK_GIVE_UP_AFTER:
                    if recoveries_used < MAX_RECOVERIES_PER_TASK:
                        recoveries_used += 1
                        log.warning("task_recovering", reason=f"stuck:{signal.reason}")
                        await self.bus.publish(
                            "task_recovering",
                            {"reason": f"stuck:{signal.reason}", "step": step, "hint": signal.hint},
                        )
                        ctx.history.append(
                            f"step {step}: RECOVERY -- previous attempt got stuck "
                            f"({signal.reason}: {signal.hint}). Try a different approach."
                        )
                        self.loop_detector.reset()
                        await asyncio.sleep(0.4)
                        continue
                    outcome.status = "failed"
                    outcome.reason = f"stuck:{signal.reason}"
                    outcome.steps_used = local_step
                    outcome.recoveries_used = recoveries_used
                    break
                log.warning("stuck_detected", reason=signal.reason, hint=signal.hint)
                await self.bus.publish(
                    "stuck_detected",
                    {"reason": signal.reason, "step": step, "hint": signal.hint},
                )
                ctx.history.append(f"step {step}: SKIPPED -- {signal.hint}")
                await asyncio.sleep(0.4)
                continue

            try:
                self._dispatch(action, shot)
            except ActionRefused as e:
                await self.bus.publish("task_paused", {"reason": str(e), "step": step})
                while self.pause.paused:
                    await asyncio.sleep(0.2)
                continue
            except Exception as e:
                log.exception("dispatch_failed")
                if recoveries_used < MAX_RECOVERIES_PER_TASK:
                    recoveries_used += 1
                    await self.bus.publish(
                        "task_recovering", {"reason": f"dispatch: {e}", "step": step}
                    )
                    ctx.history.append(
                        f"step {step}: RECOVERY -- previous attempt failed to dispatch "
                        f"({action.kind} {action.args}): {e}. Try a different approach."
                    )
                    self.loop_detector.reset()
                    await asyncio.sleep(0.4)
                    continue
                outcome.status = "failed"
                outcome.reason = f"dispatch: {e}"
                outcome.steps_used = local_step
                outcome.recoveries_used = recoveries_used
                break

            ctx.history.append(f"step {step}: {action.kind} {action.args} -- {action.reason}")
            ctx.prev_shot = shot
            ctx.prev_action_kind = action.kind
            ctx.prev_action = action
            await self._settle_after_action(step)
        else:
            outcome.status = "failed"
            outcome.reason = "max_steps_exceeded"
            outcome.steps_used = step_cap
            outcome.recoveries_used = recoveries_used

        ctx.step_offset += outcome.steps_used
        return outcome

    def _dispatch(self, action: Action, shot) -> None:
        dispatch_action(action, shot, self.actuator)

    async def _settle_after_action(self, step: int) -> None:
        """Active wait for the screen to stabilize after dispatching an action.

        Publishes a `screen_settled` bus event so dashboards / evals can see
        per-step latency. Falls back to a fixed sleep if continuous capture
        isn't wired (e.g. unit tests, headless runs).
        """
        if self.capture is None:
            await asyncio.sleep(POST_ACTION_FALLBACK_SLEEP_S)
            return

        def _frame_getter():
            shot = self.capture.latest_frame()
            return shot.image if shot is not None else None

        result = await wait_for_screen_settle(_frame_getter)
        await self.bus.publish(
            "screen_settled",
            {
                "step": step,
                "elapsed_ms": int(result.elapsed_s * 1000),
                "polls": result.polls,
                "last_diff": result.last_diff,
                "stable": result.stable,
            },
        )


def dispatch_action(action: Action, shot, actuator) -> None:
    """Translate a vision Action into actuator calls, scaling click coords from
    captured-image space to display space via `shot.scale_x` / `shot.scale_y`.

    Free function so the routing logic is unit-testable with a fake actuator
    without spinning up the full Planner / bus / capture stack.
    """
    kind = action.kind
    a = action.args
    if kind == "click":
        x = int(a["x"] * shot.scale_x)
        y = int(a["y"] * shot.scale_y)
        actuator.click(x, y, button=a.get("button", "left"), clicks=a.get("clicks", 1))
    elif kind == "double_click":
        x = int(a["x"] * shot.scale_x)
        y = int(a["y"] * shot.scale_y)
        actuator.click(x, y, button=a.get("button", "left"), clicks=2)
    elif kind == "drag":
        x1 = int(a["x1"] * shot.scale_x)
        y1 = int(a["y1"] * shot.scale_y)
        x2 = int(a["x2"] * shot.scale_x)
        y2 = int(a["y2"] * shot.scale_y)
        actuator.drag(x1, y1, x2, y2)
    elif kind == "type":
        actuator.type_text(str(a["text"]))
    elif kind == "key":
        actuator.key(a["name"], modifiers=a.get("modifiers", []))
    elif kind == "scroll":
        actuator.scroll(int(a.get("dx", 0)), int(a.get("dy", 0)))
    elif kind == "wait":
        time.sleep(min(int(a.get("ms", 200)), 5000) / 1000.0)
    elif kind == "error":
        raise RuntimeError(action.reason)
    else:
        raise ValueError(f"unknown action: {kind}")
