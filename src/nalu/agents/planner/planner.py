from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import structlog

from ... import config
from ...actuator import Actuator, ActionRefused, PauseController
from ...bus import BusClient, Event
from ...capture import capture_main_display
from ..vision import Action, VisionAgent

log = structlog.get_logger("planner")


class Planner:
    def __init__(self, bus: BusClient, actuator: Actuator, vision: VisionAgent, pause: PauseController):
        self.bus = bus
        self.actuator = actuator
        self.vision = vision
        self.pause = pause

    async def run(self) -> None:
        await self.bus.subscribe("user_intent", self._on_intent)
        await self.bus.publish("planner_ready", {"ts": time.time()})

    async def _on_intent(self, ev: Event) -> None:
        goal = ev.payload.get("text", "").strip()
        if not goal:
            return
        run_dir = config.new_run_dir()
        await self.bus.publish("task_started", {"goal": goal, "run_dir": str(run_dir)})
        actions_log = (run_dir / "actions.jsonl").open("a")
        history: list[str] = []
        deadline = time.time() + config.PLANNER_TASK_TIMEOUT_S

        try:
            for step in range(config.PLANNER_MAX_STEPS):
                if time.time() > deadline:
                    await self.bus.publish("task_failed", {"reason": "timeout", "step": step})
                    break

                shot = capture_main_display()
                shot.image.save(run_dir / f"step_{step:03d}.jpg", quality=70)

                try:
                    action: Action = await asyncio.to_thread(self.vision.decide, shot.image, goal, history)
                except Exception as e:
                    log.exception("vision_failed")
                    await self.bus.publish("task_failed", {"reason": f"vision: {e}", "step": step})
                    break

                rec = {"step": step, "action": action.kind, "args": action.args, "reason": action.reason, "ts": time.time()}
                actions_log.write(json.dumps(rec) + "\n")
                actions_log.flush()
                await self.bus.publish("action_decided", rec)

                if action.kind == "done":
                    await self.bus.publish("task_completed", {"answer": action.args.get("answer", ""), "steps": step + 1})
                    break

                try:
                    self._dispatch(action, shot)
                except ActionRefused as e:
                    await self.bus.publish("task_paused", {"reason": str(e), "step": step})
                    while self.pause.paused:
                        await asyncio.sleep(0.2)
                    continue
                except Exception as e:
                    log.exception("dispatch_failed")
                    await self.bus.publish("task_failed", {"reason": f"dispatch: {e}", "step": step})
                    break

                history.append(f"step {step}: {action.kind} {action.args} -- {action.reason}")
                await asyncio.sleep(0.4)
            else:
                await self.bus.publish("task_failed", {"reason": "max_steps_exceeded"})
        finally:
            actions_log.close()

    def _dispatch(self, action: Action, shot) -> None:
        kind = action.kind
        a = action.args
        if kind == "click":
            x = int(a["x"] * shot.scale_x)
            y = int(a["y"] * shot.scale_y)
            self.actuator.click(x, y, button=a.get("button", "left"), clicks=a.get("clicks", 1))
        elif kind == "type":
            self.actuator.type_text(str(a["text"]))
        elif kind == "key":
            self.actuator.key(a["name"], modifiers=a.get("modifiers", []))
        elif kind == "scroll":
            self.actuator.scroll(int(a.get("dx", 0)), int(a.get("dy", 0)))
        elif kind == "wait":
            time.sleep(min(int(a.get("ms", 200)), 5000) / 1000.0)
        elif kind == "error":
            raise RuntimeError(action.reason)
        else:
            raise ValueError(f"unknown action: {kind}")
