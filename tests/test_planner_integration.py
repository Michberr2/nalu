"""End-to-end planner tests with stubbed vision + decomposer + bus.

Covers the multi-subgoal flow added in Phase 6: decompose → run each subgoal →
state passing → replan on failure → final task_completed/task_failed.
No MLX models are loaded; all dependencies are stubbed.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
from PIL import Image

from nalu.agents.planner.loops import LoopDetector
from nalu.agents.planner.planner import Planner
from nalu.agents.planner_llm import Plan, Subgoal
from nalu.agents.vision import Action
from nalu.bus.bus import Event


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self._subs: dict[str, list[Callable]] = {}

    async def subscribe(self, topic: str, fn) -> None:
        self._subs.setdefault(topic, []).append(fn)

    async def publish(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))
        for fn in self._subs.get(topic, []):
            await fn(Event(topic=topic, payload=payload))

    def topics(self) -> list[str]:
        return [t for t, _ in self.events]

    def by_topic(self, topic: str) -> list[dict]:
        return [p for t, p in self.events if t == topic]


@dataclass
class FakeShot:
    image: Image.Image = field(default_factory=lambda: Image.new("RGB", (100, 80), "white"))
    scale_x: float = 1.0
    scale_y: float = 1.0


class FakeCapture:
    def __init__(self) -> None:
        self.shot = FakeShot()

    def latest_frame(self) -> FakeShot:
        return self.shot


class FakeActuator:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y))

    def type_text(self, text):
        self.calls.append(("type", text))

    def key(self, name, modifiers=()):
        self.calls.append(("key", name, tuple(modifiers)))

    def scroll(self, dx, dy):
        self.calls.append(("scroll", dx, dy))

    def drag(self, x1, y1, x2, y2):
        self.calls.append(("drag", x1, y1, x2, y2))


class FakePause:
    paused = False


class ScriptedVision:
    """Returns one canned Action per call. Records the goal each call saw so tests can
    verify state passing across subgoals."""

    def __init__(self, script: list[Action]) -> None:
        self.script = list(script)
        self.goals_seen: list[str] = []

    def decide(self, image, goal, history, conversation):
        self.goals_seen.append(goal)
        if not self.script:
            return Action(kind="error", reason="ran off the end of the test script")
        return self.script.pop(0)


class StubDecomposer:
    """Returns canned Plans for decompose / replan."""

    def __init__(self, decompose_plan: Plan, replan_plan: Plan | None = None) -> None:
        self.decompose_plan = decompose_plan
        self.replan_plan = replan_plan
        self.decompose_calls: list[str] = []
        self.replan_calls: list[dict] = []

    def decompose(self, goal: str, conversation: str = "") -> Plan:
        self.decompose_calls.append(goal)
        return self.decompose_plan

    def replan(self, **kw) -> Plan:
        self.replan_calls.append(kw)
        return self.replan_plan if self.replan_plan is not None else Plan(
            subgoals=[Subgoal(goal=kw["failed_subgoal"])], fallback=True
        )


@pytest.fixture
def _patched_planner(tmp_path, monkeypatch):
    """Builds a Planner with fake everything + redirects runs_dir into tmp."""
    from nalu import config
    from nalu.agents.planner import planner as planner_mod

    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(config, "USE_LLM_PLANNER", True)
    monkeypatch.setattr(config, "PLANNER_TASK_TIMEOUT_S", 60)
    monkeypatch.setattr(config, "PLANNER_MAX_STEPS", 10)
    monkeypatch.setattr(config, "PLANNER_SUBGOAL_MAX_STEPS", 4)
    monkeypatch.setattr(config, "PLANNER_MAX_REPLANS", 1)

    counter = {"n": 0}

    def new_run_dir():
        counter["n"] += 1
        d = tmp_path / f"run_{counter['n']:03d}"
        d.mkdir()
        return d

    monkeypatch.setattr(config, "new_run_dir", new_run_dir)
    monkeypatch.setattr(planner_mod, "evaluate_action_effect", lambda *a, **kw: None)

    def make(*, vision, decomposer=None, capture=None):
        return Planner(
            bus=FakeBus(),
            actuator=FakeActuator(),
            vision=vision,
            pause=FakePause(),
            capture=capture if capture is not None else FakeCapture(),
            loop_detector=LoopDetector(),
            judge=None,
            decomposer=decomposer,
        )

    return make


async def _drive(planner, goal: str) -> None:
    await planner._on_intent(Event(topic="user_intent", payload={"text": goal}))


def test_single_subgoal_plan_completes(_patched_planner):
    vision = ScriptedVision([Action(kind="done", args={"answer": "ok"})])
    decomposer = StubDecomposer(Plan(subgoals=[Subgoal(goal="open notes")]))
    planner = _patched_planner(vision=vision, decomposer=decomposer)

    asyncio.run(_drive(planner, "open the notes app"))

    bus: FakeBus = planner.bus  # type: ignore[assignment]
    assert "task_started" in bus.topics()
    assert "plan_decomposed" in bus.topics()
    assert "subgoal_started" in bus.topics()
    assert "subgoal_completed" in bus.topics()
    assert "task_completed" in bus.topics()
    completed = bus.by_topic("task_completed")[0]
    assert completed["answer"] == "ok"
    assert completed["subgoals"] == 1
    assert decomposer.decompose_calls == ["open the notes app"]


def test_two_subgoal_plan_passes_answer_forward(_patched_planner):
    vision = ScriptedVision(
        [
            Action(kind="done", args={"answer": "12°C"}),
            Action(kind="done", args={"answer": "saved"}),
        ]
    )
    decomposer = StubDecomposer(
        Plan(
            subgoals=[
                Subgoal(goal="read the temperature"),
                Subgoal(goal="save it to a note"),
            ]
        )
    )
    planner = _patched_planner(vision=vision, decomposer=decomposer)
    asyncio.run(_drive(planner, "find Paris weather and save"))

    assert len(vision.goals_seen) == 2
    assert vision.goals_seen[0] == "read the temperature"
    assert "12°C" in vision.goals_seen[1]
    assert "Carried context" in vision.goals_seen[1]
    bus: FakeBus = planner.bus  # type: ignore[assignment]
    assert len(bus.by_topic("subgoal_completed")) == 2
    final = bus.by_topic("task_completed")[0]
    assert final["answer"] == "saved"


def test_subgoal_failure_triggers_replan(_patched_planner, monkeypatch):
    # Force subgoal 1 to exhaust its step cap before reaching `done`.
    from nalu import config

    monkeypatch.setattr(config, "PLANNER_MAX_STEPS", 4)
    vision = ScriptedVision(
        [
            Action(kind="click", args={"x": 50, "y": 40}),
            Action(kind="click", args={"x": 50, "y": 40}),
            Action(kind="click", args={"x": 50, "y": 40}),
            Action(kind="click", args={"x": 50, "y": 40}),
            Action(kind="done", args={"answer": "recovered"}),
        ]
    )
    decomposer = StubDecomposer(
        decompose_plan=Plan(subgoals=[Subgoal(goal="search Paris")]),
        replan_plan=Plan(subgoals=[Subgoal(goal="dismiss modal then search")]),
    )
    planner = _patched_planner(vision=vision, decomposer=decomposer)
    asyncio.run(_drive(planner, "find Paris weather"))

    bus: FakeBus = planner.bus  # type: ignore[assignment]
    assert "subgoal_failed" in bus.topics()
    assert "plan_replanned" in bus.topics()
    replan_ev = bus.by_topic("plan_replanned")[0]
    assert replan_ev["replans_used"] == 1
    assert decomposer.replan_calls
    assert decomposer.replan_calls[0]["original_goal"] == "find Paris weather"
    assert decomposer.replan_calls[0]["failed_subgoal"] == "search Paris"
    assert "task_completed" in bus.topics()


def test_replan_budget_exhausted_fails_task(_patched_planner, monkeypatch):
    from nalu import config

    monkeypatch.setattr(config, "PLANNER_MAX_STEPS", 4)
    vision = ScriptedVision(
        [Action(kind="click", args={"x": 1, "y": 1}) for _ in range(40)]
    )
    decomposer = StubDecomposer(
        decompose_plan=Plan(subgoals=[Subgoal(goal="initial attempt")]),
        replan_plan=Plan(subgoals=[Subgoal(goal="retry attempt")]),
    )
    planner = _patched_planner(vision=vision, decomposer=decomposer)
    asyncio.run(_drive(planner, "do the thing"))

    bus: FakeBus = planner.bus  # type: ignore[assignment]
    assert "task_failed" in bus.topics()
    assert len(bus.by_topic("plan_replanned")) == 1
    failed = bus.by_topic("task_failed")[0]
    assert "subgoal" in failed["reason"]


def test_no_decomposer_runs_single_subgoal_back_compat(_patched_planner, monkeypatch):
    from nalu import config

    monkeypatch.setattr(config, "USE_LLM_PLANNER", False)
    vision = ScriptedVision([Action(kind="done", args={"answer": "fine"})])
    planner = _patched_planner(vision=vision, decomposer=None)
    asyncio.run(_drive(planner, "open mail"))

    bus: FakeBus = planner.bus  # type: ignore[assignment]
    # Plan event still fires (with one subgoal == the goal verbatim).
    assert bus.by_topic("plan_decomposed")[0]["subgoals"][0]["goal"] == "open mail"
    assert vision.goals_seen == ["open mail"]
    assert bus.by_topic("task_completed")[0]["answer"] == "fine"


def test_meta_json_records_subgoals_and_replans(_patched_planner, tmp_path):
    vision = ScriptedVision(
        [
            Action(kind="done", args={"answer": "first"}),
            Action(kind="done", args={"answer": "second"}),
        ]
    )
    decomposer = StubDecomposer(
        Plan(subgoals=[Subgoal(goal="step 1"), Subgoal(goal="step 2")])
    )
    planner = _patched_planner(vision=vision, decomposer=decomposer)
    asyncio.run(_drive(planner, "two-step"))

    run_dirs = sorted(tmp_path.glob("run_*"))
    assert len(run_dirs) == 1
    meta = json.loads((run_dirs[0] / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert len(meta["subgoals"]) == 2
    assert meta["subgoals"][0]["goal"] == "step 1"
    assert meta["subgoals"][0]["answer"] == "first"
    assert meta["subgoals"][1]["answer"] == "second"
    assert meta["replans_used"] == 0
    assert meta["answer"] == "second"


def test_plan_json_persisted_when_decomposer_runs(_patched_planner, tmp_path):
    vision = ScriptedVision([Action(kind="done", args={"answer": "ok"})])
    decomposer = StubDecomposer(Plan(subgoals=[Subgoal(goal="alpha")]))
    planner = _patched_planner(vision=vision, decomposer=decomposer)
    asyncio.run(_drive(planner, "user goal"))

    run_dirs = sorted(tmp_path.glob("run_*"))
    plan_json = json.loads((run_dirs[0] / "plan.json").read_text())
    assert plan_json["goal"] == "user goal"
    assert plan_json["subgoals"][0]["goal"] == "alpha"


def test_decompose_exception_falls_back_to_raw_goal(_patched_planner):
    class BrokenDecomposer:
        def decompose(self, goal, conversation=""):
            raise RuntimeError("oom")

        def replan(self, **kw):
            return Plan(subgoals=[], fallback=True)

    vision = ScriptedVision([Action(kind="done", args={"answer": "still works"})])
    planner = _patched_planner(vision=vision, decomposer=BrokenDecomposer())
    asyncio.run(_drive(planner, "describe me"))

    bus: FakeBus = planner.bus  # type: ignore[assignment]
    assert vision.goals_seen == ["describe me"]
    assert bus.by_topic("task_completed")[0]["answer"] == "still works"
    assert bus.by_topic("plan_decomposed")[0]["fallback"] is True
