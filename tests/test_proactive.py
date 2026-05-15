from __future__ import annotations

import asyncio
import random
from typing import Callable

from nalu.agents.voice.proactive import (
    DEFAULT_UTTERANCES,
    ProactiveConfig,
    ProactiveSpeaker,
    is_proactive_enabled,
)
from nalu.bus import Event


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


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def tick(self, dt: float) -> None:
        self.now += dt

    def __call__(self) -> float:
        return self.now


def _run(coro):
    return asyncio.run(coro)


def _drive(events: list[tuple[str, dict]], bus: FakeBus, speaker: ProactiveSpeaker):
    async def _go():
        await speaker.run()
        for topic, payload in events:
            await bus.publish(topic, payload)
    _run(_go())


def test_is_proactive_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("NALU_PROACTIVE_VOICE", raising=False)
    assert is_proactive_enabled() is False
    monkeypatch.setenv("NALU_PROACTIVE_VOICE", "1")
    assert is_proactive_enabled() is True
    monkeypatch.setenv("NALU_PROACTIVE_VOICE", "false")
    assert is_proactive_enabled() is False
    monkeypatch.setenv("NALU_PROACTIVE_VOICE", "yes")
    assert is_proactive_enabled() is True


def test_disabled_speaker_skips_subscriptions():
    bus = FakeBus()
    said: list[str] = []
    sp = ProactiveSpeaker(bus, speak_fn=said.append, enabled=False)
    _drive([("task_started", {})], bus, sp)
    assert said == []
    # also: no subscriptions registered (bus._subs is empty for this topic)
    assert "task_started" not in bus._subs


def test_speaks_on_known_event_when_enabled():
    bus = FakeBus()
    said: list[str] = []
    clock = FakeClock(0.0)
    sp = ProactiveSpeaker(
        bus,
        speak_fn=said.append,
        rng=random.Random(0),
        clock=clock,
        enabled=True,
    )
    _drive([("task_started", {})], bus, sp)
    assert len(said) == 1
    assert said[0] in DEFAULT_UTTERANCES["task_started"]
    assert sp.spoken_count == 1


def test_publishes_proactive_ready_when_enabled():
    bus = FakeBus()
    sp = ProactiveSpeaker(bus, speak_fn=lambda _: None, enabled=True)
    _run(sp.run())
    assert any(t == "proactive_ready" for t, _ in bus.events)


def test_global_cooldown_silences_quick_followup():
    bus = FakeBus()
    said: list[str] = []
    clock = FakeClock(0.0)
    cfg = ProactiveConfig(global_cooldown_s=1.5)
    sp = ProactiveSpeaker(
        bus, speak_fn=said.append, config=cfg, rng=random.Random(0), clock=clock, enabled=True
    )

    async def _go():
        await sp.run()
        await bus.publish("task_started", {})  # speaks
        clock.tick(0.5)
        await bus.publish("task_failed", {})  # under global cooldown → silent
        clock.tick(2.0)
        await bus.publish("task_failed", {})  # past cooldown → speaks

    _run(_go())
    assert len(said) == 2


def test_per_event_cooldown_silences_repeats():
    bus = FakeBus()
    said: list[str] = []
    clock = FakeClock(0.0)
    cfg = ProactiveConfig(
        global_cooldown_s=0.0,
        per_event_cooldown_s={"subgoal_started": 5.0},
    )
    sp = ProactiveSpeaker(
        bus, speak_fn=said.append, config=cfg, rng=random.Random(0), clock=clock, enabled=True
    )

    async def _go():
        await sp.run()
        await bus.publish("subgoal_started", {})  # speaks
        clock.tick(1.0)
        await bus.publish("subgoal_started", {})  # under per-event cooldown → silent
        clock.tick(1.0)
        await bus.publish("subgoal_started", {})  # still silent
        clock.tick(10.0)
        await bus.publish("subgoal_started", {})  # past cooldown → speaks

    _run(_go())
    assert len(said) == 2


def test_unknown_event_does_not_subscribe_or_speak():
    bus = FakeBus()
    said: list[str] = []
    cfg = ProactiveConfig(
        utterances={"task_started": ("Working on it.",)},
        per_event_cooldown_s={},
    )
    sp = ProactiveSpeaker(bus, speak_fn=said.append, config=cfg, enabled=True)
    _drive([("unrelated_topic", {})], bus, sp)
    assert said == []


def test_speak_exception_does_not_break_handler():
    bus = FakeBus()
    fail_count = {"n": 0}

    def boom(_t: str) -> None:
        fail_count["n"] += 1
        raise RuntimeError("audio device gone")

    sp = ProactiveSpeaker(bus, speak_fn=boom, enabled=True)
    _drive([("task_started", {})], bus, sp)
    assert fail_count["n"] == 1  # we did try to speak; the exception was swallowed


def test_phrase_pool_uses_rng():
    bus = FakeBus()
    said: list[str] = []
    cfg = ProactiveConfig(
        utterances={"task_started": ("a", "b", "c", "d")},
        per_event_cooldown_s={"task_started": 0.0},
        global_cooldown_s=0.0,
    )
    sp = ProactiveSpeaker(
        bus, speak_fn=said.append, config=cfg, rng=random.Random(42), clock=FakeClock(0.0), enabled=True
    )

    async def _go():
        await sp.run()
        for _ in range(20):
            await bus.publish("task_started", {})

    _run(_go())
    # 20 unique-ish random picks from 4 options should cover at least 2 different phrases
    assert len(set(said)) >= 2
    assert all(p in {"a", "b", "c", "d"} for p in said)


def test_set_enabled_false_silences_subsequent_events():
    bus = FakeBus()
    said: list[str] = []
    sp = ProactiveSpeaker(bus, speak_fn=said.append, enabled=True)
    _run(sp.run())
    _run(bus.publish("task_started", {}))
    assert len(said) == 1
    sp.set_enabled(False)
    _run(bus.publish("task_started", {}))
    assert len(said) == 1  # handler still attached but gate now blocks it
