from __future__ import annotations

import asyncio
from typing import Callable

from nalu.agents.responder import (
    Responder,
    build_user_prompt,
    clean_reply,
    make_default_generate_fn,
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

    def by_topic(self, topic: str) -> list[dict]:
        return [p for t, p in self.events if t == topic]


def _drive(coro):
    return asyncio.run(coro)


async def _fire_query(bus: FakeBus, text: str, conversation: str = "") -> None:
    await bus.publish("user_query", {"text": text, "conversation": conversation})


def test_clean_reply_strips_whitespace_and_quotes():
    assert clean_reply('  "Hello there."  ') == "Hello there."


def test_clean_reply_strips_code_fences():
    assert clean_reply("```\nHello, sir.\n```") == "Hello, sir."


def test_clean_reply_truncates_long_reply():
    raw = "word " * 200  # 1000 chars
    out = clean_reply(raw, max_chars=50)
    assert out.endswith("…")
    assert len(out) <= 51  # word boundary + ellipsis


def test_clean_reply_empty_in_empty_out():
    assert clean_reply("") == ""
    assert clean_reply("   ") == ""


def test_build_user_prompt_omits_conversation_block_when_empty():
    p = build_user_prompt("what time is it?")
    assert "Recent conversation" not in p
    assert "what time is it?" in p


def test_build_user_prompt_includes_conversation_when_present():
    p = build_user_prompt("and the weather?", "User: hi\nNalu: hello")
    assert "Recent conversation" in p
    assert "User: hi" in p
    assert "and the weather?" in p


def test_responder_publishes_ready_on_run():
    bus = FakeBus()
    r = Responder(bus, generate_fn=lambda s, u: "ok")
    _drive(r.run())
    assert bus.by_topic("responder_ready"), "expected responder_ready event"


def test_responder_publishes_reply_on_user_query():
    bus = FakeBus()
    captured: dict = {}

    def gen(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return "Five o'clock, sir."

    r = Responder(bus, generate_fn=gen)

    async def _run():
        await r.run()
        await _fire_query(bus, "what time is it?")

    _drive(_run())
    replies = bus.by_topic("responder_reply")
    assert len(replies) == 1
    assert replies[0]["reply"] == "Five o'clock, sir."
    assert replies[0]["query"] == "what time is it?"
    assert replies[0]["elapsed_ms"] >= 0
    assert "Jarvis" in captured["system"]
    assert "what time is it?" in captured["user"]


def test_responder_ignores_empty_query():
    bus = FakeBus()
    r = Responder(bus, generate_fn=lambda s, u: "noop")

    async def _run():
        await r.run()
        await _fire_query(bus, "   ")

    _drive(_run())
    assert bus.by_topic("responder_reply") == []


def test_responder_publishes_failed_when_generate_raises():
    bus = FakeBus()

    def gen(system: str, user: str) -> str:
        raise RuntimeError("model boom")

    r = Responder(bus, generate_fn=gen)

    async def _run():
        await r.run()
        await _fire_query(bus, "anything")

    _drive(_run())
    failures = bus.by_topic("responder_failed")
    assert len(failures) == 1
    assert "boom" in failures[0]["reason"]
    assert failures[0]["query"] == "anything"
    assert bus.by_topic("responder_reply") == []


def test_responder_caps_long_reply():
    bus = FakeBus()
    r = Responder(bus, generate_fn=lambda s, u: "blah " * 200, max_reply_chars=40)

    async def _run():
        await r.run()
        await _fire_query(bus, "ramble please")

    _drive(_run())
    reply = bus.by_topic("responder_reply")[0]["reply"]
    assert reply.endswith("…")
    assert len(reply) <= 41


def test_make_default_generate_fn_calls_decomposer_generate():
    class FakeDecomposer:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []

        def _generate(self, system: str, user: str) -> str:
            self.calls.append((system, user))
            return "ok"

    dec = FakeDecomposer()
    fn = make_default_generate_fn(dec)
    assert fn("sys", "usr") == "ok"
    assert dec.calls == [("sys", "usr")]


def test_responder_passes_conversation_through_to_prompt():
    bus = FakeBus()
    seen: dict = {}

    def gen(system: str, user: str) -> str:
        seen["user"] = user
        return "fine"

    r = Responder(bus, generate_fn=gen)

    async def _run():
        await r.run()
        await _fire_query(bus, "and you?", conversation="User: hi\nNalu: hello")

    _drive(_run())
    assert "User: hi" in seen["user"]
    assert "and you?" in seen["user"]
