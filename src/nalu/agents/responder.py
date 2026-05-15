"""Responder — short conversational replies via the same MLX-LM model the planner uses.

The planner is built for *acting* on the screen: decompose a goal into subgoals,
hand each to the vision agent, loop. That round-trip is the wrong shape for
quick conversational turns ("what time is it?", "thanks", "tell me a joke").
Those don't need a screen action; they need a one-shot text answer.

Dual-model orchestration in the spirit of Thinking Machines' interaction model:
the same Qwen LLM that decomposes goals also answers conversational turns,
guarded by a separate lock so a slow decompose doesn't block a quick reply
and vice-versa.

`Responder` subscribes to `user_query` events and publishes `responder_reply`.
The daemon decides which path to take (heuristic in `daemon.py`); this module
only owns the reply itself. Pure-Python parsing here — MLX is lazy-loaded via
the same `LLMDecomposer.load()` hook used elsewhere so the test path doesn't
trigger a model download.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

import structlog

from ..bus import BusClient, Event


log = structlog.get_logger("responder")


RESPONDER_SYSTEM = """You are Nalu, a concise voice assistant inspired by Jarvis from Iron Man.

## Rules
- Respond in one or two short sentences. No lists. No headers. Plain prose.
- Speak naturally — your reply will be read aloud by text-to-speech.
- If the user asks for something that requires acting on the screen, say so briefly and stop. Do not pretend to have done it.
- If you do not know, say so. Do not invent facts.
- Match the user's register — neutral, dry wit when appropriate.
"""


GenerateFn = Callable[[str, str], str]


class Responder:
    """Conversational reply agent backed by the planner LLM.

    `generate_fn` is the only injected hook so unit tests can stub the MLX path
    without loading a 7B model. In production this defaults to a thin wrapper
    around `LLMDecomposer._generate` so the two share a single loaded model.
    """

    def __init__(
        self,
        bus: BusClient,
        generate_fn: GenerateFn,
        *,
        max_reply_chars: int = 280,
        system_prompt: str = RESPONDER_SYSTEM,
    ) -> None:
        self.bus = bus
        self._generate = generate_fn
        self.max_reply_chars = max_reply_chars
        self.system_prompt = system_prompt
        self._lock = threading.Lock()

    async def run(self) -> None:
        await self.bus.subscribe("user_query", self._on_query)
        await self.bus.publish("responder_ready", {"ts": time.time()})

    async def _on_query(self, ev: Event) -> None:
        text = (ev.payload.get("text") or "").strip()
        if not text:
            return
        conv = ev.payload.get("conversation") or ""
        t0 = time.monotonic()
        try:
            raw = await asyncio.to_thread(self._safe_generate, text, conv)
        except Exception as e:
            log.exception("responder_generate_failed")
            await self.bus.publish(
                "responder_failed",
                {"reason": str(e), "query": text},
            )
            return
        reply = clean_reply(raw, max_chars=self.max_reply_chars)
        await self.bus.publish(
            "responder_reply",
            {
                "query": text,
                "reply": reply,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
            },
        )

    def _safe_generate(self, query: str, conversation: str) -> str:
        user = build_user_prompt(query, conversation)
        with self._lock:
            return self._generate(self.system_prompt, user)


def build_user_prompt(query: str, conversation: str = "") -> str:
    conv = (conversation or "").strip()
    block = f"## Recent conversation\n{conv}\n\n" if conv else ""
    return f"{block}## User\n{query.strip()}\n\nReply in one or two short sentences."


def clean_reply(raw: str, *, max_chars: int = 280) -> str:
    """Strip whitespace, drop accidental code-fence wrappers, cap length."""
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if "\n" in s:
            _first, _, rest = s.partition("\n")
            s = rest.strip()
    s = s.strip().strip('"').strip("'").strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"
    return s


def make_default_generate_fn(decomposer) -> GenerateFn:
    """Wrap an existing `LLMDecomposer` so the Responder reuses its loaded model.

    The decomposer's `_generate` already serializes access via its own lock,
    so we don't double-lock here.
    """

    def _gen(system: str, user: str) -> str:
        return decomposer._generate(system, user)

    return _gen


__all__ = [
    "Responder",
    "RESPONDER_SYSTEM",
    "build_user_prompt",
    "clean_reply",
    "make_default_generate_fn",
]
