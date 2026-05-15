"""Proactive Jarvis speech.

Most of Nalu's voice today is reactive: the user pushes a button, the agent
runs, the final answer gets spoken. The Iron-Man-Jarvis feel comes from the
moments in between — short status quips while Nalu works ("Working on it,
sir." "One moment.") and brief commentary on what just happened ("Done."
"That click didn't land, trying again.").

`ProactiveSpeaker` subscribes to interesting bus events and emits a short
spoken phrase via the TTS callable when one fires. Rate-limited so we don't
chatter on every event, gated behind `NALU_PROACTIVE_VOICE` so it's opt-in,
and parametric on the event → utterance table so users can swap in their
own quips without editing the source.

Pure-Python orchestration: the actual speech goes through an injected
`speak_fn` callable so tests run without Piper. In `daemon.py` this is
wired to `TTS.speak` on a background thread (same pattern as task answers).
"""
from __future__ import annotations

import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import structlog

from ...bus import BusClient, Event


log = structlog.get_logger("proactive")


SpeakFn = Callable[[str], None]


# Default event → phrase pool. One phrase is chosen at random per fire to
# avoid sounding scripted. Keep each phrase short (≤8 words) so it doesn't
# step on the next event.
DEFAULT_UTTERANCES: dict[str, tuple[str, ...]] = {
    "task_started": (
        "Working on it.",
        "On it, sir.",
        "One moment.",
        "Right away.",
    ),
    "subgoal_started": (
        "Next step.",
        "Moving on.",
    ),
    "action_no_effect": (
        "That didn't land. Trying again.",
        "Missed. Retrying.",
    ),
    "task_recovering": (
        "Adjusting approach.",
        "Backing up. Trying another way.",
    ),
    "stuck_detected": (
        "I'm stuck. Skipping ahead.",
    ),
    "task_paused": (
        "Paused.",
    ),
    "task_failed": (
        "Couldn't complete that one.",
        "That didn't work out.",
    ),
}


# Per-event minimum gap so we don't repeat the same phrase back-to-back when
# events fire in clusters (subgoal_started fires once per subgoal in a 6-step
# plan — speaking six times in a row would be obnoxious).
DEFAULT_PER_EVENT_COOLDOWN_S: dict[str, float] = {
    "task_started": 0.0,
    "subgoal_started": 8.0,
    "action_no_effect": 6.0,
    "task_recovering": 6.0,
    "stuck_detected": 6.0,
    "task_paused": 0.0,
    "task_failed": 0.0,
}

# Absolute floor across all events — don't speak more than once every N seconds
# regardless of event type. Prevents speech overlap entirely.
DEFAULT_GLOBAL_COOLDOWN_S = 1.5


def is_proactive_enabled() -> bool:
    """Read the env gate. Default off — Jarvis chatter is a deliberate opt-in."""
    return str(os.environ.get("NALU_PROACTIVE_VOICE", "0")).strip().lower() not in (
        "0", "", "false", "no", "off",
    )


@dataclass
class ProactiveConfig:
    utterances: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_UTTERANCES.items()}
    )
    per_event_cooldown_s: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PER_EVENT_COOLDOWN_S)
    )
    global_cooldown_s: float = DEFAULT_GLOBAL_COOLDOWN_S


class ProactiveSpeaker:
    """Bus-driven Jarvis-style commentary.

    `speak_fn` is the only side-effecting hook so tests can capture phrases
    without invoking Piper / the audio stack. `clock` is injectable for
    testing rate limits without sleeping.
    """

    def __init__(
        self,
        bus: BusClient,
        speak_fn: SpeakFn,
        *,
        config: ProactiveConfig | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        enabled: bool | None = None,
    ) -> None:
        self.bus = bus
        self._speak = speak_fn
        self.config = config or ProactiveConfig()
        self._rng = rng or random.Random()
        self._clock = clock
        self._enabled = is_proactive_enabled() if enabled is None else enabled
        self._lock = threading.Lock()
        self._last_global = -math.inf
        self._last_per_event: dict[str, float] = {}
        self.spoken_count = 0  # observable for tests / metrics

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    async def run(self) -> None:
        if not self._enabled:
            log.info("proactive_disabled")
            return
        for topic in self.config.utterances:
            await self.bus.subscribe(topic, self._make_handler(topic))
        await self.bus.publish("proactive_ready", {"ts": time.time()})

    def _make_handler(self, topic: str):
        async def _handler(_ev: Event) -> None:
            self._maybe_speak(topic)
        return _handler

    def _maybe_speak(self, topic: str) -> None:
        if not self._enabled:
            return
        with self._lock:
            now = self._clock()
            if now - self._last_global < self.config.global_cooldown_s:
                return
            per_event = self.config.per_event_cooldown_s.get(topic, 0.0)
            last = self._last_per_event.get(topic, -math.inf)
            if per_event > 0.0 and now - last < per_event:
                return
            phrases = self.config.utterances.get(topic) or ()
            if not phrases:
                return
            phrase = self._rng.choice(phrases)
            self._last_global = now
            self._last_per_event[topic] = now
            self.spoken_count += 1
        try:
            self._speak(phrase)
        except Exception:
            log.exception("proactive_speak_failed", topic=topic, phrase=phrase)


__all__ = [
    "ProactiveSpeaker",
    "ProactiveConfig",
    "DEFAULT_UTTERANCES",
    "DEFAULT_PER_EVENT_COOLDOWN_S",
    "DEFAULT_GLOBAL_COOLDOWN_S",
    "is_proactive_enabled",
]
