"""Stuck/loop detection for the planner.

The vision model occasionally emits the same wrong action over and over, or
oscillates between two actions that don't make progress (click X → click Y →
click X → click Y...). Without a circuit-breaker the planner happily burns
through the step cap reproducing the same mistake.

`LoopDetector` is a pure-Python state machine the planner consults after
every action is emitted. It tracks a small window of action *signatures*
(coarse-grained — clicks are bucketed to a 32 px grid so near-identical
clicks dedupe) and flags two patterns:

  - **repeat**       — the same signature N times in a row (default N=3)
  - **alternation**  — A-B-A-B... for K cycles (default K=3)

When the detector fires, the planner injects a hint into the action history
("you've repeated this; try something different") and *skips the dispatch*
for that step. If the very next vision turn produces the same flagged
signature anyway, `consecutive_signals` reaches 2 and the planner gives up
with `task_failed({"reason": "stuck"})`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


DEFAULT_REPEAT_THRESHOLD = 3
DEFAULT_ALTERNATION_CYCLES = 3
DEFAULT_CLICK_GRID_PX = 32
DEFAULT_TYPE_PREFIX = 40
DEFAULT_WINDOW = 12


@dataclass
class StuckSignal:
    reason: str  # "repeat" | "alternation"
    count: int
    hint: str
    signature: tuple


def action_signature(
    kind: str, args: dict[str, Any], *, click_grid_px: int = DEFAULT_CLICK_GRID_PX,
    type_prefix: int = DEFAULT_TYPE_PREFIX,
) -> tuple:
    """Coarse signature for action equivalence.

    Two `click(123,456)` calls and `click(140,470)` collapse to the same
    signature when bucketed by 32 px — they're 'the same wrong click' for
    stuck-detection purposes.
    """
    if kind == "click":
        x = int(args.get("x", 0)) // click_grid_px
        y = int(args.get("y", 0)) // click_grid_px
        return ("click", x, y, args.get("button", "left"), int(args.get("clicks", 1)))
    if kind == "type":
        return ("type", str(args.get("text", ""))[:type_prefix])
    if kind == "key":
        mods = tuple(sorted(args.get("modifiers", [])))
        return ("key", args.get("name", ""), mods)
    if kind == "scroll":
        return ("scroll", int(args.get("dx", 0)), int(args.get("dy", 0)))
    if kind == "wait":
        return ("wait",)
    try:
        return (kind, tuple(sorted((k, str(v)) for k, v in args.items())))
    except Exception:
        return (kind,)


@dataclass
class LoopDetector:
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD
    alternation_cycles: int = DEFAULT_ALTERNATION_CYCLES
    click_grid_px: int = DEFAULT_CLICK_GRID_PX
    type_prefix: int = DEFAULT_TYPE_PREFIX
    window_size: int = DEFAULT_WINDOW
    _window: deque = field(init=False)
    _last_signal_sig: tuple | None = field(init=False, default=None)
    _consecutive: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.window_size)

    @property
    def consecutive_signals(self) -> int:
        return self._consecutive

    def reset(self) -> None:
        self._window.clear()
        self._last_signal_sig = None
        self._consecutive = 0

    def observe(self, kind: str, args: dict[str, Any]) -> StuckSignal | None:
        sig = action_signature(
            kind, args, click_grid_px=self.click_grid_px, type_prefix=self.type_prefix
        )
        self._window.append(sig)

        signal = self._detect_repeat(sig, kind) or self._detect_alternation()
        if signal is None:
            self._last_signal_sig = None
            self._consecutive = 0
            return None

        if signal.signature == self._last_signal_sig:
            self._consecutive += 1
        else:
            self._consecutive = 1
            self._last_signal_sig = signal.signature
        signal.count = self._consecutive
        return signal

    def _detect_repeat(self, sig: tuple, kind: str) -> StuckSignal | None:
        if len(self._window) < self.repeat_threshold:
            return None
        tail = list(self._window)[-self.repeat_threshold:]
        if not all(s == sig for s in tail):
            return None
        hint = (
            f"You have just performed `{kind}` {self.repeat_threshold} times in a row "
            f"with the same target and the screen has not changed in the way you expected. "
            f"Stop repeating it. Re-examine the screenshot and pick a different action or target."
        )
        return StuckSignal(reason="repeat", count=self.repeat_threshold, hint=hint, signature=sig)

    def _detect_alternation(self) -> StuckSignal | None:
        needed = self.alternation_cycles * 2
        if len(self._window) < needed:
            return None
        tail = list(self._window)[-needed:]
        a, b = tail[0], tail[1]
        if a == b:
            return None
        if not all(tail[i] == (a if i % 2 == 0 else b) for i in range(needed)):
            return None
        hint = (
            f"You are oscillating between `{a[0]}` and `{b[0]}` without making progress "
            f"({self.alternation_cycles} cycles). Try a third approach — neither of those two actions "
            f"is moving the task forward."
        )
        return StuckSignal(
            reason="alternation",
            count=self.alternation_cycles,
            hint=hint,
            signature=(a, b),
        )
