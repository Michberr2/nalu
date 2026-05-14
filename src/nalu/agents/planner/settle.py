"""Active screen-stabilization wait.

Replaces a fixed post-action sleep with a poll loop that watches consecutive
frames and exits as soon as the screen has visibly stopped changing — or hits
a max cap. Most UI actions stabilize in 50–200 ms; a fixed 400 ms sleep
overpays for the fast cases and underpays for the slow ones (a click that
opens a heavy menu, a `cmd+t` that needs a fresh tab to draw). Polling lets
the planner spend exactly the time the screen actually needs.

Pure async with an injected `frame_getter` callable so unit tests can script
the frame sequence without touching ContinuousCapture or PyObjC.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from PIL import Image

from .screen_change import perceptual_diff


SETTLE_MAX_WAIT_S = 1.5
SETTLE_MIN_WAIT_S = 0.05
SETTLE_POLL_S = 0.08
SETTLE_DIFF_THRESHOLD = 0.005
SETTLE_STABLE_POLLS_REQUIRED = 2


FrameGetter = Callable[[], Optional[Image.Image]]


@dataclass
class SettleResult:
    elapsed_s: float
    polls: int
    last_diff: float
    stable: bool  # True if the stability threshold was hit; False if we capped out


async def wait_for_screen_settle(
    frame_getter: FrameGetter,
    *,
    max_wait_s: float = SETTLE_MAX_WAIT_S,
    min_wait_s: float = SETTLE_MIN_WAIT_S,
    poll_s: float = SETTLE_POLL_S,
    threshold: float = SETTLE_DIFF_THRESHOLD,
    stable_polls_required: int = SETTLE_STABLE_POLLS_REQUIRED,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SettleResult:
    """Poll `frame_getter` until consecutive frames are nearly identical, or `max_wait_s` elapses.

    Semantics:
    - Sleeps `min_wait_s` first so the OS has at least one redraw window.
    - Samples a baseline frame, then polls every `poll_s` for at most `max_wait_s`.
    - Treats `stable_polls_required` consecutive `perceptual_diff <= threshold` samples
      as "settled" and returns early with `stable=True`.
    - If the initial `frame_getter()` returns `None`, returns immediately with
      `stable=False` after `min_wait_s` — caller should fall back to a fixed sleep.
    - `sleeper` is injectable so tests run without burning wall-clock time.
    """
    t0 = time.monotonic()
    if min_wait_s > 0:
        await sleeper(min_wait_s)

    prev = frame_getter()
    if prev is None:
        return SettleResult(
            elapsed_s=time.monotonic() - t0,
            polls=0,
            last_diff=0.0,
            stable=False,
        )

    polls = 0
    consecutive_stable = 0
    last_diff = 0.0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= max_wait_s:
            return SettleResult(
                elapsed_s=elapsed,
                polls=polls,
                last_diff=last_diff,
                stable=False,
            )
        await sleeper(poll_s)
        polls += 1
        curr = frame_getter()
        if curr is None:
            continue
        last_diff = perceptual_diff(prev, curr)
        if last_diff <= threshold:
            consecutive_stable += 1
            if consecutive_stable >= stable_polls_required:
                return SettleResult(
                    elapsed_s=time.monotonic() - t0,
                    polls=polls,
                    last_diff=last_diff,
                    stable=True,
                )
        else:
            consecutive_stable = 0
        prev = curr
