from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from PIL import Image

from nalu.agents.planner.settle import (
    SETTLE_DIFF_THRESHOLD,
    SettleResult,
    wait_for_screen_settle,
)


def _solid(width: int, height: int, value: int) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8), mode="RGB")


def _make_sleeper(sleeps: list[float]):
    async def _sleeper(s: float) -> None:
        sleeps.append(s)
    return _sleeper


def _frames(seq: list[Optional[Image.Image]]):
    it = iter(seq)

    def _get():
        try:
            return next(it)
        except StopIteration:
            return seq[-1]
    return _get


def test_settle_returns_immediately_stable_when_frames_identical():
    a = _solid(64, 64, 128)
    sleeps: list[float] = []
    result: SettleResult = asyncio.run(
        wait_for_screen_settle(
            _frames([a, a, a, a]),
            max_wait_s=1.0,
            min_wait_s=0.05,
            poll_s=0.1,
            stable_polls_required=2,
            sleeper=_make_sleeper(sleeps),
        )
    )
    assert result.stable is True
    assert result.polls == 2
    assert result.last_diff == 0.0
    # one min-wait sleep + two poll sleeps
    assert sleeps == [0.05, 0.1, 0.1]


def test_settle_waits_then_stabilizes():
    a = _solid(64, 64, 0)
    b = _solid(64, 64, 200)
    # baseline=a, then changes for a couple polls, then steady on b
    seq = [a, b, b, b, b]
    sleeps: list[float] = []
    result = asyncio.run(
        wait_for_screen_settle(
            _frames(seq),
            max_wait_s=2.0,
            min_wait_s=0.0,
            poll_s=0.05,
            stable_polls_required=2,
            sleeper=_make_sleeper(sleeps),
        )
    )
    assert result.stable is True
    # poll 1: baseline vs b → diff high, reset
    # poll 2: b vs b → stable (1)
    # poll 3: b vs b → stable (2) → return
    assert result.polls == 3
    assert result.last_diff == 0.0


def test_settle_caps_when_screen_keeps_changing():
    # Alternating frames forever — never stabilizes, must hit max_wait cap.
    a = _solid(64, 64, 0)
    b = _solid(64, 64, 255)
    state = {"i": 0}

    def _alternating():
        f = a if state["i"] % 2 == 0 else b
        state["i"] += 1
        return f

    # Use real (tiny) sleeps so time.monotonic() actually advances and the cap can fire.
    result = asyncio.run(
        wait_for_screen_settle(
            _alternating,
            max_wait_s=0.08,
            min_wait_s=0.0,
            poll_s=0.01,
            stable_polls_required=2,
        )
    )
    assert result.stable is False
    assert result.polls >= 1
    assert result.last_diff > SETTLE_DIFF_THRESHOLD


def test_settle_returns_unstable_when_initial_frame_is_none():
    sleeps: list[float] = []
    result = asyncio.run(
        wait_for_screen_settle(
            _frames([None]),
            max_wait_s=1.0,
            min_wait_s=0.05,
            poll_s=0.1,
            sleeper=_make_sleeper(sleeps),
        )
    )
    assert result.stable is False
    assert result.polls == 0
    # min-wait was applied, but no polls were attempted
    assert sleeps == [0.05]


def test_settle_skips_intermittent_none_frames():
    a = _solid(64, 64, 100)
    seq = [a, None, a, a]
    sleeps: list[float] = []
    result = asyncio.run(
        wait_for_screen_settle(
            _frames(seq),
            max_wait_s=2.0,
            min_wait_s=0.0,
            poll_s=0.05,
            stable_polls_required=2,
            sleeper=_make_sleeper(sleeps),
        )
    )
    assert result.stable is True
    # poll 1: None → skip; poll 2: a vs a stable (1); poll 3: a vs a stable (2) → return
    assert result.polls == 3


def test_settle_respects_custom_threshold():
    a = _solid(64, 64, 100)
    b = _solid(64, 64, 102)  # tiny diff between consecutive frames
    state = {"i": 0}

    def _toggling():
        # Alternates a/b so consecutive polls always see a small but nonzero diff.
        f = a if state["i"] % 2 == 0 else b
        state["i"] += 1
        return f

    # Tight threshold (0.0) → small diff looks like change → caps out unstable.
    tight = asyncio.run(
        wait_for_screen_settle(
            _toggling,
            max_wait_s=0.08,
            min_wait_s=0.0,
            poll_s=0.01,
            threshold=0.0,
            stable_polls_required=2,
        )
    )
    assert tight.stable is False
    assert tight.last_diff > 0.0

    # Loose threshold (0.5) → small diff looks identical → stabilizes immediately.
    state["i"] = 0
    loose_sleeps: list[float] = []
    loose = asyncio.run(
        wait_for_screen_settle(
            _toggling,
            max_wait_s=1.0,
            min_wait_s=0.0,
            poll_s=0.05,
            threshold=0.5,
            stable_polls_required=2,
            sleeper=_make_sleeper(loose_sleeps),
        )
    )
    assert loose.stable is True
