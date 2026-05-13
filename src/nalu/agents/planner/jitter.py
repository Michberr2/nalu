"""Cheap retry for clicks that produced no observable effect.

The no-effect detector already catches the case where a click landed on a
non-interactive pixel (model missed the button by a few px, the toolbar's
hit-region is ~20px off, etc.). The default response — append a hint and
let the model retake the screenshot — wastes a vision turn on what's
essentially a 4-pixel correction.

This module provides a one-shot jittered retry: nudge the click by a small
random offset (≤8 px by default), re-dispatch immediately, and only fall
back to the model if the jitter also produced nothing. Pure-Python so it's
testable without Quartz; planner is responsible for actually dispatching
the returned args.
"""
from __future__ import annotations

import random


JITTER_MAX_PX = 8


def jitter_click_args(
    args: dict,
    width: int,
    height: int,
    *,
    max_offset_px: int = JITTER_MAX_PX,
    rng: random.Random | None = None,
) -> dict:
    """Return a copy of `args` with `x` and `y` nudged by ±max_offset_px and clamped to [0, dim).

    Caller passes the captured image's `width`/`height` so the jittered point
    can't land outside the frame. The clamp is `dim - 1` (inclusive upper
    bound) to match `validate_action`'s strict `< dim` check.

    Other args (button, clicks, x1/y1/x2/y2 for drag) are passed through
    unchanged. Drag jittering is not supported — drags rarely miss the same
    way, and shifting both endpoints by the same offset would just translate
    the gesture rather than retarget it.
    """
    if "x" not in args or "y" not in args:
        return dict(args)
    rng = rng if rng is not None else random.Random()
    out = dict(args)
    dx = rng.randint(-max_offset_px, max_offset_px)
    dy = rng.randint(-max_offset_px, max_offset_px)
    if dx == 0 and dy == 0:
        # Force at least one pixel of movement so we don't reproduce the same miss.
        dx = 1 if rng.random() < 0.5 else -1
    out["x"] = max(0, min(int(width) - 1, int(args["x"]) + dx))
    out["y"] = max(0, min(int(height) - 1, int(args["y"]) + dy))
    return out
