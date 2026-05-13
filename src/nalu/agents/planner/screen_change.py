"""Detect whether an action visibly changed the screen.

The vision model occasionally misses its target — it clicks 8 px off a button,
or types into a window that's lost focus. The screen looks identical afterward,
but the model's internal narrative ("I clicked the OK button") is decoupled
from reality. Without a feedback signal, the next perceive→reason→act turn
inherits that wrong narrative.

`perceptual_diff` downsamples both frames to a small grayscale grid and
returns the mean absolute pixel diff in [0, 1]. The downsample makes the
comparison robust to harmless cursor movement, font anti-aliasing jitter,
and 1-pixel layout shifts; we want a *semantic* "did anything happen" signal,
not a pixel-perfect equality check.

The planner uses this to inject a hint into the action history when an
effect-bearing action (click, type, key, scroll) produces no observable
change — letting the model self-correct on the next turn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


DEFAULT_DOWNSAMPLE = 64
DEFAULT_CHANGE_THRESHOLD = 0.005  # mean pixel diff (normalized 0..1)
EFFECT_BEARING_KINDS = ("click", "double_click", "type", "key", "scroll", "drag")


@dataclass
class NoEffectSignal:
    diff: float
    threshold: float
    action_kind: str
    hint: str


def perceptual_diff(
    before: Image.Image, after: Image.Image, downsample: int = DEFAULT_DOWNSAMPLE
) -> float:
    """Mean absolute pixel diff between two frames in [0, 1].

    Both frames are converted to grayscale and bilinear-downsampled to
    `downsample x downsample` before comparison.
    """
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    a = np.asarray(
        before.convert("L").resize((downsample, downsample), Image.BILINEAR),
        dtype=np.float32,
    )
    b = np.asarray(
        after.convert("L").resize((downsample, downsample), Image.BILINEAR),
        dtype=np.float32,
    )
    return float(np.mean(np.abs(a - b)) / 255.0)


def is_changed(
    before: Image.Image, after: Image.Image,
    *, threshold: float = DEFAULT_CHANGE_THRESHOLD,
    downsample: int = DEFAULT_DOWNSAMPLE,
) -> bool:
    return perceptual_diff(before, after, downsample=downsample) > threshold


def is_effect_bearing(action_kind: str) -> bool:
    return action_kind in EFFECT_BEARING_KINDS


def evaluate_action_effect(
    action_kind: str, before: Image.Image, after: Image.Image,
    *, threshold: float = DEFAULT_CHANGE_THRESHOLD,
    downsample: int = DEFAULT_DOWNSAMPLE,
) -> NoEffectSignal | None:
    """Return a `NoEffectSignal` if an effect-bearing action produced no diff, else None."""
    if not is_effect_bearing(action_kind):
        return None
    diff = perceptual_diff(before, after, downsample=downsample)
    if diff > threshold:
        return None
    hint = (
        f"Your previous `{action_kind}` action produced no observable change on the screen "
        f"(pixel diff {diff:.4f} <= {threshold}). The action likely missed its target — "
        f"re-examine the screenshot, pick a more accurate location, or try a different approach."
    )
    return NoEffectSignal(diff=diff, threshold=threshold, action_kind=action_kind, hint=hint)
