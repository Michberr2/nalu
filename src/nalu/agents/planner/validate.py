"""Refuse impossible-on-the-current-screen actions before they reach the actuator.

UI-TARS occasionally hallucinates click coordinates that fall outside the
captured image — usually a few pixels past the edge, sometimes wildly wrong
(x=4096 on a 1920-wide capture). The actuator scales those into Cocoa
coordinates and clicks anyway, the click lands at the screen edge or a random
neighbouring pixel, and the no-effect detector then has to clean up the mess
on the next turn.

Cheaper to catch it here: validate the args, refuse out-of-bounds clicks /
drags, append a self-correction hint to the action history, and let the model
retake the screenshot with the hint in context. Pure-Python, no PIL/actuator
dependency — caller passes the image's width/height.
"""
from __future__ import annotations

from dataclasses import dataclass


_COORD_KINDS = ("click", "double_click", "drag")


@dataclass(frozen=True)
class RefusalSignal:
    reason: str
    hint: str


def _coerce_int(value) -> int | None:
    if isinstance(value, bool):  # bool is a subclass of int — keep it out
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _check_point(x_raw, y_raw, width: int, height: int, label: str) -> RefusalSignal | None:
    x = _coerce_int(x_raw)
    y = _coerce_int(y_raw)
    if x is None or y is None:
        return RefusalSignal(
            reason="missing_coords",
            hint=f"{label} args missing numeric x/y (got x={x_raw!r}, y={y_raw!r}). "
            "Re-look at the screenshot and emit integer pixel coordinates.",
        )
    if x < 0 or y < 0 or x >= width or y >= height:
        return RefusalSignal(
            reason="out_of_bounds",
            hint=f"{label} at ({x}, {y}) is outside the {width}×{height} screenshot. "
            "Pick a coordinate inside the visible frame.",
        )
    return None


def validate_action(kind: str, args: dict, width: int, height: int) -> RefusalSignal | None:
    """Return a `RefusalSignal` if `kind`+`args` can't be dispatched on a `width`×`height` image.

    Returns `None` when the action is fine. Coord kinds (click/double_click/drag)
    are bounds-checked; argument-bearing non-coord kinds (type/key) are checked
    for the args they actually need so dispatch can never KeyError.
    """
    if not isinstance(args, dict):
        return RefusalSignal(
            reason="bad_args",
            hint=f"{kind} args must be a dict; got {type(args).__name__}.",
        )
    if kind in _COORD_KINDS:
        if kind in ("click", "double_click"):
            return _check_point(args.get("x"), args.get("y"), width, height, kind)
        # drag: validate both endpoints
        start = _check_point(args.get("x1"), args.get("y1"), width, height, "drag start")
        if start is not None:
            return start
        return _check_point(args.get("x2"), args.get("y2"), width, height, "drag end")
    if kind == "type":
        text = args.get("text")
        if not isinstance(text, str) or not text:
            return RefusalSignal(
                reason="missing_text",
                hint="type args must include a non-empty `text` string. "
                "Emit `type(content='...')` with the actual text you want typed.",
            )
        return None
    if kind == "key":
        name = args.get("name")
        if not isinstance(name, str) or not name:
            return RefusalSignal(
                reason="missing_key_name",
                hint="key args must include `name` (e.g. `enter`, `space`, `a`). "
                "Emit `hotkey(key='cmd space')` or `hotkey(key='enter')`.",
            )
        return None
    return None
