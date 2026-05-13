"""Draw action markers on saved screenshots.

The Runs tab shows one screenshot per step. Without a marker, you can't tell
*at a glance* whether the agent clicked the right thing — you have to read the
JSONL action log and translate coordinates in your head. Annotations close that
gap: a red ring at the click point, a yellow line for drags, a top-banner for
typed text or keystrokes.

The original screenshot is preserved (for training data fidelity); annotated
copies are saved as `step_NNN_decided.jpg` alongside.
"""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw


CLICK_RING_RADIUS = 18
CLICK_RING_WIDTH = 3
DOUBLE_CLICK_INNER_RADIUS = 6
CLICK_COLOR = (255, 60, 60, 255)        # red
DOUBLE_CLICK_COLOR = (255, 140, 60, 255)  # orange
DRAG_COLOR = (255, 220, 0, 255)          # yellow
SCROLL_COLOR = (60, 180, 255, 255)       # blue
TYPE_COLOR = (90, 220, 90, 255)          # green
KEY_COLOR = (180, 90, 220, 255)          # purple
WAIT_COLOR = (160, 160, 160, 255)        # gray
BANNER_HEIGHT = 28
BANNER_BG = (0, 0, 0, 170)
BANNER_FG = (255, 255, 255, 255)


def _draw_ring(draw: ImageDraw.ImageDraw, x: int, y: int, color, *, radius: int = CLICK_RING_RADIUS, width: int = CLICK_RING_WIDTH) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def _draw_dot(draw: ImageDraw.ImageDraw, x: int, y: int, color, *, radius: int = 3) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _draw_banner(draw: ImageDraw.ImageDraw, width: int, label: str, color) -> None:
    draw.rectangle((0, 0, width, BANNER_HEIGHT), fill=BANNER_BG)
    draw.rectangle((0, 0, 6, BANNER_HEIGHT), fill=color)
    draw.text((12, 6), label, fill=BANNER_FG)


def _truncate(text: str, n: int = 60) -> str:
    text = text.replace("\n", "\\n")
    if len(text) > n:
        return text[: n - 1] + "…"
    return text


def draw_action_marker(image: Image.Image, kind: str, args: dict[str, Any]) -> Image.Image:
    """Return a new RGBA image with a marker for `kind` overlaid on `image`.

    Coordinates in `args` are interpreted in the input image's space — the
    planner's captured screenshot, not display pixels. The original image is
    not mutated.
    """
    base = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, _ = base.size

    if kind == "click":
        x, y = int(args.get("x", 0)), int(args.get("y", 0))
        _draw_ring(draw, x, y, CLICK_COLOR)
        _draw_dot(draw, x, y, CLICK_COLOR)
    elif kind == "double_click":
        x, y = int(args.get("x", 0)), int(args.get("y", 0))
        _draw_ring(draw, x, y, DOUBLE_CLICK_COLOR)
        _draw_dot(draw, x, y, DOUBLE_CLICK_COLOR, radius=DOUBLE_CLICK_INNER_RADIUS)
    elif kind == "drag":
        x1, y1 = int(args.get("x", 0)), int(args.get("y", 0))
        x2, y2 = int(args.get("x2", x1)), int(args.get("y2", y1))
        draw.line((x1, y1, x2, y2), fill=DRAG_COLOR, width=4)
        _draw_dot(draw, x1, y1, DRAG_COLOR, radius=5)
        _draw_dot(draw, x2, y2, DRAG_COLOR, radius=5)
    elif kind == "scroll":
        dx = int(args.get("dx", 0))
        dy = int(args.get("dy", 0))
        label = f"SCROLL  dx={dx}  dy={dy}"
        _draw_banner(draw, width, label, SCROLL_COLOR)
    elif kind == "type":
        text = str(args.get("text", ""))
        _draw_banner(draw, width, f"TYPE  {_truncate(text)}", TYPE_COLOR)
    elif kind == "key":
        name = str(args.get("name", ""))
        mods = "+".join(args.get("modifiers", []) or [])
        combo = f"{mods}+{name}" if mods else name
        _draw_banner(draw, width, f"KEY  {combo}", KEY_COLOR)
    elif kind == "wait":
        ms = int(args.get("ms", 0))
        _draw_banner(draw, width, f"WAIT  {ms}ms", WAIT_COLOR)
    else:
        _draw_banner(draw, width, f"{kind.upper()}", WAIT_COLOR)

    return Image.alpha_composite(base, overlay)
