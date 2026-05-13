from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from nalu.agents.planner.annotate import (
    BANNER_HEIGHT,
    CLICK_RING_RADIUS,
    draw_action_marker,
)


def _blank(width: int = 320, height: int = 240, value: int = 0) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8), mode="RGB")


def _diff_count(a: Image.Image, b: Image.Image) -> int:
    aa = np.asarray(a.convert("RGB"), dtype=np.int16)
    bb = np.asarray(b.convert("RGB"), dtype=np.int16)
    return int(np.sum(np.any(aa != bb, axis=-1)))


def test_returns_new_image_without_mutating_original():
    src = _blank()
    src_copy = src.copy()
    out = draw_action_marker(src, "click", {"x": 100, "y": 100})
    assert _diff_count(src, src_copy) == 0  # original untouched
    assert out.size == src.size
    assert _diff_count(src, out) > 0


def test_click_marker_localized_around_target_point():
    src = _blank(width=400, height=300)
    out = draw_action_marker(src, "click", {"x": 200, "y": 150})
    # crop a tight window around the click and assert most diff is inside it
    pad = CLICK_RING_RADIUS + 4
    inside = _diff_count(
        src.crop((200 - pad, 150 - pad, 200 + pad, 150 + pad)),
        out.crop((200 - pad, 150 - pad, 200 + pad, 150 + pad)),
    )
    total = _diff_count(src, out)
    assert total > 0
    assert inside / total > 0.9


def test_double_click_renders_inner_dot_in_addition_to_ring():
    src = _blank()
    single = draw_action_marker(src, "click", {"x": 100, "y": 100})
    double = draw_action_marker(src, "double_click", {"x": 100, "y": 100})
    # Larger inner-fill than a single click
    assert _diff_count(src, double) > _diff_count(src, single)


def test_drag_draws_line_between_endpoints():
    src = _blank(width=400, height=200)
    out = draw_action_marker(src, "drag", {"x": 50, "y": 50, "x2": 350, "y2": 150})
    midpoint = _diff_count(
        src.crop((195, 95, 205, 105)),
        out.crop((195, 95, 205, 105)),
    )
    assert midpoint > 0


def test_type_renders_top_banner():
    src = _blank()
    out = draw_action_marker(src, "type", {"text": "hello world"})
    inside_banner = _diff_count(
        src.crop((0, 0, src.width, BANNER_HEIGHT)),
        out.crop((0, 0, src.width, BANNER_HEIGHT)),
    )
    below_banner = _diff_count(
        src.crop((0, BANNER_HEIGHT + 4, src.width, src.height)),
        out.crop((0, BANNER_HEIGHT + 4, src.width, src.height)),
    )
    assert inside_banner > 0
    assert below_banner == 0


def test_type_truncates_long_text_without_crashing():
    src = _blank()
    long_text = "x" * 500
    out = draw_action_marker(src, "type", {"text": long_text})
    assert out.size == src.size


def test_key_banner_includes_modifiers():
    src = _blank()
    out = draw_action_marker(src, "key", {"name": "space", "modifiers": ["cmd"]})
    assert _diff_count(src, out) > 0


def test_scroll_renders_banner():
    src = _blank()
    out = draw_action_marker(src, "scroll", {"dx": 0, "dy": -120})
    inside_banner = _diff_count(
        src.crop((0, 0, src.width, BANNER_HEIGHT)),
        out.crop((0, 0, src.width, BANNER_HEIGHT)),
    )
    assert inside_banner > 0


def test_wait_renders_banner():
    src = _blank()
    out = draw_action_marker(src, "wait", {"ms": 500})
    assert _diff_count(src, out) > 0


def test_unknown_kind_renders_fallback_banner():
    src = _blank()
    out = draw_action_marker(src, "unknown", {})
    assert _diff_count(src, out) > 0


def test_click_outside_image_does_not_crash():
    src = _blank(width=200, height=200)
    out = draw_action_marker(src, "click", {"x": 9999, "y": -50})
    assert out.size == src.size


def test_returns_rgba_compatible_image():
    src = _blank()
    out = draw_action_marker(src, "click", {"x": 10, "y": 10})
    rgb = out.convert("RGB")
    assert rgb.size == src.size
