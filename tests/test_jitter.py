from __future__ import annotations

import random

from nalu.agents.planner.jitter import JITTER_MAX_PX, jitter_click_args


W, H = 1200, 800


def test_jitter_returns_copy_does_not_mutate_input():
    args = {"x": 100, "y": 200, "button": "left"}
    snapshot = dict(args)
    out = jitter_click_args(args, W, H, rng=random.Random(0))
    assert args == snapshot
    assert out is not args


def test_jitter_passes_through_non_xy_args():
    out = jitter_click_args({"x": 50, "y": 60, "button": "right", "clicks": 2}, W, H, rng=random.Random(0))
    assert out["button"] == "right"
    assert out["clicks"] == 2


def test_jitter_within_max_offset():
    rng = random.Random(7)
    for _ in range(50):
        out = jitter_click_args({"x": 600, "y": 400}, W, H, rng=rng)
        assert abs(out["x"] - 600) <= JITTER_MAX_PX
        assert abs(out["y"] - 400) <= JITTER_MAX_PX


def test_jitter_respects_custom_max_offset():
    rng = random.Random(7)
    for _ in range(50):
        out = jitter_click_args({"x": 600, "y": 400}, W, H, max_offset_px=2, rng=rng)
        assert abs(out["x"] - 600) <= 2
        assert abs(out["y"] - 400) <= 2


def test_jitter_clamps_to_image_bounds_at_top_left():
    out = jitter_click_args({"x": 0, "y": 0}, W, H, rng=random.Random(0))
    assert 0 <= out["x"] < W
    assert 0 <= out["y"] < H


def test_jitter_clamps_to_image_bounds_at_bottom_right():
    out = jitter_click_args({"x": W - 1, "y": H - 1}, W, H, rng=random.Random(1))
    assert 0 <= out["x"] < W
    assert 0 <= out["y"] < H


def test_jitter_is_deterministic_for_same_seed():
    a = jitter_click_args({"x": 100, "y": 100}, W, H, rng=random.Random(42))
    b = jitter_click_args({"x": 100, "y": 100}, W, H, rng=random.Random(42))
    assert a == b


def test_jitter_forces_movement_even_on_zero_zero_roll():
    # A stub RNG that always rolls 0 — without the force-movement guard,
    # the jittered point would equal the original and we'd reproduce the miss.
    class ZeroRng:
        def randint(self, a, b):
            return 0

        def random(self):
            return 0.4  # < 0.5 → +1 dx branch

    out = jitter_click_args({"x": 100, "y": 100}, W, H, rng=ZeroRng())
    assert (out["x"], out["y"]) != (100, 100)


def test_jitter_passes_through_when_xy_missing():
    args = {"button": "left", "clicks": 1}
    out = jitter_click_args(args, W, H, rng=random.Random(0))
    assert out == args
    assert out is not args  # still a copy


def test_jitter_handles_only_y_missing():
    args = {"x": 100}
    out = jitter_click_args(args, W, H, rng=random.Random(0))
    assert out == args


def test_jitter_coerces_non_int_x_y_to_int_after_offset():
    out = jitter_click_args({"x": 100.7, "y": 50.2}, W, H, rng=random.Random(0))
    assert isinstance(out["x"], int) and isinstance(out["y"], int)


def test_jitter_at_image_dim_minus_one_does_not_cross_bound():
    # If original is at edge and the offset goes positive, clamp must hold.
    rng = random.Random(0)
    for _ in range(20):
        out = jitter_click_args({"x": W - 1, "y": H - 1}, W, H, rng=rng)
        assert out["x"] < W and out["y"] < H
