from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nalu.agents.planner.planner import dispatch_action
from nalu.agents.vision import Action


@dataclass
class FakeShot:
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass
class FakeActuator:
    calls: list[tuple] = field(default_factory=list)

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y, button, clicks))

    def drag(self, x1, y1, x2, y2):
        self.calls.append(("drag", x1, y1, x2, y2))

    def type_text(self, text):
        self.calls.append(("type_text", text))

    def key(self, name, modifiers=()):
        self.calls.append(("key", name, tuple(modifiers)))

    def scroll(self, dx, dy):
        self.calls.append(("scroll", dx, dy))


def test_click_routes_with_scaled_coords():
    act = FakeActuator()
    dispatch_action(Action(kind="click", args={"x": 100, "y": 200}), FakeShot(scale_x=2.0, scale_y=2.0), act)
    assert act.calls == [("click", 200, 400, "left", 1)]


def test_click_respects_button_and_clicks_args():
    act = FakeActuator()
    dispatch_action(Action(kind="click", args={"x": 10, "y": 20, "button": "right", "clicks": 3}), FakeShot(), act)
    assert act.calls == [("click", 10, 20, "right", 3)]


def test_double_click_routes_to_click_with_clicks_two():
    act = FakeActuator()
    dispatch_action(Action(kind="double_click", args={"x": 50, "y": 60}), FakeShot(), act)
    assert act.calls == [("click", 50, 60, "left", 2)]


def test_double_click_scales_coords():
    act = FakeActuator()
    dispatch_action(Action(kind="double_click", args={"x": 100, "y": 100}), FakeShot(scale_x=1.5, scale_y=2.0), act)
    assert act.calls == [("click", 150, 200, "left", 2)]


def test_drag_routes_both_endpoints_scaled():
    act = FakeActuator()
    dispatch_action(
        Action(kind="drag", args={"x1": 10, "y1": 20, "x2": 30, "y2": 40}),
        FakeShot(scale_x=2.0, scale_y=2.0),
        act,
    )
    assert act.calls == [("drag", 20, 40, 60, 80)]


def test_type_passes_text_through():
    act = FakeActuator()
    dispatch_action(Action(kind="type", args={"text": "hello world"}), FakeShot(), act)
    assert act.calls == [("type_text", "hello world")]


def test_type_coerces_non_string_text():
    act = FakeActuator()
    dispatch_action(Action(kind="type", args={"text": 42}), FakeShot(), act)
    assert act.calls == [("type_text", "42")]


def test_key_with_no_modifiers_defaults_to_empty_list():
    act = FakeActuator()
    dispatch_action(Action(kind="key", args={"name": "enter"}), FakeShot(), act)
    assert act.calls == [("key", "enter", ())]


def test_key_with_modifiers():
    act = FakeActuator()
    dispatch_action(Action(kind="key", args={"name": "c", "modifiers": ["cmd"]}), FakeShot(), act)
    assert act.calls == [("key", "c", ("cmd",))]


def test_scroll_with_int_args():
    act = FakeActuator()
    dispatch_action(Action(kind="scroll", args={"dx": 0, "dy": -100}), FakeShot(), act)
    assert act.calls == [("scroll", 0, -100)]


def test_scroll_defaults_missing_axes_to_zero():
    act = FakeActuator()
    dispatch_action(Action(kind="scroll", args={}), FakeShot(), act)
    assert act.calls == [("scroll", 0, 0)]


def test_wait_does_not_call_actuator(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("nalu.agents.planner.planner.time.sleep", lambda s: sleeps.append(s))
    act = FakeActuator()
    dispatch_action(Action(kind="wait", args={"ms": 250}), FakeShot(), act)
    assert act.calls == []
    assert sleeps == [0.25]


def test_wait_clamps_to_five_seconds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("nalu.agents.planner.planner.time.sleep", lambda s: sleeps.append(s))
    dispatch_action(Action(kind="wait", args={"ms": 60_000}), FakeShot(), FakeActuator())
    assert sleeps == [5.0]


def test_error_kind_raises():
    with pytest.raises(RuntimeError, match="parser said no"):
        dispatch_action(Action(kind="error", args={}, reason="parser said no"), FakeShot(), FakeActuator())


def test_unknown_kind_raises_value_error():
    with pytest.raises(ValueError, match="unknown action: teleport"):
        dispatch_action(Action(kind="teleport", args={}), FakeShot(), FakeActuator())
