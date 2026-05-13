from __future__ import annotations

import time
from typing import Iterable

from .hotkey import PauseController


class ActionRefused(RuntimeError):
    pass


_KEY_MAP = {
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51,
    "escape": 53, "esc": 53, "left": 123, "right": 124, "down": 125, "up": 126,
    "command": 55, "cmd": 55, "shift": 56, "option": 58, "alt": 58, "control": 59, "ctrl": 59,
}

_MODIFIER_FLAGS = {
    "cmd": 1 << 20, "command": 1 << 20,
    "shift": 1 << 17,
    "option": 1 << 19, "alt": 1 << 19,
    "ctrl": 1 << 18, "control": 1 << 18,
}


class Actuator:
    def __init__(self, pause: PauseController):
        self._pause = pause

    def _check(self) -> None:
        if self._pause.paused:
            raise ActionRefused("paused — press hotkey to resume")

    def move(self, x: int, y: int) -> None:
        self._check()
        import Quartz
        ev = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self._check()
        import Quartz
        btn = {"left": Quartz.kCGMouseButtonLeft, "right": Quartz.kCGMouseButtonRight}[button]
        down = Quartz.kCGEventLeftMouseDown if button == "left" else Quartz.kCGEventRightMouseDown
        up = Quartz.kCGEventLeftMouseUp if button == "left" else Quartz.kCGEventRightMouseUp
        for i in range(clicks):
            self._check()
            for kind in (down, up):
                ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), btn)
                Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventClickState, i + 1)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            time.sleep(0.04)

    def drag(self, x1: int, y1: int, x2: int, y2: int, steps: int = 20) -> None:
        """Press at (x1,y1), drag to (x2,y2), release. `steps` controls smoothness."""
        self._check()
        import Quartz
        btn = Quartz.kCGMouseButtonLeft
        # Press
        ev_down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, (x1, y1), btn)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)
        # Intermediate drag points so the OS sees a real drag, not a teleport-and-release.
        steps = max(1, int(steps))
        for i in range(1, steps + 1):
            self._check()
            t = i / steps
            ix = int(x1 + (x2 - x1) * t)
            iy = int(y1 + (y2 - y1) * t)
            ev = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDragged, (ix, iy), btn)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            time.sleep(0.01)
        # Release
        ev_up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, (x2, y2), btn)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)

    def scroll(self, dx: int, dy: int) -> None:
        self._check()
        import Quartz
        ev = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, int(dy), int(dx))
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def type_text(self, text: str) -> None:
        self._check()
        import Quartz
        for ch in text:
            self._check()
            ev_down = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
            ev_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
            Quartz.CGEventKeyboardSetUnicodeString(ev_down, len(ch), ch)
            Quartz.CGEventKeyboardSetUnicodeString(ev_up, len(ch), ch)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)
            time.sleep(0.01)

    def key(self, name: str, modifiers: Iterable[str] = ()) -> None:
        self._check()
        import Quartz
        code = _KEY_MAP.get(name.lower())
        if code is None:
            raise ValueError(f"unknown key: {name}")
        flags = 0
        for m in modifiers:
            flags |= _MODIFIER_FLAGS.get(m.lower(), 0)
        for is_down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(None, code, is_down)
            if flags:
                Quartz.CGEventSetFlags(ev, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.02)
