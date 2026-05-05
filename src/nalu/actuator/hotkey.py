from __future__ import annotations

import threading
from typing import Callable

from pynput import keyboard

from .. import config


class PauseController:
    """Global hotkey toggle. paused defaults to False; can be set externally."""

    def __init__(self, on_change: Callable[[bool], None] | None = None):
        self._paused = False
        self._lock = threading.Lock()
        self._on_change = on_change
        self._listener: keyboard.GlobalHotKeys | None = None

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def set(self, value: bool) -> None:
        with self._lock:
            changed = self._paused != value
            self._paused = value
        if changed and self._on_change:
            self._on_change(value)

    def toggle(self) -> None:
        self.set(not self.paused)

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.GlobalHotKeys({config.PAUSE_HOTKEY: self.toggle})
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
