from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from pynput import keyboard

from ... import config
from .stt import STT, record


class PushToTalk:
    """Tap-to-talk: hotkey press records a fixed window, then runs STT.

    Single-tap UX (not press-and-hold) so users don't need to keep
    Ctrl+Opt+Cmd+Space depressed while speaking. Concurrent triggers
    are dropped while a capture is in flight.
    """

    def __init__(
        self,
        on_transcript: Callable[[str], None],
        seconds: float = config.PTT_RECORD_SECONDS,
        on_listening: Callable[[bool], None] | None = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_listening = on_listening
        self._seconds = seconds
        self._stt = STT()
        self._lock = threading.Lock()
        self._busy = False
        self._listener: keyboard.GlobalHotKeys | None = None

    def trigger(self) -> None:
        """Start a record+transcribe cycle if one isn't already in flight."""
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._capture, daemon=True).start()

    _trigger = trigger  # back-compat alias for the hotkey callback

    def _capture(self) -> None:
        try:
            if self._on_listening:
                self._on_listening(True)
            samples, sr = record(self._seconds)
            if self._on_listening:
                self._on_listening(False)
            text = self._stt.transcribe_array(samples, sr).strip()
            if text:
                self._on_transcript(text)
        finally:
            with self._lock:
                self._busy = False

    def warm(self) -> None:
        """Pre-load the STT model so the first tap is instant."""
        self._stt.load()

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.GlobalHotKeys({config.PUSH_TO_TALK_HOTKEY: self._trigger})
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
