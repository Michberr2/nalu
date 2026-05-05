from __future__ import annotations

import threading
import time
from typing import Optional

from .. import config
from .screen import Screenshot, capture_main_display


class ContinuousCapture:
    """Background thread that continuously captures the screen.
    Once started, the planner reads the latest frame instead of capturing
    per-step. Permission is granted once, capture is always-on."""

    def __init__(self, fps: float = config.CAPTURE_FPS, max_width: int = config.CAPTURE_MAX_WIDTH):
        self.fps = max(0.5, fps)
        self.max_width = max_width
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Screenshot] = None
        self._frame_count = 0
        self._last_capture_ms: float = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nalu-capture", daemon=True)
        self._thread.start()
        # Wait for first frame so callers don't race the first read.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self.latest_frame() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("ContinuousCapture failed to produce a frame within 5s — check Screen Recording permission")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                shot = capture_main_display(max_width=self.max_width)
                with self._lock:
                    self._latest = shot
                    self._frame_count += 1
                    self._last_capture_ms = (time.time() - t0) * 1000
            except Exception:
                # Don't kill the thread on transient errors; just back off briefly.
                time.sleep(0.5)
                continue
            elapsed = time.time() - t0
            sleep_for = max(0.0, period - elapsed)
            if self._stop.wait(sleep_for):
                break

    def latest_frame(self) -> Optional[Screenshot]:
        with self._lock:
            return self._latest

    def stats(self) -> dict:
        with self._lock:
            return {
                "frames": self._frame_count,
                "fps_target": self.fps,
                "last_capture_ms": round(self._last_capture_ms, 2),
                "running": self._thread is not None and self._thread.is_alive(),
            }
