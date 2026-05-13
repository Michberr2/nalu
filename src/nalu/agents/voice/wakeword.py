"""Local "Hey Nalu" wake-word detection.

Uses [openwakeword](https://github.com/dscripka/openwakeword) — small ONNX models that
run ~1% CPU on Apple Silicon. The pretrained `hey_jarvis` keyword fits the persona
(see project memory: "Jarvis-from-Iron-Man style"); users can swap to a custom-trained
"hey_nalu" model via the `model` constructor arg.

Architecture:
- `WakeWordSpotter` is the abstract scoring interface (single method: predict).
- `OpenWakeWordSpotter` is the concrete openwakeword wrapper (lazy-imported).
- `WakeWordRunner` owns the mic loop, threshold, and cooldown — easily tested with
  a stub spotter without ever touching the audio device.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, Protocol


# openwakeword chunks are 80 ms of 16 kHz mono = 1280 samples.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_SAMPLES = 1280
DEFAULT_THRESHOLD = 0.5
DEFAULT_COOLDOWN_S = 2.0
DEFAULT_KEYWORD = "hey_jarvis"


class WakeWordSpotter(Protocol):
    """Anything that maps a 1280-sample int16 frame to per-keyword scores."""

    def predict(self, frame) -> dict[str, float]: ...


class OpenWakeWordSpotter:
    """Concrete spotter backed by openwakeword."""

    def __init__(self, models: Iterable[str] | None = None):
        self._model = None
        self._models = list(models) if models else [DEFAULT_KEYWORD]

    def load(self) -> None:
        if self._model is not None:
            return
        from openwakeword.model import Model  # type: ignore

        self._model = Model(wakeword_models=self._models)

    def predict(self, frame) -> dict[str, float]:
        self.load()
        return self._model.predict(frame)


class WakeWordRunner:
    """Polls a spotter and fires `on_wake(keyword, score)` once threshold is crossed.

    The mic loop is threaded; pass a stub `frame_source` (an iterable of frames)
    in tests to drive deterministic scenarios.
    """

    def __init__(
        self,
        on_wake: Callable[[str, float], None],
        spotter: WakeWordSpotter | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        frame_source: Iterable | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_samples: int = DEFAULT_FRAME_SAMPLES,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._on_wake = on_wake
        self._spotter = spotter or OpenWakeWordSpotter()
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._frame_source = frame_source
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._clock = clock
        self._last_fire_ts: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    def feed(self, frame) -> tuple[str, float] | None:
        """Feed one audio frame, return `(keyword, score)` if it triggered, else None.

        Honors threshold and cooldown. Pure synchronous — used by tests and by the
        background thread.
        """
        scores = self._spotter.predict(frame)
        if not scores:
            return None
        keyword, score = max(scores.items(), key=lambda kv: kv[1])
        if score < self._threshold:
            return None
        now = self._clock()
        if self._last_fire_ts is not None and now - self._last_fire_ts < self._cooldown_s:
            return None
        self._last_fire_ts = now
        try:
            self._on_wake(keyword, float(score))
        except Exception:
            pass
        return keyword, float(score)

    def warm(self) -> None:
        if hasattr(self._spotter, "load"):
            self._spotter.load()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        if self._frame_source is not None:
            for frame in self._frame_source:
                if self._stop.is_set():
                    return
                self.feed(frame)
            return
        self._run_microphone()

    def _run_microphone(self) -> None:
        import numpy as np
        import sounddevice as sd

        with sd.InputStream(
            channels=1,
            samplerate=self._sample_rate,
            dtype="int16",
            blocksize=self._frame_samples,
        ) as stream:
            while not self._stop.is_set():
                data, _overflow = stream.read(self._frame_samples)
                frame = np.frombuffer(data, dtype=np.int16)
                self.feed(frame)
