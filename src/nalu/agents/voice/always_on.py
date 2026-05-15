"""Always-on STT with VAD.

Push-to-talk is precise but transactional: the user has to remember the hotkey
every time, and short interjections ("yes" "stop" "open mail") cost a full
hotkey + record cycle. An always-on path complements PTT by letting the mic
listen continuously, gate speech through a voice-activity detector, and
transcribe each detected utterance the moment silence resumes.

This module is wiring, not models. It accepts injected:
  * `audio_source` — yields (np.int16 samples, sample_rate) chunks every ~30ms
  * `vad_fn`       — chunk → bool (or probability; threshold applied here)
  * `transcribe_fn` — accumulated samples → text
  * `is_muted`     — bool callable; when True we drop incoming audio (barge-in)
  * `on_transcript`— callable invoked with the final transcribed string

This decoupling means tests can drive the whole runner without sounddevice,
silero-vad, or faster-whisper. In `daemon.py` these are wired to:
  * `sounddevice.RawInputStream` chunks
  * silero-vad's torch model
  * `STT.transcribe_array`
  * the TTS state, so we don't transcribe Nalu speaking to itself

Barge-in is implemented as a *drop* policy here — the runner discards audio
arriving while `is_muted` is True. Stopping the in-flight TTS playback is the
caller's job (the daemon owns `sd.play` / `sd.stop`).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import structlog


log = structlog.get_logger("always_on")


# Tunables — exposed for tests, overridable via constructor.
DEFAULT_CHUNK_MS = 30  # silero-vad operates well on 16kHz, 30ms windows
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_MS = 200  # below this we treat the burst as noise
DEFAULT_END_OF_SPEECH_SILENCE_MS = 500
DEFAULT_MAX_UTTERANCE_MS = 15_000


AudioChunk = tuple[np.ndarray, int]  # (samples, sample_rate)
AudioSource = Callable[[], Optional[AudioChunk]]
VADFn = Callable[[np.ndarray, int], float]
TranscribeFn = Callable[[np.ndarray, int], str]
OnTranscript = Callable[[str], None]


@dataclass
class AlwaysOnConfig:
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS
    end_of_speech_silence_ms: int = DEFAULT_END_OF_SPEECH_SILENCE_MS
    max_utterance_ms: int = DEFAULT_MAX_UTTERANCE_MS


class _UtteranceBuffer:
    """Accumulates speech samples until the runner decides the utterance ended."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self.ms = 0

    def append(self, samples: np.ndarray, chunk_ms: int) -> None:
        self._chunks.append(samples)
        self.ms += chunk_ms

    def clear(self) -> None:
        self._chunks = []
        self.ms = 0

    def to_array(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._chunks)


class AlwaysOnRunner:
    """VAD-gated continuous transcription.

    `step()` consumes one chunk from `audio_source` and advances the state
    machine. `run_forever()` loops in a background thread until `stop()`.
    Tests should call `step()` directly with a deterministic audio source.

    State machine:
        IDLE → speech detected → SPEAKING (accumulate)
        SPEAKING → silence_ms ≥ end_of_speech_silence_ms → FLUSH (transcribe)
        SPEAKING → utterance_ms ≥ max_utterance_ms → FLUSH (force-cut)
        ANY → is_muted=True → drop chunk, reset SPEAKING
    """

    STATE_IDLE = "idle"
    STATE_SPEAKING = "speaking"

    def __init__(
        self,
        audio_source: AudioSource,
        vad_fn: VADFn,
        transcribe_fn: TranscribeFn,
        on_transcript: OnTranscript,
        *,
        is_muted: Callable[[], bool] | None = None,
        config: AlwaysOnConfig | None = None,
        chunk_ms: int = DEFAULT_CHUNK_MS,
    ) -> None:
        self._source = audio_source
        self._vad = vad_fn
        self._transcribe = transcribe_fn
        self._on_transcript = on_transcript
        self._is_muted = is_muted or (lambda: False)
        self.config = config or AlwaysOnConfig()
        self.chunk_ms = chunk_ms

        self._state = self.STATE_IDLE
        self._buffer: _UtteranceBuffer | None = None
        self._silence_ms = 0
        self._sample_rate: int | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Last N transcripts kept for observability / tests.
        self.transcripts: deque[str] = deque(maxlen=32)

    @property
    def state(self) -> str:
        return self._state

    @property
    def utterance_ms(self) -> int:
        return self._buffer.ms if self._buffer is not None else 0

    def step(self) -> bool:
        """Consume one chunk. Returns False if the source is exhausted."""
        chunk = self._source()
        if chunk is None:
            return False
        samples, sr = chunk
        if self._sample_rate is None:
            self._sample_rate = sr
        if self._is_muted():
            # Drop audio while TTS is speaking; reset any partial utterance so we
            # don't ship a fragment when the user gets through.
            if self._state == self.STATE_SPEAKING:
                self._reset_state()
            return True

        prob = float(self._vad(samples, sr))
        is_speech = prob >= self.config.vad_threshold

        if is_speech:
            if self._state == self.STATE_IDLE:
                self._state = self.STATE_SPEAKING
                self._buffer = _UtteranceBuffer(sr)
            self._buffer.append(samples, self.chunk_ms)  # type: ignore[union-attr]
            self._silence_ms = 0
            if self._buffer.ms >= self.config.max_utterance_ms:  # type: ignore[union-attr]
                self._flush(force_cut=True)
            return True

        # Silence
        if self._state == self.STATE_SPEAKING:
            # Tail-pad silence into the utterance to keep word endings intact.
            self._buffer.append(samples, self.chunk_ms)  # type: ignore[union-attr]
            self._silence_ms += self.chunk_ms
            if self._silence_ms >= self.config.end_of_speech_silence_ms:
                self._flush(force_cut=False)
        return True

    def _flush(self, *, force_cut: bool) -> None:
        if self._buffer is None or self._sample_rate is None:
            self._reset_state()
            return
        if self._buffer.ms < self.config.min_speech_ms and not force_cut:
            # Too short to bother transcribing — likely a cough.
            log.info("utterance_dropped_too_short", ms=self._buffer.ms)
            self._reset_state()
            return
        samples = self._buffer.to_array()
        self._reset_state()
        try:
            text = self._transcribe(samples, self._sample_rate or 16000).strip()
        except Exception:
            log.exception("transcribe_failed")
            return
        if not text:
            return
        self.transcripts.append(text)
        try:
            self._on_transcript(text)
        except Exception:
            log.exception("on_transcript_handler_failed")

    def _reset_state(self) -> None:
        self._state = self.STATE_IDLE
        self._buffer = None
        self._silence_ms = 0

    def run_forever(self) -> None:
        while not self._stop.is_set():
            if not self.step():
                # No data this tick — yield rather than spin.
                time.sleep(self.chunk_ms / 1000.0)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def make_silero_vad_fn(sample_rate: int = 16000) -> VADFn:
    """Build a VAD callable backed by silero-vad. Lazy import so tests skip torch.

    Falls back to a permissive "everything is speech" function if silero-vad isn't
    installed — the always-on path then degrades to "transcribe everything"
    rather than failing to start. Daemon should log a warning in that case.
    """
    try:
        import torch
        from silero_vad import load_silero_vad
    except Exception as e:  # ImportError or torch init failure
        log.warning("silero_vad_unavailable", reason=str(e))
        return lambda samples, sr: 1.0

    model = load_silero_vad()

    def _vad(samples: np.ndarray, sr: int) -> float:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32) / 32768.0
        if sr != sample_rate:
            ratio = sample_rate / sr
            n = int(len(samples) * ratio)
            xs = np.linspace(0, len(samples) - 1, n).astype(np.int64)
            samples = samples[xs]
        t = torch.from_numpy(samples)
        with torch.no_grad():
            return float(model(t, sample_rate).item())

    return _vad


def chunked_sounddevice_source(
    chunk_ms: int = DEFAULT_CHUNK_MS,
    sample_rate: int = 16000,
) -> Iterator[AudioChunk]:
    """Generator that yields chunks from the default input device.

    Returned as an iterator the caller wraps in `next()` — `AlwaysOnRunner`
    expects a callable, so wrap with `lambda: next(it, None)`.
    """
    import sounddevice as sd

    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=samples_per_chunk,
    ) as stream:
        while True:
            data, _overflow = stream.read(samples_per_chunk)
            yield data.flatten(), sample_rate


__all__ = [
    "AlwaysOnRunner",
    "AlwaysOnConfig",
    "make_silero_vad_fn",
    "chunked_sounddevice_source",
    "DEFAULT_CHUNK_MS",
    "DEFAULT_VAD_THRESHOLD",
    "DEFAULT_MIN_SPEECH_MS",
    "DEFAULT_END_OF_SPEECH_SILENCE_MS",
    "DEFAULT_MAX_UTTERANCE_MS",
]
