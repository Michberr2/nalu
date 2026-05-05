from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import config


class STT:
    def __init__(self, model_size: str = config.STT_MODEL):
        self.model_size = model_size
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        cache_dir = config.MODELS_DIR / "whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8",
            download_root=str(cache_dir),
        )

    def transcribe_array(self, samples: np.ndarray, sample_rate: int) -> str:
        self.load()
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32) / 32768.0
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            n = int(len(samples) * ratio)
            xs = np.linspace(0, len(samples) - 1, n).astype(np.int64)
            samples = samples[xs]
        segments, _info = self._model.transcribe(samples, language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()

    def transcribe_file(self, path: Path) -> str:
        self.load()
        segments, _info = self._model.transcribe(str(path), language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()


def record(duration_s: float, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    import sounddevice as sd

    audio = sd.rec(int(duration_s * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten(), sample_rate
