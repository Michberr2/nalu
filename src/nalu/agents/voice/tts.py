from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from ... import config

PIPER_HF_REPO = "rhasspy/piper-voices"


def _voice_paths(voice: str) -> tuple[Path, Path]:
    lang = voice.split("_")[0]
    region = voice.split("_")[1].split("-")[0]
    quality = voice.rsplit("-", 1)[-1]
    name = voice
    base = config.MODELS_DIR / "piper" / voice
    return base / f"{name}.onnx", base / f"{name}.onnx.json"


def _ensure_voice(voice: str) -> tuple[Path, Path]:
    onnx, cfg = _voice_paths(voice)
    if onnx.exists() and cfg.exists():
        return onnx, cfg

    from huggingface_hub import hf_hub_download

    lang = voice.split("_")[0]
    region = voice.split("_")[1].split("-")[0]
    speaker = voice.split("-")[0].replace(f"{lang}_{region}-", "")
    quality = voice.rsplit("-", 1)[-1]
    subdir = f"{lang}/{lang}_{region}/{speaker}/{quality}"

    onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx_dl = hf_hub_download(PIPER_HF_REPO, f"{subdir}/{voice}.onnx", local_dir=str(onnx.parent.parent.parent))
    cfg_dl = hf_hub_download(PIPER_HF_REPO, f"{subdir}/{voice}.onnx.json", local_dir=str(onnx.parent.parent.parent))
    return Path(onnx_dl), Path(cfg_dl)


class TTS:
    def __init__(self, voice: str = config.TTS_VOICE):
        self.voice_name = voice
        self._voice = None
        self._sr: int | None = None

    def load(self) -> None:
        if self._voice is not None:
            return
        from piper.voice import PiperVoice

        onnx, _cfg = _ensure_voice(self.voice_name)
        self._voice = PiperVoice.load(str(onnx))
        self._sr = self._voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.load()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
        return samples, sr

    def speak(self, text: str) -> None:
        import sounddevice as sd

        samples, sr = self.synthesize(text)
        sd.play(samples, sr)
        sd.wait()
