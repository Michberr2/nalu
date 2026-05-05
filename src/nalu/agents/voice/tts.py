from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import config

PIPER_HF_REPO = "rhasspy/piper-voices"


def _parse_voice(voice: str) -> tuple[str, str, str, str]:
    """en_GB-alan-medium -> (en, GB, alan, medium)"""
    parts = voice.split("-")
    if len(parts) != 3:
        raise ValueError(f"voice id must look like en_GB-alan-medium, got {voice!r}")
    lang_region, speaker, quality = parts
    lang, region = lang_region.split("_", 1)
    return lang, region, speaker, quality


def _voice_paths(voice: str) -> tuple[Path, Path]:
    base = config.MODELS_DIR / "piper" / voice
    return base / f"{voice}.onnx", base / f"{voice}.onnx.json"


def _ensure_voice(voice: str) -> tuple[Path, Path]:
    onnx, cfg = _voice_paths(voice)
    if onnx.exists() and cfg.exists():
        return onnx, cfg

    from huggingface_hub import hf_hub_download

    lang, region, speaker, quality = _parse_voice(voice)
    subdir = f"{lang}/{lang}_{region}/{speaker}/{quality}"

    onnx.parent.mkdir(parents=True, exist_ok=True)
    cache_root = config.MODELS_DIR / "piper" / "_hf_cache"
    onnx_dl = hf_hub_download(PIPER_HF_REPO, f"{subdir}/{voice}.onnx", cache_dir=str(cache_root))
    cfg_dl = hf_hub_download(PIPER_HF_REPO, f"{subdir}/{voice}.onnx.json", cache_dir=str(cache_root))
    onnx.write_bytes(Path(onnx_dl).read_bytes())
    cfg.write_bytes(Path(cfg_dl).read_bytes())
    return onnx, cfg


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
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.int16), self._sr or 22050
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        return audio, self._sr

    def speak(self, text: str) -> None:
        import sounddevice as sd

        samples, sr = self.synthesize(text)
        sd.play(samples, sr)
        sd.wait()
