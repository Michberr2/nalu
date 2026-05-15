from __future__ import annotations

import numpy as np

from nalu.agents.voice.always_on import (
    AlwaysOnConfig,
    AlwaysOnRunner,
)


def _silence(ms: int, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.int16)


def _speech(ms: int, sr: int = 16000) -> np.ndarray:
    # Synthetic int16 array — VAD is mocked, the actual samples don't matter.
    return np.full(int(sr * ms / 1000), 1000, dtype=np.int16)


def _make_runner(
    chunks: list[np.ndarray],
    vad_decisions: list[float],
    *,
    chunk_ms: int = 30,
    is_muted=None,
    config: AlwaysOnConfig | None = None,
):
    """Build a runner with deterministic source + scripted VAD output.

    `chunks` and `vad_decisions` are zipped — each step pops one of each.
    """
    sources = iter(chunks)
    vads = iter(vad_decisions)
    transcripts: list[str] = []

    def source():
        try:
            return next(sources), 16000
        except StopIteration:
            return None

    def vad(_samples: np.ndarray, _sr: int) -> float:
        return next(vads)

    def transcribe(samples: np.ndarray, _sr: int) -> str:
        ms = int(len(samples) / 16)
        return f"utterance({ms}ms)"

    runner = AlwaysOnRunner(
        audio_source=source,
        vad_fn=vad,
        transcribe_fn=transcribe,
        on_transcript=transcripts.append,
        is_muted=is_muted,
        config=config,
        chunk_ms=chunk_ms,
    )
    return runner, transcripts


def test_step_returns_false_when_source_exhausted():
    runner, _ = _make_runner([], [])
    assert runner.step() is False


def test_idle_until_speech_detected():
    runner, said = _make_runner(
        chunks=[_silence(30), _silence(30), _silence(30)],
        vad_decisions=[0.1, 0.05, 0.0],
    )
    while runner.step():
        pass
    assert runner.state == AlwaysOnRunner.STATE_IDLE
    assert said == []


def test_speech_burst_below_min_speech_ms_is_dropped():
    # 1 speech chunk (30ms) is below 200ms floor, then silence — should drop.
    cfg = AlwaysOnConfig(min_speech_ms=200, end_of_speech_silence_ms=60)
    chunks = [_speech(30)] + [_silence(30)] * 5
    vads = [0.9] + [0.0] * 5
    runner, said = _make_runner(chunks, vads, config=cfg)
    while runner.step():
        pass
    assert said == []
    assert runner.state == AlwaysOnRunner.STATE_IDLE


def test_full_utterance_transcribed_after_silence():
    # ~300ms of speech (10 chunks) + 500ms silence (17 chunks) → flush
    cfg = AlwaysOnConfig(
        min_speech_ms=200, end_of_speech_silence_ms=500, max_utterance_ms=15_000
    )
    chunks = [_speech(30)] * 10 + [_silence(30)] * 17
    vads = [0.9] * 10 + [0.0] * 17
    runner, said = _make_runner(chunks, vads, config=cfg)
    while runner.step():
        pass
    assert len(said) == 1
    assert said[0].startswith("utterance(")
    assert runner.state == AlwaysOnRunner.STATE_IDLE


def test_max_utterance_force_cuts_long_speech():
    cfg = AlwaysOnConfig(
        min_speech_ms=60, end_of_speech_silence_ms=500, max_utterance_ms=240
    )
    # 10 speech chunks @ 30ms = 300ms > 240ms cap → force cut
    chunks = [_speech(30)] * 10
    vads = [0.9] * 10
    runner, said = _make_runner(chunks, vads, config=cfg)
    while runner.step():
        pass
    assert len(said) == 1


def test_muted_drops_audio_and_resets_speaking_state():
    cfg = AlwaysOnConfig(min_speech_ms=60, end_of_speech_silence_ms=60)
    muted = {"on": False}
    chunks = (
        [_speech(30)] * 3
        + [_speech(30)] * 3  # mute kicks in on chunk 4
        + [_silence(30)] * 5
    )
    vads = [0.9] * 6 + [0.0] * 5

    step_idx = {"i": 0}

    def is_muted():
        # Mute after first 3 chunks; never unmute (the speech buffer should be
        # discarded; idle resumes).
        i = step_idx["i"]
        return i >= 3

    runner, said = _make_runner(chunks, vads, is_muted=is_muted, config=cfg)
    while runner.step():
        step_idx["i"] += 1
    # Speech accumulated for 3 chunks (90ms) then muting wiped state.
    # Below min_speech of 60ms? 90ms > 60ms, but it was dropped on entering mute,
    # not flushed — so no transcript.
    assert said == []
    assert runner.state == AlwaysOnRunner.STATE_IDLE


def test_transcribe_exception_logged_not_raised():
    def transcribe(_s: np.ndarray, _sr: int) -> str:
        raise RuntimeError("model dead")

    chunks = [_speech(30)] * 10 + [_silence(30)] * 20
    vads = [0.9] * 10 + [0.0] * 20

    sources = iter(chunks)
    vads_iter = iter(vads)
    said: list[str] = []

    def source():
        try:
            return next(sources), 16000
        except StopIteration:
            return None

    runner = AlwaysOnRunner(
        audio_source=source,
        vad_fn=lambda s, sr: next(vads_iter),
        transcribe_fn=transcribe,
        on_transcript=said.append,
        config=AlwaysOnConfig(min_speech_ms=60, end_of_speech_silence_ms=300),
    )
    while runner.step():
        pass
    assert said == []  # exception was swallowed


def test_on_transcript_exception_does_not_corrupt_state():
    def boom(_t: str) -> None:
        raise RuntimeError("handler failed")

    sources = iter([_speech(30)] * 10 + [_silence(30)] * 20 + [_speech(30)] * 10 + [_silence(30)] * 20)
    vads_iter = iter([0.9] * 10 + [0.0] * 20 + [0.9] * 10 + [0.0] * 20)

    runner = AlwaysOnRunner(
        audio_source=lambda: (next(sources, None), 16000) if True else None,
        vad_fn=lambda s, sr: next(vads_iter),
        transcribe_fn=lambda s, sr: "hello",
        on_transcript=boom,
        config=AlwaysOnConfig(min_speech_ms=60, end_of_speech_silence_ms=300),
    )
    # Need to wrap source so StopIteration becomes None:
    finished = {"done": False}
    src_iter = iter([_speech(30)] * 10 + [_silence(30)] * 20 + [_speech(30)] * 10 + [_silence(30)] * 20)
    vads_iter2 = iter([0.9] * 10 + [0.0] * 20 + [0.9] * 10 + [0.0] * 20)

    def source():
        if finished["done"]:
            return None
        try:
            return next(src_iter), 16000
        except StopIteration:
            finished["done"] = True
            return None

    runner = AlwaysOnRunner(
        audio_source=source,
        vad_fn=lambda s, sr: next(vads_iter2),
        transcribe_fn=lambda s, sr: "hello",
        on_transcript=boom,
        config=AlwaysOnConfig(min_speech_ms=60, end_of_speech_silence_ms=300),
    )
    while runner.step():
        pass
    # Two full utterances were transcribed; both handler invocations raised; state
    # returned to IDLE cleanly each time.
    assert runner.state == AlwaysOnRunner.STATE_IDLE
    assert len(runner.transcripts) == 2  # recorded internally even though handler raised


def test_silero_vad_fallback_when_unavailable(monkeypatch):
    # Force the import path to fail; the factory should return a permissive function.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "silero_vad"):
            raise ImportError(f"{name} not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from nalu.agents.voice.always_on import make_silero_vad_fn

    vad = make_silero_vad_fn()
    # Fallback returns 1.0 for everything — never gates speech.
    assert vad(np.zeros(480, dtype=np.int16), 16000) == 1.0
