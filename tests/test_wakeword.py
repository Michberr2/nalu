from __future__ import annotations

from typing import Iterable

from nalu.agents.voice.wakeword import WakeWordRunner


class _StubSpotter:
    """Returns scripted scores per call. Frame contents are ignored."""

    def __init__(self, scripted: Iterable[dict[str, float]]):
        self._iter = iter(scripted)
        self.calls = 0
        self.loaded = False

    def predict(self, frame) -> dict[str, float]:
        self.calls += 1
        try:
            return next(self._iter)
        except StopIteration:
            return {}

    def load(self) -> None:
        self.loaded = True


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _runner(scores, **kwargs):
    fires: list[tuple[str, float]] = []
    spotter = _StubSpotter(scores)
    runner = WakeWordRunner(
        on_wake=lambda k, s: fires.append((k, s)),
        spotter=spotter,
        **kwargs,
    )
    return runner, spotter, fires


def test_below_threshold_does_not_fire():
    runner, _, fires = _runner([{"hey_jarvis": 0.3}], threshold=0.5)
    assert runner.feed(b"") is None
    assert fires == []


def test_above_threshold_fires_once():
    runner, _, fires = _runner([{"hey_jarvis": 0.9}], threshold=0.5)
    out = runner.feed(b"")
    assert out == ("hey_jarvis", 0.9)
    assert fires == [("hey_jarvis", 0.9)]


def test_cooldown_suppresses_consecutive_fires():
    clock = _Clock()
    runner, _, fires = _runner(
        [{"k": 0.8}, {"k": 0.95}, {"k": 0.9}],
        threshold=0.5, cooldown_s=2.0, clock=clock,
    )
    clock.now = 0.0
    assert runner.feed(b"") is not None
    clock.now = 0.5
    assert runner.feed(b"") is None
    clock.now = 1.99
    assert runner.feed(b"") is None
    assert len(fires) == 1


def test_fires_again_after_cooldown_expires():
    clock = _Clock()
    runner, _, fires = _runner(
        [{"k": 0.8}, {"k": 0.95}],
        threshold=0.5, cooldown_s=2.0, clock=clock,
    )
    clock.now = 0.0
    runner.feed(b"")
    clock.now = 2.5
    assert runner.feed(b"") is not None
    assert len(fires) == 2


def test_picks_highest_scoring_keyword():
    runner, _, fires = _runner([{"a": 0.6, "b": 0.95, "c": 0.4}], threshold=0.5)
    out = runner.feed(b"")
    assert out == ("b", 0.95)
    assert fires == [("b", 0.95)]


def test_empty_score_dict_is_no_op():
    runner, _, fires = _runner([{}], threshold=0.0)
    assert runner.feed(b"") is None
    assert fires == []


def test_warm_calls_spotter_load():
    runner, spotter, _ = _runner([])
    runner.warm()
    assert spotter.loaded is True


def test_callback_exceptions_dont_raise():
    spotter = _StubSpotter([{"k": 0.9}])

    def bad(_k, _s):
        raise RuntimeError("boom")

    runner = WakeWordRunner(on_wake=bad, spotter=spotter, threshold=0.5)
    out = runner.feed(b"")
    assert out == ("k", 0.9)


def test_frame_source_drains_then_stops():
    fires: list[tuple[str, float]] = []
    spotter = _StubSpotter([{"k": 0.9}, {"k": 0.1}, {"k": 0.95}])
    clock = _Clock()
    runner = WakeWordRunner(
        on_wake=lambda k, s: fires.append((k, s)),
        spotter=spotter,
        threshold=0.5,
        cooldown_s=0.0,
        frame_source=[b"a", b"b", b"c"],
        clock=clock,
    )
    runner.start()
    runner._thread.join(timeout=1.0)
    assert spotter.calls == 3
    assert [f[0] for f in fires] == ["k", "k"]
