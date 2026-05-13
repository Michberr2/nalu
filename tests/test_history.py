from __future__ import annotations

from nalu.agents.planner.history import (
    COMPACT_AFTER,
    KEEP_TAIL,
    compact_history,
    summarize_head,
)


def _action(step: int, kind: str, args: str = "{}", reason: str = "r") -> str:
    return f"step {step}: {kind} {args} -- {reason}"


def test_short_history_unchanged():
    entries = [_action(i, "click") for i in range(5)]
    assert compact_history(entries) == entries


def test_history_at_threshold_unchanged():
    entries = [_action(i, "click") for i in range(COMPACT_AFTER)]
    assert compact_history(entries) == entries


def test_history_above_threshold_compacted():
    entries = [_action(i, "click") for i in range(COMPACT_AFTER + 5)]
    out = compact_history(entries)
    assert len(out) == 1 + KEEP_TAIL
    assert out[0].startswith("Earlier:")
    assert out[1:] == entries[-KEEP_TAIL:]


def test_summary_counts_action_kinds():
    head = [
        _action(0, "click"),
        _action(1, "click"),
        _action(2, "type"),
        _action(3, "scroll"),
        _action(4, "click"),
    ]
    s = summarize_head(head)
    assert "5 prior steps" in s
    assert "3× click" in s
    assert "1× type" in s
    assert "1× scroll" in s


def test_summary_counts_self_correction_events():
    head = [
        _action(0, "click"),
        "step 1: SKIPPED -- stuck hint",
        "step 2: NO EFFECT -- screen unchanged",
        "step 3: VERIFICATION DENIED -- still loading",
        "step 4: REFUSED -- click at (4096, 50) is outside the 1200x800 screenshot",
        "step 5: JITTERED RETRY -- (100,100) had no effect; retried at (104,98)",
        "step 6: RECOVERY -- previous attempt got stuck, try a different approach",
        "step 7: PARSE RETRY -- previous response was prose, not an action",
    ]
    s = summarize_head(head)
    assert "7 self-correction events" in s
    assert "1× click" in s


def test_summary_handles_singular_self_correction():
    head = [_action(0, "click"), "step 1: SKIPPED -- hint"]
    s = summarize_head(head)
    assert "1 self-correction event" in s and "1 self-correction events" not in s


def test_summary_with_only_self_correction_omits_kind_block():
    head = ["step 0: SKIPPED -- hint", "step 1: NO EFFECT -- nothing"]
    s = summarize_head(head)
    assert "2 self-correction events" in s
    assert "×" not in s  # no kind list


def test_compact_preserves_tail_order():
    entries = []
    for i in range(COMPACT_AFTER + 3):
        kind = "click" if i % 2 == 0 else "type"
        entries.append(_action(i, kind))
    out = compact_history(entries)
    assert out[1:] == entries[-KEEP_TAIL:]


def test_custom_compact_after_and_keep_tail():
    entries = [_action(i, "click") for i in range(15)]
    out = compact_history(entries, compact_after=5, keep_tail=3)
    assert len(out) == 4
    assert out[0].startswith("Earlier:")
    assert out[1:] == entries[-3:]


def test_keep_tail_zero_drops_all_verbatim():
    entries = [_action(i, "click") for i in range(15)]
    out = compact_history(entries, compact_after=5, keep_tail=0)
    assert len(out) == 1
    assert out[0].startswith("Earlier:")


def test_unparseable_entries_dont_crash():
    head = ["weird line without step prefix", "step abc: not numeric", _action(0, "click")]
    s = summarize_head(head)
    assert "3 prior steps" in s
    assert "1× click" in s


def test_compact_returns_new_list_not_mutating_input():
    entries = [_action(i, "click") for i in range(COMPACT_AFTER + 5)]
    snapshot = list(entries)
    compact_history(entries)
    assert entries == snapshot
