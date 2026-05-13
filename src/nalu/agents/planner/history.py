"""Compact long action histories before they reach the vision prompt.

Each planner step appends a one-line trace to `history`:
  step N: <kind> {<args>} -- <reason>
  step N: SKIPPED -- <stuck hint>
  step N: NO EFFECT -- <screen-change hint>
  step N: VERIFICATION DENIED -- <verifier reason>
  step N: REFUSED -- <out-of-bounds hint>
  step N: JITTERED RETRY -- <click jitter retry trace>
  step N: RECOVERY -- <auto-retry-after-stuck-or-dispatch context>
  step N: PARSE RETRY -- <model returned prose, action format reminder>

For short tasks this is fine. For long tasks (30+ steps) the trace dominates
the prompt and pushes the screenshot's pixel reasoning out of the model's
attention budget. `compact_history` keeps the last `keep_tail` entries verbatim
and folds everything older into a single "Earlier: …" summary line that names
the action-kind distribution and counts self-correction events.

Pure-Python — the planner just swaps `history` for `compact_history(history)`
in its `vision.decide` call.
"""
from __future__ import annotations

import re
from collections import Counter


COMPACT_AFTER = 20      # only start compacting once we exceed this many entries
KEEP_TAIL = 8           # last N entries always kept verbatim

_SELF_CORRECTION_MARKERS = (
    "SKIPPED",
    "NO EFFECT",
    "VERIFICATION DENIED",
    "REFUSED",
    "JITTERED RETRY",
    "RECOVERY",
    "PARSE RETRY",
)
_STEP_KIND_RE = re.compile(r"^step\s+\d+:\s*(\w+)")


def _is_self_correction(entry: str) -> bool:
    return any(marker in entry for marker in _SELF_CORRECTION_MARKERS)


def _entry_kind(entry: str) -> str | None:
    m = _STEP_KIND_RE.match(entry)
    if not m:
        return None
    head = m.group(1)
    if head in {"SKIPPED", "NO", "VERIFICATION", "REFUSED", "JITTERED", "RECOVERY", "PARSE"}:
        return None
    return head


def summarize_head(head: list[str]) -> str:
    """Return a one-line "Earlier: …" summary for the older entries."""
    kinds: Counter[str] = Counter()
    self_correction = 0
    for entry in head:
        if _is_self_correction(entry):
            self_correction += 1
            continue
        kind = _entry_kind(entry)
        if kind:
            kinds[kind] += 1

    parts = [f"Earlier: {len(head)} prior steps"]
    if kinds:
        kind_str = ", ".join(f"{n}× {k}" for k, n in kinds.most_common())
        parts.append(f"({kind_str})")
    if self_correction:
        parts.append(f"and {self_correction} self-correction event{'s' if self_correction != 1 else ''}")
    return " ".join(parts) + "."


def compact_history(
    entries: list[str],
    *,
    compact_after: int = COMPACT_AFTER,
    keep_tail: int = KEEP_TAIL,
) -> list[str]:
    """Return a compacted view of `entries` suitable for the vision prompt.

    Histories at or below `compact_after` are returned unchanged. Otherwise
    the head is folded into one summary line, followed by the last `keep_tail`
    entries verbatim.
    """
    if len(entries) <= compact_after:
        return list(entries)
    head = entries[:-keep_tail] if keep_tail > 0 else list(entries)
    tail = entries[-keep_tail:] if keep_tail > 0 else []
    return [summarize_head(head), *tail]
