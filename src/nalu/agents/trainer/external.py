"""Ingest public GUI-agent datasets into our (image, goal, action, args) JSONL format.

The training pipeline already accepts any JSONL with the schema
`{run, step, image, goal, action, args, thought}`. This module is a pure-Python
adapter that maps SeeClick's native annotation shape into that schema so it
can be mixed with locally-recorded runs.

This module is fully offline. The user supplies a local annotation file and
an images directory (obtained out-of-band — `huggingface-cli download`,
`wget`, rsync from a NAS, anything); we read, normalize, write. The agent
runtime never reaches out to a network.

SeeClick's native annotation shape (one record per example):
    {
        "img_filename": "screenshots/page_001.png",
        "instruction": "Click the 'Sign in' button",
        # Either bbox or point; coordinates are *normalized* [0, 1] floats
        # in the original Cheng et al. (2024) release.
        "bbox": [x1, y1, x2, y2],
        # or:
        "point": [x, y],
        "task_type": "click",  # SeeClick is mostly click grounding
    }

We accept either bbox (we take the center) or point. If coordinates look
absolute (any value > 1.0), we treat them as pixels; otherwise we read the
image to multiply normalized values up to pixel space.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from ... import config


@dataclass
class NormalizedExample:
    run: str
    step: int
    image: str  # absolute path; the training pipeline accepts absolute or ROOT-relative
    goal: str
    action: str
    args: dict
    thought: str = ""

    def to_dict(self) -> dict:
        return {
            "run": self.run,
            "step": self.step,
            "image": self.image,
            "goal": self.goal,
            "action": self.action,
            "args": self.args,
            "thought": self.thought,
        }


@dataclass
class FetchSummary:
    out_path: Path
    source: str
    examples_in: int = 0
    examples_out: int = 0
    skipped_no_target: int = 0
    skipped_no_image: int = 0
    skipped_unknown_action: int = 0
    actions: dict[str, int] = field(default_factory=dict)


def _looks_normalized(*coords: float) -> bool:
    """Return True if every coordinate is in [0, 1] — SeeClick's normalized shape."""
    return all(0.0 <= float(c) <= 1.0 for c in coords)


def _read_image_size(path: Path) -> tuple[int, int] | None:
    """Best-effort read image dimensions without forcing a PIL import for callers."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def _bbox_center_to_point(
    bbox: list[float], img_w: int | None, img_h: int | None
) -> tuple[int, int] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return _normalize_point(cx, cy, img_w, img_h)


def _normalize_point(
    x: float, y: float, img_w: int | None, img_h: int | None
) -> tuple[int, int] | None:
    if _looks_normalized(x, y):
        if img_w is None or img_h is None:
            return None
        return int(round(x * img_w)), int(round(y * img_h))
    return int(round(x)), int(round(y))


def _action_kind_from_seeclick(record: dict) -> str:
    """Map SeeClick's `task_type` field (when present) to our action vocabulary.

    SeeClick is dominated by click grounding; rarer subsets include `type` and
    `select`. Records without an explicit task_type are assumed to be clicks.
    """
    raw = (record.get("task_type") or record.get("action") or "click").strip().lower()
    if raw in {"click", "tap", "select", "press"}:
        return "click"
    if raw in {"type", "input", "type_text"}:
        return "type"
    if raw in {"scroll"}:
        return "scroll"
    return ""  # signals "unknown — caller should skip"


def normalize_seeclick_record(
    record: dict,
    *,
    images_root: Path,
    record_index: int,
    source_tag: str = "seeclick",
) -> NormalizedExample | str:
    """Map one SeeClick record to our schema. Returns a string error code on skip.

    Pure-Python — caller is responsible for image existence checks beyond
    the basic path resolution we do here.
    """
    img_rel = record.get("img_filename") or record.get("image") or record.get("img_path")
    if not img_rel:
        return "no_image"
    img_path = (images_root / img_rel).resolve() if not Path(img_rel).is_absolute() else Path(img_rel)
    if not img_path.exists():
        return "no_image"

    kind = _action_kind_from_seeclick(record)
    if not kind:
        return "unknown_action"

    goal = (record.get("instruction") or record.get("goal") or "").strip()
    if not goal:
        return "no_goal"

    args: dict = {}
    if kind == "click":
        img_w_h = _read_image_size(img_path)
        img_w, img_h = (img_w_h or (None, None))
        point: tuple[int, int] | None = None
        if "point" in record:
            p = record["point"]
            if isinstance(p, (list, tuple)) and len(p) == 2:
                point = _normalize_point(float(p[0]), float(p[1]), img_w, img_h)
        elif "bbox" in record:
            point = _bbox_center_to_point(list(record["bbox"]), img_w, img_h)
        if point is None:
            return "no_target"
        # Clamp to image bounds when known so downstream `validate_action` won't refuse.
        if img_w is not None and img_h is not None:
            x = max(0, min(img_w - 1, point[0]))
            y = max(0, min(img_h - 1, point[1]))
        else:
            x, y = point
        args = {"x": x, "y": y}
    elif kind == "type":
        text = record.get("text") or record.get("value") or ""
        if not text:
            return "no_target"
        args = {"text": str(text)}
    elif kind == "scroll":
        # SeeClick's scroll subset uses direction tokens; map to a coarse delta.
        direction = (record.get("direction") or "down").lower()
        args = {"dx": 0, "dy": -200 if direction == "down" else 200}

    return NormalizedExample(
        run=f"{source_tag}-{record_index:07d}",
        step=0,
        image=str(img_path),
        goal=goal,
        action=kind,
        args=args,
        thought="",
    )


def iter_seeclick_records(annotation_path: Path) -> Iterator[dict]:
    """Yield records from a SeeClick annotation file.

    Supports both line-delimited JSON (`*.jsonl`) and a single top-level JSON
    list (`*.json`) — both shapes appear in the wild for SeeClick subsets.
    """
    text = annotation_path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(data, list):
            yield from (r for r in data if isinstance(r, dict))
            return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def fetch_seeclick(
    annotation_path: Path,
    images_root: Path,
    out_dir: Path | None = None,
    *,
    limit: int | None = None,
) -> FetchSummary:
    """Walk a local SeeClick snapshot, write a JSONL in our training schema.

    `annotation_path`: path to SeeClick's annotations (.jsonl or .json).
    `images_root`: directory under which `img_filename` paths resolve.
    `out_dir`: where to write `dataset.jsonl` + `summary.json`. Defaults to
               `<config.ROOT>/training/datasets/external-<ts>/`.
    `limit`: cap on number of *output* examples (useful for smoke runs).
    """
    out_dir = out_dir or (
        config.ROOT / "training" / "datasets" / f"external-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.jsonl"

    summary = FetchSummary(out_path=out_path, source=str(annotation_path))
    actions: dict[str, int] = {}

    with out_path.open("w") as f:
        for i, rec in enumerate(iter_seeclick_records(annotation_path)):
            summary.examples_in += 1
            result = normalize_seeclick_record(rec, images_root=images_root, record_index=i)
            if isinstance(result, str):
                if result == "no_image":
                    summary.skipped_no_image += 1
                elif result == "no_target":
                    summary.skipped_no_target += 1
                elif result == "unknown_action":
                    summary.skipped_unknown_action += 1
                continue
            f.write(json.dumps(result.to_dict()) + "\n")
            summary.examples_out += 1
            actions[result.action] = actions.get(result.action, 0) + 1
            if limit is not None and summary.examples_out >= limit:
                break

    summary.actions = actions
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "out": str(out_path),
                "source": summary.source,
                "examples_in": summary.examples_in,
                "examples_out": summary.examples_out,
                "skipped_no_image": summary.skipped_no_image,
                "skipped_no_target": summary.skipped_no_target,
                "skipped_unknown_action": summary.skipped_unknown_action,
                "actions": actions,
            },
            indent=2,
        )
    )
    return summary
