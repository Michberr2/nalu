from __future__ import annotations

import json
from pathlib import Path

import pytest

from nalu.agents.trainer.external import (
    fetch_seeclick,
    iter_seeclick_records,
    normalize_seeclick_record,
)


@pytest.fixture
def fake_image(tmp_path: Path):
    """Write a real (tiny) PNG so PIL can read its dimensions."""
    pytest.importorskip("PIL")
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    def _make(name: str, w: int, h: int) -> Path:
        path = images_dir / name
        Image.new("RGB", (w, h), color=(255, 255, 255)).save(path)
        return path

    return images_dir, _make


def test_normalize_click_with_normalized_point(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 1000, 800)
    rec = {
        "img_filename": "a.png",
        "instruction": "Click the sign-in button",
        "point": [0.5, 0.25],
        "task_type": "click",
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert not isinstance(out, str)
    assert out.action == "click"
    assert out.args == {"x": 500, "y": 200}
    assert out.goal == "Click the sign-in button"
    assert out.run == "seeclick-0000000"


def test_normalize_click_with_absolute_pixel_point(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 1000, 800)
    rec = {
        "img_filename": "a.png",
        "instruction": "Open the menu",
        # Any value > 1.0 → treated as absolute pixels, no image-size lookup needed.
        "point": [42, 99],
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=1)
    assert not isinstance(out, str)
    assert out.args == {"x": 42, "y": 99}


def test_normalize_click_with_bbox_takes_center(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 1000, 800)
    rec = {
        "img_filename": "a.png",
        "instruction": "Click the panel",
        "bbox": [0.2, 0.2, 0.4, 0.4],  # normalized: center at (0.3, 0.3)
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=2)
    assert not isinstance(out, str)
    assert out.args == {"x": 300, "y": 240}


def test_normalize_clamps_to_image_bounds(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 1000, 800)
    rec = {
        "img_filename": "a.png",
        "instruction": "Click off-screen",
        "point": [1100, 50],  # x exceeds width
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=3)
    assert not isinstance(out, str)
    assert out.args["x"] == 999  # clamped to width - 1


def test_normalize_skips_missing_image(tmp_path):
    rec = {
        "img_filename": "nope.png",
        "instruction": "x",
        "point": [10, 10],
    }
    out = normalize_seeclick_record(rec, images_root=tmp_path, record_index=0)
    assert out == "no_image"


def test_normalize_skips_missing_target(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    rec = {"img_filename": "a.png", "instruction": "Click something"}
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert out == "no_target"


def test_normalize_skips_missing_goal(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    rec = {"img_filename": "a.png", "point": [10, 10]}
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert out == "no_goal"


def test_normalize_skips_unknown_task_type(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    rec = {
        "img_filename": "a.png",
        "instruction": "do thing",
        "point": [10, 10],
        "task_type": "summon_demon",
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert out == "unknown_action"


def test_normalize_type_action(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    rec = {
        "img_filename": "a.png",
        "instruction": "Type the search query",
        "task_type": "type",
        "text": "hello world",
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert not isinstance(out, str)
    assert out.action == "type"
    assert out.args == {"text": "hello world"}


def test_normalize_scroll_direction_down(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    rec = {
        "img_filename": "a.png",
        "instruction": "Scroll for more",
        "task_type": "scroll",
        "direction": "down",
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert not isinstance(out, str)
    assert out.action == "scroll"
    assert out.args == {"dx": 0, "dy": -200}


def test_normalize_scroll_direction_up(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    rec = {
        "img_filename": "a.png",
        "instruction": "Scroll back up",
        "task_type": "scroll",
        "direction": "up",
    }
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert not isinstance(out, str)
    assert out.args == {"dx": 0, "dy": 200}


def test_normalize_falls_back_to_click_when_no_task_type(fake_image):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    rec = {"img_filename": "a.png", "instruction": "click thing", "point": [50, 50]}
    out = normalize_seeclick_record(rec, images_root=images_dir, record_index=0)
    assert not isinstance(out, str)
    assert out.action == "click"


def test_iter_seeclick_records_jsonl(tmp_path):
    p = tmp_path / "ann.jsonl"
    p.write_text(
        '{"a": 1}\n'
        '{"a": 2}\n'
        "\n"
        "not json\n"
        '{"a": 3}\n'
    )
    out = list(iter_seeclick_records(p))
    assert [r["a"] for r in out] == [1, 2, 3]


def test_iter_seeclick_records_top_level_array(tmp_path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}, "not a dict"]))
    out = list(iter_seeclick_records(p))
    assert [r["a"] for r in out] == [1, 2]


def test_fetch_writes_dataset_and_summary(fake_image, tmp_path):
    images_dir, mk = fake_image
    mk("a.png", 1000, 800)
    mk("b.png", 1000, 800)
    ann = tmp_path / "ann.jsonl"
    ann.write_text(
        json.dumps({"img_filename": "a.png", "instruction": "click a", "point": [0.1, 0.1]}) + "\n" +
        json.dumps({"img_filename": "b.png", "instruction": "click b", "bbox": [0.4, 0.4, 0.6, 0.6]}) + "\n" +
        json.dumps({"img_filename": "missing.png", "instruction": "x", "point": [0.5, 0.5]}) + "\n"
    )
    out_dir = tmp_path / "out"
    summary = fetch_seeclick(ann, images_dir, out_dir=out_dir)

    assert summary.examples_in == 3
    assert summary.examples_out == 2
    assert summary.skipped_no_image == 1
    assert summary.actions == {"click": 2}

    lines = (out_dir / "dataset.jsonl").read_text().splitlines()
    records = [json.loads(l) for l in lines]
    assert [r["goal"] for r in records] == ["click a", "click b"]
    assert all(r["action"] == "click" for r in records)

    s = json.loads((out_dir / "summary.json").read_text())
    assert s["examples_out"] == 2
    assert s["actions"] == {"click": 2}


def test_fetch_respects_limit(fake_image, tmp_path):
    images_dir, mk = fake_image
    mk("a.png", 100, 100)
    ann = tmp_path / "ann.jsonl"
    ann.write_text(
        "\n".join(
            json.dumps({"img_filename": "a.png", "instruction": f"goal {i}", "point": [10, 10]})
            for i in range(5)
        )
    )
    summary = fetch_seeclick(ann, images_dir, out_dir=tmp_path / "out", limit=2)
    assert summary.examples_out == 2
