from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ... import config


@dataclass
class TrainingRunSummary:
    out_dir: Path
    adapter_dir: Path
    iters: int
    examples: int
    final_loss: float | None


def _serialize_action(action: str, args: dict[str, Any]) -> str:
    if action == "click":
        return f"click(x={int(args.get('x', 0))}, y={int(args.get('y', 0))})"
    if action == "double_click":
        return f"double_click(x={int(args.get('x', 0))}, y={int(args.get('y', 0))})"
    if action == "type":
        text = str(args.get("text", "")).replace('"', '\\"')
        return f'type(text="{text}")'
    if action == "key":
        name = args.get("name", "")
        mods = args.get("modifiers", []) or []
        if mods:
            combo = "+".join(list(mods) + [name])
            return f'hotkey(keys="{combo}")'
        return f'press(key="{name}")'
    if action == "scroll":
        return f"scroll(dx={int(args.get('dx', 0))}, dy={int(args.get('dy', 0))})"
    if action == "done":
        ans = str(args.get("answer", "")).replace('"', '\\"')
        return f'finished(content="{ans}")'
    return f"{action}({json.dumps(args)})"


def _build_completion(thought: str, action: str, args: dict[str, Any]) -> str:
    body = _serialize_action(action, args)
    if thought:
        return f"Thought: {thought}\nAction: {body}"
    return f"Action: {body}"


_METRIC_RE = re.compile(
    r"Iter (\d+): Train loss .*?(\d+\.\d+)"
    r".*?Learning Rate (\d+\.\d+e[+-]?\d+)"
    r".*?It/sec (\d+\.\d+)"
    r".*?Tokens/sec (\d+\.\d+)"
    r".*?Trained Tokens (\d+)"
    r".*?Peak mem (\d+\.\d+)"
)


class _MetricsTee:
    def __init__(self, stream, metrics_path: Path):
        self.stream = stream
        self.metrics_path = metrics_path
        self._buf = ""

    def write(self, data: str) -> int:
        self.stream.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._maybe_record(line)
        return len(data)

    def _maybe_record(self, line: str) -> None:
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = _METRIC_RE.search(clean)
        if not m:
            return
        rec = {
            "step": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "learning_rate": float(m.group(3)),
            "it_per_sec": float(m.group(4)),
            "tokens_per_sec": float(m.group(5)),
            "trained_tokens": int(m.group(6)),
            "peak_mem_gb": float(m.group(7)),
            "ts": time.time(),
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def flush(self) -> None:
        self.stream.flush()


class QLoRARunner:
    """Fine-tune the vision model on a Nalu JSONL dataset, emitting a LoRA adapter."""

    def __init__(
        self,
        dataset_path: Path,
        out_dir: Path | None = None,
        model_path: str = config.VISION_MODEL,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        learning_rate: float = 2e-5,
        batch_size: int = 1,
        iters: int | None = None,
        epochs: int = 1,
        max_seq_length: int = 2048,
        grad_checkpoint: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.out_dir = out_dir or (
            config.ROOT / "training" / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.model_path = model_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iters = iters
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.grad_checkpoint = grad_checkpoint

    def _load_examples(self) -> list[dict]:
        out = []
        for line in self.dataset_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

    def _build_hf_dataset(self, examples: list[dict]):
        from datasets import Dataset, Image as DSImage

        rows = {"question": [], "answer": [], "image": []}
        for ex in examples:
            img_path = config.ROOT / ex["image"]
            if not img_path.exists():
                continue
            rows["question"].append(ex.get("goal", "") or "")
            rows["answer"].append(
                _build_completion(ex.get("thought", ""), ex["action"], ex.get("args", {}))
            )
            rows["image"].append(str(img_path))

        if not rows["image"]:
            raise ValueError(f"no resolvable image paths in {self.dataset_path}")

        ds = Dataset.from_dict(rows)
        return ds.cast_column("image", DSImage())

    def run(self) -> TrainingRunSummary:
        import mlx.optimizers as optim
        from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
        from mlx_vlm.trainer.datasets import VisionDataset
        from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
        from mlx_vlm.trainer.utils import print_trainable_parameters
        from mlx_vlm.utils import load

        self.out_dir.mkdir(parents=True, exist_ok=True)
        # mlx_vlm.apply_lora_layers expects a directory containing
        # adapters.safetensors + adapter_config.json — so the run dir IS the adapter.
        adapter_file = self.out_dir / "adapters.safetensors"
        adapter_config = self.out_dir / "adapter_config.json"
        metrics_path = self.out_dir / "metrics.jsonl"
        config_path = self.out_dir / "config.json"

        adapter_config.write_text(
            json.dumps(
                {
                    "rank": self.lora_rank,
                    "alpha": self.lora_alpha,
                    "dropout": self.lora_dropout,
                },
                indent=2,
            )
        )

        examples = self._load_examples()
        hf_ds = self._build_hf_dataset(examples)

        if len(hf_ds) < self.batch_size:
            raise ValueError(
                f"dataset has {len(hf_ds)} examples; need at least batch_size={self.batch_size}"
            )

        iters = self.iters or max(1, (len(hf_ds) // self.batch_size) * self.epochs)

        config_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(self.dataset_path),
                    "model_path": self.model_path,
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "iters": iters,
                    "examples": len(hf_ds),
                    "started_ts": time.time(),
                },
                indent=2,
            )
        )

        model, processor = load(
            self.model_path, processor_config={"trust_remote_code": True}
        )
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        model_cfg = model.config.__dict__

        hf_ds = transform_dataset_to_messages(hf_ds, model_type, None)
        train_dataset = VisionDataset(hf_ds, model_cfg, processor, image_resize_shape=None)

        setup_args = SimpleNamespace(
            full_finetune=False,
            train_vision=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )
        model = setup_model_for_training(model, setup_args, adapter_path=None)
        print_trainable_parameters(model)

        optimizer = optim.Adam(learning_rate=self.learning_rate)

        training_args = TrainingArgs(
            batch_size=self.batch_size,
            iters=iters,
            steps_per_report=1,
            steps_per_eval=10**9,
            steps_per_save=max(50, iters // 4) if iters >= 4 else iters,
            val_batches=0,
            max_seq_length=self.max_seq_length,
            adapter_file=str(adapter_file),
            grad_checkpoint=self.grad_checkpoint,
            learning_rate=self.learning_rate,
        )

        tee = _MetricsTee(sys.stdout, metrics_path)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            train(
                model=model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=None,
                args=training_args,
            )
        finally:
            sys.stdout = old_stdout

        final_loss: float | None = None
        if metrics_path.exists():
            lines = metrics_path.read_text().splitlines()
            if lines:
                try:
                    final_loss = json.loads(lines[-1]).get("train_loss")
                except json.JSONDecodeError:
                    pass

        return TrainingRunSummary(
            out_dir=self.out_dir,
            adapter_dir=self.out_dir,
            iters=iters,
            examples=len(hf_ds),
            final_loss=final_loss,
        )


def _active_adapter_pointer() -> Path:
    return config.ROOT / "training" / "active_adapter"


def active_adapter_dir() -> Path | None:
    """Return the directory of the currently-active LoRA adapter, or None.

    The pointer is a tiny text file that holds an absolute path to the run
    directory, so apply_lora_layers can be called against it.
    """
    pointer = _active_adapter_pointer()
    if not pointer.exists():
        return None
    target = Path(pointer.read_text().strip())
    if not (target / "adapters.safetensors").exists():
        return None
    if not (target / "adapter_config.json").exists():
        return None
    return target


def activate_adapter(run_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "adapters.safetensors").exists():
        raise FileNotFoundError(f"no adapters.safetensors in {run_dir}")
    if not (run_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter_config.json in {run_dir}")
    pointer = _active_adapter_pointer()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(run_dir))
    return run_dir


def deactivate_adapter() -> bool:
    pointer = _active_adapter_pointer()
    if pointer.exists():
        pointer.unlink()
        return True
    return False


def list_runs(root: Path | None = None) -> list[dict]:
    root = root or (config.ROOT / "training" / "runs")
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        cfg = d / "config.json"
        metrics = d / "metrics.jsonl"
        adapter = d / "adapters.safetensors"
        entry = {"name": d.name, "path": str(d), "has_adapter": adapter.exists()}
        if cfg.exists():
            try:
                entry.update(json.loads(cfg.read_text()))
            except json.JSONDecodeError:
                pass
        if metrics.exists():
            lines = metrics.read_text().splitlines()
            entry["steps_logged"] = len(lines)
            if lines:
                try:
                    entry["last_loss"] = json.loads(lines[-1]).get("train_loss")
                except json.JSONDecodeError:
                    pass
        out.append(entry)
    return out
