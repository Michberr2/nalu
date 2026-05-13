"""Mergekit + MLX quantization pipeline.

Drives `mergekit-yaml` to merge two or more compatible HF checkpoints, then optionally
runs `mlx_vlm.convert` to produce a 4-bit MLX checkpoint, and finally registers the
result in Nalu's model registry so `nalu model use <id>` picks it up.

Mergekit and mlx_vlm.convert are external CLIs; we shell out so users who never merge
don't have to install mergekit. Caveats:

  - The two source models must share an architecture (e.g. both Qwen2-VL or both
    Qwen2.5-VL). Mixing across families fails inside mergekit, not here.
  - The merge runs on the HF-format weights, not on the 4-bit MLX cache. The user
    must give source paths that point at full-precision repos or local dirs.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

from ... import config


MERGES_ROOT = config.ROOT / "training" / "merges"
SUPPORTED_METHODS = {"linear", "slerp", "ties", "task_arithmetic", "dare_ties", "dare_linear"}


@dataclass
class MergeSource:
    model: str
    weight: float = 0.5
    density: float | None = None  # for ties / dare_ties

    def to_mergekit(self) -> dict:
        params: dict = {"weight": float(self.weight)}
        if self.density is not None:
            params["density"] = float(self.density)
        return {"model": self.model, "parameters": params}


@dataclass
class MergeConfig:
    sources: list[MergeSource]
    merge_method: str = "linear"
    dtype: str = "bfloat16"
    base_model: str | None = None  # required for ties/dare_ties/task_arithmetic
    tokenizer_source: str | None = None

    def validate(self) -> None:
        if len(self.sources) < 2:
            raise ValueError("merge requires at least two source models")
        if self.merge_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unsupported merge_method {self.merge_method!r}; "
                f"choose one of {sorted(SUPPORTED_METHODS)}"
            )
        if self.merge_method in {"ties", "dare_ties", "task_arithmetic"} and not self.base_model:
            raise ValueError(
                f"merge_method {self.merge_method!r} requires base_model"
            )
        if self.merge_method == "linear":
            total = sum(s.weight for s in self.sources)
            if total <= 0:
                raise ValueError("linear merge requires positive total weight")

    def to_yaml(self) -> dict:
        out: dict = {
            "models": [s.to_mergekit() for s in self.sources],
            "merge_method": self.merge_method,
            "dtype": self.dtype,
        }
        if self.base_model:
            out["base_model"] = self.base_model
        if self.tokenizer_source:
            out["tokenizer_source"] = self.tokenizer_source
        return out


@dataclass
class MergeRunSummary:
    out_dir: Path
    config_path: Path
    merged_dir: Path
    mlx_dir: Path | None
    registered_id: str | None
    elapsed_s: float
    method: str
    sources: list[str] = field(default_factory=list)


def _new_merge_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = MERGES_ROOT / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(
            f"{name!r} not found on PATH. Install it (e.g. `uv tool install mergekit`) "
            "and re-run."
        )
    return found


def _run_subprocess(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-30:]
        raise RuntimeError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            + "\n".join(tail)
        )


def write_config(cfg: MergeConfig, path: Path) -> Path:
    cfg.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_yaml(), sort_keys=False))
    return path


class MergeRunner:
    def __init__(
        self,
        cfg: MergeConfig,
        out_dir: Path | None = None,
        quantize: bool = True,
        quant_bits: int = 4,
        register_as: str | None = None,
        register_label: str = "",
    ):
        self.cfg = cfg
        self.out_dir = out_dir or _new_merge_dir()
        self.quantize = quantize
        self.quant_bits = quant_bits
        self.register_as = register_as
        self.register_label = register_label

    def run(self, runner: callable = None) -> MergeRunSummary:
        """Drive merge + (optional) MLX conversion + registry write.

        `runner` is for test injection — defaults to `_run_subprocess`.
        """
        runner = runner or _run_subprocess
        self.cfg.validate()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = write_config(self.cfg, self.out_dir / "merge.yaml")
        merged_dir = self.out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        _check_tool("mergekit-yaml")
        runner(
            ["mergekit-yaml", str(cfg_path), str(merged_dir), "--allow-crimes"],
            self.out_dir / "mergekit.log",
        )

        mlx_dir: Path | None = None
        if self.quantize:
            _check_tool("mlx_vlm.convert")
            mlx_dir = self.out_dir / "mlx"
            mlx_dir.mkdir(parents=True, exist_ok=True)
            runner(
                [
                    "mlx_vlm.convert",
                    "--hf-path", str(merged_dir),
                    "--mlx-path", str(mlx_dir),
                    "-q",
                    "--q-bits", str(self.quant_bits),
                ],
                self.out_dir / "mlx_convert.log",
            )

        registered_id: str | None = None
        if self.register_as:
            from ..vision.registry import register_model

            entry = register_model(
                self.register_as,
                str(mlx_dir if mlx_dir is not None else merged_dir),
                kind="merged",
                label=self.register_label or self.register_as,
            )
            registered_id = entry.id

        summary = MergeRunSummary(
            out_dir=self.out_dir,
            config_path=cfg_path,
            merged_dir=merged_dir,
            mlx_dir=mlx_dir,
            registered_id=registered_id,
            elapsed_s=time.time() - t0,
            method=self.cfg.merge_method,
            sources=[s.model for s in self.cfg.sources],
        )
        (self.out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "out_dir": str(summary.out_dir),
                    "config_path": str(summary.config_path),
                    "merged_dir": str(summary.merged_dir),
                    "mlx_dir": str(summary.mlx_dir) if summary.mlx_dir else None,
                    "registered_id": summary.registered_id,
                    "elapsed_s": summary.elapsed_s,
                    "method": summary.method,
                    "sources": summary.sources,
                },
                indent=2,
            )
        )
        return summary


def list_merges(root: Path | None = None) -> list[dict]:
    root = root or MERGES_ROOT
    if not root.exists():
        return []
    out: list[dict] = []
    for d in sorted(root.iterdir(), reverse=True):
        s = d / "summary.json"
        if s.exists():
            try:
                out.append(json.loads(s.read_text()))
            except json.JSONDecodeError:
                continue
    return out


def parse_sources(raw: Iterable[str]) -> list[MergeSource]:
    """Parse `repo[@weight[:density]]` strings from the CLI."""
    out: list[MergeSource] = []
    for s in raw:
        density = None
        if ":" in s:
            s, dstr = s.rsplit(":", 1)
            density = float(dstr)
        weight = 0.5
        if "@" in s:
            s, wstr = s.rsplit("@", 1)
            weight = float(wstr)
        out.append(MergeSource(model=s, weight=weight, density=density))
    return out
