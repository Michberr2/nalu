from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ... import config


SYSTEM_PROMPT = """You are Nalu's vision-action module. You see one screenshot and one user goal.
You output exactly one action as a single JSON object on a line, nothing else.
Schema:
  {"action": "click", "x": <0-1280>, "y": <0-800>, "reason": "<short>"}
  {"action": "type", "text": "<string>", "reason": "<short>"}
  {"action": "key", "name": "return|tab|escape|...", "modifiers": ["cmd"|"shift"|...], "reason": "<short>"}
  {"action": "scroll", "dx": <int>, "dy": <int>, "reason": "<short>"}
  {"action": "wait", "ms": <int>, "reason": "<short>"}
  {"action": "done", "answer": "<short>", "reason": "<short>"}
Coordinates are in the captured image's pixel space. Be precise.
"""


@dataclass
class Action:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @classmethod
    def parse(cls, text: str) -> "Action":
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            return cls(kind="error", reason=f"no JSON in model output: {text[:200]}")
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            return cls(kind="error", reason=f"bad JSON: {e}")
        kind = obj.pop("action", "error")
        reason = obj.pop("reason", "")
        return cls(kind=kind, args=obj, reason=reason)


class VisionAgent:
    """Wraps a local MLX-VLM model. Loads lazily so tests / dashboard don't need it."""

    def __init__(self, model_id: str = config.VISION_MODEL):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        cfg = load_config(self.model_id)
        self._model, self._processor = load(self.model_id, processor_config={"trust_remote_code": True})
        self._cfg = cfg

    def decide(self, image: Image.Image, goal: str, history: list[str] | None = None) -> Action:
        self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        history_str = "\n".join(history or [])
        user = f"Goal: {goal}\n\nRecent steps:\n{history_str}\n\nLook at the screenshot and emit the next action JSON."
        formatted = apply_chat_template(
            self._processor, self._cfg, user, num_images=1, system_prompt=SYSTEM_PROMPT
        )
        out = generate(
            self._model,
            self._processor,
            formatted,
            [image],
            max_tokens=256,
            temperature=0.0,
            verbose=False,
        )
        text = out.text if hasattr(out, "text") else str(out)
        return Action.parse(text)
