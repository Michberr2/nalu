from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from ... import config


SYSTEM_PROMPT = """You are a GUI agent. You see a screenshot of the user's screen and a task instruction.
You output the next action to perform on the GUI.

## Action Space
click(start_box='[x,y]')
left_double_click(start_box='[x,y]')
right_single_click(start_box='[x,y]')
type(content='string to type')
hotkey(key='cmd+a')
scroll(start_box='[x,y]', direction='up' or 'down')
wait()
finished(content='answer or completion message')

## Output Format
Thought: one sentence about what you will do next and why.
Action: one function call from the Action Space.

## Notes
- Coordinates are pixels in the screenshot you are shown.
- Always emit BOTH a Thought line and an Action line.
- When the task is informational and no further GUI action is needed, use finished(content='answer').
"""

_UI_TARS_TO_NALU = {
    "click": "click",
    "left_click": "click",
    "left_single_click": "click",
    "left_double_click": "double_click",
    "right_click": "click",
    "right_single_click": "click",
    "type": "type",
    "input": "type",
    "hotkey": "key",
    "key": "key",
    "scroll": "scroll",
    "wait": "wait",
    "finished": "done",
    "done": "done",
    "call_user": "done",
}


_BOX_TOKEN_RE = re.compile(r"<\|box_(?:start|end)\|>")
_FUNC_CALL_RE = re.compile(r"(\w+)\s*\(([^)]*)\)")
_COORD_RE = re.compile(r"\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?")


def _strip_tokens(text: str) -> str:
    return _BOX_TOKEN_RE.sub("", text)


def _parse_kwargs(arg_str: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in re.finditer(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,]+?))(?=\s*,\s*\w+\s*=|$)", arg_str):
        k = m.group(1)
        v = next((g for g in m.groups()[1:] if g is not None), "")
        out[k] = v.strip()
    return out


@dataclass
class Action:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @classmethod
    def parse(cls, text: str) -> "Action":
        text = _strip_tokens(text).strip()

        # Format 1: "Thought: ...\nAction: click(start_box='[x,y]')"
        action_match = re.search(r"Action\s*:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        thought_match = re.search(r"Thought\s*:\s*(.+?)(?:\nAction|$)", text, re.DOTALL)
        reason = thought_match.group(1).strip() if thought_match else ""
        candidate = action_match.group(1).strip() if action_match else text

        fc = _FUNC_CALL_RE.search(candidate)
        if fc:
            fn, raw_args = fc.group(1), fc.group(2)
            kind = _UI_TARS_TO_NALU.get(fn.lower(), fn.lower())
            kwargs = _parse_kwargs(raw_args)
            args: dict[str, Any] = {}
            for box_key in ("start_box", "box", "coordinate", "point"):
                if box_key in kwargs:
                    cm = _COORD_RE.search(kwargs[box_key])
                    if cm:
                        args["x"] = int(cm.group(1))
                        args["y"] = int(cm.group(2))
                    break
            if "content" in kwargs:
                args["text"] = kwargs["content"]
                if kind == "done":
                    args["answer"] = kwargs["content"]
            if "key" in kwargs:
                parts = [p.strip().lower() for p in kwargs["key"].split("+")]
                args["modifiers"] = parts[:-1]
                args["name"] = parts[-1]
            if "direction" in kwargs:
                d = kwargs["direction"].lower()
                args["dx"] = 0
                args["dy"] = -120 if d == "up" else 120 if d == "down" else 0
            return cls(kind=kind, args=args, reason=reason or fn)

        # Format 2: bare JSON {"action": "...", ...}, possibly truncated.
        jm = re.search(r"\{[^{}]*\}?", candidate, re.DOTALL)
        if jm:
            blob = jm.group(0)
            if not blob.rstrip().endswith("}"):
                blob = blob.rstrip().rstrip(",") + "}"
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                obj = None
            if obj is not None:
                raw_kind = obj.pop("action", None) or obj.pop("name", None) or "error"
                kind = _UI_TARS_TO_NALU.get(str(raw_kind).lower(), str(raw_kind).lower())
                if "coordinate" in obj and isinstance(obj["coordinate"], list) and len(obj["coordinate"]) >= 2:
                    obj["x"], obj["y"] = obj["coordinate"][0], obj["coordinate"][1]
                obj.pop("coordinate", None)
                obj.pop("name", None)
                r = obj.pop("reason", "") or reason
                return cls(kind=kind, args=obj, reason=r)

        # Format 3: regex-only fallback for badly-formed model output.
        action_field = re.search(r"\"action\"\s*:\s*\"([^\"]+)\"", candidate)
        coord_field = re.search(r"\"coordinate\"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", candidate)
        if action_field:
            kind = _UI_TARS_TO_NALU.get(action_field.group(1).lower(), action_field.group(1).lower())
            args: dict[str, Any] = {}
            if coord_field:
                args["x"] = int(coord_field.group(1))
                args["y"] = int(coord_field.group(2))
            return cls(kind=kind, args=args, reason=reason or "fallback-regex")

        return cls(kind="error", reason=f"unparseable model output: {text[:300]}")


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

        history_str = "\n".join(history or []) or "(no prior steps)"
        user = (
            f"## User Instruction\n{goal}\n\n"
            f"## Action History\n{history_str}\n\n"
            f"What is your next Thought and Action?"
        )
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
