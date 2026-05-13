from __future__ import annotations

import ast
import gc
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ... import config


SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double_click(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single_click(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x2,y2)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use English in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.
- Coordinates are absolute pixel positions in the screenshot.
- Every reply MUST contain an `Action:` line. Do not reply with only a description of the screen.
- If the target app or element is not visible, launch it via Spotlight: `hotkey(key='cmd space')`, then on the next turn `type(content='AppName\\n')`.
- When the user goal is to compute or retrieve a value, finish with `finished(content='<the value>')` once you have it.

## User Instruction
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
    "press": "key",
    "press_keys": "key",
    "keypress": "key",
    "scroll": "scroll",
    "wait": "wait",
    "finished": "done",
    "done": "done",
    "call_user": "done",
}


_SENTINEL = object()

_BOX_TOKEN_RE = re.compile(r"<\|box_(?:start|end)\|>")
_FUNC_CALL_RE = re.compile(r"(\w+)\s*\((.*)\)\s*$")
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

        # Format 2a: <verb> {<dict>} — e.g. `click {'x': 10, 'y': 10}`.
        verb_dict = re.match(r"\s*(\w+)\s*(\{.*\})\s*$", candidate, re.DOTALL)
        if verb_dict:
            try:
                d = ast.literal_eval(verb_dict.group(2))
            except (ValueError, SyntaxError):
                d = None
            if isinstance(d, dict):
                kind = _UI_TARS_TO_NALU.get(verb_dict.group(1).lower(), verb_dict.group(1).lower())
                args = {k: v for k, v in d.items()}
                if "coordinate" in args and isinstance(args["coordinate"], (list, tuple)) and len(args["coordinate"]) >= 2:
                    args["x"], args["y"] = args["coordinate"][0], args["coordinate"][1]
                    args.pop("coordinate", None)
                if "content" in args and kind == "done":
                    args["answer"] = args["content"]
                if "content" in args and kind == "type":
                    args["text"] = args.pop("content")
                return cls(kind=kind, args=args, reason=reason or verb_dict.group(1))

        # Format 2b: bare JSON {"action": "...", ...}, possibly truncated.
        jm = re.search(r"\{[^{}]*\}?", candidate, re.DOTALL)
        if jm:
            blob = jm.group(0)
            if not blob.rstrip().endswith("}"):
                blob = blob.rstrip().rstrip(",") + "}"
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(blob)
                except (ValueError, SyntaxError):
                    obj = None
            if isinstance(obj, dict):
                raw_kind = obj.pop("action", None) or obj.pop("name", None) or "error"
                kind = _UI_TARS_TO_NALU.get(str(raw_kind).lower(), str(raw_kind).lower())
                if "coordinate" in obj and isinstance(obj["coordinate"], (list, tuple)) and len(obj["coordinate"]) >= 2:
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

        # Format 4: natural-language `press "Cmd + Space"` style.
        press_match = re.search(r"\b(?:press|hotkey|keypress)\b\s*[\"']?([A-Za-z0-9+\s]+?)[\"']?\s*$", candidate, re.IGNORECASE)
        if press_match:
            parts = [p.strip().lower().replace("command", "cmd").replace("control", "ctrl").replace("option", "alt") for p in press_match.group(1).split("+")]
            parts = [p for p in parts if p]
            if parts:
                return cls(kind="key", args={"modifiers": parts[:-1], "name": parts[-1]}, reason=reason or "fallback-press")

        return cls(kind="error", reason=f"unparseable model output: {text[:300]}")


class VisionAgent:
    """Wraps a local MLX-VLM model. Loads lazily so tests / dashboard don't need it."""

    def __init__(self, model_id: str | None = None):
        from . import registry

        self._registry = registry
        self.model_id = model_id or registry.active_model().path
        self._model = None
        self._processor = None
        self._cfg = None
        self._adapter_dir: Path | None = None
        self._lock = threading.Lock()

    @property
    def adapter_dir(self) -> Path | None:
        return self._adapter_dir

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            self._load_locked(adapter_override=_SENTINEL)

    def swap_adapter(self, target: Path | None | object = None) -> Path | None:
        """Reload the base model and apply `target` (or the active pointer if omitted).

        Pass `None` explicitly to drop any adapter and run the base model.
        Acquires the same lock as `decide()`, so concurrent calls queue.
        """
        with self._lock:
            self._load_locked(adapter_override=target)
            return self._adapter_dir

    def swap_model(self, model_id: str, adapter: Path | None | object = _SENTINEL) -> str:
        """Swap to a different registered base model (and optionally re-apply an adapter).

        Defaults to the registry's active adapter pointer. Pass `None` to drop adapters.
        """
        new_path = self._registry.resolve_model_path(model_id)
        with self._lock:
            self.model_id = new_path
            self._load_locked(adapter_override=adapter)
            return self.model_id

    def _load_locked(self, adapter_override: Path | None | object) -> None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        if self._model is not None:
            self._model = None
            self._processor = None
            self._cfg = None
            self._adapter_dir = None
            gc.collect()

        cfg = load_config(self.model_id)
        model, processor = load(
            self.model_id, processor_config={"trust_remote_code": True}
        )
        self._cfg = cfg
        self._processor = processor
        self._model = model

        if adapter_override is _SENTINEL:
            from ..trainer.runner import active_adapter_dir

            adapter = active_adapter_dir()
        elif adapter_override is None:
            adapter = None
        else:
            adapter = Path(adapter_override)
            if not (adapter / "adapters.safetensors").exists():
                adapter = None

        if adapter is not None:
            from mlx_vlm.trainer.utils import apply_lora_layers

            self._model = apply_lora_layers(self._model, str(adapter))
            self._adapter_dir = adapter

    def decide(
        self,
        image: Image.Image,
        goal: str,
        history: list[str] | None = None,
        conversation: str | None = None,
    ) -> Action:
        self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        history_str = "\n".join(history or []) or "(no prior steps)"
        conv_block = f"## Conversation\n{conversation.strip()}\n\n" if conversation and conversation.strip() else ""
        user = (
            f"{conv_block}"
            f"## User Instruction\n{goal}\n\n"
            f"## Action History\n{history_str}\n\n"
            f"What is your next Thought and Action?"
        )
        with self._lock:
            formatted = apply_chat_template(
                self._processor, self._cfg, user, num_images=1, system_prompt=SYSTEM_PROMPT
            )
            out = generate(
                self._model,
                self._processor,
                formatted,
                [image],
                max_tokens=1024,
                temperature=0.0,
                verbose=False,
            )
        text = out.text if hasattr(out, "text") else str(out)
        return Action.parse(text)

    def judge(self, image: Image.Image, prompt: str, *, max_tokens: int = 256) -> str:
        """Run the model on a custom prompt and return raw output text (no Action parsing).

        Used by the planner's completion verifier. Reuses the same lock as `decide`,
        so verification calls and ordinary perceive→reason→act calls queue safely.
        """
        self.load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        with self._lock:
            formatted = apply_chat_template(
                self._processor, self._cfg, prompt, num_images=1, system_prompt=""
            )
            out = generate(
                self._model,
                self._processor,
                formatted,
                [image],
                max_tokens=max_tokens,
                temperature=0.0,
                verbose=False,
            )
        return out.text if hasattr(out, "text") else str(out)
