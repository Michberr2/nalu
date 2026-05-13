"""Pure-Python state machine for the menu-bar UI.

Subscribes to bus events and exposes a snapshot of:
- daemon status (running / not running)
- current task (idle / working / paused) + last goal
- active model id
- recent conversation turns (capped)

Cocoa view code observes this and renders. Keeping it UI-free means we can
unit-test transitions without spinning up an NSStatusBar.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


STATUS_IDLE = "idle"
STATUS_WORKING = "working"
STATUS_PAUSED = "paused"
STATUS_NO_DAEMON = "no_daemon"


@dataclass
class MenuItem:
    title: str = ""
    action: str = ""              # opaque key the Cocoa shell uses to dispatch
    payload: dict = field(default_factory=dict)
    enabled: bool = True
    separator: bool = False
    submenu: list["MenuItem"] = field(default_factory=list)


def summarize_status(state: "MenubarState") -> str:
    if not state.daemon_running:
        return "Nalu — daemon offline"
    if state.paused:
        return "Nalu — paused"
    if state.current_goal:
        return f"Nalu — working: {state.current_goal[:48]}"
    return "Nalu — idle"


def build_menu(state: "MenubarState") -> list[MenuItem]:
    """Render the current state into a flat list of menu items.

    The Cocoa shell turns each into an NSMenuItem; submenus become NSMenu.
    """
    items: list[MenuItem] = []
    items.append(MenuItem(title=summarize_status(state), enabled=False))
    items.append(MenuItem(separator=True))

    if state.daemon_running:
        items.append(MenuItem(title="Ask Nalu…", action="ask"))
        if state.paused:
            items.append(MenuItem(title="Resume", action="resume"))
        else:
            items.append(MenuItem(title="Pause", action="pause"))
        if state.recent_turns:
            convo = MenuItem(title="Recent", submenu=[])
            for t in list(state.recent_turns)[-6:]:
                role = "You" if t.get("role") == "user" else "Nalu"
                text = (t.get("text") or "").strip().splitlines()[0][:64] or "(empty)"
                convo.submenu.append(MenuItem(title=f"{role}: {text}", enabled=False))
            items.append(convo)
    else:
        items.append(MenuItem(title="Start daemon (`nalu serve`)", enabled=False))

    if state.models:
        models_sub = MenuItem(title="Model", submenu=[])
        for m in state.models:
            mark = "● " if m["id"] == state.active_model_id else "  "
            models_sub.submenu.append(MenuItem(
                title=f"{mark}{m.get('label') or m['id']}",
                action="use_model",
                payload={"id": m["id"]},
                enabled=state.daemon_running,
            ))
        items.append(models_sub)

    items.append(MenuItem(separator=True))
    items.append(MenuItem(title="Open dashboard", action="dashboard"))
    items.append(MenuItem(title="Quit menu bar", action="quit"))
    return items


class MenubarState:
    """Thread-safe state holder. Bus event handlers call the `apply_*` methods.

    `subscribe(cb)` registers a callback that fires after every state mutation;
    the Cocoa shell uses it to schedule a menu rebuild on the main thread.
    """

    def __init__(self, *, max_recent: int = 20):
        self._lock = threading.Lock()
        self._listeners: list[Callable[[], None]] = []
        self.daemon_running: bool = False
        self.paused: bool = False
        self.current_goal: str | None = None
        self.last_goal: str | None = None
        self.last_answer: str | None = None
        self.last_failure: str | None = None
        self.active_model_id: str = ""
        self.models: list[dict] = []
        self.recent_turns: deque[dict] = deque(maxlen=max_recent)
        self.last_event_ts: float = 0.0

    def subscribe(self, cb: Callable[[], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass

    def status(self) -> str:
        if not self.daemon_running:
            return STATUS_NO_DAEMON
        if self.paused:
            return STATUS_PAUSED
        if self.current_goal:
            return STATUS_WORKING
        return STATUS_IDLE

    def set_daemon_running(self, running: bool) -> None:
        with self._lock:
            if self.daemon_running == running:
                return
            self.daemon_running = running
            if not running:
                self.current_goal = None
        self._notify()

    def set_models(self, models: list[dict], active_id: str) -> None:
        with self._lock:
            self.models = list(models)
            self.active_model_id = active_id
        self._notify()

    def apply_event(self, topic: str, payload: dict, ts: float | None = None) -> None:
        ts = ts if ts is not None else time.time()
        with self._lock:
            self.last_event_ts = ts
            if topic == "user_intent":
                goal = payload.get("text", "") or ""
                self.current_goal = goal
                self.last_goal = goal
                self.recent_turns.append({"role": "user", "text": goal, "ts": ts})
            elif topic == "task_completed":
                ans = payload.get("answer", "") or ""
                self.last_answer = ans
                self.current_goal = None
                self.recent_turns.append({"role": "assistant", "text": ans, "ts": ts})
            elif topic == "task_failed":
                reason = payload.get("reason", "") or ""
                self.last_failure = reason
                self.current_goal = None
                self.recent_turns.append({"role": "assistant", "text": f"failed: {reason}", "ts": ts})
            elif topic == "task_paused":
                self.paused = True
            elif topic == "pause_state":
                self.paused = bool(payload.get("paused"))
            elif topic == "vision_swap_completed":
                pass  # adapter swap; no state change for menu
            elif topic == "vision_model_swap_completed":
                model_path = payload.get("model")
                for m in self.models:
                    if m.get("path") == model_path:
                        self.active_model_id = m["id"]
                        break
        self._notify()
