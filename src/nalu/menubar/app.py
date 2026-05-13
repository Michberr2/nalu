"""NSStatusBar shell that observes `MenubarState` and dispatches actions to the bus.

Run with `nalu menubar`. Connects to the daemon's bus over UDS, subscribes to the
event topics the state machine cares about, and rebuilds the menu on every change.
The Cocoa run loop owns the main thread; bus events arrive on a background asyncio
thread and call `state.apply_event(...)`, which schedules `_rebuild()` via
`performSelectorOnMainThread:`.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import webbrowser
from pathlib import Path

from .. import config, daemon
from ..agents.vision import active_model_id, list_models
from ..bus import BusClient
from .state import MenubarState, build_menu, summarize_status

MENU_TOPICS = (
    "user_intent",
    "task_completed",
    "task_failed",
    "task_paused",
    "pause_state",
    "vision_swap_completed",
    "vision_model_swap_completed",
)


def _refresh_models(state: MenubarState) -> None:
    try:
        rows = [
            {"id": m.id, "label": m.label, "path": m.path}
            for m in list_models()
        ]
        state.set_models(rows, active_model_id())
    except Exception:
        pass


class BusBridge:
    """Runs an asyncio loop in a background thread that mirrors bus events into state."""

    def __init__(self, state: MenubarState):
        self.state = state
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def publish(self, topic: str, payload: dict) -> None:
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._publish(topic, payload), loop)

    async def _publish(self, topic: str, payload: dict) -> None:
        client = BusClient(source="menubar")
        try:
            await client.connect()
            await client.publish(topic, payload)
        finally:
            await client.close()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception:
            pass

    async def _main(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            self.state.set_daemon_running(daemon.is_running())
            if not daemon.is_running():
                await asyncio.sleep(min(backoff, 5.0))
                backoff = min(backoff * 1.5, 5.0)
                continue
            backoff = 1.0
            sub = BusClient(source="menubar-listener")
            try:
                await sub.connect()
                for topic in MENU_TOPICS:
                    await sub.subscribe(topic, self._on_event)
                _refresh_models(self.state)
                while not self._stop.is_set() and daemon.is_running():
                    await asyncio.sleep(2.0)
            except Exception:
                pass
            finally:
                try:
                    await sub.close()
                except Exception:
                    pass
                self.state.set_daemon_running(False)
            await asyncio.sleep(1.0)

    async def _on_event(self, ev) -> None:
        self.state.apply_event(ev.topic, ev.payload, ev.ts)


def _open_dashboard() -> None:
    subprocess.Popen(
        ["uv", "run", "nalu", "dashboard"],
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ask_via_dialog(bridge: BusBridge) -> None:
    """Use AppleScript to pop a text-input dialog without dragging a Tk dependency in."""
    script = (
        'tell application "System Events" to display dialog '
        '"Ask Nalu" default answer "" with title "Nalu" '
        'buttons {"Cancel","Ask"} default button "Ask"'
    )
    try:
        out = subprocess.check_output(
            ["osascript", "-e", script],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return  # user cancelled
    text = ""
    for token in out.split(", "):
        if token.startswith("text returned:"):
            text = token.split(":", 1)[1].strip()
            break
    if text:
        bridge.publish("user_intent", {"text": text, "via": "menubar"})


def run() -> int:
    """Entry point for `nalu menubar`. Owns the main thread."""
    try:
        from AppKit import (
            NSApplication, NSApp, NSApplicationActivationPolicyAccessory,
            NSMenu, NSMenuItem, NSStatusBar, NSVariableStatusItemLength,
        )
        from Foundation import NSObject
        from PyObjCTools import AppHelper
    except ImportError as e:
        print(f"menu-bar requires PyObjC AppKit: {e}")
        return 1

    state = MenubarState()
    bridge = BusBridge(state)

    class Controller(NSObject):
        def init(self):
            self = NSObject.init(self)
            self._actions = {}
            return self

        def setStatusItem_(self, item):
            self._status_item = item

        def menuItemClicked_(self, sender):
            tag = sender.tag()
            entry = self._actions.get(tag)
            if not entry:
                return
            action = entry["action"]
            payload = entry["payload"]
            if action == "ask":
                threading.Thread(target=_ask_via_dialog, args=(bridge,), daemon=True).start()
            elif action == "pause":
                bridge.publish("pause_request", {"paused": True})
            elif action == "resume":
                bridge.publish("pause_request", {"paused": False})
            elif action == "use_model":
                bridge.publish("vision_swap_model", {"model_id": payload["id"]})
            elif action == "dashboard":
                _open_dashboard()
            elif action == "quit":
                bridge.stop()
                NSApp.terminate_(None)

        def rebuild(self):
            menu = NSMenu.alloc().init()
            self._actions = {}
            counter = 0
            for item in build_menu(state):
                counter += 1
                if item.separator:
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                ns = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    item.title,
                    "menuItemClicked:" if item.action and item.enabled else None,
                    "",
                )
                ns.setEnabled_(item.enabled)
                ns.setTarget_(self)
                ns.setTag_(counter)
                self._actions[counter] = {"action": item.action, "payload": item.payload}
                if item.submenu:
                    sub = NSMenu.alloc().init()
                    for child in item.submenu:
                        counter += 1
                        if child.separator:
                            sub.addItem_(NSMenuItem.separatorItem())
                            continue
                        cn = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                            child.title,
                            "menuItemClicked:" if child.action and child.enabled else None,
                            "",
                        )
                        cn.setEnabled_(child.enabled)
                        cn.setTarget_(self)
                        cn.setTag_(counter)
                        self._actions[counter] = {"action": child.action, "payload": child.payload}
                        sub.addItem_(cn)
                    ns.setSubmenu_(sub)
                menu.addItem_(ns)
            self._status_item.setMenu_(menu)
            self._status_item.button().setTitle_("Nalu")
            self._status_item.button().setToolTip_(summarize_status(state))

    NSApplication.sharedApplication()
    NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    controller = Controller.alloc().init()
    item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
    item.button().setTitle_("Nalu")
    controller.setStatusItem_(item)
    controller.rebuild()

    state.subscribe(lambda: controller.performSelectorOnMainThread_withObject_waitUntilDone_(
        "rebuild", None, False,
    ))

    bridge.start()
    try:
        AppHelper.runEventLoop()
    finally:
        bridge.stop()
    return 0
