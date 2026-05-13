from __future__ import annotations

from nalu.menubar.state import (
    STATUS_IDLE,
    STATUS_NO_DAEMON,
    STATUS_PAUSED,
    STATUS_WORKING,
    MenubarState,
    build_menu,
    summarize_status,
)


def _items_by_action(state):
    return {item.action: item for item in build_menu(state) if item.action}


def test_initial_state_reports_no_daemon():
    s = MenubarState()
    assert s.status() == STATUS_NO_DAEMON
    titles = [i.title for i in build_menu(s)]
    assert any("offline" in t for t in titles)


def test_set_daemon_running_transitions_to_idle():
    s = MenubarState()
    s.set_daemon_running(True)
    assert s.status() == STATUS_IDLE


def test_user_intent_marks_working():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("user_intent", {"text": "open finder"}, ts=10.0)
    assert s.status() == STATUS_WORKING
    assert s.current_goal == "open finder"
    assert s.last_goal == "open finder"
    assert list(s.recent_turns)[-1]["text"] == "open finder"


def test_task_completed_clears_goal_and_records_turn():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("user_intent", {"text": "what time is it"}, ts=1.0)
    s.apply_event("task_completed", {"answer": "5pm"}, ts=2.0)
    assert s.current_goal is None
    assert s.last_answer == "5pm"
    assert s.recent_turns[-1] == {"role": "assistant", "text": "5pm", "ts": 2.0}


def test_task_failed_records_failure_with_prefix():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("user_intent", {"text": "do thing"}, ts=1.0)
    s.apply_event("task_failed", {"reason": "timeout"}, ts=2.0)
    assert s.current_goal is None
    assert s.last_failure == "timeout"
    assert s.recent_turns[-1]["text"] == "failed: timeout"


def test_pause_state_event_toggles_paused():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("pause_state", {"paused": True}, ts=1.0)
    assert s.status() == STATUS_PAUSED
    s.apply_event("pause_state", {"paused": False}, ts=2.0)
    assert s.status() == STATUS_IDLE


def test_setting_daemon_offline_clears_current_goal():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("user_intent", {"text": "go"}, ts=1.0)
    s.set_daemon_running(False)
    assert s.current_goal is None
    assert s.status() == STATUS_NO_DAEMON


def test_set_models_records_active():
    s = MenubarState()
    s.set_models(
        [{"id": "a", "label": "A", "path": "x"}, {"id": "b", "label": "B", "path": "y"}],
        active_id="b",
    )
    assert s.active_model_id == "b"
    assert len(s.models) == 2


def test_vision_model_swap_completed_updates_active_by_path():
    s = MenubarState()
    s.set_models(
        [{"id": "a", "label": "A", "path": "x"}, {"id": "b", "label": "B", "path": "y"}],
        active_id="a",
    )
    s.apply_event("vision_model_swap_completed", {"model": "y"}, ts=1.0)
    assert s.active_model_id == "b"


def test_subscribe_fires_on_mutation():
    s = MenubarState()
    calls: list[int] = []
    s.subscribe(lambda: calls.append(1))
    s.set_daemon_running(True)
    s.apply_event("user_intent", {"text": "x"}, ts=1.0)
    s.apply_event("task_completed", {"answer": "y"}, ts=2.0)
    assert len(calls) == 3


def test_subscribe_swallows_listener_exceptions():
    s = MenubarState()

    def bad():
        raise RuntimeError("boom")

    calls: list[int] = []
    s.subscribe(bad)
    s.subscribe(lambda: calls.append(1))
    s.set_daemon_running(True)
    assert calls == [1]


def test_set_daemon_running_idempotent():
    s = MenubarState()
    calls: list[int] = []
    s.subscribe(lambda: calls.append(1))
    s.set_daemon_running(True)
    s.set_daemon_running(True)
    assert calls == [1]


def test_summarize_status_truncates_long_goal():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("user_intent", {"text": "x" * 200}, ts=1.0)
    summary = summarize_status(s)
    assert summary.startswith("Nalu — working: ")
    assert len(summary) < 80


def test_build_menu_offline_omits_ask():
    s = MenubarState()
    actions = _items_by_action(s)
    assert "ask" not in actions
    assert "dashboard" in actions
    assert "quit" in actions


def test_build_menu_running_includes_pause_and_models():
    s = MenubarState()
    s.set_daemon_running(True)
    s.set_models(
        [{"id": "a", "label": "A", "path": "x"}, {"id": "b", "label": "B", "path": "y"}],
        active_id="a",
    )
    items = build_menu(s)
    actions = [i.action for i in items]
    assert "ask" in actions
    assert "pause" in actions
    model_sub = next(i for i in items if i.title == "Model")
    use_actions = [c.payload.get("id") for c in model_sub.submenu]
    assert use_actions == ["a", "b"]
    assert model_sub.submenu[0].title.startswith("● ")
    assert model_sub.submenu[1].title.startswith("  ")


def test_build_menu_paused_swaps_pause_to_resume():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("pause_state", {"paused": True}, ts=1.0)
    actions = _items_by_action(s)
    assert "resume" in actions
    assert "pause" not in actions


def test_build_menu_includes_recent_submenu_when_history_present():
    s = MenubarState()
    s.set_daemon_running(True)
    s.apply_event("user_intent", {"text": "open foo"}, ts=1.0)
    s.apply_event("task_completed", {"answer": "ok"}, ts=2.0)
    items = build_menu(s)
    recent = next((i for i in items if i.title == "Recent"), None)
    assert recent is not None
    assert any("You: open foo" in c.title for c in recent.submenu)
    assert any("Nalu: ok" in c.title for c in recent.submenu)
