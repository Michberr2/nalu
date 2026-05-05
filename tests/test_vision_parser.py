from nalu.agents.vision import Action


def test_native_ui_tars_click_with_box_tokens():
    a = Action.parse(
        "Thought: Click the Apple menu.\n"
        "Action: click(start_box='<|box_start|>(13,13)<|box_end|>')"
    )
    assert a.kind == "click"
    assert a.args == {"x": 13, "y": 13}
    assert "Apple menu" in a.reason


def test_native_ui_tars_click_no_box_tokens():
    a = Action.parse("Action: click(start_box='[420, 360]')")
    assert a.kind == "click"
    assert a.args == {"x": 420, "y": 360}


def test_left_double_click_maps_to_double_click():
    a = Action.parse("Action: left_double_click(start_box='<|box_start|>(50,60)<|box_end|>')")
    assert a.kind == "double_click"
    assert a.args == {"x": 50, "y": 60}


def test_type_action():
    a = Action.parse("Action: type(content='hello world\\n')")
    assert a.kind == "type"
    assert a.args["text"].startswith("hello world")


def test_hotkey_action():
    a = Action.parse("Action: hotkey(key='ctrl+c')")
    assert a.kind == "key"
    assert a.args["modifiers"] == ["ctrl"]
    assert a.args["name"] == "c"


def test_scroll_down_action():
    a = Action.parse("Action: scroll(start_box='(640,400)', direction='down')")
    assert a.kind == "scroll"
    assert a.args["dy"] == 120
    assert a.args["dx"] == 0


def test_scroll_up_action():
    a = Action.parse("Action: scroll(start_box='(640,400)', direction='up')")
    assert a.args["dy"] == -120


def test_finished_maps_to_done_with_answer():
    a = Action.parse("Action: finished(content='the focused window is Terminal.')")
    assert a.kind == "done"
    assert "Terminal" in a.args["answer"]


def test_wait_action():
    a = Action.parse("Action: wait()")
    assert a.kind == "wait"


def test_bare_json_format():
    a = Action.parse('{"action": "left_click", "coordinate": [100, 200]}')
    assert a.kind == "click"
    assert a.args["x"] == 100
    assert a.args["y"] == 200


def test_truncated_json_recovers():
    a = Action.parse('{"name": "terminal", "action": "left_click", "coordinate": [1008, 230]')
    assert a.kind == "click"
    assert a.args["x"] == 1008
    assert a.args["y"] == 230


def test_natural_language_press_fallback():
    a = Action.parse('Action: press "Command + Space"')
    assert a.kind == "key"
    assert "cmd" in a.args["modifiers"]
    assert a.args["name"] == "space"


def test_press_keys_variant_is_normalized():
    a = Action.parse("Action: press_keys(key='cmd+space')")
    assert a.kind == "key"
    assert a.args["modifiers"] == ["cmd"]
    assert a.args["name"] == "space"


def test_unparseable_returns_error():
    a = Action.parse("here is some prose with no structure at all")
    assert a.kind == "error"


def test_thought_is_captured_as_reason():
    a = Action.parse(
        "Thought: I should click the search icon.\n"
        "Action: click(start_box='(10,20)')"
    )
    assert "search icon" in a.reason
