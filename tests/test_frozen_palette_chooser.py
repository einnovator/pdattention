from experiments.paper6_5_tools.run_frozen_palette_chooser import (
    arguments_match,
    choice_prompt,
    parse_call,
    parse_label,
)


def test_parse_label_maps_only_in_range_integer() -> None:
    names = ("get_user", "update_user")
    assert parse_label(" 1", names) == "update_user"
    assert parse_label("LABEL: 7", names) == ""
    assert parse_label("get_user", names) == ""


def test_choice_prompt_binds_stable_labels() -> None:
    prompt = choice_prompt("find u17", "record", ("get_user", "search_user"))
    assert "0 = get_user" in prompt
    assert "1 = search_user" in prompt
    assert prompt.endswith("LABEL:")


def test_parse_call_and_argument_matching() -> None:
    name, arguments = parse_call('prefix {"name":"get_user","arguments":{"user_id":"U17"}}')
    assert name == "get_user"
    assert arguments_match(arguments, {"user_id": "u17"})
    assert not arguments_match(arguments, {"user_id": "u18"})
