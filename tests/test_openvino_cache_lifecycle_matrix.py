import pytest

from experiments.paper6_3_openvino.run_cache_lifecycle_cell import request_plan


SHORT = ({"role": "user", "content": "short"},)
LONG = ({"role": "user", "content": "long"},)


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("short", ["short"] * 5),
        ("long", ["long"]),
        ("long_short", ["long"] + ["short"] * 5),
        ("long_long", ["long"] * 5),
    ],
)
def test_request_plan_preserves_lifecycle_transition(scenario, expected) -> None:
    plan = request_plan(scenario, SHORT, LONG, 5)
    assert [kind for kind, _ in plan] == expected


def test_request_plan_rejects_too_few_repeats() -> None:
    with pytest.raises(ValueError, match="at least two"):
        request_plan("short", SHORT, LONG, 1)
