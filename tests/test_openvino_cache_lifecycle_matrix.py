import pytest

from experiments.paper6_3_openvino.run_cache_lifecycle_cell import request_plan
from experiments.paper6_3_openvino.summarize_cache_lifecycle_versions import (
    parse_teardown_log,
)


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


def test_teardown_log_reports_physical_block_deficit() -> None:
    result = parse_teardown_log(
        "BlockManager leaked sequence block tables: 1\n"
        "BlockAllocator leaked blocks. Expected num free blocks: 128, actual: 69\n"
        "BlockAllocator leaked blocks. Expected num free blocks: 128, actual: 69\n"
    )

    assert result == {
        "leaked_sequence_tables": 1,
        "expected_free_blocks": 128,
        "actual_free_blocks": 69,
        "leaked_blocks": 59,
        "allocator_reports": 2,
    }
