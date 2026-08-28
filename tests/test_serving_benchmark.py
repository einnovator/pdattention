from __future__ import annotations

import pytest

from pra_hf.serving_benchmark import benchmark_messages, percentile, run_serving_benchmark


def test_benchmark_conditions_separate_prefix_and_pra_memory() -> None:
    conditions = benchmark_messages()
    assert list(conditions) == [
        "no_prefix_no_pra",
        "prefix_only",
        "pra_only",
        "prefix_plus_pra",
        "full_context",
    ]
    assert "PRA_EVIDENCE_4821" not in str(conditions["prefix_only"])
    assert "PRA_EVIDENCE_4821" in str(conditions["pra_only"])
    assert len(str(conditions["full_context"])) > len(str(conditions["prefix_plus_pra"]))


def test_percentile_interpolates_and_handles_empty_input() -> None:
    assert percentile([], 0.5) is None
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_serving_benchmark_rejects_single_repeat() -> None:
    with pytest.raises(ValueError, match="At least two"):
        run_serving_benchmark(
            "http://engine", model="model", engine="mlx", repeats=1
        )

