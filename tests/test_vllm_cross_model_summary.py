from experiments.paper6_vllm.summarize_cross_model import _cohort_row


def test_cohort_row_reports_exact_pairs_and_regime_ratios() -> None:
    parity = []
    aggregates = []
    for regime in (
        "cold_one_shot",
        "warm_repeated",
        "multi_query_same_resource",
        "concurrent_shared_resource",
    ):
        parity.append(
            {
                "engine": "vllm-metal",
                "dataset": "qasper",
                "regime": regime,
                "paired_requests": 2,
                "exact_output_parity": 1.0,
            }
        )
        for condition in ("e0_selected_text", "e2_native_kv"):
            aggregates.append(
                {
                    "engine": "vllm-metal",
                    "dataset": "qasper",
                    "regime": regime,
                    "condition": condition,
                    "cold_end_to_end_completion_ms": (
                        100 if condition == "e0_selected_text" else 110
                    ),
                    "completion_p50_ms": (
                        50 if condition == "e0_selected_text" else 55
                    ),
                    "requests_per_second": (
                        10 if condition == "e0_selected_text" else 12.5
                    ),
                }
            )

    row = _cohort_row("test/model", {"parity": parity, "aggregates": aggregates})

    assert row["exact_pairs"] == 8
    assert row["paired_requests"] == 8
    assert row["cold_one_shot_e2_over_e0"] == 1.1
    assert row["warm_repeated_e2_over_e0"] == 1.1
    assert row["concurrent_shared_resource_e2_over_e0"] == 0.8
