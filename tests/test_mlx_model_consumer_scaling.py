from experiments.paper6_2_mlx.run_model_consumer_scaling import (
    _percentile,
    matched_costs,
    resolve_consumer_layers,
)
from experiments.paper6_2_mlx.summarize_model_consumer_scaling import summarize_model


def test_consumer_layers_use_contiguous_suffix_and_round_up():
    assert resolve_consumer_layers(40, "last_7_8") == tuple(range(5, 40))
    assert resolve_consumer_layers(36, "last_2_3") == tuple(range(12, 36))


def test_matched_costs_keep_warm_and_cold_states_separate():
    assert matched_costs(
        {"completion_latency_ms": 40.0, "representation_encode_ms": 60.0}
    ) == {"warm_request_ms": 40.0, "cold_usable_context_ms": 100.0}


def test_nearest_rank_percentiles_expose_first_use_tail_outliers():
    values = [10.0, 11.0, 12.0, 13.0, 1_000.0]
    assert _percentile(values, 0.5) == 12.0
    assert _percentile(values, 0.95) == 1_000.0


def _row(condition, f1, logprob, agreement, warm, cold, fraction):
    return {
        "dataset": "qasper",
        "seed": 11,
        "example_id": "example",
        "condition": condition,
        "token_f1": f1,
        "gold_answer_logprob": logprob,
        "sequence_agreement_vs_e0": agreement,
        "consumer_layer_fraction": fraction,
        "warm_request_ms": warm,
        "cold_usable_context_ms": cold,
        "active_detail_bytes": 1024,
    }


def test_summary_uses_matched_ratios_and_reports_balanced_candidate():
    payload = {
        "model_id": "mlx-community/Qwen3-8B-4bit",
        "model_revision": "revision",
        "layer_count": 36,
        "rows": [
            _row("E0_WARM", 0.5, -2.0, 1.0, 100.0, 200.0, 0.0),
            _row("E2_CONCAT_WARM", 0.5, -2.0, 1.0, 80.0, 180.0, 1.0),
            _row("E2_SEGMENTED_ALL_LAYERS", 0.5, -2.0, 1.0, 90.0, 190.0, 1.0),
            _row("E2_SEGMENTED_LAST_1_2", 0.49, -2.2, 0.0, 70.0, 170.0, 0.5),
        ],
    }
    report = summarize_model(payload)
    concat = next(
        row for row in report["conditions"] if row["condition"] == "E2_CONCAT_WARM"
    )
    assert concat["warm_cost_ratio_vs_e0"] == 0.8
    assert concat["cold_cost_ratio_vs_e0"] == 0.9
    assert report["minimum_balanced_smoke_fraction"] == 0.5
    assert report["minimum_strict_transport_fraction"] == 1.0
