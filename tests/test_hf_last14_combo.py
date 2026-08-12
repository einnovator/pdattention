"""Protocol-level tests for the final last-14 adaptation comparison."""

import pytest

from experiments.paper2_hf.qa.run_last14_combo import (
    _aggregates,
    _valid_ratio,
    last_band_layers,
    variant_from_name,
)


def test_last14_layer_band_is_exact_and_validated():
    assert last_band_layers(28) == tuple(range(14, 28))
    with pytest.raises(ValueError, match="Cannot select"):
        last_band_layers(8)


def test_combo_variant_keeps_residual_and_lora_controls_independent():
    variant = variant_from_name("combo_residual_32_lora_r8")
    assert variant.residual_width == 32
    assert variant.lora_rank == 8
    assert variant_from_name("residual_16").lora_rank == 0
    assert variant_from_name("lora_o_r4").residual_width == 0


def test_recovery_ratio_rejects_invalid_context_denominators():
    assert _valid_ratio(2.0, 4.0, 0.05) == 0.5
    assert _valid_ratio(2.0, -1.0, 0.05) is None
    assert _valid_ratio(2.0, 0.01, 0.05) is None
    assert _valid_ratio(2.0, None, 0.05) is None


def test_aggregate_retains_ratios_when_first_seed_has_no_valid_full_context():
    rows = [
        {
            "seed": 11,
            "variant": "fixed",
            "dataset": "hotpotqa",
            "condition": "oracle",
            "examples": 2,
            "rho_full": None,
            "gold_sequence_logprob_delta_vs_none": 1.0,
        },
        {
            "seed": 23,
            "variant": "fixed",
            "dataset": "hotpotqa",
            "condition": "oracle",
            "examples": 2,
            "rho_full": 0.5,
            "gold_sequence_logprob_delta_vs_none": 2.0,
        },
    ]
    aggregate = _aggregates(rows)[0]
    assert aggregate["rho_full_mean"] == 0.5
    assert aggregate["gold_sequence_logprob_delta_vs_none_mean"] == 1.5


def test_aggregate_recovery_uses_ratio_of_matched_cohort_means():
    rows = [
        {
            "seed": seed,
            "variant": "fixed",
            "dataset": "hotpotqa",
            "condition": "routed",
            "examples": 8,
            "gold_sequence_logprob_delta_vs_none": gain,
            "direct_sequence_benefit": direct,
            "full_sequence_benefit": full,
        }
        for seed, gain, direct, full in (
            (11, 2.0, 8.0, 4.0),
            (23, 4.0, 12.0, 8.0),
        )
    ]
    aggregate = _aggregates(rows)[0]
    assert aggregate["rho_direct_cohort"] == pytest.approx(3.0 / 10.0)
    assert aggregate["rho_full_cohort"] == pytest.approx(3.0 / 6.0)
