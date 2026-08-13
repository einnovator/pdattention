import json

import pytest
import torch

from experiments.paper2_hf.qa.run_decision_aware_loss import (
    aggregate,
    aggregate_seed_means,
    audit_baseline_reproduction,
    class_balance,
    decision_aware_objective,
    grouped_polarity_logits,
    select_polarity_weight,
)


def test_grouped_polarity_logits_pool_tokenizer_forms():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    grouped = grouped_polarity_logits(logits, {"no": [0, 1], "yes": [3, 4]})
    assert grouped.shape == (1, 2)
    assert grouped[0, 1] > grouped[0, 0]


def test_decision_objective_reduces_to_sequence_ce_at_zero_weight():
    logits = torch.tensor([[[0.0, 2.0, -1.0], [1.0, 0.0, 2.0]]], requires_grad=True)
    targets = torch.tensor([[1, 2]])
    total, sequence, polarity = decision_aware_objective(
        logits,
        targets,
        "yes",
        {"no": [0], "yes": [1]},
        0.0,
    )
    assert total.item() == pytest.approx(sequence.item())
    assert polarity.item() > 0
    total.backward()
    assert logits.grad is not None


def test_decision_objective_penalizes_wrong_grouped_polarity():
    correct = torch.tensor([[[0.0, 4.0]]])
    wrong = torch.tensor([[[4.0, 0.0]]])
    ids = torch.tensor([[1]])
    _, _, correct_loss = decision_aware_objective(
        correct, ids, "yes", {"no": [0], "yes": [1]}, 1.0
    )
    _, _, wrong_loss = decision_aware_objective(
        wrong, ids, "yes", {"no": [0], "yes": [1]}, 1.0
    )
    assert correct_loss < wrong_loss


def test_class_balance_reports_majority_baseline():
    rows = [{"answer": value} for value in ("yes", "yes", "no")]
    assert class_balance(rows) == {
        "examples": 3,
        "yes": 2,
        "no": 1,
        "majority_label": "yes",
        "majority_accuracy": pytest.approx(2 / 3),
    }


def test_selection_prioritizes_decoded_polarity_then_lower_lambda():
    base = {
        "condition": "routed",
        "polarity_correct_mean": 0.75,
        "margin_correct_mean": 0.75,
        "f1_mean": 0.75,
        "eos_emitted_mean": 1.0,
        "answer_contained_mean": 0.75,
        "gold_polarity_margin_mean": 0.2,
    }
    selected = select_polarity_weight(
        [{**base, "polarity_weight": 1.0}, {**base, "polarity_weight": 0.5}]
    )
    assert selected["polarity_weight"] == 0.5


def test_aggregate_reports_mean_median_and_dispersion():
    rows = [
        {"polarity_weight": 0.5, "condition": "routed", "seed": 11, "example_id": "a", "polarity_correct": 1.0},
        {"polarity_weight": 0.5, "condition": "routed", "seed": 23, "example_id": "b", "polarity_correct": 0.0},
    ]
    result = aggregate(rows, ("polarity_weight", "condition"))[0]
    assert result["polarity_correct_mean"] == 0.5
    assert result["polarity_correct_median"] == 0.5
    assert result["polarity_correct_std"] == pytest.approx(2**-0.5)


def test_seed_aggregation_keeps_seed_dispersion_distinct_from_items():
    rows = [
        {"polarity_weight": 1.0, "condition": "routed", "seed": 11, "polarity_correct_mean": 0.5},
        {"polarity_weight": 1.0, "condition": "routed", "seed": 23, "polarity_correct_mean": 1.0},
    ]
    result = aggregate_seed_means(rows, ("polarity_weight", "condition"))[0]
    assert result["polarity_correct_mean"] == 0.75
    assert result["polarity_correct_seed_std"] == pytest.approx(2**-1.5)
    assert result["polarity_correct_seed_values"] == [0.5, 1.0]


def test_baseline_reproduction_audit_detects_exact_rows(tmp_path):
    row = {
        "example_id": "x",
        "seed": 11,
        "condition": "pra_routed_residual_16_qasper_trained",
        "generated_text": "yes",
        "polarity_correct": 1.0,
        "answer_contained": 1.0,
        "f1": 1.0,
        "em": 1.0,
        "eos_emitted": True,
        "hit_max_new_tokens": False,
        "gold_sequence_logprob": -1.0,
        "gold_mean_token_logprob": -1.0,
        "gold_first_token_rank": 1,
        "gold_first_token_margin": 0.5,
        "gold_polarity_margin": 0.5,
    }
    path = tmp_path / "previous.json"
    path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    current = {**row, "condition": "routed", "polarity_weight": 0.0}
    result = audit_baseline_reproduction([current], path)
    assert result["exact"]
    assert all(value == 0 for value in result["field_mismatches"].values())
