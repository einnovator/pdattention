import inspect

import pytest
import torch

from pra_hf.final_metrics import (
    decompose_path_survival,
    exact_join,
    facet_confidence_features,
    fit_linear_selector,
    pareto_flags,
    predict_linear_selector,
    require_disjoint_identifiers,
    selected_facet_group_rank,
)


def test_selected_facet_rank_is_stable_and_target_is_evaluation_only():
    scores = torch.tensor([[0.5, 0.5, 0.1], [0.0, 0.2, 0.9]])
    assert selected_facet_group_rank(scores, 0, (1,)) == 2
    assert selected_facet_group_rank(scores, 1, (1,)) == 2
    with pytest.raises(ValueError):
        selected_facet_group_rank(scores, 2, (1,))


def test_facet_confidence_uses_only_score_distribution():
    features = facet_confidence_features(torch.tensor([[3.0, 1.0], [1.0, 1.0]]))
    assert features["top_parent_margin"].tolist() == [2.0, 0.0]
    assert features["parent_score_entropy"][0] < features["parent_score_entropy"][1]


def test_linear_selector_fits_validation_rows_and_predicts_heldout_rows():
    train_x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = torch.tensor([0, 0, 1, 1])
    model = fit_linear_selector(train_x, train_y, class_count=2, ridge=0.01)
    prediction = predict_linear_selector(model, torch.tensor([[-3.0], [3.0]]))
    assert prediction.tolist() == [0, 1]


def test_validation_and_heldout_identities_must_be_disjoint():
    require_disjoint_identifiers(("validation-a",), ("test-b",))
    with pytest.raises(ValueError):
        require_disjoint_identifiers(("shared",), ("shared",))


def test_runtime_facet_selector_features_have_no_oracle_argument():
    parameters = set(inspect.signature(facet_confidence_features).parameters)
    assert parameters == {"scores"}
    assert not parameters.intersection({"target", "evidence", "oracle", "parent_group"})


def test_path_decomposition_never_calls_redundancy_negative_search_loss():
    losses = decompose_path_survival(0.8, 0.6, 0.7)
    assert losses["missing_local_edge_loss"] == pytest.approx(0.2)
    assert losses["compounded_local_error_loss"] == pytest.approx(0.2)
    assert losses["additional_frontier_search_loss"] == 0.0
    assert losses["correlation_or_redundancy_gain"] == pytest.approx(0.1)


def test_pareto_flags_respect_quality_and_cost_directions():
    rows = [
        {"quality": 0.8, "cost": 0.2},
        {"quality": 0.7, "cost": 0.3},
        {"quality": 0.9, "cost": 0.4},
    ]
    assert pareto_flags(rows, maximize=("quality",), minimize=("cost",)) == [True, False, True]


def test_exact_join_rejects_alignment_errors():
    assert exact_join([{"layer": 0, "a": 1}], [{"layer": 0, "b": 2}], keys=("layer",)) == [
        {"layer": 0, "a": 1, "b": 2}
    ]
    with pytest.raises(ValueError):
        exact_join([{"layer": 0}], [{"layer": 1}], keys=("layer",))
    with pytest.raises(ValueError):
        exact_join(
            [{"layer": 0}, {"layer": 0}],
            [{"layer": 0}],
            keys=("layer",),
        )
