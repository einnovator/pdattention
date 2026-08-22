import pytest

from experiments.pra_efficiency import normalized_metrics


def test_normalized_metrics_separate_candidate_and_evidence_denominators():
    metrics = normalized_metrics(20, [1, 2], [1, 1, 8, 9])
    assert metrics["K"] == 3
    assert metrics["R_E"] == pytest.approx(.5)
    assert metrics["P_E"] == pytest.approx(1 / 3)
    assert metrics["K_over_N"] == pytest.approx(.15)
    assert metrics["K_over_E"] == pytest.approx(1.5)
