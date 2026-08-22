import pytest

from experiments.pra_efficiency import normalized_metrics, parse_ids


def test_normalized_metrics_deduplicate_selected_identities():
    metrics = normalized_metrics(10, ["a", "b"], ["a", "a", "x"])
    assert metrics["K"] == 2
    assert metrics["R_E"] == pytest.approx(0.5)
    assert metrics["P_E"] == pytest.approx(0.5)
    assert metrics["K_over_N"] == pytest.approx(0.2)
    assert metrics["K_over_E"] == pytest.approx(1.0)
    assert metrics["C_E"] == 0


def test_parse_ids_accepts_artifact_encodings():
    assert parse_ids('["a", "b"]') == ["a", "b"]
    assert parse_ids("a|b") == ["a", "b"]
    assert parse_ids("a b") == ["a", "b"]
