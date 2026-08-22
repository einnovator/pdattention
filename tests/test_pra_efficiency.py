import pytest

from experiments.pra_efficiency import normalized_metrics, parse_ids


def test_metrics_use_unique_selected_chunks():
    metrics = normalized_metrics(8, ["a", "b"], ["a", "a", "x"])
    assert metrics["K"] == 2
    assert metrics["R_E"] == pytest.approx(.5)
    assert metrics["P_E"] == pytest.approx(.5)
    assert metrics["K_over_N"] == pytest.approx(.25)


def test_parse_identity_formats():
    assert parse_ids('["a", "b"]') == ["a", "b"]
    assert parse_ids("a|b") == ["a", "b"]
