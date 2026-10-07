import numpy as np
import pytest

from experiments.paper4_5_agent import run_mlx_live_agent_kv_lifecycle as probe


def test_max_distribution_delta_reduces_each_coordinate(monkeypatch):
    values = {
        ("a", "A"): (1.0, 0.01, 0.02),
        ("b", "B"): (0.5, 0.03, 0.01),
    }
    monkeypatch.setattr(
        probe,
        "_distribution_delta",
        lambda left, right: values[(left, right)],
    )

    assert probe._max_distribution_delta(["a", "b"], ["A", "B"]) == (
        1.0,
        0.03,
        0.02,
    )


def test_max_distribution_delta_rejects_misaligned_steps():
    assert probe._max_distribution_delta([object()], []) == (
        float("inf"),
        float("inf"),
        float("inf"),
    )


def test_max_distribution_delta_accepts_host_numpy_coordinates(monkeypatch):
    monkeypatch.setattr(
        probe,
        "_distribution_delta",
        lambda _left, _right: pytest.fail("MLX comparator received NumPy"),
    )
    candidate = np.array([1.0, 3.0, 2.0], dtype=np.float16)
    reference = np.array([6.0, 8.0, 7.0], dtype=np.float16)

    centered, probability, variation = probe._max_distribution_delta(
        [candidate], [reference]
    )

    assert centered == pytest.approx(0.0)
    assert probability == pytest.approx(0.0)
    assert variation == pytest.approx(0.0)


def test_qualification_coordinates_keep_cross_consumer_failure_separate():
    checks = {
        "same_subset_token_exact": True,
        "same_subset_logit_within_tolerance": True,
        "dense_engine_oracle_token_exact": True,
        "dense_engine_oracle_logit_within_tolerance": False,
        "zero_unrequested_history_reencoding": True,
    }

    assert probe._qualification_coordinates(checks) == (True, False, False)


def test_qualification_coordinates_require_same_consumer_checks():
    checks = {
        "same_subset_token_exact": False,
        "dense_engine_oracle_token_exact": True,
        "dense_engine_oracle_logit_within_tolerance": True,
    }

    assert probe._qualification_coordinates(checks) == (False, True, False)
