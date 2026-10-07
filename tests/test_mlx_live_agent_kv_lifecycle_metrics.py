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
