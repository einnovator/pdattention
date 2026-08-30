from experiments.paper6_2_mlx.run_tier_window_pressure import _oldest_other


def test_oldest_other_preserves_active_request() -> None:
    lru = ["a", "b", "c"]

    assert _oldest_other(lru, "a") == "b"
    assert lru == ["a", "c"]


def test_oldest_other_returns_none_for_only_active_request() -> None:
    assert _oldest_other(["a"], "a") is None
