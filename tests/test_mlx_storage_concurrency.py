import pytest

from experiments.paper6_2_mlx.run_live_storage_concurrency import _percentile


def test_nearest_rank_percentiles_are_stable_for_small_concurrency_waves() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert _percentile(values, 0.50) == 3.0
    assert _percentile(values, 0.95) == 5.0
    assert _percentile(values, 0.99) == 5.0


def test_percentile_rejects_empty_wave() -> None:
    with pytest.raises(ValueError, match="empty"):
        _percentile([], 0.50)
