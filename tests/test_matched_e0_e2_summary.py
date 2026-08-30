import pytest

from experiments.engine_serving.summarize_matched_e0_e2 import _wilson_interval


def test_wilson_interval_does_not_treat_perfect_sample_as_certainty() -> None:
    low, high = _wilson_interval(700, 700)

    assert low == pytest.approx(0.99454, abs=1e-5)
    assert high == pytest.approx(1.0)


def test_wilson_interval_rejects_empty_cohort() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _wilson_interval(0, 0)
