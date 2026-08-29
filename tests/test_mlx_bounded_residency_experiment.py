from __future__ import annotations

import pytest

from experiments.paper6_2_mlx.run_bounded_residency import _access_sequence


def test_access_sequence_preserves_legacy_single_round_shape() -> None:
    assert _access_sequence(4, 1) == (0, 1, 2, 3, 0)


def test_access_sequence_repeats_complete_session_before_final_revisit() -> None:
    assert _access_sequence(3, 2) == (0, 1, 2, 0, 1, 2, 0)


@pytest.mark.parametrize("resources,rounds", [(2, 1), (3, 0)])
def test_access_sequence_rejects_invalid_pressure_geometry(
    resources: int, rounds: int
) -> None:
    with pytest.raises(ValueError):
        _access_sequence(resources, rounds)
