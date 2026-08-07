from pathlib import Path

import pytest

from nb.pra_notebook_utils import (
    experiment_policy,
    find_repo_root,
    validate_paired_model_seeds,
)


def test_find_repo_root_from_notebook_directory():
    repo = find_repo_root(Path(__file__).parents[1] / "nb")

    assert (repo / "src").is_dir()
    assert (repo / "data").is_dir()


def test_experiment_policy_scales_capacity_and_update_budget():
    sizes = [3, 30, 300, 3_000, 30_000]
    policies = [experiment_policy(size) for size in sizes]

    assert [policy.d_model for policy in policies] == [32, 32, 64, 96, 128]
    assert [policy.n_layers for policy in policies] == [2, 2, 3, 4, 4]
    assert [policy.estimated_optimizer_steps for policy in policies] == [20, 120, 180, 375, 2_250]


def test_validate_paired_model_seeds_requires_five_shared_seeds():
    seeds = [1, 7, 21, 42, 87]
    rows = [
        {"model": model, "split_count": split_count, "seed": seed}
        for model in ("td_sa_tiny", "td_pra_tiny")
        for split_count in (2, 5)
        for seed in seeds
    ]

    assert validate_paired_model_seeds(rows) == seeds

    with pytest.raises(ValueError, match="at least 5 model seeds"):
        validate_paired_model_seeds(rows[:4])


def test_validate_paired_model_seeds_rejects_unpaired_groups():
    rows = [
        {"model": model, "split_count": 2, "seed": seed}
        for model, seeds in (
            ("td_sa_tiny", [1, 7, 21, 42, 87]),
            ("td_pra_tiny", [1, 7, 21, 42, 99]),
        )
        for seed in seeds
    ]

    with pytest.raises(ValueError, match="same paired model seeds"):
        validate_paired_model_seeds(rows)
