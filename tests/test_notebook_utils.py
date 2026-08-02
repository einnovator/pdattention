from pathlib import Path

from nb.pra_notebook_utils import experiment_policy, find_repo_root


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
