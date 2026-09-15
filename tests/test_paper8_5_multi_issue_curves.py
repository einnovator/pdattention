from pathlib import Path

from experiments.paper8_5_agent_memory.multi_issue_curves import (
    _compact_identity_label,
    aggregate_frontier,
    render_frontier_plots,
)


def _row(pair: str, family: str, strategy: str, saving: float, quality: float):
    return {
        "pair_id": pair,
        "sequence_family_id": family,
        "sequence_stratum": "related_same_repository",
        "agent_id": "mini-swe-agent",
        "agent_revision": "agent-r1",
        "model_revision": "model-r1",
        "tokenizer_revision": "tokenizer-r1",
        "harness_revision": "harness-r1",
        "issue_count": 2,
        "session_mode": "persistent",
        "strategy_id": strategy,
        "strategy_config_id": "default",
        "strategy_config_digest": "config-digest",
        "strategy_config": {},
        "official_resolution": quality,
        "failure_aware_saving_vs_persistent_full": saving,
        "failure_aware_saving_vs_fresh_full": saving - 0.1,
        "calls_delta_vs_persistent_full": 0,
        "calls_delta_vs_fresh_full": 1,
        "successful_calls_delta_vs_persistent_full": 0,
        "successful_calls_delta_vs_fresh_full": 1,
        "rediscovery_delta_vs_fresh_full": 0,
        "cost_per_resolved_issue": 100,
        "first_action_divergence_rate": 0,
        "first_divergence_preceding_saving": saving / 2,
        "selected_input_tokens": 200,
    }


def test_aggregation_clusters_repeats_by_ordered_sequence_family():
    reduction = {
        "schema_version": 1,
        "rows": [
            _row("p1", "family-a", "S02_active_episode_only", 0.2, 1.0),
            _row("p2", "family-a", "S02_active_episode_only", 0.4, 0.0),
            _row("p3", "family-b", "S02_active_episode_only", 0.6, 1.0),
        ],
    }
    summary = aggregate_frontier(reduction, seed=1, bootstrap_samples=100)
    cell = summary["cells"][0]
    assert cell["execution_rows"] == 3
    assert cell["sequence_clusters"] == 2
    # family-a contributes its mean 0.3 once; family-b contributes 0.6 once.
    assert cell["metrics"]["failure_aware_saving_vs_persistent_full"]["mean"] == 0.45
    assert cell["metrics"]["official_resolution"]["mean"] == 0.75


def test_plotter_emits_both_baseline_frontiers(tmp_path: Path):
    reduction = {
        "schema_version": 1,
        "rows": [_row("p1", "family-a", "S02_active_episode_only", 0.2, 1.0)],
    }
    summary = aggregate_frontier(reduction, seed=1, bootstrap_samples=10)
    manifest = render_frontier_plots(summary, tmp_path)
    assert manifest["plot_count"] == 5
    assert (tmp_path / "related_same_repository" / "quality_vs_persistent.png").is_file()
    assert (tmp_path / "related_same_repository" / "quality_vs_fresh.png").is_file()


def test_aggregation_never_pools_sequence_strata():
    first = _row("p1", "family-a", "S02_active_episode_only", 0.2, 1.0)
    second = _row("p2", "family-b", "S02_active_episode_only", 0.4, 1.0)
    second["sequence_stratum"] = "dependent_same_workspace"
    summary = aggregate_frontier({"schema_version": 1, "rows": [first, second]})
    assert len(summary["cells"]) == 2


def test_plot_labels_preserve_compact_model_and_policy_identity():
    prefix = ("mini-swe-agent", "agent-r1", "06c1097efce0431c", "tok", "harness")
    assert _compact_identity_label(
        (*prefix, "S01_persistent_full", "default")
    ) == "mini-swe-agent / 06c1097e / FULL"
    assert _compact_identity_label(
        (*prefix, "S03_completed_episode_spine", "r4m2v2_tasks1")
    ) == "mini-swe-agent / 06c1097e / R4/M2/V2"
