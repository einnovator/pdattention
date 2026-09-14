import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.multi_issue_frontier import (
    load_strategy_registry,
    reduce_multi_issue_runs,
)


REGISTRY = Path(
    "experiments/paper8_5_agent_memory/configs/multi_issue_strategy_registry_v1.json"
)
PROMOTION_CONTRACT = Path(
    "experiments/paper8_5_agent_memory/configs/"
    "paper4_5_profile_promotion_contract_v1.json"
)


def _run(strategy: str, mode: str, tokens: tuple[int, int], resolved=(True, True)):
    strategy_config = {}
    strategy_config_digest = hashlib.sha256(
        json.dumps(strategy_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "pair_id": "pair-1",
        "sequence_family_id": "sequence-family-1",
        "sequence_id": "sequence-1",
        "sequence_digest": "sequence-digest",
        "sequence_stratum": "related_same_repository",
        "agent_id": "mini-swe-agent",
        "agent_revision": "agent-r1",
        "ordered_instance_ids": ["repo__1", "repo__2"],
        "issue_count": 2,
        "session_mode": mode,
        "strategy_id": strategy,
        "strategy_config_id": "default",
        "strategy_config": strategy_config,
        "strategy_config_digest": strategy_config_digest,
        "model_revision": "model-r1",
        "tokenizer_revision": "tokenizer-r1",
        "harness_revision": "harness-r1",
        "decoding_digest": "decode-r1",
        "workspace_schedule_digest": "workspace-r1",
        "issues": [
            {
                "issue_index": index,
                "resolved": outcome,
                "selected_input_tokens": token_count,
                "calls": 10,
                "rediscovery_calls": index - 1,
                "first_action_diverged": False,
                "selected_tokens_before_divergence_or_terminal": token_count,
                "full_tokens_before_divergence_or_terminal": 100,
            }
            for index, (token_count, outcome) in enumerate(zip(tokens, resolved), 1)
        ],
    }


def test_registry_has_unique_reasoned_strategies_and_locked_issue_axis():
    registry = load_strategy_registry(REGISTRY)
    assert registry["issue_counts"] == [1, 2, 3, 4, 5]
    assert len(registry["strategies"]) == 10


def test_profile_promotion_contract_separates_quality_and_runtime_evidence():
    contract = json.loads(PROMOTION_CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    assert [row["id"] for row in contract["profile_candidates"]] == [
        "AGENT_FULL", "AGENT_QUALITY", "AGENT_BALANCED", "AGENT_ECONOMY"
    ]
    assert "sparse-KV identity" in contract["evidence_levels"]["runtime_qualified"]
    assert any("logical token" in rule for rule in contract["prohibitions"])


def test_reducer_compares_every_candidate_to_both_full_controls():
    registry = load_strategy_registry(REGISTRY)
    fresh = _run("S00_fresh_full", "fresh_per_issue", (50, 50))
    persistent = _run("S01_persistent_full", "persistent", (100, 100))
    candidate = _run("S03_completed_episode_spine", "persistent", (60, 60))
    result = reduce_multi_issue_runs((fresh, persistent, candidate), registry)
    row = next(row for row in result["rows"] if row["strategy_id"].startswith("S03"))
    assert row["saving_vs_persistent_full"] == pytest.approx(0.4)
    assert row["saving_vs_fresh_full"] == pytest.approx(-0.2)
    assert row["official_resolution"] == 1.0
    assert row["failure_aware_saving_vs_persistent_full"] == pytest.approx(0.4)
    assert row["in_primary_saving_target"] is True
    assert row["discovery_primary_target_met"] is True
    assert row["confirmation_primary_target_met"] is None
    assert row["successful_calls_delta_vs_persistent_full"] == 0
    assert row["first_action_divergence_rate"] == 0.0


def test_failure_aware_saving_cannot_reward_a_quality_loss():
    registry = load_strategy_registry(REGISTRY)
    fresh = _run("S00_fresh_full", "fresh_per_issue", (50, 50))
    persistent = _run("S01_persistent_full", "persistent", (100, 100))
    failed = _run(
        "S02_active_episode_only", "persistent", (30, 30), resolved=(True, False)
    )
    result = reduce_multi_issue_runs((fresh, persistent, failed), registry)
    row = next(row for row in result["rows"] if row["strategy_id"].startswith("S02"))
    assert row["saving_vs_persistent_full"] == pytest.approx(0.7)
    assert row["failure_aware_saving_vs_persistent_full"] == 0.0
    assert row["discovery_primary_target_met"] is False
    assert row["failure_aware_saving_vs_fresh_full"] == 0.0
    assert row["cost_per_resolved_issue"] == 60


def test_pairing_identity_mismatch_and_duplicate_config_fail_closed():
    registry = load_strategy_registry(REGISTRY)
    fresh = _run("S00_fresh_full", "fresh_per_issue", (50, 50))
    persistent = _run("S01_persistent_full", "persistent", (100, 100))
    candidate = _run("S03_completed_episode_spine", "persistent", (60, 60))
    mismatched = copy.deepcopy(candidate)
    mismatched["model_revision"] = "other"
    with pytest.raises(ValueError, match="strict pairing identity mismatch"):
        reduce_multi_issue_runs((fresh, persistent, mismatched), registry)

    duplicate = copy.deepcopy(candidate)
    with pytest.raises(ValueError, match="duplicate strategy/config"):
        reduce_multi_issue_runs((fresh, persistent, candidate, duplicate), registry)


def test_config_digest_and_pre_divergence_accounting_fail_closed():
    registry = load_strategy_registry(REGISTRY)
    fresh = _run("S00_fresh_full", "fresh_per_issue", (50, 50))
    persistent = _run("S01_persistent_full", "persistent", (100, 100))
    candidate = _run("S03_completed_episode_spine", "persistent", (60, 60))
    candidate["issues"][1]["first_action_diverged"] = True
    candidate["issues"][1]["selected_tokens_before_divergence_or_terminal"] = 20
    candidate["issues"][1]["full_tokens_before_divergence_or_terminal"] = 40
    result = reduce_multi_issue_runs((fresh, persistent, candidate), registry)
    row = next(row for row in result["rows"] if row["strategy_id"].startswith("S03"))
    assert row["first_action_divergence_rate"] == 0.5
    assert row["first_divergence_preceding_saving"] == pytest.approx(1 - 80 / 140)

    bad = copy.deepcopy(candidate)
    bad["strategy_config"] = {"tail_turns": 4}
    with pytest.raises(ValueError, match="strategy_config_digest"):
        reduce_multi_issue_runs((fresh, persistent, bad), registry)
