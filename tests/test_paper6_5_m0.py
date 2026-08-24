"""Scientific-contract tests for the deterministic Paper 6.5 study."""

from __future__ import annotations

from data.agent_resources import generate_agent_catalog, synthetic_semantic_vector
from experiments.paper6_5_tools.run_m0_policy_study import (
    FIXED_POLICIES,
    POLICIES,
    _add_oracle_fields,
    _hint,
    _semantic_encoder,
)
from experiments.paper6_5_tools.summarize_m0_policy_study import (
    _mean_ci,
    summarize_policy,
)
from pra_hf.agent_resources import PersistentResourceIndex


def test_required_fixed_adaptive_and_hint_policies_are_present():
    assert FIXED_POLICIES == (
        "fixed_explicit",
        "fixed_token",
        "fixed_index",
        "fixed_semantic",
        "fixed_hybrid",
    )
    assert {"auto", "user_hint", "adaptive"} <= set(POLICIES)
    assert _hint("fixed_token", "semantic_paraphrase").strict
    assert _hint("user_hint", "semantic_paraphrase").mode.value == "semantic"


def test_study_semantic_encoder_is_the_declared_compact_control():
    assert len(_semantic_encoder("archive invoice")) == 96
    assert _semantic_encoder("archive invoice") == synthetic_semantic_vector(
        "archive invoice", dimensions=96
    )


def test_index_only_scoring_uses_postings_candidate_narrowing():
    catalog = generate_agent_catalog(128, seed=11)
    index = PersistentResourceIndex(catalog.resources, semantic_encoder=_semantic_encoder)
    query = next(value for value in catalog.queries if value.stratum == "exact_name")
    from pra_hf.agent_resources import DiscoveryRequest

    rows = index.score(
        DiscoveryRequest(query=query.query, tenant_id="paper6_5", namespace="synthetic"),
        channels=("index",),
    )
    assert 0 < len(rows) < len(catalog.resources)


def test_oracle_policy_is_validation_label_free_and_prefers_low_cost_correct_policy():
    rows = [
        {
            "catalog_size": 8,
            "seed": 11,
            "query_id": "q",
            "policy": policy,
            "top1_correct": int(policy in {"fixed_token", "fixed_hybrid"}),
        }
        for policy in POLICIES
    ]
    _add_oracle_fields(rows)
    assert {row["oracle_policy"] for row in rows} == {"fixed_token"}
    token = next(row for row in rows if row["policy"] == "fixed_token")
    hybrid = next(row for row in rows if row["policy"] == "fixed_hybrid")
    assert token["quality_regret"] == 0
    assert hybrid["cost_regret"] > 0


def test_seed_bootstrap_is_deterministic_and_keeps_center():
    first = _mean_ci([0.6, 0.7, 0.8, 0.9, 1.0], seed=11, draws=500)
    second = _mean_ci([0.6, 0.7, 0.8, 0.9, 1.0], seed=11, draws=500)
    assert first == second
    assert first[0] == 0.8
    assert first[1] <= first[0] <= first[2]
