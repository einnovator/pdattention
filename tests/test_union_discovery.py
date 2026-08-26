"""Tests for bounded, provenance-preserving tool candidate unions."""

import pytest

from data.agent_workflows import realistic_tool_catalog
from pra_hf.union_discovery import (
    ToolDiscoveryMode,
    ToolDiscoveryPolicy,
    UnionStrategy,
    agreement_rerank,
    discover_candidate_set,
)


def _scores(resources):
    uris = [row.uri for row in resources]
    return {
        "lexical": {uris[0]: 1.0, uris[1]: 0.9, uris[2]: 0.1},
        "dictionary": {uris[1]: 1.0, uris[3]: 0.8, uris[0]: 0.2},
        "tags": {uris[0]: 0.9, uris[4]: 0.8},
        "embedding": {uris[5]: 1.0, uris[1]: 0.7},
    }


def test_explicit_name_resolution_returns_one_candidate() -> None:
    resources = realistic_tool_catalog()
    result = discover_candidate_set("get_user", resources, _scores(resources))

    assert result.mode == ToolDiscoveryMode.RESOLVE
    assert result.candidate_uris == (resources[1].uri,)
    assert result.explicit_resolution


def test_diversity_union_deduplicates_and_enforces_maximum() -> None:
    resources = realistic_tool_catalog()
    result = discover_candidate_set(
        "help with an account",
        resources,
        _scores(resources),
        ToolDiscoveryPolicy(mode="union", max_candidates=4),
    )

    assert len(result.candidate_uris) == 4
    assert len(set(result.candidate_uris)) == 4
    assert result.candidate_uris[:3] == (resources[0].uri, resources[1].uri, resources[4].uri)


def test_diversity_union_admits_a_candidate_unique_to_embedding() -> None:
    resources = realistic_tool_catalog()
    scores = _scores(resources)
    scores["embedding"] = {resources[6].uri: 1.0, resources[1].uri: 0.7}
    result = discover_candidate_set(
        "semantic request",
        resources,
        scores,
        ToolDiscoveryPolicy(mode="union", strategy="diversity_union", max_candidates=4),
    )

    assert resources[6].uri in result.candidate_uris
    assert result.provenance_for(resources[6].uri).sources[0].channel == "embedding"


def test_custom_automatic_channel_names_are_not_filtered() -> None:
    resources = realistic_tool_catalog()
    scores = {
        "auto_keyword": {resources[0].uri: 1.0},
        "keyword_synonym": {resources[1].uri: 1.0},
        "auto_tag_concept": {resources[2].uri: 1.0},
    }
    result = discover_candidate_set(
        "semantic request",
        resources,
        scores,
        ToolDiscoveryPolicy(
            mode="union",
            strategy="diversity_union",
            max_candidates=3,
            channels=tuple(scores),
        ),
    )

    assert result.candidate_uris == (resources[0].uri, resources[1].uri, resources[2].uri)


def test_candidate_provenance_retains_every_channel_hit() -> None:
    resources = realistic_tool_catalog()
    result = discover_candidate_set(
        "account help",
        resources,
        _scores(resources),
        ToolDiscoveryPolicy(mode="union", max_candidates=6),
    )

    sources = {row.channel for row in result.provenance_for(resources[1].uri).sources}
    assert sources == {"lexical", "dictionary", "embedding"}


def test_fused_and_raw_union_are_distinct_matched_budget_strategies() -> None:
    resources = realistic_tool_catalog()
    fused = discover_candidate_set(
        "account help",
        resources,
        _scores(resources),
        ToolDiscoveryPolicy(mode="union", strategy=UnionStrategy.FUSED_SCORE, max_candidates=3),
    )
    raw = discover_candidate_set(
        "account help",
        resources,
        _scores(resources),
        ToolDiscoveryPolicy(mode="union", strategy=UnionStrategy.RAW_UNION, max_candidates=3),
    )

    assert len(fused.candidate_uris) == len(raw.candidate_uris) == 3
    assert fused.strategy != raw.strategy


def test_destructive_candidates_are_suppressed_unless_enabled() -> None:
    resources = realistic_tool_catalog()
    destructive = next(row for row in resources if row.name == "delete_user")
    scores = {"lexical": {destructive.uri: 10.0, resources[0].uri: 1.0}}

    safe = discover_candidate_set("remove account", resources, scores, ToolDiscoveryPolicy(mode="union"))
    unsafe = discover_candidate_set(
        "remove account",
        resources,
        scores,
        ToolDiscoveryPolicy(mode="union", allow_unsafe=True),
    )

    assert destructive.uri not in safe.candidate_uris
    assert destructive.uri in unsafe.candidate_uris


def test_candidate_budget_validation() -> None:
    try:
        ToolDiscoveryPolicy(max_candidates=2, min_candidates=3)
    except ValueError as error:
        assert "min_candidates" in str(error)
    else:
        raise AssertionError("invalid candidate budgets must fail")


def test_agreement_rerank_preserves_palette_and_rewards_bounded_support() -> None:
    resources = realistic_tool_catalog()
    uris = [row.uri for row in resources[:3]]
    scores = {
        "lexical": {uris[0]: 1.0, uris[1]: 0.9, uris[2]: 0.1},
        "semantic": {uris[1]: 1.0, uris[2]: 0.9, uris[0]: 0.1},
    }

    reranked = agreement_rerank(
        scores,
        candidate_uris=(uris[0], uris[1], uris[2]),
        support_depth=2,
        agreement_weight=1.0,
    )

    assert reranked[0] == uris[1]
    assert set(reranked) == set(uris)


def test_agreement_rerank_rejects_invalid_control_values() -> None:
    with pytest.raises(ValueError):
        agreement_rerank({}, candidate_uris=("x",), support_depth=0, agreement_weight=0.1)
    with pytest.raises(ValueError):
        agreement_rerank({}, candidate_uris=("x",), support_depth=1, agreement_weight=-0.1)
