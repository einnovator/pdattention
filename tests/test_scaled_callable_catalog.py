"""Tests for scaled callable catalogs and optimized candidate ordering."""

from __future__ import annotations

import torch

from data.scaled_callable_catalog import generate_scaled_callable_catalog
from experiments.paper6_5_tools.scaled_candidate_sets import candidate_orders
from pra_hf.agent_resources import AgentResource, resource_uri
from pra_hf.tool_records import tool_record_from_callable
from pra_hf.union_discovery import ToolDiscoveryPolicy, discover_candidate_set


def test_scaled_catalog_preserves_targets_and_builds_one_facet_distractors() -> None:
    catalog = generate_scaled_callable_catalog(128, seed=23)

    assert len(catalog) == 128
    assert sum(row.target for row in catalog) == 18
    assert len({row.function.__qualname__ for row in catalog}) == 128
    assert any(row.confusion_axis == "shared_object_destructive" for row in catalog)
    for row in catalog:
        record = tool_record_from_callable(row.function, namespace="scaled", tenant_id="paper6_5")
        assert not record.manual_tags
        assert record.schema.inputs


def test_scaled_catalog_is_seed_reproducible() -> None:
    first = generate_scaled_callable_catalog(64, seed=11)
    second = generate_scaled_callable_catalog(64, seed=11)
    different = generate_scaled_callable_catalog(64, seed=37)

    assert [row.function.__name__ for row in first] == [row.function.__name__ for row in second]
    assert [row.anchor_tool for row in first] == [row.anchor_tool for row in second]
    assert [row.anchor_tool for row in first] != [row.anchor_tool for row in different]


def test_tensorized_candidate_orders_match_runtime_union_semantics() -> None:
    resources = tuple(sorted(
        (
            AgentResource(
                uri=resource_uri("tool", "scaled", f"tool_{index}", "v1"),
                kind="tool",
                namespace="scaled",
                name=f"tool_{index}",
                version="v1",
                description=f"Tool {index}",
                tenant_id="paper6_5",
            )
            for index in range(8)
        ),
        key=lambda row: row.uri,
    ))
    channels = {
        "bm25": torch.tensor([.8, .2, .9, .1, .4, .7, .3, .6]),
        "auto_keyword": torch.tensor([.1, .9, .3, .7, .8, .2, .6, .4]),
        "embedding": torch.tensor([.4, .3, .2, .9, .1, .8, .7, .6]),
    }
    optimized = candidate_orders(channels, max_candidates=6)
    score_maps = {
        name: {resource.uri: float(scores[index]) for index, resource in enumerate(resources)}
        for name, scores in channels.items()
    }
    for strategy in ("fused_score", "raw_union", "diversity_union"):
        runtime = discover_candidate_set(
            "unmatched query",
            resources,
            score_maps,
            ToolDiscoveryPolicy(
                mode="union",
                strategy=strategy,
                max_candidates=6,
                allow_unsafe=True,
                channels=tuple(channels),
            ),
        )
        expected = tuple(resources[index].uri for index in optimized[strategy])
        assert runtime.candidate_uris == expected
