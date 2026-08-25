"""Contracts for model-independent semantic-hard resource discovery."""

from __future__ import annotations

import torch

from data.agent_workflows import realistic_tool_catalog
from pra_hf.agent_disclosure import DiscoveryPolicy
from pra_hf.semantic_resource_discovery import (
    CanonicalConceptMap,
    ConceptExpansion,
    ExternalSemanticIndex,
    ToolSemanticCard,
    token_overlap,
)


def _concepts() -> CanonicalConceptMap:
    return CanonicalConceptMap(
        (
            ConceptExpansion("fix", "update", "operation", "en", "fixture"),
            ConceptExpansion("account", "user", "object", "en", "fixture"),
            ConceptExpansion("avisar", "notify", "operation", "pt", "fixture"),
            ConceptExpansion("utilizador", "user", "object", "pt", "fixture"),
        )
    )


def test_sdk_policy_keeps_external_semantics_independent_from_native_qk() -> None:
    policy = DiscoveryPolicy(
        dictionary_semantics=True,
        embedding_backend="compact-local",
    )

    assert policy.lexical_index
    assert policy.dictionary_semantics
    assert policy.embedding_backend == "compact-local"
    assert not policy.native_qk


def test_concept_map_retains_language_and_source_provenance() -> None:
    concepts = _concepts()
    expanded, matches = concepts.expanded_text("Avisar o utilizador", language="pt")

    assert expanded.endswith("notify user")
    assert {(row.surface, row.source) for row in matches} == {
        ("avisar", "fixture"),
        ("utilizador", "fixture"),
    }


def test_structured_card_preserves_operation_object_schema_and_effect() -> None:
    resource = next(row for row in realistic_tool_catalog() if row.name == "update_user")
    card = ToolSemanticCard.from_resource(resource)

    assert card.operation == "update"
    assert card.objects == ("user",)
    assert {"user_id", "status"} <= set(card.inputs)
    assert card.effect == "write"
    assert len(card.vectors) == 3


def test_external_index_aligns_dictionary_tags_and_multivector_embeddings() -> None:
    resources = realistic_tool_catalog()
    cards = tuple(ToolSemanticCard.from_resource(row) for row in resources)
    vectors = torch.zeros((len(resources), 3, 4))
    target = next(index for index, row in enumerate(resources) if row.name == "update_user")
    vectors[target, :, 0] = 1.0
    index = ExternalSemanticIndex(
        resources,
        _concepts(),
        cards=cards,
        multi_embeddings=vectors,
    )
    rows = index.score(
        "Fix the account",
        language="en",
        query_embedding=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    by_uri = {row.uri: row for row in rows}

    assert by_uri[resources[target].uri].dictionary == 1.0
    assert by_uri[resources[target].uri].embedding == 1.0
    assert index.embedding_bytes == vectors.numel() * vectors.element_size()


def test_token_overlap_uses_query_denominator() -> None:
    resource = next(row for row in realistic_tool_catalog() if row.name == "update_user")
    assert token_overlap("update user now", resource) == 2 / 3
