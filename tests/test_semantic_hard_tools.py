"""Frozen benchmark contracts for Paper 6.5 semantic-hard discovery."""

from __future__ import annotations

from collections import Counter

from data.agent_workflows import realistic_tool_catalog
from data.semantic_concepts import canonical_concept_map
from data.semantic_hard_tools import semantic_hardness_queries
from pra_hf.agent_resources import normalize_text


def test_semantic_hard_benchmark_size_balance_and_split_identity() -> None:
    rows = semantic_hardness_queries()

    assert len(rows) == 306
    assert Counter(row.split for row in rows) == {
        "audit": 18,
        "validation": 144,
        "test": 144,
    }
    assert Counter(row.hardness_level for row in rows) == {
        "H0": 54,
        "H1": 36,
        "H2": 36,
        "H3": 36,
        "H4": 36,
        "H5": 108,
    }
    assert len({row.query_id for row in rows}) == len(rows)
    assert len({row.query.casefold() for row in rows}) == len(rows)


def test_every_tool_and_multilingual_cell_has_validation_and_test_rows() -> None:
    rows = semantic_hardness_queries()
    tools = {resource.name for resource in realistic_tool_catalog()}

    for tool in tools:
        tool_rows = [row for row in rows if row.required_tool == tool]
        assert len(tool_rows) == 17
        for level in ("H0", "H1", "H2", "H3", "H4", "H5"):
            assert {row.split for row in tool_rows if row.hardness_level == level} >= {
                "validation",
                "test",
            }
        for language in ("pt", "es", "fr"):
            assert {row.split for row in tool_rows if row.language == language} == {
                "validation",
                "test",
            }


def test_context_is_present_only_for_contextual_hardness() -> None:
    rows = semantic_hardness_queries()
    assert all(bool(row.context) == (row.hardness_level == "H4") for row in rows)


def test_h2_plus_avoids_exact_tool_alias_and_h5_avoids_english_tool_name() -> None:
    resources = {resource.name: resource for resource in realistic_tool_catalog()}
    for row in semantic_hardness_queries():
        if row.hardness_level not in {"H2", "H3", "H4", "H5"}:
            continue
        resource = resources[row.required_tool]
        text = normalize_text(" ".join((row.context, row.query)))
        names = {normalize_text(resource.name), *(normalize_text(alias) for alias in resource.aliases)}
        assert not any(name and f" {name} " in f" {text} " for name in names)


def test_concept_map_recovers_each_declared_multilingual_operation_and_object() -> None:
    concepts = canonical_concept_map()
    for row in semantic_hardness_queries():
        if row.hardness_level != "H5":
            continue
        matched = concepts.concepts(row.query, language=row.language)
        assert row.canonical_operation in matched["operation"], row.query_id
        assert row.canonical_object in matched["object"], row.query_id
