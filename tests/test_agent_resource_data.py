"""Tests for opaque Paper 6.5 catalog generation."""

from data.agent_resources import (
    generate_agent_catalog,
    replace_versions,
    synthetic_semantic_vector,
)


def test_catalog_generation_is_deterministic_and_opaque():
    first = generate_agent_catalog(32, seed=11)
    second = generate_agent_catalog(32, seed=11)
    assert first == second
    assert len(first.resources) == 32
    assert all(resource.name.startswith("tool_") for resource in first.resources)
    assert all(resource.name not in resource.description for resource in first.resources)


def test_query_splits_are_identity_disjoint_and_cover_required_strata():
    catalog = generate_agent_catalog(64, seed=23)
    validation_targets = {
        query.target_uris[0]
        for query in catalog.split("validation")
        if len(query.target_uris) == 1
    }
    test_targets = {
        query.target_uris[0]
        for query in catalog.split("test")
        if len(query.target_uris) == 1
    }
    strata = {query.stratum for query in catalog.queries}
    assert validation_targets.isdisjoint(test_targets)
    assert {
        "explicit_uri",
        "exact_name",
        "alias",
        "typo",
        "semantic_paraphrase",
        "description",
        "ambiguous",
        "nonexistent",
    } <= strata


def test_mutation_changes_version_uri_and_source_without_touching_other_rows():
    catalog = generate_agent_catalog(8, seed=37)
    mutated = replace_versions(catalog.resources, (2,))
    assert mutated[0] == catalog.resources[0]
    assert mutated[2].version == "v2"
    assert mutated[2].uri != catalog.resources[2].uri
    assert "updated schema" in mutated[2].description


def test_synthetic_semantic_encoder_maps_paraphrase_to_catalog_concept():
    canonical = synthetic_semantic_vector("archive invoice in service family 2")
    paraphrase = synthetic_semantic_vector(
        "place into long term storage billing document in service family 2"
    )
    dot = sum(left * right for left, right in zip(canonical, paraphrase))
    assert dot > 0.99
