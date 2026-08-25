"""Tests for isolated zero-configuration tool discovery evidence."""

from __future__ import annotations

from data.python_tool_ingestion_cases import update_user
from data.semantic_concepts import canonical_concept_map
from pra_hf.auto_tool_discovery import (
    AutoEvidenceSource,
    auto_tag_score,
    automatic_semantic_view,
    evidence_provenance_counts,
    inferred_concept_score,
    weighted_keyword_score,
)
from pra_hf.tool_records import tool_record_from_callable


def _view():
    concepts = canonical_concept_map()
    record = tool_record_from_callable(update_user, concept_map=concepts)
    return concepts, record, automatic_semantic_view(record, concepts=concepts)


def test_keyword_only_policy_excludes_dictionary_and_manual_metadata() -> None:
    concepts, record, view = _view()
    sources = {
        AutoEvidenceSource.FUNCTION_NAME,
        AutoEvidenceSource.DOCSTRING,
        AutoEvidenceSource.PARAMETER_NAME,
        AutoEvidenceSource.PARAMETER_DESCRIPTION,
        AutoEvidenceSource.RETURN_DESCRIPTION,
        AutoEvidenceSource.TYPE_SCHEMA,
        AutoEvidenceSource.MODULE_NAMESPACE,
    }

    assert weighted_keyword_score("update user", view, sources=sources) > 0
    assert all(row.source != AutoEvidenceSource.DICTIONARY_EXPANSION for row in view.evidence if row.source in sources)
    assert not record.manual_tags


def test_synonym_mode_is_explicit_and_improves_canonical_query_coverage() -> None:
    concepts, _, view = _view()
    keyword_sources = {AutoEvidenceSource.FUNCTION_NAME, AutoEvidenceSource.DOCSTRING}
    synonym_sources = {*keyword_sources, AutoEvidenceSource.DICTIONARY_EXPANSION}

    direct = weighted_keyword_score("fix the account", view, sources=keyword_sources)
    expanded = weighted_keyword_score(
        "fix the account", view, sources=synonym_sources, concepts=concepts, expand_query=True
    )

    assert expanded > direct


def test_operation_object_and_auto_tags_are_inferred_without_manual_tags() -> None:
    concepts, record, view = _view()

    assert not record.manual_tags
    assert {"update", "user"} <= (set(view.operations) | set(view.objects))
    assert inferred_concept_score("modify the account", view, concepts) > 0
    assert auto_tag_score("modify the account", view, concepts) > 0


def test_type_schema_terms_can_be_toggled_independently() -> None:
    _, _, view = _view()
    without_types = weighted_keyword_score(
        "changed result", view, sources={AutoEvidenceSource.FUNCTION_NAME, AutoEvidenceSource.DOCSTRING}
    )
    with_types = weighted_keyword_score(
        "changed result",
        view,
        sources={AutoEvidenceSource.FUNCTION_NAME, AutoEvidenceSource.DOCSTRING, AutoEvidenceSource.TYPE_SCHEMA},
    )

    assert with_types > without_types


def test_provenance_manifest_covers_callable_and_embedding_fields() -> None:
    _, record, view = _view()
    counts = evidence_provenance_counts((view,))

    assert counts[AutoEvidenceSource.FUNCTION_NAME.value] > 0
    assert counts[AutoEvidenceSource.PARAMETER_NAME.value] > 0
    assert counts[AutoEvidenceSource.EMBEDDING_FIELD.value] > 0
    assert record.field_provenance["embedding_fields"]


def test_absent_docstring_remains_empty_and_camel_name_is_tokenized() -> None:
    def updateUserStatus(user_id: str) -> bool:
        return True

    record = tool_record_from_callable(updateUserStatus)
    view = automatic_semantic_view(record)
    name_terms = view.weighted_terms(sources={AutoEvidenceSource.FUNCTION_NAME})

    assert record.description == ""
    assert {"update", "user", "status"} <= set(name_terms)
