"""Offline correctness checks for the Paper 2 oracle-gap audit."""

from types import SimpleNamespace

import pytest
import torch

from experiments.paper2_hf.qa.run_oracle_gap_audit import (
    align_evidence_token_ids,
    audit_materialized_selection,
    build_evidence_span_audit,
    counterfactual_softmax_diagnostic,
    materialized_source_positions,
    parent_context,
    token_span_from_offsets,
)


class _CharacterTokenizer:
    """Minimal offset-producing tokenizer for exact boundary tests."""

    def __call__(self, text, **_kwargs):
        return SimpleNamespace(
            input_ids=[ord(character) for character in text],
            offset_mapping=[(index, index + 1) for index in range(len(text))],
        )


def _hit(start, end, *, chunk_id, uri="doc://one"):
    return SimpleNamespace(
        reference_uri=uri,
        chunk_id=chunk_id,
        token_start=start,
        token_end=end,
        logical_start=start,
        logical_end=end,
        selected_token_count=end - start,
    )


def test_token_span_boundaries_are_half_open_and_include_partial_tokens():
    offsets = [(0, 2), (2, 5), (6, 9)]

    assert token_span_from_offsets(offsets, 1, 4) == (0, 2)
    assert token_span_from_offsets(offsets, 5, 6) is None
    assert token_span_from_offsets(offsets, 6, 9) == (2, 3)


def test_span_audit_reports_all_occurrences_but_selects_canonical_first_match():
    example = {
        "source": "alpha beta gamma beta",
        "evidence": ["beta"],
        "evidence_annotations": [{"title": "T", "sentence_id": 2}],
        "answer": "must not affect selection",
    }

    audit = build_evidence_span_audit(_CharacterTokenizer(), example)

    assert audit["all_representable"]
    assert audit["annotations"][0]["char_occurrences"] == [[6, 10], [17, 21]]
    assert audit["annotations"][0]["selected_char_span"] == [6, 10]
    assert audit["annotations"][0]["token_span"] == [6, 10]


def test_materialization_deduplicates_overlap_and_covers_every_evidence_token():
    selected = [_hit(0, 4, chunk_id="a"), _hit(2, 6, chunk_id="b")]

    positions, duplicates = materialized_source_positions(selected)
    audit = audit_materialized_selection(selected, [(1, 3), (4, 6)])

    assert positions == [0, 1, 2, 3, 4, 5]
    assert duplicates == 2
    assert audit["all_evidence_covered"]
    assert audit["deduplicated_overlap_tokens"] == 2
    assert audit["extra_non_evidence_tokens"] == 2


def test_missing_evidence_token_is_detected_after_materialization():
    audit = audit_materialized_selection([_hit(0, 4, chunk_id="a")], [(3, 5)])

    assert audit["evidence_span_covered"] == [False]
    assert not audit["all_evidence_covered"]


def test_counterfactual_softmax_removes_only_selected_parent_distractors():
    weights = torch.tensor([[[[0.2, 0.3, 0.1, 0.4]]]])

    audit = counterfactual_softmax_diagnostic(
        weights,
        query_positions=[0],
        evidence_keys=[0],
        distractor_keys=[1],
        local_keys=[2, 3],
    )

    assert audit["evidence_mass"] == pytest.approx(0.2)
    assert audit["distractor_mass"] == pytest.approx(0.3)
    assert audit["local_mass"] == pytest.approx(0.5)
    assert audit["counterfactual_evidence_mass"] == pytest.approx(2 / 7)
    assert audit["counterfactual_entropy"] < audit["attention_entropy"]


def test_direct_reference_alignment_uses_token_identity_not_answer_text():
    source_ids = [10, 20, 21, 30, 40, 41]
    prompt_ids = [1, 20, 21, 2, 40, 41, 3]

    alignment = align_evidence_token_ids(
        source_ids,
        [(1, 3), (4, 6)],
        prompt_ids,
        context_positions=[1, 2, 4, 5],
    )

    assert alignment["pairs"] == [[1, 1], [2, 2], [4, 4], [5, 5]]
    assert alignment["alignment_fraction"] == 1.0


def test_parent_context_is_annotation_derived_and_deduplicated():
    example = {
        "answer": "unused gold answer",
        "evidence": ["evidence"],
        "parent_paragraphs": ["parent one", "parent one", "parent two"],
    }

    assert parent_context(example) == "parent one\nparent two"
