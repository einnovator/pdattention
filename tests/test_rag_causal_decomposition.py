from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.rag_causal_decomposition import (
    CausalDecompositionReceipt,
    DocumentAttentionPolicy,
    build_document_attention_mask,
    request_positions_digest,
    token_sequence_digest,
    validate_matched_abc_receipts,
)


def test_no_cross_document_mask_isolates_documents_and_exposes_all_to_query() -> None:
    mask, receipt = build_document_attention_mask(
        (3, 2), query_tokens=2, policy=DocumentAttentionPolicy.NO_CROSS_DOC
    )
    assert mask[1] == (True, True, False, False, False, False, False)
    assert mask[3] == (False, False, False, True, False, False, False)
    assert mask[5] == (True, True, True, True, True, True, False)
    assert mask[6] == (True, True, True, True, True, True, True)
    assert receipt.cross_document_attention_edges_allowed == 0
    assert receipt.query_token_boundary.start == 5


def test_full_and_sparse_mask_ladder_has_expected_document_edges() -> None:
    full, full_receipt = build_document_attention_mask(
        (4, 4, 4), policy=DocumentAttentionPolicy.FULL_CAUSAL
    )
    previous, previous_receipt = build_document_attention_mask(
        (4, 4, 4), policy=DocumentAttentionPolicy.PREVIOUS_DOC_ONLY
    )
    top, top_receipt = build_document_attention_mask(
        (4, 4, 4), policy=DocumentAttentionPolicy.TOP_RANKED_TO_ALL
    )
    boundary, boundary_receipt = build_document_attention_mask(
        (4, 4, 4),
        policy=DocumentAttentionPolicy.BOUNDARY_ONLY,
        boundary_window_size=2,
    )
    assert full[10][0]
    assert not previous[10][0] and previous[10][4]
    assert top[10][0] and not top[10][4]
    assert boundary[8][6] and not boundary[8][5]
    assert not boundary[10][6]
    assert (
        full_receipt.cross_document_attention_edges_allowed
        > previous_receipt.cross_document_attention_edges_allowed
        > boundary_receipt.cross_document_attention_edges_allowed
        > 0
    )
    assert top_receipt.cross_document_attention_edges_allowed > 0


def test_mask_receipt_is_stable_and_policy_sensitive() -> None:
    _, first = build_document_attention_mask(
        (3, 2), query_tokens=1, policy=DocumentAttentionPolicy.NO_CROSS_DOC
    )
    _, second = build_document_attention_mask(
        (3, 2), query_tokens=1, policy=DocumentAttentionPolicy.NO_CROSS_DOC
    )
    _, changed = build_document_attention_mask(
        (3, 2), query_tokens=1, policy=DocumentAttentionPolicy.FULL_CAUSAL
    )
    assert first.receipt_id == second.receipt_id
    assert first.attention_mask_digest == second.attention_mask_digest
    assert first.receipt_id != changed.receipt_id


def test_boundary_mask_requires_an_explicit_positive_window() -> None:
    with pytest.raises(ValueError, match="positive window"):
        build_document_attention_mask(
            (2, 2), policy=DocumentAttentionPolicy.BOUNDARY_ONLY
        )
    with pytest.raises(ValueError, match="valid only"):
        build_document_attention_mask(
            (2, 2),
            policy=DocumentAttentionPolicy.NO_CROSS_DOC,
            boundary_window_size=1,
        )


def _abc_receipt() -> CausalDecompositionReceipt:
    return CausalDecompositionReceipt(
        selection_receipt_id="selection-1",
        ordered_record_ids=("D1", "D2"),
        token_sequence_digest=token_sequence_digest((1, 2, 3, 4)),
        document_order_digest=token_sequence_digest((101, 202)),
        request_positions_digest=request_positions_digest((0, 1, 2, 3)),
        attention_mask_receipt_id="mask-b",
        position_binding_mode="PRE_ROPE",
        rope_frequency_digest="f" * 64,
    )


def test_abc_receipt_rejects_selection_token_order_and_position_mismatch() -> None:
    receipt = _abc_receipt()
    validate_matched_abc_receipts(
        {"A": receipt, "B": replace(receipt, attention_mask_receipt_id="mask-b"), "C": receipt}
    )
    for field, value in (
        ("selection_receipt_id", "selection-2"),
        ("ordered_record_ids", ("D2", "D1")),
        ("token_sequence_digest", "0" * 64),
        ("request_positions_digest", "1" * 64),
    ):
        with pytest.raises(ValueError, match="mismatch"):
            validate_matched_abc_receipts(
                {"A": receipt, "B": receipt, "C": replace(receipt, **{field: value})}
            )

