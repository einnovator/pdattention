"""Auditable attention masks for the Paper 3.2 causal decomposition.

The selector and serialized token stream are frozen before this module runs.
It changes only which earlier document tokens may contextualize a document;
query tokens always retain causal access to every selected document.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class DocumentAttentionPolicy(str, Enum):
    """Allowed document-to-document edges in an otherwise causal prompt."""

    FULL_CAUSAL = "FULL_CAUSAL"
    NO_CROSS_DOC = "NO_CROSS_DOC"
    PREVIOUS_DOC_ONLY = "PREVIOUS_DOC_ONLY"
    TOP_RANKED_TO_ALL = "TOP_RANKED_TO_ALL"
    BOUNDARY_ONLY = "BOUNDARY_ONLY"
    QUERY_ONLY_CROSS_DOC = "QUERY_ONLY_CROSS_DOC"


@dataclass(frozen=True)
class TokenBoundary:
    """Half-open token interval for one serialized document or query."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("token boundaries must be ordered and non-negative")

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end


@dataclass(frozen=True)
class AttentionMaskReceipt:
    """Tamper-evident description of one packed causal attention mask."""

    policy: DocumentAttentionPolicy
    document_token_boundaries: tuple[TokenBoundary, ...]
    query_token_boundary: TokenBoundary
    attention_mask_digest: str
    cross_document_attention_edges_allowed: int
    boundary_window_size: int = 0
    top_ranked_document_index: int = 0
    schema_version: str = "paper3.2-document-attention-mask-v1"

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "attention_mask_policy": self.policy.value,
            "attention_mask_digest": self.attention_mask_digest,
            "document_token_boundaries": [
                asdict(boundary) for boundary in self.document_token_boundaries
            ],
            "query_token_boundary": asdict(self.query_token_boundary),
            "cross_document_attention_edges_allowed": (
                self.cross_document_attention_edges_allowed
            ),
            "boundary_window_size": self.boundary_window_size,
            "top_ranked_document_index": self.top_ranked_document_index,
        }
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value


@dataclass(frozen=True)
class CausalDecompositionReceipt:
    """Frozen A/B/C identity contract independent of the execution backend."""

    selection_receipt_id: str
    ordered_record_ids: tuple[str, ...]
    token_sequence_digest: str
    document_order_digest: str
    request_positions_digest: str
    attention_mask_receipt_id: str
    position_binding_mode: str
    rope_frequency_digest: str
    schema_version: str = "paper3.2-prerope-causal-decomposition-v1"

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value = asdict(self)
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value


def token_sequence_digest(token_ids: Sequence[int]) -> str:
    """Hash exact token identities, including record separators."""

    return _digest([int(token_id) for token_id in token_ids])


def request_positions_digest(positions: Sequence[int]) -> str:
    """Hash the effective request position assigned to every source token."""

    return _digest([int(position) for position in positions])


def _boundaries(document_lengths: Sequence[int], query_tokens: int) -> tuple[
    tuple[TokenBoundary, ...], TokenBoundary
]:
    if not document_lengths or any(length <= 0 for length in document_lengths):
        raise ValueError("document lengths must be positive")
    if query_tokens < 0:
        raise ValueError("query token count cannot be negative")
    documents: list[TokenBoundary] = []
    cursor = 0
    for length in document_lengths:
        documents.append(TokenBoundary(cursor, cursor + int(length)))
        cursor += int(length)
    return tuple(documents), TokenBoundary(cursor, cursor + query_tokens)


def build_document_attention_mask(
    document_lengths: Sequence[int],
    *,
    query_tokens: int = 0,
    policy: DocumentAttentionPolicy = DocumentAttentionPolicy.NO_CROSS_DOC,
    boundary_window_size: int = 0,
    top_ranked_document_index: int = 0,
) -> tuple[tuple[tuple[bool, ...], ...], AttentionMaskReceipt]:
    """Build the exact packed-token mask for one causal decomposition arm.

    For ``BOUNDARY_ONLY``, the first ``window`` tokens of a document may read
    the last ``window`` tokens of its immediate predecessor. Later tokens can
    receive that information causally through their own document.
    """

    documents, query = _boundaries(document_lengths, query_tokens)
    if not 0 <= top_ranked_document_index < len(documents):
        raise ValueError("top-ranked document index is outside the document set")
    if policy is DocumentAttentionPolicy.BOUNDARY_ONLY:
        if boundary_window_size <= 0:
            raise ValueError("boundary-only attention requires a positive window")
    elif boundary_window_size != 0:
        raise ValueError("boundary window is valid only for BOUNDARY_ONLY")

    total = query.end
    owner = [-1] * total
    for document_index, boundary in enumerate(documents):
        for position in range(boundary.start, boundary.end):
            owner[position] = document_index

    rows: list[tuple[bool, ...]] = []
    cross_edges = 0
    for row_index in range(total):
        row: list[bool] = []
        row_owner = owner[row_index]
        row_is_query = query.contains(row_index)
        for column_index in range(total):
            allowed = column_index <= row_index
            column_owner = owner[column_index]
            if allowed and not row_is_query and column_owner != row_owner:
                if policy is DocumentAttentionPolicy.FULL_CAUSAL:
                    allowed = True
                elif policy is DocumentAttentionPolicy.PREVIOUS_DOC_ONLY:
                    allowed = column_owner == row_owner - 1
                elif policy is DocumentAttentionPolicy.TOP_RANKED_TO_ALL:
                    allowed = column_owner == top_ranked_document_index
                elif policy is DocumentAttentionPolicy.BOUNDARY_ONLY:
                    current = documents[row_owner]
                    previous = documents[row_owner - 1] if row_owner > 0 else None
                    allowed = bool(
                        previous
                        and column_owner == row_owner - 1
                        and row_index < current.start + boundary_window_size
                        and column_index >= previous.end - boundary_window_size
                    )
                else:
                    allowed = False
            if allowed and row_owner >= 0 and column_owner >= 0:
                cross_edges += int(row_owner != column_owner)
            row.append(bool(allowed))
        rows.append(tuple(row))

    mask = tuple(rows)
    digest = _digest([[int(value) for value in row] for row in mask])
    receipt = AttentionMaskReceipt(
        policy=policy,
        document_token_boundaries=documents,
        query_token_boundary=query,
        attention_mask_digest=digest,
        cross_document_attention_edges_allowed=cross_edges,
        boundary_window_size=boundary_window_size,
        top_ranked_document_index=top_ranked_document_index,
    )
    return mask, receipt


def validate_matched_abc_receipts(
    receipts: Mapping[str, CausalDecompositionReceipt],
) -> None:
    """Fail if A/B/C differ on any frozen identity or coordinate field."""

    required = {"A", "B", "C"}
    if set(receipts) != required:
        raise ValueError("causal decomposition requires exactly A, B, and C receipts")
    matched_fields = (
        "selection_receipt_id",
        "ordered_record_ids",
        "token_sequence_digest",
        "document_order_digest",
        "request_positions_digest",
    )
    reference = receipts["A"]
    for arm, receipt in receipts.items():
        for field in matched_fields:
            if getattr(receipt, field) != getattr(reference, field):
                raise ValueError(f"A/B/C {field} mismatch in arm {arm}")

