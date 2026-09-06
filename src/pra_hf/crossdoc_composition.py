"""Request-local cross-document gist composition contracts and reference math."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Sequence

import torch


class CrossDocumentCompositionMode(str, Enum):
    """Bounded request-local interaction applied after persistent K/V selection."""

    INDEPENDENT_PRA = "INDEPENDENT_PRA"
    GIST_SA_APPEND = "GIST_SA_APPEND"
    GIST_SA_BOUNDARY_8 = "GIST_SA_BOUNDARY_8"
    GIST_SA_BOUNDARY_32 = "GIST_SA_BOUNDARY_32"

    @property
    def boundary_tokens(self) -> int:
        return {
            self.INDEPENDENT_PRA: 0,
            self.GIST_SA_APPEND: 0,
            self.GIST_SA_BOUNDARY_8: 8,
            self.GIST_SA_BOUNDARY_32: 32,
        }[self]


class GistAttentionMask(str, Enum):
    """Which selected-record gists may exchange request-local information."""

    ALL_TO_ALL = "all_to_all"
    RANK_CAUSAL = "rank_causal"
    TOP_RANKED_HUB = "top_ranked_hub"
    SAME_DOCUMENT_ONLY = "same_document_only"


@dataclass(frozen=True)
class CrossDocumentCompositionConfig:
    """Parameter-free first-stage gist composition configuration."""

    mode: CrossDocumentCompositionMode = CrossDocumentCompositionMode.GIST_SA_APPEND
    attention_mask: GistAttentionMask = GistAttentionMask.ALL_TO_ALL
    residual_scale: float = 1.0
    position_policy: str = "query_adjacent_compact_band"
    pooling_method: str = "layerwise_mean_pre_rope_kv"
    normalization_policy: str = "scaled_dot_product_softmax"

    def __post_init__(self) -> None:
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("cross-document residual scale must be finite and non-negative")
        if self.mode is CrossDocumentCompositionMode.INDEPENDENT_PRA:
            raise ValueError("independent PRA does not require a composition module")


@dataclass(frozen=True)
class CrossDocumentCompositionReceipt:
    """Auditable geometry and measured cost of one ephemeral composition."""

    mode: str
    gist_count: int
    gist_dim: int
    gist_attention_mask: str
    gist_attention_edges: int
    boundary_tokens_per_record: int
    corrected_token_count: int
    boundary_conditioning_edges: int
    request_composition_flops_estimate: int
    request_composition_ms: float
    request_composition_bytes: int
    persistent_native_tokens: int
    request_local_native_tokens: int
    gist_positions: tuple[int, ...]
    record_ids: tuple[str, ...]
    source_memory_digest: str
    pooling_method: str
    normalization_policy: str
    position_policy: str
    kv_correction_rank: int | None = None
    schema_version: str = "paper3.2-crossdoc-composition-v2"

    @property
    def receipt_id(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class CrossDocumentResidualAdapterConfig:
    """Architecture and objective for request-local cross-document K/V repair.

    The adapter sees one token's native K/V and a pooled summary of preceding
    records at the same layer.  Its output projection is initialized to zero,
    so an untrained adapter is exactly the independent-PRA baseline.
    """

    rank: int = 16
    activation: str = "tanh"
    kv_distillation_weight: float = 1.0
    response_distillation_weight: float = 0.25
    task_loss_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("cross-document adapter rank must be positive")
        if self.activation not in {"tanh", "silu"}:
            raise ValueError("cross-document adapter activation must be tanh or silu")
        weights = (
            self.kv_distillation_weight,
            self.response_distillation_weight,
            self.task_loss_weight,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("cross-document adapter loss weights must be finite and non-negative")
        if not any(weights):
            raise ValueError("cross-document adapter requires at least one positive loss weight")


@dataclass(frozen=True)
class BoundaryReencodeSpan:
    """One local bridge around a packed-document boundary.

    ``left_*`` identifies immutable native K/V exposed as context. ``right_*``
    identifies the later-document prefix that is actually re-encoded.
    Coordinates are offsets in the frozen packed selection.
    """

    boundary_index: int
    left_start: int
    left_end: int
    right_start: int
    right_end: int

    @property
    def context_tokens(self) -> int:
        return self.left_end - self.left_start

    @property
    def reencoded_tokens(self) -> int:
        return self.right_end - self.right_start


@dataclass(frozen=True)
class SelectiveBoundaryReencodeReceipt:
    """Auditable cost and geometry for parameter-free boundary re-encoding."""

    boundary_tokens: int
    boundary_count: int
    context_native_tokens: int
    reencoded_tokens: int
    request_reencode_ms: float
    persistent_native_tokens: int
    record_ids: tuple[str, ...]
    spans: tuple[BoundaryReencodeSpan, ...]
    source_memory_digest: str
    schema_version: str = "paper3.2-selective-boundary-reencode-v1"

    @property
    def receipt_id(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["receipt_id"] = self.receipt_id
        return payload


def boundary_reencode_spans(
    resource_lengths: Sequence[int], boundary_tokens: int
) -> tuple[BoundaryReencodeSpan, ...]:
    """Plan local tail-to-prefix bridges without inspecting task labels.

    At each join, at most ``boundary_tokens`` from the preceding record are
    exposed as immutable native context and at most the same number from the
    next record are recomputed.  The first record is never recomputed.
    """

    lengths = tuple(int(length) for length in resource_lengths)
    if len(lengths) < 2:
        raise ValueError("boundary re-encoding requires at least two records")
    if any(length <= 0 for length in lengths):
        raise ValueError("boundary re-encoding requires positive record lengths")
    if boundary_tokens <= 0:
        raise ValueError("boundary re-encoding window must be positive")
    spans: list[BoundaryReencodeSpan] = []
    cursor = lengths[0]
    previous_start = 0
    for boundary_index, current_length in enumerate(lengths[1:]):
        spans.append(
            BoundaryReencodeSpan(
                boundary_index=boundary_index,
                left_start=max(previous_start, cursor - boundary_tokens),
                left_end=cursor,
                right_start=cursor,
                right_end=cursor + min(boundary_tokens, current_length),
            )
        )
        previous_start = cursor
        cursor += current_length
    return tuple(spans)


def build_gist_attention_mask(
    count: int,
    policy: GistAttentionMask | str,
    *,
    document_ids: Sequence[str] = (),
) -> torch.Tensor:
    """Return a boolean ``[M, M]`` mask with at least self access per row."""

    if count <= 0:
        raise ValueError("gist attention requires at least one record")
    policy = GistAttentionMask(policy)
    if document_ids and len(document_ids) != count:
        raise ValueError("document identities must align with gists")
    if policy is GistAttentionMask.ALL_TO_ALL:
        return torch.ones((count, count), dtype=torch.bool)
    if policy is GistAttentionMask.RANK_CAUSAL:
        return torch.ones((count, count), dtype=torch.bool).tril()
    if policy is GistAttentionMask.TOP_RANKED_HUB:
        mask = torch.eye(count, dtype=torch.bool)
        mask[0, :] = True
        mask[:, 0] = True
        return mask
    if not document_ids:
        raise ValueError("same-document gist attention requires document identities")
    return torch.tensor(
        [[left == right for right in document_ids] for left in document_ids],
        dtype=torch.bool,
    )


def contextualize_gists(
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    residual_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply identity-projection gist SA over ``[..., M, D]`` K/V tensors.

    This is the parameter-free C1 baseline. Scores are accumulated in FP32,
    while contextualized K/V return to the source dtype. The inputs are never
    modified.
    """

    if keys.shape != values.shape or keys.ndim < 2:
        raise ValueError("gist keys and values must share shape [..., records, width]")
    records, width = keys.shape[-2:]
    if tuple(mask.shape) != (records, records):
        raise ValueError("gist attention mask must have shape [records, records]")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every gist query must have at least one visible key")
    if not math.isfinite(residual_scale) or residual_scale < 0:
        raise ValueError("residual scale must be finite and non-negative")
    scores = torch.einsum(
        "...id,...jd->...ij", keys.float(), keys.float()
    ) / math.sqrt(width)
    visible = mask.to(device=scores.device)
    scores = scores.masked_fill(~visible, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    contextual_keys = keys + residual_scale * torch.einsum(
        "...ij,...jd->...id", attention.to(keys.dtype), keys
    )
    contextual_values = values + residual_scale * torch.einsum(
        "...ij,...jd->...id", attention.to(values.dtype), values
    )
    return contextual_keys, contextual_values, attention


def memory_identity_digest(
    *, record_ids: Sequence[str], source_tokens: Sequence[int], layer_count: int
) -> str:
    """Fingerprint immutable memory identities and geometry without tensor bytes."""

    payload = {
        "record_ids": list(record_ids),
        "source_tokens": list(source_tokens),
        "layer_count": layer_count,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
