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
    schema_version: str = "paper3.2-crossdoc-composition-v1"

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
