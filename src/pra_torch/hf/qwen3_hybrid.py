"""Fail-closed topology plan for Qwen3.8's hybrid GDN/QSA decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Qwen3HybridPRAPlan:
    model_type: str
    layer_count: int
    qsa_layer_ids: tuple[int, ...]
    gdn_layer_ids: tuple[int, ...]
    unsupported_layer_ids: tuple[int, ...]
    pra_scope: str = "qsa_attention_only"
    recurrent_state_ownership: str = "host_model"
    sparse_indexer_ownership: str = "host_model"
    evaluation_status: str = "STRUCTURAL_ONLY"

    def to_dict(self) -> dict:
        return asdict(self)


def describe_qwen3_hybrid_plan(config) -> Qwen3HybridPRAPlan:
    """Separate token-K/V QSA layers from host-owned recurrent GDN state."""

    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if not layer_types:
        raise ValueError("Qwen3 hybrid config must expose layer_types.")
    qsa_types = {"full_attention", "qwen_sparse_attention", "qsa"}
    gdn_types = {"linear_attention", "gated_deltanet", "gdn"}
    qsa = tuple(i for i, kind in enumerate(layer_types) if kind in qsa_types)
    gdn = tuple(i for i, kind in enumerate(layer_types) if kind in gdn_types)
    known = set(qsa) | set(gdn)
    if not qsa:
        raise ValueError("Qwen3 hybrid config exposes no QSA/full-attention layer.")
    return Qwen3HybridPRAPlan(
        model_type=str(getattr(config, "model_type", "qwen3_hybrid")),
        layer_count=len(layer_types),
        qsa_layer_ids=qsa,
        gdn_layer_ids=gdn,
        unsupported_layer_ids=tuple(i for i in range(len(layer_types)) if i not in known),
    )
