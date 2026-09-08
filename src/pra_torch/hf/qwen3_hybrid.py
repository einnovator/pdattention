"""Fail-closed structural plan for Qwen3.8's hybrid GDN/QSA decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass


_QSA_TYPES = frozenset({"full_attention", "qwen_sparse_attention", "qsa"})
_GDN_TYPES = frozenset({"linear_attention", "gated_deltanet", "gdn"})


@dataclass(frozen=True)
class Qwen3HybridPRAPlan:
    """Serializable ownership map for a hybrid recurrent/sparse decoder."""

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
    """Classify QSA and GDN layers without importing a giant model.

    Native PRA token K/V can only attach to attention-bearing QSA layers. GDN
    layers maintain recurrent state rather than token-addressable K/V and must
    remain unchanged. Unknown layer kinds are reported explicitly so loading a
    future architecture revision fails review instead of being guessed.
    """

    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if not layer_types:
        raise ValueError("Qwen3 hybrid config must expose layer_types.")
    qsa = tuple(i for i, kind in enumerate(layer_types) if kind in _QSA_TYPES)
    gdn = tuple(i for i, kind in enumerate(layer_types) if kind in _GDN_TYPES)
    known = set(qsa) | set(gdn)
    unsupported = tuple(i for i in range(len(layer_types)) if i not in known)
    if not qsa:
        raise ValueError("Qwen3 hybrid config exposes no QSA/full-attention layer.")
    return Qwen3HybridPRAPlan(
        model_type=str(getattr(config, "model_type", "qwen3_hybrid")),
        layer_count=len(layer_types),
        qsa_layer_ids=qsa,
        gdn_layer_ids=gdn,
        unsupported_layer_ids=unsupported,
    )
