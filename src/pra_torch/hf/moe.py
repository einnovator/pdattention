"""Host-owned mixture-of-experts topology contract for PRA-HF.

PRA extends attention; it must not replace, reroute, or duplicate a model's
expert MLPs.  This module records that boundary independently of any one MoE
family so later adapters can share the same isolation checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anatomy import resolve_decoder_anatomy


@dataclass(frozen=True)
class HFMoETopology:
    """Serializable description of the host model's sparse-MLP topology."""

    model_type: str
    layer_count: int
    expert_count: int
    experts_per_token: int
    moe_layer_ids: tuple[int, ...]
    dense_layer_ids: tuple[int, ...]
    attention_module_types: tuple[str, ...]
    expert_module_types: tuple[str, ...]
    router_ownership: str = "host_model"
    expert_execution_ownership: str = "host_model"
    pra_scope: str = "attention_only"


@dataclass(frozen=True)
class HFMoEIsolationSnapshot:
    """Object identities that must survive attention-module injection."""

    mlp_ids: tuple[int, ...]
    router_ids: tuple[int, ...]
    expert_container_ids: tuple[int, ...]


def _config_int(config, names: tuple[str, ...], default: int = 0) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return default


def describe_moe_topology(model) -> HFMoETopology | None:
    """Describe a decoder MoE without assuming its expert implementation.

    A model qualifies as MoE only when at least one decoder MLP exposes both a
    router-like ``gate`` and an ``experts`` container. Configuration aliases
    cover the naming used by common Hugging Face MoE families while concrete
    attention compatibility remains the responsibility of a family adapter.
    """

    try:
        anatomy = resolve_decoder_anatomy(model)
    except TypeError:
        return None
    layers = tuple(anatomy.layers)
    moe_layers = tuple(
        index
        for index, layer in enumerate(layers)
        if (
            hasattr(layer.mlp, "experts")
            and (hasattr(layer.mlp, "gate") or hasattr(layer.mlp, "router"))
        )
    )
    if not moe_layers:
        return None
    config = anatomy.config
    expert_count = _config_int(
        config,
        ("num_experts", "num_local_experts", "n_routed_experts"),
    )
    experts_per_token = _config_int(
        config,
        ("num_experts_per_tok", "num_experts_per_token", "experts_per_token"),
    )
    if expert_count <= 0 or experts_per_token <= 0:
        raise ValueError("MoE topology requires positive expert and top-k counts.")
    dense_layers = tuple(index for index in range(len(layers)) if index not in moe_layers)
    return HFMoETopology(
        model_type=str(getattr(config, "model_type", "unknown")),
        layer_count=len(layers),
        expert_count=expert_count,
        experts_per_token=experts_per_token,
        moe_layer_ids=moe_layers,
        dense_layer_ids=dense_layers,
        attention_module_types=tuple(sorted({type(layer.self_attn).__qualname__ for layer in layers})),
        expert_module_types=tuple(sorted({type(layers[index].mlp).__qualname__ for index in moe_layers})),
    )


def snapshot_moe_isolation(model) -> HFMoEIsolationSnapshot | None:
    """Capture host-owned MoE module identities before attention injection."""

    topology = describe_moe_topology(model)
    if topology is None:
        return None
    anatomy = resolve_decoder_anatomy(model)
    mlps = tuple(anatomy.layers[index].mlp for index in topology.moe_layer_ids)
    return HFMoEIsolationSnapshot(
        mlp_ids=tuple(id(module) for module in mlps),
        router_ids=tuple(
            id(getattr(module, "gate", getattr(module, "router", None)))
            for module in mlps
        ),
        expert_container_ids=tuple(id(module.experts) for module in mlps),
    )


def assert_moe_isolation(model, expected: HFMoEIsolationSnapshot | None) -> None:
    """Fail if PRA attention injection changed any host router or expert block."""

    if expected is None:
        return
    actual = snapshot_moe_isolation(model)
    if actual != expected:
        raise RuntimeError("PRA-HF attention injection modified host-owned MoE modules.")
