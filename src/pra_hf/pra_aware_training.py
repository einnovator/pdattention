"""Pretrained-model adaptation scopes for Paper 4 PRA-aware training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

from pra_torch.pra_aware_training import LoRALinear


HFAdaptationRegime = Literal[
    "native",
    "frozen_pra",
    "consumer_lora",
    "interface_lora",
    "broad_lora",
    "full_weight_pra",
]
GemmaPlacement = Literal[
    "all_global",
    "every_second_global",
    "late_global",
    "all_eligible",
]


@dataclass(frozen=True)
class GemmaLayerTopology:
    """Machine-readable native/PRA ownership for one Gemma decoder layer."""

    layer_index: int
    native_attention_type: str
    converted_attention_type: str
    native_window: int | None
    pra_enabled: bool


def gemma_layer_topology(
    config, placement: GemmaPlacement = "all_global"
) -> tuple[GemmaLayerTopology, ...]:
    """Select PRA slots from a checkpoint's declared local/global schedule."""

    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if not layer_types:
        raise ValueError("Gemma config must expose layer_types.")
    global_ids = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    )
    if not global_ids:
        raise ValueError("Gemma config contains no native global-attention slots.")
    if placement == "all_global":
        selected = set(global_ids)
    elif placement == "every_second_global":
        selected = set(global_ids[1::2] or global_ids[-1:])
    elif placement == "late_global":
        selected = set(global_ids[-min(2, len(global_ids)) :])
    elif placement == "all_eligible":
        selected = set(range(len(layer_types)))
    else:
        raise ValueError(f"Unsupported Gemma placement: {placement}")
    window = getattr(config, "sliding_window", None)
    return tuple(
        GemmaLayerTopology(
            layer_index=index,
            native_attention_type=layer_type,
            converted_attention_type="pra" if index in selected else layer_type,
            native_window=(
                int(window)
                if layer_type == "sliding_attention" and window
                else None
            ),
            pra_enabled=index in selected,
        )
        for index, layer_type in enumerate(layer_types)
    )


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_id(path: str) -> int | None:
    match = _LAYER_PATTERN.search(path)
    return int(match.group(1)) if match else None


def _replace_linear(
    root: nn.Module,
    path: str,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    parent: nn.Module = root
    pieces = path.split(".")
    for piece in pieces[:-1]:
        parent = parent[int(piece)] if piece.isdigit() else getattr(parent, piece)
    name = pieces[-1]
    layer = parent[int(name)] if name.isdigit() else getattr(parent, name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"LoRA target {path} is not linear.")
    replacement = LoRALinear(
        layer,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    ).to(device=layer.weight.device, dtype=layer.weight.dtype)
    if name.isdigit():
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)


def install_hf_adaptation_regime(
    model: nn.Module,
    regime: HFAdaptationRegime,
    *,
    pra_layer_ids: tuple[int, ...],
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> tuple[str, ...]:
    """Apply the frozen/Consumer/Interface/Broad/full ownership contract.

    Consumer LoRA exposes Q/O and the following MLP at PRA slots. Interface
    LoRA also exposes K/V. Broad LoRA covers attention and MLP linears across
    the decoder. Zero-initialized up projections preserve the initial function.
    """

    regimes = {
        "native",
        "frozen_pra",
        "consumer_lora",
        "interface_lora",
        "broad_lora",
        "full_weight_pra",
    }
    if regime not in regimes:
        raise ValueError(f"Unsupported HF adaptation regime: {regime}")
    for parameter in model.parameters():
        parameter.requires_grad = regime == "full_weight_pra"
    if regime in {"native", "frozen_pra", "full_weight_pra"}:
        return ()
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("An HF adaptation regime is already installed.")

    pra_layers = {int(layer) for layer in pra_layer_ids}
    targets = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer = _layer_id(path)
        if layer is None:
            continue
        is_attention = any(
            path.endswith(f".{name}_proj") for name in ("q", "k", "v", "o")
        )
        is_mlp = ".mlp." in path or ".feed_forward." in path
        if not (is_attention or is_mlp):
            continue
        if regime == "consumer_lora":
            selected = layer in pra_layers and (
                is_mlp or path.endswith(".q_proj") or path.endswith(".o_proj")
            )
        elif regime == "interface_lora":
            selected = layer in pra_layers
        else:
            selected = True
        if selected:
            targets.append(path)
    for path in targets:
        _replace_linear(
            model,
            path,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
    return tuple(targets)


def hf_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    """Report total and exposed parameters after one adaptation setup."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / max(total, 1)),
    }
