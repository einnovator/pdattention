"""Resident agent-history K/V selection for Transformers DynamicCache."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .live_history import LiveKVSelectionPlan


@dataclass(frozen=True)
class HFResidentKVSelection:
    """A request cache derived only from an already evaluated source cache."""

    cache: object
    plan: LiveKVSelectionPlan
    physical_kv_copy: bool
    selected_text_reencoded_tokens: int = 0


def _layer_pair(layer: object) -> tuple[object, object, str, str]:
    for key_name, value_name in (
        ("keys", "values"),
        ("key_cache", "value_cache"),
    ):
        keys = getattr(layer, key_name, None)
        values = getattr(layer, value_name, None)
        if keys is not None and values is not None:
            return keys, values, key_name, value_name
    raise TypeError(
        f"Unsupported Transformers cache layer {type(layer)!r}; expected K/V tensors."
    )


def _select_tensor(tensor, plan: LiveKVSelectionPlan):
    import torch

    pieces = [tensor[..., row.start : row.end, :] for row in plan.intervals]
    if not pieces:
        return tensor[..., :0, :], False
    if len(pieces) == 1:
        return pieces[0], False
    return torch.cat(pieces, dim=-2), True


def select_dynamic_cache(
    source_cache: object, plan: LiveKVSelectionPlan
) -> HFResidentKVSelection:
    """Select resident K/V tensors without invoking tokenization or the model.

    A single contiguous interval remains a tensor view. Multiple disjoint
    intervals require a packed tensor copy in the portable HF cache API; that
    cost is reported explicitly and is not confused with text re-encoding.
    """

    layers = getattr(source_cache, "layers", None)
    if layers is None:
        raise TypeError("Live HF K/V selection requires a Transformers cache with layers.")
    selected = copy.copy(source_cache)
    selected.layers = []
    copied = False
    for source_layer in layers:
        keys, values, key_name, value_name = _layer_pair(source_layer)
        if int(keys.shape[-2]) < plan.source_tokens:
            raise ValueError("HF source cache is shorter than the selection plan.")
        layer = copy.copy(source_layer)
        chosen_keys, key_copy = _select_tensor(keys, plan)
        chosen_values, value_copy = _select_tensor(values, plan)
        setattr(layer, key_name, chosen_keys)
        setattr(layer, value_name, chosen_values)
        selected.layers.append(layer)
        copied = copied or key_copy or value_copy
    return HFResidentKVSelection(selected, plan, copied)

