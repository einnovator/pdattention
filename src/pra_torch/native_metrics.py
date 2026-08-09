"""Canonical derived metrics for fixed-target native-KV experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping


RCB_EPSILON = 1e-8


def recovered_context_benefit(
    *,
    sa_full_loss: float,
    sa_tail_loss: float,
    pra_loss: float,
    epsilon: float = RCB_EPSILON,
) -> float | None:
    """Return the fraction of the dense context benefit recovered by PRA.

    The value is intentionally not clamped.  ``None`` marks targets for which
    full context provides too little benefit to define a stable ratio.
    """

    dependency_gain = float(sa_tail_loss) - float(sa_full_loss)
    if abs(dependency_gain) <= epsilon:
        return None
    return (float(sa_tail_loss) - float(pra_loss)) / dependency_gain


def derive_native_kv_metrics(losses: Mapping[str, float]) -> dict[str, float | None]:
    """Decompose native-KV quality into transport, sparsity, and routing terms."""

    required = {
        "sa_full",
        "sa_tail",
        "native_all",
        "native_oracle",
        "native_routed",
        "native_shuffled",
        "native_disabled",
    }
    missing = sorted(required.difference(losses))
    if missing:
        raise ValueError(f"Missing native-KV conditions: {', '.join(missing)}")
    values = {key: float(value) for key, value in losses.items()}
    result: dict[str, float | None] = {
        "transport_gap": values["native_all"] - values["sa_full"],
        "sparse_gap": values["native_oracle"] - values["native_all"],
        "routing_gap": values["native_routed"] - values["native_oracle"],
        "full_context_benefit": values["sa_tail"] - values["sa_full"],
        "dependency_gain": values["sa_tail"] - values["sa_full"],
        "content_causality_oracle": values["native_shuffled"] - values["native_oracle"],
        "content_causality_routed": values["native_shuffled"] - values["native_routed"],
        "disabled_wrapper_gap": values["native_disabled"] - values["sa_tail"],
    }
    for condition in ("all", "oracle", "routed", "shuffled"):
        result[f"memory_benefit_{condition}"] = values["sa_tail"] - values[f"native_{condition}"]
        result[f"rcb_{condition}"] = recovered_context_benefit(
            sa_full_loss=values["sa_full"],
            sa_tail_loss=values["sa_tail"],
            pra_loss=values[f"native_{condition}"],
        )
    return result


def finite_values(values):
    """Return numeric non-null values suitable for aggregation."""

    return [float(value) for value in values if value is not None and math.isfinite(float(value))]
