"""Deterministic bandwidth model for independent expert and PRA prefetch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrefetchStrategy(str, Enum):
    """Which predicted resource classes may move before first demand."""

    EXPERT_ONLY = "expert_only"
    PRA_ONLY = "pra_only"
    INDEPENDENT = "independent"
    COORDINATED = "coordinated"


@dataclass(frozen=True)
class PrefetchDemand:
    """One physical transfer demand; identity remains external to this model."""

    resource_class: str
    bytes: int
    need_time_s: float
    cache_hit: bool = False
    eventually_used: bool = True

    def __post_init__(self) -> None:
        if self.resource_class not in {"expert", "pra"}:
            raise ValueError("resource_class must be 'expert' or 'pra'.")
        if self.bytes < 0 or self.need_time_s < 0:
            raise ValueError("bytes and need_time_s must be non-negative.")


@dataclass(frozen=True)
class PrefetchOutcome:
    strategy: PrefetchStrategy
    transferred_bytes: int
    expert_bytes: int
    pra_bytes: int
    exposed_latency_s: float
    accelerator_idle_s: float
    ready_before_demand: int
    demand_count: int
    wasted_prefetch_bytes: int

    @property
    def ready_fraction(self) -> float:
        return self.ready_before_demand / max(1, self.demand_count)


def coordinate_prefetch(
    demands: tuple[PrefetchDemand, ...],
    *,
    bandwidth_bytes_per_s: float,
    strategy: PrefetchStrategy | str,
) -> PrefetchOutcome:
    """Evaluate a single-link schedule using an earliest-deadline coordinator.

    Independent mode starts all eligible transfers at time zero and models fair
    bandwidth sharing. Coordinated mode serializes prefetched transfers by the
    earliest demand deadline. Non-prefetched resources start on demand. This
    algebraic model is a controlled scheduling experiment, not a live E3 claim.
    """

    strategy = PrefetchStrategy(strategy)
    if bandwidth_bytes_per_s <= 0:
        raise ValueError("bandwidth_bytes_per_s must be positive.")
    misses = tuple(demand for demand in demands if not demand.cache_hit and demand.bytes)
    prefetch_classes = {
        PrefetchStrategy.EXPERT_ONLY: {"expert"},
        PrefetchStrategy.PRA_ONLY: {"pra"},
        PrefetchStrategy.INDEPENDENT: {"expert", "pra"},
        PrefetchStrategy.COORDINATED: {"expert", "pra"},
    }[strategy]
    prefetched = tuple(d for d in misses if d.resource_class in prefetch_classes)
    on_demand = tuple(d for d in misses if d.resource_class not in prefetch_classes)
    completion: dict[int, float] = {}

    link_free = 0.0
    if strategy == PrefetchStrategy.INDEPENDENT and prefetched:
        share = bandwidth_bytes_per_s / len(prefetched)
        for demand in prefetched:
            completion[id(demand)] = demand.bytes / share
        link_free = max(completion.values(), default=0.0)
    else:
        cursor = 0.0
        for demand in sorted(prefetched, key=lambda row: (row.need_time_s, row.resource_class)):
            cursor += demand.bytes / bandwidth_bytes_per_s
            completion[id(demand)] = cursor
        link_free = cursor

    on_demand_cursor = link_free
    for demand in sorted(on_demand, key=lambda row: row.need_time_s):
        start = max(demand.need_time_s, on_demand_cursor)
        finish = start + demand.bytes / bandwidth_bytes_per_s
        completion[id(demand)] = finish
        on_demand_cursor = finish

    lateness = [max(0.0, completion[id(d)] - d.need_time_s) for d in misses]
    ready = sum(completion[id(d)] <= d.need_time_s for d in misses)
    transferred = sum(d.bytes for d in misses)
    return PrefetchOutcome(
        strategy=strategy,
        transferred_bytes=transferred,
        expert_bytes=sum(d.bytes for d in misses if d.resource_class == "expert"),
        pra_bytes=sum(d.bytes for d in misses if d.resource_class == "pra"),
        exposed_latency_s=sum(lateness),
        accelerator_idle_s=max(lateness, default=0.0),
        ready_before_demand=ready,
        demand_count=len(misses),
        wasted_prefetch_bytes=sum(d.bytes for d in prefetched if not d.eventually_used),
    )
