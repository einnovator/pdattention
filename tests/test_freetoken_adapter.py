from __future__ import annotations

import pytest

from pra_freetoken import PrefetchDemand, PrefetchStrategy, coordinate_prefetch


def demands() -> tuple[PrefetchDemand, ...]:
    return (
        PrefetchDemand("expert", bytes=20, need_time_s=3.0),
        PrefetchDemand("pra", bytes=10, need_time_s=5.0),
    )


def test_coordinator_keeps_expert_and_pra_traffic_disjoint() -> None:
    outcome = coordinate_prefetch(
        demands(), bandwidth_bytes_per_s=10.0, strategy="coordinated"
    )
    assert outcome.expert_bytes == 20
    assert outcome.pra_bytes == 10
    assert outcome.transferred_bytes == 30
    assert outcome.ready_fraction == 1.0
    assert outcome.exposed_latency_s == 0.0


def test_pra_only_exposes_the_unprefetched_expert_transfer() -> None:
    outcome = coordinate_prefetch(
        demands(), bandwidth_bytes_per_s=10.0, strategy=PrefetchStrategy.PRA_ONLY
    )
    assert outcome.exposed_latency_s == pytest.approx(2.0)
    assert outcome.accelerator_idle_s == pytest.approx(2.0)


def test_independent_fair_sharing_can_miss_an_early_deadline() -> None:
    outcome = coordinate_prefetch(
        demands(), bandwidth_bytes_per_s=10.0, strategy="independent"
    )
    assert outcome.exposed_latency_s == pytest.approx(1.0)
    assert outcome.ready_before_demand == 1


def test_invalid_bandwidth_is_rejected() -> None:
    with pytest.raises(ValueError):
        coordinate_prefetch(demands(), bandwidth_bytes_per_s=0, strategy="coordinated")
