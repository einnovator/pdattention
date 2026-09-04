"""Logical-interval planning and native-K/V gathering for PRA memory.

Routing chunks are storage and search units.  This module deliberately works in
source-relative token coordinates so a semantic disclosure window can cross one
or more stored chunks without copying overlap or crossing a resource boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .memory import LayerKV, ReferenceChunkMemory


PROMPT_HISTORY_DOMAIN = "pra://prompt-history"


def logical_domain(source_uri: str) -> str:
    """Map cache/source identities to independent materialization domains.

    The implicit ``#__head`` resource and the live prompt tail are one logical
    prompt history.  Every explicit URI remains isolated from every other URI.
    """

    if source_uri in {"#__head", "#__prompt", PROMPT_HISTORY_DOMAIN}:
        return PROMPT_HISTORY_DOMAIN
    return source_uri


@dataclass(frozen=True, order=True)
class LogicalInterval:
    """Half-open token interval in one continuous materialization domain."""

    domain: str
    start: int
    end: int
    evidence_start: int | None = None
    evidence_end: int | None = None
    score: float = 1.0

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("A logical interval requires a nonempty domain.")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Logical intervals must satisfy 0 <= start < end.")
        evidence_start = self.start if self.evidence_start is None else self.evidence_start
        evidence_end = self.end if self.evidence_end is None else self.evidence_end
        if not (self.start <= evidence_start < evidence_end <= self.end):
            raise ValueError("Evidence bounds must be a nonempty subset of the interval.")
        object.__setattr__(self, "evidence_start", int(evidence_start))
        object.__setattr__(self, "evidence_end", int(evidence_end))

    @property
    def token_count(self) -> int:
        return self.end - self.start

    @property
    def evidence_token_count(self) -> int:
        assert self.evidence_start is not None and self.evidence_end is not None
        return self.evidence_end - self.evidence_start


@dataclass(frozen=True)
class LogicalDomainBounds:
    """Available source-relative extent for one independent resource."""

    domain: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Domain bounds must satisfy 0 <= start < end.")


@dataclass(frozen=True)
class LogicalKVShard:
    """One stored contiguous K/V shard with source-relative provenance."""

    domain: str
    start: int
    end: int
    kv: LayerKV
    shard_id: str
    source_uri: str
    encoding_granularity: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Logical K/V shards must satisfy 0 <= start < end.")
        expected = self.end - self.start
        if self.kv.k.ndim != 4 or self.kv.v.shape != self.kv.k.shape:
            raise ValueError("Shard K/V must have matching [1,H,T,Dh] shapes.")
        if self.kv.k.shape[0] != 1 or int(self.kv.k.shape[2]) != expected:
            raise ValueError("Shard K/V length must match its logical interval.")

    @property
    def token_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class GatherFragment:
    """Contiguous slice copied from one physical shard."""

    domain: str
    logical_start: int
    logical_end: int
    shard_id: str
    shard_local_start: int
    shard_local_end: int


@dataclass(frozen=True)
class MaterializationStats:
    """Logical, physical, transfer, and provenance accounting for one gather."""

    requested_tokens_pre_dedup: int
    deduplicated_tokens: int
    materialized_native_kv_tokens: int
    materialized_native_kv_bytes: int
    transferred_kv_bytes: int
    interval_resolution_seconds: float
    gather_seconds: float
    h2d_seconds: float
    evidence_tokens: int
    non_evidence_tokens: int
    storage_shards_touched: int
    cross_shard_interval_count: int
    encoding_granularities: tuple[int, ...]

    @property
    def evidence_density(self) -> float:
        return self.evidence_tokens / max(self.materialized_native_kv_tokens, 1)


@dataclass(frozen=True)
class MaterializedKV:
    """Ordered native K/V result and its exact logical provenance."""

    key: torch.Tensor
    value: torch.Tensor
    logical_positions: tuple[tuple[str, int], ...]
    intervals: tuple[LogicalInterval, ...]
    fragments: tuple[GatherFragment, ...]
    stats: MaterializationStats


class MissingLogicalKVError(ValueError):
    """Raised when a requested logical token has no stored K/V shard."""


def shards_from_chunks(chunks: Iterable[ReferenceChunkMemory]) -> list[LogicalKVShard]:
    """Convert cache chunks into source-domain K/V shards without copying tensors."""

    shards = []
    for chunk in chunks:
        assert chunk.logical_end is not None
        shards.append(
            LogicalKVShard(
                domain=logical_domain(chunk.source_uri),
                start=int(chunk.logical_start),
                end=int(chunk.logical_end),
                kv=chunk.token_kv,
                shard_id=chunk.chunk_id,
                source_uri=chunk.source_uri,
                encoding_granularity=(
                    int(chunk.metadata["encoding_block_tokens"])
                    if "encoding_block_tokens" in chunk.metadata
                    else None
                ),
            )
        )
    return shards


def domain_bounds_from_shards(shards: Iterable[LogicalKVShard]) -> dict[str, LogicalDomainBounds]:
    """Return the covered outer bounds for every independent logical domain."""

    grouped: dict[str, list[LogicalKVShard]] = {}
    for shard in shards:
        grouped.setdefault(shard.domain, []).append(shard)
    return {
        domain: LogicalDomainBounds(
            domain,
            min(shard.start for shard in values),
            max(shard.end for shard in values),
        )
        for domain, values in grouped.items()
    }


def evidence_centered_interval(
    domain: str,
    evidence_start: int,
    evidence_end: int,
    *,
    radius_left: int,
    radius_right: int,
    bounds: LogicalDomainBounds,
    score: float = 1.0,
) -> LogicalInterval:
    """Expand evidence within its own resource; expansion never changes domains."""

    if logical_domain(domain) != bounds.domain:
        raise ValueError("Evidence and bounds belong to different materialization domains.")
    if radius_left < 0 or radius_right < 0:
        raise ValueError("Materialization radii must be nonnegative.")
    if not (bounds.start <= evidence_start < evidence_end <= bounds.end):
        raise ValueError("Evidence lies outside its materialization domain.")
    return LogicalInterval(
        domain=bounds.domain,
        start=max(bounds.start, evidence_start - radius_left),
        end=min(bounds.end, evidence_end + radius_right),
        evidence_start=evidence_start,
        evidence_end=evidence_end,
        score=score,
    )


def union_intervals(intervals: Sequence[LogicalInterval]) -> list[LogicalInterval]:
    """Union overlap/adjacency per domain while preserving first-seen domain order."""

    domain_order = list(dict.fromkeys(interval.domain for interval in intervals))
    output: list[LogicalInterval] = []
    for domain in domain_order:
        values = sorted(
            (interval for interval in intervals if interval.domain == domain),
            key=lambda interval: (interval.start, interval.end),
        )
        if not values:
            continue
        start = values[0].start
        end = values[0].end
        evidence_start = values[0].evidence_start
        evidence_end = values[0].evidence_end
        score = values[0].score
        for interval in values[1:]:
            if interval.start <= end:
                end = max(end, interval.end)
                evidence_start = min(int(evidence_start), int(interval.evidence_start))
                evidence_end = max(int(evidence_end), int(interval.evidence_end))
                score = max(score, interval.score)
                continue
            output.append(
                LogicalInterval(
                    domain, start, end, int(evidence_start), int(evidence_end), score
                )
            )
            start, end = interval.start, interval.end
            evidence_start, evidence_end, score = (
                interval.evidence_start,
                interval.evidence_end,
                interval.score,
            )
        output.append(
            LogicalInterval(
                domain, start, end, int(evidence_start), int(evidence_end), score
            )
        )
    return output


def _allocate_integer_quotas(
    capacities: Sequence[int],
    weights: Sequence[float],
    total_budget: int,
    minimum_per_region: int | Sequence[int],
) -> list[int]:
    if total_budget <= 0:
        raise ValueError("A fixed K/V budget must be positive.")
    if len(capacities) != len(weights) or not capacities:
        raise ValueError("Budget allocation requires aligned nonempty regions and weights.")
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("Every materialization region must contain at least one token.")
    requested_minimums = (
        [int(minimum_per_region)] * len(capacities)
        if isinstance(minimum_per_region, int)
        else [int(value) for value in minimum_per_region]
    )
    if len(requested_minimums) != len(capacities) or any(value <= 0 for value in requested_minimums):
        raise ValueError("Every materialization region requires a positive minimum quota.")
    minimums = [min(minimum, capacity) for minimum, capacity in zip(requested_minimums, capacities)]
    if total_budget < sum(minimums):
        raise ValueError("K/V budget is too small to disclose every evidence region.")
    quotas = minimums[:]
    remaining = min(total_budget, sum(capacities)) - sum(quotas)
    while remaining > 0:
        candidates = [index for index, quota in enumerate(quotas) if quota < capacities[index]]
        if not candidates:
            break
        weight_total = sum(max(weights[index], 0.0) for index in candidates)
        if weight_total == 0:
            index = min(candidates, key=lambda item: (quotas[item], item))
        else:
            index = max(
                candidates,
                key=lambda item: (
                    max(weights[item], 0.0) / weight_total / max(quotas[item], 1),
                    -item,
                ),
            )
        quotas[index] += 1
        remaining -= 1
    return quotas


def allocate_interval_budget(
    intervals: Sequence[LogicalInterval],
    *,
    total_budget: int,
    strategy: str,
    minimum_per_region: int = 1,
) -> list[LogicalInterval]:
    """Allocate a fixed physical token budget without silently starving a region.

    Quotas are converted to centered subintervals.  If a quota can cover the
    evidence core, surrounding context is added symmetrically inside the
    candidate interval; otherwise a centered evidence slice is retained.
    """

    if minimum_per_region <= 0:
        raise ValueError("minimum_per_region must be positive.")
    capacities = [interval.token_count for interval in intervals]
    if strategy in {"equal", "minimum_core_remainder"}:
        weights = [1.0] * len(intervals)
    elif strategy == "evidence_length_proportional":
        weights = [float(interval.evidence_token_count) for interval in intervals]
    elif strategy == "score_proportional":
        weights = [max(float(interval.score), 0.0) for interval in intervals]
    else:
        raise ValueError(f"Unsupported K/V budget allocation strategy: {strategy}")
    minimums = (
        [max(minimum_per_region, interval.evidence_token_count) for interval in intervals]
        if strategy == "minimum_core_remainder"
        else minimum_per_region
    )
    quotas = _allocate_integer_quotas(capacities, weights, total_budget, minimums)
    output = []
    for interval, quota in zip(intervals, quotas):
        assert interval.evidence_start is not None and interval.evidence_end is not None
        evidence_length = interval.evidence_token_count
        if quota < evidence_length:
            start = interval.evidence_start + (evidence_length - quota) // 2
            end = start + quota
            evidence_start, evidence_end = start, end
        else:
            extra = quota - evidence_length
            left = min(extra // 2, interval.evidence_start - interval.start)
            right = min(extra - left, interval.end - interval.evidence_end)
            left = min(
                interval.evidence_start - interval.start,
                left + (extra - left - right),
            )
            start = interval.evidence_start - left
            end = interval.evidence_end + right
            if end - start < quota:
                end = min(interval.end, start + quota)
                start = max(interval.start, end - quota)
            evidence_start, evidence_end = interval.evidence_start, interval.evidence_end
        output.append(
            LogicalInterval(
                interval.domain,
                start,
                end,
                evidence_start,
                evidence_end,
                interval.score,
            )
        )
    return output


def _cover_interval(
    interval: LogicalInterval,
    shards: Sequence[LogicalKVShard],
) -> list[tuple[LogicalKVShard, int, int]]:
    candidates = sorted(
        (shard for shard in shards if shard.domain == interval.domain),
        key=lambda shard: (shard.start, -shard.end, shard.shard_id),
    )
    cursor = interval.start
    fragments = []
    while cursor < interval.end:
        covering = [shard for shard in candidates if shard.start <= cursor < shard.end]
        if not covering:
            raise MissingLogicalKVError(
                f"No K/V shard covers {interval.domain}[{cursor}] in "
                f"[{interval.start},{interval.end})."
            )
        # Prefer the shard extending furthest to avoid storage-overlap replay.
        shard = max(covering, key=lambda value: (value.end, -value.start, value.shard_id))
        fragment_end = min(interval.end, shard.end)
        fragments.append((shard, cursor, fragment_end))
        cursor = fragment_end
    return fragments


def gather_logical_kv(
    shards: Sequence[LogicalKVShard],
    intervals: Sequence[LogicalInterval],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> MaterializedKV:
    """Union intervals and gather exact ordered K/V across storage boundaries."""

    if not shards:
        raise ValueError("Logical materialization requires at least one K/V shard.")
    if not intervals:
        raise ValueError("Logical materialization requires at least one interval.")
    requested = sum(interval.token_count for interval in intervals)
    evidence_positions = {
        (interval.domain, position)
        for interval in intervals
        for position in range(int(interval.evidence_start), int(interval.evidence_end))
    }
    resolution_started = time.perf_counter()
    merged = union_intervals(intervals)
    covered = [(interval, _cover_interval(interval, shards)) for interval in merged]
    resolution_seconds = time.perf_counter() - resolution_started

    keys = []
    values = []
    positions: list[tuple[str, int]] = []
    fragments: list[GatherFragment] = []
    touched: set[str] = set()
    granularities: set[int] = set()
    gather_started = time.perf_counter()
    transfer_bytes = 0
    target_device = torch.device(device)
    target_dtype = dtype or shards[0].kv.k.dtype
    for interval, parts in covered:
        for shard, logical_start, logical_end in parts:
            local_start = logical_start - shard.start
            local_end = logical_end - shard.start
            key = shard.kv.k[:, :, local_start:local_end, :]
            value = shard.kv.v[:, :, local_start:local_end, :]
            if key.device != target_device or value.device != target_device:
                transfer_bytes += (
                    key.numel() * key.element_size() + value.numel() * value.element_size()
                )
            keys.append(key)
            values.append(value)
            touched.add(shard.shard_id)
            if shard.encoding_granularity is not None:
                granularities.add(shard.encoding_granularity)
            positions.extend(
                (interval.domain, position) for position in range(logical_start, logical_end)
            )
            fragments.append(
                GatherFragment(
                    interval.domain,
                    logical_start,
                    logical_end,
                    shard.shard_id,
                    local_start,
                    local_end,
                )
            )
    gather_seconds = time.perf_counter() - gather_started

    if target_device.type == "cuda" and transfer_bytes:
        torch.cuda.synchronize(target_device)
    transfer_started = time.perf_counter()
    keys = [key.to(target_device, target_dtype) for key in keys]
    values = [value.to(target_device, target_dtype) for value in values]
    if target_device.type == "cuda" and transfer_bytes:
        torch.cuda.synchronize(target_device)
    h2d_seconds = time.perf_counter() - transfer_started if transfer_bytes else 0.0
    key = torch.cat(keys, dim=2)
    value = torch.cat(values, dim=2)
    materialized = len(positions)
    evidence = sum(position in evidence_positions for position in positions)
    bytes_used = key.numel() * key.element_size() + value.numel() * value.element_size()
    return MaterializedKV(
        key=key,
        value=value,
        logical_positions=tuple(positions),
        intervals=tuple(merged),
        fragments=tuple(fragments),
        stats=MaterializationStats(
            requested_tokens_pre_dedup=requested,
            deduplicated_tokens=materialized,
            materialized_native_kv_tokens=materialized,
            materialized_native_kv_bytes=bytes_used,
            transferred_kv_bytes=transfer_bytes,
            interval_resolution_seconds=resolution_seconds,
            gather_seconds=gather_seconds,
            h2d_seconds=h2d_seconds,
            evidence_tokens=evidence,
            non_evidence_tokens=materialized - evidence,
            storage_shards_touched=len(touched),
            cross_shard_interval_count=sum(len(parts) > 1 for _interval, parts in covered),
            encoding_granularities=tuple(sorted(granularities)),
        ),
    )
