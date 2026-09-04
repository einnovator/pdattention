"""Auditable RAG+PRA realization and multi-resource position policies.

Paper 3.2 freezes retrieval and selection before this module is called.  The
functions below therefore change only how selected source tokens are realized
and positioned.  They never score content or silently alter selected spans.
"""

from __future__ import annotations

import hashlib
import json
import random
import itertools
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence


def _digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class RAGPRAProfile(str, Enum):
    """Named end-to-end profiles that keep selection and realization distinct."""

    RAG_ONLY_TEXT = "RAG_ONLY_TEXT"
    RAG_PLUS_PRA_SELECTED = "RAG_PLUS_PRA_SELECTED"
    RAG_PLUS_PRA_NATIVE_CONTIGUOUS = "RAG_PLUS_PRA_NATIVE_CONTIGUOUS"
    RAG_PLUS_PRA_NATIVE_INDEPENDENT = "RAG_PLUS_PRA_NATIVE_INDEPENDENT"
    RAG_PLUS_PRA_NATIVE_REBOUND = "RAG_PLUS_PRA_NATIVE_REBOUND"
    RAG_PLUS_PRA_REPAIR = "RAG_PLUS_PRA_REPAIR"


class SelectorRole(str, Enum):
    """The role PRA plays before realization."""

    EXTERNAL_ONLY = "external_only"
    PRA_SECOND_STAGE = "pra_second_stage"
    PRA_REPLACEMENT = "pra_replacement"


class MaterializationMode(str, Enum):
    """Physical representation presented to the model."""

    SELECTED_TEXT = "selected_text"
    NATIVE_CONTIGUOUS = "native_contiguous"
    NATIVE_INDEPENDENT = "native_independent"
    NATIVE_REBOUND = "native_rebound"
    NATIVE_REBOUND_REPAIR = "native_rebound_repair"


class PositionPolicy(str, Enum):
    """Request-specific coordinate systems for selected native resources."""

    SOURCE_LOCAL = "SOURCE_LOCAL"
    GLOBAL_PACKED = "GLOBAL_PACKED"
    RESOURCE_ADJACENT = "RESOURCE_ADJACENT"
    RANK_DISTANCE = "RANK_DISTANCE"
    SCORE_DISTANCE = "SCORE_DISTANCE"
    NON_OVERLAPPING_NEAR_BANDS = "NON_OVERLAPPING_NEAR_BANDS"
    RANDOM_DISTANCE = "RANDOM_DISTANCE"


@dataclass(frozen=True)
class ProfileContract:
    """Selection/realization contract for one canonical RAG+PRA profile."""

    profile: RAGPRAProfile
    default_selector_role: SelectorRole
    materialization: MaterializationMode
    requires_frozen_external_selection: bool


PROFILE_CONTRACTS: Mapping[RAGPRAProfile, ProfileContract] = {
    RAGPRAProfile.RAG_ONLY_TEXT: ProfileContract(
        RAGPRAProfile.RAG_ONLY_TEXT,
        SelectorRole.EXTERNAL_ONLY,
        MaterializationMode.SELECTED_TEXT,
        False,
    ),
    RAGPRAProfile.RAG_PLUS_PRA_SELECTED: ProfileContract(
        RAGPRAProfile.RAG_PLUS_PRA_SELECTED,
        SelectorRole.PRA_SECOND_STAGE,
        MaterializationMode.SELECTED_TEXT,
        False,
    ),
    RAGPRAProfile.RAG_PLUS_PRA_NATIVE_CONTIGUOUS: ProfileContract(
        RAGPRAProfile.RAG_PLUS_PRA_NATIVE_CONTIGUOUS,
        SelectorRole.EXTERNAL_ONLY,
        MaterializationMode.NATIVE_CONTIGUOUS,
        True,
    ),
    RAGPRAProfile.RAG_PLUS_PRA_NATIVE_INDEPENDENT: ProfileContract(
        RAGPRAProfile.RAG_PLUS_PRA_NATIVE_INDEPENDENT,
        SelectorRole.EXTERNAL_ONLY,
        MaterializationMode.NATIVE_INDEPENDENT,
        True,
    ),
    RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND: ProfileContract(
        RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
        SelectorRole.EXTERNAL_ONLY,
        MaterializationMode.NATIVE_REBOUND,
        True,
    ),
    RAGPRAProfile.RAG_PLUS_PRA_REPAIR: ProfileContract(
        RAGPRAProfile.RAG_PLUS_PRA_REPAIR,
        SelectorRole.EXTERNAL_ONLY,
        MaterializationMode.NATIVE_REBOUND_REPAIR,
        True,
    ),
}


@dataclass(frozen=True)
class SelectedResource:
    """One selected immutable source interval and its routing provenance.

    ``source_positions`` are token coordinates from the resource's original
    encoding.  Composition policies may translate them, but never change their
    length, internal offsets, identity, or source hash.
    """

    resource_id: str
    chunk_id: str
    source_sha256: str
    source_positions: tuple[int, ...]
    rank: int
    score: float

    def __post_init__(self) -> None:
        if not self.resource_id or not self.chunk_id:
            raise ValueError("resource and chunk IDs are required")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        int(self.source_sha256, 16)
        if not self.source_positions:
            raise ValueError("selected resources require at least one token")
        if any(right <= left for left, right in zip(self.source_positions, self.source_positions[1:])):
            raise ValueError("source positions must be strictly increasing")
        if self.rank <= 0:
            raise ValueError("rank must be positive")

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.resource_id, self.chunk_id, self.source_sha256


@dataclass(frozen=True)
class ResourcePlacement:
    """Effective request coordinates for one unchanged selected resource."""

    resource_id: str
    chunk_id: str
    source_sha256: str
    source_positions: tuple[int, ...]
    effective_positions: tuple[int, ...]
    rank: int
    score: float

    def __post_init__(self) -> None:
        if len(self.source_positions) != len(self.effective_positions):
            raise ValueError("position rebinding cannot change token count")
        source_deltas = tuple(
            right - left for left, right in zip(self.source_positions, self.source_positions[1:])
        )
        effective_deltas = tuple(
            right - left
            for left, right in zip(self.effective_positions, self.effective_positions[1:])
        )
        if source_deltas != effective_deltas:
            raise ValueError("position rebinding must preserve internal geometry")


@dataclass(frozen=True)
class CompositionReceipt:
    """Tamper-evident output of one request-specific composition policy."""

    selection_receipt_id: str
    profile: RAGPRAProfile
    selector_role: SelectorRole
    position_policy: PositionPolicy
    query_position: int
    placements: tuple[ResourcePlacement, ...]
    random_seed: int = 0
    repair_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.selection_receipt_id:
            raise ValueError("selection_receipt_id is required")
        if self.query_position < 0:
            raise ValueError("query_position cannot be negative")
        if not 0.0 <= self.repair_fraction <= 1.0:
            raise ValueError("repair_fraction must be in [0, 1]")
        if len({(row.resource_id, row.chunk_id) for row in self.placements}) != len(
            self.placements
        ):
            raise ValueError("composition resources must be unique")
        if any(
            position >= self.query_position
            for row in self.placements
            for position in row.effective_positions
        ):
            raise ValueError("all materialized evidence must precede the query")

    @property
    def receipt_id(self) -> str:
        return _digest(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "selection_receipt_id": self.selection_receipt_id,
            "profile": self.profile.value,
            "selector_role": self.selector_role.value,
            "position_policy": self.position_policy.value,
            "query_position": self.query_position,
            "placements": [asdict(row) for row in self.placements],
            "random_seed": self.random_seed,
            "repair_fraction": self.repair_fraction,
        }
        if include_receipt_id:
            value["receipt_id"] = self.receipt_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompositionReceipt":
        supplied = value.get("receipt_id")
        receipt = cls(
            selection_receipt_id=str(value["selection_receipt_id"]),
            profile=RAGPRAProfile(str(value["profile"])),
            selector_role=SelectorRole(str(value["selector_role"])),
            position_policy=PositionPolicy(str(value["position_policy"])),
            query_position=int(value["query_position"]),
            placements=tuple(
                ResourcePlacement(
                    resource_id=str(row["resource_id"]),
                    chunk_id=str(row["chunk_id"]),
                    source_sha256=str(row["source_sha256"]),
                    source_positions=tuple(int(item) for item in row["source_positions"]),
                    effective_positions=tuple(int(item) for item in row["effective_positions"]),
                    rank=int(row["rank"]),
                    score=float(row["score"]),
                )
                for row in value["placements"]  # type: ignore[index]
            ),
            random_seed=int(value.get("random_seed", 0)),
            repair_fraction=float(value.get("repair_fraction", 0.0)),
        )
        if supplied is not None and supplied != receipt.receipt_id:
            raise ValueError("composition receipt digest does not match its contents")
        return receipt


def _translated(source: Sequence[int], start: int) -> tuple[int, ...]:
    origin = source[0]
    return tuple(start + position - origin for position in source)


def _minimum_query_position(
    resources: Sequence[SelectedResource], policy: PositionPolicy, gap: int
) -> int:
    total = sum(resource.source_positions[-1] - resource.source_positions[0] + 1 for resource in resources)
    source_max = max(resource.source_positions[-1] for resource in resources)
    max_length = max(
        resource.source_positions[-1] - resource.source_positions[0] + 1
        for resource in resources
    )
    if policy in {
        PositionPolicy.RANK_DISTANCE,
        PositionPolicy.SCORE_DISTANCE,
        PositionPolicy.RANDOM_DISTANCE,
    }:
        stride = max_length + max(gap, 1)
        policy_minimum = gap + max_length + (len(resources) - 1) * stride
    elif policy is PositionPolicy.RESOURCE_ADJACENT:
        policy_minimum = gap + max_length
    else:
        policy_minimum = total + gap
    return max(policy_minimum, source_max + gap if policy is PositionPolicy.SOURCE_LOCAL else 0)


def compose_resources(
    resources: Sequence[SelectedResource],
    *,
    selection_receipt_id: str,
    profile: RAGPRAProfile,
    position_policy: PositionPolicy,
    selector_role: SelectorRole | None = None,
    query_position: int | None = None,
    near_gap: int = 4,
    random_seed: int = 0,
    repair_fraction: float = 0.0,
) -> CompositionReceipt:
    """Place selected resources without changing their identity or token geometry."""

    if not resources:
        raise ValueError("at least one selected resource is required")
    if near_gap < 0:
        raise ValueError("near_gap cannot be negative")
    if len({resource.identity for resource in resources}) != len(resources):
        raise ValueError("selected resource identities must be unique")

    contract = PROFILE_CONTRACTS[profile]
    role = selector_role or contract.default_selector_role
    if query_position is None:
        query_position = _minimum_query_position(resources, position_policy, near_gap)
    lengths = {
        resource.identity: resource.source_positions[-1] - resource.source_positions[0] + 1
        for resource in resources
    }
    if position_policy is PositionPolicy.SOURCE_LOCAL:
        effective = {resource.identity: resource.source_positions for resource in resources}
    elif position_policy is PositionPolicy.GLOBAL_PACKED:
        cursor = 0
        effective = {}
        for resource in resources:
            effective[resource.identity] = _translated(resource.source_positions, cursor)
            cursor = effective[resource.identity][-1] + 1
    elif position_policy is PositionPolicy.RESOURCE_ADJACENT:
        effective = {
            resource.identity: _translated(
                resource.source_positions, query_position - near_gap - lengths[resource.identity]
            )
            for resource in resources
        }
    elif position_policy is PositionPolicy.NON_OVERLAPPING_NEAR_BANDS:
        cursor = query_position - near_gap
        effective = {}
        for resource in sorted(resources, key=lambda row: (row.rank, row.resource_id), reverse=True):
            start = cursor - lengths[resource.identity]
            effective[resource.identity] = _translated(resource.source_positions, start)
            cursor = start
    else:
        ordered = sorted(resources, key=lambda row: (row.rank, row.resource_id))
        stride = max(lengths.values()) + max(near_gap, 1)
        if position_policy is PositionPolicy.RANK_DISTANCE:
            band = {resource.identity: max(resource.rank - 1, 0) for resource in resources}
        elif position_policy is PositionPolicy.SCORE_DISTANCE:
            high = max(resource.score for resource in resources)
            low = min(resource.score for resource in resources)
            scale = high - low
            band = {
                resource.identity: int(round((high - resource.score) / scale * (len(resources) - 1)))
                if scale
                else resource.rank - 1
                for resource in resources
            }
        elif position_policy is PositionPolicy.RANDOM_DISTANCE:
            bands = list(range(len(resources)))
            random.Random(random_seed).shuffle(bands)
            band = {resource.identity: value for resource, value in zip(ordered, bands)}
        else:
            raise ValueError(f"unsupported position policy: {position_policy}")
        effective = {
            resource.identity: _translated(
                resource.source_positions,
                query_position
                - near_gap
                - lengths[resource.identity]
                - band[resource.identity] * stride,
            )
            for resource in resources
        }

    placements = tuple(
        ResourcePlacement(
            resource_id=resource.resource_id,
            chunk_id=resource.chunk_id,
            source_sha256=resource.source_sha256,
            source_positions=resource.source_positions,
            effective_positions=effective[resource.identity],
            rank=resource.rank,
            score=resource.score,
        )
        for resource in resources
    )
    if any(position < 0 for row in placements for position in row.effective_positions):
        raise ValueError("query_position is too small for the selected position policy")
    return CompositionReceipt(
        selection_receipt_id=selection_receipt_id,
        profile=profile,
        selector_role=role,
        position_policy=position_policy,
        query_position=query_position,
        placements=placements,
        random_seed=random_seed,
        repair_fraction=repair_fraction,
    )


def permute_resources(
    resources: Sequence[SelectedResource], order: Sequence[str]
) -> tuple[SelectedResource, ...]:
    """Apply a declared D1/D2/resource permutation without changing content."""

    by_id = {resource.resource_id: resource for resource in resources}
    if len(by_id) != len(resources) or set(order) != set(by_id) or len(order) != len(resources):
        raise ValueError("order must contain every unique resource ID exactly once")
    return tuple(by_id[resource_id] for resource_id in order)


def permutation_orders(
    resource_ids: Sequence[str], *, seed: int = 0, max_random: int = 4
) -> tuple[tuple[str, ...], ...]:
    """Return canonical, reverse, and bounded deterministic resource orders."""

    canonical = tuple(resource_ids)
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("resource IDs must be non-empty and unique")
    orders: list[tuple[str, ...]] = [canonical]
    reverse = tuple(reversed(canonical))
    if reverse not in orders:
        orders.append(reverse)
    if len(canonical) <= 7:
        candidates = list(itertools.permutations(canonical))
        random.Random(seed).shuffle(candidates)
        for order in candidates:
            if order not in orders:
                orders.append(order)
            if len(orders) >= 2 + max_random:
                break
    else:
        rng = random.Random(seed)
        attempts = 0
        while len(orders) < 2 + max_random and attempts < max_random * 20:
            candidate = list(canonical)
            rng.shuffle(candidate)
            order = tuple(candidate)
            if order not in orders:
                orders.append(order)
            attempts += 1
    return tuple(orders)
