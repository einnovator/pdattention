"""Capability-graph disclosure policies for Paper 6.5 M5.

Discovery ranks candidate identities. Disclosure decides which bounded set of
definitions becomes visible for planning. The APIs in this module keep those
axes independent and preserve why every non-root capability was included.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from pra_hf.agent_resources import (
    AgentResource,
    DiscoveryHint,
    DiscoveryMode,
    DiscoveryTrace,
    SideEffectClass,
)


class DisclosureMode(str, Enum):
    """User-visible breadth profiles independent of discovery channels."""

    MINIMAL = "minimal"
    LOCAL = "local"
    PLANNING = "planning"
    BROAD = "broad"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class DiscoveryPolicy:
    """Resource-ranking policy embedded in the combined SDK configuration."""

    mode: DiscoveryMode | str = DiscoveryMode.ADAPTIVE
    strict: bool = False

    def hint(self) -> DiscoveryHint:
        return DiscoveryHint(self.mode, self.strict)


@dataclass(frozen=True)
class ToolDisclosurePolicy:
    """Budgets for direct roots and each capability-graph signal."""

    mode: DisclosureMode | str = DisclosureMode.PLANNING
    direct_k: int = 1
    family_k: int = 3
    tag_k: int = 2
    schema_successor_k: int = 4
    schema_predecessor_k: int = 2
    speculative_k: int = 2
    max_tools: int = 10
    schema_depth: int = 3
    allow_unsafe: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", DisclosureMode(self.mode))
        budgets = (
            self.direct_k,
            self.family_k,
            self.tag_k,
            self.schema_successor_k,
            self.schema_predecessor_k,
            self.speculative_k,
            self.max_tools,
            self.schema_depth,
        )
        if any(value < 0 for value in budgets) or self.max_tools <= 0:
            raise ValueError("Disclosure budgets must be nonnegative and max_tools positive.")


@dataclass(frozen=True)
class AgentResourcePolicy:
    """Orthogonal discovery and disclosure controls exposed to callers."""

    discovery: DiscoveryPolicy = field(default_factory=DiscoveryPolicy)
    disclosure: ToolDisclosurePolicy = field(default_factory=ToolDisclosurePolicy)


@dataclass(frozen=True)
class CapabilityEdge:
    """One typed graph relationship with direction and signal provenance."""

    source_uri: str
    target_uri: str
    edge_type: str
    weight: float
    directed: bool
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class DisclosureProvenance:
    """Why a resource entered one disclosed planning set."""

    uri: str
    source: str
    root_uri: str | None
    edge_type: str | None = None
    edge_weight: float | None = None
    direct_rank: int | None = None
    graph_depth: int = 0


@dataclass(frozen=True)
class DisclosureTrace:
    """Bounded disclosed set plus graph and budget accounting."""

    requested_mode: str
    resolved_mode: str
    root_uris: tuple[str, ...]
    disclosed_uris: tuple[str, ...]
    provenance: tuple[DisclosureProvenance, ...]
    graph_expansions: int
    candidate_edges_considered: int
    unsafe_suppressed: int


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _read_write_pair(left: str | None, right: str | None) -> bool:
    read = {"list", "search", "get", "read", "inspect", "validate"}
    write = {"create", "update", "write", "archive", "restore", "notify", "export"}
    return bool(left in read and right in write or right in read and left in write)


class ToolCapabilityGraph:
    """Offline metadata graph over typed resources.

    Undirected similarity edges retain each contributing signal. Directional
    schema edges connect producer output types to consumer input types.
    """

    def __init__(self, resources: Iterable[AgentResource]) -> None:
        self.resources = tuple(resources)
        self.by_uri = {resource.uri: resource for resource in self.resources}
        if len(self.by_uri) != len(self.resources):
            raise ValueError("Capability graph resource URIs must be unique.")
        self.edges = self._build_edges()
        self.outgoing: dict[str, list[CapabilityEdge]] = defaultdict(list)
        self.incoming: dict[str, list[CapabilityEdge]] = defaultdict(list)
        for edge in self.edges:
            self.outgoing[edge.source_uri].append(edge)
            self.incoming[edge.target_uri].append(edge)
            if not edge.directed:
                reverse = CapabilityEdge(
                    edge.target_uri,
                    edge.source_uri,
                    edge.edge_type,
                    edge.weight,
                    False,
                    edge.provenance,
                )
                self.outgoing[reverse.source_uri].append(reverse)
                self.incoming[reverse.target_uri].append(reverse)

    def _build_edges(self) -> tuple[CapabilityEdge, ...]:
        edges: list[CapabilityEdge] = []
        for index, left in enumerate(self.resources):
            for right in self.resources[index + 1 :]:
                signals: list[tuple[str, float]] = []
                category = _jaccard(left.toolset_categories, right.toolset_categories)
                objects = _jaccard(left.object_types, right.object_types)
                tags = _jaccard(left.tags, right.tags)
                keywords = _jaccard(left.keywords, right.keywords)
                if category:
                    signals.append(("same_category", category))
                if objects:
                    signals.append(("same_object", objects))
                if left.namespace == right.namespace:
                    signals.append(("same_namespace", 0.15))
                if left.api_family and left.api_family == right.api_family:
                    signals.append(("same_api", 0.70))
                if left.operation_family and left.operation_family == right.operation_family:
                    signals.append(("operation_family", 0.60))
                if _read_write_pair(left.operation_kind, right.operation_kind):
                    signals.append(("read_write_pair", 0.35))
                if tags:
                    signals.append(("tag_match", tags))
                if keywords:
                    signals.append(("keyword_match", keywords))
                if signals:
                    weight = min(1.0, sum(value for _, value in signals) / 2.5)
                    edges.append(CapabilityEdge(
                        left.uri,
                        right.uri,
                        "capability_similarity",
                        weight,
                        False,
                        tuple(name for name, _ in signals),
                    ))
                for producer, consumer in ((left, right), (right, left)):
                    shared = producer.produces & consumer.consumes
                    if shared:
                        edges.append(CapabilityEdge(
                            producer.uri,
                            consumer.uri,
                            "output_to_input",
                            min(1.0, 0.70 + 0.10 * len(shared)),
                            True,
                            tuple(f"schema:{value}" for value in sorted(shared)),
                        ))
        return tuple(edges)

    @property
    def density(self) -> float:
        count = len(self.resources)
        return len(self.edges) / max(count * (count - 1), 1)

    def _eligible(self, uri: str, policy: ToolDisclosurePolicy) -> bool:
        resource = self.by_uri[uri]
        return policy.allow_unsafe or resource.side_effect_class != SideEffectClass.DESTRUCTIVE

    def disclose(
        self,
        root_uris: Sequence[str],
        policy: ToolDisclosurePolicy,
        *,
        root_confidence: float = 1.0,
    ) -> DisclosureTrace:
        """Expand direct roots under the requested profile and source budgets."""

        roots = tuple(uri for uri in dict.fromkeys(root_uris) if uri in self.by_uri)
        resolved = policy.mode
        if resolved == DisclosureMode.ADAPTIVE:
            if root_confidence >= 0.90 and self.density < 0.30:
                resolved = DisclosureMode.LOCAL
            elif root_confidence >= 0.60:
                resolved = DisclosureMode.PLANNING
            else:
                resolved = DisclosureMode.BROAD
        selected: list[str] = []
        provenance: list[DisclosureProvenance] = []
        unsafe_suppressed = 0

        def add(uri: str, record: DisclosureProvenance) -> bool:
            nonlocal unsafe_suppressed
            if uri in selected or len(selected) >= policy.max_tools:
                return False
            if not self._eligible(uri, policy):
                unsafe_suppressed += 1
                return False
            selected.append(uri)
            provenance.append(record)
            return True

        for rank, uri in enumerate(roots[: policy.direct_k], start=1):
            add(uri, DisclosureProvenance(uri, "direct", uri, direct_rank=rank))
        if resolved == DisclosureMode.MINIMAL or not selected:
            return DisclosureTrace(policy.mode.value, resolved.value, roots, tuple(selected), tuple(provenance), 0, 0, unsafe_suppressed)

        include_local = resolved in {DisclosureMode.LOCAL, DisclosureMode.PLANNING, DisclosureMode.BROAD}
        include_schema = resolved in {DisclosureMode.PLANNING, DisclosureMode.BROAD}
        considered = expansions = 0
        if include_local:
            local = []
            for root in roots:
                for edge in self.outgoing[root]:
                    if edge.directed:
                        continue
                    considered += 1
                    local.append((edge.weight, edge.target_uri, root, edge))
            local.sort(key=lambda row: (-row[0], row[1]))
            family_used = tag_used = 0
            for _weight, uri, root, edge in local:
                tag_signal = any(value in {"tag_match", "keyword_match"} for value in edge.provenance)
                if tag_signal and tag_used < policy.tag_k:
                    source = "tag_match" if "tag_match" in edge.provenance else "keyword_match"
                    if add(uri, DisclosureProvenance(uri, source, root, edge.edge_type, edge.weight, graph_depth=1)):
                        tag_used += 1
                        expansions += 1
                elif family_used < policy.family_k:
                    source = next((value for value in edge.provenance if value in {"same_category", "same_object", "same_api", "operation_family"}), "same_namespace")
                    if add(uri, DisclosureProvenance(uri, source, root, edge.edge_type, edge.weight, graph_depth=1)):
                        family_used += 1
                        expansions += 1
        if include_schema and len(selected) < policy.max_tools:
            queue = deque((root, root, 0) for root in roots)
            seen = set(roots)
            successor_used = predecessor_used = 0
            while queue and len(selected) < policy.max_tools:
                current, root, depth = queue.popleft()
                if depth >= policy.schema_depth:
                    continue
                candidates = []
                for edge in self.outgoing[current]:
                    if edge.directed:
                        candidates.append(("schema_successor", edge.target_uri, edge))
                for edge in self.incoming[current]:
                    if edge.directed:
                        candidates.append(("schema_predecessor", edge.source_uri, edge))
                candidates.sort(key=lambda row: (-row[2].weight, row[1]))
                for source, uri, edge in candidates:
                    considered += 1
                    if uri in seen:
                        continue
                    if source == "schema_successor" and successor_used >= policy.schema_successor_k:
                        continue
                    if source == "schema_predecessor" and predecessor_used >= policy.schema_predecessor_k:
                        continue
                    seen.add(uri)
                    if add(uri, DisclosureProvenance(uri, source, root, edge.edge_type, edge.weight, graph_depth=depth + 1)):
                        expansions += 1
                        successor_used += int(source == "schema_successor")
                        predecessor_used += int(source == "schema_predecessor")
                    queue.append((uri, root, depth + 1))
        return DisclosureTrace(
            policy.mode.value,
            resolved.value,
            roots,
            tuple(selected),
            tuple(provenance),
            expansions,
            considered,
            unsafe_suppressed,
        )


def disclosure_policy_for_profile(mode: DisclosureMode | str, *, max_tools: int = 10) -> ToolDisclosurePolicy:
    """Return reproducible fixed budgets for one SDK disclosure profile."""

    mode = DisclosureMode(mode)
    if mode == DisclosureMode.MINIMAL:
        return ToolDisclosurePolicy(mode, direct_k=1, family_k=0, tag_k=0, schema_successor_k=0, schema_predecessor_k=0, speculative_k=0, max_tools=max_tools, schema_depth=0)
    if mode == DisclosureMode.LOCAL:
        return ToolDisclosurePolicy(mode, direct_k=1, family_k=3, tag_k=2, schema_successor_k=0, schema_predecessor_k=0, speculative_k=0, max_tools=max_tools, schema_depth=0)
    if mode == DisclosureMode.BROAD:
        return ToolDisclosurePolicy(mode, direct_k=2, family_k=5, tag_k=4, schema_successor_k=8, schema_predecessor_k=4, speculative_k=3, max_tools=max_tools, schema_depth=4)
    return ToolDisclosurePolicy(mode, max_tools=max_tools)
