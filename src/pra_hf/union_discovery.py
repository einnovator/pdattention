"""Bounded multi-channel candidate palettes for typed tool discovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from pra_hf.agent_resources import AgentResource, SideEffectClass, normalize_text


class ToolDiscoveryMode(str, Enum):
    """Agent-side resolution modes above individual retrieval channels."""

    AUTO = "auto"
    RESOLVE = "resolve"
    TOP_K = "top_k"
    UNION = "union"


class UnionStrategy(str, Enum):
    """Candidate selection strategies evaluated at matched budgets."""

    SINGLE_CHANNEL = "single_channel"
    FUSED_SCORE = "fused_score"
    RAW_UNION = "raw_union"
    DIVERSITY_UNION = "diversity_union"


@dataclass(frozen=True)
class ToolDiscoveryPolicy:
    """Configuration for model-independent candidate-set construction."""

    mode: ToolDiscoveryMode | str = ToolDiscoveryMode.AUTO
    strategy: UnionStrategy | str = UnionStrategy.DIVERSITY_UNION
    max_candidates: int = 6
    min_candidates: int = 1
    lexical: bool = True
    dictionary: bool = True
    tags: bool = True
    embedding: bool = True
    graph: bool = False
    allow_unsafe: bool = False
    preferred_channel: str | None = None
    channels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ToolDiscoveryMode(self.mode))
        object.__setattr__(self, "strategy", UnionStrategy(self.strategy))
        if self.min_candidates < 0 or self.max_candidates <= 0:
            raise ValueError("Candidate budgets must be nonnegative and max_candidates positive.")
        if self.min_candidates > self.max_candidates:
            raise ValueError("min_candidates cannot exceed max_candidates.")

    @property
    def enabled_channels(self) -> tuple[str, ...]:
        if self.channels is not None:
            return tuple(dict.fromkeys(self.channels))
        enabled = []
        for channel, active in (
            ("lexical", self.lexical),
            ("dictionary", self.dictionary),
            ("tags", self.tags),
            ("embedding", self.embedding),
            ("graph", self.graph),
        ):
            if active:
                enabled.append(channel)
        return tuple(enabled)


@dataclass(frozen=True)
class ChannelHit:
    """One channel's evidence for one stable resource identity."""

    channel: str
    rank: int
    score: float


@dataclass(frozen=True)
class CandidateProvenance:
    """All discovery evidence retained for one admitted URI."""

    uri: str
    sources: tuple[ChannelHit, ...]
    admission_source: str
    admission_rank: int


@dataclass(frozen=True)
class CandidateSet:
    """Bounded candidate palette returned to disclosure/materialization."""

    mode: ToolDiscoveryMode
    strategy: UnionStrategy
    candidate_uris: tuple[str, ...]
    provenance: tuple[CandidateProvenance, ...]
    max_candidates: int
    explicit_resolution: bool = False

    def provenance_for(self, uri: str) -> CandidateProvenance:
        return next(row for row in self.provenance if row.uri == uri)


_CHANNEL_ORDER = ("explicit", "lexical", "dictionary", "tags", "embedding", "graph")


def _rankings(
    channel_scores: Mapping[str, Mapping[str, float]],
    eligible: set[str],
) -> dict[str, tuple[tuple[str, float], ...]]:
    rankings = {}
    for channel, scores in channel_scores.items():
        rankings[channel] = tuple(sorted(
            ((uri, float(score)) for uri, score in scores.items() if uri in eligible),
            key=lambda row: (-row[1], row[0]),
        ))
    return rankings


def _all_hits(rankings: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, tuple[ChannelHit, ...]]:
    hits: dict[str, list[ChannelHit]] = {}
    for channel, rows in rankings.items():
        for rank, (uri, score) in enumerate(rows, start=1):
            hits.setdefault(uri, []).append(ChannelHit(channel, rank, score))
    order = {name: index for index, name in enumerate(_CHANNEL_ORDER)}
    return {
        uri: tuple(sorted(values, key=lambda row: (order.get(row.channel, 99), row.rank)))
        for uri, values in hits.items()
    }


def _normalized_fusion(rankings: Mapping[str, Sequence[tuple[str, float]]]) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for rows in rankings.values():
        if not rows:
            continue
        values = [score for _, score in rows]
        low, high = min(values), max(values)
        for uri, score in rows:
            normalized = (score - low) / (high - low) if high > low else float(score > 0)
            fused[uri] = fused.get(uri, 0.0) + normalized
    return sorted(fused.items(), key=lambda row: (-row[1], row[0]))


def agreement_rerank(
    channel_scores: Mapping[str, Mapping[str, float]],
    *,
    candidate_uris: Sequence[str],
    support_depth: int,
    agreement_weight: float,
) -> tuple[str, ...]:
    """Rerank a bounded palette by fused evidence plus channel agreement.

    Agreement is counted only when a candidate appears in a channel's first
    ``support_depth`` results. This avoids treating every item in a dense
    embedding score vector as independent support. The candidate set remains
    unchanged; this helper only orders hypotheses already admitted by another
    bounded policy.
    """

    if support_depth <= 0:
        raise ValueError("support_depth must be positive.")
    if agreement_weight < 0:
        raise ValueError("agreement_weight cannot be negative.")
    candidates = tuple(dict.fromkeys(candidate_uris))
    if not candidates:
        return ()
    eligible = set(candidates)
    rankings = _rankings(channel_scores, eligible | {
        uri for scores in channel_scores.values() for uri in scores
    })
    fused = dict(_normalized_fusion(rankings))
    support = {uri: 0 for uri in candidates}
    for rows in rankings.values():
        for uri, _ in rows[:support_depth]:
            if uri in support:
                support[uri] += 1
    return tuple(sorted(
        candidates,
        key=lambda uri: (
            -(fused.get(uri, 0.0) + agreement_weight * support[uri]),
            -support[uri],
            uri,
        ),
    ))


def _exact_resolution(query: str, resources: Sequence[AgentResource]) -> tuple[str, ...]:
    normalized = normalize_text(query)
    hits = []
    for resource in resources:
        names = (resource.uri, resource.name, *resource.aliases)
        if normalized in {normalize_text(name) for name in names}:
            hits.append(resource.uri)
    return tuple(sorted(hits))


def discover_candidate_set(
    query: str,
    resources: Sequence[AgentResource],
    channel_scores: Mapping[str, Mapping[str, float]],
    policy: ToolDiscoveryPolicy | None = None,
    *,
    explicit_reference_uris: Iterable[str] = (),
) -> CandidateSet:
    """Return a bounded, deduplicated palette with all channel provenance."""

    policy = policy or ToolDiscoveryPolicy()
    by_uri = {resource.uri: resource for resource in resources}
    explicit = tuple(dict.fromkeys(uri for uri in explicit_reference_uris if uri in by_uri))
    exact = _exact_resolution(query, resources)
    resolved = explicit or exact
    if len(resolved) == 1 and policy.mode in {ToolDiscoveryMode.AUTO, ToolDiscoveryMode.RESOLVE}:
        uri = resolved[0]
        hit = ChannelHit("explicit", 1, 1.0)
        return CandidateSet(
            mode=ToolDiscoveryMode.RESOLVE,
            strategy=policy.strategy,
            candidate_uris=(uri,),
            provenance=(CandidateProvenance(uri, (hit,), "explicit", 1),),
            max_candidates=policy.max_candidates,
            explicit_resolution=True,
        )
    eligible = {
        resource.uri
        for resource in resources
        if not resource.revoked
        and (policy.allow_unsafe or resource.side_effect_class != SideEffectClass.DESTRUCTIVE or resource.uri in explicit)
    }
    enabled = set(policy.enabled_channels)
    scores = {channel: values for channel, values in channel_scores.items() if channel in enabled}
    if explicit:
        scores = {"explicit": {uri: 1.0 for uri in explicit}, **scores}
    rankings = _rankings(scores, eligible)
    channel_order = tuple(
        dict.fromkeys((*policy.enabled_channels, *(_CHANNEL_ORDER), *(rankings.keys())))
    )
    hits = _all_hits(rankings)
    selected: list[tuple[str, str]] = []

    def admit(uri: str, source: str) -> None:
        if uri in eligible and uri not in {value for value, _ in selected} and len(selected) < policy.max_candidates:
            selected.append((uri, source))

    strategy = policy.strategy
    if strategy == UnionStrategy.SINGLE_CHANNEL:
        channel = policy.preferred_channel or next((name for name in channel_order if rankings.get(name)), "")
        for uri, _ in rankings.get(channel, ()):
            admit(uri, channel)
    elif strategy == UnionStrategy.FUSED_SCORE:
        for uri, _ in _normalized_fusion(rankings):
            admit(uri, "fused_score")
    elif strategy == UnionStrategy.RAW_UNION:
        pool: dict[str, tuple[int, float, str]] = {}
        for channel, rows in rankings.items():
            for rank, (uri, score) in enumerate(rows[: policy.max_candidates], start=1):
                previous = pool.get(uri)
                candidate = (rank, -score, channel)
                if previous is None or candidate < previous:
                    pool[uri] = candidate
        for uri, (_, _, channel) in sorted(pool.items(), key=lambda row: (row[1], row[0])):
            admit(uri, channel)
    else:
        for channel in channel_order:
            for uri, _ in rankings.get(channel, ()):
                before = len(selected)
                admit(uri, channel)
                if len(selected) > before:
                    break
        depth = 1
        while len(selected) < policy.max_candidates and any(len(rows) > depth for rows in rankings.values()):
            for channel in channel_order:
                rows = rankings.get(channel, ())
                if len(rows) > depth:
                    admit(rows[depth][0], channel)
            depth += 1
        if len(selected) < policy.min_candidates:
            for uri, _ in _normalized_fusion(rankings):
                admit(uri, "fused_fill")
    provenance = tuple(
        CandidateProvenance(uri, hits.get(uri, ()), source, rank)
        for rank, (uri, source) in enumerate(selected, start=1)
    )
    return CandidateSet(
        mode=ToolDiscoveryMode.UNION if policy.mode == ToolDiscoveryMode.AUTO else policy.mode,
        strategy=strategy,
        candidate_uris=tuple(uri for uri, _ in selected),
        provenance=provenance,
        max_candidates=policy.max_candidates,
        explicit_resolution=False,
    )
