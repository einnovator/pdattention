"""Observable channel-selection diagnostics for multi-representational PRA.

This module deliberately keeps deployment-time channel choice separate from
gold evidence geometry.  Experiment code may use gold annotations to explain
outcomes, but :func:`select_observable_channel` rejects those fields so that an
analysis cannot accidentally turn an oracle into a deployable policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


PRIMARY_CHANNELS = (
    "gist",
    "exact",
    "bm25",
    "approx",
    "hybrid",
    "iterative_hybrid",
)

_GOLD_FIELDS = {
    "answer_overlap",
    "chain_depth",
    "evidence_compactness",
    "evidence_documents",
    "evidence_gap",
    "evidence_regions",
    "evidence_tokens",
    "gold_chunk_ids",
    "query_evidence_overlap",
}


def precision_recall(
    selected: Iterable[str], gold: Iterable[str]
) -> tuple[float, float]:
    """Return evidence precision and recall over stable chunk identities."""
    selected_set = set(selected)
    gold_set = set(gold)
    hits = len(selected_set & gold_set)
    return hits / max(len(selected_set), 1), hits / max(len(gold_set), 1)


def oracle_channel(
    recalls: Mapping[str, float], channels: Iterable[str] = PRIMARY_CHANNELS
) -> tuple[str, float]:
    """Return the deterministic per-example channel oracle and its recall."""
    available = [(str(channel), float(recalls[channel])) for channel in channels]
    if not available:
        raise ValueError("oracle_channel requires at least one channel.")
    return max(available, key=lambda item: (item[1], -PRIMARY_CHANNELS.index(item[0])))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """Measure selected-chunk overlap, defining two empty sets as identical."""
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def reciprocal_rank_fusion(
    rankings: Mapping[str, Mapping[str, int]], *, constant: float = 60.0
) -> dict[str, float]:
    """Fuse channel ranks without assuming that raw scores are calibrated."""
    if constant <= 0:
        raise ValueError("RRF constant must be positive.")
    identities = {identity for ranking in rankings.values() for identity in ranking}
    return {
        identity: sum(
            1.0 / (constant + ranking[identity])
            for ranking in rankings.values()
            if identity in ranking
        )
        for identity in identities
    }


def headroom_decomposition(
    oracle: float, best_heldout_fixed: float, validation_selected: float
) -> tuple[float, float]:
    """Separate adaptive opportunity from validation-selection instability."""
    return (
        float(oracle) - float(best_heldout_fixed),
        float(best_heldout_fixed) - float(validation_selected),
    )


def new_address_tokens(
    query_tokens: Iterable[str],
    first_hop_tokens: Iterable[str],
    later_gold_tokens: Iterable[str],
    *,
    minimum_length: int = 3,
) -> set[str]:
    """Find useful lexical addresses exposed after hop zero.

    An address must be absent from the original query, present in an admitted
    first-hop chunk, and present in still-unrecovered gold evidence.  This is an
    analysis of exposure, not a claim that the router used that token causally.
    """
    query = {value for value in query_tokens if len(value) >= minimum_length}
    exposed = {value for value in first_hop_tokens if len(value) >= minimum_length}
    later = {value for value in later_gold_tokens if len(value) >= minimum_length}
    return (exposed & later) - query


def useful_address(
    *, exposed: bool, gold_linked: bool, successor_rank: int | None, rank_limit: int
) -> bool:
    """Require an exposed address to link gold and rank within retry reach."""
    if rank_limit <= 0:
        raise ValueError("rank_limit must be positive.")
    return bool(
        exposed
        and gold_linked
        and successor_rank is not None
        and int(successor_rank) <= rank_limit
    )


def select_observable_channel(features: Mapping[str, float | int | bool]) -> str:
    """Apply a fixed diagnostic channel rule using deployment-visible signals.

    The rule is intentionally simple and frozen.  It uses query rarity and
    channel score/rank disagreement, never evidence labels or answer text.
    """
    leaked = _GOLD_FIELDS.intersection(features)
    if leaked:
        raise ValueError(f"Gold-derived selector features are forbidden: {sorted(leaked)}")
    rare = float(features.get("query_rare_fraction", 0.0))
    exact = float(features.get("exact_top_score", 0.0))
    bm25_gap = float(features.get("bm25_score_gap", 0.0))
    semantic_gap = float(features.get("semantic_score_gap", 0.0))
    disagreement = int(features.get("channel_disagreement", 0))
    new_address = bool(features.get("new_address_observed", False))
    if new_address:
        return "iterative_hybrid"
    if exact >= 0.75 and rare >= 0.15:
        return "exact"
    if bm25_gap >= 0.20 and rare >= 0.08:
        return "bm25"
    if semantic_gap >= 0.15 and exact < 0.40:
        return "gist"
    if disagreement >= 3:
        return "hybrid"
    return "approx"
