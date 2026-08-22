"""Adaptive retrieval-representation control for PRA.

Paper 3.5 treats the representation used to discover memory as a controller
action.  Root discovery and successor traversal deliberately have separate
method spaces because evidence admitted at hop ``t`` changes the next query
state.  This module defines that serving-safe contract; evaluator-only oracle
selection remains in experiment code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .factorized_control import FactorizedEffortAction


ROOT_METHODS = ("semantic", "exact", "bm25", "approx", "hybrid")
SUCCESSOR_METHODS = (
    "native_semantic",
    "exact_new_address",
    "bm25_state",
    "approximate_new_address",
    "hybrid_state",
)
ROOT_METHOD_ALIASES = {"gist": "semantic", **{name: name for name in ROOT_METHODS}}
MATCHED_SUCCESSOR = {
    "semantic": "native_semantic",
    "exact": "exact_new_address",
    "bm25": "bm25_state",
    "approx": "approximate_new_address",
    "hybrid": "hybrid_state",
}
_FORBIDDEN_FEATURE_MARKERS = (
    "dataset",
    "gold",
    "oracle",
    "answer_correct",
    "evidence_recall",
    "path_recovery",
    "target_method",
)


@dataclass(frozen=True)
class SearchMethodActionSpec:
    """Validated Paper 2.6 discovery action contract.

    ``source_sha256`` pins the exact imported JSON.  Canonical root names use
    ``semantic`` even though Paper 2.6 calls the same mean-gist channel
    ``gist``.  The alias is retained in ``source_root_methods`` for audit.
    """

    schema_version: str
    root_methods: tuple[str, ...]
    successor_methods: tuple[str, ...]
    source_root_methods: tuple[str, ...]
    confidence_signals: tuple[str, ...]
    cost_metrics: tuple[str, ...]
    source_sha256: str
    materialization_performed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveSearchAction:
    """One method-aware interpretation/search/admission decision.

    The nested factorized action carries ``(F,R,K,H,B_search,B_KV)``.  Method
    fields are independent: a semantic root may expose a rare address that is
    then followed exactly, or a lexical root may transition to native semantic
    successor search.  Physical K/V admission is still bounded separately.
    """

    root_method: str
    successor_method: str
    effort: FactorizedEffortAction
    query_region_policy: str = "structural"

    def __post_init__(self) -> None:
        if self.root_method not in ROOT_METHODS:
            raise ValueError(f"Unsupported root_method={self.root_method!r}.")
        if self.successor_method not in SUCCESSOR_METHODS:
            raise ValueError(f"Unsupported successor_method={self.successor_method!r}.")
        if not self.query_region_policy:
            raise ValueError("query_region_policy must be nonempty.")

    @property
    def identifier(self) -> str:
        return f"Sr-{self.root_method}_Ss-{self.successor_method}_{self.effort.identifier}"

    @property
    def control_vector(self) -> dict[str, Any]:
        return {
            "Q_regions": self.query_region_policy,
            "S_root": self.root_method,
            "S_succ": self.successor_method,
            "F": self.effort.facets,
            "R": self.effort.roots,
            "K": self.effort.neighbors,
            "H": self.effort.hops,
            "B_search": self.effort.search_budget,
            "B_KV": self.effort.kv_budget,
        }


@dataclass(frozen=True)
class SearchTransition:
    """Auditable root-to-successor method transition for one routing hop."""

    root_method: str
    successor_method: str
    hop: int
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if self.root_method not in ROOT_METHODS or self.successor_method not in SUCCESSOR_METHODS:
            raise ValueError("Transition contains an unsupported search method.")
        if self.hop < 1 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Transition hop/confidence is outside its valid range.")


def load_search_method_action_spec(path: Path) -> SearchMethodActionSpec:
    """Load and strictly validate the deterministic Paper 2.6 handoff."""

    raw = path.read_bytes()
    payload = json.loads(raw)
    source_roots = tuple(sorted(payload.get("root_search_methods", {})))
    roots = tuple(sorted(ROOT_METHOD_ALIASES.get(name, name) for name in source_roots))
    successors = tuple(sorted(payload.get("successor_search_methods", {})))
    if set(roots) != set(ROOT_METHODS):
        raise ValueError(f"Paper 2.6 root methods do not match the adaptive contract: {roots}.")
    if set(successors) != set(SUCCESSOR_METHODS):
        raise ValueError(
            f"Paper 2.6 successor methods do not match the adaptive contract: {successors}."
        )

    entries = [
        *payload["root_search_methods"].values(),
        *payload["successor_search_methods"].values(),
    ]
    confidence = tuple(sorted({item for entry in entries for item in entry["confidence_signals"]}))
    costs = tuple(sorted({item for entry in entries for item in entry["cost_metrics"]}))
    return SearchMethodActionSpec(
        schema_version=str(payload.get("schema_version", "")),
        root_methods=tuple(ROOT_METHODS),
        successor_methods=tuple(SUCCESSOR_METHODS),
        source_root_methods=source_roots,
        confidence_signals=confidence,
        cost_metrics=costs,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        materialization_performed=bool(payload.get("materialization_performed", False)),
    )


def validate_method_feature_names(names: Sequence[str]) -> tuple[str, ...]:
    """Reject dataset identity and evaluator labels from a controller input."""

    normalized = tuple(str(name) for name in names)
    leaked = sorted(
        name
        for name in normalized
        if any(marker in name.lower() for marker in _FORBIDDEN_FEATURE_MARKERS)
    )
    if leaked:
        raise ValueError(f"Deployment-unsafe method features: {leaked}")
    return normalized


def select_method_oracle(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_fields: Sequence[str] = ("recall", "precision", "mrr"),
    cost_fields: Sequence[str] = ("comparisons", "token_span_operations", "latency_ms"),
) -> Mapping[str, Any]:
    """Select a deterministic evaluator-side upper bound from measured rows."""

    if not rows:
        raise ValueError("Method oracle requires at least one measured row.")
    return min(
        rows,
        key=lambda row: (
            *(-float(row.get(field, 0.0)) for field in quality_fields),
            *(float(row.get(field, 0.0)) for field in cost_fields),
            str(row.get("root_method", row.get("channel", ""))),
            str(row.get("successor_method", row.get("successor_channel", ""))),
        ),
    )


def useful_address_probability(observables: Mapping[str, float]) -> float:
    """Return an interpretable lexical-successor gate from runtime observables.

    This fixed score is a non-learned cascade baseline, not a calibrated model.
    Rarity, uniqueness and semantic consistency increase confidence; many
    aliases sharing the address and a poor rank reduce it.
    """

    rarity = float(observables.get("rarity", observables.get("idf", 0.0)))
    candidates = max(float(observables.get("candidate_count", 1.0)), 1.0)
    rank = max(float(observables.get("successor_rank", 1.0)), 1.0)
    semantic = float(observables.get("semantic_consistency", 0.0))
    approximate = float(observables.get("approximate_confidence", 0.0))
    linear = -1.5 + 0.55 * rarity - 0.75 * math.log(candidates) - 0.18 * (rank - 1)
    linear += 0.8 * semantic + 0.5 * approximate
    return 1.0 / (1.0 + math.exp(-max(min(linear, 30.0), -30.0)))


def choose_successor_cascade(observables: Mapping[str, float], *, threshold: float = 0.5) -> str:
    """Use a lexical successor only for a sufficiently useful exposed address."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    if useful_address_probability(observables) >= threshold:
        return "exact_new_address"
    if float(observables.get("semantic_score_gap", 0.0)) < 0.05:
        return "approximate_new_address"
    return "native_semantic"


def method_retry_action(before: AdaptiveSearchAction, after: AdaptiveSearchAction) -> str:
    """Classify the cheapest corrective delta without hiding compound changes."""

    changed = []
    if before.root_method != after.root_method:
        changed.append("change_root_method")
    if before.successor_method != after.successor_method:
        changed.append("change_successor_method")
    for name in ("facets", "roots", "neighbors", "hops", "search_budget", "kv_budget"):
        if getattr(before.effort, name) != getattr(after.effort, name):
            changed.append(f"change_{name}")
    return changed[0] if len(changed) == 1 else ("no_change" if not changed else "combined_action")


def method_cost_accounting(
    root: Mapping[str, Any], successor: Mapping[str, Any], *, materialized_kv_tokens: int = 0
) -> dict[str, float]:
    """Expose unlike search costs and physical admission without conflating them."""

    if materialized_kv_tokens < 0:
        raise ValueError("materialized_kv_tokens must be non-negative.")
    return {
        "root_comparisons": float(root.get("comparisons", 0.0)),
        "successor_comparisons": float(successor.get("comparisons", 0.0)),
        "index_lookups": float(root.get("index_lookups", 0.0))
        + float(successor.get("index_lookups", 0.0)),
        "token_span_operations": float(root.get("token_span_operations", 0.0))
        + float(successor.get("token_span_operations", 0.0)),
        "root_latency_ms": float(root.get("latency_ms", 0.0)),
        "successor_latency_ms": float(successor.get("latency_ms", 0.0)),
        "index_memory_bytes": max(
            float(root.get("index_memory_bytes", 0.0)),
            float(successor.get("index_memory_bytes", 0.0)),
        ),
        "materialized_kv_tokens": float(materialized_kv_tokens),
    }
