"""Deterministic root persistence and bounded transition competition.

This module regulates *which parent identities* survive routing.  It is
deliberately independent of attention and K/V materialization: callers provide
the ordinary root score vector and frozen transition score vectors, then map
the returned parent indices to native payloads after routing stops.

The API contains no evidence or answer labels.  Labels may therefore evaluate
the returned identities, but cannot influence a routing decision accidentally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch


_ROOT_MODES = {"fixed", "score_drop", "zscore", "seed_agreement"}
_TRANSITION_MODES = {"fixed", "adaptive"}
_GEOMETRIES = {"semantic", "native_rank"}


def deterministic_topk(
    scores: torch.Tensor,
    k: int,
    *,
    excluded: set[int] | None = None,
) -> list[int]:
    """Return descending finite-score indices with index-based tie breaking."""
    excluded = excluded or set()
    candidates = [
        index
        for index in range(scores.numel())
        if index not in excluded and math.isfinite(float(scores[index]))
    ]
    candidates.sort(key=lambda index: (-float(scores[index]), index))
    return candidates[: max(0, k)]


@dataclass(frozen=True)
class RootLockConfig:
    """Choose persistent identities from the full root Top-B ranking.

    ``threshold`` means a raw adjacent score drop for ``score_drop``, a
    full-vector z-score for ``zscore``, and an agreement fraction for
    ``seed_agreement``.  ``fixed_count`` is used only by ``fixed``.
    """

    mode: str
    fixed_count: int = 1
    threshold: float = 0.0
    minimum_locked: int = 1

    def __post_init__(self) -> None:
        if self.mode not in _ROOT_MODES:
            raise ValueError(f"Unsupported root lock mode: {self.mode}")
        if self.fixed_count < 0 or self.minimum_locked < 0:
            raise ValueError("Root lock counts must be non-negative.")
        if self.mode == "seed_agreement" and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Seed-agreement threshold must lie in [0, 1].")


@dataclass(frozen=True)
class TransitionPolicyConfig:
    """Bound transition breadth while leaving transition geometry frozen.

    Adaptive routing chooses Top-1 when both geometries agree on Top-1 and
    concentration exceeds ``high_confidence``; it chooses Top-2 when their
    Top-4 sets overlap and concentration exceeds ``moderate_confidence``;
    otherwise it uses Top-4.  The selected breadth is always capped by the
    remaining parent budget.
    """

    mode: str = "adaptive"
    fixed_k: int = 4
    moderate_confidence: float = 0.35
    high_confidence: float = 0.55

    def __post_init__(self) -> None:
        if self.mode not in _TRANSITION_MODES:
            raise ValueError(f"Unsupported transition mode: {self.mode}")
        if self.fixed_k not in {1, 2, 4}:
            raise ValueError("Fixed transition breadth must be 1, 2, or 4.")
        if not 0.0 <= self.moderate_confidence <= self.high_confidence <= 1.0:
            raise ValueError("Transition confidence thresholds are invalid.")


@dataclass(frozen=True)
class AdaptiveCompetitionConfig:
    """Configure one bounded root-to-transition routing pass."""

    total_budget: int
    root_lock: RootLockConfig
    transition: TransitionPolicyConfig
    transition_geometry: str = "semantic"

    def __post_init__(self) -> None:
        if self.total_budget < 0:
            raise ValueError("Total parent budget must be non-negative.")
        if self.transition_geometry not in _GEOMETRIES:
            raise ValueError(
                f"Unsupported transition geometry: {self.transition_geometry}"
            )


@dataclass(frozen=True)
class TransitionScores:
    """Frozen parent-level scores and measured work for one source parent.

    ``semantic`` and ``native_raw`` have shape ``[parents]``.  Native values
    remain raw pre-RoPE QK reductions; the router uses their per-source rank
    and never applies the saturated sigmoid used in earlier diagnostics.
    """

    semantic: torch.Tensor
    native_raw: torch.Tensor
    semantic_comparisons: int = 0
    native_qk_comparisons: int = 0
    scoring_seconds: float = 0.0


@dataclass(frozen=True)
class TransitionConfidence:
    """Oracle-free uncertainty and semantic/native agreement diagnostics."""

    top1_top4_spread: float
    normalized_entropy: float
    concentration: float
    same_top1: bool
    top4_overlap: float
    top1_rank_distance: int


@dataclass(frozen=True)
class AdaptiveCompetitionResult:
    """Stable routing identities, budgets, score traces, and search costs."""

    root_top_b: tuple[int, ...]
    locked_roots: tuple[int, ...]
    root_confidence: Mapping[str, float]
    propagated: tuple[int, ...]
    selected: tuple[int, ...]
    transition_ks: tuple[int, ...]
    transition_confidences: tuple[TransitionConfidence, ...]
    root_comparisons: int
    semantic_comparisons: int
    native_qk_comparisons: int
    scoring_seconds: float

    @property
    def propagation_budget(self) -> int:
        return len(self.selected) - len(self.locked_roots)


def root_seed_agreement(
    rankings: Sequence[Sequence[int]],
    top_r: int,
) -> dict[int, float]:
    """Return the fraction of independent rankings containing each parent."""
    if not rankings:
        raise ValueError("At least one root ranking is required.")
    counts: dict[int, int] = {}
    for ranking in rankings:
        for parent in set(ranking[: max(0, top_r)]):
            counts[parent] = counts.get(parent, 0) + 1
    return {parent: count / len(rankings) for parent, count in counts.items()}


def lock_root_candidates(
    root_scores: torch.Tensor,
    budget: int,
    config: RootLockConfig,
    *,
    agreement: Mapping[int, float] | None = None,
) -> tuple[list[int], list[int], dict[str, float]]:
    """Compute full root Top-B first, then select an identity-level lock set."""
    if root_scores.ndim != 1:
        raise ValueError("Root scores must have shape [parents].")
    top_b = deterministic_topk(root_scores, budget)
    if not top_b:
        return [], [], {"locked_fraction": 0.0}

    if config.mode == "fixed":
        locked = top_b[: config.fixed_count]
    elif config.mode == "score_drop":
        locked_count = config.minimum_locked
        for offset in range(len(top_b) - 1):
            drop = float(root_scores[top_b[offset]] - root_scores[top_b[offset + 1]])
            if drop >= config.threshold:
                locked_count = offset + 1
                break
        locked = top_b[:locked_count]
    elif config.mode == "zscore":
        finite = root_scores[torch.isfinite(root_scores)].float()
        mean = float(finite.mean())
        std = float(finite.std(unbiased=False))
        std = max(std, 1e-8)
        locked = [
            parent
            for parent in top_b
            if (float(root_scores[parent]) - mean) / std > config.threshold
        ]
    else:
        if agreement is None:
            raise ValueError("Seed-agreement locking requires agreement scores.")
        locked = [
            parent
            for parent in top_b
            if float(agreement.get(parent, 0.0)) >= config.threshold
        ]

    minimum = min(config.minimum_locked, len(top_b))
    if len(locked) < minimum:
        locked = top_b[:minimum]
    locked = locked[:budget]
    top_values = torch.tensor([float(root_scores[parent]) for parent in top_b])
    finite = root_scores[torch.isfinite(root_scores)].float()
    std = max(float(finite.std(unbiased=False)), 1e-8)
    confidence = {
        "top1_top2_drop": (
            float(top_values[0] - top_values[1]) if len(top_b) > 1 else math.inf
        ),
        "top1_topB_spread_z": (
            float((top_values[0] - top_values[-1]) / std) if len(top_b) > 1 else math.inf
        ),
        "locked_fraction": len(locked) / len(top_b),
    }
    return top_b, locked, confidence


def _finite_order(scores: torch.Tensor, excluded: set[int]) -> list[int]:
    return deterministic_topk(scores, scores.numel(), excluded=excluded)


def transition_confidence(
    semantic: torch.Tensor,
    native_raw: torch.Tensor,
    *,
    excluded: set[int] | None = None,
) -> TransitionConfidence:
    """Measure normalized score shape and agreement without oracle identities."""
    if semantic.ndim != 1 or native_raw.ndim != 1 or semantic.shape != native_raw.shape:
        raise ValueError("Transition scores must be aligned [parents] vectors.")
    excluded = excluded or set()
    semantic_order = _finite_order(semantic, excluded)
    native_order = _finite_order(native_raw, excluded)
    if not semantic_order or not native_order:
        return TransitionConfidence(0.0, 1.0, 0.0, False, 0.0, semantic.numel())

    native_finite = torch.tensor(
        [float(native_raw[parent]) for parent in native_order], dtype=torch.float32
    )
    mean = native_finite.mean()
    std = native_finite.std(unbiased=False).clamp_min(1e-8)
    normalized = (native_finite - mean) / std
    top = normalized[: min(4, normalized.numel())]
    probabilities = torch.softmax(top, dim=0)
    entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    )
    normalized_entropy = entropy / math.log(len(top)) if len(top) > 1 else 0.0
    spread = float(top[0] - top[-1]) if len(top) > 1 else math.inf
    semantic_top4, native_top4 = set(semantic_order[:4]), set(native_order[:4])
    overlap = len(semantic_top4 & native_top4) / max(
        len(semantic_top4 | native_top4), 1
    )
    semantic_rank = {parent: rank for rank, parent in enumerate(semantic_order)}
    return TransitionConfidence(
        top1_top4_spread=spread,
        normalized_entropy=normalized_entropy,
        concentration=float(probabilities[0]),
        same_top1=semantic_order[0] == native_order[0],
        top4_overlap=overlap,
        top1_rank_distance=abs(semantic_rank.get(native_order[0], len(semantic_order))),
    )


def adaptive_transition_k(
    confidence: TransitionConfidence,
    config: TransitionPolicyConfig,
) -> int:
    """Choose a deterministic bounded breadth of one, two, or four."""
    if config.mode == "fixed":
        return config.fixed_k
    if confidence.same_top1 and confidence.concentration >= config.high_confidence:
        return 1
    if confidence.top4_overlap > 0.0 and confidence.concentration >= config.moderate_confidence:
        return 2
    return 4


def monotonic_final_selection(
    locked: Sequence[int],
    proposals: Sequence[tuple[int, float]],
    root_order: Sequence[int],
    budget: int,
) -> tuple[list[int], list[int]]:
    """Preserve locked roots, deduplicate proposals, then root-fill to budget."""
    selected = list(dict.fromkeys(locked))[:budget]
    locked_set = set(selected)
    best: dict[int, float] = {}
    for parent, score in proposals:
        if parent not in locked_set and (
            parent not in best or score > best[parent]
        ):
            best[parent] = score
    ranked = sorted(best, key=lambda parent: (-best[parent], parent))
    propagated = []
    for parent in ranked:
        if len(selected) >= budget:
            break
        selected.append(parent)
        propagated.append(parent)
    for parent in root_order:
        if len(selected) >= budget:
            break
        if parent not in selected:
            selected.append(parent)
    if not locked_set.issubset(selected):
        raise AssertionError("A locked root was evicted from final selection.")
    return selected, propagated


class AdaptiveCompetitionRouter:
    """Apply monotonic root locking and one bounded propagation pass.

    ``transition_provider(source_parent)`` supplies both frozen geometries for
    a locked source.  Propagation candidates compete only for non-locked slots;
    unfilled slots fall back to the original root order so every matched run
    materializes exactly the same number of parents whenever enough exist.
    """

    def route(
        self,
        root_scores: torch.Tensor,
        transition_provider: Callable[[int], TransitionScores],
        config: AdaptiveCompetitionConfig,
        *,
        agreement: Mapping[int, float] | None = None,
    ) -> AdaptiveCompetitionResult:
        top_b, locked, root_confidence = lock_root_candidates(
            root_scores,
            config.total_budget,
            config.root_lock,
            agreement=agreement,
        )
        if len(locked) >= config.total_budget:
            return AdaptiveCompetitionResult(
                tuple(top_b), tuple(locked), root_confidence,
                (), tuple(locked), (), (),
                root_scores.numel(), 0, 0, 0.0,
            )

        excluded = set(locked)
        proposals: list[tuple[int, float]] = []
        decisions: list[int] = []
        confidences: list[TransitionConfidence] = []
        semantic_work = native_work = 0
        scoring_seconds = 0.0
        root_rank = {parent: rank + 1 for rank, parent in enumerate(top_b)}
        for source in locked:
            scores = transition_provider(source)
            confidence = transition_confidence(
                scores.semantic, scores.native_raw, excluded=excluded
            )
            breadth = min(
                adaptive_transition_k(confidence, config.transition),
                config.total_budget - len(locked),
            )
            decisions.append(breadth)
            confidences.append(confidence)
            semantic_work += scores.semantic_comparisons
            native_work += scores.native_qk_comparisons
            scoring_seconds += scores.scoring_seconds
            chosen_scores = (
                scores.semantic
                if config.transition_geometry == "semantic"
                else scores.native_raw
            )
            order = deterministic_topk(chosen_scores, breadth, excluded=excluded)
            for transition_rank, parent in enumerate(order, start=1):
                if config.transition_geometry == "semantic":
                    source_affinity = (float(root_scores[source]) + 1.0) / 2.0
                    target_affinity = (float(chosen_scores[parent]) + 1.0) / 2.0
                    proposal_score = source_affinity * target_affinity
                else:
                    # Raw QK magnitude is not calibrated across sources.  Its
                    # deterministic rank retains the frozen geometry without
                    # reintroducing the saturated sigmoid path score.
                    proposal_score = 1.0 / (
                        root_rank.get(source, len(top_b) + 1) * transition_rank
                    )
                proposals.append((parent, proposal_score))

        selected, propagated = monotonic_final_selection(
            locked, proposals, top_b, config.total_budget
        )
        return AdaptiveCompetitionResult(
            tuple(top_b), tuple(locked), root_confidence,
            tuple(propagated), tuple(selected),
            tuple(decisions), tuple(confidences), root_scores.numel(),
            semantic_work, native_work, scoring_seconds,
        )
