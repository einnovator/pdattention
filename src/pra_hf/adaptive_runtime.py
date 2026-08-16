"""Adaptive effort selection and monotonic retry control for PRA inference.

The controller consumes only runtime-observable features.  Ground-truth labels
may be used by offline evaluators, but are deliberately rejected by the public
feature constructor so they cannot leak into serving decisions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


_ORACLE_MARKERS = ("oracle", "gold", "ground_truth", "answer_correct", "evidence_recall")


@dataclass(frozen=True)
class EffortProfile:
    """One validated discrete point in the PRA control-vector space.

    The fields instantiate ``(Q,F,R,K,H,B,theta,L,G,M)``.  Conceptual parent and
    physical native-K/V budgets are separate because broad search does not have
    to imply broad disclosure.
    """

    name: str
    level: int
    facet_policy: str
    facet_count: int
    retained_roots: int
    neighbors_per_expansion: int
    hop_depth: int
    conceptual_budget: int
    native_kv_budget: int
    routing_threshold: float
    search_layers: tuple[int, ...]
    consumer_layers: tuple[int, ...]
    granularity_tokens: int
    materialization_policy: str
    query_region_policy: str = "head"

    def __post_init__(self) -> None:
        if not self.name or self.level < 0:
            raise ValueError("An effort profile requires a name and non-negative level.")
        positive = (
            self.facet_count,
            self.retained_roots,
            self.neighbors_per_expansion,
            self.conceptual_budget,
            self.native_kv_budget,
            self.granularity_tokens,
        )
        if any(value <= 0 for value in positive) or self.hop_depth < 0:
            raise ValueError("Effort counts and budgets must be positive; hop depth is non-negative.")
        if not 0.0 <= self.routing_threshold <= 1.0:
            raise ValueError("routing_threshold must lie in [0, 1].")
        if not self.search_layers or not self.consumer_layers:
            raise ValueError("Effort profiles require search and consumer layers.")
        if not self.query_region_policy:
            raise ValueError("Effort profiles require a query-region policy.")

    @property
    def control_vector(self) -> dict[str, Any]:
        """Return the public ``Q/F/R/K/H/B/theta/L/G/M`` representation."""

        return {
            "Q": {"policy": self.query_region_policy},
            "F": {"policy": self.facet_policy, "count": self.facet_count},
            "R": self.retained_roots,
            "K": self.neighbors_per_expansion,
            "H": self.hop_depth,
            "B": {
                "conceptual_parents": self.conceptual_budget,
                "native_kv_tokens": self.native_kv_budget,
            },
            "theta": self.routing_threshold,
            "L": {
                "search": list(self.search_layers),
                "consumer": list(self.consumer_layers),
            },
            "G": self.granularity_tokens,
            "M": self.materialization_policy,
        }

    @property
    def cost_units(self) -> float:
        """Stable abstract effort used for controller training and accounting."""

        search = self.retained_roots * max(self.neighbors_per_expansion, 1) * (self.hop_depth + 1)
        return float(search + self.conceptual_budget + self.native_kv_budget / 64.0)

    def dominates(self, cheaper: "EffortProfile") -> bool:
        """Whether this profile permits at least every monotonic resource below it."""

        return (
            self.level >= cheaper.level
            and self.facet_count >= cheaper.facet_count
            and self.retained_roots >= cheaper.retained_roots
            and self.neighbors_per_expansion >= cheaper.neighbors_per_expansion
            and self.hop_depth >= cheaper.hop_depth
            and self.conceptual_budget >= cheaper.conceptual_budget
            and self.native_kv_budget >= cheaper.native_kv_budget
            and self.routing_threshold <= cheaper.routing_threshold
            and set(self.search_layers).issuperset(cheaper.search_layers)
            and set(self.consumer_layers).issuperset(cheaper.consumer_layers)
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["control_vector"] = self.control_vector
        value["cost_units"] = self.cost_units
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EffortProfile":
        fields = cls.__dataclass_fields__
        kwargs = {name: value[name] for name in fields if name in value}
        kwargs["search_layers"] = tuple(kwargs["search_layers"])
        kwargs["consumer_layers"] = tuple(kwargs["consumer_layers"])
        return cls(**kwargs)


def validate_effort_ladder(profiles: Sequence[EffortProfile]) -> tuple[EffortProfile, ...]:
    """Sort and verify that every escalation is resource-monotonic."""

    ordered = tuple(sorted(profiles, key=lambda profile: profile.level))
    if not ordered or len({profile.name for profile in ordered}) != len(ordered):
        raise ValueError("The effort ladder must contain uniquely named profiles.")
    if len({profile.level for profile in ordered}) != len(ordered):
        raise ValueError("Effort levels must be unique.")
    for cheaper, broader in zip(ordered, ordered[1:]):
        if not broader.dominates(cheaper):
            raise ValueError(f"Effort escalation {cheaper.name}->{broader.name} is not monotonic.")
    return ordered


def save_effort_profiles(path: Path, profiles: Sequence[EffortProfile]) -> None:
    ordered = validate_effort_ladder(profiles)
    path.write_text(json.dumps([profile.to_dict() for profile in ordered], indent=2), encoding="utf-8")


def load_effort_profiles(path: Path) -> tuple[EffortProfile, ...]:
    return validate_effort_ladder(
        [EffortProfile.from_dict(value) for value in json.loads(path.read_text(encoding="utf-8"))]
    )


def default_effort_profiles() -> tuple[EffortProfile, ...]:
    """Return the initial low/medium/high ladder used by the SDK and study."""

    return validate_effort_ladder(
        (
            EffortProfile(
                "E0_low", 0, "last_span", 1, 1, 2, 0, 2, 256, 0.65,
                (27,), (27,), 256, "evidence_centered", "head",
            ),
            EffortProfile(
                "E1_medium", 1, "multi_span", 2, 2, 4, 1, 4, 512, 0.45,
                (24, 27), (26, 27), 128, "evidence_centered", "structural",
            ),
            EffortProfile(
                "E2_high", 2, "multi_scale", 4, 4, 8, 3, 8, 1024, 0.25,
                (20, 24, 27), (24, 25, 26, 27), 64, "threshold_gated_radius", "multi_region",
            ),
        )
    )


@dataclass(frozen=True)
class ControllerFeatures:
    """Cheap query, routing, output, and memory observations for effort choice."""

    query_length: float = 0.0
    prompt_length: float = 0.0
    query_region_count: float = 1.0
    query_region_confidence: float = 0.0
    query_region_score_gap: float = 0.0
    query_region_disagreement: float = 0.0
    query_region_expansion: float = 0.0
    sentence_count: float = 0.0
    entity_density: float = 0.0
    relation_density: float = 0.0
    question_type_id: float = 0.0
    facet_disagreement: float = 0.0
    top_root_score: float = 0.0
    root_score_gap: float = 0.0
    topk_score_gap: float = 0.0
    routing_entropy: float = 0.0
    competitive_roots: float = 0.0
    facet_agreement: float = 1.0
    frontier_mean: float = 0.0
    frontier_std: float = 0.0
    newly_discovered_memory: float = 0.0
    path_convergence: float = 0.0
    output_entropy_mean: float = 0.0
    output_entropy_max: float = 0.0
    answer_log_probability: float = 0.0
    answer_margin: float = 0.0
    answer_length: float = 0.0
    retry_consistency: float = 1.0
    confidence_delta: float = 0.0
    selected_source_fraction: float = 0.0
    active_native_kv: float = 0.0
    memory_attention_ratio: float = 0.0
    evidence_density_proxy: float = 0.0
    residual_effect: float = 0.0
    attempt: float = 0.0

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__)

    @classmethod
    def from_runtime_mapping(cls, values: Mapping[str, Any]) -> "ControllerFeatures":
        """Construct features while rejecting accidental evaluator-label leakage."""

        lowered = {str(key).lower() for key in values}
        leaked = sorted(
            key for key in lowered if any(marker in key for marker in _ORACLE_MARKERS)
        )
        if leaked:
            raise ValueError(f"Evaluator-only fields cannot enter the controller: {leaked}")
        kwargs = {
            name: float(values.get(name, 0.0))
            for name in cls.names()
        }
        return cls(**kwargs)

    def vector(self, names: Sequence[str] | None = None) -> torch.Tensor:
        selected = tuple(names or self.names())
        unknown = set(selected) - set(self.names())
        if unknown:
            raise ValueError(f"Unknown controller features: {sorted(unknown)}")
        return torch.tensor([float(getattr(self, name)) for name in selected], dtype=torch.float64)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.names()}


def token_entropy(values: torch.Tensor, *, logits: bool = True) -> torch.Tensor:
    """Return entropy over the last (vocabulary) axis for any leading shape."""

    probabilities = torch.softmax(values.float(), dim=-1) if logits else values.float()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def semantic_consistency(left: str, right: str) -> float:
    """Cheap normalized token-F1 proxy for retry answer consistency."""

    left_tokens = _TOKEN.findall(left.lower())
    right_tokens = _TOKEN.findall(right.lower())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    remaining = list(right_tokens)
    overlap = 0
    for token in left_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-12)


@dataclass(frozen=True)
class LinearEffortController:
    """Dependency-free standardized ridge classifier over discrete profiles."""

    profile_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]

    @classmethod
    def fit(
        cls,
        features: Sequence[ControllerFeatures],
        targets: Sequence[str],
        profile_names: Sequence[str],
        *,
        feature_names: Sequence[str] | None = None,
        ridge: float = 1e-3,
    ) -> "LinearEffortController":
        if len(features) != len(targets) or not features:
            raise ValueError("Controller training requires aligned nonempty examples and targets.")
        names = tuple(feature_names or ControllerFeatures.names())
        profiles = tuple(profile_names)
        target_index = {name: index for index, name in enumerate(profiles)}
        if set(targets) - set(profiles):
            raise ValueError("A controller target is not present in the effort ladder.")
        matrix = torch.stack([feature.vector(names) for feature in features])
        mean = matrix.mean(dim=0)
        scale = matrix.std(dim=0, unbiased=False).clamp_min(1e-8)
        standardized = (matrix - mean) / scale
        design = torch.cat([standardized, torch.ones(len(features), 1, dtype=torch.float64)], dim=1)
        target = torch.zeros(len(features), len(profiles), dtype=torch.float64)
        for row, name in enumerate(targets):
            target[row, target_index[name]] = 1.0
        identity = torch.eye(design.shape[1], dtype=torch.float64)
        identity[-1, -1] = 0.0
        solution = torch.linalg.solve(design.T @ design + ridge * identity, design.T @ target)
        return cls(
            profiles,
            names,
            tuple(mean.tolist()),
            tuple(scale.tolist()),
            tuple(tuple(row) for row in solution.tolist()),
        )

    def probabilities(self, features: ControllerFeatures) -> dict[str, float]:
        vector = features.vector(self.feature_names)
        mean = torch.tensor(self.mean, dtype=torch.float64)
        scale = torch.tensor(self.scale, dtype=torch.float64)
        design = torch.cat([(vector - mean) / scale, torch.ones(1, dtype=torch.float64)])
        weights = torch.tensor(self.weights, dtype=torch.float64)
        probabilities = torch.softmax(design @ weights, dim=0)
        return {name: float(probabilities[index]) for index, name in enumerate(self.profile_names)}

    def choose(self, features: ControllerFeatures) -> str:
        probabilities = self.probabilities(features)
        return max(self.profile_names, key=lambda name: (probabilities[name], -self.profile_names.index(name)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinearEffortController":
        return cls(
            tuple(value["profile_names"]),
            tuple(value["feature_names"]),
            tuple(value["mean"]),
            tuple(value["scale"]),
            tuple(tuple(row) for row in value["weights"]),
        )


@dataclass(frozen=True)
class HandRuleController:
    """Transparent E0/E1/E2 rule frozen from validation diagnostics."""

    medium_entropy: float
    high_entropy: float
    medium_root_gap: float
    high_root_gap: float

    def choose(self, features: ControllerFeatures, profile_names: Sequence[str]) -> str:
        if len(profile_names) != 3:
            raise ValueError("The initial hand rule expects exactly three effort profiles.")
        if (
            features.routing_entropy >= self.high_entropy
            or features.root_score_gap <= self.high_root_gap
            or features.facet_disagreement >= 0.65
        ):
            return profile_names[2]
        if (
            features.routing_entropy >= self.medium_entropy
            or features.root_score_gap <= self.medium_root_gap
            or features.facet_disagreement >= 0.35
        ):
            return profile_names[1]
        return profile_names[0]


@dataclass(frozen=True)
class StopPolicy:
    """Observable confidence gates for stopping or escalating a retry."""

    max_incorrect_probability: float = 0.35
    max_routing_entropy: float = 0.72
    min_answer_margin: float = 0.0
    min_retry_consistency: float = 0.45
    require_path_convergence: float = 0.0

    def evaluate(
        self,
        features: ControllerFeatures,
        incorrect_probability: float,
        *,
        has_previous_answer: bool,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []
        if incorrect_probability > self.max_incorrect_probability:
            reasons.append("predicted_error")
        if features.routing_entropy > self.max_routing_entropy:
            reasons.append("routing_entropy")
        if features.answer_margin < self.min_answer_margin:
            reasons.append("answer_margin")
        if has_previous_answer and features.retry_consistency < self.min_retry_consistency:
            reasons.append("retry_instability")
        if features.path_convergence < self.require_path_convergence:
            reasons.append("path_not_converged")
        return not reasons, tuple(reasons)


@dataclass
class AttemptResult:
    """Observable result returned by one search/materialize/generate attempt."""

    answer: str
    features: ControllerFeatures
    incorrect_probability: float
    search_seconds: float
    materialization_seconds: float
    generation_seconds: float
    active_native_kv: int
    selected_parents: tuple[int, ...] = ()
    reusable_state: Any = None
    reused_search_items: int = 0
    reused_kv_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return self.search_seconds + self.materialization_seconds + self.generation_seconds


@dataclass(frozen=True)
class RetryTrace:
    """Public control trace; contains decisions and metrics, never hidden reasoning."""

    attempt: int
    effort: str
    control_vector: Mapping[str, Any]
    incorrect_probability: float
    routing_entropy: float
    answer_margin: float
    retry_consistency: float
    active_native_kv: int
    selected_parent_count: int
    reused_search_items: int
    reused_kv_tokens: int
    search_seconds: float
    materialization_seconds: float
    generation_seconds: float
    escalation_reasons: tuple[str, ...]
    stop_reason: str
    query_region_policy: str = "head"
    query_start: int | None = None
    query_end: int | None = None
    query_region_count: int = 0
    query_region_confidence: float = 0.0
    query_region_method: str = "unreported"
    query_region_expansion: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["control_vector"] = dict(self.control_vector)
        return value


@dataclass(frozen=True)
class RetryResult:
    answer: str
    final_effort: str
    attempts: int
    stopped: bool
    traces: tuple[RetryTrace, ...]
    result: AttemptResult


class AdaptiveRetryAgent:
    """Execute a bounded, monotonic, state-reusing PRA effort ladder."""

    def __init__(
        self,
        profiles: Sequence[EffortProfile],
        stop_policy: StopPolicy,
        *,
        max_retries: int = 2,
        max_search_budget: int | None = None,
        max_active_kv: int | None = None,
        latency_budget_seconds: float | None = None,
    ) -> None:
        self.profiles = validate_effort_ladder(profiles)
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        self.stop_policy = stop_policy
        self.max_retries = max_retries
        self.max_search_budget = max_search_budget
        self.max_active_kv = max_active_kv
        self.latency_budget_seconds = latency_budget_seconds

    def admissible_profiles(self) -> tuple[EffortProfile, ...]:
        values = []
        for profile in self.profiles:
            if self.max_search_budget is not None and profile.conceptual_budget > self.max_search_budget:
                continue
            if self.max_active_kv is not None and profile.native_kv_budget > self.max_active_kv:
                continue
            values.append(profile)
        if not values:
            raise ValueError("No effort profile fits the configured runtime budgets.")
        return tuple(values)

    def run(
        self,
        executor: Callable[[EffortProfile, AttemptResult | None], AttemptResult],
        initial_features: ControllerFeatures,
        *,
        controller: LinearEffortController | HandRuleController | None = None,
        mode: str = "auto",
        effort: str = "low",
        retry_with_more_effort: bool = False,
    ) -> RetryResult:
        """Run manual or automatic control while passing prior state to retries."""

        profiles = self.admissible_profiles()
        names = tuple(profile.name for profile in profiles)
        if mode not in {"auto", "manual"}:
            raise ValueError("mode must be 'auto' or 'manual'.")
        if mode == "manual":
            aliases = {"low": 0, "medium": min(1, len(profiles) - 1), "high": len(profiles) - 1}
            index = aliases.get(effort, names.index(effort) if effort in names else -1)
            if index < 0:
                raise ValueError(f"Unknown effort profile: {effort}")
        elif isinstance(controller, LinearEffortController):
            chosen = controller.choose(initial_features)
            index = names.index(chosen)
        elif isinstance(controller, HandRuleController):
            index = names.index(controller.choose(initial_features, names))
        else:
            index = 0

        prior: AttemptResult | None = None
        traces: list[RetryTrace] = []
        elapsed = 0.0
        for attempt in range(self.max_retries + 1):
            profile = profiles[index]
            current = executor(profile, prior)
            elapsed += current.total_seconds
            consistency = (
                semantic_consistency(prior.answer, current.answer) if prior is not None else 1.0
            )
            features = ControllerFeatures(
                **{
                    **current.features.to_dict(),
                    "retry_consistency": consistency,
                    "attempt": float(attempt),
                    "active_native_kv": float(current.active_native_kv),
                }
            )
            stop, reasons = self.stop_policy.evaluate(
                features,
                current.incorrect_probability,
                has_previous_answer=prior is not None,
            )
            budget_stop = False
            stop_reason = "confidence"
            if self.latency_budget_seconds is not None and elapsed >= self.latency_budget_seconds:
                stop, budget_stop, stop_reason = True, True, "latency_budget"
            elif attempt >= self.max_retries:
                stop, budget_stop, stop_reason = True, True, "max_retries"
            elif index >= len(profiles) - 1:
                stop, budget_stop, stop_reason = True, True, "max_effort"
            elif mode == "manual" and not retry_with_more_effort:
                stop, stop_reason = True, "manual_complete"
            trace = RetryTrace(
                attempt=attempt,
                effort=profile.name,
                control_vector=profile.control_vector,
                incorrect_probability=float(current.incorrect_probability),
                routing_entropy=features.routing_entropy,
                answer_margin=features.answer_margin,
                retry_consistency=consistency,
                active_native_kv=current.active_native_kv,
                selected_parent_count=len(current.selected_parents),
                reused_search_items=current.reused_search_items,
                reused_kv_tokens=current.reused_kv_tokens,
                search_seconds=current.search_seconds,
                materialization_seconds=current.materialization_seconds,
                generation_seconds=current.generation_seconds,
                escalation_reasons=reasons if not stop else (),
                stop_reason=stop_reason if stop else "",
                query_region_policy=profile.query_region_policy,
                query_start=(
                    int(current.metadata["query_spans"][0][0])
                    if current.metadata.get("query_spans")
                    else None
                ),
                query_end=(
                    int(current.metadata["query_spans"][0][1])
                    if current.metadata.get("query_spans")
                    else None
                ),
                query_region_count=len(current.metadata.get("query_spans", ())),
                query_region_confidence=float(current.metadata.get("query_region_confidence", 0.0)),
                query_region_method=str(current.metadata.get("query_region_method", "unreported")),
                query_region_expansion=int(current.metadata.get("query_region_expansion", 0)),
            )
            traces.append(trace)
            current.features = features
            if stop:
                return RetryResult(
                    current.answer,
                    profile.name,
                    len(traces),
                    not budget_stop or stop_reason in {"max_effort", "max_retries"},
                    tuple(traces),
                    current,
                )
            prior = current
            index += 1
        raise AssertionError("Retry loop must terminate within its configured bound.")


def calibration_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> dict[str, float]:
    """Return dependency-free Brier, ECE, AUROC, and average precision."""

    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Calibration inputs must be aligned and nonempty.")
    pairs = [(min(max(float(p), 0.0), 1.0), int(y)) for p, y in zip(probabilities, labels)]
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [(p, y) for p, y in pairs if low <= p < high or (index == bins - 1 and p == 1.0)]
        if bucket:
            confidence = sum(p for p, _ in bucket) / len(bucket)
            frequency = sum(y for _, y in bucket) / len(bucket)
            ece += len(bucket) / len(pairs) * abs(confidence - frequency)
    positives = [p for p, y in pairs if y]
    negatives = [p for p, y in pairs if not y]
    if positives and negatives:
        wins = sum((left > right) + 0.5 * (left == right) for left in positives for right in negatives)
        auroc = wins / (len(positives) * len(negatives))
    else:
        auroc = math.nan
    ranked = sorted(pairs, key=lambda item: -item[0])
    positive_total = sum(y for _, y in ranked)
    correct = 0
    precision_sum = 0.0
    for rank, (_probability, label) in enumerate(ranked, start=1):
        if label:
            correct += 1
            precision_sum += correct / rank
    average_precision = precision_sum / positive_total if positive_total else math.nan
    return {"brier": brier, "ece": ece, "auroc": auroc, "auprc": average_precision}


def risk_coverage_curve(
    probabilities: Sequence[float], labels: Sequence[int]
) -> list[dict[str, float]]:
    """Selective risk when retaining increasingly uncertain examples."""

    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rows = []
    for count in range(1, len(ordered) + 1):
        accepted = ordered[:count]
        rows.append(
            {
                "coverage": count / len(ordered),
                "risk": sum(label for _, label in accepted) / count,
                "selective_accuracy": 1.0 - sum(label for _, label in accepted) / count,
                "threshold": float(accepted[-1][0]),
            }
        )
    return rows
