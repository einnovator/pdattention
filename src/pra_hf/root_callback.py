"""Single-event root callbacks for adaptive PRA request/reply execution.

The callback contract deliberately stops at ``ROOT_SELECTED``.  It receives a
compact structured state produced by retrieval and returns bounded successor,
retry, or graph-refinement controls.  Evaluator labels are not represented in
the state, so persisted traces can be reused for offline controller training.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

import torch

from .adaptive_search import SUCCESSOR_METHODS


ROOT_CALLBACK_ACTIONS = (
    "continue",
    "change_successor_method",
    "change_successor_k",
    "change_hop_depth",
    "graph_refine",
    "retry_root",
    "stop",
)

ROOT_STATE_FEATURE_NAMES = (
    "root_top1_score",
    "root_score_gap",
    "candidate_entropy",
    "channel_agreement",
    "channel_disagreement",
    "address_count",
    "address_rarity",
    "facet_agreement",
    "root_dispersion",
    "evidence_proxy",
    "searched_fraction",
    "remaining_search_fraction",
    "remaining_kv_fraction",
)


@dataclass(frozen=True)
class RootState:
    """Observable state emitted after root retrieval and before one successor hop.

    ``root_embedding`` is the mean selected routing gist with shape
    ``[routing_width]`` serialized as floats.  Candidate scores and selected IDs
    retain their rank order.  No gold evidence, answer, or dataset identity is
    available to a serving controller.
    """

    example_id: str
    query_features: Mapping[str, float]
    facet_mode: str
    facet_count: int
    root_method: str
    root_ids: tuple[str, ...]
    root_scores: tuple[float, ...]
    root_top1_score: float
    root_score_gap: float
    candidate_entropy: float
    channel_agreement: float
    channel_disagreement: float
    root_embedding: tuple[float, ...]
    new_entities: tuple[str, ...]
    new_addresses: tuple[str, ...]
    address_count: int
    address_rarity: float
    facet_agreement: float
    root_dispersion: float
    evidence_proxy: float
    searched_fraction: float
    remaining_search_budget: int
    total_search_budget: int
    remaining_kv_budget: int
    total_kv_budget: int

    def __post_init__(self) -> None:
        if not self.example_id or self.facet_count <= 0:
            raise ValueError("Root state requires an example ID and positive facet count.")
        if len(self.root_ids) != len(self.root_scores):
            raise ValueError("Selected root IDs and scores must align.")
        if any(not math.isfinite(float(value)) for value in self.root_scores):
            raise ValueError("Selected root scores must be finite.")
        for name in (
            "candidate_entropy",
            "channel_agreement",
            "facet_agreement",
            "evidence_proxy",
            "searched_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.channel_disagreement < 0 or self.root_dispersion < 0:
            raise ValueError("Disagreement and dispersion cannot be negative.")
        if min(
            self.remaining_search_budget,
            self.total_search_budget,
            self.remaining_kv_budget,
            self.total_kv_budget,
        ) < 0:
            raise ValueError("Callback budgets cannot be negative.")
        if self.remaining_search_budget > self.total_search_budget:
            raise ValueError("Remaining search budget exceeds its total.")
        if self.remaining_kv_budget > self.total_kv_budget:
            raise ValueError("Remaining K/V budget exceeds its total.")

    @property
    def remaining_search_fraction(self) -> float:
        return self.remaining_search_budget / max(self.total_search_budget, 1)

    @property
    def remaining_kv_fraction(self) -> float:
        return self.remaining_kv_budget / max(self.total_kv_budget, 1)

    def feature_vector(self, query_feature_names: Sequence[str] = ()) -> tuple[float, ...]:
        """Return a stable deployment-safe vector for small callback models."""

        callback = tuple(float(getattr(self, name)) for name in ROOT_STATE_FEATURE_NAMES)
        query = tuple(float(self.query_features.get(name, 0.0)) for name in query_feature_names)
        return (*query, *callback)

    def audit_dict(self, *, include_embedding: bool = False) -> dict:
        """Return a JSON-safe callback trace, optionally retaining the full gist."""

        values = asdict(self)
        values["query_features"] = dict(self.query_features)
        if not include_embedding:
            values["root_embedding"] = {
                "width": len(self.root_embedding),
                "norm": math.sqrt(sum(value * value for value in self.root_embedding)),
            }
        values["remaining_search_fraction"] = self.remaining_search_fraction
        values["remaining_kv_fraction"] = self.remaining_kv_fraction
        return values


@dataclass(frozen=True)
class RootDecision:
    """Bounded action returned by the single ``ROOT_SELECTED`` callback."""

    action: str = "continue"
    successor_method: str = "native_semantic"
    successor_k: int = 2
    hop_depth: int = 1
    graph_refine: bool = False
    retry_root: bool = False
    search_budget: int = 2
    kv_budget: int = 4
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.action not in ROOT_CALLBACK_ACTIONS:
            raise ValueError(f"Unsupported root callback action={self.action!r}.")
        if self.successor_method not in SUCCESSOR_METHODS:
            raise ValueError(f"Unsupported successor method={self.successor_method!r}.")
        if min(self.successor_k, self.hop_depth, self.search_budget, self.kv_budget) <= 0:
            raise ValueError("Successor, hop, search, and K/V budgets must be positive.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Decision confidence must lie in [0, 1].")
        if self.action == "graph_refine" and not self.graph_refine:
            raise ValueError("A graph_refine action must enable graph refinement.")
        if self.action == "retry_root" and not self.retry_root:
            raise ValueError("A retry_root action must enable root retry.")

    @property
    def label(self) -> str:
        """Return the compact class label used by offline callback training."""

        return (
            f"{self.action}|{self.successor_method}|k{self.successor_k}|h{self.hop_depth}"
            f"|g{int(self.graph_refine)}|r{int(self.retry_root)}"
            f"|s{self.search_budget}|kv{self.kv_budget}"
        )


class AdaptiveController(Protocol):
    """Minimal two-stage controller interface; implementations remain opt-in."""

    def initial_action(self, query_state: Mapping[str, float]):
        """Choose controls needed before root retrieval."""

    def on_root_selected(self, root_state: RootState) -> RootDecision:
        """Choose one bounded post-root decision."""


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RootCallbackResult(Generic[ResultT]):
    """Result and public trace from exactly one root-state callback event."""

    initial_action: Any
    root_state: RootState
    decision: RootDecision
    result: ResultT
    callback_events: int = 1


class RootCallbackExecutor:
    """Dispatch one opt-in callback between root and successor execution.

    The caller owns concrete retrieval operations. ``root_executor`` performs
    the initial action and returns its observable :class:`RootState`;
    ``successor_executor`` applies the callback decision, including optional
    graph refinement.  The executor cannot loop, so retry-style multi-event
    control is impossible through this API.
    """

    def __init__(self, controller: AdaptiveController) -> None:
        self.controller = controller

    def run(
        self,
        query_state: Mapping[str, float],
        root_executor: Callable[[Any], RootState],
        successor_executor: Callable[[Any, RootState, RootDecision], ResultT],
        *,
        initial_action: Any = None,
    ) -> RootCallbackResult[ResultT]:
        selected = (
            self.controller.initial_action(query_state)
            if initial_action is None
            else initial_action
        )
        root_state = root_executor(selected)
        if not isinstance(root_state, RootState):
            raise TypeError("root_executor must return RootState.")
        decision = self.controller.on_root_selected(root_state)
        if not isinstance(decision, RootDecision):
            raise TypeError("on_root_selected must return RootDecision.")
        result = successor_executor(selected, root_state, decision)
        return RootCallbackResult(selected, root_state, decision, result)


class NoOpRootCallback:
    """Compatibility controller that preserves the preselected successor action."""

    def __init__(self, decision: RootDecision | None = None) -> None:
        self.decision = decision or RootDecision()

    def initial_action(self, query_state: Mapping[str, float]):
        return None

    def on_root_selected(self, root_state: RootState) -> RootDecision:
        return self.decision


class ThresholdRootCallback(NoOpRootCallback):
    """Cheap uncertainty rule for conditional graph refinement or root retry."""

    def __init__(
        self,
        *,
        successor_method: str = "native_semantic",
        top1_threshold: float = 0.0,
        gap_threshold: float = 0.05,
        entropy_threshold: float = 0.75,
        disagreement_threshold: float = 2.0,
        graph_refine: bool = True,
        retry_root: bool = False,
        successor_k: int = 2,
        search_budget: int = 2,
        kv_budget: int = 4,
    ) -> None:
        super().__init__()
        self.successor_method = successor_method
        self.top1_threshold = float(top1_threshold)
        self.gap_threshold = float(gap_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.disagreement_threshold = float(disagreement_threshold)
        self.enable_graph_refine = bool(graph_refine)
        self.enable_retry_root = bool(retry_root)
        self.successor_k = int(successor_k)
        self.search_budget = int(search_budget)
        self.kv_budget = int(kv_budget)

    def on_root_selected(self, root_state: RootState) -> RootDecision:
        uncertain = (
            root_state.root_top1_score < self.top1_threshold
            or root_state.root_score_gap < self.gap_threshold
            or root_state.candidate_entropy > self.entropy_threshold
            or root_state.channel_disagreement >= self.disagreement_threshold
        )
        if uncertain and self.enable_graph_refine:
            return RootDecision(
                action="graph_refine",
                successor_method=self.successor_method,
                successor_k=self.successor_k,
                graph_refine=True,
                search_budget=self.search_budget,
                kv_budget=self.kv_budget,
            )
        if uncertain and self.enable_retry_root:
            return RootDecision(
                action="retry_root",
                successor_method=self.successor_method,
                successor_k=self.successor_k,
                retry_root=True,
                search_budget=self.search_budget,
                kv_budget=self.kv_budget,
            )
        return RootDecision(
            action="continue",
            successor_method=self.successor_method,
            successor_k=self.successor_k,
            search_budget=self.search_budget,
            kv_budget=self.kv_budget,
        )


class LinearRootCallback:
    """Bootstrap ridge classifier over compact query-plus-root state features."""

    def __init__(
        self,
        *,
        query_feature_names: Sequence[str],
        classes: Sequence[str],
        decisions: Mapping[str, RootDecision],
        mean: torch.Tensor,
        scale: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        self.query_feature_names = tuple(query_feature_names)
        self.classes = tuple(classes)
        self.decisions = dict(decisions)
        self.mean = mean.float()
        self.scale = scale.float()
        self.weights = weights.float()

    @classmethod
    def fit(
        cls,
        states: Sequence[RootState],
        labels: Sequence[str],
        decisions: Mapping[str, RootDecision],
        *,
        query_feature_names: Sequence[str] = (),
        ridge: float = 0.1,
        seed: int = 0,
    ) -> "LinearRootCallback":
        """Fit a deterministic bootstrap ridge model without dataset features."""

        if not states or len(states) != len(labels):
            raise ValueError("Callback fitting requires aligned non-empty states and labels.")
        classes = tuple(sorted(set(labels)))
        if set(classes) - set(decisions):
            raise ValueError("Every callback label requires a concrete RootDecision.")
        features = torch.tensor(
            [state.feature_vector(query_feature_names) for state in states],
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(int(seed))
        sample = torch.randint(len(states), (len(states),), generator=generator)
        x = features[sample]
        sampled_labels = [labels[index] for index in sample.tolist()]
        mean = x.mean(dim=0)
        scale = x.std(dim=0, unbiased=False)
        scale[scale < 1e-8] = 1.0
        design = torch.column_stack(((x - mean) / scale, torch.ones(len(x), dtype=x.dtype)))
        target = torch.zeros((len(x), len(classes)), dtype=x.dtype)
        class_index = {name: index for index, name in enumerate(classes)}
        for row, label in enumerate(sampled_labels):
            target[row, class_index[label]] = 1.0
        penalty = torch.eye(design.shape[1], dtype=x.dtype) * float(ridge)
        penalty[-1, -1] = 0.0
        weights = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
        return cls(
            query_feature_names=query_feature_names,
            classes=classes,
            decisions=decisions,
            mean=mean,
            scale=scale,
            weights=weights,
        )

    def on_root_selected(self, root_state: RootState) -> RootDecision:
        values = torch.tensor(
            root_state.feature_vector(self.query_feature_names), dtype=torch.float32
        )
        design = torch.cat(((values - self.mean) / self.scale, torch.ones(1)))
        scores = design @ self.weights
        probabilities = torch.softmax(scores, dim=0)
        index = int(torch.argmax(probabilities))
        decision = self.decisions[self.classes[index]]
        return RootDecision(**{**asdict(decision), "confidence": float(probabilities[index])})

    def initial_action(self, query_state: Mapping[str, float]):
        """Defer initial-action choice to the caller's one-shot controller."""

        return None
