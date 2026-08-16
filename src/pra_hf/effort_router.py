"""Swappable discriminative routers for minimum-sufficient PRA effort.

The profile selector in :mod:`pra_hf.adaptive_runtime` is the R0 baseline.
This module provides compositional categorical heads (R1/R2) and an
autoregressive interaction-aware variant (R3A) without creating a Cartesian
class over all PRA controls.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn

from .adaptive_runtime import EffortProfile


@dataclass(frozen=True)
class ActionField:
    """One finite categorical control and its validation-defined values."""

    name: str
    values: tuple[Any, ...]
    ordered: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.values or len(set(map(str, self.values))) != len(self.values):
            raise ValueError("Action fields require a name and unique candidate values.")


@dataclass(frozen=True)
class RouterActionSpace:
    """Finite compositional action space shared by every router family."""

    fields: tuple[ActionField, ...]

    def __post_init__(self) -> None:
        if not self.fields or len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("Router action fields must be nonempty and uniquely named.")

    @classmethod
    def from_profiles(
        cls,
        profiles: Sequence[EffortProfile],
        *,
        core_only: bool = False,
    ) -> "RouterActionSpace":
        """Derive supported categorical values from a frozen profile ladder."""

        if not profiles:
            raise ValueError("Profiles are required to derive an action space.")
        controls: list[tuple[str, Sequence[Any]]] = [
            ("query_region_policy", [profile.query_region_policy for profile in profiles]),
            ("facet_policy", [profile.facet_policy for profile in profiles]),
            ("retained_roots", [profile.retained_roots for profile in profiles]),
            ("neighbors", [profile.neighbors_per_expansion for profile in profiles]),
            ("hops", [profile.hop_depth for profile in profiles]),
            ("conceptual_budget", [profile.conceptual_budget for profile in profiles]),
            ("native_kv_budget", [profile.native_kv_budget for profile in profiles]),
        ]
        if not core_only:
            controls.extend(
                [
                    ("threshold", [profile.routing_threshold for profile in profiles]),
                    ("layer_policy", [profile.consumer_layers for profile in profiles]),
                    ("granularity", [profile.granularity_tokens for profile in profiles]),
                    ("materialization", [profile.materialization_policy for profile in profiles]),
                ]
            )
        fields = []
        for name, values in controls:
            unique = tuple(dict.fromkeys(values))
            fields.append(ActionField(name, unique, ordered=name != "materialization"))
        return cls(tuple(fields))

    def field(self, name: str) -> ActionField:
        try:
            return next(field for field in self.fields if field.name == name)
        except StopIteration as error:
            raise KeyError(name) from error

    def index_targets(self, values: Mapping[str, Any]) -> dict[str, int]:
        targets = {}
        for field in self.fields:
            try:
                targets[field.name] = field.values.index(values[field.name])
            except (KeyError, ValueError) as error:
                raise ValueError(f"Unsupported target for action field {field.name}.") from error
        return targets

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [asdict(field) for field in self.fields]}


def profile_actions(profile: EffortProfile) -> dict[str, Any]:
    """Flatten one effort profile into compositional router targets."""

    return {
        "query_region_policy": profile.query_region_policy,
        "facet_policy": profile.facet_policy,
        "retained_roots": profile.retained_roots,
        "neighbors": profile.neighbors_per_expansion,
        "hops": profile.hop_depth,
        "conceptual_budget": profile.conceptual_budget,
        "native_kv_budget": profile.native_kv_budget,
        "threshold": profile.routing_threshold,
        "layer_policy": profile.consumer_layers,
        "granularity": profile.granularity_tokens,
        "materialization": profile.materialization_policy,
    }


@dataclass(frozen=True)
class EffortDecision:
    """Auditable output shared by profile and compositional effort routers."""

    actions: Mapping[str, Any]
    probabilities: Mapping[str, tuple[float, ...]]
    confidence: Mapping[str, float]
    architecture: str
    latency_seconds: float


class EffortPlanner(Protocol):
    """Runtime contract for swappable initial and retry effort planners."""

    def decide(
        self,
        features: torch.Tensor,
        *,
        semantic: torch.Tensor | None = None,
        conservative: bool = False,
    ) -> EffortDecision: ...


class HashingQueryEncoder:
    """Dependency-free cached semantic baseline for small-router experiments.

    It is not presented as a pretrained language model.  The signed hashing
    projection gives R2/R3 a reproducible content channel while preserving the
    public interface expected by a future MiniLM/BERT-like encoder.
    """

    def __init__(self, width: int = 64) -> None:
        if width <= 0:
            raise ValueError("Semantic encoder width must be positive.")
        self.width = width
        self._cache: dict[str, torch.Tensor] = {}

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            if text not in self._cache:
                vector = torch.zeros(self.width, dtype=torch.float32)
                tokens = [token.lower() for token in text.split() if token.strip()]
                for token in tokens:
                    digest = hashlib.sha256(token.encode("utf-8")).digest()
                    bucket = int.from_bytes(digest[:4], "little") % self.width
                    sign = 1.0 if digest[4] & 1 else -1.0
                    vector[bucket] += sign
                self._cache[text] = torch.nn.functional.normalize(vector, dim=0) if vector.any() else vector
            rows.append(self._cache[text])
        return torch.stack(rows)


class MultiHeadEffortRouter(nn.Module):
    """R1/R2 shared MLP with one categorical head per PRA control."""

    def __init__(
        self,
        feature_width: int,
        action_space: RouterActionSpace,
        *,
        semantic_width: int = 0,
        hidden_width: int = 48,
        architecture: str = "R1_feature_mlp",
    ) -> None:
        super().__init__()
        if feature_width <= 0 or semantic_width < 0 or hidden_width <= 0:
            raise ValueError("Router widths must be valid positive dimensions.")
        self.feature_width = feature_width
        self.semantic_width = semantic_width
        self.action_space = action_space
        self.architecture = architecture
        self.trunk = nn.Sequential(
            nn.Linear(feature_width + semantic_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {field.name: nn.Linear(hidden_width, len(field.values)) for field in action_space.fields}
        )

    def _input(self, features: torch.Tensor, semantic: torch.Tensor | None) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[1] != self.feature_width:
            raise ValueError("Feature tensor width does not match the effort router.")
        if self.semantic_width:
            if semantic is None:
                raise ValueError("This effort router requires a semantic query embedding.")
            if semantic.ndim == 1:
                semantic = semantic.unsqueeze(0)
            if semantic.shape != (features.shape[0], self.semantic_width):
                raise ValueError("Semantic tensor shape does not match the effort router.")
            features = torch.cat([features, semantic.to(features)], dim=1)
        elif semantic is not None:
            raise ValueError("A feature-only effort router does not accept semantic embeddings.")
        return features.float()

    def forward(
        self, features: torch.Tensor, semantic: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        state = self.trunk(self._input(features, semantic))
        return {name: head(state) for name, head in self.heads.items()}

    def loss(
        self,
        features: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
        *,
        semantic: torch.Tensor | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        logits = self(features, semantic)
        losses = [
            float((weights or {}).get(name, 1.0)) * torch.nn.functional.cross_entropy(value, targets[name])
            for name, value in logits.items()
        ]
        return torch.stack(losses).sum()

    @torch.no_grad()
    def decide(
        self,
        features: torch.Tensor,
        *,
        semantic: torch.Tensor | None = None,
        conservative: bool = False,
    ) -> EffortDecision:
        started = time.perf_counter()
        logits = self(features, semantic)
        if next(iter(logits.values())).shape[0] != 1:
            raise ValueError("decide expects exactly one feature row.")
        actions, probabilities, confidence = {}, {}, {}
        for field in self.action_space.fields:
            distribution = torch.softmax(logits[field.name][0], dim=0)
            if conservative and field.ordered:
                cumulative = torch.cumsum(distribution, dim=0)
                index = int(torch.nonzero(cumulative >= 0.9, as_tuple=False)[0])
            else:
                index = int(torch.argmax(distribution))
            actions[field.name] = field.values[index]
            probabilities[field.name] = tuple(float(value) for value in distribution)
            confidence[field.name] = float(distribution[index])
        return EffortDecision(
            actions,
            probabilities,
            confidence,
            self.architecture,
            time.perf_counter() - started,
        )


class AutoregressiveEffortRouter(nn.Module):
    """R3A controller whose categorical heads condition on earlier choices."""

    def __init__(
        self,
        feature_width: int,
        action_space: RouterActionSpace,
        *,
        semantic_width: int = 0,
        hidden_width: int = 48,
        context_width: int = 16,
    ) -> None:
        super().__init__()
        if feature_width <= 0 or hidden_width <= 0 or context_width <= 0:
            raise ValueError("Autoregressive router widths must be positive.")
        self.feature_width = feature_width
        self.semantic_width = semantic_width
        self.action_space = action_space
        self.trunk = nn.Sequential(
            nn.Linear(feature_width + semantic_width, hidden_width), nn.ReLU(), nn.Linear(hidden_width, hidden_width), nn.ReLU()
        )
        self.embeddings = nn.ModuleDict(
            {
                field.name: nn.Embedding(len(field.values), context_width)
                for field in action_space.fields[:-1]
            }
        )
        self.heads = nn.ModuleDict(
            {field.name: nn.Linear(hidden_width + context_width, len(field.values)) for field in action_space.fields}
        )
        self.context_width = context_width

    def forward(
        self,
        features: torch.Tensor,
        semantic: torch.Tensor | None = None,
        teacher_targets: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[1] != self.feature_width:
            raise ValueError("Feature tensor width does not match the autoregressive router.")
        if self.semantic_width:
            if semantic is None:
                raise ValueError("This autoregressive router requires semantic embeddings.")
            if semantic.ndim == 1:
                semantic = semantic.unsqueeze(0)
            features = torch.cat([features, semantic.to(features)], dim=1)
        state = self.trunk(features.float())
        context = torch.zeros(len(state), self.context_width, device=state.device)
        logits = {}
        for field in self.action_space.fields:
            logits[field.name] = self.heads[field.name](torch.cat([state, context], dim=1))
            choice = (
                teacher_targets[field.name]
                if teacher_targets is not None
                else torch.argmax(logits[field.name], dim=1)
            )
            if field.name in self.embeddings:
                context = context + self.embeddings[field.name](choice)
        return logits

    def loss(
        self,
        features: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
        *,
        semantic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self(features, semantic, teacher_targets=targets)
        return torch.stack(
            [torch.nn.functional.cross_entropy(logits[name], targets[name]) for name in logits]
        ).sum()

    @torch.no_grad()
    def decide(
        self,
        features: torch.Tensor,
        *,
        semantic: torch.Tensor | None = None,
        conservative: bool = False,
    ) -> EffortDecision:
        if conservative:
            # Autoregressive conservative decoding needs sequence-level search;
            # keep the first implementation explicit rather than silently
            # applying incompatible independent quantiles.
            raise ValueError("Conservative autoregressive decoding is not implemented.")
        started = time.perf_counter()
        logits = self(features, semantic)
        if next(iter(logits.values())).shape[0] != 1:
            raise ValueError("decide expects exactly one feature row.")
        actions, probabilities, confidence = {}, {}, {}
        for field in self.action_space.fields:
            distribution = torch.softmax(logits[field.name][0], dim=0)
            index = int(torch.argmax(distribution))
            actions[field.name] = field.values[index]
            probabilities[field.name] = tuple(float(value) for value in distribution)
            confidence[field.name] = float(distribution[index])
        return EffortDecision(
            actions,
            probabilities,
            confidence,
            "R3A_autoregressive",
            time.perf_counter() - started,
        )
