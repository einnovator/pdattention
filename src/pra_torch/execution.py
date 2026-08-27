"""Engine-neutral execution policy and lifecycle contracts for PRA.

The types in this module deliberately contain no Hugging Face or PyTorch
objects.  Engines map logical selected identities to their own native payloads
after semantic routing has produced a :class:`PRASelectionPlan`.
"""

from __future__ import annotations

import statistics
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class PRASelectionStage(str, Enum):
    """Frequency at which semantic selection may change."""

    REQUEST = "request"
    PHASE = "phase"
    TOKEN = "token"


class PRASelectionLayerScope(str, Enum):
    """Whether logical identities are shared or independently selected."""

    SHARED = "shared"
    PER_LAYER = "per_layer"


class PRAMaterializationScope(str, Enum):
    """Lifetime at which selected native payloads may be rebuilt."""

    REQUEST = "request"
    PHASE = "phase"
    LAYER = "layer"
    TOKEN = "token"


class PRAResidencyPolicy(str, Enum):
    """Physical retention policy for a materialized native payload."""

    KEEP = "keep"
    LAYER_LIFETIME = "layer_lifetime"
    LRU = "lru"
    EXTERNAL_ONLY = "external_only"


class PRARoutingLayerPolicy(str, Enum):
    """Named choices for the source layer of shared selection."""

    EXPLICIT = "explicit"
    FIRST_PRA_LAYER = "first_pra_layer"
    MIDDLE_PRA_LAYER = "middle_pra_layer"
    LAST_PRA_LAYER = "last_pra_layer"


@dataclass(frozen=True)
class PRAExecutionPolicy:
    """Orthogonal semantic and physical choices for one PRA request."""

    selection_stage: PRASelectionStage | str = PRASelectionStage.REQUEST
    selection_layer_scope: PRASelectionLayerScope | str = PRASelectionLayerScope.SHARED
    materialization_scope: PRAMaterializationScope | str = PRAMaterializationScope.REQUEST
    residency_policy: PRAResidencyPolicy | str = PRAResidencyPolicy.KEEP
    routing_layer_policy: PRARoutingLayerPolicy | str = PRARoutingLayerPolicy.EXPLICIT
    routing_layer: int | None = None
    phase_policy: str = "prefill_and_completion"
    reselection_interval_tokens: int | None = None
    allow_stale_selection_before_routing_layer: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_stage", PRASelectionStage(self.selection_stage))
        object.__setattr__(
            self, "selection_layer_scope", PRASelectionLayerScope(self.selection_layer_scope)
        )
        object.__setattr__(
            self, "materialization_scope", PRAMaterializationScope(self.materialization_scope)
        )
        object.__setattr__(self, "residency_policy", PRAResidencyPolicy(self.residency_policy))
        object.__setattr__(
            self, "routing_layer_policy", PRARoutingLayerPolicy(self.routing_layer_policy)
        )
        if self.phase_policy != "prefill_and_completion":
            raise ValueError("The initial PHASE implementation supports prefill_and_completion only.")
        if self.reselection_interval_tokens is not None and self.reselection_interval_tokens <= 0:
            raise ValueError("reselection_interval_tokens must be positive when provided.")
        if self.allow_stale_selection_before_routing_layer:
            raise ValueError("Stale shared selections before the routing layer are not implemented.")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "selection_stage",
            "selection_layer_scope",
            "materialization_scope",
            "residency_policy",
            "routing_layer_policy",
        ):
            value[key] = getattr(self, key).value
        return value

    @classmethod
    def from_value(
        cls, value: "PRAExecutionPolicy | Mapping[str, object] | None"
    ) -> "PRAExecutionPolicy | None":
        if value is None or isinstance(value, cls):
            return value
        return cls(**dict(value))


@dataclass(frozen=True)
class PRAExecutionCapabilities:
    """Static policy support reported by an engine adapter."""

    engine: str
    request_selection: bool = True
    phase_selection: bool = False
    token_selection: bool = False
    shared_layer_selection: bool = True
    per_layer_selection: bool = False
    request_materialization: bool = True
    phase_materialization: bool = False
    layer_materialization: bool = False
    token_materialization: bool = False
    keep_residency: bool = True
    layer_lifetime_residency: bool = False
    lru_residency: bool = False
    external_only_residency: bool = False
    external_kv: bool = False
    async_prefetch: bool = False
    streaming: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedPRAExecutionPolicy:
    """Validated effective policy plus field-level precedence provenance."""

    policy: PRAExecutionPolicy
    field_sources: Mapping[str, str]
    capabilities: PRAExecutionCapabilities
    downgrades: tuple[str, ...] = ()

    @property
    def policy_source(self) -> str:
        values = set(self.field_sources.values())
        return next(iter(values)) if len(values) == 1 else "mixed"

    def to_dict(self) -> dict[str, object]:
        return {
            **self.policy.to_dict(),
            "policy_source": self.policy_source,
            "field_sources": dict(self.field_sources),
            "downgrades": list(self.downgrades),
            "engine": self.capabilities.engine,
        }


_DEFAULT_LOCK = threading.RLock()
_GLOBAL_DEFAULT = PRAExecutionPolicy()


def get_default_execution_policy() -> PRAExecutionPolicy:
    """Return the process default from its single authoritative location."""

    with _DEFAULT_LOCK:
        return _GLOBAL_DEFAULT


def set_default_execution_policy(
    policy: PRAExecutionPolicy | Mapping[str, object],
) -> PRAExecutionPolicy:
    """Replace the process default for future requests and return it."""

    global _GLOBAL_DEFAULT
    resolved = PRAExecutionPolicy.from_value(policy)
    assert resolved is not None
    with _DEFAULT_LOCK:
        _GLOBAL_DEFAULT = resolved
    return resolved


def _overlay(
    base: PRAExecutionPolicy,
    value: PRAExecutionPolicy | Mapping[str, object] | None,
) -> tuple[PRAExecutionPolicy, set[str]]:
    if value is None:
        return base, set()
    if isinstance(value, PRAExecutionPolicy):
        return value, set(value.to_dict())
    changes = dict(value)
    unknown = set(changes).difference(base.to_dict())
    if unknown:
        raise ValueError(f"Unknown PRA execution-policy fields: {sorted(unknown)}")
    return PRAExecutionPolicy(**{**base.to_dict(), **changes}), set(changes)


def resolve_routing_layer(
    policy: PRAExecutionPolicy,
    active_layers: Sequence[int],
    configured_layer: int,
) -> int:
    """Resolve a named routing-layer policy against active PRA layers."""

    layers = tuple(sorted(set(int(layer) for layer in active_layers)))
    if not layers:
        raise ValueError("At least one active PRA layer is required.")
    if policy.routing_layer_policy == PRARoutingLayerPolicy.EXPLICIT:
        selected = configured_layer if policy.routing_layer is None else int(policy.routing_layer)
    elif policy.routing_layer_policy == PRARoutingLayerPolicy.FIRST_PRA_LAYER:
        selected = layers[0]
    elif policy.routing_layer_policy == PRARoutingLayerPolicy.MIDDLE_PRA_LAYER:
        selected = layers[(len(layers) - 1) // 2]
    else:
        selected = layers[-1]
    if selected not in layers:
        raise ValueError(f"Routing layer {selected} is not an active PRA layer {layers}.")
    return selected


def validate_execution_policy(
    policy: PRAExecutionPolicy,
    capabilities: PRAExecutionCapabilities,
    *,
    active_layers: Sequence[int] = (),
    configured_routing_layer: int | None = None,
) -> None:
    """Reject unsupported or semantically contradictory combinations."""

    stage_support = {
        PRASelectionStage.REQUEST: capabilities.request_selection,
        PRASelectionStage.PHASE: capabilities.phase_selection,
        PRASelectionStage.TOKEN: capabilities.token_selection,
    }
    layer_support = {
        PRASelectionLayerScope.SHARED: capabilities.shared_layer_selection,
        PRASelectionLayerScope.PER_LAYER: capabilities.per_layer_selection,
    }
    materialization_support = {
        PRAMaterializationScope.REQUEST: capabilities.request_materialization,
        PRAMaterializationScope.PHASE: capabilities.phase_materialization,
        PRAMaterializationScope.LAYER: capabilities.layer_materialization,
        PRAMaterializationScope.TOKEN: capabilities.token_materialization,
    }
    residency_support = {
        PRAResidencyPolicy.KEEP: capabilities.keep_residency,
        PRAResidencyPolicy.LAYER_LIFETIME: capabilities.layer_lifetime_residency,
        PRAResidencyPolicy.LRU: capabilities.lru_residency,
        PRAResidencyPolicy.EXTERNAL_ONLY: capabilities.external_only_residency,
    }
    if not stage_support[policy.selection_stage]:
        raise ValueError(
            f"Engine {capabilities.engine!r} does not support {policy.selection_stage.value} selection."
        )
    if not layer_support[policy.selection_layer_scope]:
        raise ValueError(
            f"Engine {capabilities.engine!r} does not support "
            f"{policy.selection_layer_scope.value} layer selection."
        )
    if not materialization_support[policy.materialization_scope]:
        raise ValueError(
            f"Engine {capabilities.engine!r} does not support "
            f"{policy.materialization_scope.value} materialization."
        )
    if not residency_support[policy.residency_policy]:
        raise ValueError(
            f"Engine {capabilities.engine!r} does not support "
            f"{policy.residency_policy.value} residency."
        )
    if (
        policy.selection_stage == PRASelectionStage.TOKEN
        and policy.materialization_scope == PRAMaterializationScope.REQUEST
    ):
        raise ValueError("TOKEN selection cannot use immutable REQUEST materialization.")
    if (
        policy.residency_policy == PRAResidencyPolicy.LAYER_LIFETIME
        and policy.materialization_scope == PRAMaterializationScope.REQUEST
    ):
        raise ValueError("LAYER_LIFETIME residency contradicts REQUEST materialization.")
    if policy.selection_stage == PRASelectionStage.PHASE and policy.materialization_scope not in {
        PRAMaterializationScope.PHASE,
        PRAMaterializationScope.LAYER,
    }:
        raise ValueError("PHASE selection requires PHASE or LAYER materialization.")
    if policy.selection_stage == PRASelectionStage.TOKEN and policy.materialization_scope not in {
        PRAMaterializationScope.TOKEN,
        PRAMaterializationScope.LAYER,
    }:
        raise ValueError("TOKEN selection requires TOKEN or LAYER materialization.")
    if active_layers and configured_routing_layer is not None:
        routing_layer = resolve_routing_layer(policy, active_layers, configured_routing_layer)
        if (
            policy.selection_stage == PRASelectionStage.TOKEN
            and policy.selection_layer_scope == PRASelectionLayerScope.SHARED
            and routing_layer > max(active_layers)
        ):
            raise ValueError("TOKEN+SHARED routing must precede at least one consuming layer.")


def resolve_execution_policy(
    *,
    request_policy: PRAExecutionPolicy | Mapping[str, object] | None = None,
    model_policy: PRAExecutionPolicy | Mapping[str, object] | None = None,
    global_policy: PRAExecutionPolicy | Mapping[str, object] | None = None,
    capabilities: PRAExecutionCapabilities,
    active_layers: Sequence[int] = (),
    configured_routing_layer: int | None = None,
) -> ResolvedPRAExecutionPolicy:
    """Apply request > model > global precedence and validate explicitly."""

    global_value = PRAExecutionPolicy.from_value(global_policy) or get_default_execution_policy()
    sources = {key: "global_default" for key in global_value.to_dict()}
    effective, model_fields = _overlay(global_value, model_policy)
    for key in model_fields:
        sources[key] = "model_default"
    effective, request_fields = _overlay(effective, request_policy)
    for key in request_fields:
        sources[key] = "request_override"
    validate_execution_policy(
        effective,
        capabilities,
        active_layers=active_layers,
        configured_routing_layer=configured_routing_layer,
    )
    return ResolvedPRAExecutionPolicy(effective, sources, capabilities)


@dataclass(frozen=True)
class PRASelectedIdentity:
    """Layer-independent reference/chunk identity without native tensors."""

    reference_uri: str
    chunk_id: str
    token_start: int
    token_end: int
    reference_score: float | None = None
    chunk_score: float | None = None
    winning_gist_index: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.reference_uri or not self.chunk_id or self.token_end < self.token_start:
            raise ValueError("A selected identity requires a URI, chunk ID, and ordered span.")

    @property
    def key(self) -> tuple[str, str]:
        return self.reference_uri, self.chunk_id


@dataclass(frozen=True)
class PRASelectionPlan:
    """One semantic selection epoch, shared or indexed by layer."""

    selection_stage: PRASelectionStage | str
    layer_scope: PRASelectionLayerScope | str
    source_layer: int | None
    epoch_id: int
    shared_rows: tuple[tuple[PRASelectedIdentity, ...], ...] | None = None
    per_layer_rows: Mapping[int, tuple[tuple[PRASelectedIdentity, ...], ...]] = field(
        default_factory=dict
    )
    phase: str | None = None
    token_index: int | None = None
    routing_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_stage", PRASelectionStage(self.selection_stage))
        object.__setattr__(self, "layer_scope", PRASelectionLayerScope(self.layer_scope))
        object.__setattr__(self, "per_layer_rows", dict(self.per_layer_rows))
        if self.epoch_id <= 0:
            raise ValueError("Selection epoch IDs must be positive.")
        if self.layer_scope == PRASelectionLayerScope.SHARED and self.shared_rows is None:
            raise ValueError("SHARED plans require shared_rows.")
        if self.layer_scope == PRASelectionLayerScope.PER_LAYER and not self.per_layer_rows:
            raise ValueError("PER_LAYER plans require per_layer_rows.")

    def rows_for(self, layer_id: int) -> tuple[tuple[PRASelectedIdentity, ...], ...]:
        if self.layer_scope == PRASelectionLayerScope.SHARED:
            assert self.shared_rows is not None
            return self.shared_rows
        try:
            return self.per_layer_rows[int(layer_id)]
        except KeyError as error:
            raise KeyError(f"Selection plan has no rows for layer {layer_id}.") from error

    def identity_keys(self, layer_id: int | None = None) -> frozenset[tuple[str, str]]:
        if layer_id is None:
            if self.layer_scope == PRASelectionLayerScope.SHARED:
                rows = self.shared_rows or ()
            else:
                rows = tuple(row for value in self.per_layer_rows.values() for row in value)
        else:
            rows = self.rows_for(layer_id)
        return frozenset(identity.key for row in rows for identity in row)


@dataclass
class PRARequestExecutionContext:
    """Mutable request-owned policy, plans, physical state, and trace."""

    resolved_policy: ResolvedPRAExecutionPolicy
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    selection_plan: PRASelectionPlan | None = None
    phase_plans: dict[str, PRASelectionPlan] = field(default_factory=dict)
    token_index: int = 0
    phase: str = "request"
    materialization_state: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, object]] = field(default_factory=list)
    _epoch: int = 0

    @property
    def policy(self) -> PRAExecutionPolicy:
        return self.resolved_policy.policy

    def next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    def record_plan(self, plan: PRASelectionPlan) -> None:
        previous = self.selection_plan
        self.selection_plan = plan
        if plan.phase:
            self.phase_plans[plan.phase] = plan
        current = plan.identity_keys()
        prior = previous.identity_keys() if previous is not None else frozenset()
        union = current | prior
        self.trace.append(
            {
                "event": "selection",
                "epoch_id": plan.epoch_id,
                "phase": plan.phase,
                "token_index": plan.token_index,
                "source_layer": plan.source_layer,
                "selected_identities": sorted(current),
                "temporal_jaccard": len(current & prior) / max(len(union), 1),
                "selection_additions": len(current - prior),
                "selection_removals": len(prior - current),
                "routing_seconds": plan.routing_seconds,
                "routing_operations": (
                    len(plan.per_layer_rows)
                    if plan.layer_scope == PRASelectionLayerScope.PER_LAYER
                    else 1
                ),
            }
        )

    def summary(self) -> dict[str, object]:
        selection_rows = [row for row in self.trace if row.get("event") == "selection"]
        materialization_rows = [
            row for row in self.trace if row.get("event") == "materialization"
        ]
        jaccards = [float(row["temporal_jaccard"]) for row in selection_rows[1:]]
        layer_jaccards: list[float] = []
        if self.selection_plan is not None:
            layers = sorted(self.selection_plan.per_layer_rows)
            for left_index, left in enumerate(layers):
                left_ids = self.selection_plan.identity_keys(left)
                for right in layers[left_index + 1 :]:
                    right_ids = self.selection_plan.identity_keys(right)
                    layer_jaccards.append(
                        len(left_ids & right_ids) / max(len(left_ids | right_ids), 1)
                    )
        return {
            "request_id": self.request_id,
            "execution_policy": self.resolved_policy.to_dict(),
            "selection_epochs": len(selection_rows),
            "materialization_epochs": len(materialization_rows),
            "routing_operations": sum(
                int(row.get("routing_operations", 1)) for row in selection_rows
            ),
            "temporal_selection_jaccard_mean": (
                statistics.fmean(jaccards) if jaccards else 1.0
            ),
            "layer_selection_jaccard_mean": (
                statistics.fmean(layer_jaccards) if layer_jaccards else 1.0
            ),
            "trace": list(self.trace),
        }


class PRASelectionController:
    """Decide when a logical selection epoch must be recomputed."""

    def selection_for(
        self,
        *,
        context: PRARequestExecutionContext,
        layer_id: int,
        phase: str,
        token_index: int,
        route: Callable[[int], tuple[tuple[PRASelectedIdentity, ...], ...]],
        routing_seconds: float = 0.0,
    ) -> PRASelectionPlan:
        policy = context.policy
        existing = context.selection_plan
        if policy.selection_stage == PRASelectionStage.REQUEST and existing is not None:
            if (
                policy.selection_layer_scope == PRASelectionLayerScope.SHARED
                or layer_id in existing.per_layer_rows
            ):
                return existing
        if policy.selection_stage == PRASelectionStage.PHASE and phase in context.phase_plans:
            phase_plan = context.phase_plans[phase]
            if (
                policy.selection_layer_scope == PRASelectionLayerScope.SHARED
                or layer_id in phase_plan.per_layer_rows
            ):
                return phase_plan
        if (
            policy.selection_stage == PRASelectionStage.TOKEN
            and existing is not None
            and policy.reselection_interval_tokens
            and token_index % policy.reselection_interval_tokens
        ):
            return existing
        rows = route(layer_id)
        if policy.selection_layer_scope == PRASelectionLayerScope.SHARED:
            plan = PRASelectionPlan(
                policy.selection_stage,
                policy.selection_layer_scope,
                layer_id,
                context.next_epoch(),
                shared_rows=rows,
                phase=phase,
                token_index=token_index,
                routing_seconds=routing_seconds,
            )
        else:
            previous = dict(existing.per_layer_rows) if existing is not None else {}
            previous[layer_id] = rows
            plan = PRASelectionPlan(
                policy.selection_stage,
                policy.selection_layer_scope,
                layer_id,
                context.next_epoch(),
                per_layer_rows=previous,
                phase=phase,
                token_index=token_index,
                routing_seconds=routing_seconds,
            )
        context.record_plan(plan)
        return plan


class PRALayerPayloadResolver:
    """Map logical plan rows to one engine layer's native payload handles."""

    def __init__(self, resolve_rows: Callable[[Sequence[Sequence[PRASelectedIdentity]], int], Any]):
        self._resolve_rows = resolve_rows

    def resolve(self, plan: PRASelectionPlan, layer_id: int) -> Any:
        return self._resolve_rows(plan.rows_for(layer_id), int(layer_id))


class PRAMaterializationManager:
    """Cache opaque engine payloads according to materialization lifetime."""

    def __init__(self, materialize: Callable[[PRASelectionPlan, int, Mapping[str, object]], Any]):
        self._materialize = materialize
        self._cache: dict[tuple[object, ...], Any] = {}

    def begin_request(self, context: PRARequestExecutionContext) -> None:
        self._cache.clear()
        context.materialization_state.clear()

    def begin_phase(self, context: PRARequestExecutionContext, phase: str) -> None:
        context.phase = phase
        if context.policy.materialization_scope == PRAMaterializationScope.PHASE:
            self._cache.clear()

    def get_layer_memory(
        self,
        *,
        context: PRARequestExecutionContext,
        plan: PRASelectionPlan,
        layer_id: int,
        direct_tokens: int = 0,
        metadata: Mapping[str, object] | None = None,
    ) -> Any:
        metadata = {**dict(metadata or {}), "direct_tokens": int(direct_tokens)}
        scope = context.policy.materialization_scope
        cacheable = (
            context.policy.residency_policy == PRAResidencyPolicy.KEEP
            and scope in {PRAMaterializationScope.REQUEST, PRAMaterializationScope.PHASE}
        )
        lifetime = "request" if scope == PRAMaterializationScope.REQUEST else context.phase
        key = (
            lifetime,
            int(layer_id),
            tuple(sorted(plan.identity_keys(layer_id))),
            tuple(sorted((str(key), repr(value)) for key, value in metadata.items())),
        )
        hit = cacheable and key in self._cache
        if hit:
            payload = self._cache[key]
        else:
            payload = self._materialize(plan, int(layer_id), metadata)
            if cacheable:
                self._cache[key] = payload
        context.trace.append(
            {
                "event": "materialization",
                "epoch_id": plan.epoch_id,
                "phase": context.phase,
                "layer_id": int(layer_id),
                "cache_hit": bool(hit),
            }
        )
        return payload

    def end_layer(self, context: PRARequestExecutionContext, layer_id: int) -> None:
        if context.policy.residency_policy == PRAResidencyPolicy.LAYER_LIFETIME:
            self._cache = {
                key: value for key, value in self._cache.items() if key[1] != int(layer_id)
            }

    def end_phase(self, context: PRARequestExecutionContext) -> None:
        if context.policy.materialization_scope == PRAMaterializationScope.PHASE:
            self._cache.clear()

    def end_request(self, context: PRARequestExecutionContext) -> None:
        self._cache.clear()
        context.materialization_state.clear()
