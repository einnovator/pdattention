"""Stable user-facing configuration for the PRA Hugging Face integration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig

from .layer_profiles import (
    AddressEncodingPolicy,
    DetailKVEncodingPolicy,
    LAYER_PROFILE_OBJECTIVES,
    LayerProfileRegistry,
    LayerSelection,
    MissingDetailKVPolicy,
    ResolvedLayerRoles,
    eligible_layers,
)
from .native_geometry import NativeMaterializationMode, materialization_profile
from .profile_benchmarks import (
    MeasurementStatus,
    ProductProfile,
    ProfileBenchmarkRegistry,
    profile_objective,
)


_DEFAULT_CONSUMPTION_LAYERS = tuple(range(-8, 0))


@dataclass
class PRAConfig:
    """Configure semantic routing and bounded native-K/V consumption.

    ``selected_fraction`` takes precedence over ``top_k`` when it is not
    ``None`` in one-shot mode.  Iterative mode instead uses the hard
    ``max_unique_chunks`` closure budget.  Negative layer indices are resolved
    relative to the decoder depth.
    """

    enabled: bool = True
    profile: str | None = None
    workload: str | None = None
    product_profile_registry: str | None = None
    routing_layer: int = -1
    routing_layers: tuple[int, ...] | None = None
    consumption_layers: tuple[int, ...] = _DEFAULT_CONSUMPTION_LAYERS
    address_layers: tuple[int, ...] | None = None
    detail_kv_layers: tuple[int, ...] | None = None
    consumption_profile: str | dict[str, Any] | None = None
    layer_profile_name: str | None = None
    layer_profile_objective: str = "balanced"
    layer_profile_registry: str | None = None
    workload_class: str | None = None
    materialization_class: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    detail_kv_encoding_policy: str = DetailKVEncodingPolicy.MINIMAL.value
    address_encoding_policy: str = AddressEncodingPolicy.ROUTING_ONLY.value
    missing_detail_kv_policy: str = MissingDetailKVPolicy.REENCODE_MISSING.value
    address_mode: str = "native"
    routing_representation: str = ATTENTION_INPUT_HIDDEN_STATE
    chunk_tokens: int = 32
    chunk_overlap_tokens: int = 0
    local_gist_tokens: int | None = None
    selected_fraction: float | None = 0.20
    top_k: int = 8
    max_direct_context: int = 256
    native_operation_limit: int = 512
    max_materialized_tokens: int = 256
    materialization_profile: str | None = None
    materialization_mode: str = NativeMaterializationMode.SELECTED_CHUNK.value
    materialization_context_tokens: int = 0
    materialization_left_context_tokens: int | None = None
    materialization_right_context_tokens: int | None = None
    materialization_target_tokens: int | None = None
    materialization_full_selected_record: bool = False
    context_safety_reserve_tokens: int = 4
    encoding_block_tokens: int = 256
    reference_device: str = "cpu"
    pin_reference_memory: bool = False
    non_blocking_transfer: bool = False
    query_strategy: str = "last"
    query_window: int = 16
    query_half_life: float = 4.0
    trigger_threshold: float = float("-inf")
    routing_mode: str = "one_shot"
    routing_depth: int = 2
    branch_top_k: int = 2
    beam_size: int = 8
    max_unique_chunks: int = 8
    root_anchor_alpha: float = 0.5
    frontier_mode: str = "direct"
    frontier_projection: str = "memory"
    residual_beta: float = 1.0
    path_score_mode: str = "product"
    iterative_min_confidence: float | None = None
    hybrid_discovery_mode: str = "iterative_hybrid"
    hybrid_semantic_weight: float = 0.65
    hybrid_token_weight: float = 0.35
    hybrid_later_semantic_weight: float = 0.25
    hybrid_later_token_weight: float = 0.75
    hybrid_exact_min_tokens: int = 2
    hybrid_cascade_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.profile is not None:
            self.profile = ProductProfile(str(self.profile).upper()).value
            if self.layer_profile_name is None:
                self.layer_profile_name = "product_profile"
            self.layer_profile_objective = profile_objective(self.profile)
        if self.workload is not None and self.workload_class is None:
            self.workload_class = self.workload
        self.consumption_layers = tuple(int(layer) for layer in self.consumption_layers)
        self.routing_layers = (
            None
            if self.routing_layers is None
            else tuple(int(layer) for layer in self.routing_layers)
        )
        self.address_layers = (
            None
            if self.address_layers is None
            else tuple(int(layer) for layer in self.address_layers)
        )
        self.detail_kv_layers = (
            None
            if self.detail_kv_layers is None
            else tuple(int(layer) for layer in self.detail_kv_layers)
        )
        self.detail_kv_encoding_policy = DetailKVEncodingPolicy(
            self.detail_kv_encoding_policy
        ).value
        self.address_encoding_policy = AddressEncodingPolicy(
            self.address_encoding_policy
        ).value
        self.missing_detail_kv_policy = MissingDetailKVPolicy(
            self.missing_detail_kv_policy
        ).value
        if self.layer_profile_objective not in LAYER_PROFILE_OBJECTIVES:
            raise ValueError(
                f"layer_profile_objective must be one of {LAYER_PROFILE_OBJECTIVES}."
            )
        if self.address_mode not in {"native", "external"}:
            raise ValueError("address_mode must be 'native' or 'external'.")
        if self.materialization_profile is not None:
            profile = materialization_profile(self.materialization_profile)
            self.chunk_tokens = profile.routing_chunk_tokens
            self.chunk_overlap_tokens = profile.routing_chunk_overlap_tokens
            self.materialization_mode = profile.mode.value
            self.materialization_left_context_tokens = profile.left_context_tokens
            self.materialization_right_context_tokens = profile.right_context_tokens
        self.materialization_mode = NativeMaterializationMode(
            self.materialization_mode
        ).value
        if not self.consumption_layers:
            raise ValueError("consumption_layers cannot be empty.")
        if self.selected_fraction is not None and not 0 < self.selected_fraction <= 1:
            raise ValueError("selected_fraction must lie in (0, 1].")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive.")
        if self.materialization_target_tokens is not None and self.materialization_target_tokens <= 0:
            raise ValueError("materialization_target_tokens must be positive or None.")
        if self.materialization_full_selected_record and self.materialization_target_tokens is not None:
            raise ValueError(
                "materialization_target_tokens and materialization_full_selected_record are mutually exclusive."
            )
        contexts = (
            self.materialization_context_tokens,
            self.materialization_left_context_tokens,
            self.materialization_right_context_tokens,
        )
        if any(value is not None and value < 0 for value in contexts):
            raise ValueError("Materialization context widths must be non-negative.")
        if self.materialization_full_selected_record:
            self.materialization_mode = NativeMaterializationMode.FULL_SELECTED_RECORD.value
        elif self.materialization_target_tokens is not None:
            self.materialization_mode = NativeMaterializationMode.EXPANDED_WINDOW.value
        if not 0 <= self.chunk_overlap_tokens < self.chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be in [0, chunk_tokens).")
        local_tokens = self.local_gist_tokens or min(32, self.chunk_tokens)
        if local_tokens <= 0 or local_tokens > self.chunk_tokens:
            raise ValueError("local_gist_tokens must lie in [1, chunk_tokens].")
        if self.routing_mode == "local_iterative" and self.chunk_tokens % local_tokens:
            raise ValueError(
                "local_iterative currently requires chunk_tokens divisible by local_gist_tokens."
            )
        if self.reference_device not in {"cpu", "gpu"}:
            raise ValueError("reference_device must be 'cpu' or 'gpu'.")
        iterative_modes = {
            "iterative",
            "local_iterative",
            "token_iterative",
            "hybrid_iterative",
        }
        if self.routing_mode not in {"one_shot", *iterative_modes}:
            raise ValueError(
                "routing_mode must be 'one_shot', 'iterative', 'local_iterative', "
                "'token_iterative', or 'hybrid_iterative'."
            )
        # Import lazily so the stable config module does not create an import cycle.
        if self.routing_mode in iterative_modes:
            self.iterative_config
        if self.routing_mode in {"token_iterative", "hybrid_iterative"}:
            self.hybrid_discovery_policy

    @property
    def selection_policy(self) -> str:
        """Return the active budget policy, including documented precedence."""
        if self.routing_mode in {
            "iterative",
            "local_iterative",
            "token_iterative",
            "hybrid_iterative",
        }:
            return f"{self.routing_mode}_closure"
        return "selected_fraction" if self.selected_fraction is not None else "top_k"

    @property
    def materialization_context(self) -> tuple[int, int]:
        """Resolve symmetric and side-specific context into left/right widths."""

        return (
            self.materialization_context_tokens
            if self.materialization_left_context_tokens is None
            else self.materialization_left_context_tokens,
            self.materialization_context_tokens
            if self.materialization_right_context_tokens is None
            else self.materialization_right_context_tokens,
        )

    @property
    def iterative_config(self):
        """Translate public closure controls into the tensor-only router config."""
        from .iterative import IterativeRoutingConfig

        return IterativeRoutingConfig(
            depth=self.routing_depth,
            branch_top_k=self.branch_top_k,
            beam_size=self.beam_size,
            max_unique_chunks=self.max_unique_chunks,
            root_anchor_alpha=self.root_anchor_alpha,
            frontier_mode=self.frontier_mode,
            frontier_projection=self.frontier_projection,
            residual_beta=self.residual_beta,
            path_score_mode=self.path_score_mode,
            min_confidence=self.iterative_min_confidence,
        )

    @property
    def hybrid_discovery_policy(self):
        """Translate public channel controls into the discovery-only policy."""
        from .hybrid_discovery import HybridDiscoveryPolicy

        mode = (
            "token_weighted"
            if self.routing_mode == "token_iterative"
            else self.hybrid_discovery_mode
        )
        return HybridDiscoveryPolicy(
            mode=mode,
            semantic_weight=self.hybrid_semantic_weight,
            token_weight=self.hybrid_token_weight,
            later_semantic_weight=self.hybrid_later_semantic_weight,
            later_token_weight=self.hybrid_later_token_weight,
            exact_min_tokens=self.hybrid_exact_min_tokens,
            cascade_threshold=self.hybrid_cascade_threshold,
        )

    def resolved_layer_roles(self, model_config_or_layer_count) -> ResolvedLayerRoles:
        """Resolve and validate independent address, detail, route, and consume roles.

        Explicit request fields take precedence.  A named registry profile is
        consulted only for roles that the request leaves unspecified.  Gemma 3
        profiles normalize over native full-attention layers rather than over
        its interleaved sliding-attention stack.
        """

        layer_count, allowed, family = eligible_layers(model_config_or_layer_count)
        profile: dict[str, Any] = {}
        profile_source = "request"
        registry_version = None
        if self.layer_profile_name is not None:
            registry = (
                LayerProfileRegistry.from_path(self.layer_profile_registry)
                if self.layer_profile_registry
                else LayerProfileRegistry.default()
            )
            profile, profile_source = registry.resolve(
                family=family,
                model_id=self.model_id,
                workload=self.workload_class,
                materialization=self.materialization_class,
                objective=self.layer_profile_objective,
            )
            registry_version = registry.version

        if self.routing_layers is not None:
            routing_spec: Any = self.routing_layers
        elif "routing" in profile:
            routing_spec = profile["routing"]
        else:
            routing_spec = (self.routing_layer,)
        routing = LayerSelection.from_value(routing_spec).resolve(
            layer_count, allowed_layers=allowed
        )

        if self.consumption_profile is not None:
            consumption_spec: Any = self.consumption_profile
        elif self.consumption_layers != _DEFAULT_CONSUMPTION_LAYERS:
            consumption_spec = self.consumption_layers
        elif "consumption" in profile:
            consumption_spec = profile["consumption"]
        elif family == "gemma3" and self.consumption_layers == _DEFAULT_CONSUMPTION_LAYERS:
            consumption_spec = tuple(
                layer for layer in range(max(0, layer_count - 8), layer_count)
                if layer in allowed
            ) or (allowed[-1],)
        else:
            consumption_spec = self.consumption_layers
        consumption = LayerSelection.from_value(consumption_spec).resolve(
            layer_count, allowed_layers=allowed
        )

        detail_policy = DetailKVEncodingPolicy(self.detail_kv_encoding_policy)
        if detail_policy == DetailKVEncodingPolicy.ALL_LAYERS:
            detail = allowed
        elif detail_policy == DetailKVEncodingPolicy.EXPLICIT:
            if self.detail_kv_layers is None:
                raise ValueError("detail_kv_layers is required by the explicit detail policy.")
            detail = LayerSelection.from_value(self.detail_kv_layers).resolve(
                layer_count, allowed_layers=allowed
            )
        elif self.detail_kv_layers is not None:
            detail = LayerSelection.from_value(self.detail_kv_layers).resolve(
                layer_count, allowed_layers=allowed
            )
        else:
            # PROFILE_UNION is populated by calibration with an explicit union;
            # without one, preserving the active consumers is the bounded choice.
            detail = consumption

        address_policy = AddressEncodingPolicy(self.address_encoding_policy)
        if address_policy == AddressEncodingPolicy.EXTERNAL_ONLY:
            address = ()
            address_mode = "external"
        elif address_policy == AddressEncodingPolicy.EXPLICIT:
            if self.address_layers is None:
                raise ValueError("address_layers is required by the explicit address policy.")
            address = LayerSelection.from_value(self.address_layers).resolve(
                layer_count, allowed_layers=allowed
            )
            address_mode = self.address_mode
        elif self.address_layers is not None:
            address = LayerSelection.from_value(self.address_layers).resolve(
                layer_count, allowed_layers=allowed
            )
            address_mode = self.address_mode
        else:
            address = routing
            address_mode = self.address_mode

        return ResolvedLayerRoles(
            address_layers=address,
            detail_kv_layers=detail,
            routing_layers=routing,
            consumption_layers=consumption,
            detail_kv_encoding_policy=detail_policy.value,
            address_encoding_policy=address_policy.value,
            missing_detail_kv_policy=self.missing_detail_kv_policy,
            address_mode=address_mode,
            profile_name=(profile.get("name") if profile else self.layer_profile_name),
            profile_source=profile_source,
            registry_version=registry_version,
            model_family=family,
            model_id=self.model_id,
            model_revision=self.model_revision,
            workload_class=self.workload_class,
            materialization_class=self.materialization_class,
            objective=self.layer_profile_objective,
        )

    def product_profile_trace(self) -> dict[str, Any]:
        """Resolve product evidence without changing explicit mechanism fields."""

        requested = self.profile or self.layer_profile_objective.upper()
        trace: dict[str, Any] = {
            "profile_requested": requested,
            "profile_resolved": requested,
            "profile_source": "explicit_request" if self.profile else "layer_objective",
            "registry_version": None,
            "measurement_status": MeasurementStatus.CALIBRATION_PENDING.value,
        }
        if self.model_id is None:
            return trace
        registry = (
            ProfileBenchmarkRegistry.from_path(self.product_profile_registry)
            if self.product_profile_registry
            else ProfileBenchmarkRegistry.default()
        )
        try:
            resolution = registry.resolve(
                self.model_id,
                workload=self.workload_class,
                profile=requested,
            )
        except KeyError:
            trace["registry_version"] = registry.registry_version
            return trace
        trace.update(resolution.trace())
        trace["measurement_status"] = resolution.row["measurement_status"]
        trace["evidence_tier"] = resolution.row["evidence_tier"]
        return trace

    def resolved_layers(self, model_config_or_layer_count) -> tuple[int, tuple[int, ...]]:
        """Return the legacy primary-route/consumer view of resolved layer roles."""

        roles = self.resolved_layer_roles(model_config_or_layer_count)
        return roles.primary_routing_layer, roles.consumption_layers

    def to_internal(self, model_config_or_layer_count) -> PRAHFConfig:
        """Translate stable product fields into the shared research-core config."""
        roles = self.resolved_layer_roles(model_config_or_layer_count)
        return PRAHFConfig(
            layer_ids=roles.injected_layers,
            address_layer_ids=roles.address_layers,
            detail_kv_layer_ids=roles.detail_kv_layers,
            routing_layer_ids=roles.routing_layers,
            consumption_layer_ids=roles.consumption_layers,
            missing_detail_kv_policy=roles.missing_detail_kv_policy,
            address_mode=roles.address_mode,
            model_max_context_tokens=self.native_operation_limit,
            max_prompt_direct_tokens=self.max_direct_context,
            encoding_block_tokens=self.encoding_block_tokens,
            routing_chunk_tokens=self.chunk_tokens,
            routing_chunk_overlap_tokens=self.chunk_overlap_tokens,
            routing_representation=self.routing_representation,
            query_strategy=self.query_strategy,
            query_window=self.query_window,
            query_half_life=self.query_half_life,
            max_materialized_memory_tokens=self.max_materialized_tokens,
            context_safety_reserve_tokens=self.context_safety_reserve_tokens,
            # Product selection is applied globally after one complete ranking.
            top_k_references=1_000_000,
            top_k_chunks_per_reference=1_000_000,
            trigger_threshold=self.trigger_threshold,
            gist_mode=("segment_mean" if self.routing_mode == "local_iterative" else "mean"),
            gists_per_chunk=(
                self.chunk_tokens // (self.local_gist_tokens or min(32, self.chunk_tokens))
                if self.routing_mode == "local_iterative"
                else 1
            ),
            kv_cache_residency=self.reference_device,
            kv_cache_pin_memory=self.pin_reference_memory,
            kv_cache_non_blocking=self.non_blocking_transfer,
            collect_detailed_timing=True,
            collect_routing_metrics=True,
            store_associative_gists=(
                self.routing_mode == "local_iterative"
                or self.frontier_projection == "query"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        values = asdict(self)
        values["consumption_layers"] = list(self.consumption_layers)
        for field_name in ("routing_layers", "address_layers", "detail_kv_layers"):
            if values[field_name] is not None:
                values[field_name] = list(values[field_name])
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PRAConfig":
        """Construct a config while ignoring no unknown fields."""
        return cls(**values)

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write ``pra_config.json`` using a Hugging Face-style directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "pra_config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "PRAConfig":
        """Load ``pra_config.json`` from a local artifact directory."""
        path = Path(directory) / "pra_config.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
