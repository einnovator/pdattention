"""Stable user-facing configuration for the PRA Hugging Face integration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig


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
    routing_layer: int = -1
    consumption_layers: tuple[int, ...] = _DEFAULT_CONSUMPTION_LAYERS
    routing_representation: str = ATTENTION_INPUT_HIDDEN_STATE
    chunk_tokens: int = 32
    chunk_overlap_tokens: int = 0
    local_gist_tokens: int | None = None
    selected_fraction: float | None = 0.20
    top_k: int = 8
    max_direct_context: int = 256
    native_operation_limit: int = 512
    max_materialized_tokens: int = 256
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
        self.consumption_layers = tuple(int(layer) for layer in self.consumption_layers)
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

    def resolved_layers(self, model_config_or_layer_count) -> tuple[int, tuple[int, ...]]:
        """Resolve layer IDs while preserving a host model's native scope.

        Ordinary decoder families resolve negative indices against the complete
        stack. Gemma 3 defaults instead select the late native-global layers;
        explicit local-layer requests are rejected rather than silently turning
        sliding attention into full external-memory attention.
        """
        if isinstance(model_config_or_layer_count, int):
            layer_count = model_config_or_layer_count
            layer_types: tuple[str, ...] = ()
        else:
            model_config = model_config_or_layer_count
            layer_count = int(model_config.num_hidden_layers)
            layer_types = (
                tuple(getattr(model_config, "layer_types", ()) or ())
                if getattr(model_config, "model_type", None) == "gemma3_text"
                else ()
            )

        def resolve(value: int) -> int:
            result = layer_count + value if value < 0 else value
            if result < 0 or result >= layer_count:
                raise ValueError(f"Layer {value} is outside a {layer_count}-layer model.")
            return result

        if not layer_types:
            routing = resolve(self.routing_layer)
            consumption = tuple(sorted({resolve(layer) for layer in self.consumption_layers}))
            return routing, consumption

        global_layers = tuple(
            index for index, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        )
        if not global_layers:
            raise ValueError("Gemma 3 exposes no native full-attention layers.")
        routing = (
            global_layers[-1]
            if self.routing_layer == -1
            else resolve(self.routing_layer)
        )
        if routing not in global_layers:
            raise ValueError(
                f"Gemma 3 routing layer {routing} is sliding attention; "
                f"choose one of the native global layers {global_layers}."
            )
        if self.consumption_layers == _DEFAULT_CONSUMPTION_LAYERS:
            requested = tuple(range(max(0, layer_count - 8), layer_count))
            consumption = tuple(layer for layer in requested if layer in global_layers)
            if not consumption:
                consumption = (global_layers[-1],)
        else:
            requested = tuple(sorted({resolve(layer) for layer in self.consumption_layers}))
            local = tuple(layer for layer in requested if layer not in global_layers)
            if local:
                raise ValueError(
                    f"Gemma 3 PRA consumption layers {local} are sliding attention; "
                    f"choose native global layers from {global_layers}."
                )
            consumption = requested
        return routing, consumption

    def to_internal(self, model_config_or_layer_count) -> PRAHFConfig:
        """Translate stable product fields into the shared research-core config."""
        routing, consumption = self.resolved_layers(model_config_or_layer_count)
        layers = tuple(sorted({routing, *consumption}))
        return PRAHFConfig(
            layer_ids=layers,
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
