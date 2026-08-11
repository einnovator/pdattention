"""Stable user-facing configuration for the PRA Hugging Face integration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig


@dataclass
class PRAConfig:
    """Configure one-shot semantic routing and bounded native-K/V consumption.

    ``selected_fraction`` takes precedence over ``top_k`` when it is not
    ``None``. Negative layer indices are resolved relative to the decoder depth.
    """

    enabled: bool = True
    routing_layer: int = -1
    consumption_layers: tuple[int, ...] = tuple(range(-8, 0))
    routing_representation: str = ATTENTION_INPUT_HIDDEN_STATE
    chunk_tokens: int = 32
    chunk_overlap_tokens: int = 0
    selected_fraction: float | None = 0.20
    top_k: int = 8
    max_direct_context: int = 256
    native_operation_limit: int = 512
    max_materialized_tokens: int = 256
    context_safety_reserve_tokens: int = 4
    encoding_block_tokens: int = 256
    reference_device: str = "cpu"
    pin_reference_memory: bool = False
    non_blocking_transfer: bool = False
    query_strategy: str = "last"
    query_window: int = 16
    query_half_life: float = 4.0
    trigger_threshold: float = float("-inf")

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
        if not 0 <= self.chunk_overlap_tokens < self.chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be in [0, chunk_tokens).")
        if self.reference_device not in {"cpu", "gpu"}:
            raise ValueError("reference_device must be 'cpu' or 'gpu'.")

    @property
    def selection_policy(self) -> str:
        """Return the active budget policy, including documented precedence."""
        return "selected_fraction" if self.selected_fraction is not None else "top_k"

    def resolved_layers(self, layer_count: int) -> tuple[int, tuple[int, ...]]:
        """Resolve routing and consumption layer IDs for a concrete model."""
        def resolve(value: int) -> int:
            result = layer_count + value if value < 0 else value
            if result < 0 or result >= layer_count:
                raise ValueError(f"Layer {value} is outside a {layer_count}-layer model.")
            return result

        routing = resolve(self.routing_layer)
        consumption = tuple(sorted({resolve(layer) for layer in self.consumption_layers}))
        return routing, consumption

    def to_internal(self, layer_count: int) -> PRAHFConfig:
        """Translate stable product fields into the shared research-core config."""
        routing, consumption = self.resolved_layers(layer_count)
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
            gist_mode="mean",
            gists_per_chunk=1,
            kv_cache_residency=self.reference_device,
            kv_cache_pin_memory=self.pin_reference_memory,
            kv_cache_non_blocking=self.non_blocking_transfer,
            collect_detailed_timing=True,
            collect_routing_metrics=True,
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
