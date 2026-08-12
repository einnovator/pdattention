"""Configuration boundary between Hugging Face models and the PRA core."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import PRAConfig
from .memory_gate import MEMORY_GATE_FIXED, MEMORY_GATE_MODES
from .query import QUERY_LAST, RUNTIME_QUERY_STRATEGIES


ATTENTION_INPUT_HIDDEN_STATE = "attention_input_hidden_state"
CENTERED_ROPE_KEY = "centered_rope_key"
_ROUTING_REPRESENTATION_ALIASES = {
    "hidden_state": ATTENTION_INPUT_HIDDEN_STATE,
}


def canonical_routing_representation(value: str) -> str:
    """Return the explicit persistent name for a supported HF routing space."""
    return _ROUTING_REPRESENTATION_ALIASES.get(value, value)


@dataclass
class PRAHFConfig:
    """Small HF-facing configuration that expands into canonical ``PRAConfig``."""

    layer_ids: tuple[int, ...] = (-1,)
    model_max_context_tokens: int | None = None
    max_prompt_direct_tokens: int | None = None
    encoding_block_tokens: int = 512
    routing_chunk_tokens: int = 128
    routing_chunk_overlap_tokens: int = 0
    routing_representation: str = ATTENTION_INPUT_HIDDEN_STATE
    query_strategy: str = QUERY_LAST
    query_window: int = 16
    query_half_life: float = 4.0
    centered_rope_center_policy: str = "exact"
    max_materialized_memory_tokens: int = 512
    context_safety_reserve_tokens: int = 0
    top_k_references: int = 2
    top_k_chunks_per_reference: int = 1
    trigger_threshold: float = 0.2
    gist_mode: str = "mean"
    gists_per_chunk: int = 1
    kv_cache_residency: str = "cpu"
    kv_cache_pin_memory: bool = False
    kv_cache_non_blocking: bool = False
    collect_detailed_timing: bool = True
    collect_routing_metrics: bool = False
    memory_gate_mode: str = MEMORY_GATE_FIXED
    memory_gate_initial_value: float = 1.0
    residual_adapter_bottleneck: int = 0

    def __post_init__(self) -> None:
        """Reject routing modes that the installed family adapter cannot represent."""
        self.routing_representation = canonical_routing_representation(
            self.routing_representation
        )
        supported = {
            "post_rope_key",
            "pre_rope_key",
            CENTERED_ROPE_KEY,
            ATTENTION_INPUT_HIDDEN_STATE,
        }
        if self.routing_representation not in supported:
            raise ValueError(
                f"Unsupported HF routing representation: {self.routing_representation}"
            )
        if self.query_strategy not in RUNTIME_QUERY_STRATEGIES:
            raise ValueError(
                "HF runtime query_strategy must be last, uniform, exponential, or linear."
            )
        if self.query_window <= 0:
            raise ValueError("query_window must be positive.")
        if self.query_half_life <= 0:
            raise ValueError("query_half_life must be positive.")
        if self.centered_rope_center_policy not in {"exact", "floor", "ceil"}:
            raise ValueError(
                "centered_rope_center_policy must be 'exact', 'floor', or 'ceil'."
            )
        if self.routing_representation == CENTERED_ROPE_KEY and self.gist_mode not in {
            "mean",
            "segment_mean",
        }:
            raise ValueError(
                "centered_rope_key supports gist_mode='mean' or 'segment_mean'."
            )
        if self.memory_gate_mode not in MEMORY_GATE_MODES:
            raise ValueError(
                f"Unsupported HF PRA memory_gate_mode: {self.memory_gate_mode}"
            )
        if int(self.residual_adapter_bottleneck) < 0:
            raise ValueError("residual_adapter_bottleneck must be non-negative.")

    def build_pra_config(self, hf_config) -> PRAConfig:
        """Translate native model dimensions without changing pretrained parameters."""
        max_positions = int(getattr(hf_config, "max_position_embeddings"))
        hard_limit = int(self.model_max_context_tokens or max_positions)
        if hard_limit > max_positions:
            raise ValueError("PRA's native-operation limit cannot exceed the HF model limit.")
        direct_limit = int(self.max_prompt_direct_tokens or hard_limit)
        if self.encoding_block_tokens > hard_limit:
            raise ValueError("encoding_block_tokens must fit the native-operation limit.")
        if self.routing_chunk_tokens <= 0:
            raise ValueError("routing_chunk_tokens must be positive.")
        return PRAConfig(
            vocab_size=int(hf_config.vocab_size),
            d_model=int(hf_config.hidden_size),
            n_heads=int(hf_config.num_attention_heads),
            n_layers=int(hf_config.num_hidden_layers),
            d_ff=int(hf_config.intermediate_size),
            max_seq_len=max_positions,
            model_max_context_tokens=hard_limit,
            max_prompt_direct_tokens=direct_limit,
            prompt_overflow_mode="implicit_reference",
            position_encoding="rope",
            rope_theta=float(getattr(hf_config, "rope_theta", 10_000.0)),
            model_variant="custom",
            pra_layer_ids=tuple(self.layer_ids),
            top_k_references=self.top_k_references,
            top_k_chunks_per_reference=self.top_k_chunks_per_reference,
            trigger_threshold=self.trigger_threshold,
            memory_transport="native_kv",
            max_materialized_memory_tokens=self.max_materialized_memory_tokens,
            context_safety_reserve_tokens=self.context_safety_reserve_tokens,
            gist_mode=self.gist_mode,
            gists_per_chunk=self.gists_per_chunk,
            reference_position_mode="global",
            prompt_position_mode="historical",
            reference_encoding_strategy="native_slice",
            chunking_mode="fixed",
            fixed_chunk_tokens=self.routing_chunk_tokens,
            fixed_chunk_overlap_tokens=self.routing_chunk_overlap_tokens,
            kv_cache_residency=self.kv_cache_residency,
            kv_cache_pin_memory=self.kv_cache_pin_memory,
            kv_cache_non_blocking=self.kv_cache_non_blocking,
            collect_detailed_timing=self.collect_detailed_timing,
            collect_routing_metrics=self.collect_routing_metrics,
        )
