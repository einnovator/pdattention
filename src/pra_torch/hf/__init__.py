"""Thin Hugging Face adapters backed by the shared PRA execution core."""

from .config import (
    ATTENTION_INPUT_HIDDEN_STATE,
    CENTERED_ROPE_KEY,
    PRAHFConfig,
    canonical_routing_representation,
)
from .injection import PRAHFModel, inject_pra
from .anatomy import HFDecoderAnatomy, resolve_decoder_anatomy
from .memory_gate import (
    MEMORY_GATE_FIXED,
    MEMORY_GATE_PER_LAYER,
    MEMORY_GATE_SINGLE,
    PRAHFMemoryGate,
)
from .residual_adapter import PRAHFResidualAdapter, PRAHFResidualAdapterBank
from .late_band_lora import (
    PRAHFConditionalOutputLoRA,
    PRAHFConditionalOutputLoRABank,
)
from .llama import LlamaPRAAttentionAdapter
from .qwen3_moe import Qwen3MoePRAAttentionAdapter
from .gemma3 import Gemma3PRAAttentionAdapter, gemma3_global_layer_ids
from .moe import HFMoETopology, describe_moe_topology
from .mistral import MistralPRAAttentionAdapter
from .gpt_oss import GptOssPRAAttentionAdapter, gpt_oss_full_layer_ids
from .qwen3_hybrid import Qwen3HybridPRAPlan, describe_qwen3_hybrid_plan
from .query import (
    QUERY_EXPONENTIAL,
    QUERY_LAST,
    QUERY_LINEAR,
    QUERY_QUESTION_EXPONENTIAL,
    QUERY_QUESTION_MEAN,
    QUERY_STRATEGIES,
    QUERY_UNIFORM,
    aggregate_query_states,
    half_life_to_decay,
    streaming_exponential_query,
    token_span_from_offsets,
)
from .routing_adapter import HFRoutingProjection, load_hf_routing_projection

__all__ = [
    "ATTENTION_INPUT_HIDDEN_STATE",
    "CENTERED_ROPE_KEY",
    "PRAHFConfig",
    "PRAHFModel",
    "canonical_routing_representation",
    "inject_pra",
    "HFDecoderAnatomy",
    "resolve_decoder_anatomy",
    "MEMORY_GATE_FIXED",
    "MEMORY_GATE_SINGLE",
    "MEMORY_GATE_PER_LAYER",
    "PRAHFMemoryGate",
    "PRAHFResidualAdapter",
    "PRAHFResidualAdapterBank",
    "PRAHFConditionalOutputLoRA",
    "PRAHFConditionalOutputLoRABank",
    "LlamaPRAAttentionAdapter",
    "Qwen3MoePRAAttentionAdapter",
    "Gemma3PRAAttentionAdapter",
    "gemma3_global_layer_ids",
    "HFMoETopology",
    "describe_moe_topology",
    "MistralPRAAttentionAdapter",
    "GptOssPRAAttentionAdapter",
    "gpt_oss_full_layer_ids",
    "Qwen3HybridPRAPlan",
    "describe_qwen3_hybrid_plan",
    "QUERY_EXPONENTIAL",
    "QUERY_LAST",
    "QUERY_LINEAR",
    "QUERY_QUESTION_EXPONENTIAL",
    "QUERY_QUESTION_MEAN",
    "QUERY_STRATEGIES",
    "QUERY_UNIFORM",
    "aggregate_query_states",
    "half_life_to_decay",
    "streaming_exponential_query",
    "token_span_from_offsets",
    "HFRoutingProjection",
    "load_hf_routing_projection",
]
