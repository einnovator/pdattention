"""Generic contract implemented by thin Hugging Face attention adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..core import PRAExecutionCore
from ..memory import LayerKV, PRAMemoryCache


class PRAHFAttentionAdapter(nn.Module, ABC):
    """Delegate native attention unchanged unless routed PRA memory is active."""

    family = "unknown"

    def __init__(self, original_attention: nn.Module, cache: PRAMemoryCache, config) -> None:
        super().__init__()
        self.original_attention = original_attention
        self.layer_idx = int(original_attention.layer_idx)
        self.config = original_attention.config
        self.pra_config = config
        self.memory_enabled = False
        self.capture_enabled = False
        self.capture_position_ids: torch.Tensor | None = None
        self.captured_kv: LayerKV | None = None
        self.last_selected_chunks = []
        self.last_routing_rankings = []
        self.last_diagnostics: dict[str, float] = {}
        self.pra_core = PRAExecutionCore(
            cache=cache,
            config=config,
            layer_id=self.layer_idx,
            num_query_heads=int(original_attention.config.num_attention_heads),
            num_key_value_heads=int(original_attention.config.num_key_value_heads),
            head_dim=int(original_attention.head_dim),
        )

    @property
    def cache(self) -> PRAMemoryCache:
        """Return the shared URI-indexed PRA cache."""
        return self.pra_core.cache

    def set_memory_enabled(self, enabled: bool) -> None:
        """Enable or disable routed memory without changing native parameters."""
        self.memory_enabled = bool(enabled)

    def begin_capture(self, position_ids: torch.Tensor) -> None:
        """Capture this layer's post-position native K/V during the next prefill."""
        self.capture_enabled = True
        self.capture_position_ids = position_ids.detach().clone()
        self.captured_kv = None

    def consume_capture(self) -> LayerKV:
        """Return one completed capture and reset the temporary capture state."""
        if self.captured_kv is None:
            raise RuntimeError(f"Layer {self.layer_idx} did not capture reference K/V.")
        captured = self.captured_kv
        self.capture_enabled = False
        self.capture_position_ids = None
        self.captured_kv = None
        return captured

    @abstractmethod
    def project_qkv(self, hidden_states: torch.Tensor):
        """Apply the family's native Q/K/V projections and per-head norms."""

    @abstractmethod
    def apply_native_position_encoding(self, query, key, position_embeddings):
        """Apply the family's own positional implementation to Q/K."""

    @abstractmethod
    def normalize_qkv_layout(self, query, key, value):
        """Return canonical ``[B,H,T,Dh]`` query and native-head K/V tensors."""

    @abstractmethod
    def build_native_mask(self, local_key, local_value, prepared, attention_mask, query_tokens):
        """Combine memory and local K/V while preserving the native additive mask."""

    @abstractmethod
    def invoke_pra(self, query, key, value, attention_mask, **kwargs):
        """Invoke the family's native eager attention implementation."""

    @abstractmethod
    def project_output(self, attention_output: torch.Tensor, input_shape) -> torch.Tensor:
        """Restore hidden layout and apply the pretrained output projection."""
