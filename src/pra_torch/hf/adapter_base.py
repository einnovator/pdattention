"""Generic contract implemented by thin Hugging Face attention adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..core import PRAExecutionCore
from ..memory import LayerKV, PRAMemoryCache, SelectedChunk


@dataclass(frozen=True)
class HFRoutingCapture:
    """Matched routing features and native detail K/V from one bounded prefill.

    ``pre_query`` and ``post_query`` use ``[1,H_q,T,Dh]``; ``pre_key`` and
    ``detail_kv`` use native ``[1,H_kv,T,Dh]``; ``hidden_states`` uses
    ``[1,T,D_model]``.
    Full pre-RoPE tensors are transient and are discarded after gist pooling.
    """

    pre_query: torch.Tensor
    post_query: torch.Tensor
    pre_key: torch.Tensor
    hidden_states: torch.Tensor
    detail_kv: LayerKV


class PRAHFAttentionAdapter(nn.Module, ABC):
    """Delegate native attention unchanged unless routed PRA memory is active."""

    family = "unknown"

    def __init__(
        self,
        original_attention: nn.Module,
        cache: PRAMemoryCache,
        config,
        memory_gate=None,
        residual_adapter=None,
    ) -> None:
        super().__init__()
        self.original_attention = original_attention
        self.layer_idx = int(original_attention.layer_idx)
        self.config = original_attention.config
        self.pra_config = config
        self.memory_enabled = False
        self.capture_enabled = False
        self.capture_position_ids: torch.Tensor | None = None
        self.captured_routing: HFRoutingCapture | None = None
        self.last_selected_chunks = []
        self.last_routing_rankings = []
        self.last_diagnostics: dict[str, float] = {}
        self.fixed_selected_chunks: list[list[SelectedChunk]] | None = None
        self.collect_attention_diagnostics = False
        self.last_attention_weights: torch.Tensor | None = None
        # The owning HF model registers this module exactly once.
        self.__dict__["memory_gate"] = memory_gate
        self.__dict__["residual_adapter"] = residual_adapter
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

    def set_fixed_selected_chunks(
        self,
        selections: list[list[SelectedChunk]] | None,
    ) -> None:
        """Override semantic routing with explicit row-local chunk identities.

        The override stops at the selection boundary. Thresholding, budgeting,
        native-K/V transfer, masking, and attention remain on the ordinary PRA
        path, which makes this suitable for oracle and router-control studies.
        """
        self.fixed_selected_chunks = (
            None if selections is None else [list(row) for row in selections]
        )

    def set_attention_diagnostics(self, enabled: bool) -> None:
        """Opt into retaining the latest eager-kernel attention probabilities."""
        self.collect_attention_diagnostics = bool(enabled)
        if not enabled:
            self.last_attention_weights = None

    def begin_capture(self, position_ids: torch.Tensor) -> None:
        """Capture this layer's post-position native K/V during the next prefill."""
        self.capture_enabled = True
        self.capture_position_ids = position_ids.detach().clone()
        self.captured_routing = None

    def consume_capture(self) -> HFRoutingCapture:
        """Return one completed routing/detail capture and reset temporary state."""
        if self.captured_routing is None:
            raise RuntimeError(f"Layer {self.layer_idx} did not capture reference K/V.")
        captured = self.captured_routing
        self.capture_enabled = False
        self.capture_position_ids = None
        self.captured_routing = None
        return captured

    @abstractmethod
    def project_qkv(self, hidden_states: torch.Tensor):
        """Apply the family's native Q/K/V projections and per-head norms."""

    @abstractmethod
    def apply_native_position_encoding(self, query, key, position_embeddings):
        """Apply the family's own positional implementation to Q/K."""

    @abstractmethod
    def rotate_routing_keys(self, flattened_keys, positions):
        """Position pooled native-key gists using the family's own encoding."""

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
