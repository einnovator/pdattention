"""Model injection and bounded reference encoding for Hugging Face PRA."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..gists import GistContext, compute_gists, projected_tokens
from ..memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
)
from .adapter_base import HFRoutingCapture, PRAHFAttentionAdapter
from .config import PRAHFConfig
from .qwen import QwenPRAAttentionAdapter


@dataclass(frozen=True)
class PreparedLongPrompt:
    """Bounded direct tail plus source-relative positions after publishing ``#__head``."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    head_tokens: int


class PRAHFModel:
    """Operational handle for an injected HF model and its shared PRA cache."""

    def __init__(self, model, adapters, cache, hf_config, pra_config) -> None:
        self.model = model
        self.adapters: dict[int, PRAHFAttentionAdapter] = adapters
        self.cache = cache
        self.hf_config = hf_config
        self.pra_config = pra_config
        self.max_native_operation_tokens = 0
        self.native_limit_violations = 0

    @property
    def device(self) -> torch.device:
        """Return the device of the pretrained embedding table."""
        return self.model.get_input_embeddings().weight.device

    def set_memory_enabled(self, enabled: bool) -> None:
        """Toggle PRA across every selected layer."""
        for adapter in self.adapters.values():
            adapter.set_memory_enabled(enabled)

    def diagnostics_by_layer(self) -> dict[int, dict[str, float]]:
        """Return latest routing, materialization, GQA, and timing metrics."""
        return {layer: dict(adapter.last_diagnostics) for layer, adapter in self.adapters.items()}

    def _record_native_operation(self, tokens: int) -> None:
        self.max_native_operation_tokens = max(self.max_native_operation_tokens, int(tokens))
        if tokens > self.pra_config.effective_model_max_context_tokens:
            self.native_limit_violations += 1
            raise ValueError("Reference encoding exceeded the configured native-operation limit.")

    def _resident_kv(self, kv: LayerKV) -> LayerKV:
        """Move full cache K/V to configured residency without expanding GQA heads."""
        if self.pra_config.kv_cache_residency == "gpu":
            return kv
        key = kv.k.detach().to("cpu")
        value = kv.v.detach().to("cpu")
        if self.pra_config.kv_cache_pin_memory and torch.cuda.is_available():
            key = key.pin_memory()
            value = value.pin_memory()
        return LayerKV(key, value, kv.position_ids, kv.position_state)

    def _routing_points(
        self,
        captured: HFRoutingCapture,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return token routing features while leaving detail K/V post-RoPE."""
        representation = self.hf_config.routing_representation
        detail = captured.detail_kv
        if representation == "post_rope_key":
            keys = projected_tokens(detail.k[:, :, start:end, :])
            values = projected_tokens(detail.v[:, :, start:end, :])
        elif representation == "pre_rope_key":
            keys = projected_tokens(captured.pre_key[:, :, start:end, :])
            values = projected_tokens(detail.v[:, :, start:end, :])
        elif representation == "hidden_state":
            keys = captured.hidden_states[0, start:end, :]
            values = None
        else:
            raise ValueError(f"Unsupported routing representation: {representation}")
        return keys, values

    @torch.no_grad()
    def add_reference(
        self,
        uri: str,
        input_ids: torch.Tensor,
        *,
        text: str = "",
    ) -> PRACacheEntry:
        """Encode one logical source in bounded blocks and publish routable native K/V."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("HF reference encoding currently accepts one tokenized source.")
        input_ids = input_ids.to(self.device)
        total = int(input_ids.shape[1])
        if total == 0:
            raise ValueError("Cannot publish an empty PRA reference.")
        entry = PRACacheEntry(
            uri=uri,
            text=text,
            metadata={
                "hf_family": next(iter(self.adapters.values())).family,
                "source_tokens": total,
                "encoding_block_tokens": self.hf_config.encoding_block_tokens,
                "routing_chunk_tokens": self.hf_config.routing_chunk_tokens,
                "routing_representation": self.hf_config.routing_representation,
                "position_state": "post_position",
            },
        )
        prior_memory_state = {layer: adapter.memory_enabled for layer, adapter in self.adapters.items()}
        self.set_memory_enabled(False)
        try:
            for block_start in range(0, total, self.hf_config.encoding_block_tokens):
                block_end = min(block_start + self.hf_config.encoding_block_tokens, total)
                block_ids = input_ids[:, block_start:block_end]
                self._record_native_operation(int(block_ids.shape[1]))
                positions = torch.arange(block_start, block_end, device=self.device).unsqueeze(0)
                for adapter in self.adapters.values():
                    adapter.begin_capture(positions)
                self.model(
                    input_ids=block_ids,
                    attention_mask=torch.ones_like(block_ids),
                    position_ids=positions,
                    use_cache=False,
                )
                captures = {layer: adapter.consume_capture() for layer, adapter in self.adapters.items()}
                step = self.hf_config.routing_chunk_tokens - self.hf_config.routing_chunk_overlap_tokens
                if step <= 0:
                    raise ValueError("Routing overlap must be smaller than routing chunk size.")
                for local_start in range(0, int(block_ids.shape[1]), step):
                    local_end = min(local_start + self.hf_config.routing_chunk_tokens, int(block_ids.shape[1]))
                    logical_start = block_start + local_start
                    logical_end = block_start + local_end
                    chunk_token_ids = block_ids[0, local_start:local_end].tolist()
                    for layer, captured in captures.items():
                        detail = captured.detail_kv
                        kv = LayerKV(
                            k=detail.k[:, :, local_start:local_end, :],
                            v=detail.v[:, :, local_start:local_end, :],
                            position_ids=positions[:, local_start:local_end],
                            position_state="post_position",
                        )
                        routing_keys, routing_values = self._routing_points(
                            captured, local_start, local_end
                        )
                        computed = compute_gists(
                            keys=routing_keys,
                            values=routing_values,
                            mode=self.pra_config.gist_mode,
                            num_gists=self.pra_config.gists_per_chunk,
                            config=self.pra_config,
                            context=GistContext(level="chunk", token_ids=chunk_token_ids),
                        )
                        chunk = ReferenceChunkMemory(
                            chunk_id=f"{uri}#chunk={logical_start}:{logical_end}",
                            source_uri=uri,
                            token_start=logical_start,
                            token_end=logical_end,
                            logical_start=logical_start,
                            logical_end=logical_end,
                            token_kv=self._resident_kv(kv),
                            routing_gist=ChunkRoutingGist(
                                k=computed.k.detach(),
                                v=computed.v.detach() if computed.v is not None else None,
                                method=self.pra_config.gist_mode,
                                metadata=computed.metadata,
                            ),
                            metadata={
                                "encoding_block_start": block_start,
                                "routing_representation": self.hf_config.routing_representation,
                                "routing_gist_bytes": int(
                                    computed.k.numel() * computed.k.element_size()
                                ),
                                "detail_kv_bytes": int(
                                    (kv.k.numel() * kv.k.element_size())
                                    + (kv.v.numel() * kv.v.element_size())
                                ),
                            },
                        )
                        entry.layer_memory.setdefault(layer, LayerReferenceMemory()).chunks.append(chunk)
                    if local_end == int(block_ids.shape[1]):
                        break
        finally:
            for layer, enabled in prior_memory_state.items():
                self.adapters[layer].set_memory_enabled(enabled)
        self.cache.put(entry)
        return entry

    def prepare_long_prompt(self, input_ids: torch.Tensor) -> PreparedLongPrompt:
        """Publish displaced prompt history as ``#__head`` and return a bounded tail."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        direct = self.pra_config.effective_prompt_direct_tokens
        if input_ids.shape[1] <= direct:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            return PreparedLongPrompt(input_ids, torch.ones_like(input_ids), positions, 0)
        head_tokens = int(input_ids.shape[1] - direct)
        self.add_reference("#__head", input_ids[:, :head_tokens], text="implicit prompt head")
        tail = input_ids[:, head_tokens:].to(self.device)
        positions = torch.arange(head_tokens, head_tokens + tail.shape[1], device=self.device).unsqueeze(0)
        return PreparedLongPrompt(tail, torch.ones_like(tail), positions, head_tokens)


def _normalize_layer_ids(layer_ids, layer_count: int) -> tuple[int, ...]:
    normalized = tuple(sorted({layer_count + item if item < 0 else item for item in layer_ids}))
    if not normalized or normalized[0] < 0 or normalized[-1] >= layer_count:
        raise ValueError("PRA layer selection is empty or outside the decoder stack.")
    return normalized


def inject_pra(model, config: PRAHFConfig | None = None) -> PRAHFModel:
    """Wrap selected Qwen attention modules while reusing every pretrained parameter."""
    config = config or PRAHFConfig()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a Hugging Face decoder model exposing model.layers.")
    if getattr(model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("Paper 2 correctness integration requires attn_implementation='eager'.")
    layers = model.model.layers
    selected = _normalize_layer_ids(config.layer_ids, len(layers))
    pra_config = config.build_pra_config(model.config)
    pra_config.pra_layer_ids = selected
    cache = PRASimpleMemoryCache()
    adapters: dict[int, PRAHFAttentionAdapter] = {}
    for layer_id in selected:
        original = layers[layer_id].self_attn
        if ".qwen2." not in original.__class__.__module__ and ".qwen3." not in original.__class__.__module__:
            raise TypeError("Only Qwen2/Qwen2.5/Qwen3 is implemented in the first Paper 2 milestone.")
        adapter = QwenPRAAttentionAdapter(
            original,
            cache,
            pra_config,
            routing_representation=config.routing_representation,
        )
        layers[layer_id].self_attn = adapter
        adapters[layer_id] = adapter
    return PRAHFModel(model, adapters, cache, config, pra_config)
