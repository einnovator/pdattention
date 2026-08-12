"""Model injection and bounded reference encoding for Hugging Face PRA."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..gists import GistContext, compute_gists, projected_tokens
from ..memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
    SelectedChunk,
)
from .adapter_base import HFRoutingCapture, PRAHFAttentionAdapter
from .config import (
    ATTENTION_INPUT_HIDDEN_STATE,
    CENTERED_ROPE_KEY,
    PRAHFConfig,
    canonical_routing_representation,
)
from .qwen import QwenPRAAttentionAdapter
from .llama import LlamaPRAAttentionAdapter
from .gemma3 import Gemma3PRAAttentionAdapter
from .memory_gate import PRAHFMemoryGate
from .residual_adapter import PRAHFResidualAdapterBank
from .late_band_lora import PRAHFConditionalOutputLoRABank


@dataclass(frozen=True)
class PreparedLongPrompt:
    """Bounded direct tail plus source-relative positions after publishing ``#__head``."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    head_tokens: int


class PRAHFModel:
    """Operational handle for an injected HF model and its shared PRA cache."""

    def __init__(
        self,
        model,
        adapters,
        cache,
        hf_config,
        pra_config,
        routing_projection=None,
        memory_gate=None,
        residual_adapter=None,
        late_band_lora=None,
    ) -> None:
        self.model = model
        self.adapters: dict[int, PRAHFAttentionAdapter] = adapters
        self.cache = cache
        self.hf_config = hf_config
        self.pra_config = pra_config
        self.routing_projection = routing_projection
        self.memory_gate = memory_gate
        self.residual_adapter = residual_adapter
        self.late_band_lora = late_band_lora
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

    def configure_memory_gate(
        self,
        mode: str,
        *,
        initial_value: float | None = None,
    ) -> None:
        """Select fixed, shared-scalar, or per-layer memory calibration."""
        self.memory_gate.configure(mode, initial_value=initial_value)

    def memory_gate_parameters(self) -> list[torch.nn.Parameter]:
        """Return the active gate variant's complete trainable parameter set."""
        return self.memory_gate.trainable_parameters()

    def memory_gate_values(self) -> dict[int, float]:
        """Return effective per-layer scales for checkpoints and diagnostics."""
        return self.memory_gate.values()

    def configure_residual_adapter(
        self,
        bottleneck: int,
        *,
        reset: bool = True,
    ) -> None:
        """Enable a late-layer bottleneck correction or disable it with zero."""
        self.residual_adapter.configure(bottleneck, reset=reset)

    def residual_adapter_parameters(self) -> list[torch.nn.Parameter]:
        """Return the active residual adapter's complete parameter set."""
        return self.residual_adapter.trainable_parameters()

    def configure_late_band_lora(
        self,
        rank: int,
        *,
        alpha: float | None = None,
        dropout: float = 0.0,
        reset: bool = True,
    ) -> None:
        """Enable conditional output-projection LoRA or disable it with rank zero."""
        self.late_band_lora.configure(
            rank,
            alpha=alpha,
            dropout=dropout,
            reset=reset,
        )

    def late_band_lora_parameters(self) -> list[torch.nn.Parameter]:
        """Return only factors belonging to the active conditional LoRA rank."""
        return self.late_band_lora.trainable_parameters()

    def memory_use_parameters(self) -> list[torch.nn.Parameter]:
        """Return every active PRA-conditional memory-use parameter.

        Residual correction and conditional output LoRA are independent banks.
        Returning both makes their joint ownership explicit for matched
        optimization while all pretrained and router parameters stay frozen.
        """
        return [
            *self.residual_adapter_parameters(),
            *self.late_band_lora_parameters(),
        ]

    def configure_memory_layers(
        self,
        layer_ids: set[int] | tuple[int, ...],
        *,
        fixed_selections: dict[int, list[list]] | None = None,
    ) -> None:
        """Activate a layer subset and optionally force its selected identities.

        Omitting ``fixed_selections`` uses each active layer's configured router.
        Supplying it makes listed layers replay those identities through the same
        materialization and attention path. Inactive layers always delegate to
        the original Hugging Face attention module.
        """
        active = {int(layer_id) for layer_id in layer_ids}
        unknown = active.difference(self.adapters)
        if unknown:
            raise ValueError(f"PRA layers were not injected: {sorted(unknown)}")
        fixed_selections = fixed_selections or {}
        for layer_id, adapter in self.adapters.items():
            adapter.set_memory_enabled(layer_id in active)
            adapter.set_fixed_selected_chunks(fixed_selections.get(layer_id))

    def set_attention_diagnostics(self, enabled: bool) -> None:
        """Toggle opt-in eager attention-probability capture on injected layers."""
        for adapter in self.adapters.values():
            adapter.set_attention_diagnostics(enabled)

    def map_chunk_identities_to_layers(
        self,
        selections: list[list[SelectedChunk]],
        layer_ids: tuple[int, ...] | set[int],
    ) -> dict[int, list[list[SelectedChunk]]]:
        """Reuse selected parent IDs while resolving each layer's native K/V.

        Chunk IDs and source spans are stable across layer-specific cache views.
        Only that identity is shared: every returned hit owns the target layer's
        independently projected and positioned K/V payload.
        """
        targets = tuple(sorted({int(layer_id) for layer_id in layer_ids}))
        unknown = set(targets).difference(self.adapters)
        if unknown:
            raise ValueError(f"PRA layers were not injected: {sorted(unknown)}")
        mapped: dict[int, list[list[SelectedChunk]]] = {}
        for layer_id in targets:
            layer_rows: list[list[SelectedChunk]] = []
            for row in selections:
                mapped_row = []
                for hit in row:
                    memory = hit.entry.layer_memory.get(layer_id)
                    if memory is None:
                        raise ValueError(
                            f"Reference {hit.reference_uri} has no K/V for layer {layer_id}."
                        )
                    by_id = {chunk.chunk_id: chunk for chunk in memory.chunks}
                    chunk = by_id.get(hit.chunk_id)
                    if chunk is None:
                        raise ValueError(
                            f"Chunk {hit.chunk_id} has no payload at layer {layer_id}."
                        )
                    mapped_row.append(
                        replace(
                            hit,
                            chunk=chunk,
                            layer_id=layer_id,
                            metadata={
                                **hit.metadata,
                                "selection_source_layer": hit.layer_id,
                                "identity_reused_across_layers": True,
                            },
                        )
                    )
                layer_rows.append(mapped_row)
            mapped[layer_id] = layer_rows
        return mapped

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
        representation = canonical_routing_representation(
            self.hf_config.routing_representation
        )
        detail = captured.detail_kv
        if representation == "post_rope_key":
            keys = projected_tokens(detail.k[:, :, start:end, :])
            values = projected_tokens(detail.v[:, :, start:end, :])
        elif representation in {"pre_rope_key", CENTERED_ROPE_KEY}:
            keys = projected_tokens(captured.pre_key[:, :, start:end, :])
            values = projected_tokens(detail.v[:, :, start:end, :])
        elif representation == ATTENTION_INPUT_HIDDEN_STATE:
            keys = captured.hidden_states[0, start:end, :]
            values = None
        else:
            raise ValueError(f"Unsupported routing representation: {representation}")
        return keys, values

    def _centered_rope_gists(
        self,
        adapter: PRAHFAttentionAdapter,
        computed,
        *,
        logical_start: int,
        token_count: int,
    ):
        """Place pooled pre-RoPE key gists at exact known source-span centers."""
        spans = computed.metadata.get("segment_token_spans")
        if spans is None:
            spans = [[0, token_count]]
        if len(spans) != int(computed.k.shape[0]):
            raise ValueError("Centered gist spans must match the computed gist count.")
        exact_centers = torch.tensor(
            [
                logical_start + (int(start) + int(end) - 1) / 2.0
                for start, end in spans
            ],
            device=computed.k.device,
            dtype=torch.float32,
        )
        policy = self.hf_config.centered_rope_center_policy
        if policy == "floor":
            applied_centers = exact_centers.floor()
        elif policy == "ceil":
            applied_centers = exact_centers.ceil()
        else:
            applied_centers = exact_centers
        computed.k = adapter.rotate_routing_keys(computed.k, applied_centers)
        computed.metadata.update(
            {
                "gist_position_policy": "span_center",
                "center_rounding_policy": policy,
                "exact_center_positions": exact_centers.detach().cpu().tolist(),
                "applied_center_positions": applied_centers.detach().cpu().tolist(),
                "source_token_spans": [
                    [logical_start + int(start), logical_start + int(end)]
                    for start, end in spans
                ],
            }
        )
        return computed

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
                        if self.hf_config.routing_representation == CENTERED_ROPE_KEY:
                            computed = self._centered_rope_gists(
                                self.adapters[layer],
                                computed,
                                logical_start=logical_start,
                                token_count=local_end - local_start,
                            )
                        if self.routing_projection is not None:
                            computed.k = self.routing_projection.project_memory(computed.k)
                            computed.metadata.update(
                                {
                                    "routing_projection": self.routing_projection.architecture,
                                    "routing_projection_width": self.routing_projection.routing_width,
                                }
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


def inject_pra(
    model,
    config: PRAHFConfig | None = None,
    *,
    routing_projection=None,
) -> PRAHFModel:
    """Wrap selected supported attention modules while reusing pretrained parameters."""
    config = config or PRAHFConfig()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a Hugging Face decoder model exposing model.layers.")
    if getattr(model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("Paper 2 correctness integration requires attn_implementation='eager'.")
    layers = model.model.layers
    selected = _normalize_layer_ids(config.layer_ids, len(layers))
    pra_config = config.build_pra_config(model.config)
    if routing_projection is not None:
        if config.routing_representation != ATTENTION_INPUT_HIDDEN_STATE:
            raise ValueError("Learned routing projections require hidden-state routing.")
        if routing_projection.input_width != int(model.config.hidden_size):
            raise ValueError("Routing projection input width must match the HF hidden size.")
    pra_config.pra_layer_ids = selected
    cache = PRASimpleMemoryCache()
    memory_gate = PRAHFMemoryGate(
        selected,
        mode=config.memory_gate_mode,
        initial_value=config.memory_gate_initial_value,
    ).to(model.get_input_embeddings().weight.device)
    model.add_module("pra_memory_gate", memory_gate)
    residual_adapter = PRAHFResidualAdapterBank(
        int(model.config.hidden_size),
        selected,
        bottleneck=config.residual_adapter_bottleneck,
    ).to(model.get_input_embeddings().weight.device)
    model.add_module("pra_residual_adapter", residual_adapter)
    sample_output_projection = layers[selected[0]].self_attn.o_proj
    late_band_lora = PRAHFConditionalOutputLoRABank(
        int(sample_output_projection.in_features),
        int(sample_output_projection.out_features),
        selected,
        rank=config.late_band_lora_rank,
        alpha=config.late_band_lora_alpha,
        dropout=config.late_band_lora_dropout,
    ).to(model.get_input_embeddings().weight.device)
    model.add_module("pra_late_band_lora", late_band_lora)
    adapters: dict[int, PRAHFAttentionAdapter] = {}
    for layer_id in selected:
        original = layers[layer_id].self_attn
        module_name = original.__class__.__module__
        if ".qwen2." in module_name or ".qwen3." in module_name:
            adapter_class = QwenPRAAttentionAdapter
        elif ".llama." in module_name:
            adapter_class = LlamaPRAAttentionAdapter
        elif ".gemma3." in module_name:
            adapter_class = Gemma3PRAAttentionAdapter
        else:
            raise TypeError(
                "PRA-HF supports Qwen2/Qwen2.5/Qwen3, Llama, and Gemma 3 "
                "global attention modules; "
                f"received {original.__class__.__qualname__}."
            )
        adapter = adapter_class(
            original,
            cache,
            pra_config,
            model.model.rotary_emb,
            routing_representation=config.routing_representation,
            query_strategy=config.query_strategy,
            query_window=config.query_window,
            query_half_life=config.query_half_life,
            routing_projection=routing_projection,
            memory_gate=memory_gate,
            residual_adapter=residual_adapter,
            late_band_lora=late_band_lora,
        )
        layers[layer_id].self_attn = adapter
        adapters[layer_id] = adapter
    return PRAHFModel(
        model,
        adapters,
        cache,
        config,
        pra_config,
        routing_projection,
        memory_gate,
        residual_adapter,
        late_band_lora,
    )
