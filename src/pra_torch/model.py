"""Tiny standalone transformer used to study Progressive Retrieval Attention."""

from contextlib import nullcontext
from dataclasses import replace
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PRAConfig
from .attention import PRAttention
from .chunking import partition_reference_tokens, partition_source
from .gists import GRUGistPooler, GistContext, compute_gists, projected_tokens
from .memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRAMemoryCache,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
    ReferenceRoutingGists,
)
from .positions import build_position_encoding


def causal_attention_mask(seq_len: int, device) -> torch.Tensor:
    """Return ``[T,T]`` mask where true entries hide future key positions."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class VanillaTransformerBlock(nn.Module):
    """Causally masked PyTorch block with no reference-memory branch."""

    def __init__(self, cfg: PRAConfig, layer_id: int):
        """Create one baseline block for ``td_sa`` or a model's vanilla prefix."""
        super().__init__()
        self.layer_id = layer_id  # Stable depth used by traces and block ordering.
        self.layer = nn.TransformerEncoderLayer(  # Complete baseline attention/MLP block.
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(
        self,
        x,
        use_pra_memory: bool = True,
        attention_mask=None,
        position_ids=None,
    ):
        """Transform ``[B,T,D]`` states; ``use_pra_memory`` has no effect here."""
        _ = use_pra_memory, position_ids
        mask = causal_attention_mask(x.shape[1], x.device)
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        return self.layer(x, src_mask=mask, src_key_padding_mask=padding_mask)


class PositionAwareSelfAttention(nn.Module):
    """Explicit causal self-attention used when Q/K need positional transforms."""

    def __init__(self, cfg: PRAConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attention_dropout = nn.Dropout(cfg.dropout)
        self.position_encoding = build_position_encoding(
            cfg.position_encoding,
            head_dim=self.head_dim,
            rope_theta=cfg.rope_theta,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, width = x.shape
        return x.transpose(1, 2).contiguous().view(batch, tokens, heads * width)

    def forward(self, x, *, position_ids, attention_mask=None):
        """Attend causally over ``[B,T,D]`` after positioning ``[B,H,T,Dh]`` Q/K."""
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        query, key = self.position_encoding.transform_qk(query, key, position_ids)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        tokens = x.shape[1]
        future = torch.triu(
            torch.ones(tokens, tokens, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future[None, None], float("-inf"))
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :],
                float("-inf"),
            )
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        return self.o_proj(self._merge_heads(weights @ value))


class PositionAwareTransformerBlock(nn.Module):
    """Pre-norm decoder block for RoPE and other projection-level position modes."""

    def __init__(self, cfg: PRAConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = PositionAwareSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x,
        use_pra_memory: bool = True,
        attention_mask=None,
        position_ids=None,
    ):
        _ = use_pra_memory
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)
        x = x + self.residual_dropout(
            self.attn(self.ln1(x), position_ids=position_ids, attention_mask=attention_mask)
        )
        return x + self.residual_dropout(self.ff(self.ln2(x)))


class PRATransformerBlock(nn.Module):
    """Decoder block whose attention branch combines local and PRA memory output."""

    def __init__(self, cfg: PRAConfig, layer_id: int, pra_cache: PRAMemoryCache):
        """Create a pre-norm PRA attention branch and feed-forward branch."""
        super().__init__()
        self.layer_id = layer_id  # Selects this depth's independently encoded cache K/V.
        self.ln1 = nn.LayerNorm(cfg.d_model)  # Pre-normalizes local and routing queries.
        self.attn = PRAttention(  # Combined causal self-attention and memory branch.
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            max_seq_len=cfg.max_seq_len,
            layer_id=layer_id,
            pra_cache=pra_cache,
            config=cfg,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)  # Pre-normalizes the feed-forward branch.
        self.ff = nn.Sequential(  # Position-wise [D] -> [d_ff] -> [D] transform.
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(
        self,
        x,
        use_pra_memory: bool = True,
        attention_mask=None,
        position_ids=None,
    ):
        """Run pre-norm attention/MLP residuals on ``[B,T,D]`` hidden states."""
        x = x + self.attn(
            self.ln1(x),
            use_pra_memory=use_pra_memory,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = x + self.ff(self.ln2(x))
        return x

    def set_pra_cache(self, pra_cache: PRAMemoryCache) -> None:
        """Attach a cache to this block's PRA attention."""
        self.attn.set_pra_cache(pra_cache)

    def project_reference_kv(self, x, *, detach: bool = True, position_ids=None):
        """Map ``[1,M,D]`` reference states to this layer's ``[1,H,M,Dh]`` K/V."""
        return self.attn.project_kv(
            self.ln1(x),
            detach=detach,
            position_ids=position_ids,
        )


class PRASATransformerBlock(nn.Module):
    """Mixed block with an extra vanilla self-attention before the PRA branch.

    ``PRAttention`` itself contains local causal attention, so this experimental
    mode intentionally has a vanilla residual followed by a second local-plus-
    memory residual. It tests adding PRA to a conventional decoder sublayer.
    """

    def __init__(self, cfg: PRAConfig, layer_id: int, pra_cache: PRAMemoryCache):
        """Create vanilla attention, PRA attention, and MLP pre-norm sublayers."""
        super().__init__()
        self.layer_id = layer_id  # Selects matching layer cache K/V and trace identity.
        self.position_mode = cfg.position_encoding
        self.ln1 = nn.LayerNorm(cfg.d_model)  # Pre-normalizes the vanilla branch.
        self.self_attn = (
            nn.MultiheadAttention(
                embed_dim=cfg.d_model,
                num_heads=cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            if cfg.position_encoding == "absolute"
            else PositionAwareSelfAttention(cfg)
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)  # Pre-normalizes the PRA branch.
        self.pra_attn = PRAttention(  # Second local attention plus reference memory.
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            max_seq_len=cfg.max_seq_len,
            layer_id=layer_id,
            pra_cache=pra_cache,
            config=cfg,
        )
        self.ln3 = nn.LayerNorm(cfg.d_model)  # Pre-normalizes the MLP branch.
        self.ff = nn.Sequential(  # Position-wise mixed-block feed-forward network.
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def _apply_self_attention(self, x, attention_mask=None, position_ids=None):
        """Apply the leading vanilla causal residual to ``[B,T,D]`` states."""
        norm_x = self.ln1(x)
        mask = causal_attention_mask(x.shape[1], x.device)
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        if self.position_mode == "absolute":
            attn_out, _ = self.self_attn(
                norm_x,
                norm_x,
                norm_x,
                attn_mask=mask,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
        else:
            if position_ids is None:
                position_ids = torch.arange(x.shape[1], device=x.device)
            attn_out = self.self_attn(
                norm_x,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        return x + attn_out

    def forward(
        self,
        x,
        use_pra_memory: bool = True,
        attention_mask=None,
        position_ids=None,
    ):
        """Run vanilla attention, local-plus-memory PRA attention, then the MLP."""
        x = self._apply_self_attention(x, attention_mask, position_ids)
        x = x + self.pra_attn(
            self.ln2(x),
            use_pra_memory=use_pra_memory,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = x + self.ff(self.ln3(x))
        return x

    def set_pra_cache(self, pra_cache: PRAMemoryCache) -> None:
        """Attach a cache to this block's PRA attention."""
        self.pra_attn.set_pra_cache(pra_cache)

    def project_reference_kv(self, x, *, detach: bool = True, position_ids=None):
        """Project reference K/V from the same post-vanilla state queried by PRA."""
        x = self._apply_self_attention(x, position_ids=position_ids)
        return self.pra_attn.project_kv(
            self.ln2(x),
            detach=detach,
            position_ids=position_ids,
        )


class TinyPRAModel(nn.Module):
    """Compact decoder-only language model and reference-cache encoder.

    The token/position embeddings and block stack are reused in two contexts:
    normal prompt inference returns logits, while independent reference encoding
    captures K/V before each PRA sublayer. The cache object is shared across PRA
    layers, but every entry owns different K/V and gists for each layer ID.
    """

    def __init__(self, cfg: PRAConfig, pra_cache: PRAMemoryCache | None = None):
        """Construct the configured vanilla, mixed, and PRA block sequence."""
        super().__init__()
        self.cfg = cfg  # Architecture and all routing/cache operating modes.
        self.pra_cache = (  # Shared service storing all URI entries and layer payloads.
            pra_cache if pra_cache is not None else PRASimpleMemoryCache()
        )
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # IDs -> [B,T,D].
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)  # Absolute local positions.
        self.position_encoding = build_position_encoding(
            cfg.position_encoding,
            head_dim=cfg.d_model // cfg.n_heads,
            rope_theta=cfg.rope_theta,
        )
        needs_gru_pooler = (
            "gru" in {cfg.gist_mode, cfg.reference_level_gist_mode}
            or (
                "hybrid" in {cfg.gist_mode, cfg.reference_level_gist_mode}
                and cfg.gist_hybrid_global_mode == "gru"
            )
        )
        self.gist_pooler = (
            GRUGistPooler(
                cfg.d_model,
                hidden_size=cfg.gist_gru_hidden_size,
                num_layers=cfg.gist_gru_num_layers,
                bidirectional=cfg.gist_gru_bidirectional,
                dropout=cfg.dropout,
            )
            if needs_gru_pooler
            else None
        )
        self.blocks = nn.ModuleList(self._build_blocks(cfg))  # Ordered decoder depth.
        self.ln = nn.LayerNorm(cfg.d_model)  # Final decoder normalization.
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # [B,T,D] -> logits.

    def _build_blocks(self, cfg: PRAConfig) -> list[nn.Module]:
        """Lay out vanilla prefix, mixed middle, then PRA-only remainder."""
        blocks: list[nn.Module] = []
        for layer_id in range(cfg.n_layers):
            if layer_id < cfg.n_vanilla_layers:
                block_type = (
                    VanillaTransformerBlock
                    if cfg.position_encoding == "absolute"
                    else PositionAwareTransformerBlock
                )
                blocks.append(block_type(cfg, layer_id))
            elif layer_id < cfg.n_vanilla_layers + cfg.n_mixed_layers:
                blocks.append(PRASATransformerBlock(cfg, layer_id, self.pra_cache))
            else:
                blocks.append(PRATransformerBlock(cfg, layer_id, self.pra_cache))
        return blocks

    def set_pra_cache(self, pra_cache: PRAMemoryCache) -> None:
        """Attach a cache to the model and every PRA attention block."""
        self.pra_cache = pra_cache
        for block in self.blocks:
            if hasattr(block, "set_pra_cache"):
                block.set_pra_cache(pra_cache)

    def clear_pra_cache(self) -> None:
        """Remove all cached reference entries from the current cache."""
        self.pra_cache.clear()

    def selected_chunks_by_layer(self) -> dict[int, list[list]]:
        """Return row-local selections as ``layer -> batch row -> ranked chunks``."""
        selections = {}
        for block in self.blocks:
            attention = getattr(block, "attn", None) or getattr(block, "pra_attn", None)
            if attention is not None:
                selections[block.layer_id] = [list(items) for items in attention.last_selected_chunks]
        return selections

    def routing_rankings_by_layer(self) -> dict[int, list[list[dict]]]:
        """Return complete pre-top-k candidate rankings from the latest forward."""
        rankings = {}
        for block in self.blocks:
            attention = getattr(block, "attn", None) or getattr(block, "pra_attn", None)
            if attention is not None:
                rankings[block.layer_id] = [
                    list(items) for items in attention.last_routing_rankings
                ]
        return rankings

    def rebuild_cache_routing_gists(
        self,
        cache: PRAMemoryCache | None = None,
        *,
        tokenizer=None,
    ) -> None:
        """Re-index stored native K/V under the current gist configuration.

        Token K/V depends on model weights and encoding context, not on the
        selector's pooling strategy. Sensitivity sweeps can therefore reuse the
        expensive encoded payload while rebuilding only its cheap routing index.
        """
        cache = cache or self.pra_cache
        detach = self.cfg.cache_build_mode == "detached"
        with torch.no_grad() if detach else nullcontext():
            for entry in cache.all_entries():
                for layer_id, memory in entry.layer_memory.items():
                    for chunk in memory.chunks:
                        token_ids = tuple(chunk.metadata.get("source_token_ids") or ())
                        computed = compute_gists(
                            keys=projected_tokens(chunk.token_kv.k),
                            values=projected_tokens(chunk.token_kv.v),
                            mode=self.cfg.gist_mode,
                            num_gists=self.cfg.gists_per_chunk,
                            config=self.cfg,
                            context=GistContext(
                                level="chunk",
                                token_ids=token_ids,
                                tokenizer=tokenizer,
                                ref_end_token=self.cfg.ref_end_token,
                                gru_pooler=self.gist_pooler,
                            ),
                        )
                        chunk.routing_gist = ChunkRoutingGist(
                            k=computed.k.detach() if detach else computed.k,
                            v=(
                                computed.v.detach()
                                if detach and computed.v is not None
                                else computed.v
                            ),
                            method=self.cfg.gist_mode,
                            summary_k=chunk.routing_gist.summary_k,
                            metadata={
                                **computed.metadata,
                                "summary_available": chunk.routing_gist.summary_k is not None,
                            },
                        )
                    if self.cfg.reference_level_gist_mode is None or not memory.chunks:
                        entry.reference_gists_by_layer.pop(layer_id, None)
                        continue
                    keys = torch.cat(
                        [chunk.routing_gist.k for chunk in memory.chunks], dim=0
                    )
                    values = (
                        torch.cat(
                            [chunk.routing_gist.v for chunk in memory.chunks], dim=0
                        )
                        if all(chunk.routing_gist.v is not None for chunk in memory.chunks)
                        else None
                    )
                    computed = compute_gists(
                        keys=keys,
                        values=values,
                        mode=self.cfg.reference_level_gist_mode,
                        num_gists=self.cfg.reference_gists_per_reference,
                        config=self.cfg,
                        context=GistContext(level="reference", gru_pooler=self.gist_pooler),
                    )
                    entry.reference_gists_by_layer[layer_id] = ReferenceRoutingGists(
                        k=computed.k.detach() if detach else computed.k,
                        v=(
                            computed.v.detach()
                            if detach and computed.v is not None
                            else computed.v
                        ),
                        mode=self.cfg.reference_level_gist_mode,
                        metadata=computed.metadata,
                    )
                entry.metadata.update(
                    {
                        "gist_mode": self.cfg.gist_mode,
                        "gists_per_chunk": self.cfg.gists_per_chunk,
                        "reference_level_gist_mode": self.cfg.reference_level_gist_mode,
                        "reference_gists_per_reference": (
                            self.cfg.reference_gists_per_reference
                        ),
                    }
                )
        invalidate = getattr(cache, "invalidate_routing_indexes", None)
        if invalidate is not None:
            invalidate()

    def selected_references_by_layer(self) -> dict[int, list[list[tuple[str, float]]]]:
        """Deprecated compatibility view derived from chunk-aware selections."""
        warnings.warn(
            "selected_references_by_layer() is deprecated; use selected_chunks_by_layer().",
            DeprecationWarning,
            stacklevel=2,
        )
        result = {}
        for layer_id, batches in self.selected_chunks_by_layer().items():
            layer_batches = []
            for selected in batches:
                by_uri = {}
                for hit in selected:
                    by_uri.setdefault(hit.reference_uri, hit.reference_score)
                layer_batches.append(list(by_uri.items()))
            result[layer_id] = layer_batches
        return result

    def pra_diagnostics_by_layer(self) -> dict[int, dict]:
        """Collect latest batch-level routing/materialization metrics by PRA layer.

        Unlike ``selected_chunks_by_layer``, output norms, padding totals, and
        timing summarize the layer's complete logical batch rather than one row.
        """
        diagnostics = {}
        for block in self.blocks:
            attention = getattr(block, "attn", None) or getattr(block, "pra_attn", None)
            if attention is not None:
                diagnostics[block.layer_id] = {
                    **attention.last_diagnostics,
                    "batching": attention.last_memory_batching_stats,
                }
        return diagnostics

    def forward(
        self,
        input_ids,
        use_pra_memory: bool = True,
        attention_mask=None,
        position_offset: int | torch.Tensor = 0,
    ):
        """Map ``[B,T]`` IDs to logits, optionally continuing historical positions."""
        b, t = input_ids.shape
        native_limit = self.cfg.effective_model_max_context_tokens
        position_limit = self.cfg.position_capacity
        if t > native_limit:
            raise ValueError(
                f"Native forward has {t} tokens, exceeding "
                f"model_max_context_tokens={native_limit}."
            )
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same [batch,tokens] shape as input_ids.")
        if torch.is_tensor(position_offset):
            offsets = position_offset.to(device=input_ids.device, dtype=torch.long).reshape(-1)
            if offsets.numel() != b:
                raise ValueError("Per-row position offsets must have shape [batch].")
            pos = offsets[:, None] + torch.arange(t, device=input_ids.device)[None, :]
            valid = (
                attention_mask.bool()
                if attention_mask is not None
                else torch.ones_like(pos, dtype=torch.bool)
            )
            valid_positions = pos[valid]
            if position_limit is not None and (
                valid_positions.numel()
                and (
                    int(valid_positions.min()) < 0
                    or int(valid_positions.max()) >= position_limit
                )
            ):
                raise ValueError("Prompt positions exceed the model positional table.")
            pos = pos.masked_fill(~valid, 0)
        else:
            scalar_offset = int(position_offset)
            if scalar_offset < 0 or (
                position_limit is not None
                and scalar_offset + t > position_limit
            ):
                raise ValueError(
                    "Prompt position range exceeds the model positional table: "
                    f"[{scalar_offset}, {scalar_offset + t}) vs "
                    f"position_capacity={position_limit}."
                )
            pos = torch.arange(scalar_offset, scalar_offset + t, device=input_ids.device)
        x = self.position_encoding.apply_embeddings(
            self.token_emb(input_ids),
            pos,
            self.pos_emb,
        )
        for block in self.blocks:
            x = block(
                x,
                use_pra_memory=use_pra_memory,
                attention_mask=attention_mask,
                position_ids=pos,
            )
        x = self.ln(x)
        return self.head(x)

    def _encode_reference_tokens(
        self,
        token_ids,
        device,
        *,
        detach: bool,
        use_pra_memory: bool,
        position_offset: int = 0,
    ):
        """Run one chunk independently and capture K/V before each PRA sublayer.

        ``token_ids`` has length ``M``. Each captured value is ``LayerKV`` with
        shape ``[1,H,M,Dh]``. Normal cache builds disable PRA memory to avoid
        self-dependence; recursive parent builds may read already-ready children.
        """
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        position_offset = int(position_offset)
        native_limit = self.cfg.effective_model_max_context_tokens
        position_limit = self.cfg.position_capacity
        if ids.shape[1] > native_limit:
            raise ValueError(
                f"Reference encoding call has {ids.shape[1]} tokens, exceeding "
                f"model_max_context_tokens={native_limit}."
            )
        if position_offset < 0 or (
            position_limit is not None
            and position_offset + ids.shape[1] > position_limit
        ):
            raise ValueError(
                "Reference position range exceeds the model positional table: "
                f"[{position_offset}, {position_offset + ids.shape[1]}) vs "
                f"position_capacity={position_limit}."
            )
        pos = torch.arange(
            position_offset,
            position_offset + ids.shape[1],
            device=device,
        )
        x = self.position_encoding.apply_embeddings(
            self.token_emb(ids),
            pos,
            self.pos_emb,
        )
        layer_kv = {}
        for block in self.blocks:
            # Capture the exact normalized state consumed by this depth's memory branch.
            if hasattr(block, "project_reference_kv"):
                layer_kv[block.layer_id] = block.project_reference_kv(
                    x,
                    detach=detach,
                    position_ids=pos,
                )
            x = block(x, use_pra_memory=use_pra_memory, position_ids=pos)
        return layer_kv

    def _cache_resident_kv(self, kv: LayerKV) -> LayerKV:
        """Place full token K/V in the configured backing tier after gist creation."""
        if self.cfg.kv_cache_residency == "gpu":
            return kv
        key = kv.k.detach().to("cpu")
        value = kv.v.detach().to("cpu")
        if self.cfg.kv_cache_pin_memory and torch.cuda.is_available():
            key = key.pin_memory()
            value = value.pin_memory()
        positions = (
            kv.position_ids.detach().to("cpu")
            if kv.position_ids is not None
            else None
        )
        return LayerKV(
            k=key,
            v=value,
            position_ids=positions,
            position_state=kv.position_state,
        )

    def _bounded_reference_payloads(
        self,
        uri,
        token_ids,
        tokenizer,
        device,
        metadata,
        *,
        detach,
        use_pra_memory,
        historical_encoding,
        max_chunks,
        use_configured_max_chunks,
    ):
        """Encode model-safe source blocks and slice their K/V into routing chunks."""
        native_limit = self.cfg.effective_model_max_context_tokens
        encoding = self.cfg.encoding_chunking_config
        # Encoding overlap is context-only. Core spans remain non-overlapping in storage.
        core_policy = replace(encoding, overlap_fraction=0.0, overlap_tokens=0)
        cores = partition_source(
            uri,
            token_ids,
            tokenizer,
            core_policy,
            {**metadata, "partition_level": "encoding"},
            max_chunks=None,
            overflow_policy="error",
            max_chunk_tokens=native_limit,
        )
        payloads = []
        max_encoding_tokens = 0
        encoded_tokens = 0
        context_mode = (
            "historical_window"
            if historical_encoding and self.cfg.encoding_context_mode == "independent"
            else self.cfg.encoding_context_mode
        )
        for encoding_index, core in enumerate(cores):
            core_ids = list(core.token_ids)
            requested_overlap = (
                encoding.resolved_overlap_tokens(len(core_ids))
                if context_mode in {"overlap", "historical_window"}
                else 0
            )
            left_context = min(
                core.token_start,
                requested_overlap,
                max(native_limit - len(core_ids), 0),
            )
            encode_start = core.token_start - left_context
            encode_ids = list(token_ids[encode_start : core.token_end])
            max_encoding_tokens = max(max_encoding_tokens, len(encode_ids))
            encoded_tokens += len(encode_ids)
            position_offset = (
                encode_start
                if self.cfg.reference_position_mode == "global"
                and (
                    not self.position_encoding.has_bounded_positions
                    or encode_start + len(encode_ids) <= self.cfg.max_seq_len
                )
                else 0
            )
            encoded = self._encode_reference_tokens(
                encode_ids,
                device,
                detach=detach,
                use_pra_memory=use_pra_memory and not historical_encoding,
                position_offset=position_offset,
            )
            core_kv = {
                layer_id: LayerKV(
                    k=kv.k[:, :, left_context : left_context + len(core_ids), :],
                    v=kv.v[:, :, left_context : left_context + len(core_ids), :],
                    position_ids=(
                        kv.position_ids[..., left_context : left_context + len(core_ids)]
                        if kv.position_ids is not None
                        else None
                    ),
                    position_state=kv.position_state,
                )
                for layer_id, kv in encoded.items()
            }
            local_chunks = partition_source(
                uri,
                core_ids,
                tokenizer,
                self.cfg.routing_chunking_config,
                {**metadata, "partition_level": "routing"},
                max_chunks=None,
                overflow_policy="error",
                max_chunk_tokens=native_limit,
            )
            for local in local_chunks:
                global_start = core.token_start + local.token_start
                global_end = core.token_start + local.token_end
                routed = replace(
                    local,
                    chunk_id=f"{uri}#chunk={len(payloads)}",
                    token_start=global_start,
                    token_end=global_end,
                    logical_start=global_start,
                    logical_end=global_end,
                    metadata={
                        **local.metadata,
                        "encoding_block_id": encoding_index,
                        "encoding_input_start": encode_start,
                        "encoding_input_end": core.token_end,
                        "encoding_input_tokens": len(encode_ids),
                        "encoding_context_tokens": left_context,
                        "logical_start": global_start,
                        "logical_end": global_end,
                        "base_model_position_offset": position_offset,
                    },
                )
                payloads.append(
                    (
                        routed,
                        {
                            layer_id: LayerKV(
                                k=kv.k[:, :, local.token_start : local.token_end, :],
                                v=kv.v[:, :, local.token_start : local.token_end, :],
                                position_ids=(
                                    kv.position_ids[..., local.token_start : local.token_end]
                                    if kv.position_ids is not None
                                    else None
                                ),
                                position_state=kv.position_state,
                            )
                            for layer_id, kv in core_kv.items()
                        },
                    )
                )
        chunk_limit = (
            self.cfg.max_gists_per_reference
            if use_configured_max_chunks
            else max_chunks
        )
        discarded = max(len(payloads) - chunk_limit, 0) if chunk_limit is not None else 0
        if discarded:
            if self.cfg.gist_overflow_policy == "error":
                raise ValueError(
                    f"Reference produced {len(payloads)} routing chunks, exceeding {chunk_limit}."
                )
            payloads = payloads[:chunk_limit]
        return payloads, {
            "encoding_call_count": len(cores),
            "max_encoding_input_tokens": max_encoding_tokens,
            "encoding_input_tokens_total": encoded_tokens,
            "encoding_context_mode": context_mode,
            "encoding_chunking": {
                "mode": encoding.mode,
                "chunk_tokens": encoding.chunk_tokens,
                "overlap_tokens": encoding.overlap_tokens,
                "overlap_fraction": encoding.overlap_fraction,
            },
            "routing_chunking": {
                "mode": self.cfg.routing_chunking_config.mode,
                "chunk_tokens": self.cfg.routing_chunking_config.chunk_tokens,
                "overlap_tokens": self.cfg.routing_chunking_config.overlap_tokens,
                "overlap_fraction": self.cfg.routing_chunking_config.overlap_fraction,
            },
            "discarded_chunk_count": discarded,
        }

    def encode_reference_to_cache(
        self,
        uri: str,
        text: str,
        tokenizer,
        device,
        metadata: dict | None = None,
        *,
        use_pra_memory: bool = False,
    ) -> PRACacheEntry:
        """Tokenize a resolved text reference once, then build its PRA cache entry."""
        return self.encode_reference_tokens_to_cache(
            uri,
            tokenizer.encode(text),
            tokenizer,
            device,
            metadata,
            text=text,
            use_pra_memory=use_pra_memory,
        )

    def encode_reference_group_to_cache(
        self,
        references: list[dict],
        tokenizer,
        device,
    ) -> list[PRACacheEntry]:
        """Contextualize ordered URIs jointly, then slice K/V back by URI.

        ``block_slice`` encodes consecutive URI groups with optional left
        overlap; ``native_slice`` encodes the complete historical source once.
        Stored K/V remains non-overlapping and independently addressable even
        when its hidden states were produced with a larger causal context.
        """
        if self.cfg.reference_encoding_strategy == "independent":
            raise ValueError("Grouped encoding requires block_slice or native_slice.")
        if self.cfg.use_summary:
            raise NotImplementedError("Grouped reference encoding does not yet support summaries.")
        if not references:
            return []
        rows = []
        flat_ids = []
        starts = []
        for reference in references:
            token_ids = tuple(int(value) for value in tokenizer.encode(str(reference["text"])))
            if not token_ids:
                continue
            starts.append(len(flat_ids))
            flat_ids.extend(token_ids)
            rows.append({**reference, "token_ids": token_ids})
        if not rows:
            return []
        starts.append(len(flat_ids))
        block_size = (
            len(rows)
            if self.cfg.reference_encoding_strategy == "native_slice"
            else self.cfg.encoding_block_references
        )
        block_ranges = [
            (start, min(start + block_size, len(rows)))
            for start in range(0, len(rows), block_size)
        ]
        block_specs = []
        encoded_token_total = 0
        for block_id, (row_start, row_end) in enumerate(block_ranges):
            core_start = starts[row_start]
            core_end = starts[row_end]
            core_tokens = core_end - core_start
            requested_overlap = (
                max(1, int(core_tokens * self.cfg.encoding_overlap_fraction))
                if self.cfg.encoding_overlap_fraction > 0.0
                else 0
            )
            overlap = min(core_start, requested_overlap)
            encode_start = core_start - overlap
            encode_ids = flat_ids[encode_start:core_end]
            if len(encode_ids) > self.cfg.effective_model_max_context_tokens:
                raise ValueError(
                    f"Encoding block {block_id} has {len(encode_ids)} tokens, exceeding "
                    "model_max_context_tokens="
                    f"{self.cfg.effective_model_max_context_tokens}."
                )
            encoded_token_total += len(encode_ids)
            block_specs.append(
                {
                    "block_id": block_id,
                    "row_start": row_start,
                    "row_end": row_end,
                    "encode_start": encode_start,
                    "core_start": core_start,
                    "core_end": core_end,
                    "overlap_tokens": overlap,
                    "encode_ids": encode_ids,
                }
            )
        run_id = (
            f"{rows[0]['uri']}|{self.cfg.reference_encoding_strategy}|"
            f"{self.cfg.encoding_block_references}|{self.cfg.encoding_overlap_fraction}|"
            f"{self.cfg.reference_position_mode}"
        )
        common_metadata = {
            "encoding_run_id": run_id,
            "reference_encoding_strategy": self.cfg.reference_encoding_strategy,
            "encoding_block_references": self.cfg.encoding_block_references,
            "encoding_overlap_fraction": self.cfg.encoding_overlap_fraction,
            "reference_position_mode": self.cfg.reference_position_mode,
            "encoding_run_unique_source_tokens": len(flat_ids),
            "encoding_run_encoded_tokens_including_overlap": encoded_token_total,
            "encoding_run_stored_kv_tokens": len(flat_ids),
            "encoding_run_duplication_factor": encoded_token_total / max(len(flat_ids), 1),
            "max_encoding_input_tokens": max(
                (len(block["encode_ids"]) for block in block_specs),
                default=0,
            ),
        }
        entries = {
            str(row["uri"]): PRACacheEntry(
                uri=str(row["uri"]),
                text=str(row["text"]),
                metadata={
                    **dict(row.get("metadata") or {}),
                    **common_metadata,
                    "chunking_mode": "grouped_uri_slice",
                    "gist_mode": self.cfg.gist_mode,
                    "gists_per_chunk": self.cfg.gists_per_chunk,
                    "unique_source_tokens": len(row["token_ids"]),
                    "encoded_tokens_including_overlap": len(row["token_ids"]),
                    "stored_kv_tokens_including_overlap": len(row["token_ids"]),
                },
            )
            for row in rows
        }
        detach = self.cfg.cache_build_mode == "detached"
        context = torch.no_grad() if detach else nullcontext()
        with context:
            for block in block_specs:
                position_offset = (
                    block["encode_start"]
                    if self.cfg.reference_position_mode == "global"
                    and (
                        not self.position_encoding.has_bounded_positions
                        or block["encode_start"] + len(block["encode_ids"])
                        <= self.cfg.max_seq_len
                    )
                    else 0
                )
                layer_kv = self._encode_reference_tokens(
                    block["encode_ids"],
                    device,
                    detach=detach,
                    use_pra_memory=False,
                    position_offset=position_offset,
                )
                for row_index in range(block["row_start"], block["row_end"]):
                    row = rows[row_index]
                    local_start = starts[row_index] - block["encode_start"]
                    local_end = starts[row_index + 1] - block["encode_start"]
                    entry = entries[str(row["uri"])]
                    for layer_id, kv in layer_kv.items():
                        sliced = LayerKV(
                            k=kv.k[:, :, local_start:local_end, :],
                            v=kv.v[:, :, local_start:local_end, :],
                            position_ids=(
                                kv.position_ids[..., local_start:local_end]
                                if kv.position_ids is not None
                                else None
                            ),
                            position_state=kv.position_state,
                        )
                        computed = compute_gists(
                            keys=projected_tokens(sliced.k),
                            values=projected_tokens(sliced.v),
                            mode=self.cfg.gist_mode,
                            num_gists=self.cfg.gists_per_chunk,
                            config=self.cfg,
                            context=GistContext(
                                level="chunk",
                                token_ids=row["token_ids"],
                                tokenizer=tokenizer,
                                ref_end_token=self.cfg.ref_end_token,
                                gru_pooler=self.gist_pooler,
                            ),
                        )
                        chunk = ReferenceChunkMemory(
                            chunk_id=f"{row['uri']}#chunk=0",
                            source_uri=str(row["uri"]),
                            token_start=0,
                            token_end=len(row["token_ids"]),
                            logical_start=starts[row_index],
                            logical_end=starts[row_index + 1],
                            token_kv=self._cache_resident_kv(sliced),
                            routing_gist=ChunkRoutingGist(
                                k=computed.k.detach() if detach else computed.k,
                                v=(
                                    computed.v.detach()
                                    if detach and computed.v is not None
                                    else computed.v
                                ),
                                method=self.cfg.gist_mode,
                                metadata=computed.metadata,
                            ),
                            metadata={
                                "source_token_ids": row["token_ids"],
                                "encoding_block_id": block["block_id"],
                                "encoding_block_core_tokens": (
                                    block["core_end"] - block["core_start"]
                                ),
                                "encoding_overlap_tokens": block["overlap_tokens"],
                                "global_token_start": starts[row_index],
                                "global_token_end": starts[row_index + 1],
                                "logical_start": starts[row_index],
                                "logical_end": starts[row_index + 1],
                                "base_model_position_offset": position_offset,
                            },
                        )
                        entry.layer_memory.setdefault(
                            layer_id, LayerReferenceMemory()
                        ).chunks.append(chunk)
        if self.cfg.reference_level_gist_mode is not None:
            temporary = PRASimpleMemoryCache()
            for entry in entries.values():
                temporary.put(entry)
            self.rebuild_cache_routing_gists(temporary, tokenizer=tokenizer)
        return [entries[str(row["uri"])] for row in rows]

    def encode_reference_tokens_to_cache(
        self,
        uri: str,
        token_ids,
        tokenizer,
        device,
        metadata: dict | None = None,
        *,
        text: str | None = None,
        use_pra_memory: bool = False,
        max_chunks: int | None = None,
        use_configured_max_chunks: bool = True,
        max_chunk_tokens: int | None = None,
        historical_encoding: bool = False,
    ) -> PRACacheEntry:
        """Encode exact reference token IDs into layer-specific routing and K/V.

        The method first partitions the supplied IDs. It either runs each retained
        chunk independently or encodes the complete source once and slices historical
        K/V by chunk boundary. For every PRA layer it stores full token K/V
        ``[1,H,M,Dh]`` plus content/value gist sets ``[G_chunk,D]``. Optional
        per-layer URI gists are cached as ``[G_ref,D]`` for reference-first routing.
        An optional summary is
        encoded separately and contributes only a routing key. The returned entry
        is not visible to attention until a cache backend publishes it with ``put``.
        """
        metadata = dict(metadata or {})
        token_ids = [int(token_id) for token_id in token_ids]
        text = tokenizer.decode(token_ids) if text is None else text
        detach = self.cfg.cache_build_mode == "detached"
        context = torch.no_grad() if detach else nullcontext()
        bounded_encoding = (
            historical_encoding
            or self.cfg.encoding_chunking is not None
            or len(token_ids) > self.cfg.effective_model_max_context_tokens
        )
        preencoded_by_chunk = {}
        encoding_stats = {}
        if bounded_encoding:
            if use_pra_memory and historical_encoding:
                raise ValueError("Historical source encoding cannot consume its own PRA memory.")
            with context:
                payloads, encoding_stats = self._bounded_reference_payloads(
                    uri,
                    token_ids,
                    tokenizer,
                    device,
                    metadata,
                    detach=detach,
                    use_pra_memory=use_pra_memory,
                    historical_encoding=historical_encoding,
                    max_chunks=max_chunks,
                    use_configured_max_chunks=use_configured_max_chunks,
                )
            chunks = [chunk for chunk, _layer_kv in payloads]
            preencoded_by_chunk = {
                chunk.chunk_id: layer_kv for chunk, layer_kv in payloads
            }
        else:
            chunks = partition_reference_tokens(
                uri,
                token_ids,
                tokenizer,
                self.cfg,
                metadata,
                text=text,
                max_chunks=max_chunks,
                use_configured_max_chunks=use_configured_max_chunks,
                max_chunk_tokens=(
                    max_chunk_tokens or self.cfg.effective_model_max_context_tokens
                ),
            )
            encoding_stats = {
                "encoding_call_count": len(chunks),
                "max_encoding_input_tokens": max(
                    (len(chunk.token_ids) for chunk in chunks), default=0
                ),
                "encoding_input_tokens_total": sum(
                    len(chunk.token_ids) for chunk in chunks
                ),
                "encoding_context_mode": "independent",
            }
        entry = PRACacheEntry(
            uri=uri,
            text=text,
            child_uris=list(metadata.get("child_uris") or []),
            metadata={
                **metadata,
                "chunking_mode": self.cfg.routing_chunking_config.mode,
                "gist_mode": self.cfg.gist_mode,
                "gists_per_chunk": self.cfg.gists_per_chunk,
                "reference_level_gist_mode": self.cfg.reference_level_gist_mode,
                "reference_gists_per_reference": self.cfg.reference_gists_per_reference,
                "cache_build_mode": self.cfg.cache_build_mode,
                "use_summary": self.cfg.use_summary,
                "summary_mode": self.cfg.summary_mode,
                "chunk_count": len(chunks),
                "chunk_overlap_fraction": self.cfg.chunk_overlap_fraction,
                "chunk_overlap_tokens": self.cfg.resolved_chunk_overlap_tokens,
                "overlap_materialization": self.cfg.overlap_materialization,
                "historical_encoding": bool(historical_encoding),
                "reference_encoding_strategy": (
                    "bounded_native_slice" if bounded_encoding else "independent"
                ),
                "model_max_context_tokens": self.cfg.effective_model_max_context_tokens,
                "logical_context_tokens": len(token_ids),
                "logical_to_native_context_ratio": len(token_ids)
                / max(self.cfg.effective_model_max_context_tokens, 1),
                **encoding_stats,
                "unique_source_tokens": len(token_ids),
                "encoded_tokens_including_overlap": sum(
                    len(chunk.token_ids) for chunk in chunks
                ),
                "duplication_factor": sum(len(chunk.token_ids) for chunk in chunks)
                / max(len(token_ids), 1),
            },
        )
        # Summary routing is computed once per layer and shared by all URI chunks.
        summary_by_layer = {}
        summary = metadata.get("summary")
        with context:
            if self.cfg.use_summary and summary:
                summary_ids = list(tokenizer.encode(str(summary)))
                if len(summary_ids) > self.cfg.effective_model_max_context_tokens:
                    summary_ids = summary_ids[: self.cfg.effective_model_max_context_tokens]
                if summary_ids:
                    summary_kv = self._encode_reference_tokens(
                        summary_ids, device, detach=detach, use_pra_memory=False
                    )
                    for layer_id, kv in summary_kv.items():
                        summary_mode = "gru" if self.cfg.gist_mode == "gru" else "mean"
                        summary_by_layer[layer_id] = compute_gists(
                            keys=projected_tokens(kv.k),
                            values=None,
                            num_gists=1,
                            config=self.cfg,
                            context=GistContext(
                                level="chunk",
                                token_ids=summary_ids,
                                tokenizer=tokenizer,
                                ref_end_token=self.cfg.ref_end_token,
                                gru_pooler=self.gist_pooler,
                            ),
                            mode=summary_mode,
                        ).k

            # Each chunk receives a fresh positional context starting at zero. Source
            # offsets remain on the chunk for provenance and overlap removal.
            for chunk in chunks:
                token_ids = list(chunk.token_ids)
                original_length = len(token_ids)
                if original_length > self.cfg.effective_model_max_context_tokens:
                    if self.cfg.reference_overflow_policy == "error":
                        raise ValueError(
                            f"Chunk {chunk.chunk_id} has {original_length} tokens, exceeding "
                            "model_max_context_tokens="
                            f"{self.cfg.effective_model_max_context_tokens}."
                        )
                    token_ids = token_ids[: self.cfg.effective_model_max_context_tokens]
                if chunk.chunk_id not in preencoded_by_chunk:
                    layer_kv = self._encode_reference_tokens(
                        token_ids,
                        device,
                        detach=detach,
                        use_pra_memory=use_pra_memory,
                    )
                else:
                    layer_kv = preencoded_by_chunk[chunk.chunk_id]
                # Gists are pooled from projected keys/values in the same layer space
                # later used by routing queries and cross-attention.
                for layer_id, kv in layer_kv.items():
                    computed = compute_gists(
                        keys=projected_tokens(kv.k),
                        values=projected_tokens(kv.v),
                        mode=self.cfg.gist_mode,
                        num_gists=self.cfg.gists_per_chunk,
                        config=self.cfg,
                        context=GistContext(
                            level="chunk",
                            token_ids=token_ids,
                            tokenizer=tokenizer,
                            ref_end_token=self.cfg.ref_end_token,
                            gru_pooler=self.gist_pooler,
                        ),
                    )
                    gist_k = computed.k
                    gist_v = computed.v
                    if detach:
                        gist_k = gist_k.detach()
                        gist_v = gist_v.detach() if gist_v is not None else None
                    chunk_memory = ReferenceChunkMemory(
                        chunk_id=chunk.chunk_id,
                        source_uri=chunk.source_uri,
                        token_start=chunk.token_start,
                        token_end=min(chunk.token_start + len(token_ids), chunk.token_end),
                        logical_start=chunk.logical_start,
                        logical_end=chunk.logical_start + len(token_ids),
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        token_kv=self._cache_resident_kv(kv),
                        routing_gist=ChunkRoutingGist(
                            k=gist_k,
                            v=gist_v,
                            method=self.cfg.gist_mode,
                            summary_k=summary_by_layer.get(layer_id),
                            metadata={
                                **computed.metadata,
                                "summary_available": layer_id in summary_by_layer,
                            },
                        ),
                        metadata={
                            **chunk.metadata,
                            "source_token_ids": tuple(token_ids),
                            "original_token_count": original_length,
                            "retained_token_count": len(token_ids),
                            "truncated": original_length != len(token_ids),
                        },
                    )
                    entry.layer_memory.setdefault(layer_id, LayerReferenceMemory()).chunks.append(
                        chunk_memory
                    )
            # URI gists compress all chunk gists at this layer once during cache build.
            if self.cfg.reference_level_gist_mode is not None:
                for layer_id, memory in entry.layer_memory.items():
                    if not memory.chunks:
                        continue
                    keys = torch.cat([chunk.routing_gist.k for chunk in memory.chunks], dim=0)
                    values = (
                        torch.cat([chunk.routing_gist.v for chunk in memory.chunks], dim=0)
                        if all(chunk.routing_gist.v is not None for chunk in memory.chunks)
                        else None
                    )
                    computed = compute_gists(
                        keys=keys,
                        values=values,
                        mode=self.cfg.reference_level_gist_mode,
                        num_gists=self.cfg.reference_gists_per_reference,
                        config=self.cfg,
                        context=GistContext(
                            level="reference",
                            gru_pooler=self.gist_pooler,
                        ),
                    )
                    reference_k = computed.k.detach() if detach else computed.k
                    reference_v = (
                        computed.v.detach() if detach and computed.v is not None else computed.v
                    )
                    entry.reference_gists_by_layer[layer_id] = ReferenceRoutingGists(
                        k=reference_k,
                        v=reference_v,
                        mode=self.cfg.reference_level_gist_mode,
                        metadata=computed.metadata,
                    )
        return entry

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=64,
        temperature=1.0,
        use_pra_memory: bool = True,
        *,
        tokenizer=None,
        do_sample: bool = True,
    ):
        """Generate indefinitely while rolling expired direct tokens into ``#__head``."""
        self.eval()
        output_ids = input_ids
        direct_ids = input_ids
        from .memory import PRABatchedMemoryCache
        from .prompt import IMPLICIT_PROMPT_HEAD_NAME, IMPLICIT_PROMPT_HEAD_URI

        if (
            self.cfg.prompt_overflow_mode == "implicit_reference"
            and use_pra_memory
            and input_ids.shape[1] + max_new_tokens
            > self.cfg.effective_prompt_direct_tokens
            and tokenizer is None
        ):
            raise ValueError("tokenizer is required for streaming prompt-history rollover.")
        if (
            input_ids.shape[0] > 1
            and not isinstance(self.pra_cache, PRABatchedMemoryCache)
        ):
            if not self.pra_cache.is_empty():
                raise ValueError(
                    "Batched generation with explicit references requires row-local caches."
                )
            self.set_pra_cache(
                PRABatchedMemoryCache(
                    [PRASimpleMemoryCache() for _ in range(input_ids.shape[0])]
                )
            )
        active_caches = (
            self.pra_cache.row_caches
            if isinstance(self.pra_cache, PRABatchedMemoryCache)
            else [self.pra_cache]
        )
        for cache in active_caches:
            if hasattr(cache, "invalidate"):
                cache.invalidate(IMPLICIT_PROMPT_HEAD_URI)
        head_rows = [[] for _ in range(input_ids.shape[0])]
        if input_ids.shape[1] > self.cfg.effective_prompt_direct_tokens:
            if self.cfg.prompt_overflow_mode == "implicit_reference" and use_pra_memory:
                if tokenizer is None:
                    raise ValueError(
                        "tokenizer is required to build implicit memory for a long prompt."
                    )
                from .prompt import prepare_prompt_batch_for_pra

                if isinstance(self.pra_cache, PRABatchedMemoryCache):
                    caches = self.pra_cache.row_caches
                elif input_ids.shape[0] == 1:
                    caches = [self.pra_cache]
                elif self.pra_cache.is_empty():
                    caches = [PRASimpleMemoryCache() for _ in range(input_ids.shape[0])]
                else:
                    raise ValueError(
                        "Batched generation with explicit references requires row-local caches."
                    )
                prepared = prepare_prompt_batch_for_pra(
                    self,
                    tokenizer,
                    input_ids,
                    caches=caches,
                )
                direct_ids = prepared.input_ids
                head_rows = [list(split.implicit_ids) for split in prepared.splits]
                self.set_pra_cache(
                    caches[0] if len(caches) == 1 else PRABatchedMemoryCache(caches)
                )
                active_caches = caches
            else:
                from .prompt import prepare_prompt_for_pra

                splits = [prepare_prompt_for_pra(row, self.cfg) for row in input_ids]
                direct_ids = input_ids.new_tensor([split.direct_ids for split in splits])
        direct_limit = self.cfg.effective_prompt_direct_tokens
        routing_size = self.cfg.routing_chunking_config.chunk_tokens
        rollover_unit = max(
            1,
            min(int(routing_size or max(direct_limit // 2, 1)), direct_limit),
        )
        rollover_events = 0
        tokens_migrated = 0
        max_direct_tokens_observed = int(direct_ids.shape[1])
        max_native_operation_tokens = int(direct_ids.shape[1])
        routing_steps = 0
        routing_duration_seconds_total = 0.0
        materialized_memory_tokens_total = 0.0
        max_materialized_memory_tokens_observed = 0.0
        for _ in range(max_new_tokens):
            idx = direct_ids[:, -direct_limit:]
            position_offset = torch.tensor(
                [
                    self.cfg.prompt_tail_position_offset(len(head), idx.shape[1])
                    for head in head_rows
                ],
                dtype=torch.long,
                device=idx.device,
            )
            logits = self(
                idx,
                use_pra_memory=use_pra_memory,
                position_offset=position_offset,
            )
            step_diagnostics = list(self.pra_diagnostics_by_layer().values())
            step_materialized_by_layer = [
                float(row.get("memory_tokens_materialized", 0.0))
                for row in step_diagnostics
            ]
            step_materialized = sum(step_materialized_by_layer) / max(
                len(step_materialized_by_layer), 1
            )
            if any(row.get("routing_candidates", 0.0) for row in step_diagnostics):
                routing_steps += 1
            routing_duration_seconds_total += sum(
                float(row.get("routing_duration_seconds", 0.0))
                for row in step_diagnostics
            )
            materialized_memory_tokens_total += step_materialized
            max_materialized_memory_tokens_observed = max(
                max_materialized_memory_tokens_observed,
                max(step_materialized_by_layer, default=0.0),
            )
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            direct_ids = torch.cat([direct_ids, next_id], dim=1)
            output_ids = torch.cat([output_ids, next_id], dim=1)
            if direct_ids.shape[1] > direct_limit:
                if self.cfg.prompt_overflow_mode == "error":
                    raise ValueError("Generated direct history exceeded its configured limit.")
                if self.cfg.prompt_overflow_mode == "implicit_reference" and use_pra_memory:
                    migrate = min(rollover_unit, int(direct_ids.shape[1]) - 1)
                    migrated = direct_ids[:, :migrate]
                    direct_ids = direct_ids[:, migrate:]
                    for row_index, cache in enumerate(active_caches):
                        head_rows[row_index].extend(
                            int(value) for value in migrated[row_index]
                        )
                        if hasattr(cache, "invalidate"):
                            cache.invalidate(IMPLICIT_PROMPT_HEAD_URI)
                        entry = self.encode_reference_tokens_to_cache(
                            IMPLICIT_PROMPT_HEAD_URI,
                            head_rows[row_index],
                            tokenizer,
                            input_ids.device,
                            metadata={
                                "implicit": True,
                                "source": "prompt",
                                "kind": "streaming_head",
                                "display_name": IMPLICIT_PROMPT_HEAD_NAME,
                                "prompt_row": row_index,
                                "streaming": True,
                                "rollover_events": rollover_events + 1,
                                "tokens_migrated": len(head_rows[row_index]),
                            },
                            max_chunks=self.cfg.max_prompt_gists,
                            use_configured_max_chunks=False,
                            max_chunk_tokens=self.cfg.effective_model_max_context_tokens,
                            historical_encoding=(
                                self.cfg.prompt_position_mode == "historical"
                            ),
                        )
                        cache.put(entry)
                    rollover_events += 1
                    tokens_migrated += migrate * int(input_ids.shape[0])
                else:
                    direct_ids = direct_ids[:, -direct_limit:]
            max_direct_tokens_observed = max(
                max_direct_tokens_observed, int(direct_ids.shape[1])
            )
            max_native_operation_tokens = max(
                max_native_operation_tokens,
                int(idx.shape[1]),
                max(
                    (
                        int(
                            cache.get(IMPLICIT_PROMPT_HEAD_URI).metadata.get(
                                "max_encoding_input_tokens", 0
                            )
                        )
                        for cache in active_caches
                        if cache.get(IMPLICIT_PROMPT_HEAD_URI) is not None
                    ),
                    default=0,
                ),
            )
        self.last_generation_stats = {
            "generated_tokens": int(max_new_tokens),
            "direct_tokens": int(direct_ids.shape[1]),
            "head_tokens": sum(len(head) for head in head_rows),
            "rollover_events": rollover_events,
            "tokens_migrated": tokens_migrated,
            "max_direct_tokens_observed": max_direct_tokens_observed,
            "max_native_operation_tokens": max_native_operation_tokens,
            "model_max_context_tokens": self.cfg.effective_model_max_context_tokens,
            "routing_steps": routing_steps,
            "routing_duration_seconds_total": routing_duration_seconds_total,
            "materialized_memory_tokens_total": materialized_memory_tokens_total,
            "max_materialized_memory_tokens_observed": (
                max_materialized_memory_tokens_observed
            ),
        }
        return output_ids


TinyPRALanguageModel = TinyPRAModel
TransformerBlock = PRATransformerBlock


def convert_sa_model_to_pra(
    source: TinyPRAModel,
    target_config: PRAConfig,
    *,
    pra_cache: PRAMemoryCache | None = None,
) -> TinyPRAModel:
    """Convert a trained ``td_sa`` model into a native-KV PRA architecture.

    The conversion copies each PyTorch multi-head attention Q/K/V slice into
    the corresponding PRA projections and preserves output projection, layer
    norms, MLP, embeddings, and LM head. It introduces no transport parameters
    in the canonical native mode. The resulting model therefore matches the SA
    checkpoint exactly when memory is disabled (with dropout inactive).
    """
    if source.cfg.model_variant != "td_sa":
        raise ValueError("Source model must use model_variant='td_sa'.")
    if target_config.memory_transport != "native_kv":
        raise ValueError("SA conversion targets canonical memory_transport='native_kv'.")
    architecture = (
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "max_seq_len",
        "position_encoding",
        "rope_theta",
    )
    mismatches = [
        name
        for name in architecture
        if getattr(source.cfg, name) != getattr(target_config, name)
    ]
    if mismatches:
        raise ValueError(f"Source and target architecture differ in: {', '.join(mismatches)}")

    target = TinyPRAModel(target_config, pra_cache=pra_cache)
    with torch.no_grad():
        target.token_emb.load_state_dict(source.token_emb.state_dict())
        target.pos_emb.load_state_dict(source.pos_emb.state_dict())
        target.ln.load_state_dict(source.ln.state_dict())
        target.head.load_state_dict(source.head.state_dict())
        for source_block, target_block in zip(source.blocks, target.blocks):
            if not isinstance(
                source_block,
                (VanillaTransformerBlock, PositionAwareTransformerBlock),
            ):
                raise TypeError("Every source td_sa block must be self-attention only.")
            if isinstance(
                target_block,
                (VanillaTransformerBlock, PositionAwareTransformerBlock),
            ):
                if type(target_block) is not type(source_block):
                    raise TypeError("Source and target position-aware block types differ.")
                target_block.load_state_dict(source_block.state_dict())
                continue
            if not isinstance(target_block, PRATransformerBlock):
                raise TypeError("SA conversion supports vanilla or PRA target blocks.")

            if isinstance(source_block, VanillaTransformerBlock):
                layer = source_block.layer
                target_block.ln1.load_state_dict(layer.norm1.state_dict())
                target_block.ln2.load_state_dict(layer.norm2.state_dict())
                q_weight, k_weight, v_weight = layer.self_attn.in_proj_weight.chunk(3, dim=0)
                q_bias, k_bias, v_bias = layer.self_attn.in_proj_bias.chunk(3, dim=0)
                for projection, weight, bias in (
                    (target_block.attn.q_proj, q_weight, q_bias),
                    (target_block.attn.k_proj, k_weight, k_bias),
                    (target_block.attn.v_proj, v_weight, v_bias),
                ):
                    projection.weight.copy_(weight)
                    projection.bias.copy_(bias)
                target_block.attn.o_proj.load_state_dict(layer.self_attn.out_proj.state_dict())
                target_block.ff[0].load_state_dict(layer.linear1.state_dict())
                target_block.ff[2].load_state_dict(layer.linear2.state_dict())
            else:
                target_block.ln1.load_state_dict(source_block.ln1.state_dict())
                target_block.ln2.load_state_dict(source_block.ln2.state_dict())
                for target_projection, source_projection in (
                    (target_block.attn.q_proj, source_block.attn.q_proj),
                    (target_block.attn.k_proj, source_block.attn.k_proj),
                    (target_block.attn.v_proj, source_block.attn.v_proj),
                    (target_block.attn.o_proj, source_block.attn.o_proj),
                ):
                    target_projection.load_state_dict(source_projection.state_dict())
                target_block.ff[0].load_state_dict(source_block.ff[0].state_dict())
                target_block.ff[2].load_state_dict(source_block.ff[3].state_dict())
    return target


def convert_sa_checkpoint_to_pra(
    checkpoint: dict,
    target_config: PRAConfig,
    *,
    map_location: str | torch.device = "cpu",
) -> TinyPRAModel:
    """Instantiate and convert a serialized project ``td_sa`` checkpoint."""
    source_config_values = dict(checkpoint["cfg"])
    source_config_values["device"] = str(map_location)
    source_config = PRAConfig(**source_config_values)
    source = TinyPRAModel(source_config).to(map_location)
    source.load_state_dict(checkpoint["model"])
    return convert_sa_model_to_pra(source, target_config).to(map_location)
