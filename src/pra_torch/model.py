"""Tiny standalone transformer used to study Progressive Retrieval Attention."""

from contextlib import nullcontext
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PRAConfig
from .attention import PRAttention
from .chunking import partition_reference
from .gist import GRUGistPooler, compute_routing_gist
from .memory import (
    ChunkRoutingGist,
    LayerReferenceMemory,
    PRAMemoryCache,
    PRACacheEntry,
    PRASimpleMemoryCache,
    ReferenceChunkMemory,
)


def causal_attention_mask(seq_len: int, device) -> torch.Tensor:
    """Return a boolean causal mask for PyTorch attention modules."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class VanillaTransformerBlock(nn.Module):
    """Standard PyTorch transformer encoder layer used as an early block."""

    def __init__(self, cfg: PRAConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, use_pra_memory: bool = True):
        """Run vanilla causal transformer attention without PRA memory."""
        _ = use_pra_memory
        mask = causal_attention_mask(x.shape[1], x.device)
        return self.layer(x, src_mask=mask)


class PRATransformerBlock(nn.Module):
    """One decoder block with causal self-attention plus optional PRA memory."""

    def __init__(self, cfg: PRAConfig, layer_id: int, pra_cache: PRAMemoryCache):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = PRAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            max_seq_len=cfg.max_seq_len,
            layer_id=layer_id,
            pra_cache=pra_cache,
            config=cfg,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x, use_pra_memory: bool = True):
        """Run the block, optionally allowing the attention layer to read memory."""
        x = x + self.attn(self.ln1(x), use_pra_memory=use_pra_memory)
        x = x + self.ff(self.ln2(x))
        return x

    def set_pra_cache(self, pra_cache: PRAMemoryCache) -> None:
        """Attach a cache to this block's PRA attention."""
        self.attn.pra_cache = pra_cache

    def project_reference_kv(self, x, *, detach: bool = True):
        """Project reference hidden states into this block's PRA K/V space."""
        return self.attn.project_kv(self.ln1(x), detach=detach)


class PRASATransformerBlock(nn.Module):
    """Mixed block with vanilla self-attention followed by PRA attention."""

    def __init__(self, cfg: PRAConfig, layer_id: int, pra_cache: PRAMemoryCache):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.pra_attn = PRAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            max_seq_len=cfg.max_seq_len,
            layer_id=layer_id,
            pra_cache=pra_cache,
            config=cfg,
        )
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def _apply_self_attention(self, x):
        norm_x = self.ln1(x)
        mask = causal_attention_mask(x.shape[1], x.device)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False)
        return x + attn_out

    def forward(self, x, use_pra_memory: bool = True):
        """Run vanilla causal self-attention, then PRA attention, then MLP."""
        x = self._apply_self_attention(x)
        x = x + self.pra_attn(self.ln2(x), use_pra_memory=use_pra_memory)
        x = x + self.ff(self.ln3(x))
        return x

    def set_pra_cache(self, pra_cache: PRAMemoryCache) -> None:
        """Attach a cache to this block's PRA attention."""
        self.pra_attn.pra_cache = pra_cache

    def project_reference_kv(self, x, *, detach: bool = True):
        """Project reference states after the vanilla sublayer into PRA K/V."""
        x = self._apply_self_attention(x)
        return self.pra_attn.project_kv(self.ln2(x), detach=detach)


class TinyPRAModel(nn.Module):
    """A compact GPT-style model with a shared PRA cache across all layers."""

    def __init__(self, cfg: PRAConfig, pra_cache: PRAMemoryCache | None = None):
        super().__init__()
        self.cfg = cfg
        self.pra_cache = pra_cache if pra_cache is not None else PRASimpleMemoryCache()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.gist_pooler = (
            GRUGistPooler(
                cfg.d_model,
                hidden_size=cfg.gist_gru_hidden_size,
                num_layers=cfg.gist_gru_num_layers,
                bidirectional=cfg.gist_gru_bidirectional,
                dropout=cfg.dropout,
            )
            if cfg.gist_mode == "gru"
            else None
        )
        self.blocks = nn.ModuleList(self._build_blocks(cfg))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _build_blocks(self, cfg: PRAConfig) -> list[nn.Module]:
        blocks: list[nn.Module] = []
        for layer_id in range(cfg.n_layers):
            if layer_id < cfg.n_vanilla_layers:
                blocks.append(VanillaTransformerBlock(cfg, layer_id))
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
        """Return per-layer, per-batch selected chunks from the latest forward."""
        selections = {}
        for block in self.blocks:
            attention = getattr(block, "attn", None) or getattr(block, "pra_attn", None)
            if attention is not None:
                selections[block.layer_id] = [list(items) for items in attention.last_selected_chunks]
        return selections

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
        diagnostics = {}
        for block in self.blocks:
            attention = getattr(block, "attn", None) or getattr(block, "pra_attn", None)
            if attention is not None:
                diagnostics[block.layer_id] = {
                    **attention.last_diagnostics,
                    "batching": attention.last_memory_batching_stats,
                }
        return diagnostics

    def forward(self, input_ids, use_pra_memory: bool = True):
        """Return next-token logits for a batch of token ids."""
        b, t = input_ids.shape
        assert t <= self.cfg.max_seq_len
        pos = torch.arange(t, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, use_pra_memory=use_pra_memory)
        x = self.ln(x)
        return self.head(x)

    def _encode_reference_tokens(self, token_ids, device, *, detach: bool, use_pra_memory: bool):
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        pos = torch.arange(ids.shape[1], device=device)
        x = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        layer_kv = {}
        for block in self.blocks:
            if hasattr(block, "project_reference_kv"):
                layer_kv[block.layer_id] = block.project_reference_kv(x, detach=detach)
            x = block(x, use_pra_memory=use_pra_memory)
        return layer_kv

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
        """Encode content chunks into layer-specific routing gists and token K/V."""
        metadata = dict(metadata or {})
        detach = self.cfg.cache_build_mode == "detached"
        context = torch.no_grad() if detach else nullcontext()
        chunks = partition_reference(uri, text, tokenizer, self.cfg, metadata)
        entry = PRACacheEntry(
            uri=uri,
            text=text,
            child_uris=list(metadata.get("child_uris") or []),
            metadata={
                **metadata,
                "chunking_mode": self.cfg.chunking_mode,
                "gist_mode": self.cfg.gist_mode,
                "cache_build_mode": self.cfg.cache_build_mode,
                "use_summary": self.cfg.use_summary,
                "summary_mode": self.cfg.summary_mode,
                "chunk_count": len(chunks),
            },
        )
        summary_by_layer = {}
        summary = metadata.get("summary")
        with context:
            if self.cfg.use_summary and summary:
                summary_ids = list(tokenizer.encode(str(summary)))
                if len(summary_ids) > self.cfg.max_seq_len:
                    summary_ids = summary_ids[: self.cfg.max_seq_len]
                if summary_ids:
                    summary_kv = self._encode_reference_tokens(
                        summary_ids, device, detach=detach, use_pra_memory=False
                    )
                    for layer_id, kv in summary_kv.items():
                        summary_by_layer[layer_id] = compute_routing_gist(
                            kv.k,
                            mode="gru" if self.cfg.gist_mode == "gru" else "mean",
                            token_ids=summary_ids,
                            tokenizer=tokenizer,
                            ref_end_token=self.cfg.ref_end_token,
                            gru_pooler=self.gist_pooler,
                        )

            for chunk in chunks:
                token_ids = list(chunk.token_ids)
                original_length = len(token_ids)
                if original_length > self.cfg.max_seq_len:
                    if self.cfg.reference_overflow_policy == "error":
                        raise ValueError(
                            f"Chunk {chunk.chunk_id} has {original_length} tokens, exceeding "
                            f"max_seq_len={self.cfg.max_seq_len}."
                        )
                    token_ids = token_ids[: self.cfg.max_seq_len]
                layer_kv = self._encode_reference_tokens(
                    token_ids,
                    device,
                    detach=detach,
                    use_pra_memory=use_pra_memory,
                )
                for layer_id, kv in layer_kv.items():
                    gist_k = compute_routing_gist(
                        kv.k,
                        mode=self.cfg.gist_mode,
                        token_ids=token_ids,
                        tokenizer=tokenizer,
                        ref_end_token=self.cfg.ref_end_token,
                        gru_pooler=self.gist_pooler,
                    )
                    gist_v = compute_routing_gist(
                        kv.v,
                        mode=self.cfg.gist_mode,
                        token_ids=token_ids,
                        tokenizer=tokenizer,
                        ref_end_token=self.cfg.ref_end_token,
                        gru_pooler=self.gist_pooler,
                    )
                    if detach:
                        gist_k = gist_k.detach()
                        gist_v = gist_v.detach()
                    chunk_memory = ReferenceChunkMemory(
                        chunk_id=chunk.chunk_id,
                        source_uri=chunk.source_uri,
                        token_start=chunk.token_start,
                        token_end=min(chunk.token_start + len(token_ids), chunk.token_end),
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        token_kv=kv,
                        routing_gist=ChunkRoutingGist(
                            k=gist_k,
                            v=gist_v,
                            method=self.cfg.gist_mode,
                            summary_k=summary_by_layer.get(layer_id),
                            metadata={"summary_available": layer_id in summary_by_layer},
                        ),
                        metadata={
                            **chunk.metadata,
                            "original_token_count": original_length,
                            "retained_token_count": len(token_ids),
                            "truncated": original_length != len(token_ids),
                        },
                    )
                    entry.layer_memory.setdefault(layer_id, LayerReferenceMemory()).chunks.append(
                        chunk_memory
                    )
        return entry

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, use_pra_memory: bool = True):
        """Sample autoregressive continuations from the model."""
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(idx, use_pra_memory=use_pra_memory)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


TinyPRALanguageModel = TinyPRAModel
TransformerBlock = PRATransformerBlock
