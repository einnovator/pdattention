"""Tiny standalone transformer used to study Progressive Retrieval Attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PRAConfig
from .attention import PRAttention
from .memory import PRAMemoryCache, PRACacheEntry, PRASimpleMemoryCache


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
            dim_feedforward=4 * cfg.d_model,
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
            top_k_refs=cfg.top_k_refs,
            trigger_threshold=cfg.trigger_threshold,
            memory_alpha=cfg.memory_alpha,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, use_pra_memory: bool = True):
        """Run the block, optionally allowing the attention layer to read memory."""
        x = x + self.attn(self.ln1(x), use_pra_memory=use_pra_memory)
        x = x + self.ff(self.ln2(x))
        return x

    def set_pra_cache(self, pra_cache: PRAMemoryCache) -> None:
        """Attach a cache to this block's PRA attention."""
        self.attn.pra_cache = pra_cache

    def project_reference_kv(self, x):
        """Project reference hidden states into this block's PRA K/V space."""
        return self.attn.project_kv(self.ln1(x))


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
            top_k_refs=cfg.top_k_refs,
            trigger_threshold=cfg.trigger_threshold,
            memory_alpha=cfg.memory_alpha,
        )
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
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

    def project_reference_kv(self, x):
        """Project reference states after the vanilla sublayer into PRA K/V."""
        x = self._apply_self_attention(x)
        return self.pra_attn.project_kv(self.ln2(x))


class TinyPRAModel(nn.Module):
    """A compact GPT-style model with a shared PRA cache across all layers."""

    def __init__(self, cfg: PRAConfig, pra_cache: PRAMemoryCache | None = None):
        super().__init__()
        self.cfg = cfg
        self.pra_cache = pra_cache if pra_cache is not None else PRASimpleMemoryCache()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
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

    @torch.no_grad()
    def encode_reference_to_cache(self, uri: str, text: str, summary: str, tokenizer, device) -> PRACacheEntry:
        """Encode a reference document separately and capture per-layer K/V.

        This is the standalone version. It runs the reference text through the same
        model path, but projects/stores K/V for each layer from that layer's normalized
        hidden state before attention.
        """
        ids = torch.tensor([tokenizer.encode(text)[: self.cfg.max_seq_len]], dtype=torch.long, device=device)
        sum_ids = torch.tensor([tokenizer.encode(summary)[: self.cfg.max_seq_len]], dtype=torch.long, device=device)

        # Summary vector from input embeddings for initial prototype. TODO: optionally use final hidden.
        summary_hidden = self.token_emb(sum_ids).mean(dim=1).squeeze(0).detach()
        entry = PRACacheEntry(uri=uri, text=text, summary=summary, summary_vector=summary_hidden)

        b, t = ids.shape
        pos = torch.arange(t, device=device)
        x = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]

        for block in self.blocks:
            if hasattr(block, "project_reference_kv"):
                entry.layer_kv[block.layer_id] = block.project_reference_kv(x)
            x = block(x, use_pra_memory=False)  # no recursive PRA while building first cache

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
