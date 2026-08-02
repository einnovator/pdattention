"""Progressive Retrieval Attention layer for the standalone transformer."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory import PRAMemoryCache, LayerKV


class PRAttention(nn.Module):
    """Standalone PRA attention for TinyGPT.

    This version preserves normal causal self-attention, then optionally adds a
    cross-attention memory branch from layer-specific cached reference K/V.

    The memory cache must contain K/V for the same layer_id.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        max_seq_len,
        layer_id,
        pra_cache: PRAMemoryCache,
        top_k_refs=2,
        trigger_threshold=0.2,
        memory_alpha=0.5,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.layer_id = layer_id
        self.top_k_refs = top_k_refs
        self.trigger_threshold = trigger_threshold
        self.memory_alpha = memory_alpha
        self.pra_cache = pra_cache
        self.last_selected_references: list[tuple[str, float]] = []

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Separate output projection for memory branch. Simpler than sharing o_proj.
        self.mem_o_proj = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def split_heads(self, x):
        """Convert ``[batch, seq, model]`` tensors to multi-head layout."""
        b, t, d = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        """Convert multi-head attention output back to model layout."""
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def project_kv(self, hidden_states) -> LayerKV:
        """Project hidden states into detached K/V tensors for cache storage."""
        k = self.split_heads(self.k_proj(hidden_states))
        v = self.split_heads(self.v_proj(hidden_states))
        return LayerKV(k=k.detach(), v=v.detach())

    def forward(self, x, use_pra_memory: bool = True):
        """Apply causal self-attention and optionally add retrieved memory output."""
        self.last_selected_references = []
        b, t, _ = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :t, :t]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        local_out = self.merge_heads(weights @ v)
        local_out = self.o_proj(local_out)

        if not use_pra_memory or not self.pra_cache.entries:
            return local_out

        # Retrieval query = last hidden state before attention. For better version,
        # use attention to explicit ref token positions too.
        retrieved = self.pra_cache.search_by_summary(x[:, -1, :], top_k=self.top_k_refs)
        selected_k = []
        selected_v = []
        for entry, sim in retrieved:
            if sim < self.trigger_threshold:
                continue
            if self.layer_id not in entry.layer_kv:
                continue
            self.last_selected_references.append((entry.uri, sim))
            kv = entry.layer_kv[self.layer_id]
            selected_k.append(kv.k.to(x.device))
            selected_v.append(kv.v.to(x.device))

        if not selected_k:
            return local_out

        mem_k = torch.cat(selected_k, dim=2)  # [1,h,mem_len,d]
        mem_v = torch.cat(selected_v, dim=2)
        if mem_k.shape[0] == 1 and b > 1:
            mem_k = mem_k.expand(b, -1, -1, -1)
            mem_v = mem_v.expand(b, -1, -1, -1)

        mem_scores = q @ mem_k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mem_weights = F.softmax(mem_scores, dim=-1)
        mem_out = self.merge_heads(mem_weights @ mem_v)
        mem_out = self.mem_o_proj(mem_out)
        return local_out + self.memory_alpha * mem_out
