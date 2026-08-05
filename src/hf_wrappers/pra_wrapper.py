"""Experimental HF PRA adapter.

This is intentionally a compatibility adapter, not a true attention replacement.
It preserves original attention and adds a trainable cross-attention memory branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleHFMemoryIndex:
    def __init__(self, memory_texts, tokenizer, model, device):
        self.memory_texts = memory_texts
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.memory_hidden = []
        self.content_routing_vectors = []
        self.build()

    @torch.no_grad()
    def build(self):
        emb = self.model.get_input_embeddings()
        for text in self.memory_texts:
            ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(self.device)
            hidden = emb(ids).squeeze(0)
            routing_vector = hidden.mean(dim=0)
            self.memory_hidden.append(hidden.detach())
            self.content_routing_vectors.append(routing_vector.detach())
        self.content_routing_vectors = torch.stack(self.content_routing_vectors, dim=0)

    def search(self, query, top_k=2, threshold=0.2):
        query = F.normalize(query[0], dim=-1)
        routing_vectors = F.normalize(self.content_routing_vectors.to(query.device), dim=-1)
        scores = routing_vectors @ query
        vals, idx = torch.topk(scores, k=min(top_k, len(self.memory_hidden)))
        blocks = []
        for v, i in zip(vals, idx):
            if float(v) >= threshold:
                blocks.append(self.memory_hidden[int(i)].to(query.device))
        if not blocks:
            return None
        return torch.cat(blocks, dim=0).unsqueeze(0)


class PRAAttentionAdapter(nn.Module):
    def __init__(self, original_attn, hidden_size, memory_index, alpha=0.05, top_k=2, threshold=0.2):
        super().__init__()
        self.original_attn = original_attn
        self.memory_index = memory_index
        self.alpha = alpha
        self.top_k = top_k
        self.threshold = threshold
        self.mem_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mem_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mem_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mem_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        original_result = self.original_attn(hidden_states, *args, **kwargs)
        if isinstance(original_result, tuple):
            original_hidden = original_result[0]
            rest = original_result[1:]
        else:
            original_hidden = original_result
            rest = ()

        retrieved = self.memory_index.search(hidden_states[:, -1, :], self.top_k, self.threshold)
        if retrieved is None:
            return original_result

        retrieved = retrieved.to(hidden_states.device).to(hidden_states.dtype)
        q = self.mem_q(hidden_states)
        k = self.mem_k(retrieved)
        v = self.mem_v(retrieved)
        scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        mem_out = self.mem_o(weights @ v)
        mixed = original_hidden + self.alpha * mem_out
        if isinstance(original_result, tuple):
            return (mixed, *rest)
        return mixed


def patch_decoder_layers(model, memory_index, layer_ids, alpha=0.05):
    hidden_size = model.config.hidden_size
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        layer.self_attn = PRAAttentionAdapter(layer.self_attn, hidden_size, memory_index, alpha=alpha)
    return model
