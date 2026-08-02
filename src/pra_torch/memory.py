from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class LayerKV:
    """Cached key/value tensors for one transformer layer."""

    # [batch, n_heads, seq_len, head_dim] in standalone TinyGPT
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class PRACacheEntry:
    """A resolved reference encoded into summary and per-layer memory tensors."""

    uri: str
    text: str
    summary: str
    summary_vector: torch.Tensor
    layer_kv: dict[int, LayerKV] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class PRAMemoryCache(ABC):
    """Abstract cache interface consumed by PRA attention layers."""

    @abstractmethod
    def put(self, entry: PRACacheEntry) -> None:
        """Insert or replace a cache entry."""
        raise NotImplementedError

    @abstractmethod
    def get(self, uri: str) -> PRACacheEntry | None:
        """Return a cache entry by URI when present."""
        raise NotImplementedError

    @abstractmethod
    def has(self, uri: str) -> bool:
        """Return whether a URI is already cached."""
        raise NotImplementedError

    @abstractmethod
    def all_entries(self) -> list[PRACacheEntry]:
        """Return all entries in retrieval order."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""
        raise NotImplementedError

    @abstractmethod
    def search_by_summary(self, query: torch.Tensor, top_k: int = 2):
        """Find entries whose summary vectors best match the query vector."""
        raise NotImplementedError

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        return {entry.uri: entry for entry in self.all_entries()}

    def __len__(self) -> int:
        return len(self.all_entries())


class PRASimpleMemoryCache(PRAMemoryCache):
    """In-memory dictionary-backed cache implementation for experiments."""

    def __init__(self):
        self._entries: dict[str, PRACacheEntry] = {}

    @property
    def entries(self) -> dict[str, PRACacheEntry]:
        return self._entries

    def put(self, entry: PRACacheEntry) -> None:
        self._entries[entry.uri] = entry

    def get(self, uri: str) -> PRACacheEntry | None:
        return self._entries.get(uri)

    def has(self, uri: str) -> bool:
        return uri in self._entries

    def all_entries(self) -> list[PRACacheEntry]:
        return list(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def search_by_summary(self, query: torch.Tensor, top_k: int = 2):
        """Return top-k entries by cosine similarity.

        query: [batch, d_model] or [d_model]
        """
        if not self._entries:
            return []
        if query.dim() == 2:
            query = query[0]
        query = torch.nn.functional.normalize(query, dim=-1)
        entries = self.all_entries()
        summaries = torch.stack([e.summary_vector.to(query.device) for e in entries], dim=0)
        summaries = torch.nn.functional.normalize(summaries, dim=-1)
        scores = summaries @ query
        k = min(top_k, len(entries))
        vals, idx = torch.topk(scores, k=k)
        return [(entries[int(i)], float(v.detach())) for v, i in zip(vals, idx)]
