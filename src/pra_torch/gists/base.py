"""Shared types for chunk-level and reference-level gist strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch


@dataclass(frozen=True)
class GistContext:
    """Information needed by strategies that depend on source position or modules."""

    level: str  # ``chunk`` or ``reference`` for diagnostics and future policy overrides.
    token_ids: Sequence[int] = ()  # Source token IDs used by positional single-gist modes.
    tokenizer: Any = None  # Tokenizer required to locate an atomic ``ref_end`` marker.
    ref_end_token: str = "<REF_END>"  # Marker selected by ``ref_end`` mode.
    gru_pooler: Any = None  # Registered learned pooler owned by ``TinyPRAModel``.


@dataclass
class ComputedGists:
    """Paired routing keys/values with a uniform ``[G,D]`` shape contract."""

    k: torch.Tensor  # Routing keys shaped [actual_gists, d_model].
    v: torch.Tensor | None = None  # Paired values shaped [actual_gists, d_model].
    metadata: dict[str, Any] = field(default_factory=dict)  # Strategy and occupancy details.


class GistStrategy(Protocol):
    """Interface implemented by every single- or multi-gist strategy."""

    def compute(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor | None,
        num_gists: int,
        config: Any,
        context: GistContext,
    ) -> ComputedGists:
        """Compress paired ``[N,D]`` points into at most ``num_gists`` gists."""
        ...
