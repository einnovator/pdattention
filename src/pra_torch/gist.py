"""Compatibility facade for the strategy-based multi-gist implementation."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from .gists import GRUGistPooler, GistContext, compute_gists, projected_tokens


def projected_token_keys(layer_k: torch.Tensor) -> torch.Tensor:
    """Merge projected heads per token: ``[1,H,T,Dh] -> [T,D]``."""
    return projected_tokens(layer_k)


def mean_pool_layer_keys(layer_k: torch.Tensor) -> torch.Tensor:
    """Compatibility helper returning one unbatched ``[D]`` mean vector."""
    return projected_tokens(layer_k).mean(dim=0)


def compute_routing_gist(
    layer_k: torch.Tensor,
    *,
    mode: str,
    token_ids,
    tokenizer=None,
    ref_end_token: str = "<REF_END>",
    gru_pooler: GRUGistPooler | None = None,
) -> torch.Tensor:
    """Compute a single routing-gist collection shaped ``[1,D]``.

    New model code uses :func:`pra_torch.gists.compute_gists` with the complete
    ``PRAConfig``. This wrapper preserves the former import path for notebooks.
    """
    points = projected_tokens(layer_k)
    config = SimpleNamespace()
    result = compute_gists(
        keys=points,
        values=None,
        mode=mode,
        num_gists=1,
        config=config,
        context=GistContext(
            level="chunk",
            token_ids=token_ids,
            tokenizer=tokenizer,
            ref_end_token=ref_end_token,
            gru_pooler=gru_pooler,
        ),
    )
    return result.k


__all__ = [
    "GRUGistPooler",
    "compute_routing_gist",
    "mean_pool_layer_keys",
    "projected_token_keys",
]
