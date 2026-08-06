"""Layer-specific routing-gist computation from projected attention keys."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def projected_token_keys(layer_k: torch.Tensor) -> torch.Tensor:
    """Merge projected heads per token: [1,H,T,Dh] -> [T,H*Dh]."""
    if layer_k.ndim != 4:
        raise ValueError(
            "Expected layer keys [batch, heads, tokens, head_dim], "
            f"got {tuple(layer_k.shape)}."
        )
    if layer_k.shape[0] != 1:
        raise ValueError("Reference gist construction expects one chunk per encoding call.")
    return (
        layer_k.transpose(1, 2)
        .contiguous()
        .view(1, layer_k.shape[2], -1)
        .squeeze(0)
    )


def mean_pool_layer_keys(layer_k: torch.Tensor) -> torch.Tensor:
    """Pool ``[1,H,T,Dh]`` keys into one ``[d_model]`` routing vector."""
    return projected_token_keys(layer_k).mean(dim=0)


class GRUGistPooler(nn.Module):
    """Learned recurrent reduction from ``[B,T,d_model]`` to ``[B,d_model]``.

    Unlike mean/last pooling, this module has parameters and must be registered
    on ``TinyPRAModel`` so optimizers and checkpoints include it.
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        """Create the GRU and a projection back to the model/routing width."""
        super().__init__()
        hidden_size = int(hidden_size or d_model)
        self.bidirectional = bool(bidirectional)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        directions = 2 if self.bidirectional else 1
        self.output = nn.Linear(hidden_size * directions, d_model)

    def forward(self, token_keys: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Pool padded token-key sequences, optionally using true lengths."""
        if token_keys.ndim != 3:
            raise ValueError(f"Expected token keys [batch,tokens,model], got {token_keys.shape}.")
        if lengths is not None:
            packed = pack_padded_sequence(
                token_keys,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(token_keys)
        if self.bidirectional:
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            pooled = hidden[-1]
        return self.output(pooled)


def compute_routing_gist(
    layer_k: torch.Tensor,
    *,
    mode: str,
    token_ids: tuple[int, ...] | list[int],
    tokenizer=None,
    ref_end_token: str = "<REF_END>",
    gru_pooler: GRUGistPooler | None = None,
) -> torch.Tensor:
    """Compute one ``[d_model]`` routing gist for a chunk at a specific layer.

    Gists are derived from projected attention keys, not raw token embeddings,
    so the query and gist inhabit the same layer-specific key space. ``ref_end``
    selects an explicit terminator position; ``gru`` delegates to the model's
    registered learned pooler.
    """
    token_keys = projected_token_keys(layer_k)
    if token_keys.shape[0] == 0:
        raise ValueError("Cannot compute a routing gist from an empty chunk.")
    if mode == "mean":
        return token_keys.mean(dim=0)
    if mode == "last":
        return token_keys[-1]
    if mode == "ref_end":
        if tokenizer is None:
            raise ValueError("ref_end gist mode requires a tokenizer.")
        marker_ids = tokenizer.encode(ref_end_token)
        if len(marker_ids) != 1:
            raise ValueError(f"Reference terminator {ref_end_token!r} is not one atomic token.")
        indices = [index for index, token_id in enumerate(token_ids) if token_id == marker_ids[0]]
        if len(indices) != 1:
            raise ValueError(
                f"Expected exactly one {ref_end_token!r} token in the chunk, found {len(indices)}."
            )
        return token_keys[indices[0]]
    if mode == "gru":
        if gru_pooler is None:
            raise ValueError("gist_mode='gru' requires the model's registered GRU gist pooler.")
        lengths = torch.tensor([token_keys.shape[0]], device=token_keys.device)
        return gru_pooler(token_keys.unsqueeze(0), lengths=lengths).squeeze(0)
    raise ValueError(f"Unsupported gist mode: {mode}")
