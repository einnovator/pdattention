"""Attention masks shared by vanilla and PRA decoder blocks."""

from __future__ import annotations

import torch


def causal_attention_mask(
    seq_len: int,
    device: torch.device | str,
    window: int | None = None,
) -> torch.Tensor:
    """Return a boolean ``[T,T]`` mask for causal global or strict local attention.

    ``True`` entries are hidden.  A finite ``window`` includes the current token,
    so query position ``i`` can read keys ``max(0, i-window+1)..i``.
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if window is not None:
        window = int(window)
        if window <= 0:
            raise ValueError("window must be positive or None.")
    query = torch.arange(seq_len, device=device)[:, None]
    key = torch.arange(seq_len, device=device)[None, :]
    hidden = key > query
    if window is not None:
        hidden = hidden | (key < query - window + 1)
    return hidden
