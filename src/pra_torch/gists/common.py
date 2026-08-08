"""Tensor conversion, paired aggregation, deterministic initialization, and scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import ComputedGists


@dataclass(frozen=True)
class GistScore:
    """Reduced query-to-gist score plus the winning gist diagnostic."""

    aggregate_score: float
    winning_index: int | None
    per_gist_scores: torch.Tensor | None


def projected_tokens(layer_tensor: torch.Tensor) -> torch.Tensor:
    """Merge projected heads per token: ``[1,H,T,Dh] -> [T,H*Dh]``."""
    if layer_tensor.ndim != 4:
        raise ValueError(
            "Expected projected tensor [batch,heads,tokens,head_dim], "
            f"got {tuple(layer_tensor.shape)}."
        )
    if layer_tensor.shape[0] != 1:
        raise ValueError("Gist construction expects one independently encoded source.")
    return layer_tensor.transpose(1, 2).contiguous().view(layer_tensor.shape[2], -1)


def validate_points(keys: torch.Tensor, values: torch.Tensor | None) -> None:
    """Validate the shared point representation and K/V row correspondence."""
    if keys.ndim != 2:
        raise ValueError(f"Expected gist source keys [points,model], got {tuple(keys.shape)}.")
    if values is not None and (values.ndim != 2 or values.shape != keys.shape):
        raise ValueError(
            "Gist source values must match key shape exactly: "
            f"{tuple(values.shape)} != {tuple(keys.shape)}."
        )


def empty_gists(keys: torch.Tensor, values: torch.Tensor | None, **metadata) -> ComputedGists:
    """Return an empty collection while preserving width, device, and dtype."""
    width = int(keys.shape[-1]) if keys.ndim == 2 else 0
    empty_k = keys.new_empty((0, width))
    empty_v = empty_k.clone() if values is not None else None
    return ComputedGists(k=empty_k, v=empty_v, metadata=metadata)


def local_generator(seed: int) -> torch.Generator:
    """Create a CPU generator so strategies never consume global RNG state."""
    return torch.Generator(device="cpu").manual_seed(int(seed))


def normalize_rows(points: torch.Tensor) -> torch.Tensor:
    """L2-normalize point rows without producing NaNs for zero vectors."""
    return F.normalize(points, dim=-1, eps=1e-12)


def assign_to_centers(
    points: torch.Tensor,
    centers: torch.Tensor,
    *,
    distance: str,
) -> torch.Tensor:
    """Assign every point to its nearest center under cosine or Euclidean distance."""
    if centers.shape[0] == 0:
        return torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    if distance == "cosine":
        scores = normalize_rows(points) @ normalize_rows(centers).transpose(0, 1)
        return scores.argmax(dim=1)
    if distance == "euclidean":
        return torch.cdist(points, centers).argmin(dim=1)
    raise ValueError(f"Unsupported gist distance: {distance}")


def paired_means(
    keys: torch.Tensor,
    values: torch.Tensor | None,
    assignments: torch.Tensor,
    cluster_count: int,
) -> tuple[torch.Tensor, torch.Tensor | None, list[int], list[int]]:
    """Mean K and V over the same occupied assignments, dropping empty clusters."""
    occupied = [
        index for index in range(int(cluster_count)) if bool((assignments == index).any())
    ]
    if not occupied:
        empty = empty_gists(keys, values)
        return empty.k, empty.v, [], []
    key_gists = torch.stack([keys[assignments == index].mean(dim=0) for index in occupied])
    value_gists = (
        torch.stack([values[assignments == index].mean(dim=0) for index in occupied])
        if values is not None
        else None
    )
    occupancies = [int((assignments == index).sum().item()) for index in occupied]
    return key_gists, value_gists, occupied, occupancies


def mean_cosine_separation(gists: torch.Tensor) -> float:
    """Return mean pairwise cosine distance for inexpensive diversity diagnostics."""
    if gists.shape[0] < 2:
        return 0.0
    normalized = normalize_rows(gists.detach())
    similarity = normalized @ normalized.transpose(0, 1)
    row, column = torch.triu_indices(gists.shape[0], gists.shape[0], offset=1)
    return float((1.0 - similarity[row, column]).mean().cpu())


def score_gist_set(
    query: torch.Tensor,
    gists: torch.Tensor,
    aggregation: str = "max",
) -> GistScore:
    """Cosine-score one query against ``[G,D]`` gists and reduce the set."""
    if query.ndim != 1:
        raise ValueError(f"Expected one routing query [model], got {tuple(query.shape)}.")
    if gists.ndim != 2:
        raise ValueError(f"Expected routing gists [gists,model], got {tuple(gists.shape)}.")
    if gists.shape[0] == 0:
        return GistScore(float("-inf"), None, None)
    local = gists.to(query.device, query.dtype)
    scores = normalize_rows(local) @ F.normalize(query, dim=-1, eps=1e-12)
    winning_index = int(scores.argmax().item())
    if aggregation == "max":
        aggregate = scores[winning_index]
    elif aggregation == "mean":
        aggregate = scores.mean()
    elif aggregation == "logsumexp":
        aggregate = torch.logsumexp(scores, dim=0)
    else:
        raise ValueError(f"Unsupported gist score aggregation: {aggregation}")
    return GistScore(
        aggregate_score=float(aggregate.detach().cpu()),
        winning_index=winning_index,
        per_gist_scores=scores.detach(),
    )
