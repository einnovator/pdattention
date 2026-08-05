"""Dynamic rectangular batching for variable-length selected PRA memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryBucket:
    original_indices: tuple[int, ...]
    lengths: tuple[int, ...]
    max_length: int


@dataclass(frozen=True)
class MemoryBatchingStats:
    selected_lengths: tuple[int, ...]
    requested_bucket_count: int
    actual_bucket_count: int
    bucket_membership: tuple[tuple[int, ...], ...]
    bucket_max_lengths: tuple[int, ...]
    valid_positions: int
    allocated_positions: int
    padding_positions: int
    padding_fraction: float
    attention_entropy: float = 0.0
    attention_max_weight: float = 0.0

    def as_metrics(self) -> dict[str, float]:
        lengths = self.selected_lengths
        return {
            "memory_valid_positions": float(self.valid_positions),
            "memory_allocated_positions": float(self.allocated_positions),
            "memory_padding_positions": float(self.padding_positions),
            "memory_padding_fraction": self.padding_fraction,
            "memory_actual_bucket_count": float(self.actual_bucket_count),
            "memory_max_selected_length": float(max(lengths, default=0)),
            "memory_mean_selected_length": float(sum(lengths) / max(len(lengths), 1)),
            "memory_zero_selection_fraction": float(
                sum(length == 0 for length in lengths) / max(len(lengths), 1)
            ),
            "memory_attention_entropy": self.attention_entropy,
            "memory_attention_max_weight": self.attention_max_weight,
        }


class MemoryBucketPlanner:
    def plan(
        self,
        lengths: Sequence[int],
        max_buckets: int,
        strategy: str = "optimal_contiguous",
    ) -> list[MemoryBucket]:
        if max_buckets <= 0:
            return [MemoryBucket((index,), (int(length),), int(length)) for index, length in enumerate(lengths) if length]
        items = sorted(
            ((int(length), index) for index, length in enumerate(lengths) if int(length) > 0),
            key=lambda item: (item[0], item[1]),
        )
        if not items:
            return []
        bucket_count = min(int(max_buckets), len(items))
        if bucket_count == 1:
            groups = [items]
        elif strategy == "equal_count":
            groups = []
            for bucket_index in range(bucket_count):
                start = bucket_index * len(items) // bucket_count
                end = (bucket_index + 1) * len(items) // bucket_count
                if start < end:
                    groups.append(items[start:end])
        elif strategy == "optimal_contiguous":
            groups = self._optimal_groups(items, bucket_count)
        else:
            raise ValueError(f"Unsupported memory bucket strategy: {strategy}")
        return [
            MemoryBucket(
                original_indices=tuple(index for _length, index in group),
                lengths=tuple(length for length, _index in group),
                max_length=max(length for length, _index in group),
            )
            for group in groups
        ]

    @staticmethod
    def _optimal_groups(items, bucket_count):
        n = len(items)
        infinity = float("inf")
        costs = [[infinity] * (n + 1) for _ in range(bucket_count + 1)]
        previous = [[-1] * (n + 1) for _ in range(bucket_count + 1)]
        costs[0][0] = 0
        for groups in range(1, bucket_count + 1):
            for end in range(groups, n + 1):
                for start in range(groups - 1, end):
                    cost = costs[groups - 1][start] + (end - start) * items[end - 1][0]
                    if cost < costs[groups][end]:
                        costs[groups][end] = cost
                        previous[groups][end] = start
        groups = []
        end = n
        for group_count in range(bucket_count, 0, -1):
            start = previous[group_count][end]
            groups.append(items[start:end])
            end = start
        return list(reversed(groups))


def padded_memory_attention(
    q: torch.Tensor,
    memory_k_by_item: Sequence[torch.Tensor],
    memory_v_by_item: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, float, float]:
    """Attend over one masked rectangle and return reduced diagnostics."""
    if len(memory_k_by_item) != q.shape[0] or len(memory_v_by_item) != q.shape[0]:
        raise ValueError("Memory rows must match the query batch.")
    lengths = [int(value.shape[2]) for value in memory_k_by_item]
    if not lengths or max(lengths, default=0) == 0:
        return torch.zeros_like(q), 0.0, 0.0
    max_length = max(lengths)
    padded_k = []
    padded_v = []
    for key, value, length in zip(memory_k_by_item, memory_v_by_item, lengths):
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError("Each memory K/V pair must have matching [1,H,M,Dh] shapes.")
        if key.shape[0] != 1 or key.shape[1] != q.shape[1] or key.shape[3] != q.shape[3]:
            raise ValueError("Memory K/V head dimensions do not match the query.")
        pad = max_length - length
        padded_k.append(F.pad(key.to(q.device, q.dtype), (0, 0, 0, pad)))
        padded_v.append(F.pad(value.to(q.device, q.dtype), (0, 0, 0, pad)))
    memory_k = torch.cat(padded_k, dim=0)
    memory_v = torch.cat(padded_v, dim=0)
    length_tensor = torch.tensor(lengths, device=q.device)
    mask = torch.arange(max_length, device=q.device)[None, :] < length_tensor[:, None]
    scores = q @ memory_k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    weights = weights * mask[:, None, None, :]
    output = weights @ memory_v
    safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = float((-(weights * safe_weights.log()).sum(dim=-1).mean()).detach().cpu())
    maximum = float(weights.max().detach().cpu())
    return output, entropy, maximum


def dynamic_memory_attention(
    q: torch.Tensor,
    memory_k_by_item: Sequence[torch.Tensor],
    memory_v_by_item: Sequence[torch.Tensor],
    *,
    bucket_count: int,
    bucket_strategy: str,
) -> tuple[torch.Tensor, MemoryBatchingStats]:
    """Run isolated or bucketed attention and restore original batch order."""
    if len(memory_k_by_item) != q.shape[0] or len(memory_v_by_item) != q.shape[0]:
        raise ValueError("Memory list length must match query batch size.")
    lengths = [int(key.shape[2]) for key in memory_k_by_item]
    planner = MemoryBucketPlanner()
    buckets = planner.plan(lengths, bucket_count, strategy=bucket_strategy)
    output = torch.zeros_like(q)
    entropies = []
    maxima = []
    for bucket in buckets:
        indices = torch.tensor(bucket.original_indices, device=q.device, dtype=torch.long)
        bucket_q = q.index_select(0, indices)
        bucket_k = [memory_k_by_item[index] for index in bucket.original_indices]
        bucket_v = [memory_v_by_item[index] for index in bucket.original_indices]
        bucket_output, entropy, maximum = padded_memory_attention(bucket_q, bucket_k, bucket_v)
        output = output.index_copy(0, indices, bucket_output)
        entropies.append(entropy)
        maxima.append(maximum)
    valid = sum(lengths)
    allocated = sum(len(bucket.original_indices) * bucket.max_length for bucket in buckets)
    padding = allocated - valid
    stats = MemoryBatchingStats(
        selected_lengths=tuple(lengths),
        requested_bucket_count=int(bucket_count),
        actual_bucket_count=len(buckets),
        bucket_membership=tuple(bucket.original_indices for bucket in buckets),
        bucket_max_lengths=tuple(bucket.max_length for bucket in buckets),
        valid_positions=valid,
        allocated_positions=allocated,
        padding_positions=padding,
        padding_fraction=padding / max(allocated, 1),
        attention_entropy=sum(entropies) / max(len(entropies), 1),
        attention_max_weight=sum(maxima) / max(len(maxima), 1),
    )
    return output, stats
