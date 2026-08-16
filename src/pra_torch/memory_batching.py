"""Dynamic rectangular batching for variable-length selected PRA memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryBucket:
    """Batch rows grouped into one padded memory-attention rectangle."""

    original_indices: tuple[int, ...]  # Row positions restored after bucket attention.
    lengths: tuple[int, ...]  # Valid selected-memory lengths for those rows.
    max_length: int  # Rectangle width and maximum member length.


@dataclass(frozen=True)
class MemoryBatchingStats:
    """Diagnostics for variable-length memory packing and attention concentration."""

    selected_lengths: tuple[int, ...]  # Materialized memory positions per input row.
    requested_bucket_count: int  # Configured cap; zero requests isolated rows.
    actual_bucket_count: int  # Non-empty rectangles executed.
    bucket_membership: tuple[tuple[int, ...], ...]  # Original rows in each rectangle.
    bucket_max_lengths: tuple[int, ...]  # Padded memory width per rectangle.
    valid_positions: int  # Sum of real memory positions across rows.
    allocated_positions: int  # Rectangle positions allocated across all buckets.
    padding_positions: int  # Allocated positions hidden by masks.
    padding_fraction: float  # Padding divided by allocated positions.
    attention_entropy: float = 0.0  # Mean entropy over memory-attention distributions.
    attention_max_weight: float = 0.0  # Mean bucket maximum attention weight.
    memory_attention_mass: float = 0.0  # Mean joint-softmax mass assigned to memory.
    final_token_memory_attention_mass: float = 0.0  # Memory mass at the next-token query.
    # Per-row [H,M] weights for the final direct-token query. These compact
    # diagnostics let controlled studies distinguish evidence from distractor
    # memory without retaining the full [B,H,T,M+T] attention tensor.
    final_token_memory_weights: tuple[tuple[tuple[float, ...], ...], ...] = ()

    def as_metrics(self) -> dict[str, float]:
        """Flatten packing/attention statistics into experiment metric names."""
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
            "memory_attention_mass": self.memory_attention_mass,
            "final_token_memory_attention_mass": self.final_token_memory_attention_mass,
        }


class MemoryBucketPlanner:
    """Group similar memory lengths to reduce masked rectangular padding."""

    def plan(
        self,
        lengths: Sequence[int],
        max_buckets: int,
        strategy: str = "optimal_contiguous",
    ) -> list[MemoryBucket]:
        """Plan at most ``max_buckets`` without mixing data between batch rows.

        ``optimal_contiguous`` minimizes allocated positions after sorting by
        length; ``equal_count`` is a cheaper deterministic baseline. A zero cap
        deliberately creates one unpadded bucket per non-empty row.
        """
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
        """Use dynamic programming to minimize total padded positions."""
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
    """Attend over one masked variable-memory rectangle.

    Queries are ``[B,H,T,Dh]``. Each input K/V row is ``[1,H,M_i,Dh]``;
    padding forms ``[B,H,max(M_i),Dh]`` and a mask keeps rows isolated. The
    output restores the query shape and includes reduced attention diagnostics.
    """
    if len(memory_k_by_item) != q.shape[0] or len(memory_v_by_item) != q.shape[0]:
        raise ValueError("Memory rows must match the query batch.")
    lengths = [int(value.shape[2]) for value in memory_k_by_item]
    if not lengths or max(lengths, default=0) == 0:
        return torch.zeros_like(q), 0.0, 0.0
    # Pad only the memory-token dimension; head and feature dimensions stay fixed.
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
    # [B,H,T,Dh] x [B,H,Dh,M] -> [B,H,T,M].
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
    """Run bucketed memory attention and restore ``[B,H,T,Dh]`` row order."""
    if len(memory_k_by_item) != q.shape[0] or len(memory_v_by_item) != q.shape[0]:
        raise ValueError("Memory list length must match query batch size.")
    lengths = [int(key.shape[2]) for key in memory_k_by_item]
    planner = MemoryBucketPlanner()
    buckets = planner.plan(lengths, bucket_count, strategy=bucket_strategy)
    output = torch.zeros_like(q)
    entropies = []
    maxima = []
    # Each bucket is an efficiency device only; attention remains row-local.
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
    nonempty_fraction = sum(length > 0 for length in lengths) / max(len(lengths), 1)
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
        # Separate memory attention normalizes entirely over memory for each
        # nonempty row; native-KV attention below measures a joint-softmax share.
        memory_attention_mass=nonempty_fraction,
        final_token_memory_attention_mass=nonempty_fraction,
    )
    return output, stats


def native_kv_attention(
    q: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    memory_k_by_item: Sequence[torch.Tensor],
    memory_v_by_item: Sequence[torch.Tensor],
    *,
    attention_mask: torch.Tensor | None = None,
    max_context_tokens: int | None = None,
    collect_final_token_memory_weights: bool = False,
) -> tuple[torch.Tensor, MemoryBatchingStats]:
    """Attend jointly over selected historical K/V and causal local K/V.

    Queries use ``[B,H_q,T,Dh]`` while local and memory K/V use native
    ``H_kv`` heads. ``H_q`` may equal ``H_kv`` (ordinary multi-head attention)
    or be an integer multiple (GQA/MQA). Memory positions precede local positions and are
    visible to every local query. Local positions retain the ordinary causal and
    padding masks. Crucially, one softmax normalizes both sources; no separate
    memory branch or output projection is introduced.
    """
    if q.ndim != 4 or local_k.ndim != 4 or local_v.shape != local_k.shape:
        raise ValueError("Local Q and K/V must be rank-four tensors with matching K/V shapes.")
    batch_size, query_heads, token_count, head_dim = q.shape
    if (
        local_k.shape[0] != batch_size
        or local_k.shape[2] != token_count
        or local_k.shape[3] != head_dim
    ):
        raise ValueError("Local K/V batch, token, and head dimensions must match Q.")
    kv_heads = int(local_k.shape[1])
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("Query heads must be an integer multiple of native K/V heads.")
    kv_groups = query_heads // kv_heads
    if len(memory_k_by_item) != batch_size or len(memory_v_by_item) != batch_size:
        raise ValueError("Memory list length must match query batch size.")
    if attention_mask is not None and attention_mask.shape != (batch_size, token_count):
        raise ValueError("attention_mask must match the local [B,T] shape.")

    scale = q.shape[-1] ** -0.5
    causal = torch.tril(
        torch.ones(token_count, token_count, dtype=torch.bool, device=q.device)
    )
    output_rows = []
    lengths = []
    entropies = []
    maxima = []
    memory_masses = []
    final_token_memory_masses = []
    final_token_memory_weights = []
    for row_index, (memory_k, memory_v) in enumerate(
        zip(memory_k_by_item, memory_v_by_item)
    ):
        if memory_k.ndim != 4 or memory_v.shape != memory_k.shape:
            raise ValueError("Each memory K/V pair must have matching [1,H,M,Dh] shapes.")
        if (
            memory_k.shape[0] != 1
            or memory_k.shape[1] != kv_heads
            or memory_k.shape[3] != q.shape[3]
        ):
            raise ValueError("Memory K/V head dimensions do not match the query.")
        memory_k = memory_k.to(q.device, q.dtype)
        memory_v = memory_v.to(q.device, q.dtype)
        memory_length = int(memory_k.shape[2])
        if (
            max_context_tokens is not None
            and memory_length + token_count > int(max_context_tokens)
        ):
            raise ValueError(
                "Native attention context exceeds model_max_context_tokens: "
                f"{memory_length} memory + {token_count} local > {max_context_tokens}."
            )
        lengths.append(memory_length)

        keys = torch.cat((memory_k, local_k[row_index : row_index + 1]), dim=2)
        values = torch.cat((memory_v, local_v[row_index : row_index + 1]), dim=2)
        if kv_groups > 1:
            keys = keys.repeat_interleave(kv_groups, dim=1)
            values = values.repeat_interleave(kv_groups, dim=1)
        scores = q[row_index : row_index + 1] @ keys.transpose(-2, -1) * scale
        memory_visible = torch.ones(
            token_count, memory_length, dtype=torch.bool, device=q.device
        )
        visible = torch.cat((memory_visible, causal), dim=1)
        if attention_mask is not None:
            local_valid = attention_mask[row_index].to(device=q.device, dtype=torch.bool)
            visible[:, memory_length:] &= local_valid[None, :]
        scores = scores.masked_fill(~visible[None, None, :, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output_rows.append(weights @ values)
        memory_weights = weights[..., :memory_length]
        memory_masses.append(float(memory_weights.sum(dim=-1).mean().detach().cpu()))
        final_token_memory_masses.append(
            float(memory_weights[:, :, -1, :].sum(dim=-1).mean().detach().cpu())
        )
        if collect_final_token_memory_weights:
            final_weights = memory_weights[0, :, -1, :].detach().float().cpu()
            final_token_memory_weights.append(
                tuple(tuple(float(value) for value in head) for head in final_weights)
            )
        safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
        entropies.append(
            float((-(weights * safe_weights.log()).sum(dim=-1).mean()).detach().cpu())
        )
        maxima.append(float(weights.max().detach().cpu()))

    valid = sum(lengths)
    stats = MemoryBatchingStats(
        selected_lengths=tuple(lengths),
        requested_bucket_count=0,
        actual_bucket_count=sum(length > 0 for length in lengths),
        bucket_membership=tuple((index,) for index, length in enumerate(lengths) if length),
        bucket_max_lengths=tuple(length for length in lengths if length),
        valid_positions=valid,
        allocated_positions=valid,
        padding_positions=0,
        padding_fraction=0.0,
        attention_entropy=sum(entropies) / max(len(entropies), 1),
        attention_max_weight=sum(maxima) / max(len(maxima), 1),
        memory_attention_mass=sum(memory_masses) / max(len(memory_masses), 1),
        final_token_memory_attention_mass=(
            sum(final_token_memory_masses) / max(len(final_token_memory_masses), 1)
        ),
        final_token_memory_weights=tuple(final_token_memory_weights),
    )
    return torch.cat(output_rows, dim=0), stats
