"""Fused interval-addressed CUDA attention for HF live-history K/V.

The kernel reads selected rows from the canonical source cache through compact
interval descriptors.  It never packs, casts, or copies selected K/V.  A small
request-local tail is passed separately because it is owned by the current
request rather than by the reusable source.
"""

from __future__ import annotations

from typing import Sequence


def triton_disjoint_selected_attention(
    query,
    source_keys,
    source_values,
    source_intervals: Sequence[tuple[int, int, int]],
    local_keys,
    local_values,
    query_positions,
    *,
    scale: float,
):
    """Run one online softmax over canonical source intervals and local K/V."""

    import torch
    import triton
    import triton.language as tl

    if query.ndim != 4 or int(query.shape[0]) != 1:
        raise ValueError("Fused HF disjoint attention requires rank-four batch-one Q.")
    arrays = (source_keys, source_values, local_keys, local_values)
    if any(array.ndim != 4 or int(array.shape[0]) != 1 for array in arrays):
        raise ValueError("Fused HF disjoint attention requires rank-four batch-one K/V.")
    if any(array.device != query.device for array in arrays):
        raise ValueError("Fused HF disjoint attention requires one CUDA device.")
    if any(array.dtype != query.dtype for array in arrays):
        raise ValueError("Fused HF disjoint attention requires one model-native dtype.")
    if not query.is_cuda:
        raise ValueError("Fused HF disjoint attention requires CUDA tensors.")
    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    del batch
    kv_heads = int(source_keys.shape[1])
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by K/V heads.")
    if int(source_values.shape[1]) != kv_heads:
        raise ValueError("Canonical source K/V head counts differ.")
    if int(local_keys.shape[1]) != kv_heads or local_values.shape != local_keys.shape:
        raise ValueError("Local K/V geometry differs from the canonical source.")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("Fused HF disjoint attention supports head dimensions 1..256.")
    intervals = tuple(
        (int(storage_start), int(storage_end), int(position_start))
        for storage_start, storage_end, position_start in source_intervals
    )
    if not intervals or any(
        storage_start < 0 or storage_end <= storage_start or position_start < 0
        for storage_start, storage_end, position_start in intervals
    ):
        raise ValueError("Fused HF disjoint attention requires valid source intervals.")
    if any(storage_end > int(source_keys.shape[2]) for _, storage_end, _ in intervals):
        raise ValueError("A fused HF source interval exceeds canonical K/V storage.")
    if query_positions.ndim == 2:
        query_positions = query_positions[0]
    if query_positions.ndim != 1 or int(query_positions.shape[0]) != query_tokens:
        raise ValueError("Fused HF disjoint attention requires one position per query.")

    interval_table = torch.tensor(
        intervals, dtype=torch.int32, device=query.device
    )
    output = torch.empty_like(query)
    block_dim = triton.next_power_of_2(head_dim)
    local_position_start = int(query_positions[0].item()) - (
        int(local_keys.shape[2]) - query_tokens
    )

    @triton.jit
    def _kernel(
        q_ptr,
        source_k_ptr,
        source_v_ptr,
        local_k_ptr,
        local_v_ptr,
        interval_ptr,
        query_position_ptr,
        out_ptr,
        q_stride_h: tl.constexpr,
        q_stride_t: tl.constexpr,
        q_stride_d: tl.constexpr,
        source_k_stride_h: tl.constexpr,
        source_k_stride_t: tl.constexpr,
        source_k_stride_d: tl.constexpr,
        source_v_stride_h: tl.constexpr,
        source_v_stride_t: tl.constexpr,
        source_v_stride_d: tl.constexpr,
        local_k_stride_h: tl.constexpr,
        local_k_stride_t: tl.constexpr,
        local_k_stride_d: tl.constexpr,
        local_v_stride_h: tl.constexpr,
        local_v_stride_t: tl.constexpr,
        local_v_stride_d: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_d: tl.constexpr,
        query_length: tl.constexpr,
        local_length: tl.constexpr,
        local_position_start: tl.constexpr,
        groups: tl.constexpr,
        head_dimension: tl.constexpr,
        interval_count: tl.constexpr,
        attention_scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        query_head = row // query_length
        query_index = row - query_head * query_length
        kv_head = query_head // groups
        offsets = tl.arange(0, BLOCK_D)
        valid_d = offsets < head_dimension
        q = tl.load(
            q_ptr
            + query_head * q_stride_h
            + query_index * q_stride_t
            + offsets * q_stride_d,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        query_position = tl.load(query_position_ptr + query_index)
        running_max = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((BLOCK_D,), tl.float32)

        for interval_index in range(0, interval_count):
            storage_start = tl.load(interval_ptr + interval_index * 3)
            storage_end = tl.load(interval_ptr + interval_index * 3 + 1)
            position_start = tl.load(interval_ptr + interval_index * 3 + 2)
            storage_index = storage_start
            while storage_index < storage_end:
                position = position_start + storage_index - storage_start
                k = tl.load(
                    source_k_ptr
                    + kv_head * source_k_stride_h
                    + storage_index * source_k_stride_t
                    + offsets * source_k_stride_d,
                    mask=valid_d,
                    other=0.0,
                ).to(tl.float32)
                score = tl.sum(q * k, axis=0) * attention_scale
                score = tl.where(position <= query_position, score, -float("inf"))
                next_max = tl.maximum(running_max, score)
                prior_scale = tl.exp(running_max - next_max)
                weight = tl.exp(score - next_max)
                v = tl.load(
                    source_v_ptr
                    + kv_head * source_v_stride_h
                    + storage_index * source_v_stride_t
                    + offsets * source_v_stride_d,
                    mask=valid_d,
                    other=0.0,
                ).to(tl.float32)
                accumulator = accumulator * prior_scale + v * weight
                denominator = denominator * prior_scale + weight
                running_max = next_max
                storage_index += 1

        for local_index in range(0, local_length):
            position = local_position_start + local_index
            k = tl.load(
                local_k_ptr
                + kv_head * local_k_stride_h
                + local_index * local_k_stride_t
                + offsets * local_k_stride_d,
                mask=valid_d,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k, axis=0) * attention_scale
            score = tl.where(position <= query_position, score, -float("inf"))
            next_max = tl.maximum(running_max, score)
            prior_scale = tl.exp(running_max - next_max)
            weight = tl.exp(score - next_max)
            v = tl.load(
                local_v_ptr
                + kv_head * local_v_stride_h
                + local_index * local_v_stride_t
                + offsets * local_v_stride_d,
                mask=valid_d,
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * prior_scale + v * weight
            denominator = denominator * prior_scale + weight
            running_max = next_max

        tl.store(
            out_ptr
            + query_head * out_stride_h
            + query_index * out_stride_t
            + offsets * out_stride_d,
            accumulator / denominator,
            mask=valid_d,
        )

    _kernel[(query_heads * query_tokens,)](
        query,
        source_keys,
        source_values,
        local_keys,
        local_values,
        interval_table,
        query_positions,
        output,
        query.stride(1),
        query.stride(2),
        query.stride(3),
        source_keys.stride(1),
        source_keys.stride(2),
        source_keys.stride(3),
        source_values.stride(1),
        source_values.stride(2),
        source_values.stride(3),
        local_keys.stride(1),
        local_keys.stride(2),
        local_keys.stride(3),
        local_values.stride(1),
        local_values.stride(2),
        local_values.stride(3),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        query_length=query_tokens,
        local_length=int(local_keys.shape[2]),
        local_position_start=local_position_start,
        groups=query_heads // kv_heads,
        head_dimension=head_dim,
        interval_count=len(intervals),
        attention_scale=float(scale),
        BLOCK_D=block_dim,
        num_warps=max(1, min(8, block_dim // 32)),
    )
    return output, interval_table
