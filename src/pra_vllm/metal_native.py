"""Selected PRA block storage on vLLM-Metal's native paged K/V cache."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VLLMMetalBlockHandle:
    """Physical vLLM-Metal pages for one immutable selected K/V payload."""

    logical_key: str
    block_ids: tuple[int, ...]
    token_count: int
    byte_count: int


class VLLMMetalPRAStore:
    """Allocate selected K/V in ordinary vLLM-Metal physical pages.

    Query arrays are already projected and positioned.  The paged-attention
    primitive performs one normalization over all pages in each request's block
    table, exactly as it does for sequential vLLM K/V.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int = 16,
        dtype=None,
    ) -> None:
        import mlx.core as mx
        from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache

        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype or mx.float16
        self.cache = MetalPagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=self.dtype,
        )
        self._free = list(range(num_blocks))
        self._handles: dict[str, VLLMMetalBlockHandle] = {}

    def materialize(self, logical_key: str, layers: Sequence[object]) -> VLLMMetalBlockHandle:
        """Pack ``[1,Hkv,T,D]`` arrays into scheduler-compatible pages."""

        import mlx.core as mx

        existing = self._handles.get(logical_key)
        if existing is not None:
            return existing
        if len(layers) != self.num_layers:
            raise ValueError("Selected PRA K/V does not match vLLM layer count.")
        token_count = int(layers[0].keys.shape[2])
        if any(int(layer.keys.shape[2]) != token_count for layer in layers):
            raise ValueError("All PRA K/V layers must have one token geometry.")
        count = math.ceil(token_count / self.block_size)
        if count > len(self._free):
            raise MemoryError("vLLM-Metal has insufficient free PRA pages.")
        block_ids = tuple(self._free[:count])
        del self._free[:count]
        padded_tokens = count * self.block_size
        for layer_idx, layer in enumerate(layers):
            keys = layer.keys[0].transpose(1, 0, 2)
            values = layer.values[0].transpose(1, 0, 2)
            if padded_tokens != token_count:
                pad_shape = (
                    padded_tokens - token_count,
                    self.num_kv_heads,
                    self.head_dim,
                )
                keys = mx.concatenate((keys, mx.zeros(pad_shape, dtype=keys.dtype)))
                values = mx.concatenate((values, mx.zeros(pad_shape, dtype=values.dtype)))
            self.cache.key_caches[layer_idx][list(block_ids)] = keys.reshape(
                count, self.block_size, self.num_kv_heads, self.head_dim
            )
            self.cache.value_caches[layer_idx][list(block_ids)] = values.reshape(
                count, self.block_size, self.num_kv_heads, self.head_dim
            )
        mx.eval(*self.cache.key_caches, *self.cache.value_caches)
        byte_count = sum(int(layer.keys.nbytes + layer.values.nbytes) for layer in layers)
        handle = VLLMMetalBlockHandle(logical_key, block_ids, token_count, byte_count)
        self._handles[logical_key] = handle
        return handle

    def release(self, logical_key: str) -> None:
        handle = self._handles.pop(logical_key, None)
        if handle is not None:
            self._free.extend(handle.block_ids)
            self._free.sort()

    def attend(
        self,
        layer_idx: int,
        queries,
        handles: Sequence[VLLMMetalBlockHandle],
        *,
        scale: float,
    ):
        """Run one-token-per-request native paged attention over selected blocks."""

        import mlx.core as mx
        from vllm_metal.metal import get_ops

        if not handles:
            raise ValueError("Paged PRA attention requires selected handles.")
        width = max(len(handle.block_ids) for handle in handles)
        block_tables = mx.full((len(handles), width), -1, dtype=mx.int32)
        for row, handle in enumerate(handles):
            block_tables[row, : len(handle.block_ids)] = mx.array(
                handle.block_ids, dtype=mx.int32
            )
        lengths = mx.array([handle.token_count for handle in handles], dtype=mx.int32)
        cu_query = mx.arange(len(handles) + 1, dtype=mx.int32)
        output = mx.array(0)
        get_ops().paged_attention_primitive(
            queries,
            self.cache.key_caches[layer_idx],
            self.cache.value_caches[layer_idx],
            self.num_kv_heads,
            float(scale),
            0.0,
            block_tables,
            lengths,
            cu_query,
            self.block_size,
            max(handle.token_count for handle in handles),
            -1,
            output,
        )
        return output

    @property
    def resident_bytes(self) -> int:
        return sum(handle.byte_count for handle in self._handles.values())

    @property
    def physical_capacity_bytes(self) -> int:
        return sum(array.nbytes for array in self.cache.key_caches) + sum(
            array.nbytes for array in self.cache.value_caches
        )
