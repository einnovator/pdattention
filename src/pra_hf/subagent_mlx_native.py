"""Live MLX native-K/V port for Paper 9 subagent reuse experiments.

This is a deliberately narrow engine adapter. It encodes an immutable tool
result once, retains the host model's post-RoPE K/V arrays, and gives each
child a fresh local cache that attends to the shared arrays by reference.
Persistence and tiering remain the Paper 4.5 runtime's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context_records import ContextRecord
from .subagent_context import NativeKVHandle


@dataclass(frozen=True)
class MLXLayerKV:
    """One immutable layer in host ``[batch, kv_head, token, dim]`` layout."""

    keys: object
    values: object

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)


@dataclass(frozen=True)
class MLXSubagentMemory:
    """Layer-aligned native state for one reusable typed record."""

    layers: tuple[MLXLayerKV, ...]
    token_count: int

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


class MLXSubagentKVCache:
    """Expose immutable shared K/V before a request-local MLX prompt cache."""

    def __init__(self, local_cache: object, memory: MLXLayerKV, position_base: int):
        self.local_cache = local_cache
        self.memory = memory
        self.position_base = int(position_base)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def state(self):
        local = self.local_cache.state
        if not isinstance(local, tuple):
            local = tuple(local)
        return (self.memory.keys, self.memory.values, *local)

    @state.setter
    def state(self, value) -> None:
        raise RuntimeError("Shared subagent memory is immutable.")

    @property
    def nbytes(self) -> int:
        return self.memory.nbytes + int(getattr(self.local_cache, "nbytes", 0))

    def empty(self) -> bool:
        return False

    def is_trimmable(self) -> bool:
        operation = getattr(self.local_cache, "is_trimmable", None)
        return bool(operation and operation())

    def trim(self, n: int) -> int:
        return int(self.local_cache.trim(n))

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        local_keys, local_values = self.local_cache.update_and_fetch(keys, values)
        return (
            mx.concatenate((self.memory.keys, local_keys), axis=2),
            mx.concatenate((self.memory.values, local_values), axis=2),
        )

    def make_mask(self, n: int, return_array: bool = False, window_size=None, **kwargs):
        import mlx.core as mx
        from mlx_lm.models.base import create_causal_mask

        local_offset = int(self.local_cache.offset)
        if window_size is None:
            local = create_causal_mask(n, local_offset)
        else:
            local = create_causal_mask(
                n, local_offset, window_size=min(int(window_size), local_offset + n)
            )
        memory = mx.ones((n, int(self.memory.keys.shape[2])), dtype=mx.bool_)
        return mx.concatenate((memory, local), axis=1)


class MLXSubagentNativePort:
    """Concrete ``NativeKVReusePort`` backed by public ``mlx-lm`` caches."""

    encoding_revision = "paper9-mlx-post-rope-v1"
    position_contract = "contiguous_source_local"

    def __init__(self, model: object, tokenizer: object, *, model_id: str, revision: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.memories: dict[str, MLXSubagentMemory] = {}
        self.target_records: dict[str, str] = {}

    def encode(self, record: ContextRecord) -> NativeKVHandle:
        """Encode a tool result with the unchanged host model exactly once."""

        payload = record.payload
        if isinstance(payload, dict):
            value = payload.get("output", payload)
        else:
            value = payload
        token_ids = tuple(self.tokenizer.encode(str(value), add_special_tokens=False))
        if not token_ids:
            raise ValueError("Cannot encode an empty reusable record.")
        memory = encode_mlx_subagent_memory(self.model, token_ids)
        self.memories[record.record_id] = memory
        return NativeKVHandle(
            record_uuid=record.record_id,
            model_id=self.model_id,
            model_revision=self.revision,
            encoding_revision=self.encoding_revision,
            position_contract=self.position_contract,
            token_count=memory.token_count,
            byte_count=memory.nbytes,
        )

    def materialize(self, handle: NativeKVHandle, *, target_agent_uuid: str) -> bool:
        """Bind a compatible immutable record to one child request."""

        if handle.record_uuid not in self.memories:
            return False
        self.target_records[target_agent_uuid] = handle.record_uuid
        return True

    def requested_handle(self) -> NativeKVHandle:
        """Return the compatibility template supplied during runtime lookup."""

        return NativeKVHandle(
            record_uuid="requested",
            model_id=self.model_id,
            model_revision=self.revision,
            encoding_revision=self.encoding_revision,
            position_contract=self.position_contract,
            token_count=0,
            byte_count=0,
        )

    def prompt_cache(self, target_agent_uuid: str):
        """Create fresh local caches around the target's selected shared state."""

        try:
            memory = self.memories[self.target_records[target_agent_uuid]]
        except KeyError as error:
            raise KeyError(f"No native memory is attached to {target_agent_uuid!r}.") from error
        return make_mlx_subagent_cache(self.model, memory)


def encode_mlx_subagent_memory(model: object, token_ids: tuple[int, ...]) -> MLXSubagentMemory:
    """Capture post-RoPE K/V produced by a normal contiguous host prefill."""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    caches = make_prompt_cache(model)
    model(mx.array(token_ids, dtype=mx.int32)[None], cache=caches)
    states = [cache.state for cache in caches]
    mx.eval(states)
    layers = []
    for state in states:
        if not isinstance(state, tuple) or len(state) < 2:
            raise RuntimeError("The MLX model exposed a non-attention cache layer.")
        layers.append(MLXLayerKV(state[0], state[1]))
    return MLXSubagentMemory(tuple(layers), len(token_ids))


def make_mlx_subagent_cache(model: object, memory: MLXSubagentMemory):
    """Wrap one immutable memory with fresh request-local cache objects."""

    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model)
    if len(local) != len(memory.layers):
        raise ValueError("Native memory does not match the model layer count.")
    return [
        MLXSubagentKVCache(cache, layer, memory.token_count)
        for cache, layer in zip(local, memory.layers)
    ]
