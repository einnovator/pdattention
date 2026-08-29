"""In-process MLX executor for query-selected native K/V memory.

The implementation uses the public mlx-lm model and cache protocols.  Selected
resources are encoded into post-RoPE per-layer K/V once.  A request-local cache
then presents ``selected memory + local sequential cache`` to each attention
layer, so both sources participate in one softmax without making selected text
part of the visible prompt.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_hf.engine_invariants import EnginePRAIsolationGuard
from pra_hf.engine_memory import LogicalPRABlock, LogicalPRABlockId, LogicalPRABlockStore
from pra_hf.engine_residency import EnginePRAResidencyManager, PRAEvictionPolicy


@dataclass(frozen=True)
class MLXNativeLayerKV:
    """One layer's post-position K/V arrays in ``[B,Hkv,T,D]`` layout."""

    keys: object
    values: object

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)


@dataclass(frozen=True)
class MLXNativeMemory:
    """Immutable selected-resource payload aligned to all model layers."""

    layers: tuple[MLXNativeLayerKV, ...]
    source_tokens: int

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    def selected_nbytes(self, layer_indices: Iterable[int] | None = None) -> int:
        """Return active bytes for a consumer-layer profile."""

        if layer_indices is None:
            return self.nbytes
        selected = set(map(int, layer_indices))
        return sum(layer.nbytes for index, layer in enumerate(self.layers) if index in selected)


@dataclass(frozen=True)
class MLXQuantizedLayerKV:
    """Symmetric int8 K/V plus per-head scales for one model layer."""

    keys: object
    values: object
    key_scale: object
    value_scale: object
    original_dtype: str

    @property
    def nbytes(self) -> int:
        return int(
            self.keys.nbytes
            + self.values.nbytes
            + self.key_scale.nbytes
            + self.value_scale.nbytes
        )


@dataclass(frozen=True)
class MLXQuantizedMemory:
    """Compact selected-K/V residency dequantized only for materialization."""

    layers: tuple[MLXQuantizedLayerKV, ...]
    source_tokens: int
    scheme: str = "symmetric_int8_per_head"

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


def quantize_native_memory(memory: MLXNativeMemory) -> MLXQuantizedMemory:
    """Quantize selected K/V to symmetric int8 with ``[B,H,1,1]`` scales."""

    import mlx.core as mx

    layers = []
    for layer in memory.layers:
        arrays = []
        for value in (layer.keys, layer.values):
            maximum = mx.max(mx.abs(value.astype(mx.float32)), axis=(2, 3), keepdims=True)
            scale = mx.maximum(maximum / 127.0, mx.array(1e-8, dtype=mx.float32))
            quantized = mx.clip(mx.round(value.astype(mx.float32) / scale), -127, 127)
            arrays.append((quantized.astype(mx.int8), scale.astype(mx.float16)))
        layers.append(
            MLXQuantizedLayerKV(
                keys=arrays[0][0],
                values=arrays[1][0],
                key_scale=arrays[0][1],
                value_scale=arrays[1][1],
                original_dtype=str(layer.keys.dtype),
            )
        )
    result = MLXQuantizedMemory(tuple(layers), memory.source_tokens)
    mx.eval(
        *(
            array
            for layer in result.layers
            for array in (layer.keys, layer.values, layer.key_scale, layer.value_scale)
        )
    )
    return result


def dequantize_native_memory(memory: MLXQuantizedMemory) -> MLXNativeMemory:
    """Restore model-native arrays for the current MLX attention protocol."""

    import mlx.core as mx

    layers = []
    for layer in memory.layers:
        dtype = getattr(mx, layer.original_dtype, mx.float16)
        layers.append(
            MLXNativeLayerKV(
                (layer.keys.astype(mx.float32) * layer.key_scale).astype(dtype),
                (layer.values.astype(mx.float32) * layer.value_scale).astype(dtype),
            )
        )
    result = MLXNativeMemory(tuple(layers), memory.source_tokens)
    mx.eval(*(array for layer in result.layers for array in (layer.keys, layer.values)))
    return result


@dataclass(frozen=True)
class MLXNativeFingerprint:
    """Compatibility fields required before persisted K/V may be reused."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    dtype: str
    position_policy: str
    consumer_profile: str
    resource_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "position_policy": self.position_policy,
            "consumer_profile": self.consumer_profile,
            "resource_version": self.resource_version,
        }


class MLXPositionedKVCache:
    """Local-only cache whose queries retain the source-position frame.

    Non-consumer layers do not receive selected memory, but their query and
    local K/V positions must still match ordinary split-cache execution.
    """

    def __init__(self, local_cache: object, position_base: int) -> None:
        self.local_cache = local_cache
        self.position_base = int(position_base)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def local_offset(self) -> int:
        return int(self.local_cache.offset)

    @property
    def state(self):
        return self.local_cache.state

    @property
    def nbytes(self) -> int:
        return int(getattr(self.local_cache, "nbytes", 0))

    def empty(self) -> bool:
        return bool(getattr(self.local_cache, "empty", lambda: self.local_offset == 0)())

    def is_trimmable(self) -> bool:
        operation = getattr(self.local_cache, "is_trimmable", None)
        return bool(operation and operation())

    def trim(self, n: int) -> int:
        return int(self.local_cache.trim(n))

    def update_and_fetch(self, keys, values):
        return self.local_cache.update_and_fetch(keys, values)

    def make_mask(self, n: int, return_array: bool = False, window_size=None, **kwargs):
        return self.local_cache.make_mask(
            n,
            return_array=return_array,
            window_size=window_size,
            **kwargs,
        )


class MLXSelectedKVCache:
    """mlx-lm cache wrapper combining selected memory with local sequential K/V.

    ``offset`` includes ``position_base`` for RoPE while causal mask construction
    uses only the local cache offset.  Every query may see all selected memory;
    local prompt/decode keys remain causal.
    """

    def __init__(self, local_cache: object, memory: MLXNativeLayerKV, position_base: int):
        if position_base < 0:
            raise ValueError("MLX native query position base cannot be negative.")
        self.local_cache = local_cache
        self.memory = memory
        self.position_base = int(position_base)

    @property
    def offset(self) -> int:
        return self.position_base + int(self.local_cache.offset)

    @property
    def local_offset(self) -> int:
        return int(self.local_cache.offset)

    @property
    def memory_tokens(self) -> int:
        return int(self.memory.keys.shape[2])

    @property
    def state(self):
        local = self.local_cache.state
        if not isinstance(local, tuple):
            local = tuple(local)
        return (self.memory.keys, self.memory.values, *local)

    @state.setter
    def state(self, value) -> None:
        raise RuntimeError("A PRA selected-memory cache cannot replace its immutable state.")

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

    def make_mask(
        self,
        n: int,
        return_array: bool = False,
        window_size: int | None = None,
        **_: object,
    ):
        import mlx.core as mx
        from mlx_lm.models.base import create_causal_mask

        if window_size is not None:
            local_window = min(window_size, self.local_offset + n)
            local = create_causal_mask(n, self.local_offset, window_size=local_window)
        else:
            local = create_causal_mask(n, self.local_offset)
        memory = mx.ones((n, self.memory_tokens), dtype=mx.bool_)
        return mx.concatenate((memory, local), axis=1)


def encode_native_memory(model: object, token_ids: Sequence[int]) -> MLXNativeMemory:
    """Encode one complete resource and retain post-RoPE K/V for every layer."""

    if not token_ids:
        raise ValueError("Cannot encode an empty PRA resource as native K/V.")
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
        layers.append(MLXNativeLayerKV(state[0], state[1]))
    return MLXNativeMemory(tuple(layers), source_tokens=len(token_ids))


def combine_native_memories(memories: Sequence[MLXNativeMemory]) -> MLXNativeMemory:
    """Concatenate immutable resource K/V without changing source-local positions."""

    if not memories:
        raise ValueError("At least one native PRA memory is required.")
    if len({len(memory.layers) for memory in memories}) != 1:
        raise ValueError("PRA memories have incompatible layer counts.")
    if len(memories) == 1:
        return memories[0]
    import mlx.core as mx

    layers = []
    for layer_idx in range(len(memories[0].layers)):
        layers.append(
            MLXNativeLayerKV(
                mx.concatenate(
                    tuple(memory.layers[layer_idx].keys for memory in memories), axis=2
                ),
                mx.concatenate(
                    tuple(memory.layers[layer_idx].values for memory in memories), axis=2
                ),
            )
        )
    return MLXNativeMemory(
        tuple(layers), source_tokens=max(memory.source_tokens for memory in memories)
    )


def make_native_prompt_cache(
    model: object,
    memory: MLXNativeMemory,
    *,
    max_kv_size: int | None = None,
    selected_layers: Iterable[int] | None = None,
):
    """Create request-local sequential caches backed by immutable selected K/V."""

    from mlx_lm.models.cache import make_prompt_cache

    local = make_prompt_cache(model, max_kv_size=max_kv_size)
    if len(local) != len(memory.layers):
        raise ValueError("MLX native memory does not match the model layer count.")
    selected = (
        set(range(len(memory.layers)))
        if selected_layers is None
        else set(map(int, selected_layers))
    )
    invalid = selected - set(range(len(memory.layers)))
    if invalid:
        raise ValueError(f"MLX consumer layers are out of range: {sorted(invalid)}")
    return [
        (
            MLXSelectedKVCache(cache, layer, memory.source_tokens)
            if index in selected
            else MLXPositionedKVCache(cache, memory.source_tokens)
        )
        for index, (cache, layer) in enumerate(zip(local, memory.layers))
    ]


def save_native_memory(
    path: str | Path,
    memory: MLXNativeMemory,
    fingerprint: MLXNativeFingerprint,
) -> tuple[Path, Path]:
    """Persist immutable selected K/V and a strict compatibility manifest."""

    import mlx.core as mx

    target = Path(path)
    arrays_path = target.with_suffix(".npz")
    manifest_path = target.with_suffix(".json")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, layer in enumerate(memory.layers):
        arrays[f"layer_{index:04d}_k"] = layer.keys
        arrays[f"layer_{index:04d}_v"] = layer.values
    mx.savez(str(arrays_path), **arrays)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_tokens": memory.source_tokens,
                "layer_count": len(memory.layers),
                "fingerprint": fingerprint.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return arrays_path, manifest_path


def load_native_memory(
    path: str | Path,
    *,
    expected_fingerprint: MLXNativeFingerprint,
) -> MLXNativeMemory:
    """Load persisted K/V only when every compatibility field agrees."""

    import mlx.core as mx

    target = Path(path)
    arrays_path = target.with_suffix(".npz")
    manifest = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != expected_fingerprint.to_dict():
        raise ValueError("Persisted MLX PRA cache fingerprint does not match runtime.")
    arrays = mx.load(str(arrays_path))
    layers = tuple(
        MLXNativeLayerKV(
            arrays[f"layer_{index:04d}_k"],
            arrays[f"layer_{index:04d}_v"],
        )
        for index in range(int(manifest["layer_count"]))
    )
    memory = MLXNativeMemory(layers, source_tokens=int(manifest["source_tokens"]))
    return memory


class MLXInProcessNativeExecutor:
    """Capability-honest E2 executor using an in-process mlx-lm model.

    Resources are immutable and shareable only when their complete logical key
    agrees.  Selection is supplied by ``pra_policy.selected_resource_ids``;
    absent an explicit list, resources are consumed in request order up to the
    typed budget.
    """

    integration_level = "E2"

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        model_id: str,
        model_revision: str,
        block_store: LogicalPRABlockStore,
        max_resident_bytes: int = 512 * 1024 * 1024,
        eviction_policy: PRAEvictionPolicy | str = PRAEvictionPolicy.LRU,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self.block_store = block_store
        self.residency = EnginePRAResidencyManager(
            block_store,
            max_resident_bytes=max_resident_bytes,
            policy=eviction_policy,
        )
        self._session_keys: dict[str, set[str]] = {}
        self.isolation = EnginePRAIsolationGuard()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        model_revision: str,
        block_store: LogicalPRABlockStore,
        **kwargs: object,
    ) -> "MLXInProcessNativeExecutor":
        from mlx_lm import load

        model, tokenizer = load(model_id, revision=model_revision)
        return cls(
            model,
            tokenizer,
            model_id=model_id,
            model_revision=model_revision,
            block_store=block_store,
            **kwargs,
        )

    def _selected_resources(self, request: PRAWireRequest) -> tuple[PRAWireResource, ...]:
        selected = tuple(
            map(str, request.pra_policy.get("selected_resource_ids", ()))
        )
        resources = {resource.resource_id: resource for resource in request.resources}
        if selected:
            missing = [resource_id for resource_id in selected if resource_id not in resources]
            if missing:
                raise KeyError(f"Selected PRA resources are absent from the request: {missing}")
            return tuple(resources[resource_id] for resource_id in selected)
        return request.resources[: request.budget.max_resources]

    def _resource_tokens(self, resource: PRAWireResource) -> tuple[int, ...]:
        if resource.text is None:
            raise ValueError(f"Resource {resource.resource_id!r} has no encodable text.")
        values = self.tokenizer.encode(resource.text, add_special_tokens=False)
        return tuple(map(int, values[:]))

    def _register(
        self, request: PRAWireRequest, resource: PRAWireResource, token_count: int
    ) -> str:
        version = str(
            resource.metadata.get("version")
            or hashlib.sha256((resource.text or "").encode("utf-8")).hexdigest()
        )
        shareable = bool(resource.metadata.get("shareable", False))
        identity = LogicalPRABlockId(
            tenant_id=request.tenant_id,
            session_id=None if shareable else request.session_id,
            resource_id=resource.resource_id,
            resource_version=version,
            record_type=resource.record_type,
            token_start=0,
            token_end=token_count,
            layer=0,
            model_revision=self.model_revision,
            dtype="mlx-model-native",
            layout="per-layer-bhld",
            materialization_profile=str(
                request.pra_policy.get("materialization_profile", "full_selected_resource")
            ),
            position_policy=str(request.pra_policy.get("position_policy", "source_local")),
            security_scope=resource.authorization_scope,
        )
        key = self.block_store.register(
            LogicalPRABlock(identity, address_bytes=32, detail_bytes=0)
        )
        if request.session_id:
            self._session_keys.setdefault(request.session_id, set()).add(key)
        return key

    def _resolve_memory(
        self, request: PRAWireRequest
    ) -> tuple[MLXNativeMemory, tuple[str, ...], int]:
        resources = self._selected_resources(request)
        if not resources:
            raise ValueError("Native PRA execution requires at least one selected resource.")
        tokenized = [(resource, self._resource_tokens(resource)) for resource in resources]
        total = sum(len(tokens) for _, tokens in tokenized)
        if total > request.budget.max_selected_tokens:
            raise ValueError("Selected PRA resources exceed max_selected_tokens.")
        keys = tuple(
            self._register(request, resource, len(tokens))
            for resource, tokens in tokenized
        )
        scopes = tuple(map(str, request.metadata.get("authorization_scopes", ())))
        self.block_store.select(keys, tenant_id=request.tenant_id, authorization_scopes=scopes)

        futures = []
        for key, (_, tokens) in zip(keys, tokenized):
            futures.append(
                self.residency.prefetch(
                    key,
                    lambda tokens=tokens: (
                        (memory := encode_native_memory(self.model, tokens)),
                        memory.nbytes,
                    ),
                )
            )
        memories = tuple(
            self.residency.resolve(
                key,
                lambda future=future: (
                    (memory := future.result()),
                    memory.nbytes,
                ),
                request_id=request.request_id,
            )
            for key, future in zip(keys, futures)
        )
        return combine_native_memories(memories), keys, total

    def _prompt(self, request: PRAWireRequest) -> str:
        return self.tokenizer.apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(request.engine_hints.get("enable_thinking", False)),
        )

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]:
        if block_store is not self.block_store:
            raise ValueError("MLX executor and adapter must share one logical block store.")
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        memory, keys, selected_tokens = self._resolve_memory(request)
        self.isolation.open_request(request.request_id, keys)
        try:
            self.isolation.attach_once(request.request_id, keys)
            detail_layers = request.pra_policy.get("detail_kv_layers")
            cache = make_native_prompt_cache(
                self.model,
                memory,
                selected_layers=None if detail_layers is None else tuple(detail_layers),
            )
            prompt = self._prompt(request)
            started = time.perf_counter()
            with self.residency.pin_request(request.request_id, keys):
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=request.resolved_max_new_tokens,
                    prompt_cache=cache,
                    sampler=make_sampler(
                        temp=float(request.engine_hints.get("temperature", 0))
                    ),
                ):
                    yield {
                        "text": response.text,
                        "finish_reason": getattr(response, "finish_reason", None),
                        "prompt_tokens": int(response.prompt_tokens),
                        "generation_tokens": int(response.generation_tokens),
                        "selected_native_tokens": selected_tokens,
                        "active_native_kv_bytes": memory.selected_nbytes(detail_layers),
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        "native_kv_used": True,
                    }
        finally:
            self.isolation.close_request(request.request_id, require_attached=False)

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult:
        rows = tuple(self.stream(request, block_store))
        text = "".join(str(row.get("text", "")) for row in rows)
        final = dict(rows[-1]) if rows else {}
        final["residency"] = self.residency.metrics().to_dict()
        return PRAEngineResult(text=text, raw=final, trace=self.residency.events())

    def close_session(self, session_id: str) -> None:
        keys = tuple(self._session_keys.pop(session_id, ()))
        self.residency.invalidate(keys)
        for key in keys:
            identity = self.block_store.get(key).identity
            self.block_store.invalidate_resource(
                identity.tenant_id,
                identity.resource_id,
                resource_version=identity.resource_version,
            )

    def close(self) -> None:
        self.isolation.close()
        self.residency.close()
