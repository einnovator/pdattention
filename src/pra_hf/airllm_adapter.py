"""PRA lifecycle integration for AirLLM's layer-streamed HF execution.

AirLLM owns model-weight movement. The shared PRA runtime continues to own
record identity, routing, authorization, storage policy, and selected native
K/V. This module only aligns already-selected, layer-local PRA detail with
AirLLM's module hooks and exposes an opt-in bridge to the existing HF PRA path.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from .deployment import PRAEngineCapabilities
from .engine_profiles import EngineType, PrefixCacheMode


_ATTENTION_PARAMETER = re.compile(r"^(model\.layers\.(\d+)\.self_attn\.)(.+)$")


def _map_airllm_parameter_name(name: str, pra_layers: frozenset[int]) -> str:
    """Map an upstream shard key to the path introduced by PRA injection."""

    match = _ATTENTION_PARAMETER.match(name)
    if match is None or int(match.group(2)) not in pra_layers:
        return name
    if match.group(3).startswith("original_attention."):
        return name
    return f"{match.group(1)}original_attention.{match.group(3)}"


def _install_parameter_name_bridge(airllm_model: Any, pra_layers: Sequence[int]) -> None:
    """Teach AirLLM's loader the one module-path change made by PRA.

    AirLLM shards retain ordinary HF names such as
    ``model.layers.7.self_attn.q_proj.weight``. PRA preserves that projection
    inside ``self_attn.original_attention``. Translating keys at placement time
    avoids rewriting or duplicating the checkpoint shards.
    """

    if getattr(airllm_model, "_pra_parameter_bridge_installed", False):
        return
    original_move = airllm_model.move_layer_to_device
    selected = frozenset(int(layer) for layer in pra_layers)

    def move_layer_to_device(state_dict: Mapping[str, Any]):
        mapped = {
            _map_airllm_parameter_name(name, selected): value
            for name, value in state_dict.items()
        }
        return original_move(mapped)

    airllm_model.move_layer_to_device = move_layer_to_device
    airllm_model._pra_parameter_bridge_installed = True
    airllm_model._pra_parameter_layers = tuple(sorted(selected))


class AirLLMResidencyMode(str, Enum):
    """Lifetime of selected per-layer PRA detail during streamed execution."""

    HOT = "hot"
    LAYER_STREAMED = "layer_streamed"
    HYBRID = "hybrid"


class AirLLMPrefetchMode(str, Enum):
    """How the next consumer layer's PRA detail is prepared."""

    NONE = "none"
    PRA_ONLY = "pra_only"
    INDEPENDENT_PARALLEL = "independent_parallel"
    COORDINATED = "coordinated"


@dataclass(frozen=True)
class AirLLMTransfer:
    """One layer-local PRA transfer returned by a physical detail store."""

    layer_id: int
    bytes_read: int = 0
    bytes_transferred: int = 0
    disk_seconds: float = 0.0
    h2d_seconds: float = 0.0
    cache_hit: bool = False
    payload: Any = None


@dataclass(frozen=True)
class AirLLMLayerEvent:
    """Measured PRA lifecycle event aligned with one AirLLM module hook."""

    event: str
    layer_id: int
    elapsed_seconds: float
    bytes_read: int = 0
    bytes_transferred: int = 0
    cache_hit: bool = False
    prefetched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AirLLMPRAStore(Protocol):
    """Physical interface for selected K/V; semantic policy stays upstream."""

    def load_layer(self, layer_id: int, device: object) -> AirLLMTransfer: ...

    def activate_layer(self, layer_id: int, payload: Any) -> None: ...

    def release_layer(self, layer_id: int) -> None: ...


class InMemoryAirLLMPRAStore:
    """Deterministic store used by tests and lifecycle microbenchmarks."""

    def __init__(self, layer_bytes: Mapping[int, int]) -> None:
        self.layer_bytes = {int(key): int(value) for key, value in layer_bytes.items()}
        self.loaded: set[int] = set()
        self.active: set[int] = set()
        self.loads: dict[int, int] = {}

    def load_layer(self, layer_id: int, device: object) -> AirLLMTransfer:
        started = time.perf_counter()
        cache_hit = layer_id in self.loaded
        self.loaded.add(layer_id)
        self.loads[layer_id] = self.loads.get(layer_id, 0) + int(not cache_hit)
        size = self.layer_bytes.get(layer_id, 0)
        return AirLLMTransfer(
            layer_id=layer_id,
            bytes_read=0 if cache_hit else size,
            bytes_transferred=0 if cache_hit else size,
            disk_seconds=time.perf_counter() - started,
            cache_hit=cache_hit,
            payload=layer_id,
        )

    def activate_layer(self, layer_id: int, payload: Any) -> None:
        self.active.add(layer_id)

    def release_layer(self, layer_id: int) -> None:
        self.active.discard(layer_id)
        self.loaded.discard(layer_id)


class AirLLMPRAAdapter:
    """Align selected PRA detail residency with AirLLM's layer hooks.

    The adapter is attached after AirLLM installs its weight hooks. PyTorch
    consequently loads a layer's weights first, then activates the selected
    PRA detail, executes attention, and finally runs the two release hooks.
    Coordinated prefetch reuses AirLLM's single executor, avoiding an extra
    Python I/O thread; independent mode owns a separate one-worker executor.
    """

    def __init__(
        self,
        store: AirLLMPRAStore | None = None,
        *,
        consumer_layers: Sequence[int] = (),
        residency_mode: AirLLMResidencyMode | str = AirLLMResidencyMode.LAYER_STREAMED,
        hot_layers: Sequence[int] = (),
        prefetch_mode: AirLLMPrefetchMode | str = AirLLMPrefetchMode.NONE,
        storage_managed: bool = False,
    ) -> None:
        self.store = store
        self.consumer_layers = tuple(sorted({int(layer) for layer in consumer_layers}))
        self.residency_mode = AirLLMResidencyMode(residency_mode)
        self.hot_layers = frozenset(int(layer) for layer in hot_layers)
        self.prefetch_mode = AirLLMPrefetchMode(prefetch_mode)
        self.storage_managed = bool(storage_managed)
        self.events: list[AirLLMLayerEvent] = []
        self._handles: list[Any] = []
        self._model: Any = None
        self._future: Future[AirLLMTransfer] | None = None
        self._future_layer: int | None = None
        self._executor: ThreadPoolExecutor | None = None

    @property
    def native_enabled(self) -> bool:
        return self.store is not None

    def capabilities(self) -> PRAEngineCapabilities:
        native = self.native_enabled
        level = "E2" if native and self.storage_managed else ("E1" if native else "E0")
        return PRAEngineCapabilities(
            adapter="airllm-hf" if native else "airllm-e0",
            engine_type=EngineType.AIRLLM,
            integration_level=level,
            prefix_cache_mode=PrefixCacheMode.SESSION_STATE,
            session_state=True,
            logical_refs=native,
            typed_records=native,
            text_fallback=True,
            native_kv=native,
            external_kv_residency=native,
            cpu_kv=native,
            pinned_kv=native,
            gpu_kv=native,
            streaming=True,
            selected_interval_materialization=native,
            request_lifetime=native,
            host_device_residency=native,
            tenant_isolation=native,
        )

    def bind(self, airllm_model: Any) -> "AirLLMPRAAdapter":
        """Register PRA hooks on eligible AirLLM decoder modules."""

        if self._handles:
            raise RuntimeError("AirLLM PRA adapter is already bound.")
        self._model = airllm_model
        layers = getattr(airllm_model, "layers", None)
        if layers is None:
            raise TypeError("AirLLM HF integration requires a model exposing .layers.")
        streamed = set(getattr(airllm_model, "_streamed_indices", range(len(layers))))
        missing = set(self.consumer_layers) - set(range(len(layers)))
        if missing:
            raise ValueError(f"Consumer layers are outside the AirLLM module list: {sorted(missing)}")
        for layer_id in self.consumer_layers:
            if layer_id not in streamed:
                continue
            module = layers[layer_id]
            module._pra_airllm_layer_id = layer_id
            self._handles.append(module.register_forward_pre_hook(self._pre_hook))
            self._handles.append(module.register_forward_hook(self._post_hook))
        if self.prefetch_mode in {
            AirLLMPrefetchMode.PRA_ONLY,
            AirLLMPrefetchMode.INDEPENDENT_PARALLEL,
        }:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self

    def _next_consumer(self, layer_id: int) -> int | None:
        return next((layer for layer in self.consumer_layers if layer > layer_id), None)

    def _submit_prefetch(self, layer_id: int) -> None:
        if self.store is None or self.prefetch_mode == AirLLMPrefetchMode.NONE:
            return
        if self._future is not None and not self._future.done():
            return
        executor = self._executor
        if self.prefetch_mode == AirLLMPrefetchMode.COORDINATED:
            executor = getattr(self._model, "_executor", None)
        if executor is None:
            return
        self._future = executor.submit(self.store.load_layer, layer_id, self._model.device)
        self._future_layer = layer_id

    def _pre_hook(self, module: Any, args: tuple[Any, ...]) -> None:
        if self.store is None:
            return
        layer_id = int(module._pra_airllm_layer_id)
        started = time.perf_counter()
        prefetched = self._future_layer == layer_id and self._future is not None
        if prefetched:
            transfer = self._future.result()
            self._future = None
            self._future_layer = None
        else:
            transfer = self.store.load_layer(layer_id, self._model.device)
        self.store.activate_layer(layer_id, transfer.payload)
        self.events.append(
            AirLLMLayerEvent(
                event="activate",
                layer_id=layer_id,
                elapsed_seconds=time.perf_counter() - started,
                bytes_read=transfer.bytes_read,
                bytes_transferred=transfer.bytes_transferred,
                cache_hit=transfer.cache_hit,
                prefetched=prefetched,
            )
        )
        next_layer = self._next_consumer(layer_id)
        if next_layer is not None:
            self._submit_prefetch(next_layer)

    def _keeps_hot(self, layer_id: int) -> bool:
        return self.residency_mode == AirLLMResidencyMode.HOT or (
            self.residency_mode == AirLLMResidencyMode.HYBRID
            and layer_id in self.hot_layers
        )

    def _post_hook(self, module: Any, args: tuple[Any, ...], output: Any) -> Any:
        if self.store is None:
            return output
        layer_id = int(module._pra_airllm_layer_id)
        if not self._keeps_hot(layer_id):
            started = time.perf_counter()
            self.store.release_layer(layer_id)
            self.events.append(
                AirLLMLayerEvent(
                    event="release",
                    layer_id=layer_id,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        return output

    def summary(self) -> dict[str, Any]:
        activations = [event for event in self.events if event.event == "activate"]
        return {
            "residency_mode": self.residency_mode.value,
            "prefetch_mode": self.prefetch_mode.value,
            "consumer_layers": list(self.consumer_layers),
            "events": [event.to_dict() for event in self.events],
            "pra_bytes_read": sum(event.bytes_read for event in activations),
            "pra_bytes_transferred": sum(event.bytes_transferred for event in activations),
            "prefetch_hits": sum(event.prefetched for event in activations),
            "cache_hits": sum(event.cache_hit for event in activations),
            "capabilities": self.capabilities().to_dict(),
        }

    def close(self) -> None:
        if self._future is not None:
            if not self._future.cancel():
                self._future.result()
            self._future = None
            self._future_layer = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.store is not None:
            for layer_id in self.consumer_layers:
                self.store.release_layer(layer_id)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def wrap_airllm_hf_model(
    airllm_model: Any,
    *,
    pra_config: Any = None,
    router: Any = None,
    memory_adapter: Any = None,
    pra_execution_policy: Any = None,
):
    """Reuse the existing HF PRA injection on AirLLM's Transformers model.

    AirLLM's macOS MLX implementation does not expose an HF ``.model`` and is
    intentionally rejected. It remains an E0 selected-text evaluation path.
    """

    from .config import PRAConfig
    from .model import PRAForCausalLM

    class AirLLMPRAForCausalLM(PRAForCausalLM):
        @property
        def device(self):
            owner = getattr(self, "_airllm_owner", None)
            if owner is None:
                return super().device
            import torch

            return torch.device(owner.running_device)

    model = getattr(airllm_model, "model", None)
    tokenizer = getattr(airllm_model, "tokenizer", None)
    if model is None or tokenizer is None:
        raise TypeError(
            "Native AirLLM PRA requires the HF-backed AirLLM path exposing model and tokenizer."
        )
    # AirLLM prefers SDPA for ordinary execution. PRA's correctness path uses
    # eager attention so it can compose the local and selected-memory score
    # domains explicitly. Transformers' Llama/Qwen modules read this dispatch
    # choice from config at call time, so no pretrained module is rebuilt.
    propagate = getattr(airllm_model, "_propagate_attn_implementation", None)
    if callable(propagate):
        propagate("eager")
    model.config._attn_implementation = "eager"
    wrapped = AirLLMPRAForCausalLM.from_model(
        model,
        tokenizer,
        pra_config=pra_config or PRAConfig(),
        router=router,
        memory_adapter=memory_adapter,
        pra_execution_policy=pra_execution_policy,
    )
    wrapped._airllm_owner = airllm_model
    wrapped._handle.set_execution_device(airllm_model.running_device)
    _install_parameter_name_bridge(airllm_model, wrapped._handle.adapters)
    return wrapped
