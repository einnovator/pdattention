"""Semantic-keyed K/V transfer candidate for vLLM V1 on CUDA.

The connector loads selected source K/V into request-owned paged-cache slots.
The source content is not recomputed or represented by its token IDs during a
load request. vLLM still accounts for placeholder prefix slots, however, so
this mechanism is a prefix-shaped native-transfer candidate rather than the
scheduler-invisible detached-page design used by the Metal bridge.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata

from pra_vllm.cuda_protocol import CudaConnectorCommand
from pra_vllm.cuda_detached import (
    active_detached_rows,
    detached_layout,
    install_detached_runner_hooks,
    prepend_detached_blocks,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _layer_file_name(layer_name: str) -> str:
    digest = hashlib.sha256(layer_name.encode("utf-8")).hexdigest()[:24]
    return f"layer-{digest}.safetensors"


@dataclass
class _RequestTransfer:
    request_id: str
    logical_key: str
    source_tokens: int
    slot_mapping: torch.Tensor
    mode: str
    residency: str
    detached: bool = False

    @classmethod
    def create(
        cls,
        command: CudaConnectorCommand,
        block_ids: list[int],
        block_size: int,
        request_id: str,
        detached: bool = False,
    ) -> "_RequestTransfer":
        if command.source_tokens % block_size:
            raise ValueError("PRA CUDA source K/V must be block aligned.")
        required_blocks = command.source_tokens // block_size
        if not detached and len(block_ids) < required_blocks:
            raise RuntimeError("vLLM allocated too few blocks for PRA source K/V.")
        if detached:
            slots = torch.empty(0, dtype=torch.int64)
        else:
            blocks = torch.tensor(block_ids[:required_blocks], dtype=torch.int64)
            offsets = torch.arange(block_size, dtype=torch.int64)
            slots = (blocks[:, None] * block_size + offsets[None, :]).flatten()
        return cls(
            request_id=request_id,
            logical_key=command.logical_key,
            source_tokens=command.source_tokens,
            slot_mapping=slots,
            mode=command.mode,
            residency=command.residency,
            detached=detached,
        )


@dataclass
class PRASemanticConnectorMetadata(KVConnectorMetadata):
    """Worker transfer plan for one scheduler step."""

    requests: list[_RequestTransfer] = field(default_factory=list)


class PRASemanticConnector(KVConnectorBase_V1):
    """Store and load source K/V by PRA logical resource identity."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size = int(vllm_config.cache_config.block_size)
        self._detached = bool(
            self._kv_transfer_config.get_from_extra_config(
                "detached_pages", False
            )
        )
        self._detached_reserve_blocks = int(
            self._kv_transfer_config.get_from_extra_config(
                "detached_reserve_blocks", 64
            )
        )
        if self._detached:
            install_detached_runner_hooks(self._detached_reserve_blocks)
        configured = self._kv_transfer_config.get_from_extra_config(
            "storage_path", "/tmp/pra-vllm-cuda"
        )
        self._storage = Path(str(configured)).expanduser().resolve()
        self._storage.mkdir(parents=True, exist_ok=True)
        telemetry = self._kv_transfer_config.get_from_extra_config(
            "telemetry_path", None
        )
        self._telemetry = (
            None if telemetry is None else Path(str(telemetry)).expanduser().resolve()
        )
        if self._telemetry is not None:
            self._telemetry.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry_lock = threading.Lock()
        self._hot_layers: dict[tuple[str, str], torch.Tensor] = {}
        self._commands: dict[str, CudaConnectorCommand] = {}
        self._loads: dict[str, "Request"] = {}
        self._active_store_keys: set[str] = set()
        self._stored_tensor_bytes: dict[str, int] = {}
        self._detached_free: list[int] | None = None
        self._detached_handles: dict[tuple[str, str], tuple[int, ...]] = {}
        self._detached_materialized: set[tuple[str, str]] = set()
        self._detached_tensor_bytes: dict[tuple[str, str], int] = {}
        self._detached_active_requests: dict[str, tuple[str, str]] = {}
        self._detached_refcounts: dict[tuple[str, str], int] = {}

    def _write_telemetry(self, event: dict[str, Any]) -> None:
        """Append one transfer event without sharing mutable benchmark state."""

        if self._telemetry is None:
            return
        encoded = json.dumps(event, sort_keys=True) + "\n"
        with self._telemetry_lock:
            with self._telemetry.open("a", encoding="utf-8") as stream:
                stream.write(encoded)

    def _resident_hot_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._hot_layers.values())

    def _directory(self, logical_key: str) -> Path:
        digest = hashlib.sha256(logical_key.encode("utf-8")).hexdigest()
        return self._storage / digest

    def _ready(self, command: CudaConnectorCommand) -> bool:
        manifest = self._directory(command.logical_key) / "manifest.json"
        if not manifest.exists():
            return False
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return (
            payload.get("logical_key") == command.logical_key
            and int(payload.get("source_tokens", -1)) == command.source_tokens
        )

    def on_new_request(self, request: "Request") -> None:
        command = CudaConnectorCommand.parse(request.cache_salt)
        if command is not None:
            self._commands[request.request_id] = command

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        command = self._commands.get(request.request_id)
        if command is None:
            command = CudaConnectorCommand.parse(request.cache_salt)
            if command is not None:
                self._commands[request.request_id] = command
        if command is None or command.mode != "load" or not self._ready(command):
            return 0, False
        if self._detached:
            return 0, False
        prompt_tokens = request.prompt_token_ids or []
        if command.source_tokens >= len(prompt_tokens):
            raise ValueError("PRA CUDA load request must include a query suffix.")
        matched = max(0, command.source_tokens - int(num_computed_tokens))
        return matched, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens > 0 and not self._detached:
            self._loads[request.request_id] = request

    def _add_new_request(
        self,
        metadata: PRASemanticConnectorMetadata,
        req_id: str,
        block_ids: list[int],
    ) -> None:
        command = self._commands.get(req_id)
        if command is None:
            return
        if (
            command.mode == "load"
            and not self._detached
            and req_id not in self._loads
        ):
            return
        metadata.requests.append(
            _RequestTransfer.create(
                command,
                block_ids,
                self._block_size,
                request_id=req_id,
                detached=self._detached and command.mode == "load",
            )
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        metadata = PRASemanticConnectorMetadata()
        for request in scheduler_output.scheduled_new_reqs:
            self._add_new_request(metadata, request.req_id, request.block_ids[0])

        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            command = self._commands.get(req_id)
            if self._detached and command is not None and command.mode == "load":
                self._add_new_request(metadata, req_id, [])
                continue
            if req_id not in self._loads or req_id not in cached.resumed_req_ids:
                continue
            block_groups = cached.new_block_ids[index]
            if block_groups is None:
                raise RuntimeError("Resumed PRA CUDA load has no replacement blocks.")
            self._add_new_request(metadata, req_id, block_groups[0])

        for request in metadata.requests:
            if request.mode == "load":
                self._loads.pop(request.request_id, None)
        return metadata

    def _detached_slots(self, request: _RequestTransfer) -> tuple[tuple[int, ...], torch.Tensor]:
        """Allocate scheduler-invisible blocks for one immutable resource."""

        handle_key = (request.logical_key, request.residency)
        blocks = self._detached_handles.get(handle_key)
        if blocks is None:
            if self._detached_free is None:
                runner = getattr(self, "_model_runner", None)
                layout = detached_layout(runner) if runner is not None else None
                if layout is None:
                    layout = detached_layout()
                if layout is None:
                    raise RuntimeError("Detached CUDA worker reserve is unavailable.")
                self._detached_free = list(layout.block_ids)
            required = request.source_tokens // self._block_size
            if required > len(self._detached_free):
                raise MemoryError("Detached CUDA reserve has insufficient free blocks.")
            blocks = tuple(self._detached_free[:required])
            del self._detached_free[:required]
            self._detached_handles[handle_key] = blocks
        block_tensor = torch.tensor(blocks, dtype=torch.int64)
        offsets = torch.arange(self._block_size, dtype=torch.int64)
        slots = (block_tensor[:, None] * self._block_size + offsets[None, :]).flatten()
        return blocks, slots

    def _start_detached_load(
        self,
        forward_context: "ForwardContext",
        requests: list[_RequestTransfer],
    ) -> None:
        """Materialize each resource once and attach it to every active row."""

        attention = forward_context.attn_metadata
        active_rows = active_detached_rows()
        self._reap_inactive_detached_requests(set(active_rows))
        request_rows = {
            request_id: row for row, request_id in enumerate(active_rows)
        }
        grouped: dict[tuple[str, str], list[_RequestTransfer]] = {}
        for request in requests:
            grouped.setdefault((request.logical_key, request.residency), []).append(request)
        for handle_key, group in grouped.items():
            exemplar = group[0]
            blocks, slots = self._detached_slots(exemplar)
            newly_materialized = handle_key not in self._detached_materialized
            started = time.perf_counter()
            storage_read_ms = 0.0
            storage_read_bytes = 0
            h2d_bytes = 0
            d2d_bytes = 0
            native_tensor_bytes = 0
            if newly_materialized:
                directory = self._directory(exemplar.logical_key)
                for layer_name, layer in forward_context.no_compile_layers.items():
                    cache = getattr(layer, "kv_cache", None)
                    if cache is None:
                        continue
                    layer_attention = (
                        attention[layer_name]
                        if isinstance(attention, dict)
                        else attention
                    )
                    path = directory / _layer_file_name(layer_name)
                    read_started = time.perf_counter()
                    tensor = safetensors.torch.load_file(str(path))["kv_cache"]
                    storage_read_ms += (time.perf_counter() - read_started) * 1000.0
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    native_tensor_bytes += tensor_bytes
                    storage_read_bytes += tensor_bytes
                    if tensor.device == cache.device:
                        d2d_bytes += tensor_bytes
                    else:
                        h2d_bytes += tensor_bytes
                    self._inject(
                        cache, tensor, slots, layer_attention, self._block_size
                    )
                self._detached_materialized.add(handle_key)
                self._detached_tensor_bytes[handle_key] = native_tensor_bytes
            for group_index, request in enumerate(group):
                transfer_owner = newly_materialized and group_index == 0
                if request.request_id not in request_rows:
                    raise RuntimeError(
                        f"Detached CUDA request row is unavailable: {request.request_id}"
                    )
                if request.request_id not in self._detached_active_requests:
                    self._detached_active_requests[request.request_id] = handle_key
                    self._detached_refcounts[handle_key] = (
                        self._detached_refcounts.get(handle_key, 0) + 1
                    )
                prepend_detached_blocks(
                    attention,
                    row=request_rows[request.request_id],
                    block_ids=blocks,
                    source_tokens=request.source_tokens,
                    block_size=self._block_size,
                )
                self._write_telemetry(
                    {
                        "schema_version": "pra-vllm-cuda-transfer-v2",
                        "event": "load",
                        "attachment": "scheduler-invisible",
                        "request_id": request.request_id,
                        "logical_key": request.logical_key,
                        "residency": request.residency,
                        "source_tokens": request.source_tokens,
                        "scheduler_source_tokens": 0,
                        "pra_shared_blocks": len(blocks),
                        "storage_read_bytes": (
                            storage_read_bytes if transfer_owner else 0
                        ),
                        "storage_read_ms": (
                            storage_read_ms if transfer_owner else 0.0
                        ),
                        "h2d_bytes": h2d_bytes if transfer_owner else 0,
                        "d2d_bytes": d2d_bytes if transfer_owner else 0,
                        "shared_resident_hit": not transfer_owner,
                        "resident_detached_bytes": self._detached_tensor_bytes.get(
                            handle_key, 0
                        ),
                        "load_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )

    def _reap_inactive_detached_requests(self, active_request_ids: set[str]) -> None:
        """Release WARM pages after their last request leaves the worker batch."""

        for request_id in tuple(self._detached_active_requests):
            if request_id in active_request_ids:
                continue
            handle_key = self._detached_active_requests.pop(request_id)
            remaining = self._detached_refcounts.get(handle_key, 1) - 1
            if remaining > 0:
                self._detached_refcounts[handle_key] = remaining
                continue
            self._detached_refcounts.pop(handle_key, None)
            if handle_key[1] != "warm":
                continue
            blocks = self._detached_handles.pop(handle_key, ())
            self._detached_materialized.discard(handle_key)
            self._detached_tensor_bytes.pop(handle_key, None)
            if self._detached_free is not None:
                self._detached_free.extend(blocks)
                self._detached_free.sort()

    @staticmethod
    def _inject(
        destination: torch.Tensor,
        source: torch.Tensor,
        slots: torch.Tensor,
        attention_metadata: "AttentionMetadata",
        block_size: int,
    ) -> None:
        slots = slots.to(destination.device, non_blocking=True)
        source = source.to(destination.device, non_blocking=True)
        if isinstance(attention_metadata, MLACommonMetadata):
            destination.reshape(-1, destination.shape[-1])[slots] = source
            return
        blocks = slots // block_size
        offsets = slots % block_size
        destination[blocks, :, offsets] = source

    @staticmethod
    def _extract(
        source: torch.Tensor,
        slots: torch.Tensor,
        attention_metadata: "AttentionMetadata",
        block_size: int,
    ) -> torch.Tensor:
        slots = slots.to(source.device, non_blocking=True)
        if isinstance(attention_metadata, MLACommonMetadata):
            return source.reshape(-1, source.shape[-1])[slots]
        blocks = slots // block_size
        offsets = slots % block_size
        return source[blocks, :, offsets]

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PRASemanticConnectorMetadata):
            raise TypeError("Unexpected PRA CUDA connector metadata.")
        attention = forward_context.attn_metadata
        if attention is None:
            return
        detached_requests = [
            request
            for request in metadata.requests
            if request.mode == "load" and request.detached
        ]
        if detached_requests:
            self._start_detached_load(forward_context, detached_requests)
        for request in metadata.requests:
            if request.mode != "load" or request.detached:
                continue
            started = time.perf_counter()
            storage_read_ms = 0.0
            storage_read_bytes = 0
            h2d_bytes = 0
            d2d_bytes = 0
            hot_hits = 0
            hot_misses = 0
            directory = self._directory(request.logical_key)
            for layer_name, layer in forward_context.no_compile_layers.items():
                cache = getattr(layer, "kv_cache", None)
                if cache is None:
                    continue
                layer_attention = (
                    attention[layer_name] if isinstance(attention, dict) else attention
                )
                path = directory / _layer_file_name(layer_name)
                cache_key = (request.logical_key, layer_name)
                tensor = self._hot_layers.get(cache_key)
                if request.residency == "hot" and tensor is not None:
                    hot_hits += 1
                else:
                    read_started = time.perf_counter()
                    tensor = safetensors.torch.load_file(str(path))["kv_cache"]
                    storage_read_ms += (time.perf_counter() - read_started) * 1000.0
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    storage_read_bytes += tensor_bytes
                    if request.residency == "hot":
                        hot_misses += 1
                        tensor = tensor.to(cache.device, non_blocking=True)
                        h2d_bytes += tensor_bytes
                        self._hot_layers[cache_key] = tensor
                tensor_bytes = tensor.numel() * tensor.element_size()
                if tensor.device == cache.device:
                    d2d_bytes += tensor_bytes
                else:
                    h2d_bytes += tensor_bytes
                self._inject(
                    cache,
                    tensor,
                    request.slot_mapping,
                    layer_attention,
                    self._block_size,
                )
            if self._telemetry is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._write_telemetry(
                {
                    "schema_version": "pra-vllm-cuda-transfer-v1",
                    "event": "load",
                    "request_id": request.request_id,
                    "logical_key": request.logical_key,
                    "residency": request.residency,
                    "source_tokens": request.source_tokens,
                    "storage_read_bytes": storage_read_bytes,
                    "storage_read_ms": storage_read_ms,
                    "h2d_bytes": h2d_bytes,
                    "d2d_bytes": d2d_bytes,
                    "hot_layer_hits": hot_hits,
                    "hot_layer_misses": hot_misses,
                    "resident_hot_bytes": self._resident_hot_bytes(),
                    "load_ms": (time.perf_counter() - started) * 1000.0,
                }
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PRASemanticConnectorMetadata):
            raise TypeError("Unexpected PRA CUDA connector metadata.")
        for request in metadata.requests:
            if request.mode != "store":
                continue
            if request.logical_key not in self._active_store_keys:
                self._stored_tensor_bytes[request.logical_key] = 0
            for cache_key in tuple(self._hot_layers):
                if cache_key[0] == request.logical_key:
                    del self._hot_layers[cache_key]
            directory = self._directory(request.logical_key)
            directory.mkdir(parents=True, exist_ok=True)
            selected = self._extract(
                kv_layer,
                request.slot_mapping,
                attn_metadata,
                self._block_size,
            )
            self._stored_tensor_bytes[request.logical_key] += (
                selected.numel() * selected.element_size()
            )
            safetensors.torch.save_file(
                {"kv_cache": selected.detach().cpu().contiguous()},
                str(directory / _layer_file_name(layer_name)),
            )
            self._active_store_keys.add(request.logical_key)

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PRASemanticConnectorMetadata):
            return
        for request in metadata.requests:
            if request.mode != "store" or request.logical_key not in self._active_store_keys:
                continue
            directory = self._directory(request.logical_key)
            manifest = {
                "schema_version": "pra-vllm-cuda-kv-v1",
                "logical_key": request.logical_key,
                "source_tokens": request.source_tokens,
                "layer_files": len(list(directory.glob("layer-*.safetensors"))),
                "native_tensor_bytes": self._stored_tensor_bytes.get(
                    request.logical_key, 0
                ),
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._active_store_keys.discard(request.logical_key)
            self._stored_tensor_bytes.pop(request.logical_key, None)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._commands.pop(request.request_id, None)
        self._loads.pop(request.request_id, None)
        handle_key = self._detached_active_requests.pop(request.request_id, None)
        if handle_key is not None:
            remaining = self._detached_refcounts.get(handle_key, 1) - 1
            if remaining > 0:
                self._detached_refcounts[handle_key] = remaining
            else:
                self._detached_refcounts.pop(handle_key, None)
                if handle_key[1] == "warm":
                    blocks = self._detached_handles.pop(handle_key, ())
                    self._detached_materialized.discard(handle_key)
                    self._detached_tensor_bytes.pop(handle_key, None)
                    if self._detached_free is not None:
                        self._detached_free.extend(blocks)
                        self._detached_free.sort()
        return False, None
