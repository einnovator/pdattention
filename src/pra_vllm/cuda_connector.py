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

    @classmethod
    def create(
        cls,
        command: CudaConnectorCommand,
        block_ids: list[int],
        block_size: int,
        request_id: str,
    ) -> "_RequestTransfer":
        if command.source_tokens % block_size:
            raise ValueError("PRA CUDA source K/V must be block aligned.")
        required_blocks = command.source_tokens // block_size
        if len(block_ids) < required_blocks:
            raise RuntimeError("vLLM allocated too few blocks for PRA source K/V.")
        blocks = torch.tensor(block_ids[:required_blocks], dtype=torch.int64)
        offsets = torch.arange(block_size, dtype=torch.int64)
        slots = (blocks[:, None] * block_size + offsets[None, :]).flatten()
        return cls(
            request_id=request_id,
            logical_key=command.logical_key,
            source_tokens=command.source_tokens,
            slot_mapping=slots,
            mode=command.mode,
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
        configured = self._kv_transfer_config.get_from_extra_config(
            "storage_path", "/tmp/pra-vllm-cuda"
        )
        self._storage = Path(str(configured)).expanduser().resolve()
        self._storage.mkdir(parents=True, exist_ok=True)
        self._commands: dict[str, CudaConnectorCommand] = {}
        self._loads: dict[str, "Request"] = {}
        self._active_store_keys: set[str] = set()

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
        if num_external_tokens > 0:
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
        if command.mode == "load" and req_id not in self._loads:
            return
        metadata.requests.append(
            _RequestTransfer.create(
                command, block_ids, self._block_size, request_id=req_id
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
        for request in metadata.requests:
            if request.mode != "load":
                continue
            directory = self._directory(request.logical_key)
            for layer_name, layer in forward_context.no_compile_layers.items():
                cache = getattr(layer, "kv_cache", None)
                if cache is None:
                    continue
                layer_attention = (
                    attention[layer_name] if isinstance(attention, dict) else attention
                )
                path = directory / _layer_file_name(layer_name)
                tensor = safetensors.torch.load_file(str(path))["kv_cache"]
                self._inject(
                    cache,
                    tensor,
                    request.slot_mapping,
                    layer_attention,
                    self._block_size,
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
            directory = self._directory(request.logical_key)
            directory.mkdir(parents=True, exist_ok=True)
            selected = self._extract(
                kv_layer,
                request.slot_mapping,
                attn_metadata,
                self._block_size,
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
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._active_store_keys.discard(request.logical_key)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._commands.pop(request.request_id, None)
        self._loads.pop(request.request_id, None)
        return False, None
