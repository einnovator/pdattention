"""In-process SGLang executor for typed PRA gateway requests."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Iterator, Mapping

from pra_hf.deployment import PRAEngineResult, PRAWireRequest, PRAWireResource
from pra_hf.engine_memory import LogicalPRABlockStore
from pra_hf.storage_lifecycle import (
    PRARetentionClass,
    PRAStorageEntry,
    PRAStorageManager,
)
from pra_mlx.native import encode_native_memory, serialize_native_memory

from .mlx_native import SGLangMLXNativeBridge


class SGLangInProcessNativeExecutor:
    """Consume selected logical resources through an MLX-backed SGLang runner.

    The executor is the missing product boundary between ``SGLangEngineAdapter``
    and ``SGLangMLXNativeBridge``.  Resource encoding and residency survive
    requests, while bridge registration, request pinning, and selected-cache
    attachment last only for one generation.  Runner calls are serialized
    because the current in-process MLX runner does not expose its HTTP
    scheduler; this is an honest online queue baseline, not continuous batching.
    """

    def __init__(
        self,
        runner: object,
        tokenizer: object,
        *,
        model_id: str,
        model_revision: str,
        block_store: LogicalPRABlockStore,
        storage_manager: PRAStorageManager,
        bridge: SGLangMLXNativeBridge | None = None,
    ) -> None:
        self.runner = runner
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self.block_store = block_store
        self.storage = storage_manager
        self.bridge = bridge or SGLangMLXNativeBridge(
            runner, storage_manager=storage_manager
        )
        self._runner_lock = threading.RLock()
        self._session_keys: dict[str, set[str]] = {}

    @staticmethod
    def _selected_resources(request: PRAWireRequest) -> tuple[PRAWireResource, ...]:
        selected = tuple(
            map(str, request.pra_policy.get("selected_resource_ids", ()))
        )
        resources = {resource.resource_id: resource for resource in request.resources}
        if selected:
            missing = [key for key in selected if key not in resources]
            if missing:
                raise KeyError(f"Selected PRA resources are absent: {missing}")
            return tuple(resources[key] for key in selected)
        return request.resources[: request.budget.max_resources]

    def _logical_key(self, request: PRAWireRequest, resource: PRAWireResource) -> str:
        shareable = bool(resource.metadata.get("shareable", False))
        identity = {
            "tenant_id": request.tenant_id,
            "session_id": None if shareable else request.session_id,
            "resource_id": resource.resource_id,
            "resource_version": resource.metadata.get("version"),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "layout": "sglang-mlx-per-layer-bhld",
            "position_policy": "source-local",
        }
        return "sglang:" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _resource_tokens(self, resource: PRAWireResource) -> list[int]:
        if resource.text is None:
            raise ValueError(f"Resource {resource.resource_id!r} has no text.")
        return list(map(int, self.tokenizer.encode(resource.text, add_special_tokens=False)))

    def _prepare_keys(self, request: PRAWireRequest) -> tuple[tuple[str, ...], int]:
        resources = self._selected_resources(request)
        if not resources:
            raise ValueError("Native SGLang PRA requires one selected resource.")
        prepared = [(resource, self._resource_tokens(resource)) for resource in resources]
        total_tokens = sum(len(tokens) for _, tokens in prepared)
        if total_tokens > request.budget.max_selected_tokens:
            raise ValueError("Selected PRA resources exceed max_selected_tokens.")
        keys = []
        for resource, tokens in prepared:
            key = self._logical_key(request, resource)
            keys.append(key)
            if key not in self.storage.entries:
                memory = encode_native_memory(self.runner.model, tokens)
                payload = serialize_native_memory(memory)
                self.storage.register(
                    PRAStorageEntry(
                        logical_key=key,
                        record_type=resource.record_type,
                        retention_class=PRARetentionClass.RECONSTRUCTABLE,
                        tenant_id=request.tenant_id,
                        session_id=(
                            None
                            if resource.metadata.get("shareable", False)
                            else request.session_id
                        ),
                        task_id=request.task_id,
                        task_status=None,
                        resource_version=str(resource.metadata.get("version", "source-hash")),
                        detail_bytes=memory.nbytes,
                        security_scope=resource.authorization_scope,
                        source_reconstructable=True,
                        shared_reference_count=int(
                            bool(resource.metadata.get("shareable", False))
                        ),
                    ),
                    payload,
                    hot_value=memory,
                    source_loader=lambda payload=payload: payload,
                    fingerprint=(
                        f"{self.model_id}:{self.model_revision}:"
                        f"{len(memory.layers)}:{len(tokens)}"
                    ),
                )
            self.storage.record_access(key, selected=True)
        if request.session_id:
            self._session_keys.setdefault(request.session_id, set()).update(keys)
        return tuple(keys), total_tokens

    def _prompt_tokens(self, request: PRAWireRequest) -> list[int]:
        apply = getattr(self.tokenizer, "apply_chat_template", None)
        if apply is not None:
            return list(
                map(
                    int,
                    apply(
                        list(request.messages),
                        tokenize=True,
                        add_generation_prompt=True,
                    ),
                )
            )
        text = "\n".join(str(row.get("content", "")) for row in request.messages)
        return list(map(int, self.tokenizer.encode(text, add_special_tokens=False)))

    def stream(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> Iterator[Mapping[str, object]]:
        if block_store is not self.block_store:
            raise ValueError("SGLang executor and adapter must share one block store.")
        keys, selected_tokens = self._prepare_keys(request)
        query = self._prompt_tokens(request)
        request_id = str(request.request_id)
        max_tokens = request.resolved_max_new_tokens

        def rows() -> Iterator[Mapping[str, object]]:
            with self._runner_lock:
                self.bridge.register(request_id, logical_keys=keys)
                started = time.perf_counter()
                try:
                    pending = self.runner.prefill_start(
                        request_id, query, query, [], [], 0
                    )
                    self.runner.eval_pending(pending)
                    token = int(self.runner.prefill_finalize(pending))
                    yield {
                        "text": self.tokenizer.decode([token]),
                        "token_id": token,
                        "token_index": 0,
                        "ttft_ms": (time.perf_counter() - started) * 1000.0,
                        "selected_native_tokens": selected_tokens,
                        "native_kv_used": True,
                    }
                    for index in range(1, max_tokens):
                        decode = self.runner.decode_batch_start([request_id])
                        self.runner.eval_pending(decode)
                        token = int(self.runner.decode_batch_finalize(decode)[0])
                        yield {
                            "text": self.tokenizer.decode([token]),
                            "token_id": token,
                            "token_index": index,
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                            "selected_native_tokens": selected_tokens,
                            "native_kv_used": True,
                        }
                finally:
                    self.runner.remove_request(request_id)
                    self.bridge.unregister(request_id)

        return rows()

    def generate(
        self, request: PRAWireRequest, block_store: LogicalPRABlockStore
    ) -> PRAEngineResult:
        rows = tuple(self.stream(request, block_store))
        text = "".join(str(row.get("text", "")) for row in rows)
        final = dict(rows[-1]) if rows else {}
        final["storage"] = self.storage.metrics.to_dict()
        return PRAEngineResult(text=text, raw=final, trace=rows)

    def close_session(self, session_id: str) -> None:
        self.storage.close_session(session_id)
        self._session_keys.pop(session_id, None)

    def close(self) -> None:
        self.bridge.close()
        self.storage.close()
