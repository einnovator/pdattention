"""Request-owned sparse live-K/V lifecycle for in-process mlx-lm."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Callable

from pra_hf.live_history import LiveKVSelectionPlan, LiveKVSourceRegistry

from .native import (
    MLXNativeMemory,
    MLXResidentKVSelection,
    make_native_prompt_cache,
    select_live_native_memory,
)


class MLXLiveKVRequestCancelled(RuntimeError):
    """Raised after cooperative cancellation releases a request borrow."""


@dataclass(frozen=True)
class MLXLiveKVGeneration:
    """Greedy output plus K/V materialization accounting for one request."""

    token_ids: tuple[int, ...]
    step_logits: tuple[object, ...]
    source_position_base: int
    selected_kv_tokens: int
    selected_text_reencoded_tokens: int
    physical_kv_copy: bool


@dataclass
class MLXLiveKVRequest:
    """One request-scoped borrow of immutable canonical MLX agent K/V."""

    runtime: "MLXLiveKVRuntime"
    request_id: str
    source_id: str
    tenant_id: str
    session_id: str
    generation: int
    selection: MLXResidentKVSelection
    _closed: bool = field(default=False, init=False, repr=False)
    _outcome: str | None = field(default=None, init=False, repr=False)

    @property
    def active(self) -> bool:
        return not self._closed

    @property
    def outcome(self) -> str | None:
        return self._outcome

    def _close(self, outcome: str) -> bool:
        return self.runtime._release_request(self, outcome)

    def finish(self) -> bool:
        return self._close("finished")

    def cancel(self) -> bool:
        return self._close("cancelled")

    def fail(self) -> bool:
        return self._close("error")

    def __enter__(self) -> "MLXLiveKVRequest":
        if self._closed:
            raise RuntimeError("A closed MLX live-K/V request cannot be re-entered.")
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        if exc_type is None:
            self.finish()
        elif issubclass(exc_type, MLXLiveKVRequestCancelled):
            self.cancel()
        else:
            self.fail()
        return False

    def _check_cancelled(self, cancelled: Callable[[], bool] | None) -> None:
        if self._closed:
            if self._outcome == "cancelled":
                raise MLXLiveKVRequestCancelled(
                    f"MLX live-K/V request {self.request_id!r} was cancelled."
                )
            raise RuntimeError("A closed MLX live-K/V request cannot generate.")
        if cancelled is not None and cancelled():
            raise MLXLiveKVRequestCancelled(
                f"MLX live-K/V request {self.request_id!r} was cancelled."
            )

    def generate(
        self,
        model: object,
        tail_input_ids,
        *,
        max_new_tokens: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> MLXLiveKVGeneration:
        """Decode from selected live K/V without evaluating selected text.

        mlx-lm cache offsets position the wire tail at the full source extent,
        not at the packed selected width.  The model-runner lock reflects the
        current in-process mlx-lm non-reentrancy while still allowing multiple
        requests to hold independent borrows concurrently.
        """

        if self._closed:
            raise RuntimeError("A closed MLX live-K/V request cannot generate.")
        try:
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive.")
            values = [int(value) for value in tail_input_ids]
            if not values:
                raise ValueError(
                    "MLX live-K/V generation requires one non-empty wire tail."
                )
            import mlx.core as mx

            cache = make_native_prompt_cache(
                model,
                self.selection.memory,
                query_position_base=self.selection.plan.source_position_base,
            )
        except BaseException:
            self.fail()
            raise

        generated: list[int] = []
        logits_trace: list[object] = []
        try:
            with self.runtime._model_runner_lock:
                self._check_cancelled(cancelled)
                logits = model(mx.array([values], dtype=mx.int32), cache=cache)
                for step in range(max_new_tokens):
                    self._check_cancelled(cancelled)
                    current = logits[0, -1].astype(mx.float32)
                    mx.eval(current)
                    logits_trace.append(current)
                    token = int(mx.argmax(current).item())
                    generated.append(token)
                    if step + 1 == max_new_tokens:
                        break
                    logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        except MLXLiveKVRequestCancelled:
            self.cancel()
            raise
        except BaseException:
            self.fail()
            raise
        else:
            if not self.finish():
                raise MLXLiveKVRequestCancelled(
                    f"MLX live-K/V request {self.request_id!r} was cancelled."
                )

        return MLXLiveKVGeneration(
            tuple(generated),
            tuple(logits_trace),
            self.selection.plan.source_position_base,
            self.selection.plan.selected_tokens,
            self.selection.selected_text_reencoded_tokens,
            self.selection.physical_kv_copy,
        )


class MLXLiveKVRuntime:
    """Bind canonical MLX K/V ownership to synchronous mlx-lm requests."""

    def __init__(
        self,
        *,
        dump: Callable[[MLXNativeMemory], object] | None = None,
        load: Callable[[object], MLXNativeMemory] | None = None,
    ) -> None:
        self.registry = LiveKVSourceRegistry[MLXNativeMemory](dump=dump, load=load)
        self._requests: dict[str, MLXLiveKVRequest] = {}
        self._terminal_counts = {"finished": 0, "cancelled": 0, "error": 0}
        self._lock = RLock()
        self._model_runner_lock = RLock()

    def register_source(
        self,
        source_id: str,
        memory: MLXNativeMemory,
        *,
        tenant_id: str,
        session_id: str,
        generation: int,
    ) -> None:
        self.registry.register(
            source_id,
            memory,
            tenant_id=tenant_id,
            session_id=session_id,
            generation=generation,
        )

    def begin_request(
        self,
        request_id: str,
        source_id: str,
        plan: LiveKVSelectionPlan,
        *,
        tenant_id: str,
        session_id: str,
        expected_generation: int,
    ) -> MLXLiveKVRequest:
        """Borrow the canonical source before constructing selected K/V."""

        request_key = str(request_id)
        with self._lock:
            if request_key in self._requests:
                raise RuntimeError(
                    f"MLX live-K/V request {request_key!r} is already active."
                )
            (source_memory,) = self.registry.borrow(
                request_key,
                (source_id,),
                tenant_id=tenant_id,
                session_id=session_id,
                expected_generations=(expected_generation,),
            )
            try:
                selection = select_live_native_memory(source_memory, plan)
            except BaseException:
                self.registry.release(request_key)
                raise
            request = MLXLiveKVRequest(
                self,
                request_key,
                str(source_id),
                str(tenant_id),
                str(session_id),
                int(expected_generation),
                selection,
            )
            self._requests[request_key] = request
            return request

    def _release_request(self, request: MLXLiveKVRequest, outcome: str) -> bool:
        with self._lock:
            if request._closed:
                return False
            active = self._requests.get(request.request_id)
            if active is not request:
                raise RuntimeError("MLX live-K/V request ownership is inconsistent.")
            if not self.registry.release(request.request_id):
                raise RuntimeError("MLX live-K/V registry lost an active request borrow.")
            del self._requests[request.request_id]
            request._closed = True
            request._outcome = outcome
            self._terminal_counts[outcome] += 1
            return True

    def cancel_request(self, request_id: str) -> bool:
        with self._lock:
            request = self._requests.get(str(request_id))
            return False if request is None else request.cancel()

    def offload_source(self, source_id: str) -> object:
        return self.registry.offload(source_id)

    def terminate_session(self, tenant_id: str, session_id: str) -> int:
        """Cancel all request borrows, then tombstone and remove the session."""

        with self._lock:
            affected = tuple(
                request
                for request in self._requests.values()
                if (request.tenant_id, request.session_id)
                == (str(tenant_id), str(session_id))
            )
            for request in affected:
                request.cancel()
            return self.registry.terminate_session(tenant_id, session_id)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active_request_ids": tuple(sorted(self._requests)),
                "terminal_counts": dict(self._terminal_counts),
            }
