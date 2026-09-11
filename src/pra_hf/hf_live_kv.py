"""Resident agent-history K/V selection for Transformers DynamicCache."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Generic, TypeVar

from .live_history import LiveKVSelectionPlan, LiveKVSourceRegistry


T = TypeVar("T")


@dataclass(frozen=True)
class HFResidentKVSelection:
    """A request cache derived only from an already evaluated source cache."""

    cache: object
    plan: LiveKVSelectionPlan
    physical_kv_copy: bool
    selected_text_reencoded_tokens: int = 0


class HFLiveKVRequestCancelled(RuntimeError):
    """Raised after a cooperative cancellation releases its source borrow."""


@dataclass(frozen=True)
class HFLiveKVGeneration:
    """Deterministic decode output and lifecycle accounting for one request."""

    token_ids: tuple[int, ...]
    step_logits: tuple[object, ...]
    source_position_base: int
    selected_kv_tokens: int
    selected_text_reencoded_tokens: int
    physical_kv_copy: bool


@dataclass
class HFLiveKVRequest(Generic[T]):
    """One request-scoped borrow of immutable canonical agent-history K/V.

    A request receives a private cache descriptor assembled from the borrowed
    source.  Transformers may extend that descriptor during decode, while the
    canonical source remains pinned and immutable until exactly one terminal
    transition runs.
    """

    runtime: "HFLiveKVRuntime[T]"
    request_id: str
    source_id: str
    tenant_id: str
    session_id: str
    generation: int
    selection: HFResidentKVSelection
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

    def __enter__(self) -> "HFLiveKVRequest[T]":
        if self._closed:
            raise RuntimeError("A closed HF live-K/V request cannot be re-entered.")
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        if exc_type is None:
            self.finish()
        elif issubclass(exc_type, HFLiveKVRequestCancelled):
            self.cancel()
        else:
            self.fail()
        return False

    def generate(
        self,
        model,
        tail_input_ids,
        *,
        max_new_tokens: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> HFLiveKVGeneration:
        """Run greedy HF decode at the source's original position extent.

        The selected history is never tokenized or evaluated.  Only the wire
        tail and generated tokens enter the model.  The request borrow spans
        every model call and is released through the same terminal path on
        success, cooperative cancellation, or an arbitrary model exception.
        """

        if self._closed:
            raise RuntimeError("A closed HF live-K/V request cannot generate.")
        try:
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive.")

            import torch

            ids = tail_input_ids
            if not torch.is_tensor(ids):
                ids = torch.tensor([list(ids)], dtype=torch.long)
            elif ids.ndim == 1:
                ids = ids.unsqueeze(0)
            if ids.ndim != 2 or int(ids.shape[0]) != 1 or int(ids.shape[1]) == 0:
                raise ValueError(
                    "HF live-K/V generation requires one non-empty wire tail."
                )
            try:
                device = next(model.parameters()).device
            except (AttributeError, StopIteration):
                device = ids.device
            ids = ids.to(device=device, dtype=torch.long)
            position_base = self.selection.plan.source_position_base
            positions = torch.arange(
                position_base,
                position_base + int(ids.shape[1]),
                dtype=torch.long,
                device=device,
            )
            generated: list[int] = []
            logits_trace: list[object] = []
            cache = self.selection.cache
        except BaseException:
            self.fail()
            raise

        try:
            if cancelled is not None and cancelled():
                raise HFLiveKVRequestCancelled(
                    f"HF live-K/V request {self.request_id!r} was cancelled."
                )
            with torch.inference_mode():
                output = model(
                    input_ids=ids,
                    past_key_values=cache,
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    use_cache=True,
                    return_dict=True,
                )
            logits = output.logits
            cache = output.past_key_values
            for step in range(max_new_tokens):
                if cancelled is not None and cancelled():
                    raise HFLiveKVRequestCancelled(
                        f"HF live-K/V request {self.request_id!r} was cancelled."
                    )
                current = logits[:, -1, :]
                logits_trace.append(current.detach().float().cpu())
                token = int(torch.argmax(current, dim=-1).item())
                generated.append(token)
                if step + 1 == max_new_tokens:
                    break
                position = torch.tensor(
                    [position_base + int(ids.shape[1]) + step],
                    dtype=torch.long,
                    device=device,
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                        past_key_values=cache,
                        position_ids=position.unsqueeze(0),
                        cache_position=position,
                        use_cache=True,
                        return_dict=True,
                    )
                logits = output.logits
                cache = output.past_key_values
        except HFLiveKVRequestCancelled:
            self.cancel()
            raise
        except BaseException:
            self.fail()
            raise
        else:
            self.finish()

        return HFLiveKVGeneration(
            tuple(generated),
            tuple(logits_trace),
            position_base,
            self.selection.plan.selected_tokens,
            self.selection.selected_text_reencoded_tokens,
            self.selection.physical_kv_copy,
        )


class HFLiveKVRuntime(Generic[T]):
    """Bind canonical K/V ownership to synchronous Transformers requests."""

    def __init__(
        self,
        *,
        dump: Callable[[T], object] | None = None,
        load: Callable[[object], T] | None = None,
    ) -> None:
        self.registry = LiveKVSourceRegistry[T](dump=dump, load=load)
        self._requests: dict[str, HFLiveKVRequest[T]] = {}
        self._terminal_counts = {"finished": 0, "cancelled": 0, "error": 0}
        self._lock = RLock()

    def register_source(
        self,
        source_id: str,
        cache: T,
        *,
        tenant_id: str,
        session_id: str,
        generation: int,
    ) -> None:
        self.registry.register(
            source_id,
            cache,
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
    ) -> HFLiveKVRequest[T]:
        """Borrow the canonical source before creating request-local cache state."""

        request_key = str(request_id)
        with self._lock:
            if request_key in self._requests:
                raise RuntimeError(f"HF live-K/V request {request_key!r} is already active.")
            (source_cache,) = self.registry.borrow(
                request_key,
                (source_id,),
                tenant_id=tenant_id,
                session_id=session_id,
                expected_generations=(expected_generation,),
            )
            try:
                selection = select_dynamic_cache(source_cache, plan)
            except BaseException:
                self.registry.release(request_key)
                raise
            request = HFLiveKVRequest(
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

    def _release_request(self, request: HFLiveKVRequest[T], outcome: str) -> bool:
        with self._lock:
            if request._closed:
                return False
            active = self._requests.get(request.request_id)
            if active is not request:
                raise RuntimeError("HF live-K/V request ownership is inconsistent.")
            released = self.registry.release(request.request_id)
            if not released:
                raise RuntimeError("HF live-K/V registry lost an active request borrow.")
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
        """Cancel active work first, then atomically tombstone the session."""

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


def _layer_pair(layer: object) -> tuple[object, object, str, str]:
    for key_name, value_name in (
        ("keys", "values"),
        ("key_cache", "value_cache"),
    ):
        keys = getattr(layer, key_name, None)
        values = getattr(layer, value_name, None)
        if keys is not None and values is not None:
            return keys, values, key_name, value_name
    raise TypeError(
        f"Unsupported Transformers cache layer {type(layer)!r}; expected K/V tensors."
    )


def _select_tensor(tensor, plan: LiveKVSelectionPlan):
    import torch

    pieces = [tensor[..., row.start : row.end, :] for row in plan.intervals]
    if not pieces:
        return tensor[..., :0, :], False
    if len(pieces) == 1:
        return pieces[0], False
    return torch.cat(pieces, dim=-2), True


def select_dynamic_cache(
    source_cache: object, plan: LiveKVSelectionPlan
) -> HFResidentKVSelection:
    """Select resident K/V tensors without invoking tokenization or the model.

    A single contiguous interval remains a tensor view. Multiple disjoint
    intervals require a packed tensor copy in the portable HF cache API; that
    cost is reported explicitly and is not confused with text re-encoding.
    """

    layers = getattr(source_cache, "layers", None)
    if layers is None:
        raise TypeError("Live HF K/V selection requires a Transformers cache with layers.")
    selected = copy.copy(source_cache)
    selected.layers = []
    copied = False
    for source_layer in layers:
        keys, values, key_name, value_name = _layer_pair(source_layer)
        if int(keys.shape[-2]) < plan.source_tokens:
            raise ValueError("HF source cache is shorter than the selection plan.")
        layer = copy.copy(source_layer)
        chosen_keys, key_copy = _select_tensor(keys, plan)
        chosen_values, value_copy = _select_tensor(values, plan)
        setattr(layer, key_name, chosen_keys)
        setattr(layer, value_name, chosen_values)
        selected.layers.append(layer)
        copied = copied or key_copy or value_copy
    return HFResidentKVSelection(selected, plan, copied)
