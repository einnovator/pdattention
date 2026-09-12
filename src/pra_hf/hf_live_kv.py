"""Resident agent-history K/V selection for Transformers DynamicCache."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from types import MethodType
from threading import RLock
from typing import Callable, Generic, Iterable, TypeVar

from .live_history import LiveKVSelectionPlan, LiveKVSourceRegistry


T = TypeVar("T")


@dataclass(frozen=True)
class HFResidentKVSelection:
    """A request cache derived only from an already evaluated source cache."""

    cache: object
    plan: LiveKVSelectionPlan
    physical_kv_copy: bool
    selected_text_reencoded_tokens: int = 0
    interval_pack_bytes: int = 0

    @property
    def transient_attention_bytes(self) -> int:
        metrics = getattr(self.cache, "metrics", None)
        return int(getattr(metrics, "transient_attention_bytes", 0))

    @property
    def transient_kv_copy_bytes(self) -> int:
        metrics = getattr(self.cache, "metrics", None)
        return int(getattr(metrics, "transient_kv_copy_bytes", 0))

    @property
    def max_transient_kv_tile_bytes(self) -> int:
        metrics = getattr(self.cache, "metrics", None)
        return int(getattr(metrics, "max_transient_kv_tile_bytes", 0))

    @property
    def request_tail_copy_bytes(self) -> int:
        metrics = getattr(self.cache, "metrics", None)
        return int(getattr(metrics, "request_tail_copy_bytes", 0))

    @property
    def fused_attention_calls(self) -> int:
        metrics = getattr(self.cache, "metrics", None)
        return int(getattr(metrics, "fused_attention_calls", 0))


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
    interval_pack_bytes: int = 0
    transient_attention_bytes: int = 0
    transient_kv_copy_bytes: int = 0
    max_transient_kv_tile_bytes: int = 0


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
            # Real Transformers models require the explicit sparse consumer.
            # Minimal protocol fakes used by lifecycle tests consume the cache
            # opaquely and intentionally have no ``modules`` traversal.
            if hasattr(model, "modules"):
                enable_qwen_sparse_live_kv(model)
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
            bool(
                self.selection.physical_kv_copy
                or self.selection.transient_kv_copy_bytes > 0
            ),
            self.selection.interval_pack_bytes,
            self.selection.transient_attention_bytes,
            self.selection.transient_kv_copy_bytes,
            self.selection.max_transient_kv_tile_bytes,
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


@dataclass(frozen=True)
class HFSparseKVSegment:
    """One K/V tensor view with positions in the canonical source frame."""

    keys: object
    values: object
    position_start: int
    position_end: int
    record_id: str = ""
    causal_group_id: str = ""

    @property
    def tokens(self) -> int:
        return self.position_end - self.position_start


@dataclass
class HFSparseAttentionMetrics:
    """Mechanism costs that must not be conflated with text re-encoding."""

    interval_pack_bytes: int = 0
    transient_attention_bytes: int = 0
    transient_kv_copy_bytes: int = 0
    max_transient_kv_tile_bytes: int = 0
    selected_text_reencoded_tokens: int = 0
    request_tail_copy_bytes: int = 0
    fused_attention_calls: int = 0


@dataclass
class _HFSparseKVLayer:
    source_segments: tuple[HFSparseKVSegment, ...]
    source_keys: object | None = None
    source_values: object | None = None
    source_intervals: tuple[tuple[int, int, int], ...] = ()
    tail_segments: list[HFSparseKVSegment] = field(default_factory=list)

    def segments(self) -> tuple[HFSparseKVSegment, ...]:
        return self.source_segments + tuple(self.tail_segments)


class HFSparseDynamicCache:
    """Non-packing request cache for Qwen2/Qwen3 full attention.

    Source segments are basic tensor slices and therefore alias the canonical
    source cache. Request-local tail K/V is appended as separate segments. The
    class deliberately does not expose a dense ``keys`` tensor: accidentally
    routing it through ordinary HF attention must fail instead of silently
    materializing selected history.
    """

    def __init__(
        self,
        layers: Iterable[_HFSparseKVLayer],
        plan: LiveKVSelectionPlan,
    ) -> None:
        self.layers = list(layers)
        self.plan = plan
        self.metrics = HFSparseAttentionMetrics()
        self.is_sliding = [False for _ in self.layers]

    @property
    def is_compileable(self) -> bool:
        return False

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        if not self.layers:
            return self.plan.source_position_base
        layer = self.layers[layer_idx]
        tail = sum(segment.tokens for segment in layer.tail_segments)
        return self.plan.source_position_base + tail

    def get_mask_sizes(self, cache_position, layer_idx: int) -> tuple[int, int]:
        # HF constructs a conventional mask before entering each decoder layer.
        # The Qwen sparse wrapper below uses original segment positions instead;
        # this extent merely keeps model-level mask construction well-defined.
        query_length = (
            int(cache_position)
            if isinstance(cache_position, int)
            else int(cache_position.shape[0])
        )
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def append(
        self,
        layer_idx: int,
        keys,
        values,
        cache_position,
    ) -> tuple[HFSparseKVSegment, ...]:
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(f"Sparse HF cache has no layer {layer_idx}.")
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("Sparse HF Qwen K/V must be equal-shape rank-four tensors.")
        positions = tuple(int(value) for value in cache_position.detach().cpu().tolist())
        if not positions or positions != tuple(range(positions[0], positions[0] + len(positions))):
            raise ValueError("Sparse HF tail cache positions must be contiguous and ordered.")
        segment = HFSparseKVSegment(
            keys,
            values,
            positions[0],
            positions[-1] + 1,
            record_id="request-tail",
            causal_group_id="request-tail",
        )
        self.layers[layer_idx].tail_segments.append(segment)
        return self.layers[layer_idx].segments()

    def segments(self, layer_idx: int) -> tuple[HFSparseKVSegment, ...]:
        return self.layers[layer_idx].segments()


def _tensor_storage_pointer(tensor) -> int:
    storage = tensor.untyped_storage()
    return int(storage.data_ptr())


def _source_segment_views(keys, values, plan: LiveKVSelectionPlan):
    segments: list[HFSparseKVSegment] = []
    for interval in plan.intervals:
        key_view = keys[..., interval.start : interval.end, :]
        value_view = values[..., interval.start : interval.end, :]
        if _tensor_storage_pointer(key_view) != _tensor_storage_pointer(keys):
            raise RuntimeError("HF sparse key interval unexpectedly copied source storage.")
        if _tensor_storage_pointer(value_view) != _tensor_storage_pointer(values):
            raise RuntimeError("HF sparse value interval unexpectedly copied source storage.")
        segments.append(
            HFSparseKVSegment(
                key_view,
                value_view,
                interval.start,
                interval.end,
                interval.record_id,
                interval.causal_group_id,
            )
        )
    return tuple(segments)


def segmented_qwen_attention(
    query,
    segments: Iterable[HFSparseKVSegment],
    query_positions,
    *,
    scaling: float,
    metrics: HFSparseAttentionMetrics | None = None,
    score_tile_tokens: int = 256,
    source_keys=None,
    source_values=None,
    source_intervals: tuple[tuple[int, int, int], ...] = (),
):
    """Evaluate GQA over disjoint K/V views without packing selected K/V.

    This portable implementation uses an online softmax over intervals. It is
    a correctness consumer, not a fused sparse kernel: score/probability
    tensors are transient and their logical allocation is accounted separately.
    """

    import torch

    if query.ndim != 4:
        raise ValueError("Qwen sparse attention expects [batch, heads, query, dim].")
    if int(query.shape[0]) != 1:
        raise ValueError("Qwen sparse live-history currently supports batch size one.")
    if query_positions.ndim == 2:
        if int(query_positions.shape[0]) != 1:
            raise ValueError("Qwen sparse query positions currently support one batch.")
        query_positions = query_positions[0]
    if query_positions.ndim != 1 or int(query_positions.shape[0]) != int(query.shape[-2]):
        raise ValueError("Query positions must provide one original position per query.")
    rows = tuple(segments)
    if not rows:
        raise ValueError("Sparse attention requires at least the current query K/V segment.")

    if query.is_cuda and source_keys is not None and source_values is not None:
        source_count = len(source_intervals)
        tail_rows = rows[source_count:]
        if source_count <= 0 or len(rows) < source_count or not tail_rows:
            raise ValueError(
                "Fused HF attention requires source intervals and request-local K/V."
            )
        expected_position = tail_rows[0].position_start
        for row in tail_rows:
            if row.position_start != expected_position:
                raise ValueError("Fused HF request-local K/V positions must be contiguous.")
            expected_position = row.position_end
        if len(tail_rows) == 1:
            local_keys = tail_rows[0].keys
            local_values = tail_rows[0].values
            tail_copy_bytes = 0
        else:
            local_keys = torch.cat([row.keys for row in tail_rows], dim=-2)
            local_values = torch.cat([row.values for row in tail_rows], dim=-2)
            tail_copy_bytes = (
                local_keys.numel() * local_keys.element_size()
                + local_values.numel() * local_values.element_size()
            )
        from .hf_triton_attention import triton_disjoint_selected_attention

        output, interval_table = triton_disjoint_selected_attention(
            query,
            source_keys,
            source_values,
            source_intervals,
            local_keys,
            local_values,
            query_positions,
            scale=scaling,
        )
        if metrics is not None:
            metrics.fused_attention_calls += 1
            metrics.request_tail_copy_bytes += int(tail_copy_bytes)
            metrics.transient_attention_bytes += int(
                tail_copy_bytes
                + interval_table.numel() * interval_table.element_size()
                + output.numel() * output.element_size()
            )
        return output

    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    kv_heads = int(rows[0].keys.shape[1])
    if query_heads % kv_heads:
        raise ValueError("Qwen query heads must be divisible by K/V heads.")
    groups = query_heads // kv_heads
    grouped_query = query.reshape(batch, kv_heads, groups, query_tokens, head_dim)
    # Scale before the native-dtype dot product.  Scaling the fp16 result after
    # reduction can overflow even when the final score is representable.  The
    # query is request-local and small; selected K/V must remain storage views.
    scaled_query = grouped_query * float(scaling)
    accumulator = torch.zeros(
        (batch, kv_heads, groups, query_tokens, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    denominator = torch.zeros(
        (batch, kv_heads, groups, query_tokens, 1),
        dtype=torch.float32,
        device=query.device,
    )
    running_max = torch.full_like(denominator, -torch.inf)
    transient_peak = accumulator.numel() * accumulator.element_size()
    transient_peak += denominator.numel() * denominator.element_size() * 2
    transient_peak += scaled_query.numel() * scaled_query.element_size()

    if score_tile_tokens <= 0:
        raise ValueError("score_tile_tokens must be positive.")
    def tiles():
        """Yield validated K/V views and their original source positions."""

        for segment in rows:
            keys, values = segment.keys, segment.values
            if keys.ndim != 4 or values.shape != keys.shape:
                raise ValueError(
                    "Every sparse K/V segment must contain equal rank-four tensors."
                )
            if tuple(keys.shape[:2]) != (batch, kv_heads) or int(keys.shape[-1]) != head_dim:
                raise ValueError("Sparse K/V segment geometry does not match the query.")
            key_tokens = int(keys.shape[-2])
            if key_tokens != segment.tokens:
                raise ValueError(
                    "Sparse K/V tensor width does not match its position interval."
                )
            for tile_start in range(0, key_tokens, score_tile_tokens):
                tile_end = min(tile_start + score_tile_tokens, key_tokens)
                yield (
                    keys[..., tile_start:tile_end, :],
                    values[..., tile_start:tile_end, :],
                    segment.position_start + tile_start,
                    segment.position_start + tile_end,
                )

    # First pass computes one global normalization without copying or casting
    # selected K/V.  Only bounded score tiles and small reduction tensors are
    # materialized.  The second pass recomputes scores and applies normalized
    # native-dtype probabilities to the resident value views.  This mirrors
    # eager Qwen attention more closely than accumulating fp32 value copies.
    for key_tile, _value_tile, position_start, position_end in tiles():
        logits32 = torch.einsum(
            "bhgqd,bhkd->bhgqk", scaled_query, key_tile
        ).float()
        if not bool(torch.all(torch.isfinite(logits32))):
            raise RuntimeError(
                "Sparse HF Qwen native score kernel produced a non-finite value."
            )
        tile_tokens = position_end - position_start
        key_positions = torch.arange(
            position_start,
            position_end,
            dtype=query_positions.dtype,
            device=query_positions.device,
        )
        visible = key_positions.view(1, 1, 1, 1, tile_tokens) <= query_positions.view(
            1, 1, 1, query_tokens, 1
        )
        masked_logits = logits32.masked_fill(~visible, -torch.inf)
        segment_max = masked_logits.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, segment_max)
        finite_new_max = torch.isfinite(new_max)
        safe_new_max = torch.where(finite_new_max, new_max, 0.0)
        prior_scale = torch.where(
            torch.isfinite(running_max) & finite_new_max,
            torch.exp(running_max - safe_new_max),
            0.0,
        )
        weights = torch.exp(masked_logits - safe_new_max)
        weights = torch.where(visible & finite_new_max, weights, 0.0)
        denominator = denominator * prior_scale + weights.sum(dim=-1, keepdim=True)
        running_max = new_max
        transient_peak = max(
            transient_peak,
            logits32.numel() * logits32.element_size()
            + masked_logits.numel() * masked_logits.element_size()
            + weights.numel() * weights.element_size(),
        )

    if not bool(torch.all(torch.isfinite(denominator))) or bool(torch.any(denominator <= 0)):
        raise RuntimeError("Sparse causal attention produced a non-finite or empty denominator.")

    for key_tile, value_tile, position_start, position_end in tiles():
        logits32 = torch.einsum(
            "bhgqd,bhkd->bhgqk", scaled_query, key_tile
        ).float()
        tile_tokens = position_end - position_start
        key_positions = torch.arange(
            position_start,
            position_end,
            dtype=query_positions.dtype,
            device=query_positions.device,
        )
        visible = key_positions.view(1, 1, 1, 1, tile_tokens) <= query_positions.view(
            1, 1, 1, query_tokens, 1
        )
        unnormalized = torch.where(
            visible, torch.exp(logits32 - running_max), 0.0
        )
        tile_mass = unnormalized.sum(dim=-1, keepdim=True) / denominator
        # Let PyTorch's native SDPA perform the value reduction for this view.
        # Five-dimensional batch geometry expresses GQA without repeat_kv:
        # [B,Hkv,G,Q,D] attends to [B,Hkv,1,K,D] through a stride-only view.
        segment_output = torch.nn.functional.scaled_dot_product_attention(
            grouped_query,
            key_tile.unsqueeze(2),
            value_tile.unsqueeze(2),
            attn_mask=visible,
            dropout_p=0.0,
            is_causal=False,
            scale=float(scaling),
        )
        accumulator = accumulator + segment_output.float() * tile_mass
        transient_peak = max(
            transient_peak,
            logits32.numel() * logits32.element_size()
            + unnormalized.numel() * unnormalized.element_size()
            + tile_mass.numel() * tile_mass.element_size()
            + segment_output.numel() * segment_output.element_size(),
        )

    # No selected K/V tensor is cast, packed, or copied by this consumer.
    if metrics is not None:
        metrics.transient_attention_bytes += int(transient_peak)
    if not bool(torch.all(torch.isfinite(accumulator))):
        raise RuntimeError("Sparse causal attention produced a non-finite numerator.")
    return accumulator.to(query.dtype).reshape(
        batch, query_heads, query_tokens, head_dim
    )

def _qwen_sparse_forward(original_forward):
    def forward(
        module,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        plural_cache = kwargs.get("past_key_values")
        effective_cache = plural_cache if plural_cache is not None else past_key_value
        if not isinstance(effective_cache, HFSparseDynamicCache):
            if "past_key_values" in kwargs:
                return original_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    cache_position=cache_position,
                    **kwargs,
                )
            return original_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
        kwargs.pop("past_key_values", None)
        past_key_value = effective_cache
        if cache_position is None:
            cache_position = kwargs.pop("position_ids", None)
            if cache_position is not None and cache_position.ndim == 2:
                if int(cache_position.shape[0]) != 1:
                    raise ValueError(
                        "Sparse HF Qwen attention currently supports one batch."
                    )
                cache_position = cache_position[0]
        if module.training:
            raise RuntimeError("Sparse HF Qwen attention is inference-only.")
        if kwargs.get("output_attentions"):
            raise RuntimeError("Sparse HF Qwen attention does not materialize attention weights.")
        if cache_position is None:
            raise ValueError("Sparse HF Qwen attention requires original cache_position values.")
        if getattr(module, "sliding_window", None) is not None:
            raise RuntimeError("Sparse HF Qwen attention currently supports full-attention layers only.")

        module_name = type(module).__module__
        if module_name == "transformers.models.qwen2.modeling_qwen2":
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        elif module_name == "transformers.models.qwen3.modeling_qwen3":
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        else:  # pragma: no cover - guarded during installation
            raise TypeError(f"Unsupported sparse HF attention module {type(module)!r}.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query = module.q_proj(hidden_states).view(hidden_shape)
        key = module.k_proj(hidden_states).view(hidden_shape)
        if hasattr(module, "q_norm"):
            query = module.q_norm(query)
        if hasattr(module, "k_norm"):
            key = module.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        segments = past_key_value.append(module.layer_idx, key, value, cache_position)
        sparse_layer = past_key_value.layers[module.layer_idx]
        output = segmented_qwen_attention(
            query,
            segments,
            cache_position,
            scaling=module.scaling,
            metrics=past_key_value.metrics,
            source_keys=sparse_layer.source_keys,
            source_values=sparse_layer.source_values,
            source_intervals=sparse_layer.source_intervals,
        )
        output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return module.o_proj(output), None

    return forward


def enable_qwen_sparse_live_kv(model) -> int:
    """Install a fail-closed sparse-cache consumer on Qwen2/Qwen3 attention.

    The supported boundary covers the Transformers 4.55--4.57 explicit
    ``cache_position`` contract and the 5.12 forwarded ``position_ids``
    contract. Qwen2 (including Qwen2.5) and Qwen3 full-attention modules are
    supported. Their configured dense attention backend is bypassed only for
    sparse-cache requests.
    Sliding-window layers, training, batching, attention-weight output, and
    Qwen3.5 recurrent/DeltaNet layers remain outside this correctness path.
    """

    supported_modules = {
        "transformers.models.qwen2.modeling_qwen2": "Qwen2Attention",
        "transformers.models.qwen3.modeling_qwen3": "Qwen3Attention",
    }
    installed = 0
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("HF sparse live K/V requires a Transformers model module tree.")
    for module in modules():
        module_name = type(module).__module__
        expected_name = supported_modules.get(module_name)
        if expected_name is None or type(module).__name__ != expected_name:
            continue
        if getattr(module, "_pra_sparse_live_kv_enabled", False):
            installed += 1
            continue
        parameters = inspect.signature(module.forward).parameters
        supports_explicit_position = "cache_position" in parameters
        supports_forwarded_position = (
            "past_key_values" in parameters
            and any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        if "position_embeddings" not in parameters or not (
            supports_explicit_position or supports_forwarded_position
        ):
            raise RuntimeError(
                "Installed Transformers Qwen attention lacks an explicit or forwarded "
                "position contract for sparse live K/V."
            )
        original = module.forward
        module.forward = MethodType(_qwen_sparse_forward(original), module)
        module._pra_sparse_live_kv_enabled = True
        module._pra_sparse_live_kv_original_forward = original
        installed += 1
    if not installed:
        raise TypeError(
            "HF non-packing live K/V supports Qwen2/Qwen2.5 and Qwen3 attention only."
        )
    return installed


def _pack_tensor_reference(tensor, plan: LiveKVSelectionPlan):
    """Materialize a dense oracle; never use this as a PRA request cache."""

    import torch

    pieces = [tensor[..., row.start : row.end, :] for row in plan.intervals]
    if not pieces:
        return tensor[..., :0, :], False
    if len(pieces) == 1:
        return pieces[0], False
    return torch.cat(pieces, dim=-2), True


def pack_dynamic_cache_reference(source_cache: object, plan: LiveKVSelectionPlan):
    """Build the explicitly packed dense reference for qualification only."""

    layers = getattr(source_cache, "layers", None)
    if layers is None:
        raise TypeError("HF dense reference requires a Transformers cache with layers.")
    selected = copy.copy(source_cache)
    selected.layers = []
    copied = False
    packed_bytes = 0
    for source_layer in layers:
        keys, values, key_name, value_name = _layer_pair(source_layer)
        layer = copy.copy(source_layer)
        chosen_keys, key_copy = _pack_tensor_reference(keys, plan)
        chosen_values, value_copy = _pack_tensor_reference(values, plan)
        setattr(layer, key_name, chosen_keys)
        setattr(layer, value_name, chosen_values)
        selected.layers.append(layer)
        copied = copied or key_copy or value_copy
        if key_copy:
            packed_bytes += chosen_keys.numel() * chosen_keys.element_size()
        if value_copy:
            packed_bytes += chosen_values.numel() * chosen_values.element_size()
    return HFResidentKVSelection(
        selected,
        plan,
        copied,
        interval_pack_bytes=int(packed_bytes),
    )


def pack_segmented_dynamic_cache_reference(
    source_cache: object, plan: LiveKVSelectionPlan
) -> HFResidentKVSelection:
    """Pack oracle values but preserve segment boundaries and source positions.

    Qualification must compare alias-backed and copied K/V with the identical
    attention consumer.  Otherwise backend-specific reduction order is
    incorrectly attributed to selection.  This oracle is never a PRA path.
    """

    packed = pack_dynamic_cache_reference(source_cache, plan)
    selected_layers: list[_HFSparseKVLayer] = []
    compact_source_intervals = []
    compact_cursor = 0
    for interval in plan.intervals:
        width = interval.end - interval.start
        compact_source_intervals.append(
            (compact_cursor, compact_cursor + width, interval.start)
        )
        compact_cursor += width
    for packed_layer in packed.cache.layers:
        keys, values, _key_name, _value_name = _layer_pair(packed_layer)
        cursor = 0
        segments = []
        for interval in plan.intervals:
            width = interval.end - interval.start
            segments.append(
                HFSparseKVSegment(
                    keys[..., cursor : cursor + width, :],
                    values[..., cursor : cursor + width, :],
                    interval.start,
                    interval.end,
                    interval.record_id,
                    interval.causal_group_id,
                )
            )
            cursor += width
        selected_layers.append(
            _HFSparseKVLayer(
                tuple(segments),
                source_keys=keys,
                source_values=values,
                source_intervals=tuple(compact_source_intervals),
            )
        )
    return HFResidentKVSelection(
        HFSparseDynamicCache(selected_layers, plan),
        plan,
        packed.physical_kv_copy,
        interval_pack_bytes=packed.interval_pack_bytes,
    )


def dense_reference_attention_mask(
    plan: LiveKVSelectionPlan,
    query_positions,
    *,
    dtype,
    device=None,
):
    """Create the dense-oracle mask for compact K/V with original positions."""

    import torch

    if query_positions.ndim == 2:
        if int(query_positions.shape[0]) != 1:
            raise ValueError("HF dense sparse-reference masks support one batch.")
        query_positions = query_positions[0]
    prior_tail_tokens = max(int(query_positions[0]) - plan.source_position_base, 0)
    positions: list[int] = []
    for interval in plan.intervals:
        positions.extend(range(interval.start, interval.end))
    positions.extend(
        range(
            plan.source_position_base,
            plan.source_position_base + prior_tail_tokens + int(query_positions.shape[0]),
        )
    )
    key_positions = torch.tensor(positions, dtype=query_positions.dtype, device=device)
    visible = key_positions.view(1, 1, 1, -1) <= query_positions.to(device).view(1, 1, -1, 1)
    mask = torch.zeros(visible.shape, dtype=dtype, device=device)
    return mask.masked_fill(~visible, torch.finfo(dtype).min)


def select_dynamic_cache(
    source_cache: object, plan: LiveKVSelectionPlan
) -> HFResidentKVSelection:
    """Select source K/V as disjoint views for the Qwen sparse consumer."""

    layers = getattr(source_cache, "layers", None)
    if layers is None:
        raise TypeError("Live HF K/V selection requires a Transformers cache with layers.")
    selected_layers: list[_HFSparseKVLayer] = []
    for source_layer in layers:
        keys, values, _key_name, _value_name = _layer_pair(source_layer)
        if int(keys.shape[-2]) < plan.source_tokens:
            raise ValueError("HF source cache is shorter than the selection plan.")
        selected_layers.append(
            _HFSparseKVLayer(
                _source_segment_views(keys, values, plan),
                source_keys=keys,
                source_values=values,
                source_intervals=tuple(
                    (interval.start, interval.end, interval.start)
                    for interval in plan.intervals
                ),
            )
        )
    selected = HFSparseDynamicCache(selected_layers, plan)
    return HFResidentKVSelection(selected, plan, False, interval_pack_bytes=0)


def select_full_dynamic_cache_noop(
    source_cache: object, plan: LiveKVSelectionPlan
) -> HFResidentKVSelection:
    """Return the canonical dense cache unchanged for a 100% owner continuation.

    This is the production PRA-100 semantic no-op. Unlike
    :func:`select_dynamic_cache`, it must only be used by the canonical cache
    owner because ordinary Transformers decode appends K/V in place. A request
    borrowing an immutable source still needs a request-local descriptor.
    """

    if not plan.full_retention:
        raise ValueError("Dense HF cache continuation requires full retention.")
    layers = getattr(source_cache, "layers", None)
    if layers is None:
        raise TypeError("Dense HF cache continuation requires a cache with layers.")
    observed_layers = 0
    for source_layer in layers:
        keys, values, _key_name, _value_name = _layer_pair(source_layer)
        if int(keys.shape[-2]) != plan.source_tokens:
            raise ValueError(
                "Dense HF cache length disagrees with the full-retention plan."
            )
        if int(values.shape[-2]) != plan.source_tokens:
            raise ValueError(
                "Dense HF value-cache length disagrees with the full-retention plan."
            )
        observed_layers += 1
    if plan.source_tokens and observed_layers == 0:
        raise ValueError("Dense HF cache has no materialized layers.")
    return HFResidentKVSelection(
        source_cache,
        plan,
        False,
        selected_text_reencoded_tokens=0,
        interval_pack_bytes=0,
    )
