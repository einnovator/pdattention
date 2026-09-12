"""Qualify real MLX sparse-cache inference under request-owned K/V lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from pra_hf.live_history import LiveKVSelectionPlan
from pra_hf.agent_executor import split_generation_prompt
from pra_mlx.mlx_live_kv import MLXLiveKVRequestCancelled, MLXLiveKVRuntime
from pra_mlx.qwen3_segmented import install_qwen3_segmented_attention
from pra_mlx.native import (
    MLXDisjointLayerKV,
    MLXDisjointNativeMemory,
    MLXNativeLayerKV,
    MLXNativeMemory,
    MLXResidentKVSelection,
    capture_live_native_memory,
    deserialize_native_memory,
    disjoint_segmented_selected_attention,
    serialize_native_memory,
)

from .run_mlx_agent_cache_equivalence import _max_delta
from .sparse_gate_common import sparse_causal_plan


def _fingerprint(memory: MLXNativeMemory) -> str:
    """Hash logical K/V values independent of the persistence container."""

    import mlx.core as mx

    digest = hashlib.sha256()
    digest.update(str(memory.source_tokens).encode("ascii"))
    for index, layer in enumerate(memory.layers):
        for name, value in (("k", layer.keys), ("v", layer.values)):
            digest.update(f"{index}:{name}:{value.dtype}:{value.shape}".encode("ascii"))
            try:
                host = np.asarray(value)
            except (TypeError, ValueError, RuntimeError):
                host = np.asarray(value.astype(mx.float32))
            digest.update(host.tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(args.model)
    use_disjoint = args.materialization_policy == "disjoint_segmented"
    patched_layers = (
        install_qwen3_segmented_attention(model, compiled=False)
        if use_disjoint
        else 0
    )
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    assistant_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turn]
    assistant_index = assistant_indexes[-1]
    prompt_ids, source_ids, wire_tail = split_generation_prompt(
        tokenizer,
        messages[:assistant_index],
    )

    source_cache = make_prompt_cache(model)
    source_logits = model(mx.array([source_ids], dtype=mx.int32), cache=source_cache)
    mx.eval(source_logits)
    canonical = capture_live_native_memory(
        source_cache, LiveKVSelectionPlan.full(len(source_ids))
    ).memory
    plan = sparse_causal_plan(
        tokenizer,
        messages[:assistant_index],
        prompt_ids,
        source_tokens=len(source_ids),
        retention_fraction=args.retention_fraction,
    )
    source_fingerprint = _fingerprint(canonical)
    restored_fingerprints: list[str] = []

    def load_memory(payload: object) -> MLXNativeMemory:
        if not isinstance(payload, bytes):
            raise TypeError("MLX lifecycle offload payload must be bytes.")
        memory = deserialize_native_memory(payload)
        restored_fingerprints.append(_fingerprint(memory))
        return memory

    runtime = MLXLiveKVRuntime(
        dump=lambda memory: serialize_native_memory(memory, quantization="none"),
        load=load_memory,
    )
    identities = {
        "source_id": "task02-agent-history",
        "tenant_id": "paper4.5",
        "session_id": "task02",
        "generation": 1,
    }
    runtime.register_source(
        identities["source_id"],
        canonical,
        tenant_id=identities["tenant_id"],
        session_id=identities["session_id"],
        generation=identities["generation"],
    )

    def begin(
        request_id: str,
        *,
        generation: int = 1,
        segmented: bool | None = None,
        disjoint_selection: bool | None = None,
    ):
        return runtime.begin_request(
            request_id,
            identities["source_id"],
            plan,
            tenant_id=identities["tenant_id"],
            session_id=identities["session_id"],
            expected_generation=generation,
            segmented=use_disjoint if segmented is None else segmented,
            disjoint_selection=disjoint_selection,
        )

    active_before_selection = int(getattr(mx, "get_active_memory", lambda: 0)())
    reset_peak = getattr(mx, "reset_peak_memory", None)
    if reset_peak is not None:
        reset_peak()
    candidate_selection_start_ns = time.perf_counter_ns()
    candidate = begin("candidate")
    selected_arrays = []
    for layer in candidate.selection.memory.layers:
        segments = getattr(layer, "segments", (layer,))
        selected_arrays.extend(
            value
            for segment in segments
            for value in (segment.keys, segment.values)
        )
    mx.eval(*selected_arrays)
    candidate_selection_elapsed_ms = (
        time.perf_counter_ns() - candidate_selection_start_ns
    ) / 1_000_000
    active_after_selection = int(getattr(mx, "get_active_memory", lambda: 0)())
    peak_after_selection = int(getattr(mx, "get_peak_memory", lambda: 0)())
    disjoint_attention_allocation = None
    if use_disjoint:
        first_layer = candidate.selection.memory.layers[0]
        first_segment = first_layer.segments[0]
        attention = model.layers[0].self_attn.inner
        head_dim = int(first_segment.keys.shape[-1])
        probe_queries = mx.zeros(
            (1, int(attention.n_heads), 1, head_dim),
            dtype=first_segment.keys.dtype,
        )
        probe_local_keys = mx.zeros(
            (1, int(attention.n_kv_heads), 1, head_dim),
            dtype=first_segment.keys.dtype,
        )
        probe_local_values = mx.zeros(
            (1, int(attention.n_kv_heads), 1, head_dim),
            dtype=first_segment.values.dtype,
        )
        mx.eval(probe_queries, probe_local_keys, probe_local_values)
        if reset_peak is not None:
            reset_peak()
        attention_active_before = int(getattr(mx, "get_active_memory", lambda: 0)())
        probe_output = disjoint_segmented_selected_attention(
            probe_queries,
            tuple(segment.keys for segment in first_layer.segments),
            tuple(segment.values for segment in first_layer.segments),
            probe_local_keys,
            probe_local_values,
            scale=float(attention.scale),
            source_keys=first_layer.source_keys,
            source_values=first_layer.source_values,
            source_intervals=first_layer.intervals,
        )
        mx.eval(probe_output)
        attention_active_after = int(getattr(mx, "get_active_memory", lambda: 0)())
        attention_peak = int(getattr(mx, "get_peak_memory", lambda: 0)())
        selected_layer_bytes = int(first_layer.nbytes)
        attention_peak_delta = max(attention_peak - attention_active_before, 0)
        attention_active_delta = max(
            attention_active_after - attention_active_before, 0
        )
        disjoint_attention_allocation = {
            "selected_layer_kv_bytes": selected_layer_bytes,
            "active_before_bytes": attention_active_before,
            "active_after_bytes": attention_active_after,
            "active_delta_bytes": attention_active_delta,
            "absolute_peak_bytes": attention_peak,
            "peak_delta_bytes": attention_peak_delta,
            "peak_delta_to_selected_layer_kv_ratio": (
                attention_peak_delta / selected_layer_bytes
                if selected_layer_bytes
                else None
            ),
            "full_selected_kv_sized_allocation_observed": (
                attention_peak_delta >= selected_layer_bytes * 0.9
            ),
        }
    # The reference consumes the identical selected K/V and original position
    # frame through the established packed path. It never re-encodes text.
    dense_reference_selection_start_ns = time.perf_counter_ns()
    reference = begin(
        "reference", segmented=use_disjoint, disjoint_selection=False
    )
    dense_reference_arrays = [
        value
        for layer in reference.selection.memory.layers
        for value in (layer.keys, layer.values)
    ]
    mx.eval(*dense_reference_arrays)
    if use_disjoint:
        # The packed oracle is allowed to copy, but selection correctness must
        # not be confounded with a different attention implementation. Present
        # the copied values to the same interval-addressed Metal consumer using
        # the candidate's exact logical segment boundaries.
        compact_intervals = []
        cursor = 0
        for interval in plan.intervals:
            width = interval.end - interval.start
            compact_intervals.append((cursor, cursor + width))
            cursor += width
        packed_layers = []
        for layer in reference.selection.memory.layers:
            segments = tuple(
                MLXNativeLayerKV(
                    layer.keys[:, :, start:end, :],
                    layer.values[:, :, start:end, :],
                )
                for start, end in compact_intervals
            )
            packed_layers.append(
                MLXDisjointLayerKV(
                    segments,
                    source_keys=layer.keys,
                    source_values=layer.values,
                    intervals=tuple(compact_intervals),
                )
            )
        reference.selection = MLXResidentKVSelection(
            MLXDisjointNativeMemory(tuple(packed_layers), plan.source_tokens),
            plan,
            physical_kv_copy=True,
        )
        reference.disjoint_selection = True
    dense_reference_selection_elapsed_ms = (
        time.perf_counter_ns() - dense_reference_selection_start_ns
    ) / 1_000_000
    two_borrowers = runtime.registry.view(identities["source_id"]).active_request_ids
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        two_borrower_offload_error = str(exc)
    else:
        two_borrower_offload_error = ""

    candidate_result = candidate.generate(
        model, wire_tail, max_new_tokens=args.continuation_tokens
    )
    one_borrower = runtime.registry.view(identities["source_id"]).active_request_ids
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        one_borrower_offload_error = str(exc)
    else:
        one_borrower_offload_error = ""
    reference_result = reference.generate(
        model, wire_tail, max_new_tokens=args.continuation_tokens
    )

    cancelled = begin("cancelled")
    try:
        cancelled.generate(
            model,
            wire_tail,
            max_new_tokens=args.continuation_tokens,
            cancelled=lambda: True,
        )
    except MLXLiveKVRequestCancelled:
        cancellation_observed = True
    else:
        cancellation_observed = False

    errored = begin("errored")
    try:
        errored.generate(model, [], max_new_tokens=1)
    except ValueError as exc:
        error_observed = "non-empty wire tail" in str(exc)
    else:
        error_observed = False

    try:
        begin("stale", generation=0)
    except RuntimeError as exc:
        stale_error = str(exc)
    else:
        stale_error = ""

    offloaded = runtime.offload_source(identities["source_id"])
    offloaded_view = runtime.registry.view(identities["source_id"])
    restored = begin("restored")
    restored_view = runtime.registry.view(identities["source_id"])
    restored_source_fingerprint = restored_fingerprints[-1]
    restored_result = restored.generate(
        model, wire_tail, max_new_tokens=args.continuation_tokens
    )

    active_at_termination = begin("terminated-active")
    removed = runtime.terminate_session(
        identities["tenant_id"], identities["session_id"]
    )
    try:
        runtime.register_source(
            "replacement",
            canonical,
            tenant_id=identities["tenant_id"],
            session_id=identities["session_id"],
            generation=2,
        )
    except RuntimeError as exc:
        tombstone_error = str(exc)
    else:
        tombstone_error = ""

    candidate_logits = list(candidate_result.step_logits)
    reference_logits = list(reference_result.step_logits)
    restored_logits = list(restored_result.step_logits)
    checks = {
        "two_concurrent_borrowers": two_borrowers == ("candidate", "reference"),
        "offload_rejected_with_two_borrowers": (
            "while requests borrow" in two_borrower_offload_error
        ),
        "normal_finish_released_exactly_once": (
            candidate.outcome == "finished"
            and candidate.finish() is False
            and one_borrower == ("reference",)
        ),
        "offload_rejected_with_one_borrower": "while requests borrow" in one_borrower_offload_error,
        "same_subset_token_exact": candidate_result.token_ids == reference_result.token_ids,
        "same_subset_logit_within_tolerance": (
            _max_delta(candidate_logits, reference_logits)
            <= args.max_abs_logit_delta
        ),
        "cooperative_cancel_released_exactly_once": (
            cancellation_observed
            and cancelled.outcome == "cancelled"
            and cancelled.cancel() is False
        ),
        "error_released_exactly_once": (
            error_observed and errored.outcome == "error" and errored.fail() is False
        ),
        "stale_generation_rejected": "Stale" in stale_error,
        "idle_offload_reached_non_hot_tier": offloaded_view.tier == "offloaded",
        "restore_reached_hot_tier": restored_view.tier == "hot",
        "restored_source_fingerprint_exact": restored_source_fingerprint == source_fingerprint,
        "restored_generation_token_exact": candidate_result.token_ids == restored_result.token_ids,
        "restored_generation_logit_exact": _max_delta(candidate_logits, restored_logits) == 0.0,
        "termination_cancelled_active_request": (
            active_at_termination.outcome == "cancelled"
            and active_at_termination.cancel() is False
        ),
        "termination_removed_source": (
            removed == 1 and runtime.registry.view(identities["source_id"]) is None
        ),
        "termination_tombstone_rejected_recreation": "terminated" in tombstone_error,
        "original_positions_preserved": candidate_result.source_position_base == len(source_ids),
        "zero_selected_history_reencoding": all(
            row.selected_text_reencoded_tokens == 0
            for row in (candidate_result, reference_result, restored_result)
        ),
        "physical_kv_copy_reported": candidate_result.physical_kv_copy == (
            None if use_disjoint else len(plan.intervals) > 1
        ),
        "disjoint_selection_did_not_allocate": (
            not use_disjoint
            or (
                active_after_selection == active_before_selection
                and peak_after_selection == 0
            )
        ),
        "disjoint_attention_has_no_full_selected_kv_sized_allocation": (
            not use_disjoint
            or not bool(
                disjoint_attention_allocation[
                    "full_selected_kv_sized_allocation_observed"
                ]
            )
        ),
        "selection_pack_bytes_reported": candidate_result.selection_pack_bytes == (
            candidate.selection.memory.nbytes
            if candidate_result.physical_kv_copy
            else 0
        ),
    }
    result = {
        "schema_version": "paper4.5.mlx-live-kv-lifecycle.v2",
        "probe": "mlx_real_model_request_owned_sparse_kv",
        "engine": "mlx-lm",
        "model": args.model,
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "hardware": args.hardware_label,
        "materialization_policy": args.materialization_policy,
        "segmented_attention_patched_layers": patched_layers,
        "trajectory": str(args.trajectory),
        "turn": args.turn,
        "retention_fraction": args.retention_fraction,
        "source_tokens": len(source_ids),
        "wire_suffix_tokens": len(wire_tail),
        "selected_kv_tokens": plan.selected_tokens,
        "realized_retention_fraction": plan.selected_tokens / max(len(source_ids), 1),
        "source_position_base": plan.source_position_base,
        "has_holes": plan.has_holes,
        "physical_kv_copy": (
            bool(
                disjoint_attention_allocation[
                    "full_selected_kv_sized_allocation_observed"
                ]
            )
            if use_disjoint
            else candidate_result.physical_kv_copy
        ),
        "runtime_physical_kv_copy_report": candidate_result.physical_kv_copy,
        "pra_interval_pack_copy": False if use_disjoint else candidate_result.physical_kv_copy,
        "physical_kv_copy_qualification": (
            "qualified_zero_copy_interval_addressed_metal"
            if use_disjoint
            else "measured_explicit_interval_pack"
        ),
        "consumer_implementation": (
            "fused_interval_addressed_metal" if use_disjoint else "mlx_lm_dense"
        ),
        "reference_condition": (
            "packed-value oracle with identical segment boundaries, original "
            "positions, and Metal consumer"
            if use_disjoint
            else "packed dense oracle"
        ),
        "selected_kv_segments": candidate_result.selected_kv_segments,
        "selection_pack_bytes": candidate_result.selection_pack_bytes,
        "selection_materialization_elapsed_ms": candidate_selection_elapsed_ms,
        "dense_reference_pack_bytes": reference.selection.memory.nbytes,
        "dense_reference_materialization_elapsed_ms": (
            dense_reference_selection_elapsed_ms
        ),
        "selection_active_memory_delta_bytes": max(
            active_after_selection - active_before_selection, 0
        ),
        "selection_peak_memory_bytes": peak_after_selection,
        "disjoint_attention_allocation": disjoint_attention_allocation,
        "allocation_measurement_method": {
            "selection": (
                "Reset the MLX peak allocator, construct the selected interval "
                "views, force every selected key/value array with mx.eval, and "
                "compare active and peak bytes with the pre-selection baseline."
            ),
            "consumption": (
                "Reset the MLX peak allocator, run one real first-layer attention "
                "query over all selected interval key/value views plus one local "
                "token, force the output with mx.eval, and compare peak delta "
                "against the selected first-layer K/V byte extent."
            ),
            "full_kv_sized_threshold": (
                "peak_delta_bytes >= 0.9 * selected_layer_kv_bytes"
            ),
        },
        "required_runtime_fix": None,
        "selected_text_reencoded_tokens": candidate_result.selected_text_reencoded_tokens,
        "offloaded_payload_type": type(offloaded).__name__,
        "offloaded_payload_bytes": len(offloaded) if isinstance(offloaded, bytes) else None,
        "max_abs_logit_delta_same_subset": _max_delta(candidate_logits, reference_logits),
        "same_subset_logit_exact": _max_delta(candidate_logits, reference_logits) == 0.0,
        "max_abs_logit_delta_after_restore": _max_delta(candidate_logits, restored_logits),
        "checks": checks,
        "runtime_snapshot": runtime.snapshot(),
        "selection_plan": plan.to_dict(),
        "engine_request_callback_integration": True,
        "engine_lifecycle_qualified": all(checks.values()),
        "qualification_blockers": []
        if all(checks.values())
        else [name for name, passed in checks.items() if not passed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--turn", type=int, default=4)
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--hardware-label", default="unspecified")
    parser.add_argument(
        "--materialization-policy",
        choices=("dense_pack", "disjoint_segmented"),
        default="dense_pack",
    )
    parser.add_argument("--max-abs-logit-delta", type=float, default=5e-3)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "engine": result["engine"],
                "model": result["model"],
                "engine_lifecycle_qualified": result["engine_lifecycle_qualified"],
                "qualification_blockers": result["qualification_blockers"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["engine_lifecycle_qualified"] else 1)


if __name__ == "__main__":
    main()
