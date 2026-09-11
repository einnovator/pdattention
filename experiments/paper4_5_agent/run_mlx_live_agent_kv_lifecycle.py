"""Qualify real MLX sparse-cache inference under request-owned K/V lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from pra_hf.live_history import LiveKVSelectionPlan
from pra_mlx.mlx_live_kv import MLXLiveKVRequestCancelled, MLXLiveKVRuntime
from pra_mlx.native import (
    MLXNativeMemory,
    capture_live_native_memory,
    deserialize_native_memory,
    serialize_native_memory,
)

from .run_mlx_agent_cache_equivalence import _assistant_prompts, _max_delta
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
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turn)
    messages = trajectory["messages"]
    assistant_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turn]
    assistant_index = assistant_indexes[-1]
    prompt_ids = prompts[-1]
    tail_tokens = min(args.wire_tail_tokens, max(1, len(prompt_ids) // 4))
    source_ids = prompt_ids[:-tail_tokens]
    wire_tail = prompt_ids[-tail_tokens:]

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

    def begin(request_id: str, *, generation: int = 1):
        return runtime.begin_request(
            request_id,
            identities["source_id"],
            plan,
            tenant_id=identities["tenant_id"],
            session_id=identities["session_id"],
            expected_generation=generation,
        )

    candidate = begin("candidate")
    reference = begin("reference")
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
        "same_subset_logit_exact": _max_delta(candidate_logits, reference_logits) == 0.0,
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
        "physical_kv_copy_reported": candidate_result.physical_kv_copy == (len(plan.intervals) > 1),
    }
    result = {
        "schema_version": "paper4.5.mlx-live-kv-lifecycle.v1",
        "probe": "mlx_real_model_request_owned_sparse_kv",
        "engine": "mlx-lm",
        "model": args.model,
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "hardware": args.hardware_label,
        "trajectory": str(args.trajectory),
        "turn": args.turn,
        "retention_fraction": args.retention_fraction,
        "source_tokens": len(source_ids),
        "wire_suffix_tokens": len(wire_tail),
        "selected_kv_tokens": plan.selected_tokens,
        "realized_retention_fraction": plan.selected_tokens / max(len(source_ids), 1),
        "source_position_base": plan.source_position_base,
        "has_holes": plan.has_holes,
        "physical_kv_copy": candidate_result.physical_kv_copy,
        "selected_text_reencoded_tokens": candidate_result.selected_text_reencoded_tokens,
        "offloaded_payload_type": type(offloaded).__name__,
        "offloaded_payload_bytes": len(offloaded) if isinstance(offloaded, bytes) else None,
        "max_abs_logit_delta_same_subset": _max_delta(candidate_logits, reference_logits),
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
