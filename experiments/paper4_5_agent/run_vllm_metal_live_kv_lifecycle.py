"""Qualify request-owned sparse live K/V on matched vLLM-Metal 0.29."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np

from experiments.paper4_5_agent.run_hf_agent_cache_equivalence import (
    _assistant_prompts,
)
from experiments.paper4_5_agent.run_vllm_live_agent_kv_gate import (
    _distribution_provenance,
    _qualify_packaged_runtime,
)
from experiments.paper4_5_agent.sparse_gate_common import sparse_causal_plan
from experiments.paper6_vllm.run_matched_e0_e2 import _run
from pra_hf.live_history import LiveKVSelectionPlan


def _memory_fingerprint(memory) -> str:
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


def _enqueue(llm, tokens: list[int], sampling, cache_salt: str) -> str:
    [request_id] = llm.enqueue(
        {"prompt_token_ids": tokens, "cache_salt": cache_salt},
        sampling,
        use_tqdm=False,
    )
    return str(request_id)


def _output_by_id(outputs) -> dict[str, object]:
    return {str(output.request_id): output for output in outputs}


def _lookup_output(outputs: dict[str, object], request_id: str) -> object:
    """Resolve the UUID-suffixed enqueue ID used by offline vLLM V1."""

    return outputs.get(request_id) or outputs[request_id.split("-", 1)[0]]


def _token_ids(output) -> list[int]:
    return list(map(int, output.outputs[0].token_ids))


def _drain_aborted_engine(llm, *, max_steps: int = 8) -> int:
    """Let V1 reconcile an abort that was submitted between engine steps."""

    steps = 0
    while llm.llm_engine.has_unfinished_requests() and steps < max_steps:
        llm.llm_engine.step()
        steps += 1
    return steps


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import mlx.core as mx
    import vllm
    import vllm_metal
    from pra_vllm.metal_live_kv import VLLMMetalLiveKVRuntime
    from pra_vllm.v1_native import (
        VLLMMetalV1NativeBridge,
        capture_paged_memory,
        native_request_cache_salt,
    )
    from vllm import LLM, SamplingParams

    vllm_provenance = _distribution_provenance("vllm")
    metal_provenance = _distribution_provenance("vllm-metal")
    metal_version = importlib.metadata.version("vllm-metal")
    exact_packaged_runtime, packaging_failures = _qualify_packaged_runtime(
        vllm_version=getattr(vllm, "__version__", "unknown"),
        metal_version=metal_version,
        source_revision=args.engine_source_revision,
        vllm_provenance=vllm_provenance,
        metal_provenance=metal_provenance,
        artifact_overlay=args.artifact_overlay,
    )
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=args.reserve_blocks)
    engine_core = llm.llm_engine.engine_core.engine_core
    block_pool = engine_core.scheduler.kv_cache_manager.block_pool
    runtime = VLLMMetalLiveKVRuntime(bridge, block_pool)
    tokenizer = llm.get_tokenizer()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    prompts = _assistant_prompts(tokenizer, trajectory, args.turn)
    messages = trajectory["messages"]
    assistant_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turn]
    assistant_index = assistant_indexes[-1]
    prompt = prompts[-1]
    page_tokens = (len(prompt) // bridge.block_size) * bridge.block_size
    if page_tokens == 0 or page_tokens == len(prompt):
        raise RuntimeError("Agent prompt did not provide a page source plus suffix.")
    source_tokens = prompt[:page_tokens]
    suffix = prompt[page_tokens:]
    plan = sparse_causal_plan(
        tokenizer,
        messages[:assistant_index],
        prompt,
        source_tokens=page_tokens,
        retention_fraction=args.retention_fraction,
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.continuation_tokens)
    prime = SamplingParams(temperature=0, max_tokens=1)
    salt = "paper45-vllm-metal-lifecycle-task02-source"
    observation_start = len(bridge.prefill_page_observations())
    _run(llm, bridge, prime, source_tokens, cache_salt=salt)
    observations = bridge.prefill_page_observations()[observation_start:]
    fresh = [row for row in observations if row["scheduler_cache_start"] == 0]
    if not fresh:
        raise RuntimeError("vLLM-Metal did not expose source prefill pages.")
    page_count = page_tokens // bridge.block_size
    source_blocks = tuple(fresh[0]["block_ids_by_group"][0][:page_count])
    canonical_memory = capture_paged_memory(bridge, source_blocks, page_tokens)
    canonical_fingerprint = _memory_fingerprint(canonical_memory)
    del canonical_memory

    identities = {
        "source_id": "task02-agent-history",
        "tenant_id": "paper4.5",
        "session_id": "task02",
        "generation": 1,
    }
    runtime.register_source(
        identities["source_id"],
        source_blocks,
        source_tokens=page_tokens,
        tenant_id=identities["tenant_id"],
        session_id=identities["session_id"],
        generation=identities["generation"],
    )
    selected_page_indices = runtime.selected_page_indices(plan, bridge.block_size)
    selected_blocks = tuple(source_blocks[index] for index in selected_page_indices)
    selected_kv_tokens = len(selected_blocks) * bridge.block_size
    request_salt = native_request_cache_salt(
        (identities["source_id"],),
        selected_token_count=selected_kv_tokens,
        source_position_base=plan.source_position_base,
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

    copy_events_before_alias = bridge._materialize_copy_events
    memory_before_alias = int(mx.get_active_memory())
    candidate_id = _enqueue(llm, suffix, sampling, request_salt)
    reference_id = _enqueue(llm, suffix, sampling, request_salt)
    candidate = begin(candidate_id)
    reference = begin(reference_id)
    mx.synchronize()
    memory_after_alias = int(mx.get_active_memory())
    alias_copy_events = bridge._materialize_copy_events - copy_events_before_alias
    two_borrowers = runtime.registry.view(identities["source_id"]).active_request_ids
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        two_borrower_offload_error = str(exc)
    else:
        two_borrower_offload_error = ""
    try:
        concurrent_outputs = llm.wait_for_completion(use_tqdm=False)
    except BaseException:
        candidate.fail()
        reference.fail()
        raise
    by_id = _output_by_id(concurrent_outputs)
    candidate_output = _lookup_output(by_id, candidate_id)
    reference_output = _lookup_output(by_id, reference_id)
    candidate.finish()
    one_borrower = runtime.registry.view(identities["source_id"]).active_request_ids
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        one_borrower_offload_error = str(exc)
    else:
        one_borrower_offload_error = ""
    reference.finish()

    control_id = _enqueue(llm, suffix, sampling, request_salt)
    pre_restore_control = begin(control_id)
    control_output = pre_restore_control.execute(
        lambda: _lookup_output(
            _output_by_id(llm.wait_for_completion(use_tqdm=False)), control_id
        )
    )

    def abort_engine_requests(request_ids: list[str]) -> None:
        llm.llm_engine.abort_request(
            [request_id.split("-", 1)[0] for request_id in request_ids]
        )

    cancel_id = _enqueue(llm, suffix, sampling, request_salt)
    cancelled = begin(cancel_id)
    cancel_registered = cancel_id in bridge.registry.active_request_ids()
    runtime.cancel_request(
        cancel_id, abort_requests=abort_engine_requests
    )
    cancellation_drain_steps = _drain_aborted_engine(llm)
    cancellation_left_engine_idle = not llm.llm_engine.has_unfinished_requests()

    error_id = _enqueue(llm, suffix, sampling, request_salt)
    errored = begin(error_id)
    original_step = llm.llm_engine.step

    def fail_engine_step():
        raise RuntimeError("injected vLLM-Metal engine-step failure")

    llm.llm_engine.step = fail_engine_step
    try:
        errored.execute(llm.llm_engine.step)
    except RuntimeError as exc:
        engine_error_observed = "engine-step failure" in str(exc)
    else:
        engine_error_observed = False
    finally:
        llm.llm_engine.step = original_step
        abort_engine_requests([error_id])
    error_drain_steps = _drain_aborted_engine(llm)
    error_left_engine_idle = not llm.llm_engine.has_unfinished_requests()

    try:
        begin("stale-not-enqueued", generation=0)
    except RuntimeError as exc:
        stale_error = str(exc)
    else:
        stale_error = ""

    offloaded = runtime.offload_source(identities["source_id"])
    offloaded_view = runtime.registry.view(identities["source_id"])
    restored_id = _enqueue(llm, suffix, sampling, request_salt)
    restored = begin(restored_id)
    restored_source = runtime.hot_source(identities["source_id"])
    if restored_source is None:
        raise RuntimeError("vLLM-Metal restore did not produce a hot source.")
    restored_memory = capture_paged_memory(
        bridge, restored_source.block_ids, restored_source.source_tokens
    )
    restored_fingerprint = _memory_fingerprint(restored_memory)
    del restored_memory
    restored_view = runtime.registry.view(identities["source_id"])
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        restored_borrower_offload_error = str(exc)
    else:
        restored_borrower_offload_error = ""
    restored_output = restored.execute(
        lambda: _lookup_output(
            _output_by_id(llm.wait_for_completion(use_tqdm=False)), restored_id
        )
    )
    second_offload = runtime.offload_source(identities["source_id"])
    second_offloaded_view = runtime.registry.view(identities["source_id"])

    termination_id = _enqueue(llm, suffix, sampling, request_salt)
    active_at_termination = begin(termination_id)
    removed = runtime.terminate_session(
        identities["tenant_id"],
        identities["session_id"],
        abort_requests=abort_engine_requests,
    )
    termination_drain_steps = _drain_aborted_engine(llm)
    termination_left_engine_idle = not llm.llm_engine.has_unfinished_requests()
    try:
        runtime.register_source(
            "replacement",
            source_blocks,
            source_tokens=page_tokens,
            tenant_id=identities["tenant_id"],
            session_id=identities["session_id"],
            generation=2,
        )
    except RuntimeError as exc:
        tombstone_error = str(exc)
    else:
        tombstone_error = ""

    candidate_tokens = _token_ids(candidate_output)
    reference_tokens = _token_ids(reference_output)
    control_tokens = _token_ids(control_output)
    restored_tokens = _token_ids(restored_output)
    checks = {
        "exact_packaged_runtime": exact_packaged_runtime,
        "sparse_disjoint_pages": (
            plan.has_holes and len(selected_page_indices) < len(source_blocks)
        ),
        "original_positions_preserved": plan.source_position_base == page_tokens,
        "selected_pages_alias_original_source_ids": (
            candidate.selection.block_ids == selected_blocks
            and reference.selection.block_ids == selected_blocks
        ),
        "initial_request_attachment_zero_copy": (
            not candidate.selection.attachment_physical_kv_copy
            and not reference.selection.attachment_physical_kv_copy
            and alias_copy_events == 0
            and memory_after_alias == memory_before_alias
        ),
        "zero_selected_history_reencoding": all(
            request.selection.selected_text_reencoded_tokens == 0
            for request in (candidate, reference, restored)
        ),
        "two_concurrent_borrowers": len(two_borrowers) == 2,
        "offload_rejected_with_two_borrowers": (
            "while requests borrow" in two_borrower_offload_error
        ),
        "same_subset_token_exact": candidate_tokens == reference_tokens,
        "normal_finish_released_exactly_once": (
            candidate.outcome == "finished"
            and candidate.finish() is False
            and len(one_borrower) == 1
        ),
        "offload_rejected_with_one_borrower": (
            "while requests borrow" in one_borrower_offload_error
        ),
        "real_engine_cancel_released_exactly_once": (
            cancel_registered
            and cancellation_left_engine_idle
            and cancelled.outcome == "cancelled"
            and cancelled.cancel() is False
        ),
        "engine_error_released_exactly_once": (
            engine_error_observed
            and error_left_engine_idle
            and errored.outcome == "error"
            and errored.fail() is False
        ),
        "stale_generation_rejected": "Stale" in stale_error,
        "idle_offload_reached_non_hot_tier": offloaded_view.tier == "offloaded",
        "restore_reached_hot_tier": restored_view.tier == "hot",
        "restored_source_fingerprint_exact": restored_fingerprint == canonical_fingerprint,
        "restored_request_attachment_zero_copy": (
            restored.selection.source_restored_with_physical_copy
            and not restored.selection.attachment_physical_kv_copy
        ),
        "offload_rejected_while_restored_source_borrowed": (
            "while requests borrow" in restored_borrower_offload_error
        ),
        "restored_generation_token_exact": control_tokens == restored_tokens,
        "restored_source_can_be_offloaded_again": (
            second_offloaded_view.tier == "offloaded"
            and isinstance(second_offload.payload, bytes)
        ),
        "termination_cancelled_active_request": (
            active_at_termination.outcome == "cancelled"
            and active_at_termination.cancel() is False
            and termination_left_engine_idle
        ),
        "termination_removed_source": (
            removed == 1 and runtime.registry.view(identities["source_id"]) is None
        ),
        "termination_tombstone_rejected_recreation": "terminated" in tombstone_error,
    }
    result = {
        "schema_version": "paper4.5.vllm-metal-live-kv-lifecycle.v1",
        "probe": "vllm_metal_real_request_owned_sparse_kv",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "engine_source_revision": args.engine_source_revision,
        "vllm_metal_distribution_version": metal_version,
        "vllm_metal_module": str(Path(vllm_metal.__file__).resolve()),
        "exact_packaged_runtime": exact_packaged_runtime,
        "packaging_qualification_failures": packaging_failures,
        "package_provenance": {
            "vllm": vllm_provenance,
            "vllm_metal": metal_provenance,
        },
        "python_version": platform.python_version(),
        "hardware": args.hardware_label,
        "model": args.model,
        "trajectory": str(args.trajectory),
        "turn": args.turn,
        "retention_fraction": args.retention_fraction,
        "source_tokens": page_tokens,
        "wire_suffix_tokens": len(suffix),
        "selected_kv_tokens": selected_kv_tokens,
        "realized_page_retention_fraction": selected_kv_tokens / page_tokens,
        "source_position_base": plan.source_position_base,
        "has_holes": plan.has_holes,
        "selected_page_indices": list(selected_page_indices),
        "source_block_ids": list(source_blocks),
        "selected_block_ids": list(selected_blocks),
        "selected_text_reencoded_tokens": 0,
        "initial_attachment_physical_kv_copy": False,
        "restore_materialization_physical_kv_copy": True,
        "physical_kv_copy_semantics": (
            "Initial and per-request attachment reuse the exact scheduler or reserve "
            "page IDs without copying. Lossless offload and subsequent restoration "
            "copy K/V once across the residency boundary."
        ),
        "active_memory_before_two_aliases_bytes": memory_before_alias,
        "active_memory_after_two_aliases_bytes": memory_after_alias,
        "alias_active_memory_delta_bytes": memory_after_alias - memory_before_alias,
        "alias_materialize_copy_events": alias_copy_events,
        "cancellation_drain_steps": cancellation_drain_steps,
        "error_drain_steps": error_drain_steps,
        "termination_drain_steps": termination_drain_steps,
        "offloaded_payload_bytes": len(offloaded.payload),
        "second_offloaded_payload_bytes": len(second_offload.payload),
        "canonical_source_fingerprint": canonical_fingerprint,
        "restored_source_fingerprint": restored_fingerprint,
        "candidate_token_ids": candidate_tokens,
        "reference_token_ids": reference_tokens,
        "pre_restore_single_request_token_ids": control_tokens,
        "restored_token_ids": restored_tokens,
        "selection_plan": plan.to_dict(),
        "checks": checks,
        "runtime_snapshot": runtime.snapshot(),
        "bridge_capabilities": bridge.capabilities(),
        "cancellation_boundary": "real vLLM abort_request before first synchronous Metal step",
        "error_boundary": "injected engine-step dispatch exception after request registration",
        "engine_lifecycle_qualified": all(checks.values()),
        "qualification_blockers": []
        if all(checks.values())
        else [name for name, passed in checks.items() if not passed],
    }
    bridge.close()
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
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--reserve-blocks", type=int, default=256)
    parser.add_argument("--engine-source-revision", default="unknown")
    parser.add_argument("--artifact-overlay", action="store_true")
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
