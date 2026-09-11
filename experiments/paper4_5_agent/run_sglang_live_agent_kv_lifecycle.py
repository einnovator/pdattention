"""Qualify real SGLang-MLX Radix requests under sparse K/V lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from pra_hf.live_history import LiveKVSelectionPlan
from pra_mlx.native import MLXNativeMemory, capture_live_native_memory
from pra_sglang.mlx_native import (
    SGLangMLXLiveKVRuntime,
    SGLangMLXNativeBridge,
    SGLangSelectedKVCache,
)

from .run_hf_agent_cache_equivalence import _assistant_prompts
from .sparse_gate_common import sparse_causal_plan


def _fingerprint(memory: MLXNativeMemory) -> str:
    """Hash logical K/V values independently from the offload container."""

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


def _prefill(runner, request_id: str, wire_tail: list[int]) -> int:
    pending = runner.prefill_start(
        request_id, wire_tail, wire_tail, [], [], 0
    )
    runner.eval_pending(pending)
    return int(runner.prefill_finalize(pending))


def _decode(runner, request_ids: list[str]) -> list[int]:
    pending = runner.decode_batch_start(request_ids)
    runner.eval_pending(pending)
    return list(map(int, runner.decode_batch_finalize(pending)))


def _generate(
    runner,
    request_id: str,
    wire_tail: list[int],
    *,
    max_tokens: int,
) -> tuple[int, ...]:
    generated = [_prefill(runner, request_id, wire_tail)]
    while len(generated) < max_tokens:
        generated.extend(_decode(runner, [request_id]))
    return tuple(generated)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(args: argparse.Namespace) -> dict[str, object]:
    import mlx.core as mx
    import sglang
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    started = time.perf_counter()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    runner = MlxModelRunner(
        args.model,
        revision=args.revision,
        disable_radix_cache=False,
        enable_sampling=False,
    )
    runner.init_cache_pools(None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
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
    plan = sparse_causal_plan(
        tokenizer,
        messages[:assistant_index],
        prompt_ids,
        source_tokens=len(source_ids),
        retention_fraction=args.retention_fraction,
    )
    if not plan.has_holes or len(plan.intervals) < 2:
        raise RuntimeError("Lifecycle qualification requires a disjoint sparse plan.")

    prime_id = "task02-source-owner"
    pending = runner.prefill_start(prime_id, source_ids, source_ids, [], [], 0)
    runner.eval_pending(pending)
    runner.prefill_finalize(pending)
    source_caches = runner._req_caches[prime_id]
    canonical = capture_live_native_memory(
        source_caches, LiveKVSelectionPlan.full(len(source_ids))
    ).memory
    source_fingerprint = _fingerprint(canonical)

    bridge = SGLangMLXNativeBridge(runner)
    runtime = SGLangMLXLiveKVRuntime(bridge)
    identities = {
        "source_id": "task02-agent-history",
        "tenant_id": "paper4.5",
        "session_id": "task02",
        "generation": 1,
    }
    runtime.register_source(
        identities["source_id"],
        source_caches,
        owner_request_id=prime_id,
        tenant_id=identities["tenant_id"],
        session_id=identities["session_id"],
        generation=identities["generation"],
        source_tokens=len(source_ids),
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

    reference = begin("reference")
    reference_tokens = _generate(
        runner,
        reference.request_id,
        wire_tail,
        max_tokens=args.continuation_tokens,
    )
    reference_cache = runner._req_caches[reference.request_id][
        runner._cache_layout.first_attention_layer_index
    ]
    reference_offsets = {
        "scheduler_local": int(reference_cache.offset),
        "attention_rope": int(reference_cache.rope_offset),
        "position_base": int(reference_cache.position_base),
    }
    reference_selected_fingerprint = _fingerprint(reference.selection.memory)
    reference.finish()
    reference_second_finish = reference.finish()

    survivor = begin("survivor")
    cancelled = begin("cancelled")
    two_borrowers = runtime.registry.view(
        identities["source_id"]
    ).active_request_ids
    owner_remove_error = ""
    try:
        runner.remove_request(prime_id)
    except RuntimeError as exc:
        owner_remove_error = str(exc)
    offload_error = ""
    try:
        runtime.offload_source(identities["source_id"])
    except RuntimeError as exc:
        offload_error = str(exc)
    eviction_error = ""
    try:
        runtime.evict_source(identities["source_id"])
    except RuntimeError as exc:
        eviction_error = str(exc)

    concurrent_tokens = {
        survivor.request_id: [_prefill(runner, survivor.request_id, wire_tail)],
        cancelled.request_id: [_prefill(runner, cancelled.request_id, wire_tail)],
    }
    for _ in range(min(2, max(0, args.continuation_tokens - 1))):
        request_ids = [survivor.request_id, cancelled.request_id]
        values = _decode(runner, request_ids)
        for request_id, token in zip(request_ids, values):
            concurrent_tokens[request_id].append(token)
    cancelled.cancel()
    cancelled_second_cancel = cancelled.cancel()
    one_borrower = runtime.registry.view(
        identities["source_id"]
    ).active_request_ids
    while len(concurrent_tokens[survivor.request_id]) < args.continuation_tokens:
        concurrent_tokens[survivor.request_id].extend(
            _decode(runner, [survivor.request_id])
        )
    survivor_tokens = tuple(concurrent_tokens[survivor.request_id])
    survivor.finish()

    errored = begin("errored")
    _prefill(runner, errored.request_id, wire_tail)
    native_error = ""
    try:
        runner.decode_batch_start([errored.request_id, "missing-request"])
    except Exception as exc:
        native_error = f"{type(exc).__name__}: {exc}"
        errored.fail()
    else:
        errored.fail()
    errored_second_fail = errored.fail()

    stale_error = ""
    try:
        begin("stale", generation=0)
    except RuntimeError as exc:
        stale_error = str(exc)

    offloaded = runtime.offload_source(identities["source_id"])
    offloaded_view = runtime.registry.view(identities["source_id"])
    offloaded_bytes = (
        len(offloaded[1])
        if isinstance(offloaded, tuple)
        and len(offloaded) == 2
        and isinstance(offloaded[1], bytes)
        else None
    )
    offloaded_sha256 = (
        hashlib.sha256(offloaded[1]).hexdigest()
        if offloaded_bytes is not None
        else None
    )
    restored = begin("restored")
    restored_view = runtime.registry.view(identities["source_id"])
    restored_selected_fingerprint = _fingerprint(restored.selection.memory)
    restored_tokens = _generate(
        runner,
        restored.request_id,
        wire_tail,
        max_tokens=args.continuation_tokens,
    )
    restored.finish()

    terminated = begin("terminated-active")
    _prefill(runner, terminated.request_id, wire_tail)
    removed = runtime.terminate_session(
        identities["tenant_id"], identities["session_id"]
    )
    tombstone_error = ""
    try:
        runtime.register_source(
            "replacement",
            canonical,
            owner_request_id=prime_id,
            tenant_id=identities["tenant_id"],
            session_id=identities["session_id"],
            generation=2,
        )
    except RuntimeError as exc:
        tombstone_error = str(exc)

    runner.remove_request(prime_id)
    radix_pool_contains_selected_wrapper = any(
        isinstance(cache, SGLangSelectedKVCache)
        for cache_list in runner._cache_pool
        for cache in cache_list
    )
    final_snapshot = runtime.snapshot()
    bridge_capabilities = dict(bridge.capabilities())
    runtime.close()
    bridge.close()
    mx.clear_cache()

    checks = {
        "radix_cache_enabled": not runner.disable_radix_cache,
        "two_concurrent_borrowers": two_borrowers == ("cancelled", "survivor"),
        "source_owner_removal_rejected_while_borrowed": (
            "source owner" in owner_remove_error
        ),
        "offload_rejected_while_borrowed": "while requests borrow" in offload_error,
        "eviction_rejected_while_borrowed": "while requests borrow" in eviction_error,
        "native_batched_decode_executed": (
            len(concurrent_tokens[cancelled.request_id]) >= 2
        ),
        "normal_finish_released_exactly_once": (
            reference.outcome == "finished" and reference_second_finish is False
        ),
        "cancel_released_exactly_once": (
            cancelled.outcome == "cancelled"
            and cancelled_second_cancel is False
            and one_borrower == ("survivor",)
        ),
        "error_callback_released_exactly_once": (
            bool(native_error)
            and errored.outcome == "error"
            and errored_second_fail is False
        ),
        "concurrent_survivor_token_exact": survivor_tokens == reference_tokens,
        "stale_generation_rejected": "Stale" in stale_error,
        "idle_offload_reached_non_hot_tier": offloaded_view.tier == "offloaded",
        "restore_reached_hot_tier": restored_view.tier == "hot",
        "restored_selected_kv_fingerprint_exact": (
            restored_selected_fingerprint == reference_selected_fingerprint
        ),
        "restored_generation_token_exact": restored_tokens == reference_tokens,
        "termination_cancelled_active_request": (
            terminated.outcome == "cancelled" and terminated.cancel() is False
        ),
        "termination_removed_source": (
            removed == 1 and runtime.registry.view(identities["source_id"]) is None
        ),
        "termination_tombstone_rejected_recreation": (
            "terminated" in tombstone_error
        ),
        "original_positions_preserved": (
            reference_offsets["position_base"] == plan.source_position_base
            and reference_offsets["attention_rope"]
            == plan.source_position_base + reference_offsets["scheduler_local"]
        ),
        "zero_selected_history_reencoding": (
            reference.selection.selected_text_reencoded_tokens == 0
            and survivor.selection.selected_text_reencoded_tokens == 0
            and restored.selection.selected_text_reencoded_tokens == 0
        ),
        "physical_kv_copy_reported": (
            reference.selection.physical_kv_copy
            == (len(plan.intervals) > 1)
        ),
        "radix_pool_ownership_isolated": not radix_pool_contains_selected_wrapper,
    }
    result = {
        "schema_version": "paper4.5.sglang-mlx-live-kv-lifecycle.v1",
        "probe": "sglang_mlx_real_radix_request_owned_sparse_kv",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "engine_revision": provenance["sglang_source"]["revision"],
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "python_version": platform.python_version(),
        "hardware": args.hardware_label,
        "model": args.model,
        "model_revision": args.revision,
        "trajectory": str(args.trajectory),
        "turn": args.turn,
        "retention_fraction": args.retention_fraction,
        "source_tokens": len(source_ids),
        "wire_suffix_tokens": len(wire_tail),
        "selected_kv_tokens": plan.selected_tokens,
        "realized_retention_fraction": plan.selected_tokens / len(source_ids),
        "source_position_base": plan.source_position_base,
        "has_holes": plan.has_holes,
        "physical_kv_copy": reference.selection.physical_kv_copy,
        "selected_text_reencoded_tokens": (
            reference.selection.selected_text_reencoded_tokens
        ),
        "reference_token_ids": list(reference_tokens),
        "survivor_token_ids": list(survivor_tokens),
        "restored_token_ids": list(restored_tokens),
        "source_kv_sha256": source_fingerprint,
        "selected_kv_sha256": reference_selected_fingerprint,
        "restored_selected_kv_sha256": restored_selected_fingerprint,
        "offloaded_payload_type": type(offloaded).__name__,
        "offloaded_payload_bytes": offloaded_bytes,
        "offloaded_payload_sha256": offloaded_sha256,
        "native_error": native_error,
        "request_offsets": reference_offsets,
        "runtime_snapshot": final_snapshot,
        "bridge_capabilities": bridge_capabilities,
        "selection_plan": plan.to_dict(),
        "checks": checks,
        "experiment_revision": _git_revision(),
        "implementation_files": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "src/pra_sglang/mlx_native.py",
                "src/pra_sglang/native_executor.py",
                "experiments/paper4_5_agent/run_sglang_live_agent_kv_lifecycle.py",
            )
        },
        "provenance_artifact": str(args.provenance),
        "provenance_artifact_sha256": hashlib.sha256(
            args.provenance.read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "engine_request_callback_integration": True,
        "engine_lifecycle_qualified": all(checks.values()),
        "qualification_blockers": [
            name for name, passed in checks.items() if not passed
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--turn", type=int, default=4)
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--wire-tail-tokens", type=int, default=32)
    parser.add_argument("--continuation-tokens", type=int, default=8)
    parser.add_argument("--hardware-label", default="bigmac-m4pro-48gb")
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "engine": result["engine"],
                "model": result["model"],
                "engine_lifecycle_qualified": result["engine_lifecycle_qualified"],
                "qualification_blockers": result["qualification_blockers"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["engine_lifecycle_qualified"] else 1)


if __name__ == "__main__":
    main()
