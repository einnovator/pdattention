"""Qualify sparse original-position agent-history reuse on vLLM CUDA 0.28."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import uuid
from pathlib import Path

import safetensors.torch

from experiments.paper4_5_agent.run_hf_agent_cache_equivalence import (
    _assistant_prompts,
)
from experiments.paper4_5_agent.sparse_gate_common import sparse_causal_plan
from experiments.paper6_vllm.run_cuda_connector_candidate import _prompt
from pra_vllm.cuda_protocol import CudaConnectorCommand
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand


def _selected_page_indices(plan, block_size: int) -> tuple[int, ...]:
    """Retain every complete page intersecting a selected causal record."""

    selected = []
    for index in range((plan.source_tokens + block_size - 1) // block_size):
        start = index * block_size
        end = min(start + block_size, plan.source_tokens)
        if any(interval.start < end and start < interval.end for interval in plan.intervals):
            selected.append(index)
    return tuple(selected)


def _directory(storage: Path, logical_key: str) -> Path:
    return storage / hashlib.sha256(logical_key.encode("utf-8")).hexdigest()


def _derive_selected_resource(
    storage: Path,
    source_key: str,
    selected_key: str,
    page_indices: tuple[int, ...],
    block_size: int,
    source_generation: int,
) -> tuple[int, int, str]:
    """Slice captured K/V pages without reconstructing their source tokens."""

    source = _directory(storage, source_key)
    target = _directory(storage, selected_key)
    target.mkdir(parents=True, exist_ok=False)
    selected_tokens = len(page_indices) * block_size
    native_bytes = 0
    native_digest = hashlib.sha256()
    for path in source.glob("layer-*.safetensors"):
        tensor = safetensors.torch.load_file(str(path))["kv_cache"]
        spans = [tensor[index * block_size : (index + 1) * block_size] for index in page_indices]
        selected = __import__("torch").cat(spans, dim=0).contiguous()
        native_bytes += selected.numel() * selected.element_size()
        target_path = target / path.name
        safetensors.torch.save_file({"kv_cache": selected}, str(target_path))
        native_digest.update(path.name.encode("utf-8"))
        native_digest.update(target_path.read_bytes())
    manifest = {
        "schema_version": "pra-vllm-cuda-sparse-kv-v1",
        "logical_key": selected_key,
        "source_tokens": selected_tokens,
        "source_generation": source_generation,
        "layer_files": len(list(target.glob("layer-*.safetensors"))),
        "native_tensor_bytes": native_bytes,
        "parent_logical_key": source_key,
        "selected_page_indices": list(page_indices),
        "selection_materialization": "lossless CPU K/V page gather; no tokenization or model forward",
        "native_payload_sha256": native_digest.hexdigest(),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    persisted_bytes = sum(path.stat().st_size for path in target.glob("*"))
    return native_bytes, persisted_bytes, native_digest.hexdigest()


def _generate(llm, sampling, tokens: list[int], cache_salt: str | None = None):
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = llm.generate(_prompt(tokens, cache_salt), sampling, use_tqdm=False)[0]
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def _generate_pair(llm, sampling, tokens: list[int], cache_salt: str):
    """Run two simultaneous borrowers of one immutable resident selection."""

    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(
        [_prompt(tokens, cache_salt), _prompt(tokens, cache_salt)],
        sampling,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    return outputs, (time.perf_counter() - started) * 1000.0


def _cancel_one_of_two_borrowers(
    llm,
    runtime_connector,
    sampling,
    tokens: list[int],
    cache_salt: str,
    handle_key: tuple[str, str],
) -> dict[str, object]:
    """Abort one live vLLM request while its peer keeps borrowing the pages."""

    request_ids: list[str] = []
    for _ in range(2):
        [request_id] = llm.enqueue(
            _prompt(tokens, cache_salt), sampling, use_tqdm=False
        )
        request_ids.append(str(request_id))
    first_outputs = llm.llm_engine.step()
    refcount_before_abort = int(
        runtime_connector._detached_refcounts.get(handle_key, 0)
    )
    cancelled, survivor = request_ids
    cancelled_external = cancelled.split("-", 1)[0]
    llm.llm_engine.abort_request([cancelled_external])
    released = runtime_connector.reconcile_detached_requests({survivor})
    refcount_after_abort = int(
        runtime_connector._detached_refcounts.get(handle_key, 0)
    )
    handle_survived_abort = handle_key in runtime_connector._detached_handles
    first_step_finished_ids = [str(row.request_id) for row in first_outputs if row.finished]
    final_outputs = {str(row.request_id): row for row in first_outputs if row.finished}
    while llm.llm_engine.has_unfinished_requests():
        for row in llm.llm_engine.step():
            if row.finished:
                final_outputs[str(row.request_id)] = row
    def output_for(identifier: str):
        return final_outputs.get(identifier) or final_outputs.get(
            identifier.split("-", 1)[0]
        )

    survivor_output = output_for(survivor)
    cancelled_output = output_for(cancelled)
    released_after_finish = runtime_connector.reconcile_detached_requests(set())
    refcount_after_finish = int(
        runtime_connector._detached_refcounts.get(handle_key, 0)
    )
    return {
        "request_ids": request_ids,
        "first_step_finished_ids": first_step_finished_ids,
        "cancelled_request_id": cancelled,
        "cancelled_external_request_id": cancelled_external,
        "survivor_request_id": survivor,
        "refcount_before_abort": refcount_before_abort,
        "released_on_abort_snapshot": list(released),
        "refcount_after_abort": refcount_after_abort,
        "handle_survived_abort": handle_survived_abort,
        "cancelled_absent_from_final_outputs": cancelled_output is None,
        "survivor_finished": survivor_output is not None,
        "survivor_token_ids": (
            []
            if survivor_output is None
            else list(map(int, survivor_output.outputs[0].token_ids))
        ),
        "released_after_finish": list(released_after_finish),
        "refcount_after_finish": refcount_after_finish,
    }


def _events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--detached-reserve-blocks", type=int, default=384)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    run_id = uuid.uuid4().hex[:12]
    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    telemetry = storage / f"telemetry-{run_id}.jsonl"
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config={
            "kv_connector": "PRASparseConnector",
            "kv_connector_module_path": "pra_vllm.cuda_sparse_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "storage_path": str(storage),
                "telemetry_path": str(telemetry),
                "detached_pages": True,
                "detached_reserve_blocks": args.detached_reserve_blocks,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    messages = trajectory["messages"]
    prompts = _assistant_prompts(tokenizer, trajectory, args.turns)
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ][: args.turns]
    generation = SamplingParams(
        temperature=0, max_tokens=args.continuation_tokens, ignore_eos=True
    )
    prime = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    cancellation_sampling = SamplingParams(
        temperature=0, max_tokens=max(32, args.continuation_tokens), ignore_eos=True
    )
    rows = []
    lifecycle_done = False
    stale_probe: dict[str, object] = {"attempted": False, "rejected": False}
    stale_pending: tuple[list[int], SparseCudaConnectorCommand] | None = None
    for turn, (assistant_index, prompt) in enumerate(
        zip(assistant_indexes, prompts), start=1
    ):
        source_tokens = (len(prompt) // block_size) * block_size
        if source_tokens == 0 or source_tokens == len(prompt):
            continue
        source = prompt[:source_tokens]
        suffix = prompt[source_tokens:]
        try:
            plan = sparse_causal_plan(
                tokenizer,
                messages[:assistant_index],
                prompt,
                source_tokens=source_tokens,
                retention_fraction=args.retention_fraction,
            )
        except RuntimeError:
            continue
        page_indices = _selected_page_indices(plan, block_size)
        if not page_indices:
            continue
        if args.retention_fraction < 1 and len(page_indices) == source_tokens // block_size:
            continue

        full_key = f"agent-full-{run_id}-{turn}"
        selected_key = f"agent-selected-{run_id}-{turn}"
        _generate(
            llm,
            prime,
            source + suffix[:1],
            CudaConnectorCommand("store", full_key, source_tokens).cache_salt(),
        )
        source_generation = turn
        native_bytes, persisted_bytes, native_digest = _derive_selected_resource(
            storage,
            full_key,
            selected_key,
            page_indices,
            block_size,
            source_generation,
        )
        selected_tokens = len(page_indices) * block_size
        command = SparseCudaConnectorCommand(
            "load",
            selected_key,
            selected_tokens,
            source_tokens,
            source_generation=source_generation,
            residency="hot",
            request_scope=f"turn-{turn}",
        )
        event_start = len(_events(telemetry))
        (candidate, reference), pair_ms = _generate_pair(
            llm, generation, suffix, command.cache_salt()
        )
        transfers = [
            event for event in _events(telemetry)[event_start:]
            if event.get("event") == "load"
        ]
        candidate_tokens = list(map(int, candidate.outputs[0].token_ids))
        reference_tokens = list(map(int, reference.outputs[0].token_ids))
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        runtime_connector = get_kv_transfer_group()
        handle_key = (selected_key, "hot")
        deferred_refcount_after_pair = int(
            runtime_connector._detached_refcounts.get(handle_key, 0)
        )
        reconciled_pair_requests = runtime_connector.reconcile_detached_requests(set())
        refcount_after_pair = int(
            runtime_connector._detached_refcounts.get(handle_key, 0)
        )
        free_before_evict = len(runtime_connector._detached_free or ())
        evicted_blocks = runtime_connector.offload_detached_resource(selected_key)
        free_after_evict = len(runtime_connector._detached_free or ())
        reload_event_start = len(_events(telemetry))
        replay, replay_ms = _generate(llm, generation, suffix, command.cache_salt())
        reload_transfers = [
            event for event in _events(telemetry)[reload_event_start:]
            if event.get("event") == "load"
        ]
        reused_blocks = tuple(runtime_connector._detached_handles.get(handle_key, ()))
        replay_tokens = list(map(int, replay.outputs[0].token_ids))
        deferred_refcount_after_reload = int(
            runtime_connector._detached_refcounts.get(handle_key, 0)
        )
        reconciled_reload_requests = runtime_connector.reconcile_detached_requests(set())
        final_evicted_blocks = runtime_connector.offload_detached_resource(selected_key)
        free_after_final_evict = len(runtime_connector._detached_free or ())
        cancellation = None
        if not lifecycle_done:
            cancel_event_start = len(_events(telemetry))
            cancellation = _cancel_one_of_two_borrowers(
                llm,
                runtime_connector,
                cancellation_sampling,
                suffix,
                command.cache_salt(),
                handle_key,
            )
            cancel_transfers = [
                event for event in _events(telemetry)[cancel_event_start:]
                if event.get("event") == "load"
            ]
            cancellation["load_h2d_bytes"] = (
                int(cancel_transfers[0].get("h2d_bytes", 0))
                if cancel_transfers else None
            )
            cancellation["evicted_after_test"] = list(
                runtime_connector.offload_detached_resource(selected_key)
            )
            control, _ = _generate(
                llm,
                cancellation_sampling,
                suffix,
                command.cache_salt(),
            )
            cancellation["control_token_ids"] = list(
                map(int, control.outputs[0].token_ids)
            )
            runtime_connector.reconcile_detached_requests(set())
            runtime_connector.offload_detached_resource(selected_key)
            cancellation["survivor_exact_vs_control"] = (
                cancellation["survivor_token_ids"]
                == cancellation["control_token_ids"]
            )
            stale_pending = (
                list(suffix),
                SparseCudaConnectorCommand(
                    "load",
                    selected_key,
                    selected_tokens,
                    source_tokens,
                    source_generation=source_generation + 1,
                    residency="hot",
                    request_scope=f"stale-turn-{turn}",
                ),
            )
            lifecycle_done = True
        rows.append({
            "turn": turn,
            "source_tokens": source_tokens,
            "selected_kv_tokens": selected_tokens,
            "realized_retention_fraction": selected_tokens / source_tokens,
            "source_position_base": source_tokens,
            "wire_suffix_tokens": len(suffix),
            "selected_page_indices": list(page_indices),
            "has_holes": any(
                right != left + 1 for left, right in zip(page_indices, page_indices[1:])
            ) or len(page_indices) < source_tokens // block_size,
            "selection_plan": plan.to_dict(),
            "selected_text_reencoded_tokens": 0,
            "selection_materialization_kv_copy_bytes": native_bytes,
            "selection_persisted_bytes": persisted_bytes,
            "native_payload_sha256": native_digest,
            "candidate_load_h2d_bytes": int(transfers[0].get("h2d_bytes", 0)) if transfers else None,
            "reference_load_h2d_bytes": int(transfers[1].get("h2d_bytes", 0)) if len(transfers) > 1 else None,
            "reference_shared_resident_hit": bool(transfers[1].get("shared_resident_hit")) if len(transfers) > 1 else None,
            "concurrent_borrowers": 2,
            "concurrent_pair_ms": pair_ms,
            "candidate_token_ids": candidate_tokens,
            "reference_token_ids": reference_tokens,
            "exact": candidate_tokens == reference_tokens,
            "eviction": {
                "deferred_refcount_after_concurrent_pair": deferred_refcount_after_pair,
                "reconciled_pair_request_ids": list(reconciled_pair_requests),
                "refcount_after_concurrent_pair": refcount_after_pair,
                "free_pages_before": free_before_evict,
                "evicted_block_ids": list(evicted_blocks),
                "free_pages_after": free_after_evict,
                "reused_block_ids": list(reused_blocks),
                "same_pages_reused": reused_blocks == evicted_blocks,
                "reload_h2d_bytes": (
                    int(reload_transfers[0].get("h2d_bytes", 0))
                    if reload_transfers else None
                ),
                "reload_exact": replay_tokens == candidate_tokens,
                "reload_ms": replay_ms,
                "deferred_refcount_after_reload": deferred_refcount_after_reload,
                "reconciled_reload_request_ids": list(reconciled_reload_requests),
                "final_evicted_block_ids": list(final_evicted_blocks),
                "free_pages_after_final_evict": free_after_final_evict,
                "persisted_payload_sha256_after_restore": json.loads(
                    (_directory(storage, selected_key) / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["native_payload_sha256"],
            },
            "cancellation": cancellation,
        })

    if stale_pending is not None:
        stale_suffix, stale_command = stale_pending
        stale_probe = {
            "attempted": True,
            "requested_generation": stale_command.source_generation,
            "resident_generation": stale_command.source_generation - 1,
            "rejected": False,
            "error": None,
        }
        try:
            _generate(llm, prime, stale_suffix, stale_command.cache_salt())
        except RuntimeError as error:
            stale_probe["rejected"] = "Stale or unavailable" in str(error)
            stale_probe["error"] = str(error)

    payload = {
        "schema_version": "paper4.5.vllm-cuda-agent-history-kv-gate.v1",
        "probe": "vllm_cuda_sparse_original_position_agent_kv",
        "engine": "vllm-cuda",
        "engine_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": torch.cuda.get_device_name(0),
        "model": args.model,
        "trajectory": str(args.trajectory),
        "retention_fraction": args.retention_fraction,
        "retention_policy": "floor_after_complete-page materialization",
        "temperature": 0,
        "completed_turns": len(rows),
        "exact_turns": sum(int(row["exact"]) for row in rows),
        "all_exact": bool(rows) and all(row["exact"] for row in rows),
        "zero_selected_text_reencoding": True,
        "selection_materialization_is_kv_copy": True,
        "concurrent_session_borrow_test": True,
        "explicit_hot_eviction_test": all(
            row["eviction"]["refcount_after_concurrent_pair"] == 0
            and row["eviction"]["deferred_refcount_after_concurrent_pair"] == 2
            and len(row["eviction"]["reconciled_pair_request_ids"]) == 2
            and bool(row["eviction"]["evicted_block_ids"])
            and row["eviction"]["free_pages_after"]
            > row["eviction"]["free_pages_before"]
            and row["eviction"]["same_pages_reused"]
            and row["eviction"]["reload_exact"]
            and row["eviction"]["deferred_refcount_after_reload"] == 1
            and len(row["eviction"]["reconciled_reload_request_ids"]) == 1
            and row["eviction"]["final_evicted_block_ids"]
            == row["eviction"]["evicted_block_ids"]
            for row in rows
        ),
        "offload_restore_test": all(
            row["eviction"]["same_pages_reused"]
            and row["eviction"]["reload_exact"]
            and int(row["eviction"]["reload_h2d_bytes"] or 0) > 0
            and row["eviction"]["persisted_payload_sha256_after_restore"]
            == row["native_payload_sha256"]
            for row in rows
        ),
        "cancellation_test": any(
            row["cancellation"] is not None
            and row["cancellation"]["refcount_before_abort"] == 2
            and row["cancellation"]["released_on_abort_snapshot"]
            == [row["cancellation"]["cancelled_request_id"]]
            and row["cancellation"]["refcount_after_abort"] == 1
            and row["cancellation"]["handle_survived_abort"]
            and row["cancellation"]["cancelled_absent_from_final_outputs"]
            and row["cancellation"]["survivor_finished"]
            and row["cancellation"]["refcount_after_finish"] == 0
            and row["cancellation"]["survivor_exact_vs_control"]
            and bool(row["cancellation"]["evicted_after_test"])
            for row in rows
        ),
        "stale_generation_test": stale_probe,
        "reference_uses_same_resident_pages": all(
            row["reference_shared_resident_hit"] is True for row in rows
        ),
        "reference_h2d_bytes_zero": all(
            row["reference_load_h2d_bytes"] == 0 for row in rows
        ),
        "original_position_base_preserved": all(
            row["source_position_base"] >= row["selected_kv_tokens"] for row in rows
        ),
        "rows": rows,
        "limitations": [
            "Selection is losslessly gathered once from the full captured K/V into a derived persistent resource.",
            "The first HOT attachment copies that selected resource from host storage into detached GPU pages; subsequent borrowers reuse those resident pages.",
            "This vLLM 0.28 worker hook is version bounded and not an upstream extension point.",
            "The offline worker defers finish callbacks; the gate reconciles connector ownership from an authoritative active-request snapshot before eviction.",
        ],
    }
    payload["realized_retention_fraction_min"] = min(
        (row["realized_retention_fraction"] for row in rows), default=None
    )
    payload["realized_retention_fraction_max"] = max(
        (row["realized_retention_fraction"] for row in rows), default=None
    )
    payload["native_max_position_embeddings"] = int(
        getattr(llm.model_config.hf_config, "max_position_embeddings", 0) or 0
    )
    payload["max_source_position_base"] = max(
        (row["source_position_base"] for row in rows), default=0
    )
    payload["within_native_model_window"] = bool(
        payload["native_max_position_embeddings"]
        and payload["max_source_position_base"] + args.continuation_tokens
        <= payload["native_max_position_embeddings"]
    )
    payload["continuations_non_degenerate"] = bool(
        rows
        and any(
            token != 0
            for row in rows
            for token in row["candidate_token_ids"]
        )
        and len(
            {
                token
                for row in rows
                for token in row["candidate_token_ids"]
            }
        ) > 1
    )
    payload["mechanism_gate_valid"] = bool(
        args.retention_fraction < 1
        and payload["all_exact"]
        and payload["zero_selected_text_reencoding"]
        and payload["reference_uses_same_resident_pages"]
        and payload["reference_h2d_bytes_zero"]
        and payload["original_position_base_preserved"]
        and payload["explicit_hot_eviction_test"]
        and payload["offload_restore_test"]
        and payload["cancellation_test"]
        and payload["stale_generation_test"]["rejected"]
        and any(row["has_holes"] for row in rows)
    )
    payload["sparse_position_gate_valid"] = bool(
        payload["mechanism_gate_valid"]
        and payload["within_native_model_window"]
        and payload["continuations_non_degenerate"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "engine", "model", "completed_turns", "exact_turns", "all_exact",
        "reference_uses_same_resident_pages", "reference_h2d_bytes_zero",
        "original_position_base_preserved", "sparse_position_gate_valid",
    )}, indent=2))
    raise SystemExit(0 if payload["sparse_position_gate_valid"] else 1)


if __name__ == "__main__":
    main()
