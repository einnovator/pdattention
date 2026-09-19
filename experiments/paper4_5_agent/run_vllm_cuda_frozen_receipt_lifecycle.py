"""Run the Paper 8.5 receipt-aware frozen request through vLLM CUDA pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import uuid
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import safetensors.torch
import torch
import vllm
from vllm import LLM

from pra_vllm.cuda_protocol import CudaConnectorCommand
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand

from .frozen_agent_plan import frozen_live_kv_geometry, load_frozen_agent_decisions
from .run_vllm_cuda_receipt_capture_smoke import _generate, _write_alias_manifest


def _directory(connector: object, logical_key: str) -> Path:
    return Path(connector._directory(logical_key))


def _pack_capture(
    connector: object,
    *,
    prior_key: str | None,
    capture_key: str,
    packed_key: str,
    total_tokens: int,
) -> tuple[int, int]:
    """Pack receipt tensors on host and return native bytes and host-copy bytes."""

    capture_dir = _directory(connector, capture_key)
    packed_dir = _directory(connector, packed_key)
    packed_dir.mkdir(parents=True, exist_ok=False)
    capture_manifest = json.loads(
        (capture_dir / "manifest.json").read_text(encoding="utf-8")
    )
    prior_dir = None if prior_key is None else _directory(connector, prior_key)
    native_bytes = 0
    host_copy_bytes = 0
    files = sorted(capture_dir.glob("layer-*.safetensors"))
    if not files:
        raise RuntimeError("Receipt capture produced no layer tensors.")
    for capture_path in files:
        current = safetensors.torch.load_file(str(capture_path))["kv_cache"]
        if prior_dir is None:
            packed = current.contiguous()
        else:
            prior_path = prior_dir / capture_path.name
            prior = safetensors.torch.load_file(str(prior_path))["kv_cache"]
            packed = torch.cat((prior, current), dim=0).contiguous()
            host_copy_bytes += int(packed.numel() * packed.element_size())
        native_bytes += int(packed.numel() * packed.element_size())
        safetensors.torch.save_file(
            {"kv_cache": packed}, str(packed_dir / capture_path.name)
        )
    manifest = {
        "schema_version": "pra-vllm-cuda-materialized-history-v1",
        "logical_key": packed_key,
        "source_tokens": int(total_tokens),
        "layer_files": len(files),
        "native_tensor_bytes": native_bytes,
        "capture_source_tokens": int(capture_manifest["source_tokens"]),
    }
    (packed_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return native_bytes, host_copy_bytes


def _page_indices(plan: object, *, block_size: int) -> tuple[int, ...]:
    pages: set[int] = set()
    for interval in plan.intervals:
        pages.update(range(
            int(interval.start) // block_size,
            (int(interval.end) + block_size - 1) // block_size,
        ))
    return tuple(sorted(pages))


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().resolve()
    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config={
            "kv_connector": "PRASparseConnector",
            "kv_connector_module_path": "pra_vllm.cuda_sparse_connector",
            "kv_role": "kv_both",
            "kv_buffer_size": args.kv_transfer_buffer_bytes,
            "kv_connector_extra_config": {
                "storage_path": str(storage),
                "scheduler_page_aliases": True,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    decisions = load_frozen_agent_decisions(
        args.request_replay.expanduser().resolve(),
        args.selection_fixture.expanduser().resolve(),
    )
    if not 1 <= args.request_index <= len(decisions):
        raise ValueError("--request-index is outside the frozen replay")
    decision = decisions[args.request_index - 1]
    geometry = frozen_live_kv_geometry(
        tokenizer, decision, full_retention=args.frozen_full_retention
    )
    connector = llm.llm_engine.engine_core.engine_core.scheduler.connector
    registry = connector._scheduler_alias_registry
    block = int(llm.llm_engine.vllm_config.cache_config.block_size)
    source_ids = list(geometry.source_ids)
    wire_tail = list(geometry.wire_tail_ids)
    pad = (-len(source_ids)) % block
    if pad >= len(wire_tail):
        raise RuntimeError("Wire tail cannot complete the terminal source page.")
    resident_source_ids = [*source_ids, *wire_tail[:pad]]
    wire_suffix = wire_tail[pad:]
    source_tokens = len(resident_source_ids)
    if source_tokens % block:
        raise AssertionError("Resident source must be page aligned.")
    original_pages = _page_indices(geometry.plan, block_size=block)
    terminal_page = (len(source_ids) - 1) // block
    if terminal_page not in original_pages:
        raise RuntimeError("Frozen selection omitted the active-tail boundary page.")
    selected_original_tokens = len(original_pages) * block
    query = wire_suffix[0]

    source_key = f"frozen-{run_id}-source"
    source_command = SparseCudaConnectorCommand(
        "store",
        source_key,
        source_tokens,
        source_tokens,
        source_generation=1,
        residency="warm",
    )
    _prime_tokens, prime_s = _generate(
        llm, [*resident_source_ids, query], source_command
    )

    receipt_ids: list[int] = []
    packed_storage_key: str | None = None
    packed_source_key: str | None = None
    packed_source_generation = 1
    packed_native_bytes = 0
    previous_prefix_key: str | None = None
    previous_prefix_generation = 1
    materialization_rows: list[dict[str, object]] = []
    materialized_copy_bytes = 0
    materialized_h2d_bytes = 0
    materialized_host_pack_bytes = 0
    materialized_cow_bytes = 0

    for ordinal, span in enumerate(geometry.materialized_history_spans, 1):
        prior_pages = tuple(
            page for page in original_pages if (page + 1) * block <= span.position_start
        )
        if not prior_pages:
            raise RuntimeError("Receipt has no causally prior selected source page.")
        prior_original_ids = [
            token
            for page in prior_pages
            for token in resident_source_ids[page * block : (page + 1) * block]
        ]
        selected_count = len(prior_original_ids) + len(receipt_ids)
        if packed_source_key is None:
            parent_key = source_key
            parent_generation = 1
            parent_pages = prior_pages
        else:
            prefix_key = f"frozen-{run_id}-receipt-prefix-{ordinal}"
            prefix_generation = 100 + ordinal
            packed_pages = tuple(range(math.ceil(len(receipt_ids) / block)))
            page_bytes = packed_native_bytes * block // max(len(receipt_ids), 1)
            registry.publish_composite_source(
                prefix_key,
                generation=prefix_generation,
                position_extent=span.position_start,
                components=(
                    (source_key, 1, prior_pages),
                    (packed_source_key, packed_source_generation, packed_pages),
                ),
                selected_token_count=selected_count,
                partial_terminal_copy_bytes=page_bytes,
            )
            parent_key = prefix_key
            parent_generation = prefix_generation
            parent_pages = tuple(range(math.ceil(selected_count / block)))
            previous_prefix_key = prefix_key
            previous_prefix_generation = prefix_generation

        current = list(span.token_ids)
        capture_key = f"frozen-{run_id}-capture-{ordinal}"
        capture_command = SparseCudaConnectorCommand(
            "load",
            f"frozen-{run_id}-capture-request-{ordinal}",
            selected_count,
            span.position_start,
            source_generation=parent_generation,
        )
        _write_alias_manifest(
            connector,
            capture_command,
            parent=parent_key,
            pages=parent_pages,
            capture={
                "logical_key": capture_key,
                "token_start": selected_count,
                "token_count": len(current),
            },
        )
        before_capture = registry.snapshot()["telemetry"]
        capture_tokens, capture_s = _generate(
            llm,
            [*prior_original_ids, *receipt_ids, *current, query],
            capture_command,
        )
        after_capture = registry.snapshot()["telemetry"]
        capture_cow = (
            int(after_capture["materialized_history_copy_bytes"])
            - int(before_capture["materialized_history_copy_bytes"])
        )
        materialized_cow_bytes += capture_cow

        receipt_ids.extend(current)
        next_storage_key = f"frozen-{run_id}-packed-storage-{ordinal}"
        native_bytes, host_pack_bytes = _pack_capture(
            connector,
            prior_key=packed_storage_key,
            capture_key=capture_key,
            packed_key=next_storage_key,
            total_tokens=len(receipt_ids),
        )
        materialized_host_pack_bytes += host_pack_bytes
        packed_manifest_path = _directory(connector, next_storage_key) / "manifest.json"
        packed_manifest = json.loads(packed_manifest_path.read_text(encoding="utf-8"))
        next_source_key = f"frozen-{run_id}-receipt-source-{ordinal}"
        next_source_generation = 200 + ordinal
        packed_manifest["publish_scheduler_source"] = {
            "logical_key": next_source_key,
            "generation": next_source_generation,
            "position_extent": span.position_end,
        }
        packed_manifest_path.write_text(
            json.dumps(packed_manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        materialize_command = CudaConnectorCommand(
            "load",
            next_storage_key,
            len(receipt_ids),
            "hot",
            f"frozen-{run_id}-materialize-{ordinal}",
        )
        _materialize_tokens, materialize_s = _generate(
            llm, [*receipt_ids, query], materialize_command
        )
        materialized_h2d_bytes += native_bytes
        materialized_copy_bytes += int(
            json.loads((_directory(connector, capture_key) / "manifest.json").read_text())[
                "native_tensor_bytes"
            ]
        )
        if previous_prefix_key is not None:
            registry.evict_source(
                previous_prefix_key, generation=previous_prefix_generation
            )
            previous_prefix_key = None
        if packed_source_key is not None:
            registry.evict_source(
                packed_source_key, generation=packed_source_generation
            )
        packed_storage_key = next_storage_key
        packed_source_key = next_source_key
        packed_source_generation = next_source_generation
        packed_native_bytes = native_bytes
        materialization_rows.append({
            "ordinal": ordinal,
            "record_id": span.record_id,
            "message_index": span.message_index,
            "position_start": span.position_start,
            "position_end": span.position_end,
            "receipt_tokens": len(current),
            "cumulative_receipt_tokens": len(receipt_ids),
            "prior_original_pages": len(prior_pages),
            "capture_seconds": capture_s,
            "materialize_seconds": materialize_s,
            "capture_token_ids": capture_tokens,
            "capture_cow_bytes": capture_cow,
            "packed_native_bytes": native_bytes,
            "host_pack_copy_bytes": host_pack_bytes,
        })

    expected_materialized_tokens = sum(
        len(span.token_ids) for span in geometry.materialized_history_spans
    )
    if packed_source_key is None:
        # A resident-subset policy such as M2/P1 has no compact closure
        # receipts.  Alias the selected canonical source pages directly; do
        # not invent a materialized-history object or route the request
        # through the host packing path.
        final_key = source_key
        final_generation = 1
        final_pages = original_pages
        terminal_page_copy_bytes = 0
        materialization_mode = "receipt_free_direct_page_alias"
    else:
        final_key = f"frozen-{run_id}-mixed-history"
        final_generation = 999
        receipt_pages = tuple(range(math.ceil(len(receipt_ids) / block)))
        terminal_page_copy_bytes = packed_native_bytes * block // len(receipt_ids)
        registry.publish_composite_source(
            final_key,
            generation=final_generation,
            position_extent=source_tokens,
            components=(
                (source_key, 1, original_pages),
                (packed_source_key, packed_source_generation, receipt_pages),
            ),
            selected_token_count=selected_original_tokens + len(receipt_ids),
            partial_terminal_copy_bytes=terminal_page_copy_bytes,
            materialized_history_encoded_tokens=len(receipt_ids),
            materialized_history_copy_bytes=(
                materialized_copy_bytes
                + materialized_host_pack_bytes
                + materialized_h2d_bytes
            ),
        )
        final_pages = tuple(
            range(math.ceil((selected_original_tokens + len(receipt_ids)) / block))
        )
        materialization_mode = "original_pages_plus_materialized_receipts"
    final_selected_tokens = selected_original_tokens + len(receipt_ids)
    final_command = SparseCudaConnectorCommand(
        "load",
        f"frozen-{run_id}-final-request-a",
        final_selected_tokens,
        source_tokens,
        source_generation=final_generation,
    )
    _write_alias_manifest(
        connector, final_command, parent=final_key, pages=final_pages
    )
    final_prompt = [
        *(
            token
            for page in original_pages
            for token in resident_source_ids[page * block : (page + 1) * block]
        ),
        *receipt_ids,
        *wire_suffix,
    ]
    before_final = registry.snapshot()["telemetry"]
    final_tokens_a, final_s_a = _generate(
        llm, final_prompt, final_command, max_tokens=args.continuation_tokens
    )
    after_final_a = registry.snapshot()["telemetry"]
    final_command_b = SparseCudaConnectorCommand(
        "load",
        f"frozen-{run_id}-final-request-b",
        final_selected_tokens,
        source_tokens,
        source_generation=final_generation,
    )
    _write_alias_manifest(
        connector, final_command_b, parent=final_key, pages=final_pages
    )
    final_tokens_b, final_s_b = _generate(
        llm, final_prompt, final_command_b, max_tokens=args.continuation_tokens
    )
    after_final_b = registry.snapshot()["telemetry"]
    final_cow_bytes = (
        int(after_final_b["materialized_history_copy_bytes"])
        - int(before_final["materialized_history_copy_bytes"])
    )

    registry.evict_source(source_key, generation=1)
    if packed_source_key is not None:
        registry.evict_source(
            packed_source_key, generation=packed_source_generation
        )
    if final_key != source_key:
        registry.evict_source(final_key, generation=final_generation)
    final_snapshot = registry.snapshot()
    selected_history_page_overhead = selected_original_tokens - geometry.plan.selected_tokens
    total_visible_tokens = final_selected_tokens + len(wire_suffix)
    full_visible_tokens = len(geometry.prompt_ids)
    checks = {
        "all_receipts_materialized": (
            len(receipt_ids) == expected_materialized_tokens
        ),
        "all_original_pages_zero_copy": int(after_final_b["physical_kv_copy_bytes"]) == 0,
        "selected_history_not_reencoded": int(after_final_b["selected_history_reencoded_tokens"]) == 0,
        "repeat_token_exact": final_tokens_a == final_tokens_b,
        "final_alias_installed_twice": (
            int(after_final_b["alias_install_events"])
            - int(before_final["alias_install_events"])
        ) == 2,
        "final_alias_released_twice": (
            int(after_final_b["alias_release_events"])
            - int(before_final["alias_release_events"])
        ) == 2,
        "lifecycle_empty": not any((
            final_snapshot["sources"],
            final_snapshot["active_requests"],
            final_snapshot["pending_requests"],
        )),
    }
    payload: dict[str, object] = {
        "schema_version": "paper4.5.vllm-cuda-frozen-receipt-lifecycle.v2",
        "qualified": all(checks.values()),
        "qualification_blockers": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "experiment_revision": os.environ.get("PRA_EXPERIMENT_REVISION", "uncommitted"),
        "engine": "vllm-cuda",
        "engine_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": torch.cuda.get_device_name(),
        "model": args.model,
        "request_index": decision.request_index,
        "request_input_sha256": decision.request_input_sha256,
        "source_policy": decision.source_policy,
        "frozen_full_retention": args.frozen_full_retention,
        "block_size": block,
        "logical_source_tokens": len(source_ids),
        "resident_source_tokens": source_tokens,
        "active_tail_tokens_moved_to_resident_page": pad,
        "wire_suffix_tokens": len(wire_suffix),
        "selected_logical_kv_tokens": geometry.plan.selected_tokens,
        "selected_original_page_tokens": selected_original_tokens,
        "selected_history_page_overhead_tokens": selected_history_page_overhead,
        "materialized_history_tokens": len(receipt_ids),
        "materialization_mode": materialization_mode,
        "final_selected_tokens": final_selected_tokens,
        "total_visible_tokens": total_visible_tokens,
        "full_visible_tokens": full_visible_tokens,
        "history_saving_after_receipts_fraction": (
            len(source_ids)
            - (selected_original_tokens - pad)
            - len(receipt_ids)
        ) / len(source_ids),
        "total_visible_saving_fraction": (
            full_visible_tokens - total_visible_tokens
        ) / full_visible_tokens,
        "resident_original_kv_omission_fraction": (
            len(source_ids) - (selected_original_tokens - pad)
        ) / len(source_ids),
        "selected_history_reencoded_tokens": 0,
        "selected_history_kv_copy_bytes": 0,
        "receipt_capture_d2h_bytes": materialized_copy_bytes,
        "receipt_host_pack_copy_bytes": materialized_host_pack_bytes,
        "receipt_h2d_bytes": materialized_h2d_bytes,
        "receipt_intermediate_cow_bytes": materialized_cow_bytes,
        "receipt_final_cow_bytes": final_cow_bytes,
        "terminal_page_copy_bytes_per_request": terminal_page_copy_bytes,
        "final_token_ids_a": final_tokens_a,
        "final_token_ids_b": final_tokens_b,
        "final_elapsed_seconds_a": final_s_a,
        "final_elapsed_seconds_b": final_s_b,
        "prime_elapsed_seconds": prime_s,
        "materialization_rows": materialization_rows,
        "final_registry_snapshot": final_snapshot,
        "fixture_hashes": {
            "request_replay_sha256": hashlib.sha256(
                args.request_replay.expanduser().resolve().read_bytes()
            ).hexdigest(),
            "selection_fixture_sha256": hashlib.sha256(
                args.selection_fixture.expanduser().resolve().read_bytes()
            ).hexdigest(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-replay", type=Path, required=True)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=9)
    parser.add_argument(
        "--frozen-full-retention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Select the complete resident historical source while preserving "
            "the same frozen request identity and active wire tail."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--kv-transfer-buffer-bytes", type=int, default=200_000_000)
    parser.add_argument("--continuation-tokens", type=int, default=2)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "qualified": result["qualified"],
        "qualification_blockers": result["qualification_blockers"],
        "total_visible_saving_fraction": result["total_visible_saving_fraction"],
        "final_token_ids": result["final_token_ids_a"],
        "output": str(args.output),
    }, indent=2))
    raise SystemExit(0 if result["qualified"] else 1)


if __name__ == "__main__":
    main()
