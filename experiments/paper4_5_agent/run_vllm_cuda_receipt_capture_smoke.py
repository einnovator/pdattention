"""Exercise receipt capture, partial-page publication, and mixed CUDA aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import uuid
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm
from vllm import LLM, SamplingParams

from pra_vllm.cuda_protocol import CudaConnectorCommand
from pra_vllm.cuda_sparse_protocol import SparseCudaConnectorCommand


def _generate(
    llm: LLM,
    token_ids: list[int],
    command: SparseCudaConnectorCommand | CudaConnectorCommand,
    *,
    max_tokens: int = 1,
) -> tuple[list[int], float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    rows = llm.generate(
        {"prompt_token_ids": token_ids, "cache_salt": command.cache_salt()},
        SamplingParams(temperature=0, max_tokens=int(max_tokens), ignore_eos=True),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    if len(rows) != 1 or not rows[0].outputs:
        raise RuntimeError("vLLM produced no receipt-smoke output.")
    return (
        list(map(int, rows[0].outputs[0].token_ids)),
        time.perf_counter() - started,
    )


def _write_alias_manifest(
    connector: object,
    command: SparseCudaConnectorCommand,
    *,
    parent: str,
    pages: tuple[int, ...],
    capture: dict[str, object] | None = None,
) -> None:
    directory = Path(connector._directory(command.logical_key))
    directory.mkdir(parents=True, exist_ok=False)
    payload: dict[str, object] = {
        "schema_version": "pra-vllm-cuda-agent-alias-v1",
        "logical_key": command.logical_key,
        "source_tokens": command.source_tokens,
        "source_generation": command.source_generation,
        "parent_logical_key": parent,
        "selected_page_indices": list(pages),
        "physical_kv_copy_bytes": 0,
        "host_to_device_bytes": 0,
    }
    if capture is not None:
        payload["capture_materialized_history"] = capture
    (directory / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().resolve()
    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
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
                "scheduler_page_aliases": True,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    connector = llm.llm_engine.engine_core.engine_core.scheduler.connector
    registry = connector._scheduler_alias_registry
    block = int(llm.llm_engine.vllm_config.cache_config.block_size)
    query = int(tokenizer.eos_token_id or 0)
    seed = list(map(int, tokenizer.encode("PRA receipt capture source. " * 96)))
    source_tokens = min((len(seed) // block) * block, args.source_tokens)
    source_tokens = (source_tokens // block) * block
    if source_tokens < 3 * block:
        raise RuntimeError("Receipt smoke needs at least three source pages.")
    source = seed[:source_tokens]
    source_key = f"receipt-smoke-{run_id}-source"
    source_command = SparseCudaConnectorCommand(
        "store",
        source_key,
        source_tokens,
        source_tokens,
        source_generation=1,
        residency="warm",
    )
    prime_tokens, prime_s = _generate(llm, [*source, query], source_command)

    selected_pages = (0, 2)
    selected = [
        token
        for page in selected_pages
        for token in source[page * block : (page + 1) * block]
    ]
    receipt = list(map(int, tokenizer.encode("Prior task completed successfully.")))
    if len(receipt) < 2:
        raise RuntimeError("Tokenizer produced an unusably short receipt.")
    receipt = receipt[: min(len(receipt), block - 1)]
    receipt_start = source_tokens - block
    capture_key = f"receipt-smoke-{run_id}-capture"
    capture_command = SparseCudaConnectorCommand(
        "load",
        f"receipt-smoke-{run_id}-capture-request",
        len(selected),
        receipt_start,
        source_generation=1,
    )
    _write_alias_manifest(
        connector,
        capture_command,
        parent=source_key,
        pages=selected_pages,
        capture={
            "logical_key": capture_key,
            "token_start": len(selected),
            "token_count": len(receipt),
        },
    )
    capture_tokens, capture_s = _generate(
        llm, [*selected, *receipt, query], capture_command
    )
    capture_dir = Path(connector._directory(capture_key))
    capture_manifest_path = capture_dir / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    receipt_source = f"receipt-smoke-{run_id}-receipt-source"
    capture_manifest["publish_scheduler_source"] = {
        "logical_key": receipt_source,
        "generation": 2,
        "position_extent": receipt_start + len(receipt),
    }
    capture_manifest_path.write_text(
        json.dumps(capture_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    materialize_command = CudaConnectorCommand(
        "load", capture_key, len(receipt), "hot", f"receipt-smoke-{run_id}"
    )
    materialize_tokens, materialize_s = _generate(
        llm, [*receipt, query], materialize_command
    )

    receipt_row = registry.snapshot()["sources"].get(receipt_source)
    if receipt_row is None:
        raise RuntimeError("Captured receipt was not published as scheduler K/V.")
    composite_key = f"receipt-smoke-{run_id}-mixed"
    selected_count = len(selected) + len(receipt)
    copied_bytes = int(capture_manifest.get("native_tensor_bytes", 0))
    terminal_page_copy_bytes = copied_bytes * block // len(receipt)
    registry.publish_composite_source(
        composite_key,
        generation=3,
        position_extent=source_tokens,
        components=(
            (source_key, 1, selected_pages),
            (receipt_source, 2, (0,)),
        ),
        selected_token_count=selected_count,
        partial_terminal_copy_bytes=terminal_page_copy_bytes,
        materialized_history_encoded_tokens=len(receipt),
        materialized_history_copy_bytes=copied_bytes,
    )
    final_command = SparseCudaConnectorCommand(
        "load",
        f"receipt-smoke-{run_id}-final-request",
        selected_count,
        source_tokens,
        source_generation=3,
    )
    _write_alias_manifest(
        connector,
        final_command,
        parent=composite_key,
        pages=tuple(range((selected_count + block - 1) // block)),
    )
    before = registry.snapshot()["telemetry"]
    final_tokens, final_s = _generate(
        llm, [*selected, *receipt, query], final_command
    )
    after = registry.snapshot()["telemetry"]
    delta = {key: int(after[key]) - int(before[key]) for key in after}

    registry.evict_source(source_key, generation=1)
    registry.evict_source(receipt_source, generation=2)
    registry.evict_source(composite_key, generation=3)
    final_snapshot = registry.snapshot()
    qualified = bool(
        receipt_row["source_tokens"] == len(receipt)
        and receipt_row["position_extent"] == receipt_start + len(receipt)
        and len(receipt_row["block_ids"]) == 1
        and capture_manifest["source_tokens"] == len(receipt)
        and capture_manifest["layer_files"] > 0
        and copied_bytes > 0
        and delta["alias_hit_events"] == 1
        and delta["alias_install_events"] == 1
        and delta["alias_release_events"] == 1
        and delta["selected_history_reencoded_tokens"] == 0
        and not final_snapshot["sources"]
        and not final_snapshot["active_requests"]
        and not final_snapshot["pending_requests"]
    )
    payload: dict[str, object] = {
        "schema_version": "paper4.5.vllm-cuda-receipt-capture-smoke.v1",
        "qualified": qualified,
        "experiment_revision": os.environ.get("PRA_EXPERIMENT_REVISION", "uncommitted"),
        "engine": "vllm-cuda",
        "engine_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": torch.cuda.get_device_name(),
        "model": args.model,
        "block_size": block,
        "source_tokens": source_tokens,
        "selected_original_kv_tokens": len(selected),
        "materialized_history_tokens": len(receipt),
        "composite_selected_tokens": selected_count,
        "receipt_position_start": receipt_start,
        "receipt_position_end": receipt_start + len(receipt),
        "materialized_history_copy_bytes": copied_bytes,
        "terminal_page_copy_bytes": terminal_page_copy_bytes,
        "selected_history_copy_bytes": 0,
        "selected_history_reencoded_tokens": 0,
        "tokens": {
            "prime": prime_tokens,
            "capture": capture_tokens,
            "materialize": materialize_tokens,
            "final": final_tokens,
        },
        "elapsed_seconds": {
            "prime": prime_s,
            "capture": capture_s,
            "materialize": materialize_s,
            "final": final_s,
        },
        "final_alias_telemetry_delta": delta,
        "capture_manifest_sha256": hashlib.sha256(
            capture_manifest_path.read_bytes()
        ).hexdigest(),
        "final_registry_snapshot": final_snapshot,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--source-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({
        "qualified": payload["qualified"],
        "materialized_history_tokens": payload["materialized_history_tokens"],
        "materialized_history_copy_bytes": payload["materialized_history_copy_bytes"],
        "output": str(args.output),
    }, indent=2))
    raise SystemExit(0 if payload["qualified"] else 1)


if __name__ == "__main__":
    main()
