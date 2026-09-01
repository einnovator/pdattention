"""Run the selector-frozen CUDA FULL/E0/E2 residency economics matrix.

This benchmark deliberately distinguishes physical facts that the first CUDA
connector smoke conflated. FULL and E0 may use ordinary APC. E2-HOT keeps one
immutable source copy on the GPU, while E2-WARM reloads the same persisted
source and transfers it from host to device. Scoped connector salts prevent
ordinary APC from silently serving the E2 source in either native condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from experiments.paper6_vllm.run_cuda_connector_candidate import (
    EXPECTED,
    _aligned,
    _prompt,
)
from pra_vllm.cuda_protocol import CudaConnectorCommand


CONDITIONS = ("full", "e0_selected_text", "e2_hot", "e2_warm")


def percentile(values: Iterable[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for a finite sample."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count unique Torch tensor storage exposed by a nested runtime object."""

    import torch

    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, visited) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, visited) for item in value)
    return 0


def _request_times(output: Any) -> dict[str, float | None]:
    """Extract version-tolerant vLLM request timing fields."""

    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return {
            "ttft_ms": None,
            "mean_itl_ms": None,
            "completion_latency_ms": None,
            "queue_ms": None,
            "preemptions": None,
        }
    arrival = getattr(metrics, "arrival_time", None)
    first = getattr(metrics, "first_token_time", None)
    last = getattr(metrics, "last_token_time", None)
    if last is None:
        last = getattr(metrics, "finished_time", None)
    scheduled = getattr(metrics, "first_scheduled_time", None)
    output_tokens = len(output.outputs[0].token_ids)
    return {
        "ttft_ms": (
            None if arrival is None or first is None else (first - arrival) * 1000.0
        ),
        "mean_itl_ms": (
            None
            if first is None or last is None or output_tokens < 2
            else (last - first) * 1000.0 / (output_tokens - 1)
        ),
        "completion_latency_ms": (
            None if arrival is None or last is None else (last - arrival) * 1000.0
        ),
        "queue_ms": (
            None
            if arrival is None or scheduled is None
            else (scheduled - arrival) * 1000.0
        ),
        "preemptions": getattr(metrics, "num_preemptions", 0),
    }


def _read_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[offset:] if line.strip()], len(lines)


def _prompt_for_condition(
    condition: str,
    *,
    full_tokens: list[int],
    source_tokens: list[int],
    query_tokens: list[int],
    placeholder_token: int,
    logical_key: str,
    scope: str,
    detached_pages: bool = False,
) -> dict[str, object]:
    if condition == "full":
        salt = hashlib.sha256(b"paper6-cuda-economics-full").hexdigest()
        return _prompt(full_tokens + query_tokens, salt)
    if condition == "e0_selected_text":
        salt = hashlib.sha256(b"paper6-cuda-economics-e0").hexdigest()
        return _prompt(source_tokens + query_tokens, salt)
    residency = "hot" if condition == "e2_hot" else "warm"
    command = CudaConnectorCommand(
        "load",
        logical_key,
        len(source_tokens),
        residency=residency,
        request_scope=scope,
    )
    prefix = [] if detached_pages else [placeholder_token] * len(source_tokens)
    return _prompt(prefix + query_tokens, command.cache_salt())


def _finite(samples: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in samples if row.get(field) is not None]


def _aggregate(
    condition: str,
    samples: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    transfers: list[dict[str, Any]],
    *,
    block_size: int,
    source_tokens: int,
    full_tokens: int,
    query_tokens: int,
    persisted_bytes: int,
    kv_bytes_per_token: float,
    detached_pages: bool,
) -> dict[str, Any]:
    elapsed = sum(float(row["elapsed_ms"]) for row in batches) / 1000.0
    successes = sum(bool(row["expected_recovery"]) for row in samples)
    request_rate = len(samples) / max(elapsed, 1e-9)
    cached = [int(row["cached_prompt_tokens"]) for row in samples]
    prompt_tokens = {
        "full": full_tokens + query_tokens,
        "e0_selected_text": source_tokens + query_tokens,
        "e2_hot": query_tokens if detached_pages else source_tokens + query_tokens,
        "e2_warm": query_tokens if detached_pages else source_tokens + query_tokens,
    }[condition]
    visible_source_tokens = {
        "full": full_tokens,
        "e0_selected_text": source_tokens,
        "e2_hot": 0,
        "e2_warm": 0,
    }[condition]
    native = condition.startswith("e2_")
    ttft = _finite(samples, "ttft_ms")
    itl = _finite(samples, "mean_itl_ms")
    completion = _finite(samples, "completion_latency_ms")
    queue = _finite(samples, "queue_ms")
    h2d = sum(int(event.get("h2d_bytes", 0)) for event in transfers)
    d2d = sum(int(event.get("d2d_bytes", 0)) for event in transfers)
    return {
        "condition": condition,
        "requests": len(samples),
        "successful_requests": successes,
        "success_rate": successes / len(samples),
        "requests_per_second": request_rate,
        "successful_requests_per_second": successes / max(elapsed, 1e-9),
        "useful_throughput": successes / max(elapsed, 1e-9),
        "ttft_ms": {
            "p50": percentile(ttft, 0.50),
            "p95": percentile(ttft, 0.95),
            "p99": percentile(ttft, 0.99),
        },
        "mean_itl_ms": {
            "p50": percentile(itl, 0.50),
            "p95": percentile(itl, 0.95),
            "p99": percentile(itl, 0.99),
        },
        "completion_latency_ms": {
            "p50": percentile(completion, 0.50),
            "p95": percentile(completion, 0.95),
            "p99": percentile(completion, 0.99),
        },
        "queue_ms": {
            "p50": percentile(queue, 0.50),
            "p95": percentile(queue, 0.95),
            "p99": percentile(queue, 0.99),
        },
        "visible_source_tokens_per_request": visible_source_tokens,
        "scheduler_prompt_tokens_per_request": prompt_tokens,
        "selected_native_tokens_per_request": source_tokens if native else 0,
        "apc_cached_tokens_mean": statistics.fmean(cached),
        "apc_blocks_mean": statistics.fmean(cached) / block_size,
        "pra_logical_blocks": source_tokens // block_size if native else 0,
        "pra_request_slot_blocks": (
            source_tokens // block_size if native and not detached_pages else 0
        ),
        "pra_request_slot_bytes_inferred": (
            int(source_tokens * kv_bytes_per_token)
            if native and not detached_pages
            else 0
        ),
        "pra_shared_detached_blocks": (
            source_tokens // block_size if native and detached_pages else 0
        ),
        "pra_shared_detached_bytes": (
            int(source_tokens * kv_bytes_per_token)
            if native and detached_pages
            else 0
        ),
        "pra_hot_source_bytes": max(
            (
                max(
                    int(event.get("resident_hot_bytes", 0)),
                    int(event.get("resident_detached_bytes", 0)),
                )
                for event in transfers
            ),
            default=0,
        ),
        "pra_warm_persisted_bytes": persisted_bytes if condition == "e2_warm" else 0,
        "storage_read_bytes_total": sum(
            int(event.get("storage_read_bytes", 0)) for event in transfers
        ),
        "h2d_bytes_total": h2d,
        "h2d_bytes_per_request": h2d / len(samples),
        "d2d_bytes_total": d2d,
        "d2d_bytes_per_request": d2d / len(samples),
        "reload_amplification": (
            sum(int(event.get("storage_read_bytes", 0)) for event in transfers)
            / max(1, int(source_tokens * kv_bytes_per_token))
            if condition == "e2_warm"
            else 0.0
        ),
        "connector_load_ms": {
            "p50": percentile(
                [float(event["load_ms"]) for event in transfers], 0.50
            ),
            "p95": percentile(
                [float(event["load_ms"]) for event in transfers], 0.95
            ),
            "p99": percentile(
                [float(event["load_ms"]) for event in transfers], 0.99
            ),
        },
        "peak_allocated_bytes": max(
            int(row["peak_allocated_bytes"]) for row in batches
        ),
        "peak_reserved_bytes": max(int(row["peak_reserved_bytes"]) for row in batches),
        "tail_status": "MEASURED" if len(samples) >= 100 else "SMALL_SAMPLE",
        "preemptions": sum(
            int(row.get("preemptions") or 0) for row in samples
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", type=Path, default=Path(".pra/vllm-cuda-economics"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--waves", type=int, default=16)
    parser.add_argument("--full-source-blocks", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--detached-pages", action="store_true")
    parser.add_argument("--detached-reserve-blocks", type=int, default=64)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.waves <= 0 or args.full_source_blocks <= 1:
        raise ValueError("Concurrency/waves must be positive and FULL needs >1 block.")

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
        max_model_len=max(512, args.full_source_blocks * 32),
        max_num_seqs=args.concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config={
            "kv_connector": "PRASemanticConnector",
            "kv_connector_module_path": "pra_vllm.cuda_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "storage_path": str(storage),
                "telemetry_path": str(telemetry),
                "detached_pages": args.detached_pages,
                "detached_reserve_blocks": args.detached_reserve_blocks,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    padding_token = int(tokenizer.encode(" archive", add_special_tokens=False)[0])
    query_tokens = tokenizer.encode(
        "\nWhat is the authoritative archived verification code? "
        "Answer with the four digits only:",
        add_special_tokens=False,
    )
    source_tokens = _aligned(
        tokenizer.encode(
            f"Authoritative archived fact: verification code {EXPECTED}.",
            add_special_tokens=False,
        ),
        block_size,
        padding_token,
    )
    target_full_tokens = args.full_source_blocks * block_size
    distractor_unit = tokenizer.encode(
        "Archived distractor record: obsolete verification code 7394. ",
        add_special_tokens=False,
    )
    required = target_full_tokens - len(source_tokens)
    distractors = (distractor_unit * math.ceil(required / len(distractor_unit)))[:required]
    full_tokens = distractors + source_tokens
    suffix = tokenizer.encode("\nQuestion:", add_special_tokens=False)[:1]
    logical_key = f"cuda-economics-{run_id}"
    sampling = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, ignore_eos=True
    )
    store_sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    llm.generate(
        _prompt(
            source_tokens + suffix,
            CudaConnectorCommand(
                "store", logical_key, len(source_tokens)
            ).cache_salt(),
        ),
        store_sampling,
        use_tqdm=False,
    )
    resource_dir = storage / hashlib.sha256(logical_key.encode("utf-8")).hexdigest()
    persisted_bytes = sum(path.stat().st_size for path in resource_dir.glob("*"))
    manifest = json.loads((resource_dir / "manifest.json").read_text(encoding="utf-8"))
    native_tensor_bytes = int(manifest.get("native_tensor_bytes", persisted_bytes))
    kv_bytes_per_token = native_tensor_bytes / len(source_tokens)

    # Prime only aligned source blocks in ordinary APC. Caching the complete
    # source-plus-query prompt would give E0 cached query geometry while E2
    # still prefills a fresh query, confounding representation with APC state.
    llm.generate(
        _prompt(
            full_tokens + suffix,
            hashlib.sha256(b"paper6-cuda-economics-full").hexdigest(),
        ),
        store_sampling,
        use_tqdm=False,
    )
    llm.generate(
        _prompt(
            source_tokens + suffix,
            hashlib.sha256(b"paper6-cuda-economics-e0").hexdigest(),
        ),
        store_sampling,
        use_tqdm=False,
    )
    llm.generate(
        _prompt_for_condition(
            "e2_hot",
            full_tokens=full_tokens,
            source_tokens=source_tokens,
            query_tokens=query_tokens,
            placeholder_token=padding_token,
            logical_key=logical_key,
            scope="warmup-hot",
            detached_pages=args.detached_pages,
        ),
        sampling,
        use_tqdm=False,
    )

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in runner.model.parameters()
    )
    kv_pool_bytes = _tensor_bytes(getattr(runner, "kv_caches", ()))
    baseline_allocated_bytes = int(torch.cuda.memory_allocated())
    baseline_reserved_bytes = int(torch.cuda.memory_reserved())
    event_offset = 0
    _, event_offset = _read_events(telemetry, event_offset)
    all_rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for condition in CONDITIONS:
        samples: list[dict[str, Any]] = []
        batches: list[dict[str, Any]] = []
        transfers: list[dict[str, Any]] = []
        for wave in range(args.waves):
            prompts = [
                _prompt_for_condition(
                    condition,
                    full_tokens=full_tokens,
                    source_tokens=source_tokens,
                    query_tokens=query_tokens,
                    placeholder_token=padding_token,
                    logical_key=logical_key,
                    scope=f"{condition}-{wave}-{slot}",
                    detached_pages=args.detached_pages,
                )
                for slot in range(args.concurrency)
            ]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            batches.append(
                {
                    "wave": wave,
                    "elapsed_ms": elapsed_ms,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
            for output in outputs:
                text = str(output.outputs[0].text).strip()
                samples.append(
                    {
                        "wave": wave,
                        "request_id": str(output.request_id),
                        "output_text": text,
                        "output_token_ids": list(map(int, output.outputs[0].token_ids)),
                        "expected_recovery": EXPECTED in text,
                        "cached_prompt_tokens": int(output.num_cached_tokens),
                        **_request_times(output),
                    }
                )
            events, event_offset = _read_events(telemetry, event_offset)
            transfers.extend(event for event in events if event.get("event") == "load")
        raw[condition] = {"samples": samples, "batches": batches, "transfers": transfers}
        all_rows.append(
            _aggregate(
                condition,
                samples,
                batches,
                transfers,
                block_size=block_size,
                source_tokens=len(source_tokens),
                full_tokens=len(full_tokens),
                query_tokens=len(query_tokens),
                persisted_bytes=persisted_bytes,
                kv_bytes_per_token=kv_bytes_per_token,
                detached_pages=args.detached_pages,
            )
        )

    payload = {
        "schema_version": "paper6-vllm-cuda-economics-v2",
        "evidence_tier": "CUDA_MATCHED_DETACHED_PAGES" if args.detached_pages else "CUDA_MATCHED_CONNECTOR_CANDIDATE",
        "integration_status": "E2_SCHEDULER_INVISIBLE" if args.detached_pages else "E2_CANDIDATE_PREFIX_SHAPED",
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "model_id": args.model,
        "device": torch.cuda.get_device_name(0),
        "selector_frozen": True,
        "selection_identity": logical_key,
        "source_content_visible_in_e2": False,
        "source_slots_scheduler_visible_in_e2": not args.detached_pages,
        "apc_enabled": True,
        "concurrency": args.concurrency,
        "waves": args.waves,
        "requests_per_condition": args.concurrency * args.waves,
        "block_size": block_size,
        "source_tokens": len(source_tokens),
        "full_source_tokens": len(full_tokens),
        "query_tokens": len(query_tokens),
        "detached_reserve_blocks": (
            args.detached_reserve_blocks if args.detached_pages else 0
        ),
        "hbm_decomposition": {
            "model_parameter_bytes": model_parameter_bytes,
            "vllm_kv_pool_bytes": kv_pool_bytes,
            "framework_and_other_allocated_bytes_inferred": max(
                0, baseline_allocated_bytes - model_parameter_bytes - kv_pool_bytes
            ),
            "global_allocated_bytes_before_measurement": baseline_allocated_bytes,
            "global_reserved_bytes_before_measurement": baseline_reserved_bytes,
            "device_capacity_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
            "status": "MEASURED_COMPONENTS_PLUS_GLOBAL_ALLOCATOR",
        },
        "rows": all_rows,
        "raw": raw,
        "limitations": (
            [
                "Detached CUDA pages use a version-bounded vLLM 0.28 worker hook; an upstream extension point remains preferable.",
                "E2-WARM uses the PRA connector store, not an LMCache deployment.",
                "Cancellation is not exercised by the offline LLM harness.",
                "ITL percentiles are across per-request mean inter-token latency derived from vLLM request timestamps.",
            ]
            if args.detached_pages
            else [
                "CUDA E2 uses source-length scheduler placeholder slots.",
                "E2-HOT is a persistent source tensor plus request-owned attachment, not detached shared vLLM pages.",
                "E2-WARM uses the PRA connector store, not an LMCache deployment.",
                "ITL percentiles are across per-request mean inter-token latency derived from vLLM request timestamps.",
            ]
        ),
    }
    baseline_sequences = [
        tuple(sample["output_token_ids"])
        for sample in raw["e0_selected_text"]["samples"]
    ]
    for row in payload["rows"]:
        sequences = [
            tuple(sample["output_token_ids"])
            for sample in raw[row["condition"]]["samples"]
        ]
        matches = sum(left == right for left, right in zip(sequences, baseline_sequences))
        row["sequence_agreement_vs_e0"] = matches / len(baseline_sequences)
        row["quality_gate_passed"] = (
            row["success_rate"] >= 0.95
            and row["sequence_agreement_vs_e0"] >= 0.95
        )
        row["useful_throughput"] = (
            row["requests_per_second"] if row["quality_gate_passed"] else 0.0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": all_rows}, indent=2))


if __name__ == "__main__":
    main()
