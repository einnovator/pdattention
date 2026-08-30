"""Measure event-loop-owned WARM promotion through live MLX/SGLang generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_1_sglang.run_live_storage_lifecycle import _resolve_revision
from experiments.paper6_2_mlx.run_answer_quality_pressure import _bounded_source


def _lead_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("lead-ms must contain non-negative integers")
    return values


async def _run_study(
    *,
    manager,
    prepared: list[dict[str, object]],
    generate: Callable[[str, str, object, list[int]], dict[str, object]],
    lead_ms_values: tuple[int, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    from pra_hf.async_storage import (
        PRAAsyncPromotionScheduler,
        PRAHotAdmissionCandidate,
    )

    rows = []
    for example_index, item in enumerate(prepared):
        key = str(item["key"])
        query = list(item["query"])
        payload = bytes(item["payload"])
        hot_started = time.perf_counter()
        hot = generate(
            f"async-{example_index}-hot", key, manager.hot.get_hot(key), query
        )
        hot_request_ms = (time.perf_counter() - hot_started) * 1000.0

        for lead_ms in lead_ms_values:
            manager.demote_hot(key, payload=payload)
            scheduler = PRAAsyncPromotionScheduler(manager, max_inflight=2)
            prefetch_started = time.perf_counter()
            future = scheduler.prefetch(key, tenant_id="benchmark")
            await asyncio.sleep(lead_ms / 1000.0)
            demand_started = time.perf_counter()
            ready_at_demand = future.done()
            active = await scheduler.resolve(key, tenant_id="benchmark")
            promotion_ready = time.perf_counter()
            generated = generate(
                f"async-{example_index}-lead-{lead_ms}", key, active, query
            )
            request_finished = time.perf_counter()
            await scheduler.close()
            metrics = scheduler.metrics().to_dict()
            rows.append(
                {
                    "dataset": item["dataset"],
                    "example_id": item["example_id"],
                    "lead_ms": lead_ms,
                    "actual_lead_ms": (demand_started - prefetch_started) * 1000.0,
                    "ready_at_demand": ready_at_demand,
                    "demand_stall_ms": (promotion_ready - demand_started) * 1000.0,
                    "hot_request_ms": hot_request_ms,
                    "prefetched_request_ms": (request_finished - demand_started) * 1000.0,
                    "prefetch_to_completion_ms": (
                        request_finished - prefetch_started
                    )
                    * 1000.0,
                    "demand_to_hot_ratio": (
                        (request_finished - demand_started) * 1000.0
                        / max(hot_request_ms, 1e-9)
                    ),
                    "output_exact": generated["output"] == hot["output"],
                    "hot_output": hot["output"],
                    "prefetched_output": generated["output"],
                    "native_bytes": item["native_bytes"],
                    "scheduler": metrics,
                }
            )

    for item in prepared:
        key = str(item["key"])
        if manager.entries[key].current_tier.value == "hot":
            manager.demote_hot(key, payload=bytes(item["payload"]))
    ordered = sorted(prepared, key=lambda item: str(item["key"]))
    admission_budget = sum(int(item["native_bytes"]) for item in ordered[:2])
    admission_scheduler = PRAAsyncPromotionScheduler(manager, max_inflight=2)
    decisions = admission_scheduler.admit_hot_set(
        tuple(
            PRAHotAdmissionCandidate(
                str(item["key"]),
                expected_reuse=float(len(prepared) - index),
                priority=float(-index),
            )
            for index, item in enumerate(ordered)
        ),
        max_prefetch_bytes=admission_budget,
        tenant_id="benchmark",
    )
    admitted = [row.logical_key for row in decisions if row.admitted]
    if admitted:
        await admission_scheduler.resolve(admitted[0], tenant_id="benchmark")
    await admission_scheduler.close()
    admission = {
        "byte_budget": admission_budget,
        "decisions": [row.__dict__ for row in decisions],
        "metrics": admission_scheduler.metrics().to_dict(),
    }
    return rows, admission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("mlx", "sglang"), required=True)
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--lead-ms", type=_lead_values, default=(0, 25, 50, 100, 250))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from pra_hf.storage_lifecycle import (
        PRARetentionClass,
        PRAStorageEntry,
        PRAStorageManager,
        PRAStoragePolicy,
        PRAStorageTierConfig,
    )
    from pra_mlx.native import encode_native_memory, serialize_native_memory

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    examples = examples[: args.max_examples]

    with tempfile.TemporaryDirectory(prefix=f"pra-{args.engine}-async-") as directory:
        root = Path(directory)
        revision = args.revision or _resolve_revision(args.model, None)
        bridge = None
        if args.engine == "mlx":
            from mlx_lm import load
            from pra_hf.storage_lifecycle import DecodingHotBridge
            from pra_mlx import MLXNativeSegmentStore
            from pra_mlx.native import (
                deserialize_native_memory,
                make_native_prompt_cache,
            )
            from experiments.paper6_2_mlx.run_matched_e0_e2 import _generate_timed

            model, tokenizer = load(args.model, revision=revision)
            manager = PRAStorageManager(
                PRAStoragePolicy(
                    profile="mlx-async-warm",
                    hot=PRAStorageTierConfig(max_bytes=2 * 1024**3),
                    warm=PRAStorageTierConfig(
                        path=str(root / "warm"),
                        max_bytes=4 * 1024**3,
                        representation="mmap",
                    ),
                    cold=PRAStorageTierConfig(enabled=False),
                ),
                hot=DecodingHotBridge(deserialize_native_memory),
                warm=MLXNativeSegmentStore(root / "warm"),
            )

            def generate(
                request_id: str, key: str, memory, query: list[int]
            ) -> dict[str, object]:
                with manager.pin_request(request_id, (key,)):
                    return _generate_timed(
                        model,
                        tokenizer,
                        query,
                        make_native_prompt_cache(model, memory),
                        args.max_new_tokens,
                    )

            model_for_encoding = model
        else:
            import sglang
            from experiments.paper6_1_sglang.run_builtin_hicache_backend import (
                _storage_config,
            )
            from experiments.paper6_1_sglang.run_matched_e0_e2 import _generate_timed
            from pra_sglang import (
                SGLangHiCacheByteBackend,
                SGLangHiCacheHotBridge,
                SGLangMLXNativeBridge,
                SGLangPRAHiCache,
            )
            from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
            from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
            from transformers import AutoTokenizer

            os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(
                root / "builtin-hicache"
            )
            runner = MlxModelRunner(
                args.model,
                revision=revision,
                disable_radix_cache=False,
                pool_size=4096,
                mem_fraction_static=0.3,
                enable_sampling=False,
            )
            runner.init_cache_pools(req_to_token_pool=None)
            tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
            hicache = SGLangPRAHiCache(
                root / "l1-l2", max_l1_bytes=2 * 1024**3, max_l2_bytes=2 * 1024**3
            )
            builtin = StorageBackendFactory.create_backend(
                "file", _storage_config(args.model), None
            )
            manager = PRAStorageManager(
                PRAStoragePolicy(
                    profile="sglang-async-warm",
                    hot=PRAStorageTierConfig(max_bytes=2 * 1024**3),
                    warm=PRAStorageTierConfig(
                        path=str(root / "warm-state"), max_bytes=4 * 1024**3
                    ),
                    cold=PRAStorageTierConfig(enabled=False),
                ),
                hot=SGLangHiCacheHotBridge(hicache),
                warm=SGLangHiCacheByteBackend(
                    builtin, namespace="paper6-1-async-warm"
                ),
            )
            bridge = SGLangMLXNativeBridge(runner, storage_manager=manager)

            def generate(
                request_id: str, key: str, memory, query: list[int]
            ) -> dict[str, object]:
                del memory
                bridge.register(request_id, logical_keys=(key,))
                try:
                    return _generate_timed(
                        runner,
                        tokenizer,
                        request_id,
                        query,
                        max_tokens=args.max_new_tokens,
                    )
                finally:
                    runner.remove_request(request_id)
                    bridge.unregister(request_id)

            model_for_encoding = runner.model

        prepared = []
        try:
            for example in examples:
                source = _bounded_source(
                    tokenizer, example.selected_source, args.max_source_tokens
                )
                query = list(tokenizer.encode(example.question, add_special_tokens=False))
                memory = encode_native_memory(model_for_encoding, source)
                payload = serialize_native_memory(memory)
                key = f"async-{example.dataset}-{example.example_id}"
                manager.register(
                    PRAStorageEntry(
                        logical_key=key,
                        record_type="generic_document",
                        retention_class=PRARetentionClass.RECONSTRUCTABLE,
                        tenant_id="benchmark",
                        session_id="async-warm",
                        task_id=None,
                        task_status=None,
                        resource_version=example.selected_source_sha256,
                        detail_bytes=memory.nbytes,
                    ),
                    payload,
                    hot_value=memory,
                    fingerprint=f"{args.model}:{len(memory.layers)}:{len(source)}",
                )
                prepared.append(
                    {
                        "dataset": example.dataset,
                        "example_id": example.example_id,
                        "key": key,
                        "query": query,
                        "payload": payload,
                        "native_bytes": memory.nbytes,
                    }
                )

            rows, admission = asyncio.run(
                _run_study(
                    manager=manager,
                    prepared=prepared,
                    generate=generate,
                    lead_ms_values=args.lead_ms,
                )
            )
        finally:
            if bridge is not None:
                bridge.close()
            manager.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_async_warm_promotion_v1",
        "engine": args.engine,
        "model_id": args.model,
        "model_revision": revision,
        "dataset": args.dataset,
        "lead_ms": list(args.lead_ms),
        "rows": rows,
        "hot_set_admission": admission,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
