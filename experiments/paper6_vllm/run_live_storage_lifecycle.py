"""Exercise HOT/WARM/COLD/restart lifecycle through live vLLM V1 pages."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import _bounded_source
from experiments.paper6_vllm.run_matched_e0_e2 import _aligned, _run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm
    from pra_hf.storage_lifecycle import (
        PRARetentionClass,
        PRAStorageEntry,
        PRAStorageManager,
        PRAStoragePolicy,
        PRAStorageTierConfig,
    )
    from pra_mlx.native import MLXNativeColdCodec, serialize_native_memory
    from pra_vllm import (
        VLLMMetalV1NativeBridge,
        VLLMPageHotBridge,
        capture_paged_memory,
    )
    from vllm import LLM, SamplingParams

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    examples = examples[: args.max_examples]
    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=64)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    ingestion = SamplingParams(temperature=0, max_tokens=1)

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="pra-vllm-lifecycle-") as directory:
        root = Path(directory)
        policy = PRAStoragePolicy(
            profile="vllm-live-lifecycle",
            hot=PRAStorageTierConfig(max_bytes=2 * 1024**3),
            warm=PRAStorageTierConfig(
                path=str(root / "warm"),
                max_bytes=4 * 1024**3,
                representation="mmap",
                cold_grace_seconds=0,
            ),
            cold=PRAStorageTierConfig(
                path=str(root / "cold"),
                max_bytes=8 * 1024**3,
                kv_quantization="int8",
            ),
        )
        manager = PRAStorageManager(
            policy,
            hot=VLLMPageHotBridge(bridge),
            cold_codec=MLXNativeColdCodec(),
        )
        last_key = None
        try:
            for example in examples:
                raw_source = _bounded_source(
                    tokenizer, example.selected_source, args.max_source_tokens
                )
                source = _aligned(raw_source, tokenizer, bridge.block_size)
                query = list(
                    tokenizer.encode(example.question, add_special_tokens=False)
                )
                observation_start = len(bridge.prefill_page_observations())
                _run(
                    llm,
                    bridge,
                    ingestion,
                    source,
                    cache_salt=f"lifecycle-ingest-{example.selection.selection_id}",
                )
                observations = bridge.prefill_page_observations()[observation_start:]
                fresh = [row for row in observations if row["scheduler_cache_start"] == 0]
                if not fresh:
                    raise RuntimeError("vLLM did not expose source-ingestion pages.")
                page_count = len(source) // bridge.block_size
                blocks = tuple(fresh[0]["block_ids_by_group"][0][:page_count])
                memory = capture_paged_memory(bridge, blocks, len(source))
                payload = serialize_native_memory(memory)
                key = f"lifecycle-{example.dataset}-{example.example_id}"
                last_key = key
                manager.register(
                    PRAStorageEntry(
                        logical_key=key,
                        record_type="generic_document",
                        retention_class=PRARetentionClass.RECONSTRUCTABLE,
                        tenant_id="benchmark",
                        session_id="live-storage",
                        task_id=None,
                        task_status=None,
                        resource_version=example.selected_source_sha256,
                        detail_bytes=memory.nbytes,
                    ),
                    payload,
                    hot_value=memory,
                    fingerprint=f"{args.model}:{len(memory.layers)}:{len(source)}",
                )

                outputs: dict[str, str] = {}
                latencies: dict[str, float] = {}
                for tier in ("hot", "warm", "cold"):
                    if tier != "hot":
                        manager.demote_hot(key, payload=payload)
                    if tier == "cold":
                        manager.run_maintenance(
                            now_ns=time.time_ns() + 8 * 86400 * 1_000_000_000
                        )
                    _request_id, output, timing = _run(
                        llm,
                        bridge,
                        sampling,
                        query,
                        key=key,
                        source_tokens=len(source),
                        storage=manager,
                    )
                    outputs[tier] = str(output.outputs[0].text).strip()
                    latencies[tier] = float(timing["completion_latency_ms"])
                rows.append(
                    {
                        "dataset": example.dataset,
                        "example_id": example.example_id,
                        "source_tokens": len(source),
                        "native_bytes": memory.nbytes,
                        "hot_warm_exact": outputs["hot"] == outputs["warm"],
                        "hot_cold_int8_exact": outputs["hot"] == outputs["cold"],
                        "outputs": outputs,
                        "completion_latency_ms": latencies,
                    }
                )
                manager.demote_hot(key, payload=payload)

            manager.close()
            recovered = PRAStorageManager(
                policy,
                hot=VLLMPageHotBridge(bridge),
                cold_codec=MLXNativeColdCodec(),
            )
            restart_ok = False
            if last_key is not None:
                recovered.promote(last_key)
                restart_ok = recovered.entries[last_key].current_tier.value == "hot"
            metrics = recovered.metrics.to_dict()
            usage = recovered.usage()
            recovered.close()
        finally:
            manager.close()
            capabilities = dict(bridge.capabilities())
            bridge.close()

    result = {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_live_storage_lifecycle_v1",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "rows": rows,
        "summary": {
            "examples": len(rows),
            "hot_warm_exact": sum(bool(row["hot_warm_exact"]) for row in rows),
            "hot_cold_int8_exact": sum(
                bool(row["hot_cold_int8_exact"]) for row in rows
            ),
            "restart_recovered": restart_ok,
            "request_lifetime_pinning": True,
            "metrics": metrics,
            "usage": usage,
            "capabilities": capabilities,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
