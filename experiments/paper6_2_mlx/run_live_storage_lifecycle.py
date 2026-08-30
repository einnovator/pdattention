"""Exercise HOT/WARM/COLD/restart lifecycle through live MLX generation."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import _bounded_source
from experiments.paper6_2_mlx.run_matched_e0_e2 import _generate_timed


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
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mlx_lm import load
    from pra_hf.storage_lifecycle import (
        InMemoryHotBridge,
        PRARetentionClass,
        PRAStorageEntry,
        PRAStorageManager,
        PRAStoragePolicy,
        PRAStorageTierConfig,
    )
    from pra_mlx.native import (
        MLXNativeColdCodec,
        encode_native_memory,
        make_native_prompt_cache,
        serialize_native_memory,
    )

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    examples = examples[: args.max_examples]
    model, tokenizer = load(args.model, revision=args.revision)
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="pra-mlx-lifecycle-") as directory:
        root = Path(directory)
        policy = PRAStoragePolicy(
            profile="mlx-live-lifecycle",
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
            hot=InMemoryHotBridge(),
            cold_codec=MLXNativeColdCodec(),
        )
        last_key = None
        try:
            for example in examples:
                source = _bounded_source(
                    tokenizer, example.selected_source, args.max_source_tokens
                )
                query = list(
                    tokenizer.encode(example.question, add_special_tokens=False)
                )
                memory = encode_native_memory(model, source)
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
                    request_id = f"{key}:{tier}"
                    with manager.pin_request(request_id, (key,)):
                        active = manager.hot.get_hot(key)
                        generated = _generate_timed(
                            model,
                            tokenizer,
                            query,
                            make_native_prompt_cache(model, active),
                            args.max_new_tokens,
                        )
                    outputs[tier] = str(generated["output"])
                    latencies[tier] = float(generated["completion_latency_ms"])
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
                hot=InMemoryHotBridge(),
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

    result = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_live_storage_lifecycle_v1",
        "engine": "mlx-lm",
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
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
