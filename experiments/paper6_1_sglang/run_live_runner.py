"""Exercise PRA memory through SGLang's real MLX model-runner lifecycle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.paper6_1_sglang.run_native_kv import EXPECTED, SEEDS, _query, _source


def _run_request(runner, req_id: str, prompt: list[int], max_tokens: int = 12):
    import mlx.core as mx

    started = time.perf_counter()
    pending = runner.prefill_start(req_id, prompt, prompt, [], [], 0)
    runner.eval_pending(pending)
    generated = [runner.prefill_finalize(pending)]
    for _ in range(max_tokens - 1):
        pending_decode = runner.decode_batch_start([req_id])
        runner.eval_pending(pending_decode)
        generated.extend(runner.decode_batch_finalize(pending_decode))
    mx.eval(*runner._req_caches[req_id][0].state)
    return generated, (time.perf_counter() - started) * 1000.0


def _run_batch(runner, req_ids: list[str], prompt: list[int], max_tokens: int = 12):
    """Prefill independently, then use SGLang's real batched decode path."""

    started = time.perf_counter()
    generated = {req_id: [] for req_id in req_ids}
    for req_id in req_ids:
        pending = runner.prefill_start(req_id, prompt, prompt, [], [], 0)
        runner.eval_pending(pending)
        generated[req_id].append(runner.prefill_finalize(pending))
    for _ in range(max_tokens - 1):
        pending = runner.decode_batch_start(req_ids)
        runner.eval_pending(pending)
        for req_id, token in zip(req_ids, runner.decode_batch_finalize(pending)):
            generated[req_id].append(token)
    return generated, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sglang
    from pra_mlx.native import encode_native_memory
    from pra_sglang.mlx_native import SGLangMLXNativeBridge, SGLangSelectedKVCache
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    runner = MlxModelRunner(
        args.model,
        revision=args.revision,
        disable_radix_cache=True,
        enable_sampling=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prepared = []
    for seed in SEEDS:
        source = tokenizer.encode(_source(seed), add_special_tokens=False)
        query = tokenizer.encode(_query(), add_special_tokens=False)
        memory = encode_native_memory(runner.model, source)
        prepared.append((seed, source, query, memory))

    # Native memories are encoded before installing the request-time wrapper;
    # the bridge is a consumer path and must never participate in ingestion.
    bridge = SGLangMLXNativeBridge(runner)
    rows = []
    concurrency_rows = []
    try:
        for seed, source, query, memory in prepared:
            requests = (
                (f"full-{seed}", source + query, False),
                (f"native-{seed}", query, True),
                (f"disabled-{seed}", query, False),
            )
            for req_id, prompt, native in requests:
                if native:
                    bridge.register(req_id, memory, logical_keys=(f"seed-{seed}",))
                generated, latency_ms = _run_request(runner, req_id, prompt)
                text = tokenizer.decode(generated)
                cache = runner._req_caches[req_id][
                    runner._cache_layout.first_attention_layer_index
                ]
                rows.append(
                    {
                        "seed": seed,
                        "condition": "native_runner" if native else (
                            "ordinary_full" if req_id.startswith("full-") else "disabled"
                        ),
                        "visible_prompt_tokens": len(prompt),
                        "selected_native_tokens": len(source) if native else 0,
                        "active_native_kv_bytes": memory.nbytes if native else 0,
                        "runner_cache_type": type(cache).__name__,
                        "scheduler_local_tokens": cache.offset,
                        "attention_rope_offset": (
                            cache.rope_offset
                            if isinstance(cache, SGLangSelectedKVCache)
                            else cache.offset
                        ),
                        "pra_tokens_absent_from_scheduler_count": (
                            bool(
                                native
                                and cache.offset
                                == len(query) + len(generated) - 1
                            )
                            if native
                            else None
                        ),
                        "completion_latency_ms": latency_ms,
                        "output": text,
                        "exact_recovery": EXPECTED in text,
                    }
                )
                runner.remove_request(req_id)
                bridge.unregister(req_id)

        for seed, _source_tokens, query, memory in prepared:
            for concurrency in (1, 2, 4, 8):
                req_ids = [
                    f"batch-{seed}-{concurrency}-{index}"
                    for index in range(concurrency)
                ]
                for req_id in req_ids:
                    bridge.register(req_id, memory, logical_keys=(f"seed-{seed}",))
                generated, wall_ms = _run_batch(runner, req_ids, query)
                outputs = [tokenizer.decode(generated[req_id]) for req_id in req_ids]
                caches = [
                    runner._req_caches[req_id][
                        runner._cache_layout.first_attention_layer_index
                    ]
                    for req_id in req_ids
                ]
                concurrency_rows.append(
                    {
                        "seed": seed,
                        "concurrency": concurrency,
                        "wall_ms": wall_ms,
                        "requests_per_second": concurrency
                        / max(wall_ms / 1000.0, 1e-9),
                        "exact_recovery_rate": sum(EXPECTED in text for text in outputs)
                        / concurrency,
                        "shared_native_kv_bytes": memory.nbytes,
                        "duplicate_native_kv_bytes": memory.nbytes * concurrency,
                        "sharing_bytes_saved": memory.nbytes * (concurrency - 1),
                        "scheduler_counts_exclude_pra": all(
                            cache.offset == len(query) + len(generated[req_id]) - 1
                            for req_id, cache in zip(req_ids, caches)
                        ),
                        "outputs": outputs,
                    }
                )
                for req_id in req_ids:
                    runner.remove_request(req_id)
                    bridge.unregister(req_id)
    finally:
        bridge.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_live_mlx_runner_v1",
        "evidence_tier": "CONTROLLED",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "engine_revision": "ef20fab38a03490e2cdf1b7377145ca3a3f2bfc5",
        "model_id": args.model,
        "model_revision": args.revision,
        "seeds": list(SEEDS),
        "bridge_capabilities": dict(bridge.capabilities()),
        "rows": rows,
        "concurrency_rows": concurrency_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
