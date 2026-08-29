"""Run Radix shared-prefix K/V and external PRA HiCache in one MLX request."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from experiments.paper6_1_sglang.run_native_kv import EXPECTED, SEEDS, _query, _source


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.")
    return values


def _stable_prefix(seed: int) -> str:
    sentence = (
        f"Session {seed} contains ordinary dialogue and procedural notes. "
        "This stable text is reusable across related requests. "
    )
    return sentence * 4


def _prefill(runner, req_id: str, tokens: list[int], slot_ids: list[int]) -> None:
    pending = runner.prefill_start(req_id, tokens, tokens, [], slot_ids, 0)
    runner.eval_pending(pending)
    runner.prefill_finalize(pending)


def _generate(
    runner,
    req_id: str,
    prefix: list[int],
    suffix: list[int],
    prefix_slot_ids: list[int],
    *,
    max_tokens: int = 12,
):
    started = time.perf_counter()
    pending = runner.prefill_start(
        req_id,
        suffix,
        prefix + suffix,
        prefix_slot_ids,
        [],
        1,
    )
    runner.eval_pending(pending)
    generated = [runner.prefill_finalize(pending)]
    for _ in range(max_tokens - 1):
        decode = runner.decode_batch_start([req_id])
        runner.eval_pending(decode)
        generated.extend(runner.decode_batch_finalize(decode))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    layer_index = runner._cache_layout.first_attention_layer_index
    return generated, runner._req_caches[req_id][layer_index], elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=_integers, default=SEEDS)
    args = parser.parse_args()

    import sglang
    from pra_mlx.native import encode_native_memory
    from pra_sglang.hicache import PRAHiCacheTier, SGLangPRAHiCache
    from pra_sglang.mlx_native import SGLangMLXNativeBridge, SGLangSelectedKVCache
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    runner = MlxModelRunner(
        args.model,
        revision=args.revision,
        disable_radix_cache=False,
        pool_size=4096,
        mem_fraction_static=0.3,
        enable_sampling=False,
    )
    runner.init_cache_pools(req_to_token_pool=None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prepared = []
    for seed in args.seeds:
        source = tokenizer.encode(_source(seed), add_special_tokens=False)
        memory = encode_native_memory(runner.model, source)
        prepared.append((seed, memory))

    max_bytes = max(memory.nbytes for _, memory in prepared)
    rows = []
    with tempfile.TemporaryDirectory(prefix="pra-sglang-radix-hicache-") as root:
        hicache = SGLangPRAHiCache(
            root,
            max_l1_bytes=max_bytes * 2,
            max_l2_bytes=max_bytes * 2,
        )
        bridge = SGLangMLXNativeBridge(runner, hicache=hicache)
        try:
            for seed, memory in prepared:
                key = f"combined-resource-{seed}"
                hicache.put(key, memory, tier=PRAHiCacheTier.L2)
                prefix = tokenizer.encode(
                    _stable_prefix(seed), add_special_tokens=False
                )
                query = tokenizer.encode(_query(), add_special_tokens=False)
                prefix_slots = list(range(1, len(prefix) + 1))

                warm_id = f"radix-warm-{seed}"
                _prefill(runner, warm_id, prefix, prefix_slots)
                runner.remove_request(warm_id)

                condition_rows = []
                for condition, selected in (
                    ("selected_A", True),
                    ("ordinary_B_after_cleanup", False),
                    ("reselected_C", True),
                ):
                    req_id = f"combined-{condition}-{seed}"
                    if selected:
                        bridge.register(req_id, logical_keys=(key,))
                    generated, cache, latency_ms = _generate(
                        runner, req_id, prefix, query, prefix_slots
                    )
                    text = tokenizer.decode(generated)
                    is_selected_cache = isinstance(cache, SGLangSelectedKVCache)
                    condition_rows.append(
                        {
                            "condition": condition,
                            "request_id": req_id,
                            "output": text,
                            "exact_recovery": EXPECTED in text,
                            "completion_latency_ms": latency_ms,
                            "radix_prefix_tokens": len(prefix),
                            "scheduler_local_tokens": cache.offset,
                            "selected_native_tokens": (
                                cache.memory_tokens if is_selected_cache else 0
                            ),
                            "attention_visible_tokens": (
                                cache.memory_tokens + cache.offset
                                if is_selected_cache
                                else cache.offset
                            ),
                            "selected_cache_protocol": is_selected_cache,
                            "selected_tokens_excluded_from_radix_length": (
                                cache.offset
                                == len(prefix) + len(query) + len(generated) - 1
                            ),
                            "exactly_one_selected_copy": (
                                cache.memory_tokens == memory.source_tokens
                                if is_selected_cache
                                else None
                            ),
                        }
                    )
                    runner.remove_request(req_id)
                    if selected:
                        bridge.unregister(req_id)

                rows.append(
                    {
                        "seed": seed,
                        "source_tokens": memory.source_tokens,
                        "active_native_kv_bytes": memory.nbytes,
                        "radix_prefix_slots": prefix_slots,
                        "hicache_placement_after_requests": hicache.placement(key).value,
                        "conditions": condition_rows,
                    }
                )
        finally:
            capabilities = dict(bridge.capabilities())
            hicache_metrics = hicache.metrics().to_dict()
            bridge.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_1_sglang_radix_hicache_combined_v1",
        "evidence_tier": "CONTROLLED",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "seeds": list(args.seeds),
        "radix_pool_enabled": True,
        "ordinary_radix_namespace_used_for_pra": False,
        "bridge_capabilities": capabilities,
        "hicache_metrics": hicache_metrics,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
