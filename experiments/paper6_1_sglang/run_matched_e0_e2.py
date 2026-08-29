"""Compare matched selected-text and native-K/V QA on SGLang MLX.

E0 and E2 consume the same frozen evidence tokens. E0 stores those tokens in
ordinary Radix-owned prefix slots; E2 stores their native K/V outside Radix and
attaches it through ``SGLangMLXNativeBridge``. The request suffix, decoding
policy, and answer metric are otherwise identical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.engine_serving.matched_e0_e2_contract import (
    SCHEMA_VERSION,
    benchmark_metrics,
    benchmark_row,
    regime_schedule,
    validate_payload,
)
from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import (
    _bounded_source,
    _metrics,
)


def _prefill_prefix(runner, req_id: str, tokens: list[int], slot_ids: list[int]) -> None:
    """Encode an ordinary sequential prefix into SGLang's Radix slot pool."""

    pending = runner.prefill_start(req_id, tokens, tokens, [], slot_ids, 0)
    runner.eval_pending(pending)
    runner.prefill_finalize(pending)


def _generate_timed(
    runner,
    tokenizer,
    req_id: str,
    query: list[int],
    *,
    source: list[int] | None = None,
    prefix_slot_ids: list[int] | None = None,
    max_tokens: int,
) -> dict[str, object]:
    """Generate from a pooled E0 prefix or an already-attached E2 memory."""

    prefix = source or []
    slots = prefix_slot_ids or []
    started = time.perf_counter()
    pending = runner.prefill_start(
        req_id,
        query,
        prefix + query,
        slots,
        [],
        1 if slots else 0,
    )
    runner.eval_pending(pending)
    generated = [int(runner.prefill_finalize(pending))]
    arrivals = [(time.perf_counter() - started) * 1000.0]
    for _ in range(max_tokens - 1):
        decode = runner.decode_batch_start([req_id])
        runner.eval_pending(decode)
        generated.extend(int(token) for token in runner.decode_batch_finalize(decode))
        arrivals.append((time.perf_counter() - started) * 1000.0)
    completion_ms = (time.perf_counter() - started) * 1000.0
    itl_ms = (
        sum(right - left for left, right in zip(arrivals, arrivals[1:]))
        / (len(arrivals) - 1)
        if len(arrivals) > 1
        else 0.0
    )
    return {
        "output": tokenizer.decode(generated).strip(),
        "output_token_ids": generated,
        "generated_tokens": len(generated),
        "ttft_ms": arrivals[0] if arrivals else completion_ms,
        "itl_ms": itl_ms,
        "completion_latency_ms": completion_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), required=True
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument(
        "--revision", default="73e3e38d981303bc594367cd910ea6eb48349da8"
    )
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sglang
    from pra_mlx.native import encode_native_memory
    from pra_sglang.mlx_native import SGLangMLXNativeBridge, SGLangSelectedKVCache
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from transformers import AutoTokenizer

    manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
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
    bridge = SGLangMLXNativeBridge(runner)
    rows = []
    try:
        for example_index, example in enumerate(examples):
            prepared_at = time.perf_counter()
            candidate_tokens = list(
                tokenizer.encode(example.candidate_source, add_special_tokens=False)
            )
            source = _bounded_source(
                tokenizer, example.selected_source, args.max_source_tokens
            )
            text_preparation_ms = (time.perf_counter() - prepared_at) * 1000.0

            # Reusing the same physical range is safe because every preceding
            # request is removed before the next example overwrites the slots.
            prefix_slots = list(range(1, len(source) + 1))
            prefix_id = f"matched-prefix-{example_index}"
            started = time.perf_counter()
            _prefill_prefix(runner, prefix_id, source, prefix_slots)
            e0_ingestion_ms = (time.perf_counter() - started) * 1000.0
            runner.remove_request(prefix_id)

            started = time.perf_counter()
            memory = encode_native_memory(runner.model, source)
            e2_ingestion_ms = (time.perf_counter() - started) * 1000.0

            requests = regime_schedule(
                example.question,
                warm_repeats=args.warm_repeats,
                multi_query_count=args.multi_query_count,
                concurrency=args.concurrency,
            )
            for request in requests:
                query = list(
                    tokenizer.encode(request.query.text, add_special_tokens=False)
                )
                reused = request.regime != "cold_one_shot"
                for condition in ("e0_selected_text", "e2_native_kv"):
                    req_id = (
                        f"matched-{example_index}-{request.regime}-"
                        f"{request.request_ordinal}-{condition}"
                    )
                    if condition == "e2_native_kv":
                        bridge.register(
                            req_id,
                            memory,
                            logical_keys=(f"matched-{example.example_id}",),
                        )
                    generated = _generate_timed(
                        runner,
                        tokenizer,
                        req_id,
                        query,
                        source=source if condition == "e0_selected_text" else None,
                        prefix_slot_ids=(
                            prefix_slots if condition == "e0_selected_text" else None
                        ),
                        max_tokens=args.max_new_tokens,
                    )
                    cache = runner._req_caches[req_id][
                        runner._cache_layout.first_attention_layer_index
                    ]
                    is_selected = isinstance(cache, SGLangSelectedKVCache)
                    exact, f1 = _metrics(str(generated["output"]), example.answer)
                    ingestion_ms = (
                        e0_ingestion_ms
                        if condition == "e0_selected_text"
                        else e2_ingestion_ms
                    )
                    native = condition == "e2_native_kv"
                    rows.append(
                        benchmark_row(
                            condition=condition,
                            selection=example.selection,
                            request=request,
                            output=str(generated["output"]),
                            metrics=benchmark_metrics(
                                exact_match=exact,
                                token_f1=f1,
                                gold_answer_logprob=None,
                                evidence_recall=example.evidence_recall,
                                candidate_tokens=len(candidate_tokens),
                                selected_source_tokens=len(source),
                                visible_prompt_tokens=(
                                    len(query) if native else len(source) + len(query)
                                ),
                                selected_native_kv_tokens=len(source) if native else 0,
                                active_detail_bytes=memory.nbytes if native else 0,
                                retained_detail_bytes=memory.nbytes if native else 0,
                                text_preparation_ms=text_preparation_ms,
                                kv_encode_ms=ingestion_ms,
                                index_construction_ms=0.0,
                                time_to_usable_context_ms=(
                                    text_preparation_ms + ingestion_ms
                                ),
                                ttft_ms=float(generated["ttft_ms"]),
                                itl_ms=float(generated["itl_ms"]),
                                total_latency_ms=float(
                                    generated["completion_latency_ms"]
                                ),
                                generated_tokens=int(generated["generated_tokens"]),
                                ordinary_prefix_cache_hit_tokens=(
                                    len(source) if reused and not native else 0
                                ),
                                pra_hot_hit=native and reused,
                                pra_warm_hit=False,
                                bytes_read=memory.nbytes if native else 0,
                                bytes_promoted=0,
                                bytes_avoided=(memory.nbytes if native and reused else 0),
                                duplicate_physical_kv_avoided_bytes=(
                                    memory.nbytes if native and reused else 0
                                ),
                            ),
                            extra={
                                "dataset": example.dataset,
                                "seed": example.seed,
                                "request_id": req_id,
                                "gold_answer": example.answer,
                                "scheduler_counts_exclude_pra": (
                                    cache.offset
                                    == len(query) + args.max_new_tokens - 1
                                    if is_selected
                                    else None
                                ),
                                "concurrency_execution": (
                                    "shared_residency_serialized"
                                    if request.regime
                                    == "concurrent_shared_resource"
                                    else "single_request"
                                ),
                            },
                        )
                    )
                    runner.remove_request(req_id)
                    if condition == "e2_native_kv":
                        bridge.unregister(req_id)
    finally:
        capabilities = dict(bridge.capabilities())
        bridge.close()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper6_cross_engine_matched_e0_e2_sglang_v2",
        "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
        "engine": "sglang-mlx",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "radix_pool_enabled": True,
        "bridge_capabilities": capabilities,
        "warm_repeats": args.warm_repeats,
        "multi_query_count": args.multi_query_count,
        "concurrency": args.concurrency,
        "rows": rows,
    }
    validate_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
