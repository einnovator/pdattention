"""Compare matched selected-text and native-K/V execution on MLX-LM."""

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
    _answer_logprob,
    _bounded_source,
    _metrics,
)


def _cache_snapshot(cache):
    """Trim an ordinary prefix cache to immutable state for request forks."""

    import mlx.core as mx

    states = []
    for layer in cache:
        state = layer.state
        states.append(tuple(mx.array(value) for value in state))
    mx.eval(states)
    return tuple(states)


def _restore_cache(model, states):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    if len(cache) != len(states):
        raise RuntimeError("Ordinary MLX prefix cache changed layer count.")
    for layer, state in zip(cache, states):
        layer.state = state
    return cache


def _cache_nbytes(states) -> int:
    return sum(value.nbytes for state in states for value in state[:2])


def _generate_timed(model, tokenizer, query, cache, max_tokens: int):
    """Return deterministic generation with TTFT and mean inter-token latency."""

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    generated = []
    arrivals = []
    for token, _ in generate_step(
        mx.array(query, dtype=mx.int32),
        model,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=0),
    ):
        generated.append(int(token))
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
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx.native import encode_native_memory, make_native_prompt_cache

    manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    model, tokenizer = load(args.model, revision=args.revision)
    rows = []
    for example in examples:
        prepared_at = time.perf_counter()
        candidate_tokens = list(
            tokenizer.encode(example.candidate_source, add_special_tokens=False)
        )
        source = _bounded_source(tokenizer, example.selected_source, args.max_source_tokens)
        text_preparation_ms = (time.perf_counter() - prepared_at) * 1000.0
        answer = list(tokenizer.encode(" " + example.answer, add_special_tokens=False))

        started = time.perf_counter()
        ordinary_cache = make_prompt_cache(model)
        model(mx.array(source, dtype=mx.int32)[None], cache=ordinary_cache)
        ordinary_states = _cache_snapshot(ordinary_cache)
        e0_ingestion_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        native_memory = encode_native_memory(model, source)
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
                cache_factory = (
                    (lambda: _restore_cache(model, ordinary_states))
                    if condition == "e0_selected_text"
                    else (lambda: make_native_prompt_cache(model, native_memory))
                )
                logprob = _answer_logprob(model, query, answer, cache_factory())
                generated = _generate_timed(
                    model, tokenizer, query, cache_factory(), args.max_new_tokens
                )
                exact, f1 = _metrics(generated["output"], example.answer)
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
                            gold_answer_logprob=logprob,
                            evidence_recall=example.evidence_recall,
                            candidate_tokens=len(candidate_tokens),
                            selected_source_tokens=len(source),
                            visible_prompt_tokens=(
                                len(query) if native else len(source) + len(query)
                            ),
                            selected_native_kv_tokens=len(source) if native else 0,
                            active_detail_bytes=native_memory.nbytes if native else 0,
                            retained_detail_bytes=native_memory.nbytes if native else 0,
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
                            bytes_read=native_memory.nbytes if native else 0,
                            bytes_promoted=0,
                            bytes_avoided=(
                                native_memory.nbytes if native and reused else 0
                            ),
                            duplicate_physical_kv_avoided_bytes=(
                                native_memory.nbytes if native and reused else 0
                            ),
                        ),
                        extra={
                            "dataset": example.dataset,
                            "seed": example.seed,
                            "gold_answer": example.answer,
                            "execution_source_sha256": example.selected_source_sha256,
                            "e0_prefix_kv_bytes": _cache_nbytes(ordinary_states),
                            "concurrency_execution": (
                                "shared_residency_serialized"
                                if request.regime == "concurrent_shared_resource"
                                else "single_request"
                            ),
                        },
                    )
                )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper6_cross_engine_matched_e0_e2_mlx_v2",
        "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "max_source_tokens": args.max_source_tokens,
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
