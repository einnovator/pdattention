"""Compare matched selected-text and native-page QA on vLLM V1."""

from __future__ import annotations

import argparse
import json
import os
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
from experiments.paper6_2_mlx.run_answer_quality_pressure import _bounded_source, _metrics


def _aligned(tokens: list[int], tokenizer, block_size: int) -> list[int]:
    result = list(tokens)
    padding = list(tokenizer.encode(" archive", add_special_tokens=False))
    if not padding:
        raise RuntimeError("Tokenizer did not produce a page-alignment token.")
    while len(result) % block_size:
        result.append(int(padding[0]))
    return result


def _prompt(token_ids: list[int], cache_salt: str | None = None) -> dict[str, object]:
    """Use token IDs so E0 and E2 cannot diverge through decode/re-tokenization."""

    prompt: dict[str, object] = {"prompt_token_ids": list(token_ids)}
    if cache_salt is not None:
        prompt["cache_salt"] = cache_salt
    return prompt


def _run(
    llm,
    bridge,
    sampling,
    prompt_tokens,
    *,
    key=None,
    source_tokens=0,
    source_position_base=None,
    cache_salt=None,
):
    """Step V1 explicitly so TTFT and inter-token arrivals are observable."""

    started = time.perf_counter()
    position_base = (
        source_tokens if source_position_base is None else int(source_position_base)
    )
    if key is not None and cache_salt is None:
        from pra_vllm.v1_native import native_request_cache_salt

        cache_salt = native_request_cache_salt(
            (key,),
            selected_token_count=source_tokens,
            source_position_base=position_base,
        )
    [request_id] = llm.enqueue(
        _prompt(prompt_tokens, cache_salt), sampling, use_tqdm=False
    )
    if key is not None:
        bridge.register(
            request_id,
            (key,),
            selected_token_count=source_tokens,
            source_position_base=position_base,
        )
    arrivals: list[float] = []
    observed_tokens = 0
    output = None
    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()
        if len(step_outputs) > 1:
            raise RuntimeError("Matched timing harness expects one active vLLM request.")
        for candidate in step_outputs:
            output = candidate
            token_count = len(candidate.outputs[0].token_ids)
            if token_count > observed_tokens:
                timestamp = (time.perf_counter() - started) * 1000.0
                arrivals.extend([timestamp] * (token_count - observed_tokens))
                observed_tokens = token_count
    if output is None or not output.finished:
        raise RuntimeError(f"vLLM request {request_id} did not produce a final output.")
    wall_ms = (time.perf_counter() - started) * 1000.0
    incremental_arrivals = len(set(arrivals)) > 1
    itl_ms = (
        sum(right - left for left, right in zip(arrivals, arrivals[1:]))
        / (len(arrivals) - 1)
        if len(arrivals) > 1 and incremental_arrivals
        else None
    )
    return request_id, output, {
        # vLLM-Metal's offline V1 runner currently emits a whole completion
        # from one engine step. Do not mislabel that boundary as token TTFT.
        "ttft_ms": arrivals[0] if arrivals and incremental_arrivals else None,
        "itl_ms": itl_ms,
        "completion_latency_ms": wall_ms,
    }


def _run_batch(
    llm,
    bridge,
    sampling,
    prompts,
    *,
    key=None,
    source_tokens=0,
    cache_salt=None,
):
    """Run one genuinely concurrent V1 wave over a shared selected resource."""

    if key is not None:
        from pra_vllm.v1_native import native_request_cache_salt

        cache_salt = native_request_cache_salt(
            (key,),
            selected_token_count=source_tokens,
            source_position_base=source_tokens,
        )
    started = time.perf_counter()
    request_ids = []
    for prompt_tokens in prompts:
        [request_id] = llm.enqueue(
            _prompt(prompt_tokens, cache_salt), sampling, use_tqdm=False
        )
        request_ids.append(str(request_id))
        if key is not None:
            bridge.register(
                request_id,
                (key,),
                selected_token_count=source_tokens,
                source_position_base=source_tokens,
            )
    outputs = llm.wait_for_completion(use_tqdm=False)
    wall_ms = (time.perf_counter() - started) * 1000.0
    by_id = {str(output.request_id): output for output in outputs}
    ordered = [by_id[request_id.split("-", 1)[0]] for request_id in request_ids]
    return request_ids, ordered, {
        "ttft_ms": None,
        "itl_ms": None,
        "completion_latency_ms": wall_ms,
        "requests_per_second": len(request_ids) / max(wall_ms / 1000.0, 1e-9),
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
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--reserve-blocks", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm
    from pra_vllm.v1_native import VLLMMetalV1NativeBridge
    from vllm import LLM, SamplingParams

    manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_seqs=max(1, args.concurrency),
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=args.reserve_blocks)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    ingestion_sampling = SamplingParams(temperature=0, max_tokens=1)
    rows = []
    try:
        for example in examples:
            prepared_at = time.perf_counter()
            candidate_tokens = list(
                tokenizer.encode(example.candidate_source, add_special_tokens=False)
            )
            raw_source = _bounded_source(
                tokenizer, example.selected_source, args.max_source_tokens
            )
            source = _aligned(raw_source, tokenizer, bridge.block_size)
            text_preparation_ms = (time.perf_counter() - prepared_at) * 1000.0
            key = f"matched-{example.dataset}-{example.example_id}"
            e0_source_salt = f"e0-source-{example.selection.selection_id}"

            # Prime E0's ordinary APC with source-only pages. E0 then prefills
            # only the query suffix, matching E2's query-kernel geometry.
            started = time.perf_counter()
            _run(
                llm,
                bridge,
                ingestion_sampling,
                source,
                cache_salt=e0_source_salt,
            )
            e0_ingestion_ms = (time.perf_counter() - started) * 1000.0

            # Native PRA ingestion must use vLLM's own prefill path. A generic
            # MLX cache encoder is not numerically identical to vLLM-Metal.
            observation_start = len(bridge.prefill_page_observations())
            started = time.perf_counter()
            _run(
                llm,
                bridge,
                ingestion_sampling,
                source,
                cache_salt=f"e2-ingest-{example.selection.selection_id}",
            )
            observations = bridge.prefill_page_observations()[observation_start:]
            fresh = [row for row in observations if row["scheduler_cache_start"] == 0]
            if not fresh:
                raise RuntimeError("vLLM did not expose native source-ingestion pages.")
            page_count = len(source) // bridge.block_size
            source_blocks = list(fresh[0]["block_ids_by_group"][0][:page_count])
            from pra_vllm.v1_native import capture_paged_memory

            memory = capture_paged_memory(bridge, source_blocks, len(source))
            encode_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            bridge.materialize(key, memory)
            materialize_ms = (time.perf_counter() - started) * 1000.0

            requests = regime_schedule(
                example.question,
                warm_repeats=args.warm_repeats,
                multi_query_count=args.multi_query_count,
                concurrency=args.concurrency,
            )
            for request in (
                item
                for item in requests
                if item.regime != "concurrent_shared_resource"
            ):
                query = list(
                    tokenizer.encode(request.query.text, add_special_tokens=False)
                )
                reused = request.regime != "cold_one_shot"
                for condition in ("e0_selected_text", "e2_native_kv"):
                    prompt_tokens = source + query if condition == "e0_selected_text" else query
                    request_id, output, timing = _run(
                        llm,
                        bridge,
                        sampling,
                        prompt_tokens,
                        key=key if condition == "e2_native_kv" else None,
                        source_tokens=len(source),
                        cache_salt=(
                            e0_source_salt
                            if condition == "e0_selected_text"
                            else None
                        ),
                    )
                    text = str(output.outputs[0].text).strip()
                    exact, f1 = _metrics(text, example.answer)
                    native = condition == "e2_native_kv"
                    ingestion_ms = (
                        encode_ms + materialize_ms if native else e0_ingestion_ms
                    )
                    rows.append(
                        benchmark_row(
                            condition=condition,
                            selection=example.selection,
                            request=request,
                            output=text,
                            metrics=benchmark_metrics(
                                exact_match=exact,
                                token_f1=f1,
                                gold_answer_logprob=None,
                                evidence_recall=example.evidence_recall,
                                candidate_tokens=len(candidate_tokens),
                                selected_source_tokens=len(source),
                                visible_prompt_tokens=len(prompt_tokens),
                                selected_native_kv_tokens=len(source) if native else 0,
                                active_detail_bytes=memory.nbytes if native else 0,
                                retained_detail_bytes=memory.nbytes if native else 0,
                                text_preparation_ms=text_preparation_ms,
                                kv_encode_ms=encode_ms if native else None,
                                index_construction_ms=(
                                    materialize_ms if native else None
                                ),
                                time_to_usable_context_ms=(
                                    text_preparation_ms + ingestion_ms
                                ),
                                ttft_ms=timing["ttft_ms"],
                                itl_ms=timing["itl_ms"],
                                total_latency_ms=float(
                                    timing["completion_latency_ms"]
                                ),
                                generated_tokens=len(output.outputs[0].token_ids),
                                ordinary_prefix_cache_hit_tokens=int(
                                    output.num_cached_tokens
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
                                "request_id": str(request_id),
                                "gold_answer": example.answer,
                                "selected_source_tokens_before_alignment": len(
                                    raw_source
                                ),
                                "num_cache_creation_tokens": int(
                                    output.num_cache_creation_tokens
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
            concurrent_requests = tuple(
                item
                for item in requests
                if item.regime == "concurrent_shared_resource"
            )
            for condition in ("e0_selected_text", "e2_native_kv"):
                native = condition == "e2_native_kv"
                queries = [
                    list(
                        tokenizer.encode(
                            request.query.text, add_special_tokens=False
                        )
                    )
                    for request in concurrent_requests
                ]
                prompts = [
                    query if native else source + query for query in queries
                ]
                request_ids, outputs, timing = _run_batch(
                    llm,
                    bridge,
                    sampling,
                    prompts,
                    key=key if native else None,
                    source_tokens=len(source),
                    cache_salt=None if native else e0_source_salt,
                )
                for request, query, request_id, output in zip(
                    concurrent_requests, queries, request_ids, outputs
                ):
                    text = str(output.outputs[0].text).strip()
                    exact, f1 = _metrics(text, example.answer)
                    ingestion_ms = (
                        encode_ms + materialize_ms if native else e0_ingestion_ms
                    )
                    rows.append(
                        benchmark_row(
                            condition=condition,
                            selection=example.selection,
                            request=request,
                            output=text,
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
                                kv_encode_ms=encode_ms if native else None,
                                index_construction_ms=(
                                    materialize_ms if native else None
                                ),
                                time_to_usable_context_ms=(
                                    text_preparation_ms + ingestion_ms
                                ),
                                ttft_ms=None,
                                itl_ms=None,
                                total_latency_ms=float(
                                    timing["completion_latency_ms"]
                                ),
                                generated_tokens=len(output.outputs[0].token_ids),
                                ordinary_prefix_cache_hit_tokens=int(
                                    output.num_cached_tokens
                                ),
                                pra_hot_hit=native,
                                pra_warm_hit=False,
                                bytes_read=memory.nbytes if native else 0,
                                bytes_promoted=0,
                                bytes_avoided=memory.nbytes if native else 0,
                                duplicate_physical_kv_avoided_bytes=(
                                    memory.nbytes
                                    if native and request.request_ordinal > 0
                                    else 0
                                ),
                                requests_per_second=float(
                                    timing["requests_per_second"]
                                ),
                            ),
                            extra={
                                "dataset": example.dataset,
                                "seed": example.seed,
                                "request_id": str(request_id),
                                "gold_answer": example.answer,
                                "selected_source_tokens_before_alignment": len(
                                    raw_source
                                ),
                                "num_cache_creation_tokens": int(
                                    output.num_cache_creation_tokens
                                ),
                                "concurrency_execution": (
                                    "vllm_v1_continuous_batch"
                                ),
                            },
                        )
                    )
            # Reuse is measured by the two requests above. Unrelated examples
            # must return their reserved pages before the next source is loaded.
            bridge.release(key)
    finally:
        bridge.close()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper6_cross_engine_matched_e0_e2_vllm_v2",
        "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "page_alignment_required": True,
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
