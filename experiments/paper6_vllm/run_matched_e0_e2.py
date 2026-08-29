"""Compare matched selected-text and native-page QA on vLLM V1."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

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
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reserve-blocks", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm
    from pra_mlx.native import encode_native_memory
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
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=args.reserve_blocks)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    rows = []
    try:
        for example in examples:
            raw_source = _bounded_source(
                tokenizer, example.selected_source, args.max_source_tokens
            )
            source = _aligned(raw_source, tokenizer, bridge.block_size)
            query_text = (
                "Answer the question using the available evidence. Give only the "
                f"short answer.\nQuestion: {example.question}\nAnswer:"
            )
            query = list(tokenizer.encode(query_text, add_special_tokens=False))
            started = time.perf_counter()
            memory = encode_native_memory(runner.model, source)
            encode_ms = (time.perf_counter() - started) * 1000.0
            key = f"matched-{example.dataset}-{example.example_id}"
            started = time.perf_counter()
            bridge.materialize(key, memory)
            materialize_ms = (time.perf_counter() - started) * 1000.0

            for repeat in range(args.repeats):
                for condition in ("e0_selected_text", "e2_native_kv"):
                    prompt_tokens = source + query if condition == "e0_selected_text" else query
                    request_id, output, timing = _run(
                        llm,
                        bridge,
                        sampling,
                        prompt_tokens,
                        key=key if condition == "e2_native_kv" else None,
                        source_tokens=len(source),
                    )
                    text = str(output.outputs[0].text).strip()
                    exact, f1 = _metrics(text, example.answer)
                    rows.append(
                        {
                            "dataset": example.dataset,
                            "seed": example.seed,
                            "example_id": example.example_id,
                            "source_sha256": example.selected_source_sha256,
                            "condition": condition,
                            "repeat": repeat,
                            "reuse_state": "cold" if repeat == 0 else "warm",
                            "request_id": str(request_id),
                            "gold_answer": example.answer,
                            "output": text,
                            "exact_match": exact,
                            "token_f1": f1,
                            "visible_prompt_tokens": len(prompt_tokens),
                            "selected_source_tokens": len(source),
                            "selected_source_tokens_before_alignment": len(raw_source),
                            "selected_native_tokens": (
                                len(source) if condition == "e2_native_kv" else 0
                            ),
                            "selected_kv_bytes": memory.nbytes,
                            "one_time_ingestion_ms": (
                                encode_ms + materialize_ms
                                if condition == "e2_native_kv"
                                else None
                            ),
                            "num_cached_tokens": output.num_cached_tokens,
                            "num_cache_creation_tokens": output.num_cache_creation_tokens,
                            "generated_tokens": len(output.outputs[0].token_ids),
                            "resource_reused": repeat > 0,
                            **timing,
                        }
                    )
            # Reuse is measured by the two requests above. Unrelated examples
            # must return their reserved pages before the next source is loaded.
            bridge.release(key)
    finally:
        bridge.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_cross_engine_matched_e0_e2_vllm_v1",
        "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "page_alignment_required": True,
        "repeats": args.repeats,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
