"""Run selected native K/V through live vLLM-Metal V1 generation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


SEEDS = (11, 23, 37, 53, 71)
EXPECTED = "7391"


def _aligned_source_tokens(tokenizer, text: str, block_size: int) -> list[int]:
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    padding = list(tokenizer.encode(" archive", add_special_tokens=False))
    if not padding:
        raise RuntimeError("Tokenizer did not produce a page-alignment token.")
    while len(tokens) % block_size:
        tokens.append(int(padding[0]))
    return tokens


def _text(output) -> str:
    return str(output.outputs[0].text)


def _generate(llm, sampling, prompt: str):
    started = time.perf_counter()
    output = llm.generate(prompt, sampling, use_tqdm=False)[0]
    return _text(output), (time.perf_counter() - started) * 1000.0


def _generate_native(llm, bridge, sampling, prompt: str, logical_key: str, tokens: int):
    started = time.perf_counter()
    [request_id] = llm.enqueue(prompt, sampling, use_tqdm=False)
    bridge.register(
        request_id,
        (logical_key,),
        selected_token_count=tokens,
        source_position_base=tokens,
    )
    [output] = llm.wait_for_completion(use_tqdm=False)
    return request_id, _text(output), (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-blocks", type=int, default=32)
    args = parser.parse_args()

    # In-process V1 access is required to bind request IDs to driver pages.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import vllm
    from pra_mlx.native import encode_native_memory
    from pra_vllm.v1_native import VLLMMetalV1NativeBridge
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=False,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=args.reserve_blocks)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=12)
    query = (
        "What is the authoritative archived verification code? "
        "Answer with the four digits only:"
    )
    rows = []
    for seed in SEEDS:
        source_text = (
            f"Authoritative archived fact: verification code {EXPECTED}. "
            f"Archive series {seed}."
        )
        source_tokens = _aligned_source_tokens(tokenizer, source_text, bridge.block_size)
        memory = encode_native_memory(runner.model, source_tokens)
        logical_key = f"seed-{seed}-correct"
        bridge.materialize(logical_key, memory)

        wrong_code = f"{(int(EXPECTED) + seed + 1000) % 10000:04d}"
        wrong_tokens = _aligned_source_tokens(
            tokenizer,
            f"Authoritative archived fact: verification code {wrong_code}. "
            f"Archive series {seed}.",
            bridge.block_size,
        )
        if len(wrong_tokens) != len(source_tokens):
            raise RuntimeError("Causal-control memory changed physical page geometry.")
        wrong_memory = encode_native_memory(runner.model, wrong_tokens)
        wrong_key = f"seed-{seed}-wrong"
        bridge.materialize(wrong_key, wrong_memory)

        full_prompt = tokenizer.decode(source_tokens) + "\n" + query
        full_text, full_ms = _generate(llm, sampling, full_prompt)
        disabled_text, disabled_ms = _generate(llm, sampling, query)
        request_id, native_text, native_ms = _generate_native(
            llm, bridge, sampling, query, logical_key, len(source_tokens)
        )
        post_cleanup_text, post_cleanup_ms = _generate(llm, sampling, query)
        wrong_request_id, wrong_text, wrong_ms = _generate_native(
            llm, bridge, sampling, query, wrong_key, len(wrong_tokens)
        )
        rows.append(
            {
                "seed": seed,
                "source_tokens": len(source_tokens),
                "visible_query_tokens": len(
                    tokenizer.encode(query, add_special_tokens=False)
                ),
                "active_native_kv_bytes": memory.nbytes,
                "native_request_id": request_id,
                "wrong_memory_request_id": wrong_request_id,
                "full_context_output": full_text,
                "disabled_output": disabled_text,
                "native_output": native_text,
                "post_cleanup_disabled_output": post_cleanup_text,
                "wrong_memory_output": wrong_text,
                "full_context_exact_recovery": EXPECTED in full_text,
                "disabled_exact_recovery": EXPECTED in disabled_text,
                "native_exact_recovery": EXPECTED in native_text,
                "post_cleanup_leak": EXPECTED in post_cleanup_text,
                "wrong_memory_follows_wrong_code": wrong_code in wrong_text,
                "full_context_ms": full_ms,
                "disabled_ms": disabled_ms,
                "native_ms": native_ms,
                "post_cleanup_disabled_ms": post_cleanup_ms,
                "wrong_memory_ms": wrong_ms,
            }
        )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_metal_v1_live_native_generation_v1",
        "evidence_tier": "CONTROLLED",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "native_pra_status": "MEASURED_LIVE_V1_GENERATION",
        "attention_semantics": "one_softmax_selected_native_pages_plus_local_pages",
        "page_alignment_required": True,
        "ordinary_prefix_namespace_used": False,
        "reserve_blocks": args.reserve_blocks,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    bridge.close()


if __name__ == "__main__":
    main()
