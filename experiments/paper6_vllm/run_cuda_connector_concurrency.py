"""Exercise the CUDA connector candidate under batched request allocation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from experiments.paper6_vllm.run_cuda_connector_candidate import (
    EXPECTED,
    _aligned,
    _prompt,
)
from pra_vllm.cuda_protocol import CudaConnectorCommand


def _generated(output: Any) -> tuple[str, list[int]]:
    sample = output.outputs[0]
    return str(sample.text).strip(), list(map(int, sample.token_ids))


def _batch_generate(llm: Any, sampling: Any, prompts: list[dict[str, object]]):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    return outputs, {
        "requests": len(prompts),
        "completion_ms": elapsed * 1000.0,
        "requests_per_second": len(prompts) / max(elapsed, 1e-9),
        "output_tokens_per_second": output_tokens / max(elapsed, 1e-9),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _condition_row(
    *,
    concurrency: int,
    condition: str,
    outputs: list[Any],
    metrics: dict[str, Any],
    expected_by_request: list[str | None],
    forbidden_by_request: list[str],
) -> dict[str, Any]:
    decoded = [_generated(output) for output in outputs]
    recovered = [
        expected is not None and expected in text
        for (text, _), expected in zip(decoded, expected_by_request)
    ]
    leaks = [
        forbidden in text
        for (text, _), forbidden in zip(decoded, forbidden_by_request)
    ]
    return {
        "concurrency": concurrency,
        "condition": condition,
        "outputs": [text for text, _ in decoded],
        "output_token_ids": [tokens for _, tokens in decoded],
        "expected_recoveries": sum(recovered),
        "expected_requests": sum(value is not None for value in expected_by_request),
        "forbidden_leaks": sum(leaks),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage", type=Path, default=Path(".pra/vllm-cuda-concurrency")
    )
    parser.add_argument("--concurrency", nargs="*", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    levels = tuple(sorted(set(args.concurrency)))
    if not levels or levels[0] <= 0:
        raise ValueError("Concurrency levels must be positive.")

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_seqs=max(levels),
        gpu_memory_utilization=0.72,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        kv_transfer_config={
            "kv_connector": "PRASemanticConnector",
            "kv_connector_module_path": "pra_vllm.cuda_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"storage_path": str(storage)},
        },
    )
    tokenizer = llm.get_tokenizer()
    block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    padding_token = tokenizer.encode(" archive", add_special_tokens=False)[0]
    query = tokenizer.encode(
        "\nWhat is the authoritative archived verification code? "
        "Answer with the four digits only:",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode("\nQuestion:", add_special_tokens=False)[:1]
    source = _aligned(
        tokenizer.encode(
            f"Authoritative archived fact: verification code {EXPECTED}.",
            add_special_tokens=False,
        ),
        block_size,
        padding_token,
    )
    wrong_code = "7394"
    wrong_source = _aligned(
        tokenizer.encode(
            f"Authoritative archived fact: verification code {wrong_code}.",
            add_special_tokens=False,
        ),
        block_size,
        padding_token,
    )
    if len(source) != len(wrong_source):
        raise RuntimeError("Wrong-memory control changed source geometry.")

    run_id = uuid.uuid4().hex[:12]
    correct_key = f"cuda-concurrency-{run_id}-correct"
    wrong_key = f"cuda-concurrency-{run_id}-wrong"
    store_sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    sampling = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, ignore_eos=True
    )
    for key, tokens in ((correct_key, source), (wrong_key, wrong_source)):
        _batch_generate(
            llm,
            store_sampling,
            [
                _prompt(
                    tokens + suffix,
                    CudaConnectorCommand("store", key, len(tokens)).cache_salt(),
                )
            ],
        )

    placeholders = [padding_token] * len(source)
    correct_salt = CudaConnectorCommand(
        "load", correct_key, len(source)
    ).cache_salt()
    wrong_salt = CudaConnectorCommand("load", wrong_key, len(source)).cache_salt()
    rows: list[dict[str, Any]] = []
    for concurrency in levels:
        native_prompts = [
            _prompt(placeholders + query, correct_salt) for _ in range(concurrency)
        ]
        native_outputs, native_metrics = _batch_generate(
            llm, sampling, native_prompts
        )
        rows.append(
            _condition_row(
                concurrency=concurrency,
                condition="shared_native",
                outputs=native_outputs,
                metrics=native_metrics,
                expected_by_request=[EXPECTED] * concurrency,
                forbidden_by_request=[wrong_code] * concurrency,
            )
        )

        mixed_prompts: list[dict[str, object]] = []
        mixed_expected: list[str | None] = []
        for index in range(concurrency):
            native = index % 2 == 0
            mixed_prompts.append(
                _prompt(placeholders + query, correct_salt)
                if native
                else _prompt(query)
            )
            mixed_expected.append(EXPECTED if native else None)
        mixed_outputs, mixed_metrics = _batch_generate(llm, sampling, mixed_prompts)
        rows.append(
            _condition_row(
                concurrency=concurrency,
                condition="mixed_native_ordinary",
                outputs=mixed_outputs,
                metrics=mixed_metrics,
                expected_by_request=mixed_expected,
                forbidden_by_request=[
                    wrong_code if expected is not None else EXPECTED
                    for expected in mixed_expected
                ],
            )
        )

        wrong_outputs, wrong_metrics = _batch_generate(
            llm,
            sampling,
            [_prompt(placeholders + query, wrong_salt) for _ in range(concurrency)],
        )
        rows.append(
            _condition_row(
                concurrency=concurrency,
                condition="wrong_native",
                outputs=wrong_outputs,
                metrics=wrong_metrics,
                expected_by_request=[wrong_code] * concurrency,
                forbidden_by_request=[EXPECTED] * concurrency,
            )
        )

    payload = {
        "schema_version": "paper6-vllm-cuda-connector-concurrency-v1",
        "evidence_tier": "CONTROLLED_CUDA_BATCHED_NATIVE_TRANSFER",
        "integration_status": "E2_CANDIDATE_PREFIX_SHAPED",
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "model_id": args.model,
        "device": torch.cuda.get_device_name(0),
        "block_size": block_size,
        "source_tokens": len(source),
        "source_content_visible": False,
        "source_slots_scheduler_visible": True,
        "levels": list(levels),
        "rows": rows,
        "summary": {
            "requests": sum(int(row["requests"]) for row in rows),
            "expected_recoveries": sum(int(row["expected_recoveries"]) for row in rows),
            "expected_requests": sum(int(row["expected_requests"]) for row in rows),
            "forbidden_leaks": sum(int(row["forbidden_leaks"]) for row in rows),
            "peak_allocated_bytes": max(int(row["peak_allocated_bytes"]) for row in rows),
            "median_requests_per_second": statistics.median(
                float(row["requests_per_second"]) for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
