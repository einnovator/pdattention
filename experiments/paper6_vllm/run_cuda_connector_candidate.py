"""Validate semantic-keyed native K/V loading through vLLM's CUDA connector."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from pra_vllm.cuda_protocol import CudaConnectorCommand


EXPECTED = "4821"
SEEDS = (11, 23, 37, 71, 101)


def _aligned(tokens: list[int], block_size: int, padding_token: int) -> list[int]:
    result = list(tokens)
    while len(result) % block_size:
        result.append(int(padding_token))
    return result


def _prompt(tokens: list[int], cache_salt: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"prompt_token_ids": list(tokens)}
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    return payload


def _generate(llm, sampling, tokens: list[int], *, cache_salt: str | None = None):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = llm.generate(
        _prompt(tokens, cache_salt), sampling, use_tqdm=False
    )[0]
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return output, {
        "completion_ms": elapsed_ms,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cached_prompt_tokens": int(output.num_cached_tokens),
    }


def _text(output) -> str:
    return str(output.outputs[0].text).strip()


def _token_ids(output) -> list[int]:
    return list(map(int, output.outputs[0].token_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage", type=Path, default=Path(".pra/vllm-cuda-connector")
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    run_id = uuid.uuid4().hex[:12]
    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_seqs=1,
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
    block_size = int(llm.vllm_config.cache_config.block_size)
    padding = tokenizer.encode(" archive", add_special_tokens=False)
    if not padding:
        raise RuntimeError("Tokenizer did not produce a page-alignment token.")
    query = tokenizer.encode(
        "\nWhat is the authoritative archived verification code? "
        "Answer with the four digits only:",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode("\nQuestion:", add_special_tokens=False)
    if not suffix:
        suffix = [padding[0]]
    placeholder_token = int(padding[0])
    sampling = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, ignore_eos=True
    )
    store_sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        source = _aligned(
            tokenizer.encode(
                f"Authoritative archived fact: verification code {EXPECTED}. "
                f"Archive series {seed}.",
                add_special_tokens=False,
            ),
            block_size,
            placeholder_token,
        )
        wrong_code = f"{(int(EXPECTED) + seed + 1000) % 10000:04d}"
        wrong_source = _aligned(
            tokenizer.encode(
                f"Authoritative archived fact: verification code {wrong_code}. "
                f"Archive series {seed}.",
                add_special_tokens=False,
            ),
            block_size,
            placeholder_token,
        )
        if len(source) != len(wrong_source):
            raise RuntimeError("Wrong-memory control changed source geometry.")

        correct_key = f"cuda-{run_id}-{seed}-correct"
        wrong_key = f"cuda-{run_id}-{seed}-wrong"
        correct_store = CudaConnectorCommand("store", correct_key, len(source))
        wrong_store = CudaConnectorCommand("store", wrong_key, len(wrong_source))
        _, correct_ingestion = _generate(
            llm,
            store_sampling,
            source + suffix[:1],
            cache_salt=correct_store.cache_salt(),
        )
        _, wrong_ingestion = _generate(
            llm,
            store_sampling,
            wrong_source + suffix[:1],
            cache_salt=wrong_store.cache_salt(),
        )

        full, full_metrics = _generate(llm, sampling, source + query)
        disabled, disabled_metrics = _generate(llm, sampling, query)
        placeholders = [placeholder_token] * len(source)
        native, native_metrics = _generate(
            llm,
            sampling,
            placeholders + query,
            cache_salt=CudaConnectorCommand(
                "load", correct_key, len(source)
            ).cache_salt(),
        )
        post_cleanup, post_cleanup_metrics = _generate(llm, sampling, query)
        wrong, wrong_metrics = _generate(
            llm,
            sampling,
            placeholders + query,
            cache_salt=CudaConnectorCommand(
                "load", wrong_key, len(source)
            ).cache_salt(),
        )

        resource_dir = storage / __import__("hashlib").sha256(
            correct_key.encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "seed": seed,
                "source_tokens": len(source),
                "visible_query_tokens": len(query),
                "scheduler_placeholder_tokens": len(source),
                "stored_native_bytes": sum(
                    path.stat().st_size for path in resource_dir.glob("*")
                ),
                "expected_code": EXPECTED,
                "wrong_code": wrong_code,
                "full_output": _text(full),
                "native_output": _text(native),
                "disabled_output": _text(disabled),
                "post_cleanup_output": _text(post_cleanup),
                "wrong_output": _text(wrong),
                "full_token_ids": _token_ids(full),
                "native_token_ids": _token_ids(native),
                "exact_output_parity": _token_ids(full) == _token_ids(native),
                "full_exact_recovery": EXPECTED in _text(full),
                "native_exact_recovery": EXPECTED in _text(native),
                "disabled_leak": EXPECTED in _text(disabled),
                "post_cleanup_leak": EXPECTED in _text(post_cleanup),
                "wrong_memory_follows_wrong_code": wrong_code in _text(wrong),
                "ingestion": correct_ingestion,
                "wrong_ingestion": wrong_ingestion,
                "full": full_metrics,
                "native": native_metrics,
                "disabled": disabled_metrics,
                "post_cleanup": post_cleanup_metrics,
                "wrong": wrong_metrics,
            }
        )

    payload = {
        "schema_version": "paper6-vllm-cuda-connector-candidate-v1",
        "evidence_tier": "CONTROLLED_CUDA_NATIVE_TRANSFER",
        "integration_status": "E2_CANDIDATE_PREFIX_SHAPED",
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "model_id": args.model,
        "device": torch.cuda.get_device_name(0),
        "block_size": block_size,
        "apc_enabled": False,
        "source_content_visible": False,
        "source_slots_scheduler_visible": True,
        "attention_normalization": "ordinary_vllm_self_attention_over_loaded_source_and_query",
        "rows": rows,
        "summary": {
            "seeds": len(rows),
            "exact_output_pairs": sum(bool(row["exact_output_parity"]) for row in rows),
            "full_exact_recovery": sum(bool(row["full_exact_recovery"]) for row in rows),
            "native_exact_recovery": sum(
                bool(row["native_exact_recovery"]) for row in rows
            ),
            "disabled_leaks": sum(bool(row["disabled_leak"]) for row in rows),
            "post_cleanup_leaks": sum(
                bool(row["post_cleanup_leak"]) for row in rows
            ),
            "wrong_memory_follows_wrong_code": sum(
                bool(row["wrong_memory_follows_wrong_code"]) for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
