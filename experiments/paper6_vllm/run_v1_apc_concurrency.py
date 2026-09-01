"""Measure native PRA with APC and concurrent vLLM-Metal V1 requests."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path


SEEDS = (11, 23, 37, 53, 71)
CONCURRENCY = (1, 2, 4, 8)
EXPECTED = "7391"


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return values


def _aligned_source_tokens(tokenizer, text: str, block_size: int) -> list[int]:
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    padding = list(tokenizer.encode(" archive", add_special_tokens=False))
    if not padding:
        raise RuntimeError("Tokenizer did not produce a page-alignment token.")
    while len(tokens) % block_size:
        tokens.append(int(padding[0]))
    return tokens


def _stable_prefix(seed: int) -> str:
    sentence = (
        f"Session {seed} contains ordinary conversation history and procedural notes. "
        "This stable text is eligible for exact prefix reuse. "
    )
    return (sentence * 6) + "\n"


def _prompt(prefix: str, slot: int) -> str:
    return (
        prefix
        + f"Request slot {slot}. What is the authoritative archived verification "
        "code? Answer with the four digits only:"
    )


def _enqueue_batch(
    llm,
    sampling,
    prompts: list[str],
    cache_salts: list[str | None] | None = None,
):
    request_ids: list[str] = []
    salts = cache_salts or [None] * len(prompts)
    if len(salts) != len(prompts):
        raise ValueError("Each prompt must have one APC salt decision.")
    for prompt, cache_salt in zip(prompts, salts):
        payload = (
            prompt
            if cache_salt is None
            else {"prompt": prompt, "cache_salt": cache_salt}
        )
        [request_id] = llm.enqueue(payload, sampling, use_tqdm=False)
        request_ids.append(str(request_id))
    return request_ids


def _wait_batch(llm, expected: int):
    started = time.perf_counter()
    outputs = llm.wait_for_completion(use_tqdm=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(outputs) != expected:
        raise RuntimeError(f"Expected {expected} outputs, received {len(outputs)}.")
    by_id = {str(output.request_id): output for output in outputs}
    return by_id, elapsed_ms


def _output_row(output) -> dict[str, object]:
    text = str(output.outputs[0].text)
    metrics = getattr(output, "metrics", None)
    arrival = getattr(metrics, "arrival_time", None)
    first_scheduled = getattr(metrics, "first_scheduled_time", None)
    first_token = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)
    token_ids = list(getattr(output.outputs[0], "token_ids", ()) or ())

    def elapsed_ms(end, start):
        return None if end is None or start is None else (end - start) * 1000.0

    return {
        "output_request_id": str(output.request_id),
        "text": text,
        "exact_recovery": EXPECTED in text,
        "num_cached_tokens": output.num_cached_tokens,
        "num_cache_creation_tokens": output.num_cache_creation_tokens,
        "generated_tokens": len(token_ids),
        "queue_ms": elapsed_ms(first_scheduled, arrival),
        "ttft_ms": elapsed_ms(first_token, arrival),
        "request_latency_ms": elapsed_ms(finished, arrival),
        "tpot_ms": (
            elapsed_ms(finished, first_token) / max(len(token_ids) - 1, 1)
            if finished is not None and first_token is not None and token_ids
            else None
        ),
    }


def _command_value(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _runtime_metadata() -> dict[str, object]:
    """Bind an engine receipt to the exact Apple host and kernel source."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware_model": _command_value(["sysctl", "-n", "hw.model"]),
        "cpu_brand": _command_value(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_memory_bytes": int(
            _command_value(["sysctl", "-n", "hw.memsize"]) or 0
        ),
        "git_commit": _command_value(["git", "rev-parse", "HEAD"]),
        "vllm_metal_source_commit": _command_value(
            [
                "git",
                "-C",
                str(Path.home() / "git/llm-engine-lab/src/vllm-metal"),
                "rev-parse",
                "HEAD",
            ]
        ),
    }


def _run_condition(
    llm,
    bridge,
    sampling,
    prompts: list[str],
    *,
    logical_key: str | None,
    selected_tokens: int,
    register_slots: set[int] | None = None,
) -> dict[str, object]:
    cache_salts: list[str | None] = [None] * len(prompts)
    if logical_key is not None:
        from pra_vllm.v1_native import native_request_cache_salt

        native_salt = native_request_cache_salt(
            (logical_key,),
            selected_token_count=selected_tokens,
            source_position_base=selected_tokens,
        )
        cache_salts = [
            native_salt
            if register_slots is None or slot in register_slots
            else None
            for slot in range(len(prompts))
        ]
    request_ids = _enqueue_batch(llm, sampling, prompts, cache_salts)
    if logical_key is not None:
        for slot, request_id in enumerate(request_ids):
            if register_slots is not None and slot not in register_slots:
                continue
            bridge.register(
                request_id,
                (logical_key,),
                selected_token_count=selected_tokens,
                source_position_base=selected_tokens,
            )
    outputs, elapsed_ms = _wait_batch(llm, len(prompts))
    return {
        "request_ids": request_ids,
        "elapsed_ms": elapsed_ms,
        "requests_per_second": len(prompts) / (elapsed_ms / 1000.0),
        # Offline V1 returns an internal ``<external>-<wave>`` ID from enqueue,
        # while RequestOutput intentionally exposes the external numeric ID.
        "outputs": [
            {
                "selected_registered": (
                    logical_key is not None
                    and (register_slots is None or slot in register_slots)
                ),
                **_output_row(outputs[request_id.split("-", 1)[0]]),
            }
            for slot, request_id in enumerate(request_ids)
        ],
        "scheduler_observations": list(
            bridge.scheduler_observations(request_ids)
        ),
        "native_apc_salt_applied": [salt is not None for salt in cache_salts],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-blocks", type=int, default=64)
    parser.add_argument("--seeds", type=_integers, default=SEEDS)
    parser.add_argument("--concurrency", type=_integers, default=CONCURRENCY)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import vllm
    from pra_mlx.native import encode_native_memory
    from pra_vllm.v1_native import VLLMMetalV1NativeBridge
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.concurrency),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=args.reserve_blocks)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=12)
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        correct_tokens = _aligned_source_tokens(
            tokenizer,
            f"Authoritative archived fact: verification code {EXPECTED}. Series {seed}.",
            bridge.block_size,
        )
        correct_key = f"apc-seed-{seed}-correct"
        correct_memory = encode_native_memory(runner.model, correct_tokens)
        bridge.materialize(correct_key, correct_memory)

        wrong_code = f"{(int(EXPECTED) + seed + 1000) % 10000:04d}"
        wrong_tokens = _aligned_source_tokens(
            tokenizer,
            f"Authoritative archived fact: verification code {wrong_code}. Series {seed}.",
            bridge.block_size,
        )
        if len(wrong_tokens) != len(correct_tokens):
            raise RuntimeError("Wrong-memory control changed physical page geometry.")
        wrong_key = f"apc-seed-{seed}-wrong"
        bridge.materialize(wrong_key, encode_native_memory(runner.model, wrong_tokens))

        prefix = _stable_prefix(seed)
        # Populate only the ordinary APC namespace before selected pages are used.
        llm.generate(prefix + "Warm this ordinary prefix.", sampling, use_tqdm=False)

        for concurrency in args.concurrency:
            prompts = [_prompt(prefix, slot) for slot in range(concurrency)]
            condition = _run_condition(
                llm,
                bridge,
                sampling,
                prompts,
                logical_key=correct_key,
                selected_tokens=len(correct_tokens),
            )
            rows.append(
                {
                    "seed": seed,
                    "concurrency": concurrency,
                    "condition": "native_pra_plus_apc",
                    "selected_tokens": len(correct_tokens),
                    "active_native_kv_bytes": correct_memory.nbytes,
                    **condition,
                }
            )

        control_prompts = [
            _prompt(prefix, slot) for slot in range(max(args.concurrency))
        ]
        mixed_slots = set(range(0, len(control_prompts), 2))
        mixed = _run_condition(
            llm,
            bridge,
            sampling,
            control_prompts,
            logical_key=correct_key,
            selected_tokens=len(correct_tokens),
            register_slots=mixed_slots,
        )
        rows.append(
            {
                "seed": seed,
                "concurrency": max(args.concurrency),
                "condition": "mixed_selected_and_ordinary",
                "selected_tokens": len(correct_tokens),
                **mixed,
            }
        )
        for name, key in (
            ("disabled_after_native", None),
            ("wrong_memory_plus_apc", wrong_key),
        ):
            condition = _run_condition(
                llm,
                bridge,
                sampling,
                control_prompts,
                logical_key=key,
                selected_tokens=len(correct_tokens),
            )
            for output in condition["outputs"]:
                output["wrong_memory_follows_wrong_code"] = wrong_code in str(
                    output["text"]
                )
            rows.append(
                {
                    "seed": seed,
                    "concurrency": max(args.concurrency),
                    "condition": name,
                    "selected_tokens": 0 if key is None else len(correct_tokens),
                    **condition,
                }
            )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_metal_v1_apc_concurrency_v1",
        "evidence_tier": "CONTROLLED",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "concurrency_levels": list(args.concurrency),
        "seeds": list(args.seeds),
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "ordinary_prefix_namespace_used_for_pra": False,
        "native_apc_identity": "sha256_logical_selection_and_geometry",
        "scheduler_observation_point": "before_pra_attention_augmentation",
        "runtime_metadata": _runtime_metadata(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    bridge.close()


if __name__ == "__main__":
    main()
