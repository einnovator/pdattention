"""Run the first pretrained Qwen PRA correctness and smoke milestone."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata, write_artifacts
from pra_torch.hf import PRAHFConfig, inject_pra


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def synchronize(device: torch.device) -> None:
    """Complete queued CUDA work before phase timings and memory readings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_shapes(cache) -> list[list[int]]:
    """Serialize native K/V shapes from current and legacy HF Cache layouts."""
    if cache is None:
        return []
    if hasattr(cache, "layers"):
        rows = []
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None:
                rows.append([*keys.shape, *values.shape])
        return rows
    if hasattr(cache, "key_cache"):
        return [[*key.shape, *value.shape] for key, value in zip(cache.key_cache, cache.value_cache)]
    return []


def timed_call(fn, device: torch.device):
    """Return one synchronized wall-clock duration and function result."""
    synchronize(device)
    started = time.perf_counter()
    result = fn()
    synchronize(device)
    return time.perf_counter() - started, result


def compare_tensors(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    """Compute exact and floating-point parity diagnostics."""
    difference = (expected.float() - actual.float()).abs()
    return {
        "exact": bool(torch.equal(expected, actual)),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
        "finite": bool(torch.isfinite(actual).all()),
    }


def run(args) -> dict:
    """Execute parity first, then explicit-reference and implicit-head smoke probes."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    model.config.use_cache = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prompt = "The capital of Portugal is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        baseline_seconds, baseline = timed_call(
            lambda: model(**inputs, output_hidden_states=True, use_cache=True), device
        )
        baseline_generation_seconds, baseline_tokens = timed_call(
            lambda: model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False), device
        )
    baseline_logits = baseline.logits.detach().cpu()
    baseline_hidden = [hidden.detach().cpu() for hidden in baseline.hidden_states]
    baseline_cache_shapes = cache_shapes(baseline.past_key_values)

    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(-1,),
            model_max_context_tokens=args.native_limit,
            max_prompt_direct_tokens=args.direct_tokens,
            encoding_block_tokens=args.encoding_block_tokens,
            routing_chunk_tokens=args.routing_chunk_tokens,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=2,
            trigger_threshold=float("-inf"),
            kv_cache_residency=args.kv_residency,
            kv_cache_pin_memory=args.kv_residency == "cpu" and device.type == "cuda",
            kv_cache_non_blocking=args.kv_residency == "cpu" and device.type == "cuda",
        ),
    )
    with torch.no_grad():
        adapted_seconds, adapted = timed_call(
            lambda: handle.model(**inputs, output_hidden_states=True, use_cache=True), device
        )
        adapted_generation_seconds, adapted_tokens = timed_call(
            lambda: handle.model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False), device
        )
    hidden_differences = [
        float((expected.float() - actual.detach().cpu().float()).abs().max().item())
        for expected, actual in zip(baseline_hidden, adapted.hidden_states)
    ]
    parity = {
        "logits": compare_tensors(baseline_logits, adapted.logits.detach().cpu()),
        "max_hidden_state_difference": max(hidden_differences),
        "greedy_token_agreement": float((baseline_tokens == adapted_tokens).float().mean().item()),
        "greedy_sequences_equal": bool(torch.equal(baseline_tokens, adapted_tokens)),
        "baseline_cache_shapes": baseline_cache_shapes,
        "adapted_cache_shapes": cache_shapes(adapted.past_key_values),
        "cache_shapes_equal": baseline_cache_shapes == cache_shapes(adapted.past_key_values),
        "baseline_prefill_seconds": baseline_seconds,
        "adapted_prefill_seconds": adapted_seconds,
        "baseline_generation_seconds": baseline_generation_seconds,
        "adapted_generation_seconds": adapted_generation_seconds,
    }
    if not (
        parity["logits"]["max_abs_difference"] == 0.0
        and parity["max_hidden_state_difference"] == 0.0
        and parity["greedy_sequences_equal"]
        and parity["cache_shapes_equal"]
    ):
        raise AssertionError(f"Disabled-PRA parity gate failed: {parity}")

    reference_text = (
        "Portugal's capital is Lisbon. Lisbon stands on the Tagus River and is the country's largest city."
    )
    reference_ids = tokenizer(reference_text, return_tensors="pt", add_special_tokens=False).input_ids
    cold_seconds, entry = timed_call(
        lambda: handle.add_reference("mem://portugal", reference_ids, text=reference_text), device
    )
    native_kv_heads = int(next(iter(entry.layer_memory.values())).chunks[0].token_kv.k.shape[1])
    handle.set_memory_enabled(True)
    with torch.no_grad():
        warm_seconds, memory_output = timed_call(
            lambda: handle.model(**inputs, use_cache=False), device
        )
        memory_generation_seconds, memory_tokens = timed_call(
            lambda: handle.model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False), device
        )
    adapter = next(iter(handle.adapters.values()))
    explicit_selected = [
        hit.as_trace_dict() for row in adapter.last_selected_chunks for hit in row
    ]
    explicit_diagnostics = handle.diagnostics_by_layer()

    long_text = " ".join(
        ["Earlier context records Lisbon as Portugal's capital and locates it beside the Tagus."]
        * args.long_prompt_repetitions
    )
    long_ids = tokenizer(long_text, return_tensors="pt", add_special_tokens=False).input_ids
    prepared = handle.prepare_long_prompt(long_ids)
    with torch.no_grad():
        head_seconds, head_output = timed_call(
            lambda: handle.model(
                input_ids=prepared.input_ids,
                attention_mask=prepared.attention_mask,
                position_ids=prepared.position_ids,
                use_cache=False,
            ),
            device,
        )

    result = {
        "runtime": runtime_metadata(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": str(dtype),
            "attention_backend": "eager",
            "num_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "head_dim": model.config.head_dim,
            "rope_theta": model.config.rope_theta,
            "max_position_embeddings": model.config.max_position_embeddings,
            "sliding_window": getattr(model.config, "sliding_window", None),
            "pra_layers": sorted(handle.adapters),
        },
        "settings": vars(args),
        "parity": parity,
        "native_kv_replay": {
            "reference_tokens": int(reference_ids.shape[1]),
            "native_kv_heads": native_kv_heads,
            "expanded_to_query_heads_in_cache": native_kv_heads == model.config.num_attention_heads,
            "cold_reference_build_seconds": cold_seconds,
            "warm_query_seconds": warm_seconds,
            "generation_seconds": memory_generation_seconds,
            "finite_logits": bool(torch.isfinite(memory_output.logits).all()),
            "generated_text": tokenizer.decode(memory_tokens[0], skip_special_tokens=True),
            "selected": explicit_selected,
            "diagnostics": explicit_diagnostics,
        },
        "implicit_head": {
            "logical_prompt_tokens": int(long_ids.shape[1]),
            "head_tokens": prepared.head_tokens,
            "direct_tail_tokens": int(prepared.input_ids.shape[1]),
            "tail_position_start": int(prepared.position_ids[0, 0]),
            "tail_position_end": int(prepared.position_ids[0, -1]),
            "query_seconds": head_seconds,
            "finite_logits": bool(torch.isfinite(head_output.logits).all()),
            "max_native_operation_tokens": handle.max_native_operation_tokens,
            "native_limit_violations": handle.native_limit_violations,
            "diagnostics": handle.diagnostics_by_layer(),
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if not result["native_kv_replay"]["finite_logits"] or not result["implicit_head"]["finite_logits"]:
        raise AssertionError("PRA smoke produced non-finite logits.")
    if result["implicit_head"]["native_limit_violations"]:
        raise AssertionError("PRA exceeded the configured native-operation limit.")
    if not math.isfinite(result["native_kv_replay"]["warm_query_seconds"]):
        raise AssertionError("Invalid warm-query timing.")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--native-limit", type=int, default=256)
    parser.add_argument("--direct-tokens", type=int, default=128)
    parser.add_argument("--encoding-block-tokens", type=int, default=64)
    parser.add_argument("--routing-chunk-tokens", type=int, default=32)
    parser.add_argument("--memory-tokens", type=int, default=64)
    parser.add_argument("--long-prompt-repetitions", type=int, default=24)
    parser.add_argument("--kv-residency", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "qwen",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    artifact = run(arguments)
    json_path, csv_path = write_artifacts(artifact, arguments.output_dir, "qwen3_0_6b_first_night")
    print(json_path)
    print(csv_path)
