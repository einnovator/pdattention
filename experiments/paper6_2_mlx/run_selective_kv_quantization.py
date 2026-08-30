"""Calibrate selective native-K/V int8 profiles on natural QA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import (
    _answer_logprob,
    _bounded_source,
    _metrics,
)
from experiments.paper6_2_mlx.run_live_storage_lifecycle import _common_prefix
from experiments.paper6_2_mlx.run_matched_e0_e2 import _generate_timed


def quantization_profiles(layer_count: int) -> dict[str, dict[str, object]]:
    """Return component/layer ablations with stable model-relative bands."""

    if layer_count < 2:
        raise ValueError("Selective quantization requires at least two layers.")
    half = layer_count // 2
    quarter = max(1, layer_count // 4)
    return {
        "lossless": {
            "quantization": "none",
            "layers": None,
            "keys": False,
            "values": False,
        },
        "all_kv": {
            "quantization": "int8",
            "layers": None,
            "keys": True,
            "values": True,
        },
        "all_keys": {
            "quantization": "int8",
            "layers": None,
            "keys": True,
            "values": False,
        },
        "all_values": {
            "quantization": "int8",
            "layers": None,
            "keys": False,
            "values": True,
        },
        "early_half_kv": {
            "quantization": "int8",
            "layers": tuple(range(half)),
            "keys": True,
            "values": True,
        },
        "late_half_kv": {
            "quantization": "int8",
            "layers": tuple(range(half, layer_count)),
            "keys": True,
            "values": True,
        },
        "late_quarter_kv": {
            "quantization": "int8",
            "layers": tuple(range(layer_count - quarter, layer_count)),
            "keys": True,
            "values": True,
        },
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
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from pra_mlx.native import (
        deserialize_native_memory,
        encode_native_memory,
        make_native_prompt_cache,
        serialize_native_memory,
    )

    _manifest, examples = load_matched_examples(
        args.manifest, args.dataset, args.cache_dir
    )
    examples = examples[: args.max_examples]
    model, tokenizer = load(args.model, revision=args.revision)
    layer_count = len(getattr(getattr(model, "model", model), "layers"))
    profiles = quantization_profiles(layer_count)
    rows = []

    for example in examples:
        source = _bounded_source(
            tokenizer, example.selected_source, args.max_source_tokens
        )
        query = list(tokenizer.encode(example.question, add_special_tokens=False))
        answer = list(
            tokenizer.encode(" " + example.answer, add_special_tokens=False)
        )
        native = encode_native_memory(model, source)
        baseline = _generate_timed(
            model,
            tokenizer,
            query,
            make_native_prompt_cache(model, native),
            args.max_new_tokens,
        )
        baseline_f1 = _metrics(str(baseline["output"]), example.answer)[1]
        baseline_logprob = _answer_logprob(
            model, query, answer, make_native_prompt_cache(model, native)
        )

        for profile_name, profile in profiles.items():
            encode_started = time.perf_counter()
            encoded = serialize_native_memory(
                native,
                quantization=str(profile["quantization"]),
                quantized_layers=profile["layers"],
                quantize_keys=bool(profile["keys"]),
                quantize_values=bool(profile["values"]),
            )
            encode_ms = (time.perf_counter() - encode_started) * 1000.0
            decode_started = time.perf_counter()
            restored = deserialize_native_memory(encoded)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            generated = _generate_timed(
                model,
                tokenizer,
                query,
                make_native_prompt_cache(model, restored),
                args.max_new_tokens,
            )
            output = str(generated["output"])
            f1 = _metrics(output, example.answer)[1]
            logprob = _answer_logprob(
                model, query, answer, make_native_prompt_cache(model, restored)
            )
            max_key_error = max(
                float(mx.max(mx.abs(left.keys - right.keys)).item())
                for left, right in zip(native.layers, restored.layers)
            )
            max_value_error = max(
                float(mx.max(mx.abs(left.values - right.values)).item())
                for left, right in zip(native.layers, restored.layers)
            )
            rows.append(
                {
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "profile": profile_name,
                    "quantization": profile["quantization"],
                    "quantized_layers": (
                        "all" if profile["layers"] is None else list(profile["layers"])
                    ),
                    "quantize_keys": profile["keys"],
                    "quantize_values": profile["values"],
                    "source_tokens": len(source),
                    "lossless_native_bytes": native.nbytes,
                    "serialized_bytes": len(encoded),
                    "serialized_ratio": len(encoded) / max(native.nbytes, 1),
                    "encode_ms": encode_ms,
                    "decode_ms": decode_ms,
                    "completion_latency_ms": generated["completion_latency_ms"],
                    "output": output,
                    "output_exact_vs_lossless": output == baseline["output"],
                    "first_token_equal": (
                        list(generated["output_token_ids"])[:1]
                        == list(baseline["output_token_ids"])[:1]
                    ),
                    "common_prefix_tokens": _common_prefix(
                        list(baseline["output_token_ids"]),
                        list(generated["output_token_ids"]),
                    ),
                    "answer_f1": f1,
                    "answer_f1_delta": f1 - baseline_f1,
                    "gold_answer_logprob": logprob,
                    "gold_answer_logprob_delta": logprob - baseline_logprob,
                    "max_key_error": max_key_error,
                    "max_value_error": max_value_error,
                }
            )

    summary = {}
    for profile_name in profiles:
        selected = [row for row in rows if row["profile"] == profile_name]
        summary[profile_name] = {
            "examples": len(selected),
            "exact_outputs": sum(bool(row["output_exact_vs_lossless"]) for row in selected),
            "first_token_equal": sum(bool(row["first_token_equal"]) for row in selected),
            "mean_common_prefix_tokens": sum(
                int(row["common_prefix_tokens"]) for row in selected
            )
            / max(len(selected), 1),
            "mean_f1_delta": sum(float(row["answer_f1_delta"]) for row in selected)
            / max(len(selected), 1),
            "mean_logprob_delta": sum(
                float(row["gold_answer_logprob_delta"]) for row in selected
            )
            / max(len(selected), 1),
            "mean_serialized_ratio": sum(
                float(row["serialized_ratio"]) for row in selected
            )
            / max(len(selected), 1),
        }

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_selective_kv_quantization_v1",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "layer_count": layer_count,
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
