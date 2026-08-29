"""Audit vLLM native replay against K/V captured from ordinary paged prefill."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import _bounded_source, _metrics
from experiments.paper6_vllm.run_matched_e0_e2 import _aligned, _run


def _capture_memory(bridge, block_ids: list[int], source_tokens: int):
    """Copy source pages from vLLM's ordinary cache into PRA memory layout."""

    import mlx.core as mx
    from pra_mlx.native import MLXNativeLayerKV, MLXNativeMemory

    cache = bridge.runtime.kv_cache
    layers = []
    for keys, values in zip(cache.key_caches, cache.value_caches):
        page_keys = mx.array(keys[block_ids]).reshape(
            -1, keys.shape[2], keys.shape[3]
        )[:source_tokens]
        page_values = mx.array(values[block_ids]).reshape(
            -1, values.shape[2], values.shape[3]
        )[:source_tokens]
        layers.append(
            MLXNativeLayerKV(
                page_keys.transpose(1, 0, 2)[None],
                page_values.transpose(1, 0, 2)[None],
            )
        )
    memory = MLXNativeMemory(tuple(layers), source_tokens)
    mx.eval(*(array for layer in memory.layers for array in (layer.keys, layer.values)))
    return memory


def _max_deltas(left, right) -> tuple[float, float]:
    import mlx.core as mx

    key_delta = max(
        float(mx.max(mx.abs(a.keys.astype(mx.float32) - b.keys.astype(mx.float32))).item())
        for a, b in zip(left.layers, right.layers)
    )
    value_delta = max(
        float(mx.max(mx.abs(a.values.astype(mx.float32) - b.values.astype(mx.float32))).item())
        for a, b in zip(left.layers, right.layers)
    )
    return key_delta, value_delta


def _token_comparison(reference: list[int], candidate: list[int]) -> dict[str, object]:
    """Separate first-decision parity from later autoregressive divergence."""

    common_prefix = 0
    for left, right in zip(reference, candidate):
        if left != right:
            break
        common_prefix += 1
    compared = max(len(reference), len(candidate))
    equal_positions = sum(left == right for left, right in zip(reference, candidate))
    return {
        "exact": reference == candidate,
        "first_token_equal": bool(reference and candidate and reference[0] == candidate[0]),
        "common_prefix_tokens": common_prefix,
        "position_agreement": equal_positions / compared if compared else 1.0,
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), default="qasper"
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
    parser.add_argument("--max-examples", type=int, default=5)
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
        for index, example in enumerate(examples):
            raw_source = _bounded_source(
                tokenizer, example.selected_source, args.max_source_tokens
            )
            source = _aligned(raw_source, tokenizer, bridge.block_size)
            query = list(
                tokenizer.encode(
                    "Answer the question using the available evidence. Give only the "
                    f"short answer.\nQuestion: {example.question}\nAnswer:",
                    add_special_tokens=False,
                )
            )
            generic_memory = encode_native_memory(runner.model, source)

            before = len(bridge.prefill_page_observations())
            e0_id, e0_output, _ = _run(
                llm, bridge, sampling, source + query
            )
            observations = bridge.prefill_page_observations()[before:]
            ordinary = [row for row in observations if row["scheduler_cache_start"] == 0]
            if not ordinary:
                raise RuntimeError("vLLM did not expose the ordinary source prefill pages.")
            page_count = math.ceil(len(source) / bridge.block_size)
            source_blocks = list(ordinary[0]["block_ids_by_group"][0][:page_count])
            captured_memory = _capture_memory(bridge, source_blocks, len(source))
            key_delta, value_delta = _max_deltas(generic_memory, captured_memory)

            outputs = {"e0_selected_text": str(e0_output.outputs[0].text).strip()}
            output_tokens = {
                "e0_selected_text": list(map(int, e0_output.outputs[0].token_ids))
            }
            replay_conditions = [
                ("e2_generic_encoder", generic_memory, len(source)),
                ("e2_paged_capture_base_zero", captured_memory, 0),
                ("e2_paged_capture_base_minus_1", captured_memory, len(source) - 1),
                ("e2_paged_capture_base_nominal", captured_memory, len(source)),
                ("e2_paged_capture_base_plus_1", captured_memory, len(source) + 1),
            ]
            for label, memory, position_base in replay_conditions:
                logical_key = f"audit-{index}-{label}"
                bridge.materialize(logical_key, memory)
                _request_id, output, _ = _run(
                    llm,
                    bridge,
                    sampling,
                    query,
                    key=logical_key,
                    source_tokens=len(source),
                    source_position_base=position_base,
                )
                outputs[label] = str(output.outputs[0].text).strip()
                output_tokens[label] = list(map(int, output.outputs[0].token_ids))
                bridge.release(logical_key)

            token_comparisons = {
                label: _token_comparison(
                    output_tokens["e0_selected_text"], candidate_tokens
                )
                for label, candidate_tokens in output_tokens.items()
                if label != "e0_selected_text"
            }

            rows.append(
                {
                    "dataset": example.dataset,
                    "seed": example.seed,
                    "example_id": example.example_id,
                    "source_tokens": len(source),
                    "source_blocks": source_blocks,
                    "generic_vs_paged_max_key_delta": key_delta,
                    "generic_vs_paged_max_value_delta": value_delta,
                    "generic_matches_e0": outputs["e2_generic_encoder"] == outputs["e0_selected_text"],
                    "paged_capture_matches_e0_by_base": {
                        str(base): (
                            outputs[label]
                            == outputs["e0_selected_text"]
                        )
                        for label, base in (
                            ("e2_paged_capture_base_zero", 0),
                            ("e2_paged_capture_base_minus_1", len(source) - 1),
                            ("e2_paged_capture_base_nominal", len(source)),
                            ("e2_paged_capture_base_plus_1", len(source) + 1),
                        )
                    },
                    "outputs": outputs,
                    "output_token_ids": output_tokens,
                    "token_comparison_to_e0": token_comparisons,
                    "f1": {
                        label: _metrics(text, example.answer)[1]
                        for label, text in outputs.items()
                    },
                }
            )
    finally:
        bridge.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_native_capture_replay_audit_v1",
        "evidence_tier": "MECHANISM_DIAGNOSTIC",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "rows": rows,
        "summary": {
            label: {
                "examples": len(rows),
                "exact_outputs": sum(
                    bool(row["token_comparison_to_e0"][label]["exact"])
                    for row in rows
                ),
                "first_token_matches": sum(
                    bool(row["token_comparison_to_e0"][label]["first_token_equal"])
                    for row in rows
                ),
                "mean_common_prefix_tokens": sum(
                    int(row["token_comparison_to_e0"][label]["common_prefix_tokens"])
                    for row in rows
                )
                / len(rows),
                "mean_position_agreement": sum(
                    float(row["token_comparison_to_e0"][label]["position_agreement"])
                    for row in rows
                )
                / len(rows),
            }
            for label in (
                "e2_generic_encoder",
                "e2_paged_capture_base_zero",
                "e2_paged_capture_base_minus_1",
                "e2_paged_capture_base_nominal",
                "e2_paged_capture_base_plus_1",
            )
        } if rows else {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
