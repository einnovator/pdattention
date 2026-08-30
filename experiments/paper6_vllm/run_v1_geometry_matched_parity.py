"""Diagnose vLLM parity after matching E0/E2 query-prefill geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import _bounded_source
from experiments.paper6_vllm.run_matched_e0_e2 import _aligned, _run
from experiments.paper6_vllm.run_v1_capture_replay_audit import (
    _capture_memory,
    _token_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="qasper")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import vllm
    from pra_vllm.v1_native import VLLMMetalV1NativeBridge
    from vllm import LLM, SamplingParams

    manifest, examples = load_matched_examples(args.manifest, args.dataset, args.cache_dir)
    examples = examples[: args.max_examples]
    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=64)
    tokenizer = llm.get_tokenizer()
    generation = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    prime = SamplingParams(temperature=0, max_tokens=1)
    rows = []
    try:
        for index, example in enumerate(examples):
            source = _aligned(
                _bounded_source(tokenizer, example.selected_source, args.max_source_tokens),
                tokenizer,
                bridge.block_size,
            )
            query = list(
                tokenizer.encode(
                    "Answer the question using the available evidence. Give only the "
                    f"short answer.\nQuestion: {example.question}\nAnswer:",
                    add_special_tokens=False,
                )
            )

            _full_id, full, _ = _run(
                llm, bridge, generation, source + query, cache_salt=f"full-{index}"
            )

            observation_start = len(bridge.prefill_page_observations())
            _prime_id, _prime_output, _ = _run(
                llm, bridge, prime, source, cache_salt=f"geometry-{index}"
            )
            observations = bridge.prefill_page_observations()[observation_start:]
            fresh = [row for row in observations if row["scheduler_cache_start"] == 0]
            if not fresh:
                raise RuntimeError("Geometry audit did not observe source prefill pages.")
            page_count = math.ceil(len(source) / bridge.block_size)
            source_blocks = list(fresh[0]["block_ids_by_group"][0][:page_count])
            captured = _capture_memory(bridge, source_blocks, len(source))

            observation_start = len(bridge.scheduler_observations())
            _apc_id, apc, _ = _run(
                llm,
                bridge,
                generation,
                source + query,
                cache_salt=f"geometry-{index}",
            )
            apc_observations = bridge.scheduler_observations()[observation_start:]

            key = f"geometry-native-{index}"
            bridge.materialize(key, captured)
            _native_id, native, _ = _run(
                llm,
                bridge,
                generation,
                query,
                key=key,
                source_tokens=len(source),
            )
            bridge.release(key)

            tokens = {
                "e0_full": list(map(int, full.outputs[0].token_ids)),
                "e0_apc_geometry": list(map(int, apc.outputs[0].token_ids)),
                "e2_native": list(map(int, native.outputs[0].token_ids)),
            }
            rows.append(
                {
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "source_tokens": len(source),
                    "query_tokens": len(query),
                    "e0_apc_cached_tokens": int(apc.num_cached_tokens),
                    "e0_apc_scheduler_observations": apc_observations,
                    "comparisons": {
                        "full_vs_native": _token_comparison(
                            tokens["e0_full"], tokens["e2_native"]
                        ),
                        "apc_vs_native": _token_comparison(
                            tokens["e0_apc_geometry"], tokens["e2_native"]
                        ),
                        "full_vs_apc": _token_comparison(
                            tokens["e0_full"], tokens["e0_apc_geometry"]
                        ),
                    },
                    "output_token_ids": tokens,
                    "outputs": {
                        "e0_full": str(full.outputs[0].text).strip(),
                        "e0_apc_geometry": str(apc.outputs[0].text).strip(),
                        "e2_native": str(native.outputs[0].text).strip(),
                    },
                }
            )
    finally:
        bridge.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_geometry_matched_parity_v1",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "cohort": manifest["cohort"],
        "rows": rows,
        "summary": {
            name: {
                "exact": sum(row["comparisons"][name]["exact"] for row in rows),
                "first_token": sum(
                    row["comparisons"][name]["first_token_equal"] for row in rows
                ),
                "mean_common_prefix": sum(
                    row["comparisons"][name]["common_prefix_tokens"] for row in rows
                )
                / len(rows),
            }
            for name in ("full_vs_native", "apc_vs_native", "full_vs_apc")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
