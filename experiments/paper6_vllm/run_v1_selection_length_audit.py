"""Locate vLLM native replay drift as selected page count grows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from experiments.engine_serving.matched_qa import load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import _metrics
from experiments.paper6_vllm.run_matched_e0_e2 import _run
from experiments.paper6_vllm.run_v1_capture_replay_audit import _capture_memory


def _sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 or size % 16 for size in sizes):
        raise argparse.ArgumentTypeError("Sizes must be positive multiples of 16.")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), default="qasper")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--source-sizes", type=_sizes, default=(16, 32, 64, 128, 256, 384))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--reserve-blocks", type=int, default=64)
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
    bridge = VLLMMetalV1NativeBridge(runner, reserve_blocks=args.reserve_blocks)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    rows = []
    try:
        for example_index, example in enumerate(examples):
            all_source = list(tokenizer.encode(example.selected_source, add_special_tokens=False))
            query = list(
                tokenizer.encode(
                    "Answer the question using the available evidence. Give only the "
                    f"short answer.\nQuestion: {example.question}\nAnswer:",
                    add_special_tokens=False,
                )
            )
            for source_tokens in args.source_sizes:
                if source_tokens > len(all_source):
                    continue
                source = all_source[:source_tokens]
                before = len(bridge.prefill_page_observations())
                _e0_id, e0_output, _ = _run(llm, bridge, sampling, source + query)
                observations = bridge.prefill_page_observations()[before:]
                if not observations:
                    raise RuntimeError("vLLM did not expose ordinary prefill pages.")
                page_count = math.ceil(source_tokens / bridge.block_size)
                source_blocks = list(observations[0]["block_ids_by_group"][0][:page_count])
                captured = _capture_memory(bridge, source_blocks, source_tokens)
                logical_key = f"length-{example_index}-{source_tokens}"
                bridge.materialize(logical_key, captured)
                _e2_id, e2_output, _ = _run(
                    llm,
                    bridge,
                    sampling,
                    query,
                    key=logical_key,
                    source_tokens=source_tokens,
                )
                bridge.release(logical_key)
                e0_text = str(e0_output.outputs[0].text).strip()
                e2_text = str(e2_output.outputs[0].text).strip()
                rows.append(
                    {
                        "dataset": example.dataset,
                        "seed": example.seed,
                        "example_id": example.example_id,
                        "source_tokens": source_tokens,
                        "source_pages": page_count,
                        "exact_output_parity": e0_text == e2_text,
                        "f1_e0": _metrics(e0_text, example.answer)[1],
                        "f1_e2": _metrics(e2_text, example.answer)[1],
                        "output_e0": e0_text,
                        "output_e2": e2_text,
                    }
                )
    finally:
        bridge.close()

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_vllm_native_selection_length_audit_v1",
        "evidence_tier": "MECHANISM_DIAGNOSTIC",
        "engine": "vllm-metal",
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "model_id": args.model,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "source_sizes": list(args.source_sizes),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
