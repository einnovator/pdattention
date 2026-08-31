"""Run selector-frozen natural QA through the vLLM CUDA connector candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import uuid
from pathlib import Path

from experiments.paper6_2_mlx.run_answer_quality_pressure import _metrics
from experiments.paper6_vllm.run_cuda_connector_candidate import (
    _aligned,
    _generate,
    _text,
    _token_ids,
)
from pra_vllm.cuda_protocol import CudaConnectorCommand


DATASETS = ("qasper", "hotpotqa", "2wikimultihopqa")


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/portable_e0_qa_manifest.json"),
    )
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--storage", type=Path, default=Path(".pra/vllm-cuda-natural")
    )
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples-per-dataset", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected: list[dict[str, object]] = []
    for dataset in DATASETS:
        rows = [row for row in manifest["entries"] if row["dataset"] == dataset]
        selected.extend(rows[: args.max_examples_per_dataset])

    storage = args.storage.expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
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
    block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    padding = tokenizer.encode(" archive", add_special_tokens=False)
    if not padding:
        raise RuntimeError("Tokenizer did not produce a page-alignment token.")
    placeholder = int(padding[0])
    sampling = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, ignore_eos=False
    )
    store_sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)

    rows: list[dict[str, object]] = []
    for index, example in enumerate(selected):
        question = tokenizer.encode(
            f"\nQuestion: {example['question']}\nAnswer concisely:",
            add_special_tokens=False,
        )
        source = tokenizer.encode(
            str(example["selected_source"]), add_special_tokens=False
        )[: args.max_source_tokens]
        max_source = 512 - len(question) - args.max_new_tokens - block_size
        source = source[: max(1, max_source)]
        source = _aligned(source, block_size, placeholder)
        key = f"natural-{run_id}-{index:03d}"
        command = CudaConnectorCommand("store", key, len(source))
        _, ingestion = _generate(
            llm,
            store_sampling,
            source + [placeholder],
            cache_salt=command.cache_salt(),
        )
        full, full_metrics = _generate(llm, sampling, source + question)
        native, native_metrics = _generate(
            llm,
            sampling,
            [placeholder] * len(source) + question,
            cache_salt=CudaConnectorCommand("load", key, len(source)).cache_salt(),
        )
        full_text = _text(full)
        native_text = _text(native)
        full_em, full_f1 = _metrics(full_text, str(example["answer"]))
        native_em, native_f1 = _metrics(native_text, str(example["answer"]))
        resource_dir = storage / hashlib.sha256(key.encode("utf-8")).hexdigest()
        rows.append(
            {
                "dataset": example["dataset"],
                "seed": int(example["seed"]),
                "example_id": example["example_id"],
                "selection_id": example["selection_id"],
                "question": example["question"],
                "answer": example["answer"],
                "evidence_recall_at_4": float(example["evidence_recall_at_4"]),
                "selected_source_tokens": len(source),
                "visible_query_tokens": len(question),
                "scheduler_placeholder_tokens": len(source),
                "stored_native_bytes": sum(
                    path.stat().st_size for path in resource_dir.glob("*")
                ),
                "full_output": full_text,
                "native_output": native_text,
                "full_token_ids": _token_ids(full),
                "native_token_ids": _token_ids(native),
                "exact_output_parity": _token_ids(full) == _token_ids(native),
                "full_em": full_em,
                "full_f1": full_f1,
                "native_em": native_em,
                "native_f1": native_f1,
                "ingestion": ingestion,
                "full": full_metrics,
                "native": native_metrics,
            }
        )

    aggregates = []
    for dataset in DATASETS:
        cohort = [row for row in rows if row["dataset"] == dataset]
        aggregates.append(
            {
                "dataset": dataset,
                "samples": len(cohort),
                "exact_output_parity": _mean(cohort, "exact_output_parity"),
                "full_em": _mean(cohort, "full_em"),
                "native_em": _mean(cohort, "native_em"),
                "full_f1": _mean(cohort, "full_f1"),
                "native_f1": _mean(cohort, "native_f1"),
                "mean_source_tokens": _mean(cohort, "selected_source_tokens"),
                "mean_native_bytes": _mean(cohort, "stored_native_bytes"),
                "full_completion_ms": statistics.median(
                    float(row["full"]["completion_ms"]) for row in cohort
                ),
                "native_completion_ms": statistics.median(
                    float(row["native"]["completion_ms"]) for row in cohort
                ),
            }
        )

    payload = {
        "schema_version": "paper6-vllm-cuda-connector-natural-v1",
        "evidence_tier": "NATURAL_QA_CUDA_NATIVE_TRANSFER",
        "integration_status": "E2_CANDIDATE_PREFIX_SHAPED",
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "model_id": args.model,
        "device": torch.cuda.get_device_name(0),
        "manifest": str(args.manifest),
        "selector_frozen": True,
        "apc_enabled": False,
        "source_content_visible": False,
        "source_slots_scheduler_visible": True,
        "gold_answer_log_probability": None,
        "aggregates": aggregates,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

