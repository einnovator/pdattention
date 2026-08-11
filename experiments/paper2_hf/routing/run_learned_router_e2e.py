"""Confirm a frozen learned router in the real Qwen PRA generation path."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_smoke import answer_metrics, evidence_token_spans
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import (
    _prompt_with_question_span,
    load_split_examples,
)
from experiments.paper2_hf.routing.run_representation import _overlaps
from pra_torch.hf import PRAHFConfig, inject_pra, load_hf_routing_projection


def _generate(handle, encoded, tokenizer, enabled: bool, new_tokens: int):
    handle.set_memory_enabled(enabled)
    if handle.device.type == "cuda":
        torch.cuda.synchronize(handle.device)
    started = time.perf_counter()
    with torch.no_grad():
        output = handle.model.generate(
            **encoded,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if handle.device.type == "cuda":
        torch.cuda.synchronize(handle.device)
    continuation = output[0, encoded.input_ids.shape[1] :]
    return (
        tokenizer.decode(continuation, skip_special_tokens=True).strip(),
        time.perf_counter() - started,
    )


def run(args) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = load_hf_routing_projection(args.checkpoint, device=device)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(-1,),
            model_max_context_tokens=256,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=96,
            top_k_references=1,
            top_k_chunks_per_reference=3,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    examples = load_split_examples(
        args.cache_dir,
        args.examples_per_dataset,
        args.example_offset,
        args.seed,
    )
    rows = []
    for index, example in enumerate(examples, start=1):
        handle.cache.clear()
        source = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        evidence_spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
        entry = handle.add_reference(
            f"benchmark://{example['dataset']}/{example['id']}",
            source,
            text=example["source"],
        )
        encoded, _ = _prompt_with_question_span(tokenizer, example["question"], 128)
        encoded = encoded.to(device)
        baseline_text, baseline_seconds = _generate(
            handle, encoded, tokenizer, False, args.new_tokens
        )
        pra_text, pra_seconds = _generate(handle, encoded, tokenizer, True, args.new_tokens)
        adapter = next(iter(handle.adapters.values()))
        selected = adapter.last_selected_chunks[0]
        selected_spans = [
            (int(hit.logical_start), int(hit.logical_end)) for hit in selected
        ]
        chunks = entry.layer_memory[adapter.layer_idx].chunks
        gist_bytes = sum(chunk.metadata["routing_gist_bytes"] for chunk in chunks)
        detail_bytes = sum(chunk.metadata["detail_kv_bytes"] for chunk in chunks)
        rows.append(
            {
                "dataset": example["dataset"],
                "example_id": example["id"],
                "answer": example["answer"],
                "baseline_text": baseline_text,
                "pra_text": pra_text,
                "baseline_seconds": baseline_seconds,
                "pra_seconds": pra_seconds,
                "selected_chunk_ids": [hit.chunk_id for hit in selected],
                "selected_spans": selected_spans,
                "routing_recall_at_3": float(
                    any(_overlaps(span, evidence_spans) for span in selected_spans)
                ),
                "target_coverage_at_3": sum(
                    _overlaps(span, selected_spans) for span in evidence_spans
                )
                / max(len(evidence_spans), 1),
                "routing_index_bytes": gist_bytes,
                "detail_kv_bytes": detail_bytes,
                "routing_index_fraction": gist_bytes / max(detail_bytes, 1),
                "native_detail_position_states": sorted(
                    {chunk.token_kv.position_state for chunk in chunks}
                ),
                **{
                    f"baseline_{key}": value
                    for key, value in answer_metrics(baseline_text, example["answer"]).items()
                },
                **{
                    f"pra_{key}": value
                    for key, value in answer_metrics(pra_text, example["answer"]).items()
                },
            }
        )
        print(
            f"[{index}/{len(examples)}] {example['dataset']} "
            f"route={rows[-1]['routing_recall_at_3']:.0f} "
            f"baseline_f1={rows[-1]['baseline_f1']:.2f} pra_f1={rows[-1]['pra_f1']:.2f}",
            flush=True,
        )
    metrics = (
        "routing_recall_at_3",
        "target_coverage_at_3",
        "baseline_em",
        "baseline_f1",
        "baseline_answer_contained",
        "pra_em",
        "pra_f1",
        "pra_answer_contained",
        "baseline_seconds",
        "pra_seconds",
        "routing_index_fraction",
    )
    aggregates = {
        dataset: {
            metric: statistics.fmean(
                row[metric]
                for row in rows
                if dataset == "combined" or row["dataset"] == dataset
            )
            for metric in metrics
        }
        for dataset in ("combined", "hotpotqa", "qasper")
    }
    artifact = {
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "adapter_parameters": projection.parameter_count,
        "query_strategy": "last",
        "top_k": 3,
        "rows": rows,
        "aggregates": aggregates,
        "native_limit_violations": handle.native_limit_violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def parse_args() -> argparse.Namespace:
    result_dir = (
        ROOT
        / "docs"
        / "papers"
        / "shared"
        / "results"
        / "paper2_hf"
        / "routing"
        / "learned_adapter"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=result_dir / "checkpoints" / "asymmetric_linear_d128_last_joint_seed53.pt",
    )
    parser.add_argument("--output", type=Path, default=result_dir / "learned_router_e2e.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
