"""Run the public PRA-HF API on representative QASPER and HotpotQA examples."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.qa.run_smoke import answer_metrics, evidence_token_spans
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_hf import PRAConfig, PRAForCausalLM


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _evidence_recall(selected: list[dict], evidence_spans: list[tuple[int, int]]) -> float:
    routed = [(int(row["logical_start"]), int(row["logical_end"])) for row in selected]
    covered = [
        any(max(start, left) < min(end, right) for left, right in routed)
        for start, end in evidence_spans
    ]
    return sum(covered) / max(len(covered), 1)


def run(args) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    config = PRAConfig(
        routing_layer=-1,
        consumption_layers=tuple(range(-8, 0)),
        chunk_tokens=32,
        selected_fraction=0.20,
        max_direct_context=128,
        native_operation_limit=512,
        max_materialized_tokens=128,
        context_safety_reserve_tokens=4,
        encoding_block_tokens=128,
        reference_device="cpu",
        pin_reference_memory=device.type == "cuda",
        non_blocking_transfer=device.type == "cuda",
    )
    pra = PRAForCausalLM.from_pretrained(
        args.model,
        routing_adapter=args.router,
        pra_config=config,
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    pra.model.to(device).eval()
    examples = load_split_examples(args.cache_dir, args.examples_per_dataset, 8, 20260811)
    rows = []
    for index, example in enumerate(examples, start=1):
        pra.clear_references()
        prompt = f"Answer briefly and directly.\nQuestion: {example['question']}\nAnswer:"
        pra.disable()
        _sync(device)
        baseline = pra.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_details=True,
        )
        baseline_quality = answer_metrics(baseline.text, example["answer"])
        _sync(device)
        ingest_started = time.perf_counter()
        handle = pra.add_reference(
            f"benchmark://{example['dataset']}/{example['id']}", text=example["source"]
        )
        _sync(device)
        ingest_seconds = time.perf_counter() - ingest_started
        pra.enable()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        routed_started = time.perf_counter()
        routed = pra.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_details=True,
        )
        _sync(device)
        routed_wall = time.perf_counter() - routed_started
        quality = answer_metrics(routed.text, example["answer"])
        stats = routed.stats
        diagnostics = stats["diagnostics_by_layer"]
        transfer_bytes = sum(
            float(values.get("retrieved_kv_transfer_bytes", 0))
            for values in diagnostics.values()
        )
        evidence_spans = evidence_token_spans(
            pra.tokenizer, example["source"], example["evidence"]
        )
        rows.append(
            {
                "dataset": example["dataset"],
                "example_id": example["id"],
                "answer": example["answer"],
                "baseline_text": baseline.text,
                "routed_text": routed.text,
                "baseline_f1": baseline_quality["f1"],
                "baseline_em": baseline_quality["em"],
                "routed_f1": quality["f1"],
                "routed_em": quality["em"],
                "evidence_recall": _evidence_recall(stats["selected"], evidence_spans),
                "source_tokens": handle.tokens,
                "candidate_chunks": stats["candidate_chunks"],
                "requested_chunk_fraction": stats["requested_chunk_fraction"],
                "requested_kv_token_fraction": stats["requested_kv_token_fraction"],
                "materialized_kv_token_fraction": stats["materialized_kv_token_fraction"],
                "routing_index_bytes": pra.stats()["routing_index_bytes"],
                "resident_detail_kv_bytes": pra.stats()["resident_detail_kv_bytes"],
                "ingest_seconds": ingest_seconds,
                "query_encoding_seconds": stats["query_encoding_seconds"],
                "routing_seconds": stats["routing_seconds"],
                "generation_seconds": stats["generation_seconds"],
                "routed_wall_seconds": routed_wall,
                "generated_tokens": routed.generated_tokens,
                "tokens_per_second": routed.generated_tokens / max(routed_wall, 1e-12),
                "transfer_bytes_across_layers": transfer_bytes,
                "peak_gpu_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "native_limit_violations": pra.stats()["native_limit_violations"],
            }
        )
        print(
            f"[{index}/{len(examples)}] {example['dataset']} "
            f"F1 {baseline_quality['f1']:.3f}->{quality['f1']:.3f} "
            f"evidence={rows[-1]['evidence_recall']:.3f}",
            flush=True,
        )
    aggregates = {
        dataset: {
            "examples": len(subset),
            **{
                key: _mean(subset, key)
                for key in (
                    "baseline_f1",
                    "baseline_em",
                    "routed_f1",
                    "routed_em",
                    "evidence_recall",
                    "requested_chunk_fraction",
                    "materialized_kv_token_fraction",
                    "ingest_seconds",
                    "query_encoding_seconds",
                    "routing_seconds",
                    "routed_wall_seconds",
                    "tokens_per_second",
                    "routing_index_bytes",
                    "resident_detail_kv_bytes",
                    "transfer_bytes_across_layers",
                    "peak_gpu_bytes",
                )
            },
        }
        for dataset in ("hotpotqa", "qasper", "combined")
        for subset in [[row for row in rows if dataset == "combined" or row["dataset"] == dataset]]
    }
    result = {
        "protocol": "public PRAForCausalLM API; one-shot route; final-eight-layer native-KV consumption",
        "model": args.model,
        "revision": args.revision,
        "router": str(args.router),
        "device": str(device),
        "dtype": str(dtype),
        "config": config.to_dict(),
        "aggregates": aggregates,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-dataset", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
