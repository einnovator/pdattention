"""Benchmark frozen natural selections through an OpenAI-compatible E0 server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Mapping


def _load_serving_helpers():
    source = Path(__file__).resolve().parents[2] / "src" / "pra_hf" / "serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("pra_serving_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load serving benchmark helpers from {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _quality(output: str, answer: str) -> tuple[float, float, float]:
    predicted = _normalize(output)
    gold = _normalize(answer)
    exact = float(predicted == gold)
    if not predicted or not gold:
        f1 = float(predicted == gold)
    else:
        overlap = sum((Counter(predicted) & Counter(gold)).values())
        precision = overlap / len(predicted)
        recall = overlap / len(gold)
        f1 = 0.0 if overlap == 0 else 2.0 * precision * recall / (precision + recall)
    containment = float(bool(gold) and " ".join(gold) in " ".join(predicted))
    return exact, f1, containment


def _gpu_memory() -> Mapping[str, object]:
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).splitlines()[0]
        used, total, utilization, power = [value.strip() for value in line.split(",")]
        return {
            "available": True,
            "device_used_mib": int(used),
            "device_total_mib": int(total),
            "utilization_percent": float(utilization),
            "power_watts": float(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "error": str(error)}


def _bounded_text(tokenizer, text: str, limit: int) -> tuple[str, int]:
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))[:limit]
    return tokenizer.decode(token_ids, skip_special_tokens=True), len(token_ids)


def _messages(
    entry: Mapping[str, object],
    tokenizer,
    condition: str,
    *,
    selected_limit: int,
    full_limit: int,
) -> tuple[list[dict[str, str]], int, int]:
    selected, selected_tokens = _bounded_text(
        tokenizer, str(entry["selected_source"]), selected_limit
    )
    if condition == "no_context":
        evidence = ""
        source_tokens = 0
    elif condition == "selected_context":
        evidence = selected
        source_tokens = selected_tokens
    elif condition == "full_context":
        combined = selected + "\n\n" + str(entry["distractor_source"])
        evidence, source_tokens = _bounded_text(tokenizer, combined, full_limit)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    question = str(entry["question"])
    prompt = (
        f"Evidence:\n{evidence}\n\nQuestion: {question}\n"
        "Answer briefly using only the evidence."
        if evidence
        else f"Question: {question}\nAnswer briefly."
    )
    return [{"role": "user", "content": prompt}], source_tokens, selected_tokens


def _percentiles(helpers, values: list[float]) -> dict[str, float | None]:
    return {
        "p50": helpers.percentile(values, 0.50),
        "p95": helpers.percentile(values, 0.95),
        "p99": helpers.percentile(values, 0.99),
    }


def _aggregate(helpers, rows: list[Mapping[str, object]]) -> Mapping[str, object]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    return {
        "sample_count": len(rows),
        "example_count": len({str(row["example_id"]) for row in rows}),
        "exact_match": statistics.fmean(values("exact_match")),
        "token_f1": statistics.fmean(values("token_f1")),
        "answer_containment": statistics.fmean(values("answer_containment")),
        "evidence_recall_at_4": statistics.fmean(values("evidence_recall_at_4")),
        "mean_source_tokens": statistics.fmean(values("source_tokens")),
        "mean_prompt_tokens": (
            statistics.fmean(values("prompt_tokens")) if values("prompt_tokens") else None
        ),
        "mean_completion_tokens": (
            statistics.fmean(values("completion_tokens"))
            if values("completion_tokens")
            else None
        ),
        "mean_cached_tokens": (
            statistics.fmean(values("cached_tokens")) if values("cached_tokens") else None
        ),
        "ttft_ms": _percentiles(helpers, values("ttft_ms")),
        "itl_ms": _percentiles(helpers, values("mean_itl_ms")),
        "completion_latency_ms": _percentiles(
            helpers, values("completion_latency_ms")
        ),
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    from transformers import AutoTokenizer

    helpers = _load_serving_helpers()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest["entries"]
        if not args.datasets or entry["dataset"] in args.datasets
    ]
    if args.max_examples_per_dataset > 0:
        counters: Counter[str] = Counter()
        limited = []
        for entry in entries:
            dataset = str(entry["dataset"])
            if counters[dataset] >= args.max_examples_per_dataset:
                continue
            counters[dataset] += 1
            limited.append(entry)
        entries = limited

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    gpu_before = _gpu_memory()
    for entry in entries:
        for condition in args.conditions:
            messages, source_tokens, selected_tokens = _messages(
                entry,
                tokenizer,
                condition,
                selected_limit=args.max_selected_tokens,
                full_limit=args.max_full_tokens,
            )
            salt = hashlib.sha256(
                f"{args.engine}:{entry['selection_id']}:{condition}".encode("utf-8")
            ).hexdigest()
            for repeat in range(args.repeats):
                result = helpers.stream_chat_completion(
                    args.base_url,
                    model=args.model,
                    messages=messages,
                    timeout_seconds=args.timeout_seconds,
                    cache_salt=salt if args.cache_salt else None,
                    max_tokens=args.max_new_tokens,
                    disable_native_thinking=args.disable_native_thinking,
                )
                exact, f1, containment = _quality(
                    str(result["output_text"]), str(entry["answer"])
                )
                rows.append(
                    {
                        "dataset": entry["dataset"],
                        "seed": entry["seed"],
                        "example_id": entry["example_id"],
                        "selection_id": entry["selection_id"],
                        "condition": condition,
                        "repeat": repeat,
                        "cache_state": "COLD" if repeat == 0 else "WARM_REPEAT",
                        "answer": entry["answer"],
                        "source_tokens": source_tokens,
                        "selected_source_tokens": selected_tokens,
                        "evidence_recall_at_4": entry["evidence_recall_at_4"],
                        "exact_match": exact,
                        "token_f1": f1,
                        "answer_containment": containment,
                        **result,
                    }
                )
    elapsed = time.perf_counter() - started
    aggregates = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for condition in args.conditions:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            aggregates.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    **_aggregate(helpers, selected),
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_cross_engine_portable_natural_e0_v1",
        "evidence_tier": "CONTROLLED_NATURAL_QA",
        "measurement_status": "MEASURED",
        "integration_level": "E0_SELECTED_TEXT",
        "engine": args.engine,
        "model_id": args.model,
        "model_revision": args.revision,
        "base_url": args.base_url,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "conditions": args.conditions,
        "repeats": args.repeats,
        "max_selected_tokens": args.max_selected_tokens,
        "max_full_tokens": args.max_full_tokens,
        "max_new_tokens": args.max_new_tokens,
        "cache_salt": args.cache_salt,
        "elapsed_seconds": elapsed,
        "request_throughput_s": len(rows) / max(elapsed, 1e-9),
        "gpu_before": gpu_before,
        "gpu_after": _gpu_memory(),
        "aggregates": aggregates,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/portable_e0_qa_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=("qasper", "hotpotqa", "2wikimultihopqa"),
        default=[],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("no_context", "selected_context", "full_context"),
        default=["no_context", "selected_context", "full_context"],
    )
    parser.add_argument("--max-examples-per-dataset", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-selected-tokens", type=int, default=384)
    parser.add_argument("--max-full-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--cache-salt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-native-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Send the Ollama-compatible top-level think=false extension.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.max_examples_per_dataset < 0:
        parser.error("Repeat count must be positive and example limit non-negative.")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}))


if __name__ == "__main__":
    main()
