"""Run selector-frozen natural E0/E2 AirLLM experiments on a small CUDA GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from pra_hf.airllm_adapter import wrap_airllm_hf_model
from pra_hf.config import PRAConfig


class TokenTimingStreamer:
    """Capture first-token and inter-token timing from HF ``generate``.

    Transformers sends the complete prompt through ``put`` before generated
    tokens.  The first callback is therefore excluded; every later callback is
    one or more newly decoded tokens.  A caller-provided clock keeps the small
    timing state machine directly testable.
    """

    def __init__(
        self,
        *,
        started_at: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.started_at = float(started_at)
        self.clock = clock
        self._saw_prompt = False
        self.token_times: list[float] = []

    def put(self, value: Any) -> None:
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        now = float(self.clock())
        token_count = int(value.numel()) if hasattr(value, "numel") else 1
        self.token_times.extend([now] * max(token_count, 1))

    def end(self) -> None:
        """Satisfy the HF streamer protocol; all metrics are already captured."""

    def metrics(self) -> dict[str, float | int | None]:
        ttft = (
            (self.token_times[0] - self.started_at) * 1000.0
            if self.token_times
            else None
        )
        intervals = [
            (right - left) * 1000.0
            for left, right in zip(self.token_times, self.token_times[1:])
        ]
        return {
            "ttft_ms": ttft,
            "itl_ms": statistics.fmean(intervals) if intervals else None,
            "timed_output_tokens": len(self.token_times),
        }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def quality(output: str, answer: str) -> tuple[float, float, float]:
    """Return exact match, token F1, and answer containment."""

    predicted = _tokens(output)
    expected = _tokens(answer)
    exact = float(predicted == expected)
    containment = float(bool(expected) and " ".join(expected) in " ".join(predicted))
    if not predicted or not expected:
        return exact, 0.0, containment
    predicted_counts = Counter(predicted)
    expected_counts = Counter(expected)
    overlap = sum((predicted_counts & expected_counts).values())
    if overlap == 0:
        return exact, 0.0, containment
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return exact, 2.0 * precision * recall / (precision + recall), containment


def select_entries(
    manifest: Mapping[str, Any],
    datasets: Iterable[str],
    max_examples_per_dataset: int,
) -> list[dict[str, Any]]:
    """Select a deterministic bounded cohort while preserving manifest order."""

    requested = set(datasets)
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for raw in manifest["entries"]:
        entry = dict(raw)
        dataset = str(entry["dataset"])
        if requested and dataset not in requested:
            continue
        if max_examples_per_dataset > 0 and counts[dataset] >= max_examples_per_dataset:
            continue
        counts[dataset] += 1
        selected.append(entry)
    return selected


def _bounded_text(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    bounded = token_ids[:max_tokens]
    return tokenizer.decode(bounded, skip_special_tokens=True), len(bounded)


def _prompt(question: str, evidence: str | None = None) -> str:
    prefix = "" if evidence is None else f"Evidence:\n{evidence}\n\n"
    return f"{prefix}Question: {question}\nAnswer concisely:"


def _measure_e0(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    request_started = time.perf_counter()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids.to("cuda")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    timing = TokenTimingStreamer(started_at=request_started)
    started = time.perf_counter()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        disable_compile=True,
        streamer=timing,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_ids = generated[:, input_ids.shape[1] :]
    return {
        "visible_prompt_tokens": int(input_ids.shape[1]),
        "selected_native_kv_tokens": 0,
        "output_token_ids": output_ids[0].detach().cpu().tolist(),
        "output_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
        "generated_tokens": int(output_ids.shape[1]),
        "completion_seconds": elapsed,
        "tokens_per_second": int(output_ids.shape[1]) / max(elapsed, 1e-9),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        **timing.metrics(),
    }


def _measure_e2(pra: Any, question: str, max_new_tokens: int) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    request_started = time.perf_counter()
    timing = TokenTimingStreamer(started_at=request_started)
    result = pra.generate(
        _prompt(question),
        max_new_tokens=max_new_tokens,
        return_details=True,
        do_sample=False,
        use_cache=True,
        streamer=timing,
    )
    torch.cuda.synchronize()
    return {
        "visible_prompt_tokens": int(result.prompt_tokens),
        "selected_native_kv_tokens": int(
            result.stats.get("materialized_kv_tokens") or 0
        ),
        "requested_native_kv_tokens": result.stats.get("requested_kv_tokens"),
        "output_text": result.text,
        "generated_tokens": int(result.generated_tokens),
        "completion_seconds": float(result.latency_seconds),
        "tokens_per_second": int(result.generated_tokens)
        / max(float(result.latency_seconds), 1e-9),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        **timing.metrics(),
    }


def _annotate(
    measurement: dict[str, Any],
    *,
    entry: Mapping[str, Any],
    condition: str,
    regime: str,
    repeat: int,
    reference_tokens: int,
    reference_encode_seconds: float | None,
) -> dict[str, Any]:
    exact, f1, containment = quality(
        str(measurement["output_text"]), str(entry["answer"])
    )
    return {
        "dataset": entry["dataset"],
        "seed": entry["seed"],
        "example_id": entry["example_id"],
        "selection_id": entry["selection_id"],
        "selected_source_sha256": entry["selected_source_sha256"],
        "condition": condition,
        "regime": regime,
        "repeat": repeat,
        "answer": entry["answer"],
        "reference_tokens": reference_tokens,
        "reference_encode_seconds": reference_encode_seconds,
        "exact_match": exact,
        "token_f1": f1,
        "answer_containment": containment,
        **measurement,
    }


def _write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _percentiles(values: Iterable[float]) -> dict[str, float] | None:
    """Return interpolated latency percentiles, or ``None`` when unmeasured."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None

    def at(quantile: float) -> float:
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return {"p50": at(0.50), "p95": at(0.95), "p99": at(0.99)}


def _summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def optional_mean(selected: list[Mapping[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in selected if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    summaries: list[dict[str, Any]] = []
    keys = sorted({(str(row["dataset"]), str(row["condition"]), str(row["regime"])) for row in rows})
    for dataset, condition, regime in keys:
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["condition"] == condition
            and row["regime"] == regime
        ]
        summaries.append(
            {
                "dataset": dataset,
                "condition": condition,
                "regime": regime,
                "samples": len(selected),
                "mean_exact_match": statistics.fmean(float(row["exact_match"]) for row in selected),
                "mean_token_f1": statistics.fmean(float(row["token_f1"]) for row in selected),
                "mean_answer_containment": statistics.fmean(float(row["answer_containment"]) for row in selected),
                "mean_completion_seconds": statistics.fmean(float(row["completion_seconds"]) for row in selected),
                "mean_ttft_ms": optional_mean(selected, "ttft_ms"),
                "mean_itl_ms": optional_mean(selected, "itl_ms"),
                "completion_seconds": _percentiles(
                    float(row["completion_seconds"]) for row in selected
                ),
                "ttft_ms": _percentiles(
                    float(row["ttft_ms"])
                    for row in selected
                    if row.get("ttft_ms") is not None
                ),
                "itl_ms": _percentiles(
                    float(row["itl_ms"])
                    for row in selected
                    if row.get("itl_ms") is not None
                ),
                "mean_visible_prompt_tokens": statistics.fmean(float(row["visible_prompt_tokens"]) for row in selected),
                "mean_native_kv_tokens": statistics.fmean(float(row["selected_native_kv_tokens"]) for row in selected),
                "peak_cuda_bytes": max(int(row["peak_cuda_bytes"]) for row in selected),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"D:\models\tinyllama-local")
    parser.add_argument(
        "--shard-dir", type=Path, default=Path(r"D:\models\airllm-tinyllama-cuda")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/portable_e0_qa_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=("qasper", "hotpotqa", "2wikimultihopqa"),
    )
    parser.add_argument("--max-examples-per-dataset", type=int, default=1)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--max-reference-tokens", type=int, default=384)
    parser.add_argument("--max-full-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prefetching", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from airllm import AutoModel

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = select_entries(
        manifest, args.datasets, args.max_examples_per_dataset
    )
    if not entries:
        raise ValueError("The selected AirLLM natural cohort is empty.")
    report: dict[str, Any] = {
        "schema_version": "paper6.6-airllm-cuda-natural-v1",
        "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
        "integration_levels": ("E0", "E2"),
        "selector_frozen": True,
        "model_id": args.model,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "manifest": str(args.manifest),
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "warm_repeats": args.warm_repeats,
        "status": "RUNNING",
        "rows": [],
    }
    _write(args.output, report)
    try:
        initialized = time.perf_counter()
        air = AutoModel.from_pretrained(
            args.model,
            device="cuda:0",
            dtype=torch.float16,
            max_seq_len=2048,
            layer_shards_saving_path=str(args.shard_dir),
            prefetching=args.prefetching,
            profiling_mode=True,
        )
        report["initialization_seconds"] = time.perf_counter() - initialized
        tokenizer = air.tokenizer
        prepared: list[tuple[dict[str, Any], str, int, str, int]] = []
        for entry in entries:
            selected, reference_tokens = _bounded_text(
                tokenizer, str(entry["selected_source"]), args.max_reference_tokens
            )
            full, full_tokens = _bounded_text(
                tokenizer,
                f"{selected}\n\nDistractor material:\n{entry['distractor_source']}",
                args.max_full_tokens,
            )
            prepared.append((entry, selected, reference_tokens, full, full_tokens))

        # Run every visible-text condition before mutating the AirLLM model.
        for entry, selected, reference_tokens, full, _ in prepared:
            full_row = _measure_e0(
                air.model,
                tokenizer,
                _prompt(str(entry["question"]), full),
                args.max_new_tokens,
            )
            report["rows"].append(
                _annotate(
                    full_row,
                    entry=entry,
                    condition="full_context_e0",
                    regime="cold_one_shot",
                    repeat=0,
                    reference_tokens=reference_tokens,
                    reference_encode_seconds=None,
                )
            )
            for repeat in range(args.warm_repeats + 1):
                selected_row = _measure_e0(
                    air.model,
                    tokenizer,
                    _prompt(str(entry["question"]), selected),
                    args.max_new_tokens,
                )
                report["rows"].append(
                    _annotate(
                        selected_row,
                        entry=entry,
                        condition="selected_text_e0",
                        regime="cold_one_shot" if repeat == 0 else "warm_repeated",
                        repeat=repeat,
                        reference_tokens=reference_tokens,
                        reference_encode_seconds=None,
                    )
                )
                _write(args.output, report)

        pra = wrap_airllm_hf_model(
            air,
            pra_config=PRAConfig(
                consumption_layers=(-4, -3, -2, -1),
                address_layers=(-4, -3, -2, -1),
                detail_kv_layers=(-4, -3, -2, -1),
                routing_layer=-1,
                selected_fraction=1.0,
                chunk_tokens=64,
                max_materialized_tokens=args.max_reference_tokens,
                reference_device="cpu",
            ),
        )
        for entry, selected, reference_tokens, _, _ in prepared:
            pra.clear_references()
            encode_started = time.perf_counter()
            handle = pra.add_reference(
                f"memory://paper6.6/{entry['dataset']}/{entry['selection_id']}",
                text=selected,
            )
            encode_seconds = time.perf_counter() - encode_started
            if handle.tokens != reference_tokens:
                raise RuntimeError("AirLLM reference token accounting changed during ingestion.")
            for repeat in range(args.warm_repeats + 1):
                native_row = _measure_e2(
                    pra, str(entry["question"]), args.max_new_tokens
                )
                report["rows"].append(
                    _annotate(
                        native_row,
                        entry=entry,
                        condition="native_pra_e2",
                        regime="cold_one_shot" if repeat == 0 else "warm_repeated",
                        repeat=repeat,
                        reference_tokens=reference_tokens,
                        reference_encode_seconds=encode_seconds if repeat == 0 else 0.0,
                    )
                )
                _write(args.output, report)
        report["aggregates"] = _summary(report["rows"])
        report["status"] = "COMPLETE"
    except Exception as error:
        report["status"] = "BLOCKED"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    _write(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
