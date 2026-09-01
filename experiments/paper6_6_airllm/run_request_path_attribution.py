"""Attribute AirLLM E2 latency to dispatch, routing, and memory consumption.

The existing natural benchmark compares selected-text E0 with routed E2. This
runner inserts two controls: an injected-but-disabled eager path and a frozen
native plan whose selection is prepared outside the timed request. Reference
length and consumer-layer count remain explicit so the same harness can search
for workloads where native context becomes economically plausible.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.paper6_6_airllm.run_cuda_natural import (
    TokenTimingStreamer,
    _annotate,
    _bounded_text,
    _prompt,
    _summary,
    _write,
    select_entries,
)
from pra_hf.airllm_adapter import wrap_airllm_hf_model
from pra_hf.config import PRAConfig


def _direct_generate(
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


def _native_generate(
    pra: Any,
    question: str,
    max_new_tokens: int,
    *,
    plan: Any | None,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    request_started = time.perf_counter()
    timing = TokenTimingStreamer(started_at=request_started)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "return_details": True,
        "do_sample": False,
        "use_cache": True,
        "disable_compile": True,
        "streamer": timing,
    }
    if plan is None:
        result = pra.generate(_prompt(question), **kwargs)
    else:
        result = pra.generate_with_native_plan(_prompt(question), plan, **kwargs)
    torch.cuda.synchronize()
    return {
        "visible_prompt_tokens": int(result.prompt_tokens),
        "selected_native_kv_tokens": int(
            result.stats.get("materialized_kv_tokens") or 0
        ),
        "requested_native_kv_tokens": result.stats.get("requested_kv_tokens"),
        "query_encoding_seconds": result.stats.get("query_encoding_seconds", 0.0),
        "routing_seconds": result.stats.get("routing_seconds", 0.0),
        "output_text": result.text,
        "generated_tokens": int(result.generated_tokens),
        "completion_seconds": float(result.latency_seconds),
        "tokens_per_second": int(result.generated_tokens)
        / max(float(result.latency_seconds), 1e-9),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        **timing.metrics(),
    }


def _comparison(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups = sorted(
        {
            (str(row["dataset"]), int(row["reference_limit"]), str(row["condition"]))
            for row in rows
        }
    )
    for dataset, reference_limit, condition in groups:
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["reference_limit"] == reference_limit
            and row["condition"] == condition
        ]
        result.append(
            {
                "dataset": dataset,
                "reference_limit": reference_limit,
                "condition": condition,
                "samples": len(selected),
                "mean_reference_tokens": statistics.fmean(
                    float(row["reference_tokens"]) for row in selected
                ),
                "mean_ttft_ms": statistics.fmean(
                    float(row["ttft_ms"]) for row in selected
                ),
                "mean_itl_ms": statistics.fmean(
                    float(row["itl_ms"]) for row in selected
                ),
                "mean_completion_seconds": statistics.fmean(
                    float(row["completion_seconds"]) for row in selected
                ),
                "mean_peak_cuda_bytes": statistics.fmean(
                    float(row["peak_cuda_bytes"]) for row in selected
                ),
                "mean_token_f1": statistics.fmean(
                    float(row["token_f1"]) for row in selected
                ),
            }
        )
    return result


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
        "--datasets", nargs="*", default=("qasper", "hotpotqa", "2wikimultihopqa")
    )
    parser.add_argument("--max-examples-per-dataset", type=int, default=2)
    parser.add_argument(
        "--reference-limits", nargs="+", type=int, default=(64, 128, 256, 384, 768)
    )
    parser.add_argument("--consumer-layer-count", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--prefetching", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.consumer_layer_count <= 0:
        raise ValueError("consumer-layer-count must be positive.")

    from airllm import AutoModel

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = select_entries(
        manifest, args.datasets, args.max_examples_per_dataset
    )
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
    initialization_seconds = time.perf_counter() - initialized
    tokenizer = air.tokenizer
    prepared = []
    for entry in entries:
        for limit in sorted(set(args.reference_limits)):
            selected, reference_tokens = _bounded_text(
                tokenizer, str(entry["selected_source"]), limit
            )
            prepared.append((entry, limit, selected, reference_tokens))

    rows: list[dict[str, Any]] = []
    for entry, limit, selected, reference_tokens in prepared:
        for repeat in range(args.repeats):
            measurement = _direct_generate(
                air.model,
                tokenizer,
                _prompt(str(entry["question"]), selected),
                args.max_new_tokens,
            )
            row = _annotate(
                measurement,
                entry=entry,
                condition="e0_sdpa",
                regime="warm_repeated" if repeat else "cold_one_shot",
                repeat=repeat,
                reference_tokens=reference_tokens,
                reference_encode_seconds=None,
            )
            row["reference_limit"] = limit
            rows.append(row)

    consumer_layers = tuple(range(-args.consumer_layer_count, 0))
    pra = wrap_airllm_hf_model(
        air,
        pra_config=PRAConfig(
            consumption_layers=consumer_layers,
            address_layers=consumer_layers,
            detail_kv_layers=consumer_layers,
            routing_layer=-1,
            selected_fraction=1.0,
            chunk_tokens=64,
            max_materialized_tokens=max(args.reference_limits),
            reference_device="cpu",
        ),
    )
    for entry, limit, selected, reference_tokens in prepared:
        pra.clear_references()
        for repeat in range(args.repeats):
            measurement = _direct_generate(
                air.model,
                tokenizer,
                _prompt(str(entry["question"]), selected),
                args.max_new_tokens,
            )
            row = _annotate(
                measurement,
                entry=entry,
                condition="e0_eager_injected_disabled",
                regime="warm_repeated" if repeat else "cold_one_shot",
                repeat=repeat,
                reference_tokens=reference_tokens,
                reference_encode_seconds=None,
            )
            row["reference_limit"] = limit
            rows.append(row)

        encode_started = time.perf_counter()
        pra.add_reference(
            f"memory://paper6.6-attribution/{entry['dataset']}/{entry['selection_id']}/{limit}",
            text=selected,
        )
        encode_seconds = time.perf_counter() - encode_started
        route_started = time.perf_counter()
        routed = pra.route(_prompt(str(entry["question"])))
        route_wall_seconds = time.perf_counter() - route_started
        frozen = pra.freeze_native_selection(routed.selected)
        plan = pra.plan_native_materialization(
            frozen,
            full_selected_record=True,
            consumption_layers=consumer_layers,
        )
        for condition, plan_override in (
            ("e2_preselected", plan),
            ("e2_routed", None),
        ):
            for repeat in range(args.repeats):
                measurement = _native_generate(
                    pra,
                    str(entry["question"]),
                    args.max_new_tokens,
                    plan=plan_override,
                )
                measurement["route_wall_seconds_outside_request"] = (
                    route_wall_seconds if condition == "e2_preselected" else 0.0
                )
                measurement["route_query_encoding_seconds_outside_request"] = (
                    routed.query_encoding_seconds
                    if condition == "e2_preselected"
                    else 0.0
                )
                row = _annotate(
                    measurement,
                    entry=entry,
                    condition=condition,
                    regime="warm_repeated" if repeat else "cold_one_shot",
                    repeat=repeat,
                    reference_tokens=reference_tokens,
                    reference_encode_seconds=encode_seconds if repeat == 0 else 0.0,
                )
                row["reference_limit"] = limit
                rows.append(row)

    payload = {
        "schema_version": "paper6.6-airllm-request-attribution-v1",
        "evidence_tier": "CUDA_MECHANISTIC_ATTRIBUTION",
        "model_id": args.model,
        "device": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "initialization_seconds": initialization_seconds,
        "consumer_layers": list(consumer_layers),
        "reference_limits": sorted(set(args.reference_limits)),
        "repeats": args.repeats,
        "conditions": [
            "e0_sdpa",
            "e0_eager_injected_disabled",
            "e2_preselected",
            "e2_routed",
        ],
        "rows": rows,
        "aggregates": _summary(rows),
        "comparisons": _comparison(rows),
    }
    _write(args.output, payload)
    print(json.dumps({"output": str(args.output), "comparisons": payload["comparisons"]}))


if __name__ == "__main__":
    main()
