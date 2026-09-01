"""Measure matched MLX E0/E2 economics and consumer-depth scaling.

The runner keeps source selection fixed and separates three quantities that an
earlier MLX experiment accidentally mixed through lazy evaluation:

* cold cost: source/native encoding plus query generation;
* warm cost: query generation from already synchronized reusable state;
* segmented consumer cost: warm native generation at bounded layer suffixes.

One output file is used per model. JSONL checkpoints make the long 32B run
restartable without retaining more than one model in unified memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper6_2_mlx.run_answer_quality_pressure import (  # noqa: E402
    QAExample,
    SEEDS,
    _answer_logprob,
    _bounded_source,
    _examples,
    _metrics,
)
from experiments.paper6_2_mlx.run_matched_e0_e2 import (  # noqa: E402
    _cache_snapshot,
    _generate_timed,
    _restore_cache,
)


PROFILE_FRACTIONS = {
    "all_layers": 1.0,
    "last_7_8": 7 / 8,
    "last_3_4": 3 / 4,
    "last_2_3": 2 / 3,
    "last_1_2": 1 / 2,
}
BASELINE_CONDITIONS = ("E0_WARM", "E2_CONCAT_WARM")


def resolve_consumer_layers(layer_count: int, profile: str) -> tuple[int, ...]:
    """Return a contiguous late-layer suffix for a normalized profile."""

    if profile not in PROFILE_FRACTIONS:
        raise ValueError(f"Unknown consumer profile: {profile}")
    count = max(1, math.ceil(layer_count * PROFILE_FRACTIONS[profile]))
    return tuple(range(layer_count - count, layer_count))


def matched_costs(row: dict[str, Any]) -> dict[str, float]:
    """Derive explicit cold and warm costs from synchronized measurements."""

    completion = float(row["completion_latency_ms"])
    encode = float(row["representation_encode_ms"])
    return {
        "warm_request_ms": completion,
        "cold_usable_context_ms": encode + completion,
    }


def _command_value(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, check=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _runtime_metadata() -> dict[str, object]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "transformers": version("transformers"),
        "hardware_model": _command_value(["sysctl", "-n", "hw.model"]),
        "physical_memory_bytes": int(
            _command_value(["sysctl", "-n", "hw.memsize"]) or 0
        ),
        "git_commit": _command_value(["git", "rev-parse", "HEAD"]),
    }


def _cohort(args: argparse.Namespace) -> list[tuple[int, QAExample]]:
    cohort: list[tuple[int, QAExample]] = []
    for dataset in args.dataset:
        candidates = _examples(dataset, args.cache_dir)
        for seed in SEEDS:
            shuffled = list(candidates)
            random.Random(seed).shuffle(shuffled)
            cohort.extend((seed, example) for example in shuffled[: args.examples_per_seed])
    return cohort


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["dataset"]),
        int(row["seed"]),
        str(row["example_id"]),
        str(row["condition"]),
    )


def _prepared_tokens(tokenizer: Any, example: QAExample, limit: int):
    source = _bounded_source(tokenizer, example.source, limit)
    query_text = (
        "Answer the question using the available evidence. Give only the "
        f"short answer.\nQuestion: {example.question}\nAnswer:"
    )
    query = list(tokenizer.encode(query_text, add_special_tokens=False))
    answer = list(tokenizer.encode(" " + example.answer, add_special_tokens=False))
    return source, query, answer


def _measure_condition(
    *,
    model: Any,
    tokenizer: Any,
    example: QAExample,
    seed: int,
    source: list[int],
    query: list[int],
    answer: list[int],
    condition: str,
    cache_factory: Callable[[], Any],
    representation_encode_ms: float,
    consumer_layers: tuple[int, ...],
    retained_bytes: int,
    active_bytes: int,
    max_new_tokens: int,
    baseline_ids: list[int] | None,
) -> dict[str, Any]:
    import mlx.core as mx

    score_started = time.perf_counter()
    logprob = _answer_logprob(model, query, answer, cache_factory())
    score_ms = (time.perf_counter() - score_started) * 1000.0
    mx.reset_peak_memory()
    generated = _generate_timed(
        model, tokenizer, query, cache_factory(), max_new_tokens
    )
    exact, f1 = _metrics(str(generated["output"]), example.answer)
    output_ids = list(map(int, generated["output_token_ids"]))
    row: dict[str, Any] = {
        "schema_version": "paper6.2-mlx-model-consumer-scaling-v2",
        "dataset": example.dataset,
        "seed": seed,
        "example_id": example.example_id,
        "selection_sha256": hashlib.sha256(bytes(str(source), "utf-8")).hexdigest(),
        "condition": condition,
        "source_tokens": len(source),
        "query_tokens": len(query),
        "visible_prompt_tokens": (
            len(source) + len(query) if condition == "E0_WARM" else len(query)
        ),
        "consumer_layers": list(consumer_layers),
        "consumer_layer_count": len(consumer_layers),
        "model_layer_count": len(model.layers),
        "consumer_layer_fraction": (
            len(consumer_layers) / len(model.layers) if consumer_layers else 0.0
        ),
        "representation_encode_ms": representation_encode_ms,
        "retained_detail_bytes": retained_bytes,
        "active_detail_bytes": active_bytes,
        "gold_answer": example.answer,
        "gold_answer_logprob": logprob,
        "gold_logprob_latency_ms": score_ms,
        "output": generated["output"],
        "output_token_ids": output_ids,
        "exact_match": exact,
        "token_f1": f1,
        "sequence_agreement_vs_e0": (
            1.0 if baseline_ids is None else float(output_ids == baseline_ids)
        ),
        "first_token_agreement_vs_e0": (
            1.0
            if baseline_ids is None
            else float(bool(output_ids and baseline_ids) and output_ids[0] == baseline_ids[0])
        ),
        "peak_unified_memory_bytes": int(mx.get_peak_memory()),
        "ttft_ms": generated["ttft_ms"],
        "itl_ms": generated["itl_ms"],
        "completion_latency_ms": generated["completion_latency_ms"],
        "generated_tokens": generated["generated_tokens"],
        "evidence_tier": "MODEL_BACKED_NATURAL_QA_SCALING",
    }
    row.update(matched_costs(row))
    return row


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        result.append(
            {
                "condition": condition,
                "samples": len(selected),
                "seeds": len({int(row["seed"]) for row in selected}),
                "token_f1": fmean(float(row["token_f1"]) for row in selected),
                "gold_answer_logprob": fmean(
                    float(row["gold_answer_logprob"]) for row in selected
                ),
                "sequence_agreement_vs_e0": fmean(
                    float(row["sequence_agreement_vs_e0"]) for row in selected
                ),
                "consumer_layer_fraction": fmean(
                    float(row["consumer_layer_fraction"]) for row in selected
                ),
                "warm_request_ms": fmean(
                    float(row["warm_request_ms"]) for row in selected
                ),
                "cold_usable_context_ms": fmean(
                    float(row["cold_usable_context_ms"]) for row in selected
                ),
                "active_detail_bytes": fmean(
                    float(row["active_detail_bytes"]) for row in selected
                ),
                "peak_unified_memory_bytes": max(
                    int(row["peak_unified_memory_bytes"]) for row in selected
                ),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    import mlx.core as mx
    from huggingface_hub import model_info
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx import (
        encode_native_memory,
        install_qwen3_segmented_attention,
        make_native_prompt_cache,
    )

    model, tokenizer = load(args.model, revision=args.revision)
    resolved = model_info(args.model, revision=args.revision)
    layer_count = len(model.layers)
    checkpoint = args.output.with_suffix(".jsonl")
    rows = _read_rows(checkpoint) if args.resume else []
    completed = {_key(row) for row in rows}
    baseline_outputs = {
        (str(row["dataset"]), int(row["seed"]), str(row["example_id"])): list(
            map(int, row["output_token_ids"])
        )
        for row in rows
        if row["condition"] == "E0_WARM"
    }
    cohort = _cohort(args)

    # Phase one intentionally uses the unpatched MLX model. Both ordinary and
    # native source encodings are synchronized before warm request timing.
    for seed, example in cohort:
        source, query, answer = _prepared_tokens(
            tokenizer, example, args.max_source_tokens
        )
        started = time.perf_counter()
        ordinary = make_prompt_cache(model)
        encoded = model(mx.array(source, dtype=mx.int32)[None], cache=ordinary)
        mx.eval(encoded)
        ordinary_states = _cache_snapshot(ordinary)
        e0_encode_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        memory = encode_native_memory(model, source)
        e2_encode_ms = (time.perf_counter() - started) * 1000.0
        identity = (example.dataset, seed, example.example_id)
        baseline = baseline_outputs.get(identity)
        for condition in BASELINE_CONDITIONS:
            row_key = (*identity, condition)
            if row_key in completed:
                continue
            if condition == "E0_WARM":
                factory = lambda: _restore_cache(model, ordinary_states)
                encode_ms = e0_encode_ms
                layers: tuple[int, ...] = ()
                retained = active = 0
            else:
                factory = lambda: make_native_prompt_cache(model, memory)
                encode_ms = e2_encode_ms
                layers = tuple(range(layer_count))
                retained = active = memory.nbytes
            row = _measure_condition(
                model=model,
                tokenizer=tokenizer,
                example=example,
                seed=seed,
                source=source,
                query=query,
                answer=answer,
                condition=condition,
                cache_factory=factory,
                representation_encode_ms=encode_ms,
                consumer_layers=layers,
                retained_bytes=retained,
                active_bytes=active,
                max_new_tokens=args.max_new_tokens,
                baseline_ids=baseline if condition != "E0_WARM" else None,
            )
            rows.append(row)
            completed.add(row_key)
            _append_row(checkpoint, row)
            if condition == "E0_WARM":
                baseline = list(map(int, row["output_token_ids"]))
                baseline_outputs[identity] = baseline
        del encoded, ordinary, ordinary_states, memory
        mx.clear_cache()

    installed_layers = install_qwen3_segmented_attention(model)
    profiles = tuple(args.profile or PROFILE_FRACTIONS)
    for seed, example in cohort:
        source, query, answer = _prepared_tokens(
            tokenizer, example, args.max_source_tokens
        )
        started = time.perf_counter()
        memory = encode_native_memory(model, source)
        encode_ms = (time.perf_counter() - started) * 1000.0
        identity = (example.dataset, seed, example.example_id)
        baseline = baseline_outputs[identity]
        for profile in profiles:
            condition = f"E2_SEGMENTED_{profile.upper()}"
            row_key = (*identity, condition)
            if row_key in completed:
                continue
            layers = resolve_consumer_layers(layer_count, profile)
            factory = lambda layers=layers: make_native_prompt_cache(
                model, memory, selected_layers=layers, segmented=True
            )
            row = _measure_condition(
                model=model,
                tokenizer=tokenizer,
                example=example,
                seed=seed,
                source=source,
                query=query,
                answer=answer,
                condition=condition,
                cache_factory=factory,
                representation_encode_ms=encode_ms,
                consumer_layers=layers,
                retained_bytes=memory.nbytes,
                active_bytes=memory.selected_nbytes(layers),
                max_new_tokens=args.max_new_tokens,
                baseline_ids=baseline,
            )
            rows.append(row)
            completed.add(row_key)
            _append_row(checkpoint, row)
        del memory
        mx.clear_cache()

    payload: dict[str, object] = {
        "schema_version": "paper6.2-mlx-model-consumer-scaling-v2",
        "experiment": "matched_economics_and_consumer_depth",
        "timing_contract": {
            "cold": "synchronized representation encoding plus request generation",
            "warm": "request generation from synchronized reusable representation",
            "legacy_mixed_ratio": "not reported: cold E0 divided by warm E2 is unmatched",
        },
        "runtime": _runtime_metadata(),
        "model_id": args.model,
        "model_revision": resolved.sha,
        "layer_count": layer_count,
        "segmented_layers_patched": installed_layers,
        "seeds": list(SEEDS),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("qasper", "hotpotqa", "2wikimultihopqa"),
        required=True,
    )
    parser.add_argument("--profile", action="append", choices=tuple(PROFILE_FRACTIONS))
    parser.add_argument("--examples-per-seed", type=int, default=1)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"model": result["model_id"], "rows": len(result["rows"])}, indent=2))
