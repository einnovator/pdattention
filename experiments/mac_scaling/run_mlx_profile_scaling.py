"""Run restartable MLX consumer-layer and live segmented-attention calibration."""

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

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper6_2_mlx.run_answer_quality_pressure import (  # noqa: E402
    QAExample,
    SEEDS,
    _answer_logprob,
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
    "last_3_4": 3 / 4,
    "last_2_3": 2 / 3,
    "last_1_2": 1 / 2,
    "last_1_3": 1 / 3,
    "last_1_4": 1 / 4,
}


def resolve_consumer_layers(layer_count: int, profile: str) -> tuple[int, ...]:
    """Resolve a preregistered contiguous suffix by model-normalized fraction."""

    if profile not in PROFILE_FRACTIONS:
        raise ValueError(f"Unknown Mac scaling consumer profile: {profile}")
    count = max(1, math.ceil(layer_count * PROFILE_FRACTIONS[profile]))
    return tuple(range(layer_count - count, layer_count))


def selected_evidence_source(example: QAExample) -> str:
    """Materialize annotated evidence documents without retrieval changes."""

    selected = [
        document
        for document in example.documents
        if document.document_id in example.evidence_document_ids
    ]
    if not selected:
        return example.source
    return "\n\n".join(
        f"Document: {document.title}\n{document.text}" for document in selected
    )


def _bounded_tokens(tokenizer, text: str, limit: int) -> list[int]:
    values = list(tokenizer.encode(text, add_special_tokens=False))
    return values[:limit]


def _command_value(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, check=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def runtime_metadata() -> dict[str, object]:
    """Capture the hardware and software identity needed to interpret timings."""

    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    chip = _command_value(
        ["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"]
    )
    chip_name = None
    if chip:
        chip_name = next(
            (
                line.split(":", 1)[1].strip()
                for line in chip.splitlines()
                if line.strip().startswith("Chip:")
            ),
            None,
        )
    # Apple publishes nominal unified-memory bandwidth by chip family. Keep the
    # source explicit: this is platform metadata, not a measured benchmark.
    declared_bandwidth = {
        "Apple M4 Pro": 273,
    }.get(chip_name)

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "transformers": version("transformers"),
        "hardware_model": _command_value(["sysctl", "-n", "hw.model"]),
        "chip": chip_name,
        "physical_memory_bytes": int(
            _command_value(["sysctl", "-n", "hw.memsize"]) or 0
        ),
        "declared_memory_bandwidth_gbps": declared_bandwidth,
        "memory_bandwidth_source": (
            "Apple chip specification" if declared_bandwidth is not None else None
        ),
        "gpu_wired_limit_mb": int(
            _command_value(["sysctl", "-n", "iogpu.wired_limit_mb"]) or 0
        ),
        "git_commit": _command_value(["git", "rev-parse", "HEAD"]),
    }


def _cohort(examples: list[QAExample], examples_per_seed: int):
    for seed in SEEDS:
        shuffled = list(examples)
        random.Random(seed).shuffle(shuffled)
        for example in shuffled[:examples_per_seed]:
            yield seed, example


def _read_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_checkpoint(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["condition"])), []).append(row)
    result = []
    for (dataset, condition), values in sorted(groups.items()):
        result.append(
            {
                "dataset": dataset,
                "condition": condition,
                "samples": len(values),
                "seeds": len({int(row["seed"]) for row in values}),
                "exact_match": fmean(float(row["exact_match"]) for row in values),
                "token_f1": fmean(float(row["token_f1"]) for row in values),
                "gold_answer_logprob": fmean(
                    float(row["gold_answer_logprob"]) for row in values
                ),
                "completion_latency_ms": fmean(
                    float(row["completion_latency_ms"]) for row in values
                ),
                "active_detail_bytes": fmean(
                    float(row["active_detail_bytes"]) for row in values
                ),
                "peak_unified_memory_bytes": max(
                    int(row["peak_unified_memory_bytes"]) for row in values
                ),
                "sequence_agreement_vs_e0": fmean(
                    float(row["sequence_agreement_vs_e0"]) for row in values
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
    installed_layers = install_qwen3_segmented_attention(model)
    layer_count = len(model.layers)
    model_resident_bytes = int(mx.get_active_memory())
    checkpoint = args.output.with_suffix(".jsonl")
    rows = _read_checkpoint(checkpoint) if args.resume else []
    completed = {
        (str(row["dataset"]), int(row["seed"]), str(row["example_id"]), str(row["condition"]))
        for row in rows
    }

    profiles = tuple(args.profile or PROFILE_FRACTIONS)
    for dataset in args.dataset:
        examples = _examples(dataset, args.cache_dir)
        for seed, example in _cohort(examples, args.examples_per_seed):
            source_text = selected_evidence_source(example)
            source = _bounded_tokens(tokenizer, source_text, args.max_source_tokens)
            query_text = (
                "Answer the question using the available evidence. Give only the "
                f"short answer.\nQuestion: {example.question}\nAnswer:"
            )
            query = list(tokenizer.encode(query_text, add_special_tokens=False))
            answer = list(tokenizer.encode(" " + example.answer, add_special_tokens=False))
            identity = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

            started = time.perf_counter()
            ordinary = make_prompt_cache(model)
            encoded = model(mx.array(source, dtype=mx.int32)[None], cache=ordinary)
            mx.eval(encoded)
            ordinary_states = _cache_snapshot(ordinary)
            e0_encode_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            memory = encode_native_memory(model, source)
            e2_encode_ms = (time.perf_counter() - started) * 1000.0
            conditions: list[tuple[str, tuple[int, ...] | None, bool]] = [
                ("E0_SELECTED", None, False),
                ("E2_CONCAT_ALL", tuple(range(layer_count)), False),
                ("E2_SEGMENTED_ALL", tuple(range(layer_count)), True),
            ]
            conditions.extend(
                (f"E2_SEGMENTED_{profile.upper()}", resolve_consumer_layers(layer_count, profile), True)
                for profile in profiles
                if profile != "all_layers"
            )

            baseline_output: list[int] | None = None
            pending_rows: list[dict[str, object]] = []
            for condition, layers, segmented in conditions:
                key = (dataset, seed, example.example_id, condition)
                if key in completed:
                    continue

                def cache_factory():
                    if condition == "E0_SELECTED":
                        return _restore_cache(model, ordinary_states)
                    return make_native_prompt_cache(
                        model,
                        memory,
                        selected_layers=layers,
                        segmented=segmented,
                    )

                mx.reset_peak_memory()
                score_started = time.perf_counter()
                logprob = _answer_logprob(model, query, answer, cache_factory())
                score_ms = (time.perf_counter() - score_started) * 1000.0
                generated = _generate_timed(
                    model, tokenizer, query, cache_factory(), args.max_new_tokens
                )
                exact, f1 = _metrics(generated["output"], example.answer)
                output_ids = list(map(int, generated["output_token_ids"]))
                if condition == "E0_SELECTED":
                    baseline_output = output_ids
                if baseline_output is None:
                    raise RuntimeError("E0_SELECTED must execute before native conditions.")
                active_bytes = 0 if layers is None else memory.selected_nbytes(layers)
                row = {
                    "schema_version": "pra-mac-scaling-v1",
                    "model_id": args.model,
                    "model_revision": resolved.sha,
                    "quantization": "4bit",
                    "dataset": dataset,
                    "seed": seed,
                    "example_id": example.example_id,
                    "selection_sha256": identity,
                    "selection_policy": "annotated_evidence_documents",
                    "condition": condition,
                    "representation": "selected_text" if layers is None else "native_kv",
                    "segmented_attention": segmented,
                    "consumer_layers": [] if layers is None else list(layers),
                    "consumer_layer_count": 0 if layers is None else len(layers),
                    "consumer_layer_fraction": 0.0 if layers is None else len(layers) / layer_count,
                    "source_tokens": len(source),
                    "query_tokens": len(query),
                    "visible_prompt_tokens": len(source) + len(query) if layers is None else len(query),
                    "selected_native_kv_tokens": 0 if layers is None else len(source) * len(layers),
                    "active_detail_bytes": active_bytes,
                    "retained_detail_bytes": 0 if layers is None else memory.nbytes,
                    "model_resident_bytes": model_resident_bytes,
                    "peak_unified_memory_bytes": int(mx.get_peak_memory()),
                    "e0_encode_ms": e0_encode_ms,
                    "e2_encode_ms": e2_encode_ms,
                    "gold_answer_logprob": logprob,
                    "gold_logprob_latency_ms": score_ms,
                    "gold_answer": example.answer,
                    "output": generated["output"],
                    "output_token_ids": output_ids,
                    "exact_match": exact,
                    "token_f1": f1,
                    "first_token_agreement_vs_e0": float(
                        bool(output_ids and baseline_output)
                        and output_ids[0] == baseline_output[0]
                    ),
                    "sequence_agreement_vs_e0": float(output_ids == baseline_output),
                    "ttft_ms": generated["ttft_ms"],
                    "itl_ms": generated["itl_ms"],
                    "completion_latency_ms": generated["completion_latency_ms"],
                    "generated_tokens": generated["generated_tokens"],
                    "evidence_tier": "MODEL_BACKED_NATURAL_QA_CALIBRATION",
                }
                rows.append(row)
                pending_rows.append(row)
                completed.add(key)
                _append_checkpoint(checkpoint, row)
            del ordinary_states, ordinary, memory
            mx.clear_cache()

    payload = {
        "schema_version": "pra-mac-scaling-v1",
        "experiment": "mlx_consumer_layer_segmented_scaling",
        "runtime": runtime_metadata(),
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
    print(json.dumps({"rows": len(result["rows"]), "output": result["model_id"]}, indent=2))
