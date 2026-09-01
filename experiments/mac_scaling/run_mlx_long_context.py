"""Run restartable context-dilution and RoPE-frame controls on MLX.

The selected-evidence baseline is encoded once per example.  Longer conditions
keep that evidence fixed and append deterministic documents from the same
dataset.  A second native pass encodes the exact same long token stream, which
separates positional transport error from distractor-induced quality changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.mac_scaling.run_mlx_profile_scaling import (  # noqa: E402
    runtime_metadata,
    selected_evidence_source,
)
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


def unique_seed_cohort(examples: list[QAExample], count: int) -> list[tuple[int, QAExample]]:
    """Choose deterministic, non-duplicated examples while retaining seed identity."""

    chosen: list[tuple[int, QAExample]] = []
    seen: set[str] = set()
    for seed in SEEDS:
        candidates = list(examples)
        random.Random(seed).shuffle(candidates)
        example = next((item for item in candidates if item.example_id not in seen), None)
        if example is None:
            break
        chosen.append((seed, example))
        seen.add(example.example_id)
        if len(chosen) == count:
            break
    return chosen


def build_long_source_tokens(
    tokenizer,
    selected_text: str,
    dataset_examples: list[QAExample],
    *,
    target_tokens: int,
    max_selected_tokens: int,
    seed: int,
) -> tuple[list[int], int]:
    """Append deterministic non-evidence documents to a fixed selected prefix."""

    selected = list(tokenizer.encode(selected_text, add_special_tokens=False))[
        :max_selected_tokens
    ]
    if target_tokens <= len(selected):
        return selected[:target_tokens], len(selected[:target_tokens])

    documents = []
    selected_hash = hashlib.sha256(selected_text.encode("utf-8")).hexdigest()
    for example in dataset_examples[:512]:
        for document in example.documents:
            text = f"\n\nDistractor document: {document.title}\n{document.text}"
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != selected_hash:
                documents.append(text)
    if not documents:
        documents = ["\n\nDistractor document: unrelated background material."]
    random.Random(seed).shuffle(documents)
    tokens = list(selected)
    cursor = 0
    while len(tokens) < target_tokens:
        text = documents[cursor % len(documents)]
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        cursor += 1
    return list(map(int, tokens[:target_tokens])), len(selected)


def _checkpoint_rows(path: Path) -> dict[tuple[object, ...], dict[str, object]]:
    rows: dict[tuple[object, ...], dict[str, object]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            row["dataset"], row["seed"], row["example_id"],
            row["context_target_tokens"], row["condition"],
        )
        rows[key] = row
    return rows


def _append(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _first_logits(model, query: list[int], cache):
    import mlx.core as mx
    import numpy as np

    logits = model(mx.array(query, dtype=mx.int32)[None], cache=cache)
    values = logits[0, -1].astype(mx.float32)
    mx.eval(values)
    return np.asarray(values)


def _execute(
    model,
    tokenizer,
    query: list[int],
    answer: list[int],
    answer_text: str,
    cache_factory,
    max_new_tokens: int,
) -> tuple[dict[str, object], object]:
    import mlx.core as mx

    mx.reset_peak_memory()
    logits = _first_logits(model, query, cache_factory())
    score_started = time.perf_counter()
    logprob = _answer_logprob(model, query, answer, cache_factory())
    score_ms = (time.perf_counter() - score_started) * 1000.0
    generated = _generate_timed(
        model, tokenizer, query, cache_factory(), max_new_tokens
    )
    exact, f1 = _metrics(str(generated["output"]), answer_text)
    return (
        {
            **generated,
            "gold_answer_logprob": logprob,
            "gold_logprob_latency_ms": score_ms,
            "exact_match": exact,
            "token_f1": f1,
            "peak_unified_memory_bytes": int(mx.get_peak_memory()),
        },
        logits,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    import mlx.core as mx
    import numpy as np
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
    model_resident_bytes = int(mx.get_active_memory())
    checkpoint = args.output.with_suffix(".jsonl")
    existing = _checkpoint_rows(checkpoint) if args.resume else {}

    for dataset in args.dataset:
        examples = _examples(dataset, args.cache_dir)
        cohort = unique_seed_cohort(examples, args.examples_per_dataset)
        for seed, example in cohort:
            selected_text = selected_evidence_source(example)
            selected_tokens = list(
                tokenizer.encode(selected_text, add_special_tokens=False)
            )[: args.max_selected_tokens]
            query_text = (
                "Answer the question using the available evidence. Give only the "
                f"short answer.\nQuestion: {example.question}\nAnswer:"
            )
            query = list(tokenizer.encode(query_text, add_special_tokens=False))
            answer = list(tokenizer.encode(" " + example.answer, add_special_tokens=False))
            identity = hashlib.sha256(selected_text.encode("utf-8")).hexdigest()

            selected_ordinary = make_prompt_cache(model)
            started = time.perf_counter()
            selected_encoded = model(
                mx.array(selected_tokens, dtype=mx.int32)[None],
                cache=selected_ordinary,
            )
            mx.eval(selected_encoded)
            selected_states = _cache_snapshot(selected_ordinary)
            selected_e0_encode_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            selected_memory = encode_native_memory(model, selected_tokens)
            selected_e2_encode_ms = (time.perf_counter() - started) * 1000.0

            selected_results: dict[str, tuple[dict[str, object], object]] = {
                "E0_SELECTED": _execute(
                    model, tokenizer, query, answer, example.answer,
                    lambda: _restore_cache(model, selected_states), args.max_new_tokens,
                ),
                "E2_SELECTED": _execute(
                    model, tokenizer, query, answer, example.answer,
                    lambda: make_native_prompt_cache(model, selected_memory, segmented=True),
                    args.max_new_tokens,
                ),
            }

            for target in args.context_tokens:
                conditions = (
                    "FULL_VISIBLE", "E2_SOURCE_RELATIVE", "E2_QUERY_RESTART",
                    "E0_SELECTED", "E2_SELECTED",
                )
                keys = [
                    (dataset, seed, example.example_id, target, condition)
                    for condition in conditions
                ]
                if all(key in existing for key in keys):
                    continue
                if target + len(query) + len(answer) > args.model_context_limit:
                    unsupported = []
                    for condition, key in zip(conditions, keys):
                        row = {
                            "schema_version": "pra-mac-long-context-v2",
                            "model_id": args.model,
                            "model_revision": resolved.sha,
                            "dataset": dataset,
                            "seed": seed,
                            "example_id": example.example_id,
                            "selection_sha256": identity,
                            "context_target_tokens": target,
                            "condition": condition,
                            "status": "NOT_RUN_MODEL_CONTEXT_LIMIT",
                            "model_context_limit": args.model_context_limit,
                        }
                        existing[key] = row
                        unsupported.append(row)
                    _append(checkpoint, unsupported)
                    continue

                full_tokens, selected_count = build_long_source_tokens(
                    tokenizer,
                    selected_text,
                    examples,
                    target_tokens=target,
                    max_selected_tokens=args.max_selected_tokens,
                    seed=seed,
                )
                started = time.perf_counter()
                ordinary = make_prompt_cache(model)
                encoded = model(
                    mx.array(full_tokens, dtype=mx.int32)[None], cache=ordinary
                )
                mx.eval(encoded)
                ordinary_states = _cache_snapshot(ordinary)
                e0_encode_ms = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                memory = encode_native_memory(model, full_tokens)
                e2_encode_ms = (time.perf_counter() - started) * 1000.0

                full_result, full_logits = _execute(
                    model, tokenizer, query, answer, example.answer,
                    lambda: _restore_cache(model, ordinary_states), args.max_new_tokens,
                )
                source_result, source_logits = _execute(
                    model, tokenizer, query, answer, example.answer,
                    lambda: make_native_prompt_cache(model, memory, segmented=True),
                    args.max_new_tokens,
                )
                restart_result, restart_logits = _execute(
                    model, tokenizer, query, answer, example.answer,
                    lambda: make_native_prompt_cache(
                        model, memory, segmented=True, query_position_base=0
                    ),
                    args.max_new_tokens,
                )
                computed = {
                    "FULL_VISIBLE": (full_result, full_logits, e0_encode_ms, 0),
                    "E2_SOURCE_RELATIVE": (
                        source_result, source_logits, e2_encode_ms, memory.nbytes
                    ),
                    "E2_QUERY_RESTART": (
                        restart_result, restart_logits, e2_encode_ms, memory.nbytes
                    ),
                    "E0_SELECTED": (
                        *selected_results["E0_SELECTED"], selected_e0_encode_ms, 0
                    ),
                    "E2_SELECTED": (
                        *selected_results["E2_SELECTED"],
                        selected_e2_encode_ms,
                        selected_memory.nbytes,
                    ),
                }
                pending = []
                full_ids = list(map(int, full_result["output_token_ids"]))
                for condition, key in zip(conditions, keys):
                    if key in existing:
                        continue
                    result, logits, encode_ms, active_bytes = computed[condition]
                    delta = np.asarray(logits, dtype=np.float32) - np.asarray(
                        full_logits, dtype=np.float32
                    )
                    native_full = condition.startswith("E2_") and condition not in {
                        "E2_SELECTED"
                    }
                    row = {
                        "schema_version": "pra-mac-long-context-v2",
                        "model_id": args.model,
                        "model_revision": resolved.sha,
                        "quantization": "4bit",
                        "dataset": dataset,
                        "seed": seed,
                        "example_id": example.example_id,
                        "selection_sha256": identity,
                        "selection_policy": "annotated_evidence_documents",
                        "context_target_tokens": target,
                        "actual_context_tokens": len(full_tokens),
                        "selected_evidence_tokens": selected_count,
                        "distractor_tokens": max(0, len(full_tokens) - selected_count),
                        "condition": condition,
                        "status": "MEASURED",
                        "representation": (
                            "selected_text" if condition == "E0_SELECTED"
                            else "selected_native_kv" if condition == "E2_SELECTED"
                            else "full_visible_text" if condition == "FULL_VISIBLE"
                            else "full_native_kv"
                        ),
                        "position_policy": (
                            "query_restart_bug" if condition == "E2_QUERY_RESTART"
                            else "source_relative" if native_full
                            else "ordinary_sequential"
                        ),
                        "visible_prompt_tokens": (
                            len(query)
                            if condition.startswith("E2_")
                            else len(query) + (
                                selected_count if condition == "E0_SELECTED" else len(full_tokens)
                            )
                        ),
                        "active_native_kv_tokens": (
                            (selected_count if condition == "E2_SELECTED" else len(full_tokens))
                            * len(model.layers)
                            if condition.startswith("E2_") else 0
                        ),
                        "active_detail_bytes": int(active_bytes),
                        "model_resident_bytes": model_resident_bytes,
                        "encode_ms": encode_ms,
                        **result,
                        "first_token_agreement_vs_full": float(
                            int(np.argmax(logits)) == int(np.argmax(full_logits))
                        ),
                        "sequence_agreement_vs_full": float(
                            list(map(int, result["output_token_ids"])) == full_ids
                        ),
                        "first_logit_rmse_vs_full": float(np.sqrt(np.mean(delta * delta))),
                        "first_logit_max_abs_error_vs_full": float(np.max(np.abs(delta))),
                        "evidence_tier": "MODEL_BACKED_NATURAL_QA_CALIBRATION",
                        "global_sequential_equivalence": (
                            "source_relative_for_one_contiguous_resource"
                            if condition == "E2_SOURCE_RELATIVE" else None
                        ),
                    }
                    existing[key] = row
                    pending.append(row)
                _append(checkpoint, pending)
                del encoded, ordinary, ordinary_states, memory
                mx.clear_cache()

            del selected_encoded, selected_ordinary, selected_states, selected_memory
            mx.clear_cache()

    rows = list(existing.values())
    payload = {
        "schema_version": "pra-mac-long-context-v2",
        "experiment": "mlx_context_dilution_and_position_geometry",
        "timing_contract": (
            "All ordinary and native source encodes are synchronized before "
            "encode_ms and request timing stop; occupied context is the actual "
            "encoded source length."
        ),
        "runtime": runtime_metadata(),
        "model_id": args.model,
        "model_revision": resolved.sha,
        "layer_count": len(model.layers),
        "segmented_layers_patched": installed_layers,
        "seeds": list(SEEDS),
        "context_targets": list(args.context_tokens),
        "rows": sorted(
            rows,
            key=lambda row: (
                str(row["dataset"]), int(row["seed"]), str(row["example_id"]),
                int(row["context_target_tokens"]), str(row["condition"]),
            ),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--dataset", action="append",
        choices=("qasper", "hotpotqa", "2wikimultihopqa"), required=True,
    )
    parser.add_argument("--context-tokens", action="append", type=int, required=True)
    parser.add_argument("--examples-per-dataset", type=int, default=5)
    parser.add_argument("--max-selected-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--model-context-limit", type=int, default=40960)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"model": result["model_id"], "rows": len(result["rows"])}, indent=2))
