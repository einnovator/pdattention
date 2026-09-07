"""Measure full-context and PRA-selected generation for Qwen3.5-27B MLX."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper4_5_runtime.extract_mlx_catalog_features import (  # noqa: E402
    load_split_examples,
)
from experiments.paper6_2_mlx.run_answer_quality_pressure import _metrics  # noqa: E402
from experiments.paper6_2_mlx.run_matched_e0_e2 import _generate_timed  # noqa: E402
from pra_hf.router import PRARouter  # noqa: E402


def select_chunk_indices(
    scores: torch.Tensor,
    spans: list[tuple[int, int]],
    *,
    token_budget: int,
) -> list[int]:
    """Select highest-scoring chunks within a physical source-token budget."""

    if scores.ndim != 1 or len(scores) != len(spans):
        raise ValueError("Scores and chunk spans must describe the same candidates.")
    if token_budget <= 0:
        raise ValueError("The selected-context token budget must be positive.")
    selected: list[int] = []
    consumed = 0
    for index in torch.argsort(scores, descending=True).tolist():
        start, end = spans[index]
        length = max(0, int(end) - int(start))
        if not length:
            continue
        if selected and consumed + length > token_budget:
            continue
        selected.append(int(index))
        consumed += length
        if consumed >= token_budget:
            break
    return selected or [int(torch.argmax(scores).item())]


def _scores(feature: dict[str, Any], router: Any | None) -> torch.Tensor:
    query = feature["queries"]["last"].reshape(1, -1).float()
    memory = feature["memory_gists"].float()
    if router is None:
        return (
            F.normalize(query, dim=-1)
            @ F.normalize(memory, dim=-1).transpose(0, 1)
        )[0]
    with torch.no_grad():
        return router.scores(query, memory)[0].cpu()


def _selected_text(
    tokenizer: Any,
    source_ids: list[int],
    spans: list[tuple[int, int]],
    selected: list[int],
) -> str:
    ordered = sorted(selected, key=lambda index: spans[index])
    return "\n".join(
        tokenizer.decode(source_ids[spans[index][0] : spans[index][1]]).strip()
        for index in ordered
    )


def _prompt(tokenizer: Any, source: str, question: str) -> list[int]:
    content = (
        "Use only the evidence below. Give only the short answer.\n\n"
        f"Evidence:\n{source}\n\nQuestion: {question}\nAnswer:"
    )
    if tokenizer.chat_template:
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return list(tokenizer.encode(content, add_special_tokens=False))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["dataset"], row["condition"]) for row in rows})
    for dataset, condition in keys:
        values = [
            row
            for row in rows
            if row["dataset"] == dataset and row["condition"] == condition
        ]
        output.append(
            {
                "dataset": dataset,
                "condition": condition,
                "examples": len(values),
                "exact_match": fmean(row["exact_match"] for row in values),
                "token_f1": fmean(row["token_f1"] for row in values),
                "evidence_recall": fmean(row["evidence_recall"] for row in values),
                "visible_prompt_tokens": fmean(
                    row["visible_prompt_tokens"] for row in values
                ),
                "ttft_ms": fmean(row["ttft_ms"] for row in values),
                "completion_latency_ms": fmean(
                    row["completion_latency_ms"] for row in values
                ),
                "output_tokens_per_second": fmean(
                    row["output_tokens_per_second"] for row in values
                ),
                "peak_unified_memory_bytes": max(
                    row["peak_unified_memory_bytes"] for row in values
                ),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx_lm
    from huggingface_hub import model_info
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    manifest = json.loads(
        (args.feature_dir / "feature_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    features = torch.load(
        args.feature_dir / "router_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )
    examples = load_split_examples(
        args.cache_dir,
        int(manifest["splits"]["test"]["dataset_counts"]["hotpotqa"]),
        int(manifest["splits"]["test"]["offset"]),
        int(manifest["seed"]),
    )
    by_id = {example["id"]: example for example in examples}
    router = PRARouter.from_pretrained(args.router)
    model, tokenizer_wrapper = load(args.model, revision=args.revision)
    tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
    resolved = model_info(args.model, revision=args.revision)
    rows: list[dict[str, Any]] = []
    checkpoint = args.output.with_suffix(".jsonl")
    if args.resume and checkpoint.is_file():
        rows = [
            json.loads(line)
            for line in checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    complete = {(row["example_id"], row["condition"]) for row in rows}
    for feature_index, feature in enumerate(features[: args.max_examples or None]):
        example = by_id[str(feature["example_id"])]
        source_ids = list(
            tokenizer(example["source"], add_special_tokens=False).input_ids
        )[: args.max_source_tokens]
        usable = [
            index
            for index, span in enumerate(feature["chunk_spans"])
            if int(span[0]) < len(source_ids)
        ]
        spans = [
            (int(feature["chunk_spans"][index][0]), min(int(feature["chunk_spans"][index][1]), len(source_ids)))
            for index in usable
        ]
        trimmed = {
            **feature,
            "memory_gists": feature["memory_gists"][usable],
            "positive_mask": feature["positive_mask"][usable],
            "chunk_spans": spans,
        }
        budget = max(1, math.ceil(len(source_ids) * args.selected_fraction))
        generic_scores = _scores(trimmed, None)
        learned_scores = _scores(trimmed, router)
        oracle_scores = generic_scores + trimmed["positive_mask"].float() * 1_000.0
        selections = {
            "FULL_NO_PRA": list(range(len(spans))),
            "PRA_GENERIC": select_chunk_indices(
                generic_scores, spans, token_budget=budget
            ),
            "PRA_LEARNED": select_chunk_indices(
                learned_scores, spans, token_budget=budget
            ),
            "PRA_ORACLE_CONTROL": select_chunk_indices(
                oracle_scores, spans, token_budget=budget
            ),
        }
        for condition, selected in selections.items():
            key = (feature["example_id"], condition)
            if key in complete:
                continue
            started = time.perf_counter()
            selected_source = (
                tokenizer.decode(source_ids)
                if condition == "FULL_NO_PRA"
                else _selected_text(tokenizer, source_ids, spans, selected)
            )
            prompt = _prompt(tokenizer, selected_source, example["question"])
            selection_ms = (time.perf_counter() - started) * 1_000.0
            mx.reset_peak_memory()
            generated = _generate_timed(
                model,
                tokenizer,
                prompt,
                make_prompt_cache(model),
                args.max_new_tokens,
            )
            exact, f1 = _metrics(generated["output"], example["answer"])
            positives = trimmed["positive_mask"].nonzero(as_tuple=False).flatten()
            positive_set = set(map(int, positives.tolist()))
            recalled = len(positive_set.intersection(selected)) / max(
                len(positive_set), 1
            )
            latency_s = max(float(generated["completion_latency_ms"]) / 1_000.0, 1e-9)
            row = {
                "schema_version": "pra-qwen35-qualification-v1",
                "model_id": args.model,
                "model_revision": resolved.sha,
                "quantization": "MLX-4bit",
                "engine": f"mlx-lm {getattr(mlx_lm, '__version__', 'unknown')}",
                "dataset": feature["dataset"],
                "example_id": feature["example_id"],
                "condition": condition,
                "pra_enabled": condition != "FULL_NO_PRA",
                "adaptor": "combined-router-d128" if condition == "PRA_LEARNED" else None,
                "selected_fraction": 1.0 if condition == "FULL_NO_PRA" else args.selected_fraction,
                "candidate_source_tokens": len(source_ids),
                "selected_source_tokens": sum(
                    spans[index][1] - spans[index][0] for index in selected
                ),
                "visible_prompt_tokens": len(prompt),
                "selected_chunk_count": len(selected),
                "candidate_chunk_count": len(spans),
                "evidence_recall": recalled,
                "answer": example["answer"],
                "output": generated["output"],
                "exact_match": exact,
                "token_f1": f1,
                "selection_latency_ms": selection_ms,
                "ttft_ms": generated["ttft_ms"],
                "itl_ms": generated["itl_ms"],
                "completion_latency_ms": generated["completion_latency_ms"],
                "generated_tokens": generated["generated_tokens"],
                "output_tokens_per_second": generated["generated_tokens"] / latency_s,
                "peak_unified_memory_bytes": int(mx.get_peak_memory()),
                "evidence_tier": "MODEL_BACKED_NATURAL_QA_CONTROLLED",
            }
            rows.append(row)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"[{feature_index + 1}/{len(features)}] {feature['dataset']} "
                f"{condition} F1={f1:.3f} TTFT={generated['ttft_ms']:.1f}ms",
                flush=True,
            )
            mx.clear_cache()
    payload = {
        "schema_version": "pra-qwen35-qualification-v1",
        "model_id": args.model,
        "model_revision": resolved.sha,
        "feature_manifest": str(args.feature_dir / "feature_dataset_manifest.json"),
        "router": str(args.router),
        "selected_fraction": args.selected_fraction,
        "max_source_tokens": args.max_source_tokens,
        "max_new_tokens": args.max_new_tokens,
        "rows": rows,
        "aggregate": _aggregate(rows),
        "native_memory_status": "UNAVAILABLE_HYBRID_DELTA_STATE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--selected-fraction", type=float, default=0.20)
    parser.add_argument("--max-source-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.selected_fraction <= 1:
        parser.error("--selected-fraction must be in (0, 1].")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
