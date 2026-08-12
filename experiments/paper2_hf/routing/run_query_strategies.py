"""Evaluate zero-parameter information-need representations for frozen Qwen PRA."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_smoke import evidence_token_spans
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_representation import (
    _configure,
    _hotpot_examples,
    _qasper_examples,
    _ranking_row,
    _synchronize,
    _write_csv,
    _write_json,
)
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    PRAHFConfig,
    QUERY_EXPONENTIAL,
    QUERY_LAST,
    QUERY_LINEAR,
    QUERY_QUESTION_EXPONENTIAL,
    QUERY_QUESTION_MEAN,
    QUERY_UNIFORM,
    aggregate_query_states,
    half_life_to_decay,
    inject_pra,
    token_span_from_offsets,
)


@dataclass(frozen=True)
class QuerySpec:
    """One predeclared zero-parameter query aggregation condition."""

    name: str
    strategy: str
    window: int | None = None
    half_life: float | None = None

    @property
    def decay_lambda(self) -> float | None:
        return half_life_to_decay(self.half_life) if self.half_life else None


def _registry() -> dict[str, QuerySpec]:
    specs = [QuerySpec("last", QUERY_LAST)]
    specs.extend(QuerySpec(f"uniform_w{window}", QUERY_UNIFORM, window) for window in (2, 4, 8, 16, 32))
    specs.extend(QuerySpec(f"linear_w{window}", QUERY_LINEAR, window) for window in (8, 16, 32))
    specs.extend(
        QuerySpec(f"exp_w{window}_h{half_life}", QUERY_EXPONENTIAL, window, half_life)
        for window in (8, 16, 32)
        for half_life in (2.0, 4.0, 8.0, 16.0)
    )
    specs.append(QuerySpec("question_mean", QUERY_QUESTION_MEAN))
    specs.extend(
        QuerySpec(f"question_exp_h{half_life}", QUERY_QUESTION_EXPONENTIAL, half_life=half_life)
        for half_life in (2.0, 4.0, 8.0, 16.0)
    )
    return {spec.name: spec for spec in specs}


REGISTRY = _registry()
STAGE1 = (
    "last",
    "uniform_w4",
    "uniform_w8",
    "uniform_w16",
    "exp_w16_h2.0",
    "exp_w16_h4.0",
    "exp_w16_h8.0",
    "linear_w16",
    "question_mean",
    "question_exp_h4.0",
    "question_exp_h8.0",
)


def load_split_examples(cache_dir: Path, count: int, offset: int, seed: int) -> list[dict]:
    """Load a deterministic identity-disjoint slice from each validation source."""
    stop = int(offset) + int(count)
    hotpot = _hotpot_examples(cache_dir, stop, seed)[offset:stop]
    qasper = _qasper_examples(cache_dir / "qasper", stop, seed + 1)[offset:stop]
    return [*hotpot, *qasper]


def _prompt_with_question_span(tokenizer, question: str, max_tokens: int):
    question = question.strip()
    content = f"Answer briefly and directly.\nQuestion: {question}"
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        rendered = content
    marker = f"Question: {question}"
    marker_start = rendered.rfind(marker)
    if marker_start < 0:
        raise ValueError("Rendered prompt does not contain the exact question marker.")
    char_start = marker_start + len("Question: ")
    char_end = char_start + len(question)
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_tokens,
    )
    tokenizer.truncation_side = previous
    offsets = encoded.pop("offset_mapping")[0].tolist()
    return encoded, token_span_from_offsets(offsets, char_start, char_end)


@torch.no_grad()
def _capture_query_features(handle, tokenizer, example: dict, device: torch.device):
    encoded, question_span = _prompt_with_question_span(tokenizer, example["question"], 128)
    encoded = encoded.to(device)
    positions = torch.arange(encoded.input_ids.shape[1], device=device).unsqueeze(0)
    adapter = next(iter(handle.adapters.values()))
    handle.set_memory_enabled(False)
    adapter.begin_capture(positions)
    handle.model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        position_ids=positions,
        use_cache=False,
    )
    captured = adapter.consume_capture()
    native_query = adapter.pra_core.prepare_pra_query(captured.post_query)
    return captured.hidden_states, native_query, int(encoded.input_ids.shape[1]), question_span


def _pool_length(spec: QuerySpec, prompt_tokens: int, question_span: tuple[int, int]) -> int:
    if spec.strategy == QUERY_LAST:
        return 1
    if spec.strategy.startswith("question_"):
        return question_span[1] - question_span[0]
    return min(int(spec.window or 0), prompt_tokens)


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["query_strategy"], row["top_k"])].append(row)
    metrics = (
        "recall_at_3",
        "recall_at_8",
        "recall_at_16",
        "mrr",
        "best_evidence_rank",
        "target_coverage",
        "all_evidence_recall",
        "selected_fraction",
        "score_position_correlation",
        "mean_selected_normalized_position",
        "query_vector_norm",
        "query_cosine_to_last",
        "query_pool_tokens",
        "question_tokens",
        "prompt_tokens",
        "warm_routing_topk_seconds",
    )
    output = []
    for key, values in sorted(grouped.items()):
        record = {
            "dataset": key[0],
            "query_strategy": key[1],
            "top_k": key[2],
            "examples": len(values),
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            record[metric] = statistics.fmean(samples) if samples else None
        output.append(record)
    return output


def _pairwise_aggregates(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["left_strategy"], row["right_strategy"])].append(
            float(row["cosine_similarity"])
        )
    return [
        {
            "dataset": key[0],
            "left_strategy": key[1],
            "right_strategy": key[2],
            "examples": len(values),
            "mean_cosine_similarity": statistics.fmean(values),
        }
        for key, values in sorted(grouped.items())
    ]


def _plots(aggregates: list[dict], output_dir: Path, stem: str) -> None:
    rows = [row for row in aggregates if int(row["top_k"]) == 3]
    strategies = list(dict.fromkeys(row["query_strategy"] for row in rows))
    x = list(range(len(strategies)))
    width = 0.38
    lookup = {(row["dataset"], row["query_strategy"]): row for row in rows}
    for metric, ylabel in (("recall_at_3", "Recall@3"), ("mrr", "MRR")):
        figure, axis = plt.subplots(figsize=(max(8.0, len(strategies) * 0.8), 4.5))
        for offset, dataset, color in ((-width / 2, "hotpotqa", "#4472c4"), (width / 2, "qasper", "#ed7d31")):
            axis.bar(
                [value + offset for value in x],
                [lookup[(dataset, strategy)][metric] for strategy in strategies],
                width,
                label=dataset,
                color=color,
            )
        axis.set_xticks(x, strategies, rotation=30, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"{stem}_{metric}.{suffix}", dpi=180)
        plt.close(figure)


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
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(-1,),
            model_max_context_tokens=256,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="mean",
            gists_per_chunk=1,
            max_materialized_memory_tokens=128,
            top_k_references=1,
            top_k_chunks_per_reference=max(args.top_k),
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
    )
    examples = load_split_examples(
        args.cache_dir,
        args.examples_per_dataset,
        args.example_offset,
        args.seed,
    )
    specs = [REGISTRY[name] for name in args.strategies]
    checkpoint = args.output_dir / f"{args.stem}.checkpoint.json"
    rows, pairwise_rows = [], []
    if args.resume and checkpoint.exists():
        checkpoint_data = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows = checkpoint_data.get("rows", [])
        pairwise_rows = checkpoint_data.get("pairwise_rows", [])
    completed = {
        (row["dataset"], row["example_id"], row["query_strategy"], row["top_k"])
        for row in rows
    }
    pairwise_completed = {
        (row["dataset"], row["example_id"], row["left_strategy"], row["right_strategy"])
        for row in pairwise_rows
    }

    for example_index, example in enumerate(examples, start=1):
        _configure(handle, ATTENTION_INPUT_HIDDEN_STATE, 32, "mean", 1, "exact")
        source = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        source_tokens = int(source.shape[1])
        evidence_spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
        handle.add_reference(
            f"benchmark://{example['dataset']}/{example['id']}",
            source,
            text=example["source"],
        )
        hidden, native_query, prompt_tokens, question_span = _capture_query_features(
            handle, tokenizer, example, device
        )
        queries = {
            spec.name: aggregate_query_states(
                hidden,
                spec.strategy,
                window=spec.window,
                half_life=spec.half_life,
                token_spans=[question_span] if spec.strategy.startswith("question_") else None,
            )
            for spec in specs
        }
        last = aggregate_query_states(hidden, QUERY_LAST)
        adapter = next(iter(handle.adapters.values()))
        _synchronize(device)
        started = time.perf_counter()
        handle.cache.prepare_routing_index(adapter.layer_idx, last, force_rebuild=True)
        _synchronize(device)
        index_build_seconds = time.perf_counter() - started

        for left_index, left in enumerate(specs):
            for right in specs[left_index + 1 :]:
                key = (example["dataset"], example["id"], left.name, right.name)
                if key in pairwise_completed:
                    continue
                pairwise_rows.append(
                    {
                        "dataset": example["dataset"],
                        "example_id": example["id"],
                        "left_strategy": left.name,
                        "right_strategy": right.name,
                        "cosine_similarity": float(
                            F.cosine_similarity(queries[left.name], queries[right.name]).item()
                        ),
                    }
                )
                pairwise_completed.add(key)

        for spec in specs:
            query = queries[spec.name]
            for top_k in args.top_k:
                key = (example["dataset"], example["id"], spec.name, top_k)
                if key in completed:
                    continue
                row = _ranking_row(
                    handle=handle,
                    example=example,
                    source_tokens=source_tokens,
                    evidence_spans=evidence_spans,
                    query=query,
                    native_query=native_query,
                    direct_tokens=prompt_tokens,
                    representation=ATTENTION_INPUT_HIDDEN_STATE,
                    chunk_size=32,
                    gist_mode="mean",
                    gist_count=1,
                    center_policy="exact",
                    top_k=top_k,
                    index_build_seconds=index_build_seconds,
                    warm_repeats=args.warm_repeats,
                    seed=args.seed,
                )
                row.update(
                    {
                        "split": args.split,
                        "example_offset": args.example_offset,
                        "query_strategy": spec.name,
                        "query_strategy_kind": spec.strategy,
                        "query_window": spec.window,
                        "query_half_life": spec.half_life,
                        "query_decay_lambda": spec.decay_lambda,
                        "question_span_available": True,
                        "question_token_span": list(question_span),
                        "question_tokens": question_span[1] - question_span[0],
                        "prompt_tokens": prompt_tokens,
                        "query_pool_tokens": _pool_length(spec, prompt_tokens, question_span),
                        "query_vector_norm": float(query.float().norm(dim=-1).item()),
                        "query_cosine_to_last": float(
                            F.cosine_similarity(query.float(), last.float()).item()
                        ),
                    }
                )
                rows.append(row)
                completed.add(key)
        _write_json(checkpoint, {"rows": rows, "pairwise_rows": pairwise_rows})
        print(
            f"[{example_index}/{len(examples)}] {args.split} "
            f"{example['dataset']} {example['id']}",
            flush=True,
        )

    aggregates = _aggregate(rows)
    pairwise_aggregates = _pairwise_aggregates(pairwise_rows)
    artifact = {
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "protocol": "frozen Qwen query-representation evidence ranking; no answer generation",
        "split": args.split,
        "seed": args.seed,
        "example_offset": args.example_offset,
        "examples_per_dataset": args.examples_per_dataset,
        "memory_representation": ATTENTION_INPUT_HIDDEN_STATE,
        "routing_chunk_tokens": 32,
        "gist_mode": "mean",
        "gist_count": 1,
        "strategies": [asdict(spec) | {"decay_lambda": spec.decay_lambda} for spec in specs],
        "top_k": list(args.top_k),
        "rows": rows,
        "aggregates": aggregates,
        "pairwise_rows": pairwise_rows,
        "pairwise_aggregates": pairwise_aggregates,
        "max_native_operation_tokens": handle.max_native_operation_tokens,
        "native_limit_violations": handle.native_limit_violations,
    }
    _write_json(args.output_dir / f"{args.stem}.json", artifact)
    _write_csv(args.output_dir / f"{args.stem}.csv", rows)
    _write_csv(args.output_dir / f"{args.stem}_aggregate.csv", aggregates)
    _write_csv(args.output_dir / f"{args.stem}_pairwise.csv", pairwise_aggregates)
    _plots(aggregates, args.output_dir, args.stem)
    return artifact


def _csv_tuple(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--example-offset", type=int, default=0)
    parser.add_argument("--examples-per-dataset", type=int, default=8)
    parser.add_argument("--strategies", default=",".join(STAGE1))
    parser.add_argument("--top-k", default="3,8,16")
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stem", default="query_strategy_sweep")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "papers"
            / "shared"
            / "results"
            / "paper2_hf"
            / "routing"
            / "query_strategies"
        ),
    )
    args = parser.parse_args()
    args.strategies = _csv_tuple(args.strategies, str)
    unknown = sorted(set(args.strategies) - set(REGISTRY))
    if unknown:
        parser.error(f"Unknown query strategies: {', '.join(unknown)}")
    args.top_k = _csv_tuple(args.top_k, int)
    if args.example_offset < 0 or args.examples_per_dataset <= 0:
        parser.error("example offset must be non-negative and example count positive")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"split": result["split"], "rows": len(result["rows"])}, indent=2))
