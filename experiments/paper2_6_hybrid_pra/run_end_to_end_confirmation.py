"""Run fresh two-family Paper 2.6 discovery-to-generation confirmation.

The protocol trains channel selection on identity-disjoint validation rows and
streams held-out examples through the same bounded native-K/V materializer.
Every output row is checkpointed because reference encoding dominates runtime.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.qa.run_oracle_memory_use import (  # noqa: E402
    _answer_ids,
    _generate,
    _prompt,
    _teacher_forced,
)
from experiments.paper2_hf.qa.run_smoke import answer_metrics, evidence_token_spans  # noqa: E402
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples  # noqa: E402
from pra_hf import PRAConfig, PRAForCausalLM  # noqa: E402
from pra_hf.confidence_diagnostics import (  # noqa: E402
    choose_conservative_threshold,
    percentile_calibrate,
    selective_metrics,
    summarize_calibration,
)
from pra_hf.hybrid_discovery import HybridDiscoveryPolicy, TokenNativeIndex  # noqa: E402
from pra_hf.iterative import GistIndex, IterativeGistRouter, IterativeRoutingConfig  # noqa: E402
from pra_torch.memory import SelectedChunk  # noqa: E402


MODEL_SPECS = {
    "qwen3_0_6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "smollm2_135m": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
}
CHANNELS = {
    "semantic": "gist_only",
    "exact": "token_exact",
    "bm25": "bm25",
    "approximate": "token_approx",
    "hybrid": "iterative_hybrid",
    "ngram": "token_ngram",
    "edit": "token_edit",
    "embedding": "token_embedding",
}
CORE_CHANNELS = ("semantic", "exact", "bm25", "approximate", "hybrid")
CONTROL_CONDITIONS = (
    "disabled",
    "shuffled",
    "irrelevant",
    "empty",
    "bounded_direct_context",
)
CONTROLLER_SEEDS = (1, 7, 21, 42, 87)
LEGACY_COHORT_SEED = 20260811
LEGACY_EXAMPLES_PER_DATASET = 24


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _deduplicate(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    """Keep the last checkpoint for each stable experimental identity."""
    unique = {}
    for row in rows:
        unique[tuple(str(row.get(field, "")) for field in fields)] = row
    return list(unique.values())


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _bounded_source(tokenizer, example: dict, max_tokens: int) -> dict:
    """Keep all annotated evidence first, then a bounded amount of source context."""
    source = "\n".join(dict.fromkeys([*example["evidence"], example["source"]]))
    ids = tokenizer(source, add_special_tokens=False).input_ids[:max_tokens]
    bounded = tokenizer.decode(ids, skip_special_tokens=True)
    return {**example, "source": bounded}


def _fresh_examples(cache_dir, count: int, offset: int, seed: int) -> list[dict]:
    """Load examples after explicitly excluding the original Paper 2.6 cohort."""
    legacy = load_split_examples(
        cache_dir, LEGACY_EXAMPLES_PER_DATASET, 0, LEGACY_COHORT_SEED
    )
    excluded = {(row["dataset"], row["id"]) for row in legacy}
    requested = offset + count + LEGACY_EXAMPLES_PER_DATASET + 8
    candidates = load_split_examples(cache_dir, requested, 0, seed)
    output = []
    for dataset in ("hotpotqa", "qasper"):
        fresh = [
            row
            for row in candidates
            if row["dataset"] == dataset and (dataset, row["id"]) not in excluded
        ]
        selected = fresh[offset : offset + count]
        if len(selected) != count:
            raise RuntimeError(
                f"Only {len(selected)} fresh {dataset} identities available; expected {count}."
            )
        output.extend(selected)
    return output


def _capture_query(pra, prompt_ids, prompt_mask, routing_layer: int | None = None):
    device = pra.device
    positions = torch.arange(prompt_ids.shape[1], device=device).unsqueeze(0)
    layer = pra.routing_layer if routing_layer is None else int(routing_layer)
    adapter = pra._handle.adapters[layer]
    pra._handle.configure_memory_layers(set())
    adapter.begin_capture(positions)
    with torch.no_grad():
        pra.model(
            input_ids=prompt_ids.to(device),
            attention_mask=prompt_mask.to(device),
            position_ids=positions,
            use_cache=False,
        )
    captured = adapter.consume_capture()
    return adapter._routing_query_states(
        captured.hidden_states, captured.pre_query, captured.post_query
    )[0]


def _selected_metrics(selected, evidence_spans):
    positives = [
        any(_overlaps((hit.logical_start, hit.logical_end), span) for span in evidence_spans)
        for hit in selected
    ]
    covered = sum(
        any(_overlaps((hit.logical_start, hit.logical_end), span) for hit in selected)
        for span in evidence_spans
    )
    first = next((index for index, value in enumerate(positives, 1) if value), None)
    return {
        "evidence_recall": covered / max(len(evidence_spans), 1),
        "precision": sum(positives) / max(len(selected), 1),
        "mrr": 1.0 / first if first else 0.0,
        "complete_recovery": int(covered == len(evidence_spans) and bool(evidence_spans)),
    }


def _route_channels(
    pra,
    tokenizer,
    example,
    prompt_ids,
    prompt_mask,
    evidence_spans,
    *,
    routing_layer: int | None = None,
    chunk_budget: int = 4,
):
    layer = pra.routing_layer if routing_layer is None else int(routing_layer)
    query = _capture_query(pra, prompt_ids, prompt_mask, layer)
    index = GistIndex.from_entries(
        pra._handle.cache.all_entries(),
        layer,
        device=query.device,
        dtype=query.dtype,
    )
    embedding_weight = pra.model.get_input_embeddings().weight
    token_index = TokenNativeIndex.from_gist_index(
        index,
        tokenizer,
        automatic_aliases=True,
        ngram_sizes=(2, 3),
        token_embedding_weight=embedding_weight,
    )
    root_ids = prompt_ids[0][prompt_mask[0].bool()].detach().cpu().tolist()
    iterative = IterativeGistRouter(index)
    config = IterativeRoutingConfig(
        depth=2,
        branch_top_k=2,
        beam_size=2,
        max_unique_chunks=chunk_budget,
        root_anchor_alpha=0.0,
        path_score_mode="last",
    )
    outputs = {}
    rows = []
    for channel, mode in CHANNELS.items():
        policy = HybridDiscoveryPolicy(
            mode=mode,
            semantic_weight=0.6,
            token_weight=0.4,
            later_semantic_weight=0.1,
            later_token_weight=0.9,
            indexed=True,
            candidate_pool_size=64,
            enable_extended_channels=True,
            automatic_aliases=True,
            ngram_sizes=(2, 3),
            approximate_max_distance=1,
        )
        result = iterative.route(
            query,
            config,
            token_index=token_index,
            root_token_ids=root_ids,
            tokenizer=tokenizer,
            discovery_policy=policy,
            token_embedding_weight=embedding_weight,
        )
        selected = iterative.selected_chunks(result)
        scores = sorted(result.direct_scores, reverse=True)
        metrics = _selected_metrics(selected, evidence_spans)
        top_hit = selected[0] if selected else None
        top1_correct = int(
            top_hit is not None
            and any(
                _overlaps((top_hit.logical_start, top_hit.logical_end), span)
                for span in evidence_spans
            )
        )
        outputs[channel] = selected
        rows.append(
            {
                "channel": channel,
                **metrics,
                "top_score": scores[0] if scores else -1.0,
                "score_gap": scores[0] - scores[1] if len(scores) > 1 else 0.0,
                "score_mean": statistics.fmean(scores) if scores else -1.0,
                "top1_correct": top1_correct,
                "referential_confidence": max(
                    0.0, min(1.0, ((scores[0] if scores else -1.0) + 1.0) / 2.0)
                ),
                "selected_chunk_ids": "|".join(hit.chunk_id for hit in selected),
                "selected_chunks": len(selected),
                "candidate_chunks": len(index.records),
                "source_tokens": sum(chunk.token_count for _, chunk in index.records),
                "index_bytes": token_index.memory_bytes(),
                "scored_candidates": sum(
                    int(node.discovery_channels.get("indexed_candidates", 0))
                    for node in result.graph.nodes
                )
                or int(result.graph.costs.get("token_index_scored_candidates", 0)),
                "token_index_queries": int(
                    result.graph.costs.get("token_index_queries", 0)
                ),
            }
        )
    return outputs, rows


def _feature_vector(rows: list[dict]) -> list[float]:
    by_channel = {row["channel"]: row for row in rows}
    selected_sets = [
        set(by_channel[channel]["selected_chunk_ids"].split("|"))
        for channel in CORE_CHANNELS
    ]
    disagreement = len(set.union(*selected_sets)) / max(
        sum(len(values) for values in selected_sets), 1
    )
    features = []
    for channel in CORE_CHANNELS:
        row = by_channel[channel]
        features.extend(
            [float(row["top_score"]), float(row["score_gap"]), float(row["score_mean"])]
        )
    features.extend(
        [
            math.log1p(float(rows[0]["candidate_chunks"])),
            math.log1p(float(rows[0]["source_tokens"])),
            disagreement,
        ]
    )
    return features


def _label(rows: list[dict]) -> int:
    by_channel = {row["channel"]: row for row in rows}
    return max(
        range(len(CORE_CHANNELS)),
        key=lambda index: (
            float(by_channel[CORE_CHANNELS[index]]["evidence_recall"]),
            float(by_channel[CORE_CHANNELS[index]]["precision"]),
            float(by_channel[CORE_CHANNELS[index]]["mrr"]),
            -index,
        ),
    )


def _train_controllers(validation_groups: list[list[dict]]):
    x = torch.tensor([_feature_vector(rows) for rows in validation_groups], dtype=torch.float32)
    y = torch.tensor([_label(rows) for rows in validation_groups], dtype=torch.long)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    normalized = (x - mean) / std
    controllers = []
    for seed in CONTROLLER_SEEDS:
        torch.manual_seed(seed)
        model = torch.nn.Linear(normalized.shape[1], len(CORE_CHANNELS))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(normalized), y)
            loss.backward()
            optimizer.step()
        controllers.append((seed, model.eval()))
    return mean, std, controllers


def _controller_predictions(features, mean, std, controllers):
    x = (torch.tensor(features, dtype=torch.float32) - mean) / std
    predictions = [int(model(x).argmax().item()) for _, model in controllers]
    counts = Counter(predictions)
    ensemble = min(
        (index for index, count in counts.items() if count == max(counts.values())),
        default=0,
    )
    return [CORE_CHANNELS[index] for index in predictions], CORE_CHANNELS[ensemble]


def _make_selected(entry, routing_layer, chunks, source: str) -> list[SelectedChunk]:
    return [
        SelectedChunk(
            entry=entry,
            chunk=chunk,
            reference_score=1.0,
            chunk_score=1.0 - rank * 1e-6,
            layer_id=routing_layer,
            reference_rank=1,
            rank_within_reference=rank,
            metadata={"selection_source": source},
        )
        for rank, chunk in enumerate(chunks, 1)
    ]


def _control_selections(entry, routing_layer, base, evidence_spans, seed):
    chunks = list(entry.layer_memory[routing_layer].chunks)
    selected_ids = {hit.chunk_id for hit in base}
    nonselected = [chunk for chunk in chunks if chunk.chunk_id not in selected_ids]
    rng = random.Random(seed)
    shuffled = list(nonselected)
    rng.shuffle(shuffled)
    irrelevant = [
        chunk
        for chunk in nonselected
        if not any(_overlaps((chunk.logical_start, chunk.logical_end), span) for span in evidence_spans)
    ]
    count = len(base)
    return {
        "shuffled": _make_selected(entry, routing_layer, shuffled[:count], "shuffled_control"),
        "irrelevant": _make_selected(entry, routing_layer, irrelevant[:count], "irrelevant_control"),
        "empty": [],
    }


def _oracle(entry, routing_layer, evidence_spans, budget=4):
    chunks = list(entry.layer_memory[routing_layer].chunks)
    remaining = set(range(len(evidence_spans)))
    selected = []
    while remaining and len(selected) < budget:
        selected_ids = {chunk.chunk_id for chunk in selected}
        candidate = max(
            (chunk for chunk in chunks if chunk.chunk_id not in selected_ids),
            key=lambda chunk: (
                sum(
                    _overlaps((chunk.logical_start, chunk.logical_end), evidence_spans[index])
                    for index in remaining
                ),
                -chunk.logical_start,
            ),
            default=None,
        )
        if candidate is None:
            break
        covered = {
            index
            for index in remaining
            if _overlaps(
                (candidate.logical_start, candidate.logical_end), evidence_spans[index]
            )
        }
        if not covered:
            break
        selected.append(candidate)
        remaining -= covered
    selected_ids = {chunk.chunk_id for chunk in selected}
    selected.extend(chunk for chunk in chunks if chunk.chunk_id not in selected_ids)
    return _make_selected(
        entry, routing_layer, selected[:budget], "matched_budget_evidence_oracle"
    )


def _physical_metrics(handle, layers, source_tokens):
    diagnostics = handle.diagnostics_by_layer()
    tokens = max(
        (
            int(diagnostics.get(layer, {}).get("memory_tokens_materialized", 0))
            for layer in layers
        ),
        default=0,
    )
    states = tokens * len(layers)
    config = handle.model.config
    heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    width = int(config.hidden_size // config.num_attention_heads)
    byte_count = states * heads * width * 2 * 2
    return {
        "materialized_unique_tokens": tokens,
        "native_kv_token_states": states,
        "native_kv_bytes": byte_count,
        "active_kv_fraction": tokens / max(source_tokens, 1),
    }


def _evaluate_condition(
    pra,
    tokenizer,
    example,
    evidence_spans,
    condition,
    selected,
    layers,
    args,
):
    context = example["source"] if condition == "bounded_direct_context" else None
    prompt_ids, prompt_mask, context_tokens = _prompt(
        tokenizer,
        example["question"],
        context=context,
        max_tokens=args.full_context_tokens if context else args.prompt_tokens,
    )
    if condition in {"disabled", "empty", "bounded_direct_context"}:
        pra._handle.configure_memory_layers(set())
        active_layers = ()
    else:
        fixed = pra._handle.map_chunk_identities_to_layers([selected], layers)
        pra._handle.configure_memory_layers(set(layers), fixed_selections=fixed)
        active_layers = layers
    scored, _ = _teacher_forced(
        pra._handle,
        tokenizer,
        prompt_ids,
        prompt_mask,
        _answer_ids(tokenizer, example["answer"]),
        context_tokens,
        evidence_spans,
        pra.device,
    )
    prediction, generation_seconds = _generate(
        pra._handle,
        tokenizer,
        prompt_ids,
        prompt_mask,
        pra.device,
        args.max_new_tokens,
    )
    scored.pop("attention_by_layer", None)
    return {
        **scored,
        "generated_answer": prediction,
        "generation_seconds": generation_seconds,
        **answer_metrics(prediction, example["answer"]),
        **_physical_metrics(pra._handle, active_layers, len(tokenizer(example["source"]).input_ids)),
    }


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["dataset"], row["condition"])].append(row)
    metrics = (
        "evidence_recall",
        "precision",
        "mrr",
        "em",
        "f1",
        "answer_contained",
        "gold_mean_token_logprob",
        "gold_logprob_delta_vs_disabled",
        "materialized_unique_tokens",
        "active_kv_fraction",
        "teacher_forced_seconds",
        "generation_seconds",
    )
    return [
        {
            "model": key[0],
            "dataset": key[1],
            "condition": key[2],
            "examples": len(values),
            **{
                metric: statistics.fmean(float(row[metric]) for row in values)
                for metric in metrics
            },
        }
        for key, values in sorted(grouped.items())
    ]


def _calibration(retrieval_rows: list[dict]) -> list[dict]:
    """Summarize naturally occurring wrong top-reference choices on held-out rows."""
    output = []
    for model in sorted({row["model"] for row in retrieval_rows if row.get("split") == "test"}):
        for dataset in sorted(
            {
                row["dataset"]
                for row in retrieval_rows
                if row.get("split") == "test" and row["model"] == model
            }
        ):
            for channel in sorted(CHANNELS):
                rows = [
                    row
                    for row in retrieval_rows
                    if row.get("split") == "test"
                    and row["model"] == model
                    and row["dataset"] == dataset
                    and row["channel"] == channel
                ]
                if not rows:
                    continue
                validation = [
                    row
                    for row in retrieval_rows
                    if row.get("split") == "validation"
                    and row["model"] == model
                    and row["dataset"] == dataset
                    and row["channel"] == channel
                ]
                labels = [int(float(row["top1_correct"])) for row in rows]
                validation_scores = [
                    float(row["referential_confidence"]) for row in validation
                ]
                confidence = percentile_calibrate(
                    validation_scores,
                    [float(row["referential_confidence"]) for row in rows],
                )
                validation_probability = percentile_calibrate(
                    validation_scores, validation_scores
                )
                threshold = choose_conservative_threshold(
                    [int(float(row["top1_correct"])) for row in validation],
                    validation_probability,
                    minimum_precision=0.8,
                )
                summary = summarize_calibration(labels, confidence, bins=8)
                selective = selective_metrics(labels, confidence, threshold)
                output.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "channel": channel,
                        "examples": len(rows),
                        "top1_accuracy": statistics.fmean(labels),
                        "calibration": "validation_cdf",
                        **{f"selective_{key}": value for key, value in selective.items()},
                        **summary.__dict__,
                    }
                )
    return output


def _controller_report(completed_rows: list[dict], retrieval_rows: list[dict]) -> list[dict]:
    """Reconstruct five-seed decisions from durable per-example artifacts."""
    retrieval = {
        (row["model"], row["dataset"], row["example_id"], row["channel"]): row
        for row in retrieval_rows
        if row.get("split") == "test"
    }
    output = []
    for row in completed_rows:
        if row["condition"] != "adaptive":
            continue
        predictions = str(row["controller_seed_channels"]).split("|")
        for seed, channel in zip(CONTROLLER_SEEDS, predictions):
            selected = retrieval[(row["model"], row["dataset"], row["example_id"], channel)]
            output.append(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "example_id": row["example_id"],
                    "seed": seed,
                    "selected_channel": channel,
                    "evidence_recall": selected["evidence_recall"],
                    "precision": selected["precision"],
                }
            )
    return output


def _bootstrap_ci(values: list[float], seed: int, replicates: int = 2000):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(replicates)
    )
    return draws[int(0.025 * replicates)], draws[int(0.975 * replicates)]


def _paired_effects(rows: list[dict]) -> list[dict]:
    """Estimate paired downstream effects against complete causal baselines."""
    by_example = defaultdict(dict)
    for row in rows:
        by_example[(row["model"], row["dataset"], row["example_id"])][
            row["condition"]
        ] = row
    output = []
    baselines = ("disabled", "shuffled", "irrelevant", "empty")
    treatments = (*CORE_CHANNELS, "adaptive", "oracle")
    metrics = ("gold_mean_token_logprob", "em", "f1")
    for model in sorted({key[0] for key in by_example}):
        for dataset in sorted({key[1] for key in by_example if key[0] == model}):
            groups = [
                conditions
                for key, conditions in by_example.items()
                if key[0] == model and key[1] == dataset
            ]
            for treatment in treatments:
                for baseline in baselines:
                    paired = [
                        conditions
                        for conditions in groups
                        if treatment in conditions and baseline in conditions
                    ]
                    for metric in metrics:
                        deltas = [
                            float(conditions[treatment][metric])
                            - float(conditions[baseline][metric])
                            for conditions in paired
                        ]
                        low, high = _bootstrap_ci(
                            deltas,
                            seed=20260825
                            + sum(ord(value) for value in f"{model}{dataset}{treatment}{baseline}{metric}"),
                        )
                        output.append(
                            {
                                "model": model,
                                "dataset": dataset,
                                "treatment": treatment,
                                "baseline": baseline,
                                "metric": metric,
                                "pairs": len(deltas),
                                "mean_delta": statistics.fmean(deltas) if deltas else float("nan"),
                                "bootstrap_ci95_low": low,
                                "bootstrap_ci95_high": high,
                            }
                        )
    return output


def _plot(aggregates, output):
    core = [*CORE_CHANNELS, "adaptive", "oracle"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model in sorted({row["model"] for row in aggregates}):
        values = [row for row in aggregates if row["model"] == model]
        logp = [
            statistics.fmean(
                row["gold_logprob_delta_vs_disabled"]
                for row in values
                if row["condition"] == condition
            )
            for condition in core
        ]
        f1 = [
            statistics.fmean(row["f1"] for row in values if row["condition"] == condition)
            for condition in core
        ]
        axes[0].plot(core, logp, marker="o", label=model)
        axes[1].plot(core, f1, marker="o", label=model)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Gold log-probability delta vs disabled")
    axes[1].set_ylabel("Generated token F1")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"end_to_end_confirmation.{suffix}", dpi=190)
    plt.close(figure)


def _load_model(name, args):
    model_id, revision = MODEL_SPECS[name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=args.local_files_only
    )
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    if args.device == "cuda" and torch.cuda.get_device_capability()[0] < 7:
        model.generation_config.disable_compile = True
    layer_types = tuple(getattr(model.config, "layer_types", ()) or ())
    eligible = (
        tuple(index for index, value in enumerate(layer_types) if value == "full_attention")
        if layer_types
        else tuple(range(model.config.num_hidden_layers))
    )
    layers = eligible[-min(args.consumption_layers, len(eligible)) :]
    pra = PRAForCausalLM.from_model(
        model,
        tokenizer,
        pra_config=PRAConfig(
            routing_layer=eligible[-1],
            consumption_layers=layers,
            routing_mode="hybrid_iterative",
            chunk_tokens=args.chunk_tokens,
            selected_fraction=None,
            max_direct_context=args.prompt_tokens,
            native_operation_limit=args.native_limit,
            max_materialized_tokens=args.materialized_tokens,
            context_safety_reserve_tokens=0,
            encoding_block_tokens=args.encoding_tokens,
            reference_device="cpu",
            pin_reference_memory=args.device == "cuda",
            non_blocking_transfer=args.device == "cuda",
        ),
    )
    return model_id, revision, tokenizer, pra, layers


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "end_to_end_checkpoint.jsonl"
    completed_rows = _read_jsonl(checkpoint)
    completed = {
        (row["model"], row["dataset"], row["example_id"], row["condition"])
        for row in completed_rows
    }
    all_retrieval = _read_csv(args.output / "retrieval_rows.csv")
    controller_rows = []
    model_manifest = {}
    for model_name in args.models:
        model_id, revision, tokenizer, pra, layers = _load_model(model_name, args)
        model_manifest[model_name] = {
            "model_id": model_id,
            "revision": revision,
            "routing_layer": pra.routing_layer,
            "consumption_layers": list(layers),
            "parameter_count": sum(parameter.numel() for parameter in pra.model.parameters()),
        }
        validation = _fresh_examples(
            args.cache_dir,
            args.validation_examples_per_dataset,
            args.validation_offset,
            args.seed,
        )
        validation_groups = []
        for index, raw in enumerate(validation, 1):
            example = _bounded_source(tokenizer, raw, args.source_tokens)
            pra.clear_references()
            uri = f"benchmark://{raw['dataset']}/{raw['id']}"
            pra.add_reference(example["source"], uri=uri)
            spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
            prompt_ids, prompt_mask, _ = _prompt(
                tokenizer, example["question"], max_tokens=args.prompt_tokens
            )
            _, rows = _route_channels(
                pra, tokenizer, example, prompt_ids, prompt_mask, spans
            )
            for row in rows:
                row.update(
                    model=model_name,
                    split="validation",
                    dataset=raw["dataset"],
                    example_id=raw["id"],
                )
            validation_groups.append(rows)
            all_retrieval.extend(rows)
            print(f"[{model_name} validation {index}/{len(validation)}] {raw['dataset']} {raw['id']}", flush=True)
        mean, std, controllers = _train_controllers(validation_groups)
        tests = _fresh_examples(
            args.cache_dir,
            args.test_examples_per_dataset,
            args.test_offset,
            args.seed,
        )
        for index, raw in enumerate(tests, 1):
            example = _bounded_source(tokenizer, raw, args.source_tokens)
            pending = [
                condition
                for condition in (*CORE_CHANNELS, "adaptive", "oracle", *CONTROL_CONDITIONS)
                if (model_name, raw["dataset"], raw["id"], condition) not in completed
            ]
            if not pending:
                continue
            pra.clear_references()
            uri = f"benchmark://{raw['dataset']}/{raw['id']}"
            pra.add_reference(example["source"], uri=uri)
            entry = pra._handle.cache.get(uri)
            spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
            prompt_ids, prompt_mask, _ = _prompt(
                tokenizer, example["question"], max_tokens=args.prompt_tokens
            )
            routed, route_rows = _route_channels(
                pra, tokenizer, example, prompt_ids, prompt_mask, spans
            )
            predictions, adaptive = _controller_predictions(
                _feature_vector(route_rows), mean, std, controllers
            )
            oracle = _oracle(entry, pra.routing_layer, spans)
            controls = _control_selections(
                entry,
                pra.routing_layer,
                routed[adaptive],
                spans,
                args.seed + index,
            )
            selections = {
                **{channel: routed[channel] for channel in CORE_CHANNELS},
                "adaptive": routed[adaptive],
                "oracle": oracle,
                "disabled": [],
                "bounded_direct_context": [],
                **controls,
            }
            metrics_by_channel = {row["channel"]: row for row in route_rows}
            for row in route_rows:
                row.update(
                    model=model_name,
                    split="test",
                    dataset=raw["dataset"],
                    example_id=raw["id"],
                )
            all_retrieval.extend(route_rows)
            for seed, prediction in zip(CONTROLLER_SEEDS, predictions):
                selected = metrics_by_channel[prediction]
                controller_rows.append(
                    {
                        "model": model_name,
                        "dataset": raw["dataset"],
                        "example_id": raw["id"],
                        "seed": seed,
                        "selected_channel": prediction,
                        "evidence_recall": selected["evidence_recall"],
                        "precision": selected["precision"],
                    }
                )
            baseline_logp = None
            local_rows = []
            for condition in pending:
                quality = _evaluate_condition(
                    pra,
                    tokenizer,
                    example,
                    spans,
                    condition,
                    selections[condition],
                    layers,
                    args,
                )
                retrieval = (
                    metrics_by_channel[condition]
                    if condition in CORE_CHANNELS
                    else metrics_by_channel[adaptive]
                    if condition == "adaptive"
                    else _selected_metrics(selections[condition], spans)
                )
                row = {
                    "model": model_name,
                    "dataset": raw["dataset"],
                    "example_id": raw["id"],
                    "condition": condition,
                    "adaptive_channel": adaptive,
                    "controller_seed_channels": "|".join(predictions),
                    "reference_answer": example["answer"],
                    "selected_chunk_ids": "|".join(hit.chunk_id for hit in selections[condition]),
                    **{key: retrieval[key] for key in ("evidence_recall", "precision", "mrr")},
                    **quality,
                }
                if condition == "disabled":
                    baseline_logp = row["gold_mean_token_logprob"]
                local_rows.append(row)
            if baseline_logp is None:
                prior = next(
                    row
                    for row in completed_rows
                    if row["model"] == model_name
                    and row["dataset"] == raw["dataset"]
                    and row["example_id"] == raw["id"]
                    and row["condition"] == "disabled"
                )
                baseline_logp = prior["gold_mean_token_logprob"]
            for row in local_rows:
                row["gold_logprob_delta_vs_disabled"] = row["gold_mean_token_logprob"] - baseline_logp
                _append_jsonl(checkpoint, row)
                completed_rows.append(row)
                completed.add((model_name, raw["dataset"], raw["id"], row["condition"]))
            print(f"[{model_name} test {index}/{len(tests)}] {raw['dataset']} {raw['id']} adaptive={adaptive}", flush=True)
            pra.clear_references()
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        del pra
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    all_retrieval = _deduplicate(
        all_retrieval, ("model", "split", "dataset", "example_id", "channel")
    )
    completed_rows = _deduplicate(
        completed_rows, ("model", "dataset", "example_id", "condition")
    )
    _write_csv(args.output / "retrieval_rows.csv", all_retrieval)
    _write_csv(args.output / "end_to_end_rows.csv", completed_rows)
    controller_rows = _controller_report(completed_rows, all_retrieval)
    _write_csv(args.output / "controller_five_seed_rows.csv", controller_rows)
    aggregates = _aggregate(completed_rows)
    _write_csv(args.output / "end_to_end_summary.csv", aggregates)
    calibration = _calibration(all_retrieval)
    _write_csv(args.output / "natural_wrong_reference_calibration.csv", calibration)
    paired_effects = _paired_effects(completed_rows)
    _write_csv(args.output / "paired_causal_effects.csv", paired_effects)
    _plot(aggregates, args.output)
    manifest = {
        "schema_version": "1.0",
        "device": args.device,
        "cuda_device": (
            torch.cuda.get_device_name() if args.device == "cuda" else None
        ),
        "models": model_manifest,
        "validation_examples_per_dataset": args.validation_examples_per_dataset,
        "test_examples_per_dataset": args.test_examples_per_dataset,
        "validation_offset": args.validation_offset,
        "test_offset": args.test_offset,
        "controller_seeds": list(CONTROLLER_SEEDS),
        "legacy_cohort_exclusion": {
            "seed": LEGACY_COHORT_SEED,
            "examples_per_dataset": LEGACY_EXAMPLES_PER_DATASET,
        },
        "channels": list(CHANNELS),
        "generated_conditions": [*CORE_CHANNELS, "adaptive", "oracle", *CONTROL_CONDITIONS],
        "matched_chunk_budget": 4,
        "chunk_tokens": args.chunk_tokens,
        "source_token_limit": args.source_tokens,
        "bounded_direct_context_limit": args.full_context_tokens,
        "prompt_token_limit": args.prompt_tokens,
        "encoding_block_tokens": args.encoding_tokens,
        "native_operation_limit": args.native_limit,
        "materialized_token_limit": args.materialized_tokens,
        "consumption_layer_count": args.consumption_layers,
        "max_new_tokens": args.max_new_tokens,
        "rows": len(completed_rows),
        "aggregates": aggregates,
        "natural_wrong_reference_calibration": calibration,
        "paired_causal_effects": paired_effects,
    }
    (args.output / "end_to_end_findings.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=tuple(MODEL_SPECS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--validation-examples-per-dataset", type=int, default=20)
    parser.add_argument("--test-examples-per-dataset", type=int, default=50)
    parser.add_argument("--validation-offset", type=int, default=0)
    parser.add_argument("--test-offset", type=int, default=20)
    parser.add_argument("--source-tokens", type=int, default=1536)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--full-context-tokens", type=int, default=1664)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--encoding-tokens", type=int, default=128)
    parser.add_argument("--native-limit", type=int, default=2048)
    parser.add_argument("--materialized-tokens", type=int, default=128)
    parser.add_argument("--consumption-layers", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT.parent / "pdattention/data/.hf_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/end_to_end_confirmation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
