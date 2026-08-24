"""Confirm frozen Paper 2.8 routers on a fresh identity-disjoint cohort.

The runner never trains or selects a configuration. It projects fresh query
states with the frozen Qwen layer-27 projection, loads the five committed
direct-router checkpoints, and emits exact chunk identities for downstream
native-K/V replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_gated_study import (
    MODEL_ID,
    MODEL_REVISION,
    SEEDS,
    _bootstrap,
    _case_tensors,
    _project_native_queries,
    _score_compact,
    _sha256,
    _write_csv,
)
from experiments.paper2_8_qk_compression.run_query_conditioned_study import (
    _query_feature,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.hybrid_discovery import HybridDiscoveryPolicy, TokenNativeIndex
from pra_hf.qk_compression import (
    QueryConditionedLandmarkSelector,
    kmeans_centroids,
    low_rank_response_scores,
    masked_mean_keys,
    routing_metrics,
    stable_topk_indices,
)


RESULT_ROOT = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
CHECKPOINT_ROOT = RESULT_ROOT / "low_rank_frontier/checkpoints"
CONFIRMATION_OFFSET = 24
CONFIRMATION_PER_DATASET = 64
ROUTING_BUDGET = 4


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_selector(path: Path, device: torch.device) -> tuple[QueryConditionedLandmarkSelector, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    selector = QueryConditionedLandmarkSelector(
        2048,
        feature_width=1024,
        rank=int(checkpoint["rank"]),
        use_salience=False,
        use_interaction=True,
    ).to(device)
    selector.load_state_dict(checkpoint["state_dict"])
    return selector.eval(), checkpoint


def _score_lowrank(
    selector: QueryConditionedLandmarkSelector,
    checkpoint: dict,
    case: dict,
    *,
    static_centroids: bool,
    device: torch.device,
) -> torch.Tensor:
    keys = case["keys"].to(device).float().flatten(2)
    mask = case["mask"].to(device)
    projected = selector.feature_projection(keys / float(checkpoint["native_key_rms_scale"]))
    if static_centroids:
        projected, mask = kmeans_centroids(projected, mask, 8)
    projected_query = selector.query_projection(case["query_feature"].to(device))
    return low_rank_response_scores(
        projected_query,
        projected,
        mask,
        function="top_r_mean",
        top_r=4,
    )[0].cpu()


def _semantic_unit(scores: torch.Tensor) -> torch.Tensor:
    low, high = scores.min(), scores.max()
    if float(high - low) <= 1e-12:
        return torch.zeros_like(scores)
    return 2.0 * (scores - low) / (high - low) - 1.0


def _token_index(tokenizer, example: dict, feature: dict) -> TokenNativeIndex:
    entry = SimpleNamespace(
        uri=f"benchmark://{feature['dataset']}/{feature['example_id']}",
        text=example["source"],
        metadata={},
    )
    records = []
    for index, (start, end) in enumerate(feature["local_spans"]):
        records.append(
            (
                entry,
                SimpleNamespace(
                    chunk_id=f"local-{index}",
                    token_start=int(start),
                    token_end=int(end),
                ),
            )
        )
    return TokenNativeIndex.from_gist_index(
        SimpleNamespace(records=records, layer_id=27), tokenizer
    )


def _lexical_scores(
    tokenizer,
    example: dict,
    feature: dict,
    semantic_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    index = _token_index(tokenizer, example, feature)
    query_ids = tokenizer(example["question"], add_special_tokens=False).input_ids
    output = {}
    for name, mode in (
        ("exact", "token_exact"),
        ("bm25", "bm25"),
        ("hybrid", "iterative_hybrid"),
    ):
        candidates = index.score(
            query_ids,
            _semantic_unit(semantic_scores),
            tokenizer,
            HybridDiscoveryPolicy(mode=mode),
            hop=1,
            parent_id="query",
        )
        output[name] = torch.tensor(
            [candidate.selected_score for candidate in candidates], dtype=torch.float32
        )
    return output


def _stable_random_indices(example_id: str, candidates: int, budget: int) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(example_id.encode("utf-8")).digest()[:8], "big")
    values = list(range(candidates))
    random.Random(seed).shuffle(values)
    return sorted(values[: min(budget, candidates)])


def _oracle_indices(
    positives: torch.Tensor, teacher: torch.Tensor, budget: int
) -> list[int]:
    positive = torch.nonzero(positives, as_tuple=False).flatten().tolist()
    positive.sort(key=lambda index: (-float(teacher[index]), index))
    selected = positive[:budget]
    if len(selected) < budget:
        remaining = [
            index
            for index in stable_topk_indices(teacher, len(teacher)).tolist()
            if index not in selected
        ]
        selected.extend(remaining[: budget - len(selected)])
    return sorted(selected)


def _row(
    feature: dict,
    *,
    condition: str,
    seed: int,
    scores: torch.Tensor,
    selected: list[int] | None = None,
    index_bytes_per_chunk: int | None = None,
) -> dict:
    selected = (
        stable_topk_indices(scores, ROUTING_BUDGET).tolist()
        if selected is None
        else selected
    )
    metrics = routing_metrics(selected, feature["local_positive_mask"], budget=ROUTING_BUDGET)
    materialized = sum(int(feature["local_token_mask"][index].sum()) for index in selected)
    return {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "condition": condition,
        "seed": seed,
        "candidate_chunks": len(scores),
        "selected_chunks": " ".join(map(str, selected)),
        "requested_chunks": len(selected),
        "materialized_native_kv_tokens": materialized,
        "active_memory_fraction": materialized / max(int(feature["source_tokens"]), 1),
        "index_bytes_per_chunk": index_bytes_per_chunk,
        **metrics,
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    output = []
    for (dataset, condition), group in sorted(grouped.items()):
        identities = len({row["example_id"] for row in group})
        seeds = len({row["seed"] for row in group if int(row["seed"]) >= 0})
        output.append(
            {
                "dataset": dataset,
                "condition": condition,
                "rows": len(group),
                "identities": identities,
                "seeds": seeds,
                **{
                    metric: statistics.fmean(float(row[metric]) for row in group)
                    for metric in (
                        "evidence_recall",
                        "evidence_precision",
                        "any_evidence",
                        "chain_completion",
                        "exact_identity",
                        "mrr",
                        "materialized_native_kv_tokens",
                        "active_memory_fraction",
                    )
                },
                "index_bytes_per_chunk": next(
                    (row["index_bytes_per_chunk"] for row in group if row["index_bytes_per_chunk"] is not None),
                    None,
                ),
            }
        )
    return output


def _paired(rows: list[dict], seed: int) -> list[dict]:
    ensemble = {
        (row["dataset"], row["condition"], row["example_id"]): float(row["evidence_recall"])
        for row in rows
        if int(row["seed"]) == -1
    }
    output = []
    comparisons = (
        ("lowrank_r16_ensemble", "native_mean"),
        ("lowrank_r16_ensemble", "exact"),
        ("lowrank_r16_ensemble", "bm25"),
        ("lowrank_r16_ensemble", "hybrid"),
        ("lowrank_r8_kmeans_ensemble", "native_mean"),
        ("lowrank_r8_kmeans_ensemble", "exact"),
    )
    for dataset in ("hotpotqa", "qasper"):
        for method, baseline in comparisons:
            differences = []
            for example_id in {
                key[2] for key in ensemble if key[0] == dataset and key[1] == method
            }:
                left = ensemble.get((dataset, method, example_id))
                right = ensemble.get((dataset, baseline, example_id))
                if left is not None and right is not None:
                    differences.append(left - right)
            low, high = _bootstrap(
                differences, seed + sum(map(ord, dataset + method + baseline))
            )
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "baseline": baseline,
                    "pairs": len(differences),
                    "mean_delta": statistics.fmean(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": sum(value > 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                    "losses": sum(value < 0 for value in differences),
                }
            )
    return output


def _plot(summary: list[dict], output_dir: Path) -> None:
    conditions = (
        "native_mean",
        "exact",
        "bm25",
        "hybrid",
        "lowrank_r16_ensemble",
        "lowrank_r8_kmeans_ensemble",
        "oracle_evidence",
    )
    labels = ("Mean", "Exact", "BM25", "Hybrid", "LR16", "LR8 m8", "Oracle")
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        lookup = {
            row["condition"]: row
            for row in summary
            if row["dataset"] == dataset and row["condition"] in conditions
        }
        axis.bar(labels, [lookup[name]["evidence_recall"] for name in conditions])
        axis.set_title(dataset.upper())
        axis.set_ylabel("Evidence recall at four chunks")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"confirmation_retrieval.{suffix}", dpi=190)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    if len(features) != 2 * args.examples_per_dataset:
        raise ValueError("Confirmation feature count does not match the frozen protocol.")
    old_rows = _read_csv(RESULT_ROOT / "natural_rows.csv")
    old_ids = {(row["dataset"], row["example_id"]) for row in old_rows}
    fresh_ids = {(row["dataset"], row["example_id"]) for row in features}
    overlap = sorted(old_ids & fresh_ids)
    if overlap:
        raise RuntimeError(f"Confirmation identities overlap prior evaluation: {overlap[:3]}")

    _project_native_queries({"confirmation": features}, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    from experiments.paper2_hf.routing.run_query_strategies import load_split_examples

    examples = load_split_examples(
        args.cache_dir, args.examples_per_dataset, args.offset, args.dataset_seed
    )
    examples_by_id = {(row["dataset"], row["id"]): row for row in examples}
    selectors = {}
    for rank in (8, 16):
        selectors[rank] = []
        for seed in args.seeds:
            path = args.checkpoint_dir / f"direct_lowrank_r{rank}_seed{seed}.pt"
            selectors[rank].append((*_load_selector(path, device), seed))

    rows = []
    for index, feature in enumerate(features, start=1):
        example = examples_by_id[(feature["dataset"], feature["example_id"])]
        query, keys, mask, _ = _case_tensors(feature, device)
        case = {
            "keys": feature["local_pre_key"],
            "mask": feature["local_token_mask"],
            "query_feature": _query_feature(query[0]).cpu(),
        }
        teacher = _score_compact(
            query, keys, mask, function="top_r_mean", head_reduction="mean"
        ).cpu()
        means = masked_mean_keys(keys, mask)
        mean_scores = _score_compact(
            query,
            means,
            torch.ones((len(means), 1), dtype=torch.bool, device=device),
            function="top_r_mean",
            head_reduction="mean",
        ).cpu()
        lexical = _lexical_scores(tokenizer, example, feature, mean_scores)
        rows.append(
            _row(
                feature,
                condition="full_k_teacher",
                seed=-1,
                scores=teacher,
                index_bytes_per_chunk=32 * 1024 * 4,
            )
        )
        rows.append(
            _row(
                feature,
                condition="native_mean",
                seed=-1,
                scores=mean_scores,
                index_bytes_per_chunk=1024 * 4,
            )
        )
        for name, scores in lexical.items():
            rows.append(_row(feature, condition=name, seed=-1, scores=scores))

        scores_by_condition: dict[str, list[torch.Tensor]] = defaultdict(list)
        for rank, static, condition, byte_count in (
            (16, False, "lowrank_r16", 32 * 16 * 4),
            (8, True, "lowrank_r8_kmeans", 8 * 8 * 4),
        ):
            for selector, checkpoint, seed in selectors[rank]:
                scores = _score_lowrank(
                    selector,
                    checkpoint,
                    case,
                    static_centroids=static,
                    device=device,
                )
                scores_by_condition[condition].append(scores)
                rows.append(
                    _row(
                        feature,
                        condition=condition,
                        seed=seed,
                        scores=scores,
                        index_bytes_per_chunk=byte_count,
                    )
                )
            ensemble_scores = torch.stack(scores_by_condition[condition]).mean(dim=0)
            rows.append(
                _row(
                    feature,
                    condition=f"{condition}_ensemble",
                    seed=-1,
                    scores=ensemble_scores,
                    index_bytes_per_chunk=byte_count,
                )
            )

        primary = torch.stack(scores_by_condition["lowrank_r16"]).mean(dim=0)
        rows.append(
            _row(
                feature,
                condition="oracle_evidence",
                seed=-1,
                scores=teacher,
                selected=_oracle_indices(feature["local_positive_mask"], teacher, ROUTING_BUDGET),
            )
        )
        rows.append(
            _row(
                feature,
                condition="shuffled_selection",
                seed=-1,
                scores=primary,
                selected=_stable_random_indices(
                    feature["example_id"], len(primary), ROUTING_BUDGET
                ),
            )
        )
        rows.append(
            _row(
                feature,
                condition="irrelevant_bottom",
                seed=-1,
                scores=primary,
                selected=sorted(stable_topk_indices(-primary, ROUTING_BUDGET).tolist()),
            )
        )
        print(
            f"[confirmation {index}/{len(features)}] {feature['dataset']} "
            f"{feature['example_id']}",
            flush=True,
        )

    summary = _aggregate(rows)
    paired = _paired(rows, args.bootstrap_seed)
    _write_csv(args.output_dir / "per_example.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "paired_effects.csv", paired)
    _plot(summary, args.output_dir)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_layer": 27,
        "cohort_offset_per_dataset": args.offset,
        "examples_per_dataset": args.examples_per_dataset,
        "identity_count": len(fresh_ids),
        "prior_identity_overlap": len(overlap),
        "configuration_selection_used_confirmation": False,
        "checkpoints_frozen_before_confirmation": True,
        "seeds": list(args.seeds),
        "routing_budget_chunks": ROUTING_BUDGET,
        "feature_artifact": {
            "path": str(args.features.resolve().relative_to(ROOT)),
            "bytes": args.features.stat().st_size,
            "sha256": _sha256(args.features),
            "tracked": False,
        },
        "checkpoint_hashes": {
            path.name: _sha256(path)
            for path in sorted(args.checkpoint_dir.glob("direct_lowrank_r*.pt"))
            if any(f"seed{seed}" in path.name for seed in args.seeds)
        },
        "materialized_native_kv_unchanged": True,
        "command": (
            "python experiments/paper2_8_qk_compression/run_confirmation.py "
            "--device cuda"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"rows": len(rows), "summary_rows": len(summary), "pairs": len(paired)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offset", type=int, default=CONFIRMATION_OFFSET)
    parser.add_argument("--examples-per-dataset", type=int, default=CONFIRMATION_PER_DATASET)
    parser.add_argument("--dataset-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument(
        "--seeds", type=lambda value: tuple(map(int, value.split(","))), default=SEEDS
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--features",
        type=Path,
        default=RESULT_ROOT / "native_qk_features_confirmation.pt",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "confirmation")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
