"""Train and evaluate query-conditioned native-landmark selectors.

This Paper 2.8 continuation keeps the frozen natural cohort, native-QK teacher,
32-token candidates, and four-chunk materialization budget unchanged. It crosses
low-rank query interactions, landmark counts, retrieval-aware objectives, and
five seeds without selecting a configuration on the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

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
    _selection,
    _sha256,
    _write_csv,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.qk_compression import (
    LANDMARK_TRAINING_OBJECTIVES,
    QueryConditionedLandmarkSelector,
    gather_landmarks,
    greedy_qk_landmarks,
    landmark_features,
    landmark_training_loss,
    response_metrics,
    routing_metrics,
    token_query_key_dots,
)


RANKS = (8, 16, 32)
M_VALUES = (4, 8)
OBJECTIVES = LANDMARK_TRAINING_OBJECTIVES
PRIMARY_CONFIGURATION = {"objective": "combined", "rank": 16, "m": 4}
PAPER26_CONDITIONS = (
    "B0_gist",
    "B1_bm25",
    "B2_exact",
    "H5_iterative_hybrid",
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _query_feature(query: torch.Tensor) -> torch.Tensor:
    flat = query.float().flatten()
    return torch.nn.functional.normalize(flat, dim=0) * math.sqrt(len(flat))


def _feature_statistics(validation: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for feature in validation:
        keys = feature["local_pre_key"].float()
        mask = feature["local_token_mask"]
        rows.append(landmark_features(keys, mask)[mask])
    values = torch.cat(rows)
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-5)


def _prepare_training_batch(
    validation: list[dict],
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    function: str,
    head_reduction: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    records = []
    for index, feature in enumerate(validation, start=1):
        query, keys, mask, positives = _case_tensors(feature, device)
        normalized = (
            landmark_features(keys, mask) - feature_mean.to(device)
        ) / feature_scale.to(device)
        dots, _ = token_query_key_dots(query, keys, mask)
        token_responses = dots[0].mean(dim=-1)
        teacher = _score_compact(
            query, keys, mask, function=function, head_reduction=head_reduction
        )
        oracle_indices = greedy_qk_landmarks(
            query,
            keys,
            mask,
            max(M_VALUES),
            function=function,
            top_r=4,
            head_reduction=head_reduction,
        )
        oracle_targets = {}
        for m in M_VALUES:
            targets = torch.zeros_like(mask, dtype=torch.float32)
            for chunk, selected in enumerate(oracle_indices):
                targets[chunk, selected[:m]] = 1
            oracle_targets[m] = targets.cpu()
        records.append(
            {
                "features": normalized.cpu(),
                "query": _query_feature(query[0]).cpu(),
                "token_responses": token_responses.cpu(),
                "token_mask": mask.cpu(),
                "positives": positives.cpu(),
                "teacher": teacher.cpu(),
                "oracle": oracle_targets,
            }
        )
        print(f"[training preparation {index}/{len(validation)}]", flush=True)

    batch, max_chunks, tokens = len(records), max(
        len(record["positives"]) for record in records
    ), records[0]["token_mask"].shape[1]
    features = torch.zeros(batch, max_chunks, tokens, 8)
    queries = torch.stack([record["query"] for record in records])
    responses = torch.zeros(batch, max_chunks, tokens)
    masks = torch.zeros(batch, max_chunks, tokens, dtype=torch.bool)
    positives = torch.zeros(batch, max_chunks, dtype=torch.bool)
    teachers = torch.zeros(batch, max_chunks)
    oracle = {m: torch.zeros_like(masks, dtype=torch.float32) for m in M_VALUES}
    for row, record in enumerate(records):
        chunks = len(record["positives"])
        features[row, :chunks] = record["features"]
        responses[row, :chunks] = record["token_responses"]
        masks[row, :chunks] = record["token_mask"]
        positives[row, :chunks] = record["positives"]
        teachers[row, :chunks] = record["teacher"]
        for m in M_VALUES:
            oracle[m][row, :chunks] = record["oracle"][m]
    output = {
        "features": features,
        "queries": queries,
        "token_responses": responses,
        "token_mask": masks,
        "positives": positives,
        "teacher": teachers,
    }
    output.update({f"oracle_{m}": values for m, values in oracle.items()})
    return {key: value.to(device) for key, value in output.items()}


def _prepare_test_cases(
    test: list[dict],
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    function: str,
    head_reduction: str,
    device: torch.device,
) -> list[dict]:
    output = []
    for index, feature in enumerate(test, start=1):
        query, keys, mask, positives = _case_tensors(feature, device)
        teacher = _score_compact(
            query, keys, mask, function=function, head_reduction=head_reduction
        )
        normalized = (
            landmark_features(keys, mask) - feature_mean.to(device)
        ) / feature_scale.to(device)
        output.append(
            {
                "dataset": feature["dataset"],
                "example_id": feature["example_id"],
                "features": normalized.cpu(),
                "query_feature": _query_feature(query[0]).cpu(),
                "query": query.cpu(),
                "keys": feature["local_pre_key"],
                "mask": feature["local_token_mask"],
                "positives": positives.cpu(),
                "teacher": teacher.cpu(),
                "source_tokens": int(feature["source_tokens"]),
            }
        )
        print(f"[test preparation {index}/{len(test)}]", flush=True)
    return output


def _restore_test_cases(auxiliary: list[dict], features: list[dict]) -> list[dict]:
    if len(auxiliary) != len(features):
        raise ValueError("Prepared test cache does not match the frozen feature cohort.")
    output = []
    for cached, feature in zip(auxiliary, features):
        if cached["example_id"] != feature["example_id"]:
            raise ValueError("Prepared test identities are not aligned with the feature cache.")
        output.append(
            {
                **cached,
                "keys": feature["local_pre_key"],
                "mask": feature["local_token_mask"],
                "positives": feature["local_positive_mask"],
            }
        )
    return output


def _fit_selector(
    batch: dict[str, torch.Tensor],
    *,
    objective: str,
    rank: int,
    m: int,
    seed: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[QueryConditionedLandmarkSelector, list[dict], float]:
    torch.manual_seed(seed)
    query_width = batch["queries"].shape[-1]
    selector = QueryConditionedLandmarkSelector(
        query_width, hidden_width=32, rank=rank
    ).to(device)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps):
        logits = selector(batch["features"], batch["queries"], batch["token_mask"])
        loss, components = landmark_training_loss(
            objective,
            logits,
            batch["token_responses"],
            batch["token_mask"],
            batch["positives"],
            m=m,
            teacher_scores=batch["teacher"],
            oracle_targets=batch[f"oracle_{m}"],
            budget=4,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"Non-finite loss for {objective}, rank={rank}, m={m}, seed={seed}."
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 4, steps // 2, 3 * steps // 4, steps - 1}:
            history.append(
                {
                    "objective": objective,
                    "rank": rank,
                    "m": m,
                    "seed": seed,
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    **{
                        f"{name}_loss": float(value.detach().cpu())
                        for name, value in components.items()
                    },
                }
            )
    selector.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return selector, history, time.perf_counter() - started


@torch.no_grad()
def _evaluate_selector(
    selector: QueryConditionedLandmarkSelector,
    test_cases: list[dict],
    *,
    objective: str,
    rank: int,
    m: int,
    seed: int,
    function: str,
    head_reduction: str,
    device: torch.device,
) -> list[dict]:
    rows = []
    parameter_count = sum(parameter.numel() for parameter in selector.parameters())
    for case in test_cases:
        mask = case["mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = selector(
            case["features"].to(device),
            case["query_feature"].to(device),
            mask,
        )
        indices = []
        for chunk_logits, chunk_mask in zip(logits, mask):
            count = min(m, int(chunk_mask.sum().item()))
            indices.append(
                chunk_logits.topk(count).indices.sort().values.cpu().tolist()
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        selection_ms = 1000 * (time.perf_counter() - started)
        compact, compact_mask = gather_landmarks(case["keys"], indices)
        scores = _score_compact(
            case["query"].to(device),
            compact.to(device).float(),
            compact_mask.to(device),
            function=function,
            head_reduction=head_reduction,
        )
        selected = _selection(scores, 4)
        route = routing_metrics(selected, case["positives"], budget=4)
        preservation = response_metrics(case["teacher"], scores.cpu())
        materialized_tokens = sum(int(case["mask"][chunk].sum()) for chunk in selected)
        rows.append(
            {
                "dataset": case["dataset"],
                "example_id": case["example_id"],
                "objective": objective,
                "rank": rank,
                "m": m,
                "seed": seed,
                "parameter_count": parameter_count,
                "selection_ms": selection_ms,
                "teacher_function": function,
                "head_reduction": head_reduction,
                "selected_chunks": " ".join(map(str, selected)),
                "landmark_indices": json.dumps(indices),
                "materialized_kv_tokens": materialized_tokens,
                "active_memory_fraction": materialized_tokens
                / max(case["source_tokens"], 1),
                "native_dots": len(case["keys"])
                * m
                * case["query"].shape[1],
                **route,
                **{
                    name: value
                    for name, value in asdict(preservation).items()
                    if name != "topk_overlap"
                },
                **{
                    f"teacher_top{k}_overlap": value
                    for k, value in preservation.topk_overlap.items()
                },
            }
        )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["objective"], int(row["rank"]), int(row["m"]))].append(row)
    output = []
    metrics = (
        "evidence_recall",
        "evidence_precision",
        "any_evidence",
        "chain_completion",
        "mrr",
        "teacher_top4_overlap",
        "spearman",
        "kl",
        "materialized_kv_tokens",
        "active_memory_fraction",
        "native_dots",
        "selection_ms",
    )
    for (dataset, objective, rank, m), group in sorted(grouped.items()):
        record = {
            "dataset": dataset,
            "objective": objective,
            "rank": rank,
            "m": m,
            "rows": len(group),
            "identities": len({row["example_id"] for row in group}),
            "seeds": len({int(row["seed"]) for row in group}),
            "parameter_count": int(group[0]["parameter_count"]),
        }
        for metric in metrics:
            record[metric] = sum(float(row[metric]) for row in group) / len(group)
        output.append(record)
    return output


def _paired_effects(rows: list[dict], seed: int) -> list[dict]:
    original = _read_csv(
        ROOT / "docs/papers/shared/results/paper2_8_qk_compression/natural_rows.csv"
    )
    paper26 = _read_csv(
        ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/per_example.csv"
    )
    baselines = {}
    key_only = defaultdict(list)
    for row in original:
        if row["split"] == "test" and row["method"] == "mean":
            baselines[(row["dataset"], "QK_mean", row["example_id"])] = float(
                row["evidence_recall"]
            )
        if (
            row["split"] == "test"
            and row["method"] == "learned"
            and int(row["m"]) == 8
        ):
            key_only[(row["dataset"], row["example_id"])].append(
                float(row["evidence_recall"])
            )
    for (dataset, example_id), values in key_only.items():
        baselines[(dataset, "QK_key_only_m8", example_id)] = sum(values) / len(values)
    for row in paper26:
        if row["split"] == "test" and row["condition"] in PAPER26_CONDITIONS:
            baselines[(row["dataset"], row["condition"], row["example_id"])] = float(
                row["evidence_recall"]
            )

    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["objective"],
            int(row["rank"]),
            int(row["m"]),
            row["example_id"],
        )
        grouped[key].append(float(row["evidence_recall"]))
    output = []
    configurations = sorted({key[:4] for key in grouped})
    for dataset, objective, rank, m in configurations:
        identities = {
            key[4]: sum(values) / len(values)
            for key, values in grouped.items()
            if key[:4] == (dataset, objective, rank, m)
        }
        conditions = ("QK_mean", "QK_key_only_m8") + PAPER26_CONDITIONS
        for condition in conditions:
            differences = [
                value - baselines[(dataset, condition, identity)]
                for identity, value in identities.items()
                if (dataset, condition, identity) in baselines
            ]
            if not differences:
                continue
            low, high = _bootstrap(
                differences,
                seed + rank * 101 + m * 17 + sum(map(ord, objective + condition)),
            )
            output.append(
                {
                    "dataset": dataset,
                    "objective": objective,
                    "rank": rank,
                    "m": m,
                    "baseline": condition,
                    "pairs": len(differences),
                    "mean_delta": sum(differences) / len(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": sum(value > 0 for value in differences),
                    "losses": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    return output


def _plots(summary: list[dict], output_dir: Path) -> None:
    labels = [name.replace("_", "\n") for name in OBJECTIVES]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for axis, (dataset, m) in zip(
        axes.flat,
        (("hotpotqa", 4), ("hotpotqa", 8), ("qasper", 4), ("qasper", 8)),
    ):
        for rank, marker in zip(RANKS, ("o", "s", "^")):
            values = [
                next(
                    row["evidence_recall"]
                    for row in summary
                    if row["dataset"] == dataset
                    and row["objective"] == objective
                    and row["rank"] == rank
                    and row["m"] == m
                )
                for objective in OBJECTIVES
            ]
            axis.plot(labels, values, marker=marker, label=f"rank {rank}")
        axis.set_title(f"{dataset.upper()}, m={m}")
        axis.set_ylabel("Evidence recall at four chunks")
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    axes[0, 1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"objective_rank_recall.{suffix}", dpi=180)
    plt.close(figure)

    primary = [
        row
        for row in summary
        if row["objective"] == PRIMARY_CONFIGURATION["objective"]
        and row["rank"] == PRIMARY_CONFIGURATION["rank"]
        and row["m"] == PRIMARY_CONFIGURATION["m"]
    ]
    original_comparison = _read_csv(
        ROOT
        / "docs/papers/shared/results/paper2_8_qk_compression/paper2_6_matched_comparison.csv"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        baseline_names = ("B0_gist", "B1_bm25", "B2_exact", "H5_iterative_hybrid")
        values = [
            next(
                float(row["evidence_recall"])
                for row in original_comparison
                if row["dataset"] == dataset and row["condition"] == baseline
            )
            for baseline in baseline_names
        ]
        values.append(next(row["evidence_recall"] for row in primary if row["dataset"] == dataset))
        axis.bar(
            ["gist", "BM25", "exact", "hybrid", "QC QK"],
            values,
            color=["#6c757d", "#457b9d", "#d1495b", "#edae49", "#2a9d8f"],
        )
        axis.set_title(dataset.upper())
        axis.set_ylabel("Evidence recall at four chunks")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"primary_paper2_6_comparison.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.output_dir / "per_example.csv"
    history_path = args.output_dir / "training_history.csv"
    if args.overwrite:
        row_path.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
    existing_rows = _read_csv(row_path)
    existing_history = _read_csv(history_path)
    completed = {
        (row["objective"], int(row["rank"]), int(row["m"]), int(row["seed"]))
        for row in existing_rows
    }

    features = {
        "validation": torch.load(
            args.validation_features, map_location="cpu", weights_only=False
        ),
        "test": torch.load(args.test_features, map_location="cpu", weights_only=False),
    }
    feature_hashes = {
        "validation": _sha256(args.validation_features),
        "test": _sha256(args.test_features),
    }
    prepared_path = args.output_dir / "prepared_cache.pt"
    prepared = None
    if prepared_path.exists() and not args.rebuild_prepared_cache:
        candidate = torch.load(prepared_path, map_location="cpu", weights_only=False)
        expected = {
            "feature_hashes": feature_hashes,
            "model_revision": MODEL_REVISION,
            "teacher_function": args.teacher_function,
            "head_reduction": args.head_reduction,
        }
        if all(candidate.get(key) == value for key, value in expected.items()):
            prepared = candidate
            print(f"[prepared cache] {prepared_path}", flush=True)
    if prepared is None:
        _project_native_queries(features, device)
        feature_mean, feature_scale = _feature_statistics(features["validation"])
        training_batch = _prepare_training_batch(
            features["validation"],
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            function=args.teacher_function,
            head_reduction=args.head_reduction,
            device=device,
        )
        test_cases = _prepare_test_cases(
            features["test"],
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            function=args.teacher_function,
            head_reduction=args.head_reduction,
            device=device,
        )
        prepared = {
            "feature_hashes": feature_hashes,
            "model_revision": MODEL_REVISION,
            "teacher_function": args.teacher_function,
            "head_reduction": args.head_reduction,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "training_batch": {
                key: value.cpu() for key, value in training_batch.items()
            },
            "test_auxiliary": [
                {
                    key: value
                    for key, value in case.items()
                    if key not in {"keys", "mask", "positives"}
                }
                for case in test_cases
            ],
        }
        torch.save(prepared, prepared_path)
    else:
        feature_mean = prepared["feature_mean"]
        feature_scale = prepared["feature_scale"]
        training_batch = {
            key: value.to(device) for key, value in prepared["training_batch"].items()
        }
        test_cases = _restore_test_cases(prepared["test_auxiliary"], features["test"])
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    rows = list(existing_rows)
    history = list(existing_history)
    for objective in args.objectives:
        for rank in args.ranks:
            for m in args.m_values:
                for seed in args.seeds:
                    configuration = (objective, rank, m, seed)
                    if configuration in completed:
                        print(f"[resume] {configuration}", flush=True)
                        continue
                    print(f"[train] {configuration}", flush=True)
                    selector, run_history, train_seconds = _fit_selector(
                        training_batch,
                        objective=objective,
                        rank=rank,
                        m=m,
                        seed=seed,
                        steps=args.steps,
                        learning_rate=args.learning_rate,
                        device=device,
                    )
                    for history_row in run_history:
                        history_row["train_seconds"] = train_seconds
                    run_rows = _evaluate_selector(
                        selector,
                        test_cases,
                        objective=objective,
                        rank=rank,
                        m=m,
                        seed=seed,
                        function=args.teacher_function,
                        head_reduction=args.head_reduction,
                        device=device,
                    )
                    rows.extend(run_rows)
                    history.extend(run_history)
                    _write_csv(row_path, rows)
                    _write_csv(history_path, history)
                    if seed == args.seeds[0]:
                        torch.save(
                            {
                                "state_dict": {
                                    name: value.detach().cpu()
                                    for name, value in selector.state_dict().items()
                                },
                                "feature_mean": feature_mean,
                                "feature_scale": feature_scale,
                                "objective": objective,
                                "rank": rank,
                                "m": m,
                                "seed": seed,
                                "steps": args.steps,
                                "train_seconds": train_seconds,
                                "parameter_count": sum(
                                    parameter.numel()
                                    for parameter in selector.parameters()
                                ),
                            },
                            checkpoint_dir
                            / f"qc_{objective}_rank{rank}_m{m}_seed{seed}.pt",
                        )
                    del selector
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    numeric_rows = [
        {
            key: (
                int(value)
                if key in {"rank", "m", "seed", "parameter_count"}
                else float(value)
                if key
                in {
                    "evidence_recall",
                    "evidence_precision",
                    "any_evidence",
                    "chain_completion",
                    "mrr",
                    "teacher_top4_overlap",
                    "spearman",
                    "kl",
                    "materialized_kv_tokens",
                    "active_memory_fraction",
                    "native_dots",
                    "selection_ms",
                }
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    summary = _aggregate(numeric_rows)
    paired = _paired_effects(numeric_rows, args.seed)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "paired_effects.csv", paired)
    if (
        set(args.objectives) == set(OBJECTIVES)
        and set(args.ranks) == set(RANKS)
        and set(args.m_values) == set(M_VALUES)
    ):
        _plots(summary, args.output_dir)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_layer": 27,
        "representation": "pre_rope_native_qk",
        "candidate_chunk_tokens": 32,
        "materialization_budget_chunks": 4,
        "teacher_function": args.teacher_function,
        "head_reduction": args.head_reduction,
        "ranks": args.ranks,
        "m_values": args.m_values,
        "objectives": args.objectives,
        "seeds": args.seeds,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "primary_configuration": PRIMARY_CONFIGURATION,
        "backbone_frozen": True,
        "test_used_for_model_selection": False,
        "feature_artifacts": {
            "validation": {
                "path": str(args.validation_features.relative_to(ROOT)),
                "bytes": args.validation_features.stat().st_size,
                "sha256": feature_hashes["validation"],
                "tracked": False,
            },
            "test": {
                "path": str(args.test_features.relative_to(ROOT)),
                "bytes": args.test_features.stat().st_size,
                "sha256": feature_hashes["test"],
                "tracked": False,
            },
        },
        "command": (
            "python experiments/paper2_8_qk_compression/"
            "run_query_conditioned_study.py --device cuda"
        ),
        "prepared_cache": {
            "path": str(prepared_path.relative_to(ROOT)),
            "bytes": prepared_path.stat().st_size,
            "sha256": _sha256(prepared_path),
            "tracked": False,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"manifest": manifest, "configurations": len(summary)}


def parse_args() -> argparse.Namespace:
    feature_dir = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
    output_dir = feature_dir / "query_conditioned"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--teacher-function", default="top_r_mean")
    parser.add_argument("--head-reduction", default="mean")
    parser.add_argument(
        "--ranks", type=lambda value: tuple(map(int, value.split(","))), default=RANKS
    )
    parser.add_argument(
        "--m-values",
        type=lambda value: tuple(map(int, value.split(","))),
        default=M_VALUES,
    )
    parser.add_argument(
        "--objectives",
        type=lambda value: tuple(value.split(",")),
        default=OBJECTIVES,
    )
    parser.add_argument(
        "--seeds", type=lambda value: tuple(map(int, value.split(","))), default=SEEDS
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-prepared-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--validation-features",
        type=Path,
        default=feature_dir / "native_qk_features_validation.pt",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=feature_dir / "native_qk_features_test.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
