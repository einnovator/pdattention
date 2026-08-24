"""Run the Paper 2.8 post-G3 geometric and low-rank routing extension.

The original G0-G3 artifacts remain immutable. This runner adds matched native
multi-centroid controls and trains direct bilinear low-rank routers on the same
validation identities. Test identities are used only for final row emission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

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
    _score_compact,
    _sha256,
    _write_csv,
)
from experiments.paper2_8_qk_compression.run_query_conditioned_study import (
    _restore_test_cases,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.qk_compression import (
    QueryConditionedLandmarkSelector,
    chunk_routing_loss,
    farthest_first_indices,
    gather_landmarks,
    kmeans_centroids,
    kmeans_medoid_indices,
    low_rank_response_scores,
    masked_mean_keys,
    response_metrics,
    routing_metrics,
    stable_topk_indices,
)


RANKS = (8, 16, 32)
M_VALUES = (1, 2, 4, 8)
LOW_RANK_M_VALUES = (4, 8)
NATIVE_METHODS = ("native_mean", "native_kmeans", "native_medoid", "native_farthest")
LOW_RANK_METHODS = (
    "lowrank_all",
    "lowrank_kmeans",
    "lowrank_medoid",
    "lowrank_farthest",
)
OBJECTIVE = "combined"
TEACHER_FUNCTION = "top_r_mean"
HEAD_REDUCTION = "mean"
RESULT_ROOT = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
FROZEN_ARTIFACT_HASHES = {
    "natural_rows.csv": "47d7cd7190bb163cec3f868454502e360146a8fffe3a22b0f92862620b392c2d",
    "paper2_6_inherited_rows.csv": "e16d198407854eb65843dc1d6f58d58a88292304efd7002b81ba31a8697fb3b9",
    "query_conditioned/per_example.csv": "3cd7a23fbd35d4c0158dd3a8341f5377213e9db35904891faae919226613e900",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reduce_logits(
    logits: torch.Tensor, token_mask: torch.Tensor, *, top_r: int = 4
) -> torch.Tensor:
    masked = logits.masked_fill(~token_mask, float("-inf"))
    count = min(int(top_r), logits.shape[-1])
    values = masked.topk(count, dim=-1).values
    finite = torch.isfinite(values)
    return values.masked_fill(~finite, 0).sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)


def _parameter_breakdown(selector: QueryConditionedLandmarkSelector) -> dict[str, int]:
    salience = (
        sum(parameter.numel() for parameter in selector.salience.parameters())
        if selector.salience is not None
        else 0
    )
    query = (
        sum(parameter.numel() for parameter in selector.query_projection.parameters())
        if selector.query_projection is not None
        else 0
    )
    key = (
        sum(parameter.numel() for parameter in selector.feature_projection.parameters())
        if selector.feature_projection is not None
        else 0
    )
    return {
        "salience_parameters": salience,
        "wq_parameters": query,
        "wk_parameters": key,
        "total_parameters": salience + query + key,
    }


def _fit_direct_router(
    training_cases: list[dict[str, torch.Tensor]],
    *,
    rank: int,
    seed: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[QueryConditionedLandmarkSelector, list[dict], float]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(training_cases), generator=generator).tolist()
    selector = QueryConditionedLandmarkSelector(
        training_cases[0]["query"].shape[-1],
        feature_width=training_cases[0]["keys"].shape[-1],
        rank=rank,
        use_salience=False,
        use_interaction=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history = []
    _sync(device)
    started = time.perf_counter()
    for step in range(steps):
        if step and step % len(order) == 0:
            order = torch.randperm(len(training_cases), generator=generator).tolist()
        case = training_cases[order[step % len(order)]]
        keys = case["keys"].to(device)
        query = case["query"].to(device)
        token_mask = case["token_mask"].to(device)
        positives = case["positives"].to(device)
        teacher = case["teacher"].to(device)
        logits = selector(keys, query, token_mask)
        chunk_scores = _reduce_logits(logits, token_mask)
        loss, components = chunk_routing_loss(
            OBJECTIVE,
            chunk_scores.unsqueeze(0),
            positives.unsqueeze(0),
            token_mask.any(dim=-1).unsqueeze(0),
            teacher_scores=teacher.unsqueeze(0),
            budget=4,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite direct-router loss at step {step}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            history.append(
                {
                    "rank": rank,
                    "seed": seed,
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    **{
                        name: float(value.detach())
                        for name, value in components.items()
                    },
                }
            )
    _sync(device)
    selector.eval()
    return selector, history, time.perf_counter() - started


def _prepare_native_training_cases(
    validation_features: list[dict],
    prepared_batch: dict[str, torch.Tensor],
) -> tuple[list[dict[str, torch.Tensor]], float]:
    """Align validation native keys with the frozen prepared query/target batch."""
    if len(validation_features) != len(prepared_batch["queries"]):
        raise ValueError("Prepared validation batch does not align with native features.")
    square_sum = 0.0
    scalar_count = 0
    for feature in validation_features:
        keys = feature["local_pre_key"].float().flatten(2)
        mask = feature["local_token_mask"]
        valid = keys[mask]
        square_sum += float(valid.square().sum())
        scalar_count += valid.numel()
    key_scale = math.sqrt(square_sum / max(scalar_count, 1))
    cases = []
    for row, feature in enumerate(validation_features):
        chunks = len(feature["local_positive_mask"])
        cases.append(
            {
                "keys": feature["local_pre_key"].float().flatten(2) / key_scale,
                "query": prepared_batch["queries"][row],
                "token_mask": feature["local_token_mask"],
                "positives": feature["local_positive_mask"],
                "teacher": prepared_batch["teacher"][row, :chunks],
            }
        )
    return cases, key_scale


def _materialization_fields(case: dict, selected: list[int]) -> dict[str, float]:
    materialized = sum(int(case["mask"][chunk].sum()) for chunk in selected)
    key = case["keys"]
    scalar_bytes = key.element_size()
    native_width = key.shape[2] * key.shape[3]
    return {
        "requested_chunks": len(selected),
        "materialized_kv_tokens": materialized,
        "active_memory_fraction": materialized / max(case["source_tokens"], 1),
        "transfer_bytes": materialized * native_width * scalar_bytes * 2,
        "backing_native_kv_bytes": case["source_tokens"]
        * native_width
        * scalar_bytes
        * 2,
    }


def _metric_row(
    case: dict,
    *,
    method: str,
    rank: int,
    m: int,
    seed: int,
    scores: torch.Tensor,
    teacher: torch.Tensor,
    construction_ms: float,
    online_ms: float,
    index_bytes: int,
    native_dots: int,
    low_rank_dots: int,
    peak_delta_gpu_bytes: int,
    parameter_counts: dict[str, int],
    representation_detail: str,
) -> dict:
    selected = stable_topk_indices(scores, 4).cpu().tolist()
    route = routing_metrics(selected, case["positives"], budget=4)
    preservation = response_metrics(teacher.cpu(), scores.cpu())
    chunks = len(case["keys"])
    return {
        "dataset": case["dataset"],
        "example_id": case["example_id"],
        "method": method,
        "rank": rank,
        "m": m,
        "seed": seed,
        "objective": OBJECTIVE if seed >= 0 else "none",
        "candidate_chunks": chunks,
        "selected_chunks": " ".join(map(str, selected)),
        "representation_detail": representation_detail,
        "index_bytes": index_bytes,
        "index_bytes_per_chunk": index_bytes / chunks,
        "construction_ms": construction_ms,
        "cached_online_ms": online_ms,
        "native_dots": native_dots,
        "low_rank_dots": low_rank_dots,
        "peak_delta_gpu_bytes": peak_delta_gpu_bytes,
        **parameter_counts,
        **_materialization_fields(case, selected),
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


@torch.no_grad()
def _evaluate_native_control(
    case: dict,
    *,
    method: str,
    m: int,
    device: torch.device,
    repeats: int,
) -> dict:
    keys = case["keys"].to(device).float()
    mask = case["mask"].to(device)
    query = case["query"].to(device).float()
    teacher = case["teacher"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
    else:
        before = 0
    _sync(device)
    started = time.perf_counter()
    if method == "native_mean":
        representatives = masked_mean_keys(keys, mask)
        representative_mask = torch.ones(
            (len(keys), 1), dtype=torch.bool, device=device
        )
    elif method == "native_kmeans":
        representatives, representative_mask = kmeans_centroids(keys, mask, m)
    else:
        indices = (
            kmeans_medoid_indices(keys, mask, m)
            if method == "native_medoid"
            else farthest_first_indices(keys, mask, m)
        )
        representatives, representative_mask = gather_landmarks(keys, indices)
    _sync(device)
    construction_ms = 1000 * (time.perf_counter() - started)
    for _ in range(2):
        _score_compact(
            query,
            representatives,
            representative_mask,
            function=TEACHER_FUNCTION,
            head_reduction=HEAD_REDUCTION,
        )
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        scores = _score_compact(
            query,
            representatives,
            representative_mask,
            function=TEACHER_FUNCTION,
            head_reduction=HEAD_REDUCTION,
        )
    _sync(device)
    online_ms = 1000 * (time.perf_counter() - started) / repeats
    peak = torch.cuda.max_memory_allocated(device) - before if device.type == "cuda" else 0
    effective_m = representatives.shape[1]
    return _metric_row(
        case,
        method=method,
        rank=0,
        m=m,
        seed=-1,
        scores=scores,
        teacher=teacher,
        construction_ms=construction_ms,
        online_ms=online_ms,
        index_bytes=representatives.numel() * representatives.element_size(),
        native_dots=len(keys) * effective_m * query.shape[1],
        low_rank_dots=0,
        peak_delta_gpu_bytes=peak,
        parameter_counts={
            "salience_parameters": 0,
            "wq_parameters": 0,
            "wk_parameters": 0,
            "total_parameters": 0,
        },
        representation_detail=f"{effective_m}x{keys.shape[2] * keys.shape[3]} native-K scalars",
    )


def _gather_generic(
    values: torch.Tensor, indices: list[list[int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in indices)
    output = values.new_zeros((len(values), width, values.shape[-1]))
    mask = torch.zeros((len(values), width), dtype=torch.bool, device=values.device)
    for chunk, row in enumerate(indices):
        selected = torch.tensor(row, dtype=torch.long, device=values.device)
        output[chunk, : len(row)] = values[chunk].index_select(0, selected)
        mask[chunk, : len(row)] = True
    return output, mask


@torch.no_grad()
def _evaluate_low_rank(
    selector: QueryConditionedLandmarkSelector,
    case: dict,
    *,
    method: str,
    rank: int,
    m: int,
    seed: int,
    train_seconds: float,
    key_scale: float,
    device: torch.device,
    repeats: int,
) -> dict:
    features = case["keys"].to(device).float().flatten(2) / key_scale
    query_feature = case["query_feature"].to(device)
    mask = case["mask"].to(device)
    teacher = case["teacher"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
    else:
        before = 0
    _sync(device)
    started = time.perf_counter()
    projected = selector.feature_projection(features)
    if method == "lowrank_all":
        representatives, representative_mask = projected, mask
    elif method == "lowrank_kmeans":
        representatives, representative_mask = kmeans_centroids(projected, mask, m)
    else:
        clustering_values = projected.unsqueeze(2)
        indices = (
            kmeans_medoid_indices(projected, mask, m)
            if method == "lowrank_medoid"
            else farthest_first_indices(clustering_values, mask, m)
        )
        representatives, representative_mask = _gather_generic(projected, indices)
    _sync(device)
    construction_ms = 1000 * (time.perf_counter() - started)

    def score() -> torch.Tensor:
        projected_query = selector.query_projection(query_feature)
        return low_rank_response_scores(
            projected_query,
            representatives,
            representative_mask,
            function=TEACHER_FUNCTION,
            top_r=4,
        )[0]

    for _ in range(2):
        score()
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        scores = score()
    _sync(device)
    online_ms = 1000 * (time.perf_counter() - started) / repeats
    peak = torch.cuda.max_memory_allocated(device) - before if device.type == "cuda" else 0
    counts = _parameter_breakdown(selector)
    row = _metric_row(
        case,
        method=method,
        rank=rank,
        m=m,
        seed=seed,
        scores=scores,
        teacher=teacher,
        construction_ms=construction_ms,
        online_ms=online_ms,
        index_bytes=representatives.numel() * representatives.element_size(),
        native_dots=0,
        low_rank_dots=len(representatives) * representatives.shape[1] * rank,
        peak_delta_gpu_bytes=peak,
        parameter_counts=counts,
        representation_detail=(
            f"{representatives.shape[1]}x{rank} scalars projected from "
            f"{features.shape[-1]}-wide native K"
        ),
    )
    row["train_seconds"] = train_seconds
    return row


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], int(row["rank"]), int(row["m"]))].append(row)
    metrics = (
        "evidence_recall",
        "evidence_precision",
        "any_evidence",
        "chain_completion",
        "exact_identity",
        "mrr",
        "teacher_top4_overlap",
        "spearman",
        "kl",
        "materialized_kv_tokens",
        "active_memory_fraction",
        "index_bytes_per_chunk",
        "construction_ms",
        "cached_online_ms",
        "native_dots",
        "low_rank_dots",
        "transfer_bytes",
        "backing_native_kv_bytes",
        "peak_delta_gpu_bytes",
    )
    output = []
    for key, group in sorted(grouped.items()):
        record = {
            "dataset": key[0],
            "method": key[1],
            "rank": key[2],
            "m": key[3],
            "rows": len(group),
            "identities": len({row["example_id"] for row in group}),
            "seeds": len({int(row["seed"]) for row in group if int(row["seed"]) >= 0}),
            "salience_parameters": int(group[0]["salience_parameters"]),
            "wq_parameters": int(group[0]["wq_parameters"]),
            "wk_parameters": int(group[0]["wk_parameters"]),
            "total_parameters": int(group[0]["total_parameters"]),
        }
        for metric in metrics:
            record[metric] = sum(float(row[metric]) for row in group) / len(group)
        output.append(record)
    return output


def _seed_stability(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if int(row["seed"]) < 0:
            continue
        grouped[
            (row["dataset"], row["method"], int(row["rank"]), int(row["m"]), int(row["seed"]))
        ].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        output.append(
            {
                "dataset": key[0],
                "method": key[1],
                "rank": key[2],
                "m": key[3],
                "seed": key[4],
                "evidence_recall": sum(float(row["evidence_recall"]) for row in group)
                / len(group),
                "teacher_top4_overlap": sum(
                    float(row["teacher_top4_overlap"]) for row in group
                )
                / len(group),
            }
        )
    return output


def _parity_audit(results_root: Path) -> dict:
    required = {
        "natural_rows.csv": 1728,
        "query_conditioned/per_example.csv": 3840,
        "paper2_6_inherited_rows.csv": 224,
    }
    rows = {}
    hashes = {}
    passed = True
    for relative, expected in required.items():
        path = results_root / relative
        count = len(_read_csv(path))
        rows[relative] = {"actual": count, "expected": expected, "match": count == expected}
        actual_hash = _sha256(path)
        expected_hash = FROZEN_ARTIFACT_HASHES[relative]
        hashes[relative] = {
            "actual": actual_hash,
            "expected": expected_hash,
            "match": actual_hash == expected_hash,
        }
        passed = passed and count == expected and actual_hash == expected_hash
    query_manifest = json.loads(
        (results_root / "query_conditioned/manifest.json").read_text(encoding="utf-8")
    )
    feature_hashes = {
        split: query_manifest["feature_artifacts"][split]["sha256"]
        for split in ("validation", "test")
    }
    return {
        "gate": "E0",
        "passed": passed,
        "row_counts": rows,
        "artifact_hashes": hashes,
        "feature_hashes": feature_hashes,
        "model_revision": query_manifest["model_revision"],
        "original_gates_unchanged": True,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_root = RESULT_ROOT
    parity = _parity_audit(result_root)
    (args.output_dir / "baseline_parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not parity["passed"]:
        raise RuntimeError("E0 baseline artifact parity failed.")

    prepared_path = result_root / "query_conditioned/prepared_cache.pt"
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
    test_features = torch.load(args.test_features, map_location="cpu", weights_only=False)
    validation_features = torch.load(
        args.validation_features, map_location="cpu", weights_only=False
    )
    validation_identities = {
        (feature["dataset"], feature["example_id"]) for feature in validation_features
    }
    test_identities = {
        (feature["dataset"], feature["example_id"]) for feature in test_features
    }
    identity_overlap = sorted(validation_identities & test_identities)
    if identity_overlap:
        raise RuntimeError(f"Validation/test identity overlap: {identity_overlap}")
    test_cases = _restore_test_cases(prepared["test_auxiliary"], test_features)
    training_cases, key_scale = _prepare_native_training_cases(
        validation_features, prepared["training_batch"]
    )
    del validation_features
    row_path = args.output_dir / "per_example.csv"
    history_path = args.output_dir / "training_history.csv"
    if args.overwrite:
        row_path.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
    rows = _read_csv(row_path) if row_path.exists() else []
    history = _read_csv(history_path) if history_path.exists() else []
    completed = {
        (row["method"], int(row["rank"]), int(row["m"]), int(row["seed"]))
        for row in rows
    }

    for method in NATIVE_METHODS:
        values = (1,) if method == "native_mean" else M_VALUES
        for m in values:
            key = (method, 0, m, -1)
            if key in completed:
                continue
            print(f"[native control] {method} m={m}", flush=True)
            for case in test_cases:
                rows.append(
                    _evaluate_native_control(
                        case,
                        method=method,
                        m=m,
                        device=device,
                        repeats=args.timing_repeats,
                    )
                )
            _write_csv(row_path, rows)

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for rank in args.ranks:
        for seed in args.seeds:
            needed = {
                (method, rank, m, seed)
                for method in LOW_RANK_METHODS
                for m in ((32,) if method == "lowrank_all" else LOW_RANK_M_VALUES)
            }
            if needed.issubset(completed):
                print(f"[resume direct router] rank={rank} seed={seed}", flush=True)
                continue
            print(f"[train direct router] rank={rank} seed={seed}", flush=True)
            selector, run_history, train_seconds = _fit_direct_router(
                training_cases,
                rank=rank,
                seed=seed,
                steps=args.steps,
                learning_rate=args.learning_rate,
                device=device,
            )
            for record in run_history:
                record["train_seconds"] = train_seconds
            history.extend(run_history)
            for method in LOW_RANK_METHODS:
                values = (32,) if method == "lowrank_all" else LOW_RANK_M_VALUES
                for m in values:
                    key = (method, rank, m, seed)
                    if key in completed:
                        continue
                    print(f"[evaluate] {key}", flush=True)
                    for case in test_cases:
                        rows.append(
                            _evaluate_low_rank(
                                selector,
                                case,
                                method=method,
                                rank=rank,
                                m=m,
                                seed=seed,
                                train_seconds=train_seconds,
                                key_scale=key_scale,
                                device=device,
                                repeats=args.timing_repeats,
                            )
                        )
                    _write_csv(row_path, rows)
            _write_csv(history_path, history)
            torch.save(
                {
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in selector.state_dict().items()
                    },
                    "rank": rank,
                    "seed": seed,
                    "objective": OBJECTIVE,
                    "steps": args.steps,
                    "learning_rate": args.learning_rate,
                    "train_seconds": train_seconds,
                    "native_key_rms_scale": key_scale,
                    **_parameter_breakdown(selector),
                },
                checkpoint_dir / f"direct_lowrank_r{rank}_seed{seed}.pt",
            )
            del selector
            if device.type == "cuda":
                torch.cuda.empty_cache()

    typed_rows = []
    integer_fields = {
        "rank",
        "m",
        "seed",
        "candidate_chunks",
        "index_bytes",
        "native_dots",
        "low_rank_dots",
        "peak_delta_gpu_bytes",
        "salience_parameters",
        "wq_parameters",
        "wk_parameters",
        "total_parameters",
    }
    for row in rows:
        typed_rows.append(
            {
                key: int(value) if key in integer_fields else value
                for key, value in row.items()
            }
        )
    summary = _aggregate(typed_rows)
    stability = _seed_stability(typed_rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "seed_stability.csv", stability)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_layer": 27,
        "representation": "pre_rope_native_qk_and_cached_low_rank_key_features",
        "teacher_function": TEACHER_FUNCTION,
        "head_reduction": HEAD_REDUCTION,
        "candidate_chunk_tokens": 32,
        "materialization_budget_chunks": 4,
        "ranks": list(args.ranks),
        "native_m_values": list(M_VALUES),
        "low_rank_m_values": list(LOW_RANK_M_VALUES),
        "seeds": list(args.seeds),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "direct_objective": OBJECTIVE,
        "low_rank_key_input_width": training_cases[0]["keys"].shape[-1],
        "low_rank_query_input_width": training_cases[0]["query"].shape[-1],
        "native_key_rms_scale": key_scale,
        "backbone_frozen": True,
        "native_qk_scan_used_by_deployable_lowrank_router": False,
        "materialized_native_kv_unchanged": True,
        "test_used_for_model_selection": False,
        "validation_test_identity_disjoint": True,
        "validation_identity_count": len(validation_identities),
        "test_identity_count": len(test_identities),
        "post_g3_extension": True,
        "original_gates_unchanged": True,
        "baseline_parity": parity,
        "feature_artifacts": {
            "validation": {
                "path": str(args.validation_features.relative_to(ROOT)),
                "bytes": args.validation_features.stat().st_size,
                "sha256": _sha256(args.validation_features),
                "tracked": False,
            },
            "test": {
                "path": str(args.test_features.relative_to(ROOT)),
                "bytes": args.test_features.stat().st_size,
                "sha256": _sha256(args.test_features),
                "tracked": False,
            },
            "prepared": {
                "path": str(prepared_path.relative_to(ROOT)),
                "bytes": prepared_path.stat().st_size,
                "sha256": _sha256(prepared_path),
                "tracked": False,
            },
        },
        "command": "python experiments/paper2_8_qk_compression/run_low_rank_frontier.py --device cuda",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "rows": len(rows),
        "summary_rows": len(summary),
        "baseline_parity": parity["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument(
        "--ranks", type=lambda value: tuple(map(int, value.split(","))), default=RANKS
    )
    parser.add_argument(
        "--seeds", type=lambda value: tuple(map(int, value.split(","))), default=SEEDS
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=RESULT_ROOT / "low_rank_frontier"
    )
    parser.add_argument(
        "--validation-features",
        type=Path,
        default=RESULT_ROOT / "native_qk_features_validation.pt",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=RESULT_ROOT / "native_qk_features_test.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
