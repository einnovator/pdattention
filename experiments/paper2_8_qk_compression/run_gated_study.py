"""Run the gated Paper 2.8 QK-response compression study.

The natural protocol reuses Paper 2.5's frozen layer-27 pre-RoPE Q/K cache and
Paper 2.6's 32-token candidate chunks.  Compressors affect routing scores only;
every condition materializes the same four selected native chunks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from torch import nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_hf.qk_compression import (
    NativeLandmarkSelector,
    farthest_first_indices,
    gather_landmarks,
    greedy_qk_landmarks,
    landmark_features,
    last_token_indices,
    masked_mean_keys,
    qk_response_scores,
    random_token_indices,
    response_metrics,
    routing_metrics,
)


SEEDS = (11, 23, 37, 53, 71)
M_VALUES = (1, 2, 4, 8)
TEACHER_GRID = (
    ("max", "mean"),
    ("max", "max"),
    ("top_r_mean", "mean"),
    ("logsumexp", "mean"),
    ("attention_mass", "mean"),
)
HISTORICAL_CONDITIONS = (
    "B0_gist",
    "B1_bm25",
    "B2_exact",
    "B3_weighted",
    "B4_approx",
    "H5_iterative_hybrid",
    "O1_oracle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return sum(values) / max(len(values), 1)


def _bootstrap(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)
    )
    return draws[int(0.025 * samples)], draws[min(int(0.975 * samples), samples - 1)]


def _project_native_queries(features_by_split: dict[str, list[dict]], device: torch.device) -> None:
    """Project cached attention-input states with unchanged layer-27 Q weights."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    attention = model.model.layers[27].self_attn
    for split, features in features_by_split.items():
        for index, feature in enumerate(features, start=1):
            hidden = feature["query_hidden"].to(device=device, dtype=dtype).view(1, 1, -1)
            with torch.no_grad():
                projected = attention.q_proj(hidden).view(
                    1, 1, attention.config.num_attention_heads, attention.head_dim
                )
                if hasattr(attention, "q_norm"):
                    projected = attention.q_norm(projected)
            feature["query_pre_query"] = projected[0, 0].float().cpu()
            print(
                f"[query projection {split} {index}/{len(features)}] "
                f"{feature['dataset']} {feature['example_id']}",
                flush=True,
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _case_tensors(feature: dict, device: torch.device):
    return (
        feature["query_pre_query"].to(device).unsqueeze(0),
        feature["local_pre_key"].to(device).float(),
        feature["local_token_mask"].to(device),
        feature["local_positive_mask"].to(device),
    )


def _score_compact(
    queries: torch.Tensor,
    keys: torch.Tensor,
    mask: torch.Tensor,
    *,
    function: str,
    head_reduction: str,
) -> torch.Tensor:
    return qk_response_scores(
        queries,
        keys,
        mask,
        function=function,
        top_r=4,
        temperature=1.0,
        head_reduction=head_reduction,
    )[0]


def _selection(scores: torch.Tensor, budget: int) -> list[int]:
    return scores.topk(min(int(budget), len(scores))).indices.tolist()


def _teacher_selection(
    validation: list[dict], device: torch.device, budget: int
) -> tuple[str, str, list[dict]]:
    rows = []
    for function, head_reduction in TEACHER_GRID:
        for feature in validation:
            queries, keys, mask, positives = _case_tensors(feature, device)
            teacher = _score_compact(
                queries,
                keys,
                mask,
                function=function,
                head_reduction=head_reduction,
            )
            mean_keys = masked_mean_keys(keys, mask)
            mean_scores = _score_compact(
                queries,
                mean_keys,
                torch.ones(keys.shape[0], 1, dtype=torch.bool, device=device),
                function=function,
                head_reduction=head_reduction,
            )
            teacher_route = routing_metrics(
                _selection(teacher, budget), positives, budget=budget
            )
            mean_route = routing_metrics(
                _selection(mean_scores, budget), positives, budget=budget
            )
            rows.append(
                {
                    "dataset": feature["dataset"],
                    "example_id": feature["example_id"],
                    "teacher_function": function,
                    "head_reduction": head_reduction,
                    **{f"teacher_{key}": value for key, value in teacher_route.items()},
                    **{f"mean_{key}": value for key, value in mean_route.items()},
                }
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["teacher_function"], row["head_reduction"])].append(row)
    ranking = []
    for (function, head_reduction), group in grouped.items():
        ranking.append(
            (
                _mean(group, "teacher_evidence_recall"),
                _mean(group, "teacher_chain_completion"),
                _mean(group, "teacher_mrr"),
                _mean(group, "teacher_evidence_recall")
                - _mean(group, "mean_evidence_recall"),
                function,
                head_reduction,
            )
        )
    selected = max(ranking)
    return selected[-2], selected[-1], rows


def _synthetic_cases(seed: int, examples: int, device: torch.device) -> list[dict]:
    """Create sparse native landmarks hidden by misleading chunk means."""
    generator = torch.Generator().manual_seed(seed)
    output = []
    for chunk_tokens in (32, 64, 128, 256):
        for example in range(examples):
            chunks, kv_heads, query_heads, head_dim = 16, 2, 4, 16
            queries = torch.randn(query_heads, head_dim, generator=generator)
            queries = torch.nn.functional.normalize(queries, dim=-1)
            keys = 0.35 * torch.randn(
                chunks, chunk_tokens, kv_heads, head_dim, generator=generator
            )
            evidence = example % chunks
            for key_head in range(kv_heads):
                start = key_head * (query_heads // kv_heads)
                stop = start + (query_heads // kv_heads)
                landmark = torch.nn.functional.normalize(
                    queries[start:stop].mean(dim=0), dim=0
                )
                keys[evidence, key_head, key_head] = 5.0 * landmark
                keys[evidence, key_head + kv_heads, key_head] = 4.0 * landmark
            # A coherent distractor mean makes mean pooling choose incorrectly.
            distractor = (evidence + 1) % chunks
            for key_head in range(kv_heads):
                start = key_head * (query_heads // kv_heads)
                stop = start + (query_heads // kv_heads)
                direction = torch.nn.functional.normalize(
                    queries[start:stop].mean(dim=0), dim=0
                )
                keys[distractor, :, key_head] += 0.7 * direction
            output.append(
                {
                    "dataset": "synthetic_qk",
                    "example_id": f"qk-{chunk_tokens}-{example}",
                    "chunk_tokens": chunk_tokens,
                    "queries": queries.to(device).unsqueeze(0),
                    "keys": keys.to(device),
                    "mask": torch.ones(
                        chunks, chunk_tokens, dtype=torch.bool, device=device
                    ),
                    "positives": torch.arange(chunks, device=device) == evidence,
                }
            )
    return output


def _method_scores(
    method: str,
    m: int,
    queries: torch.Tensor,
    keys: torch.Tensor,
    mask: torch.Tensor,
    *,
    function: str,
    head_reduction: str,
    seed: int,
    learned: tuple[NativeLandmarkSelector, torch.Tensor, torch.Tensor] | None = None,
    preselected: list[list[int]] | None = None,
) -> tuple[torch.Tensor, list[list[int]] | None, float]:
    started = time.perf_counter()
    indices = None
    if method == "full_k":
        compact, compact_mask = keys, mask
    elif method == "mean":
        compact = masked_mean_keys(keys, mask)
        compact_mask = torch.ones(keys.shape[0], 1, dtype=torch.bool, device=keys.device)
    else:
        if preselected is not None:
            indices = [row[: min(m, len(row))] for row in preselected]
        elif method == "last":
            indices = last_token_indices(mask, m)
        elif method == "random":
            indices = random_token_indices(
                mask, m, generator=torch.Generator().manual_seed(seed)
            )
        elif method == "farthest":
            indices = farthest_first_indices(keys, mask, m)
        elif method == "greedy_oracle":
            indices = greedy_qk_landmarks(
                queries,
                keys,
                mask,
                m,
                function=function,
                top_r=4,
                head_reduction=head_reduction,
            )
        elif method == "learned":
            if learned is None:
                raise ValueError("Learned controller requires a fitted selector.")
            selector, mean, scale = learned
            features = (landmark_features(keys, mask) - mean) / scale
            with torch.no_grad():
                logits = selector(features, mask)
            indices = []
            for row, row_mask in zip(logits, mask):
                count = min(m, int(row_mask.sum().item()))
                indices.append(row.topk(count).indices.sort().values.tolist())
        else:
            raise ValueError(method)
        compact, compact_mask = gather_landmarks(keys, indices)
    if keys.device.type == "cuda":
        torch.cuda.synchronize(keys.device)
    compression_ms = 1000 * (time.perf_counter() - started)
    scores = _score_compact(
        queries,
        compact,
        compact_mask,
        function=function,
        head_reduction=head_reduction,
    )
    return scores, indices, compression_ms


def _evaluate_cases(
    cases: list[dict],
    *,
    split: str,
    function: str,
    head_reduction: str,
    budget: int,
    seeds: tuple[int, ...],
    learned_by_seed: dict[int, tuple[NativeLandmarkSelector, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[list[dict], list[dict]]:
    rows, audits = [], []
    for case_index, case in enumerate(cases, start=1):
        if "queries" in case:
            queries, keys, mask, positives = (
                case["queries"],
                case["keys"],
                case["mask"],
                case["positives"],
            )
            dataset, example_id = case["dataset"], case["example_id"]
            chunk_tokens = int(case["chunk_tokens"])
            source_tokens = int(mask.sum().item())
        else:
            device = next(iter(learned_by_seed.values()))[0].network[0].weight.device if learned_by_seed else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            queries, keys, mask, positives = _case_tensors(case, device)
            dataset, example_id = case["dataset"], case["example_id"]
            chunk_tokens = int(mask.sum(dim=1).float().mean().item())
            source_tokens = int(case["source_tokens"])
        teacher = _score_compact(
            queries,
            keys,
            mask,
            function=function,
            head_reduction=head_reduction,
        )
        method_specs = [("full_k", 0, 0), ("mean", 1, 0)]
        for m in M_VALUES:
            method_specs.extend(
                (method, m, seed)
                for method in ("last", "farthest", "greedy_oracle")
                for seed in (0,)
            )
            method_specs.extend(("random", m, seed) for seed in seeds)
            if learned_by_seed:
                method_specs.extend(("learned", m, seed) for seed in seeds)
        index_cache = {
            ("last", 0): last_token_indices(mask, max(M_VALUES)),
            ("farthest", 0): farthest_first_indices(keys, mask, max(M_VALUES)),
            ("greedy_oracle", 0): greedy_qk_landmarks(
                queries,
                keys,
                mask,
                max(M_VALUES),
                function=function,
                top_r=4,
                head_reduction=head_reduction,
            ),
        }
        for seed in seeds:
            index_cache[("random", seed)] = random_token_indices(
                mask,
                max(M_VALUES),
                generator=torch.Generator().manual_seed(seed + case_index * 1009),
            )
            if learned_by_seed:
                selector, mean, scale = learned_by_seed[seed]
                normalized = (landmark_features(keys, mask) - mean) / scale
                with torch.no_grad():
                    logits = selector(normalized, mask)
                index_cache[("learned", seed)] = [
                    row.topk(min(max(M_VALUES), int(row_mask.sum().item())))
                    .indices.sort()
                    .values.tolist()
                    for row, row_mask in zip(logits, mask)
                ]
        mean_selected = None
        for method, m, seed in method_specs:
            learned = learned_by_seed.get(seed) if learned_by_seed else None
            scores, indices, compression_ms = _method_scores(
                method,
                m,
                queries,
                keys,
                mask,
                function=function,
                head_reduction=head_reduction,
                seed=seed + case_index * 1009,
                learned=learned,
                preselected=index_cache.get((method, seed)),
            )
            preservation = response_metrics(teacher, scores)
            selected = _selection(scores, budget)
            route = routing_metrics(selected, positives, budget=budget)
            if method == "mean":
                mean_selected = selected
            landmarks = (
                float(mask.sum(dim=1).float().mean().item())
                if method == "full_k"
                else (1 if method == "mean" else m)
            )
            selected_tokens = sum(int(mask[index].sum().item()) for index in selected)
            evidence_tokens = int(mask[positives].sum().item())
            row = {
                "split": split,
                "dataset": dataset,
                "example_id": example_id,
                "method": method,
                "m": m,
                "seed": seed,
                "teacher_function": function,
                "head_reduction": head_reduction,
                "candidate_chunks": int(keys.shape[0]),
                "chunk_tokens": chunk_tokens,
                "source_tokens": source_tokens,
                "requested_token_budget": budget * chunk_tokens,
                "materialized_kv_tokens": selected_tokens,
                "active_memory_fraction": selected_tokens / max(source_tokens, 1),
                "evidence_normalized_overhead": selected_tokens / max(evidence_tokens, 1),
                "native_dots": int(keys.shape[0] * max(landmarks, 1) * queries.shape[1]),
                "compression_ms": compression_ms,
                **route,
                **{key: value for key, value in asdict(preservation).items() if key != "topk_overlap"},
                **{f"teacher_top{k}_overlap": value for k, value in preservation.topk_overlap.items()},
                "selected_chunks": " ".join(map(str, selected)),
                "landmark_indices": json.dumps(indices) if indices is not None else "",
            }
            rows.append(row)
            if mean_selected is not None and method not in {"full_k", "mean"}:
                audits.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "example_id": example_id,
                        "method": method,
                        "m": m,
                        "seed": seed,
                        "mean_selected": " ".join(map(str, mean_selected)),
                        "method_selected": " ".join(map(str, selected)),
                        "selection_changed": float(set(selected) != set(mean_selected)),
                        "delta_evidence_recall": route["evidence_recall"]
                        - routing_metrics(mean_selected, positives, budget=budget)[
                            "evidence_recall"
                        ],
                    }
                )
        print(f"[{split} {case_index}/{len(cases)}] {dataset} {example_id}", flush=True)
    return rows, audits


def _oracle_labels(
    features: list[dict],
    *,
    function: str,
    head_reduction: str,
    m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_rows, labels = [], []
    for feature in features:
        queries, keys, mask, _ = _case_tensors(feature, device)
        indices = greedy_qk_landmarks(
            queries,
            keys,
            mask,
            m,
            function=function,
            top_r=4,
            head_reduction=head_reduction,
        )
        key_features = landmark_features(keys, mask)
        targets = torch.zeros_like(mask, dtype=torch.float32)
        for chunk, row in enumerate(indices):
            targets[chunk, row] = 1.0
        feature_rows.append(key_features[mask].cpu())
        labels.append(targets[mask].cpu())
    values = torch.cat(feature_rows)
    targets = torch.cat(labels)
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (values - mean) / scale, targets, (mean, scale)


def _fit_selectors(
    validation: list[dict],
    *,
    function: str,
    head_reduction: str,
    device: torch.device,
    output_dir: Path,
) -> dict[int, tuple[NativeLandmarkSelector, torch.Tensor, torch.Tensor]]:
    values, targets, stats = _oracle_labels(
        validation,
        function=function,
        head_reduction=head_reduction,
        m=max(M_VALUES),
        device=device,
    )
    mean, scale = (value.to(device) for value in stats)
    values, targets = values.to(device), targets.to(device)
    positive_weight = ((targets.numel() - targets.sum()) / targets.sum().clamp_min(1)).clamp(max=20)
    fitted = {}
    checkpoint_dir = output_dir / "selector_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        selector = NativeLandmarkSelector(hidden_width=32).to(device)
        optimizer = torch.optim.AdamW(selector.parameters(), lr=3e-3, weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(seed)
        for _ in range(400):
            indices = torch.randint(
                len(values),
                (min(4096, len(values)),),
                generator=generator,
                device=device,
            )
            logits = selector.network(values.index_select(0, indices)).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets.index_select(0, indices),
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        selector.eval()
        fitted[seed] = (selector, mean, scale)
        torch.save(
            {
                "state_dict": selector.state_dict(),
                "feature_mean": mean.cpu(),
                "feature_scale": scale.cpu(),
                "seed": seed,
                "parameter_count": sum(p.numel() for p in selector.parameters()),
                "teacher_function": function,
                "head_reduction": head_reduction,
                "training_examples": len(validation),
            },
            checkpoint_dir / f"native_landmark_selector_seed{seed}.pt",
        )
    return fitted


def _aggregate(rows: list[dict]) -> list[dict]:
    dimensions = ("split", "dataset", "method", "m")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["dataset"], row["method"], row["m"])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        record = dict(zip(dimensions, key))
        record["rows"] = len(group)
        record["identities"] = len({row["example_id"] for row in group})
        for metric in (
            "evidence_recall",
            "evidence_precision",
            "any_evidence",
            "chain_completion",
            "exact_identity",
            "mrr",
            "mae",
            "rmse",
            "spearman",
            "kendall",
            "kl",
            "teacher_top1_overlap",
            "teacher_top2_overlap",
            "teacher_top4_overlap",
            "teacher_top8_overlap",
            "active_memory_fraction",
            "evidence_normalized_overhead",
            "native_dots",
            "compression_ms",
            "materialized_kv_tokens",
        ):
            record[metric] = _mean(group, metric)
        output.append(record)
    return output


def _paired_effects(rows: list[dict], seed: int) -> list[dict]:
    test = [row for row in rows if row["split"] == "test"]
    output = []
    for dataset in sorted({row["dataset"] for row in test}):
        baseline = {
            row["example_id"]: row
            for row in test
            if row["dataset"] == dataset and row["method"] == "mean"
        }
        methods = sorted({(row["method"], row["m"]) for row in test if row["dataset"] == dataset})
        for method, m in methods:
            if method in {"full_k", "mean"}:
                continue
            candidates = defaultdict(list)
            for row in test:
                if row["dataset"] == dataset and row["method"] == method and row["m"] == m:
                    candidates[row["example_id"]].append(row)
            for metric in ("evidence_recall", "chain_completion", "teacher_top4_overlap"):
                differences = [
                    _mean(values, metric) - float(baseline[identity][metric])
                    for identity, values in candidates.items()
                    if identity in baseline
                ]
                low, high = _bootstrap(differences, seed + m)
                output.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "m": m,
                        "metric": metric,
                        "pairs": len(differences),
                        "mean_delta": sum(differences) / max(len(differences), 1),
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": sum(value > 0 for value in differences),
                        "losses": sum(value < 0 for value in differences),
                        "ties": sum(value == 0 for value in differences),
                    }
                )
    return output


def _inherited_rows(root: Path) -> tuple[list[dict], dict]:
    source = root / "docs/papers/shared/results/paper2_6_hybrid_pra/per_example.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    output = []
    for row in rows:
        if row["condition"] not in HISTORICAL_CONDITIONS:
            continue
        output.append(
            {
                "dataset": row["dataset"],
                "example_id": row["example_id"],
                "condition": row["condition"],
                "evidence_recall": float(row["evidence_recall"]),
                "precision": float(row["precision"]),
                "mrr": float(row["mrr"]),
                "chain_completion": float(row["path_completion"]),
                "requested_chunks": float(row["requested_chunks"]),
                "source": "Paper 2.6 frozen per-example row",
            }
        )
    # Re-aggregation is the G0 parity test; raw rows are not rerouted or relabeled.
    tracked = root / "docs/papers/shared/results/paper2_6_hybrid_pra/summary.csv"
    with tracked.open(newline="", encoding="utf-8") as stream:
        expected = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    mismatches = []
    for row in expected:
        group = [
            item
            for item in output
            if item["dataset"] == row["dataset"] and item["condition"] == row["condition"]
        ]
        if not group:
            continue
        for metric, inherited_key in (
            ("evidence_recall", "evidence_recall"),
            ("precision", "precision"),
            ("mrr", "mrr"),
            ("chain_completion", "path_completion"),
        ):
            difference = abs(_mean(group, metric) - float(row[inherited_key]))
            if difference > 1e-12:
                mismatches.append((row["dataset"], row["condition"], metric, difference))
    return output, {"passed": not mismatches, "mismatches": mismatches, "rows": len(output)}


def _gate_decisions(
    synthetic: list[dict],
    natural: list[dict],
    paired: list[dict],
    g0: dict,
) -> dict:
    def metric(rows, method, metric, *, dataset=None, m=None):
        selected = [
            row
            for row in rows
            if row["method"] == method
            and (dataset is None or row["dataset"] == dataset)
            and (m is None or row["m"] == m)
        ]
        return _mean(selected, metric)

    synthetic_teacher = metric(synthetic, "full_k", "evidence_recall")
    synthetic_mean = metric(synthetic, "mean", "evidence_recall")
    natural_teacher_gain = {
        dataset: metric(natural, "full_k", "evidence_recall", dataset=dataset)
        - metric(natural, "mean", "evidence_recall", dataset=dataset)
        for dataset in ("hotpotqa", "qasper")
    }
    oracle_candidates = []
    for m in M_VALUES:
        oracle_candidates.append(
            {
                "m": m,
                "top4_gain": metric(natural, "greedy_oracle", "teacher_top4_overlap", m=m)
                - metric(natural, "mean", "teacher_top4_overlap"),
                "recall_gain": max(
                    metric(natural, "greedy_oracle", "evidence_recall", dataset=dataset, m=m)
                    - metric(natural, "mean", "evidence_recall", dataset=dataset)
                    for dataset in ("hotpotqa", "qasper")
                ),
            }
        )
    best_oracle = max(oracle_candidates, key=lambda row: (row["top4_gain"], row["recall_gain"]))
    g2 = best_oracle["top4_gain"] > 0 and best_oracle["recall_gain"] > 0
    learned_rows = [row for row in natural if row["method"] == "learned"]
    if learned_rows:
        m = int(best_oracle["m"])
        mean_overlap = metric(natural, "mean", "teacher_top4_overlap")
        oracle_overlap = metric(natural, "greedy_oracle", "teacher_top4_overlap", m=m)
        learned_overlap = metric(natural, "learned", "teacher_top4_overlap", m=m)
        recovery = (learned_overlap - mean_overlap) / max(oracle_overlap - mean_overlap, 1e-12)
        recall_deltas = {
            dataset: metric(natural, "learned", "evidence_recall", dataset=dataset, m=m)
            - metric(natural, "mean", "evidence_recall", dataset=dataset)
            for dataset in ("hotpotqa", "qasper")
        }
        g3 = recovery >= 0.8 and max(recall_deltas.values()) > 0 and min(recall_deltas.values()) >= -0.02
    else:
        recovery, recall_deltas, g3 = None, {}, False
    return {
        "G0_inherited_parity": g0,
        "G1_teacher_sanity": {
            "passed": synthetic_teacher > synthetic_mean and max(natural_teacher_gain.values()) > 0,
            "synthetic_full_k_recall": synthetic_teacher,
            "synthetic_mean_recall": synthetic_mean,
            "natural_full_k_minus_mean_recall": natural_teacher_gain,
        },
        "G2_oracle_feasibility": {
            "passed": g2,
            "candidates": oracle_candidates,
            "selected": best_oracle,
        },
        "G3_learned_selector": {
            "run": bool(learned_rows),
            "passed": g3,
            "oracle_gain_recovered": recovery,
            "natural_recall_deltas": recall_deltas,
        },
        "G4_synthetic_slots": {"run": False, "reason": "Requires G3 pass."},
        "G5_streaming_memory": {"run": False, "reason": "Requires offline compressor pass."},
    }


def _plots(summary: list[dict], output_dir: Path) -> None:
    test = [row for row in summary if row["split"] == "test"]
    methods = ("mean", "last", "random", "farthest", "greedy_oracle", "learned")
    colors = {
        "mean": "#6c757d",
        "last": "#d1495b",
        "random": "#edae49",
        "farthest": "#457b9d",
        "greedy_oracle": "#2a9d8f",
        "learned": "#6a4c93",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("teacher_top4_overlap", "evidence_recall"),
        ("Full-K teacher top-4 preservation", "Natural evidence recall"),
    ):
        for method in methods:
            values = [row for row in test if row["method"] == method]
            if not values:
                continue
            if method == "mean":
                axis.axhline(values[0][metric], color=colors[method], label="mean")
            else:
                axis.plot(
                    [row["m"] for row in values],
                    [row[metric] for row in values],
                    marker="o",
                    color=colors[method],
                    label=method.replace("_", " "),
                )
        axis.set_xlabel("Native landmarks per chunk (m)")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(title)
        axis.set_xticks(M_VALUES)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"qk_preservation_retrieval.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        for method in methods:
            values = [row for row in test if row["dataset"] == dataset and row["method"] == method]
            if not values:
                continue
            if method == "mean":
                axis.axhline(values[0]["evidence_recall"], color=colors[method], label="mean")
            else:
                axis.plot(
                    [row["m"] for row in values],
                    [row["evidence_recall"] for row in values],
                    marker="o",
                    color=colors[method],
                    label=method.replace("_", " "),
                )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Native landmarks per chunk (m)")
        axis.set_ylabel("Evidence recall at four chunks")
        axis.set_xticks(M_VALUES)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"natural_retrieval_by_dataset.{suffix}", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = {
        "validation": torch.load(args.validation_features, map_location="cpu", weights_only=False),
        "test": torch.load(args.test_features, map_location="cpu", weights_only=False),
    }
    _project_native_queries(features, device)
    teacher_function, head_reduction, teacher_rows = _teacher_selection(
        features["validation"], device, args.budget
    )
    _write_csv(args.output_dir / "teacher_validation.csv", teacher_rows)

    synthetic_cases = _synthetic_cases(args.seed, args.synthetic_examples, device)
    synthetic_rows, synthetic_audits = _evaluate_cases(
        synthetic_cases,
        split="synthetic",
        function=teacher_function,
        head_reduction=head_reduction,
        budget=args.budget,
        seeds=SEEDS,
    )

    # Evaluate no-training controllers first.  The learned stage is opened only
    # after the G2 decision is computed from these rows.
    natural_rows, natural_audits = _evaluate_cases(
        features["test"],
        split="test",
        function=teacher_function,
        head_reduction=head_reduction,
        budget=args.budget,
        seeds=SEEDS,
    )
    inherited, g0 = _inherited_rows(ROOT)
    preliminary_paired = _paired_effects(natural_rows, args.seed)
    preliminary_gates = _gate_decisions(
        synthetic_rows, natural_rows, preliminary_paired, g0
    )
    if preliminary_gates["G2_oracle_feasibility"]["passed"] and not args.skip_learned:
        fitted = _fit_selectors(
            features["validation"],
            function=teacher_function,
            head_reduction=head_reduction,
            device=device,
            output_dir=args.output_dir,
        )
        learned_rows, learned_audits = _evaluate_cases(
            features["test"],
            split="test",
            function=teacher_function,
            head_reduction=head_reduction,
            budget=args.budget,
            seeds=SEEDS,
            learned_by_seed=fitted,
        )
        natural_rows.extend(row for row in learned_rows if row["method"] == "learned")
        natural_audits.extend(
            row for row in learned_audits if row["method"] == "learned"
        )
    paired = _paired_effects(natural_rows, args.seed)
    gates = _gate_decisions(synthetic_rows, natural_rows, paired, g0)
    summary = _aggregate(synthetic_rows + natural_rows)
    _write_csv(args.output_dir / "synthetic_rows.csv", synthetic_rows)
    _write_csv(args.output_dir / "natural_rows.csv", natural_rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "paired_effects.csv", paired)
    _write_csv(
        args.output_dir / "changed_selection_audit.csv",
        synthetic_audits + natural_audits,
    )
    _write_csv(args.output_dir / "paper2_6_inherited_rows.csv", inherited)
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
        "materialization_budget_chunks": args.budget,
        "teacher_function": teacher_function,
        "head_reduction": head_reduction,
        "m_values": M_VALUES,
        "seeds": SEEDS,
        "backbone_frozen": True,
        "selector_parameter_count": sum(
            parameter.numel() for parameter in NativeLandmarkSelector().parameters()
        ),
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
        },
        "natural_examples": {
            split: {
                dataset: sum(row["dataset"] == dataset for row in split_features)
                for dataset in ("hotpotqa", "qasper")
            }
            for split, split_features in features.items()
        },
        "commands": {
            "validation_cache": (
                "python experiments/paper2_5_iterative_pra/"
                "precompute_native_qk_features.py --device cuda --split validation "
                "--offset 0 --examples 8 --output-dir "
                "docs/papers/shared/results/paper2_8_qk_compression"
            ),
            "study": (
                "python experiments/paper2_8_qk_compression/run_gated_study.py "
                "--device cuda"
            ),
        },
    }
    result = {"manifest": manifest, "gates": gates}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "gate_decisions.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    output_dir = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--synthetic-examples", type=int, default=20)
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--validation-features",
        type=Path,
        default=output_dir / "native_qk_features_validation.pt",
    )
    parser.add_argument(
        "--test-features",
        type=Path,
        default=output_dir / "native_qk_features_test.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
