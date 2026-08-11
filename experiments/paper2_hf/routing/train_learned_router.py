"""Train tiny evidence-supervised metrics over frozen Qwen routing features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common import recall_sparsity_curve
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_torch.hf import HFRoutingProjection


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def _load_features(directory: Path) -> dict[str, list[dict]]:
    return {
        split: torch.load(directory / f"router_features_{split}.pt", weights_only=False)
        for split in ("train", "validation", "test")
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _scores(model, feature: dict, query_strategy: str, device: torch.device):
    query = feature["queries"][query_strategy].to(device).unsqueeze(0)
    memory = feature["memory_gists"].to(device)
    if model is None:
        return F.normalize(query, dim=-1) @ F.normalize(memory, dim=-1).transpose(0, 1)
    return model.scores(query, memory)


@torch.no_grad()
def evaluate(model, features, query_strategy: str, device: torch.device, cost: dict) -> dict:
    if model is not None:
        model.eval()
    rows = []
    projection_elapsed = 0.0
    routing_elapsed = 0.0
    for feature in features:
        _sync(device)
        started = time.perf_counter()
        if model is None:
            scores = _scores(model, feature, query_strategy, device)[0]
        else:
            query = feature["queries"][query_strategy].to(device).unsqueeze(0)
            memory = feature["memory_gists"].to(device)
            projected_query = model.project_query(query)
            projected_memory = model.project_memory(memory)
            scores = (projected_query @ projected_memory.transpose(0, 1))[0]
        _sync(device)
        projection_elapsed += time.perf_counter() - started
        ranking = torch.argsort(scores, descending=True)
        _sync(device)
        routing_elapsed += time.perf_counter() - started
        positive = feature["positive_mask"].to(device)[ranking]
        ranks = torch.nonzero(positive, as_tuple=False).flatten() + 1
        best_rank = int(ranks.min().item()) if len(ranks) else None
        ranked_positions = feature["normalized_positions"].to(device)[ranking]
        ranking_ids = ranking.cpu().tolist()
        rows.append(
            {
                "dataset": feature["dataset"],
                "example_id": feature["example_id"],
                "candidate_chunks": len(ranking),
                "positive_chunks": int(feature["positive_mask"].sum().item()),
                "best_evidence_rank": best_rank,
                "recall_at_3": float(bool(positive[:3].any())),
                "recall_at_8": float(bool(positive[:8].any())),
                "recall_at_16": float(bool(positive[:16].any())),
                "all_evidence_recall_at_3": float(
                    int(positive[:3].sum().item()) == int(positive.sum().item())
                ),
                "target_coverage_at_3": float(positive[:3].sum().item())
                / max(int(positive.sum().item()), 1),
                "mrr": 1.0 / best_rank if best_rank else 0.0,
                "selected_fraction_at_3": min(3, len(ranking)) / max(len(ranking), 1),
                "mean_selected_normalized_position": float(ranked_positions[:3].mean().item()),
                "score_position_correlation": _correlation(
                    feature["normalized_positions"].tolist(), scores.float().cpu().tolist()
                ),
                "ranking": ranking_ids,
                "evidence_candidate_ids": torch.nonzero(
                    feature["positive_mask"], as_tuple=False
                ).flatten().tolist(),
                "ranked_chunk_token_counts": [
                    int(feature["chunk_spans"][index][1])
                    - int(feature["chunk_spans"][index][0])
                    for index in ranking_ids
                ],
            }
        )
    metrics = (
        "recall_at_3",
        "recall_at_8",
        "recall_at_16",
        "all_evidence_recall_at_3",
        "target_coverage_at_3",
        "mrr",
        "selected_fraction_at_3",
        "mean_selected_normalized_position",
        "score_position_correlation",
    )
    aggregates = {}
    for dataset in ("combined", "hotpotqa", "qasper"):
        selected = rows if dataset == "combined" else [row for row in rows if row["dataset"] == dataset]
        if not selected:
            aggregates[dataset] = {metric: None for metric in metrics}
            aggregates[dataset]["examples"] = 0
            continue
        aggregates[dataset] = {
            metric: statistics.fmean(
                float(row[metric]) for row in selected if row.get(metric) is not None
            )
            for metric in metrics
        }
        aggregates[dataset]["examples"] = len(selected)
    return {
        "rows": rows,
        "aggregates": aggregates,
        "recall_sparsity": recall_sparsity_curve(
            [row["ranking"] for row in rows],
            [set(row["evidence_candidate_ids"]) for row in rows],
            candidate_token_lengths=[row["ranked_chunk_token_counts"] for row in rows],
            require_complete_endpoint=True,
        ),
        "adapter_seconds_per_example": (
            projection_elapsed / max(len(features), 1) if model is not None else 0.0
        ),
        "routing_seconds_per_example": routing_elapsed / max(len(features), 1),
        "routing_index_bytes_mean": statistics.fmean(
            len(feature["positive_mask"])
            * (model.routing_width if model is not None else int(cost["feature_width"]))
            * 4
            for feature in features
        ),
        "detail_kv_bytes_mean": statistics.fmean(
            int(feature["source_tokens"])
            * 2
            * int(cost["native_kv_heads"])
            * int(cost["native_head_dim"])
            * int(cost["native_kv_dtype_bytes"])
            for feature in features
        ),
    }


def _training_mask(feature: dict, shuffled: bool, seed: int) -> torch.Tensor:
    mask = feature["positive_mask"].clone()
    if not shuffled:
        return mask
    generator = torch.Generator().manual_seed(seed + sum(map(ord, feature["example_id"])))
    return mask[torch.randperm(len(mask), generator=generator)]


def _rank(values: torch.Tensor, negative: torch.Tensor) -> list[int]:
    """Sort negative candidate indices by descending hardness values."""
    indices = torch.nonzero(negative, as_tuple=False).flatten()
    order = torch.argsort(values[indices], descending=True)
    return indices[order].tolist()


def select_training_candidates(
    feature: dict,
    query_strategy: str,
    *,
    policy: str,
    negatives_per_positive: int,
    seed: int,
    mined_scores: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Select a deterministic mixture of lexical, routing, positional, and random negatives."""
    positive = (
        feature["positive_mask"].bool()
        if positive_mask is None
        else positive_mask.cpu().bool()
    )
    if policy == "exhaustive":
        return torch.ones_like(positive), {"exhaustive": int((~positive).sum())}
    if policy not in {"mixed", "mined"}:
        raise ValueError(f"Unsupported negative policy: {policy}")
    if "lexical_scores" not in feature:
        raise ValueError("Frozen features must be enriched with lexical_scores.")

    negative = ~positive
    query = feature["queries"][query_strategy].float().unsqueeze(0)
    memory = feature["memory_gists"].float()
    baseline = (F.normalize(query, dim=-1) @ F.normalize(memory, dim=-1).T)[0]
    positive_position = feature["normalized_positions"][positive].mean()
    position_hardness = -(feature["normalized_positions"] - positive_position).abs()
    generator = torch.Generator().manual_seed(
        seed + sum(map(ord, str(feature["example_id"])))
    )
    random_hardness = torch.rand(len(positive), generator=generator)
    sources = []
    if policy == "mined":
        if mined_scores is None:
            raise ValueError("Mined-negative selection requires learned-router scores.")
        sources.append(("learned_false_positive", _rank(mined_scores.cpu(), negative)))
    sources.extend(
        [
            ("zero_shot_false_positive", _rank(baseline, negative)),
            ("lexical", _rank(feature["lexical_scores"], negative)),
            ("position_matched", _rank(position_hardness, negative)),
            ("random", _rank(random_hardness, negative)),
        ]
    )
    target = min(
        int(negative.sum()),
        max(int(positive.sum()) * int(negatives_per_positive), 1),
    )
    selected: set[int] = set()
    composition = {name: 0 for name, _ in sources}
    cursors = [0] * len(sources)
    while len(selected) < target:
        progressed = False
        for source_index, (name, ranked) in enumerate(sources):
            while cursors[source_index] < len(ranked):
                candidate = int(ranked[cursors[source_index]])
                cursors[source_index] += 1
                if candidate in selected:
                    continue
                selected.add(candidate)
                composition[name] += 1
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break
    mask = positive.clone()
    if selected:
        mask[torch.tensor(sorted(selected), dtype=torch.long)] = True
    return mask, composition


def _loss(
    model,
    feature,
    query_strategy,
    device,
    temperature,
    shuffled,
    seed,
    objective,
    margin,
    negative_policy,
    negatives_per_positive,
    mined_scores=None,
):
    scores = _scores(model, feature, query_strategy, device)[0]
    positive = _training_mask(feature, shuffled, seed).to(device)
    candidate, composition = select_training_candidates(
        feature,
        query_strategy,
        policy=negative_policy,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
        mined_scores=mined_scores,
        positive_mask=positive,
    )
    candidate = candidate.to(device)
    negative = candidate & ~positive
    if objective == "contrastive":
        scaled = scores / temperature
        loss = torch.logsumexp(scaled[candidate], dim=0) - torch.logsumexp(
            scaled[positive & candidate], dim=0
        )
    elif objective == "margin":
        loss = F.relu(
            margin - scores[positive & candidate, None] + scores[None, negative]
        ).mean()
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    return loss, composition


def _domain(features: list[dict], domain: str) -> list[dict]:
    return features if domain == "joint" else [row for row in features if row["dataset"] == domain]


def train_one(
    *,
    features,
    architecture,
    routing_width,
    query_strategy,
    train_domain,
    seed,
    device,
    steps,
    learning_rate,
    temperature,
    objective,
    margin,
    negative_policy,
    negatives_per_positive,
    hard_negative_steps,
    shuffled_labels,
    checkpoint_dir,
    cost,
):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    input_width = int(features["train"][0]["memory_gists"].shape[1])
    model = HFRoutingProjection(input_width, routing_width, architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_rows = _domain(features["train"], train_domain)
    validation_rows = _domain(features["validation"], train_domain)
    best_score = float("-inf")
    best_state = None
    best_step = 0
    losses = []
    composition_totals = defaultdict(int)
    completed_steps = 0
    started = time.perf_counter()

    def optimize(round_steps, policy, mined_by_example=None):
        nonlocal best_score, best_state, best_step, completed_steps
        stale = 0
        for local_step in range(1, round_steps + 1):
            completed_steps += 1
            model.train()
            feature = train_rows[(completed_steps - 1) % len(train_rows)]
            optimizer.zero_grad(set_to_none=True)
            mined = (
                mined_by_example.get((feature["dataset"], feature["example_id"]))
                if mined_by_example is not None
                else None
            )
            loss, composition = _loss(
                model,
                feature,
                query_strategy,
                device,
                temperature,
                shuffled_labels,
                seed + completed_steps,
                objective,
                margin,
                policy,
                negatives_per_positive,
                mined,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, count in composition.items():
                composition_totals[name] += count
            if local_step % 25 != 0 and local_step != round_steps:
                continue
            validation = evaluate(model, validation_rows, query_strategy, device, cost)
            score = (
                validation["aggregates"]["combined"]["recall_at_3"]
                + 0.25 * validation["aggregates"]["combined"]["mrr"]
            )
            if score > best_score + 1e-9:
                best_score = score
                best_step = completed_steps
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break

    initial_policy = "mixed" if negative_policy == "mined" else negative_policy
    optimize(steps, initial_policy)
    mined_rounds = 0
    if negative_policy == "mined" and hard_negative_steps > 0:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            mined_by_example = {
                (feature["dataset"], feature["example_id"]): _scores(
                    model, feature, query_strategy, device
                )[0].float().cpu()
                for feature in train_rows
            }
        optimize(hard_negative_steps, "mined", mined_by_example)
        mined_rounds = 1
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    test = evaluate(model, features["test"], query_strategy, device, cost)
    validation = evaluate(model, validation_rows, query_strategy, device, cost)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / (
        f"{architecture}_d{routing_width}_{query_strategy}_{train_domain}_seed{seed}"
        f"_{objective}_{negative_policy}"
        f"{'_shuffled' if shuffled_labels else ''}.pt"
    )
    torch.save(
        {
            "state_dict": best_state,
            "input_width": input_width,
            "routing_width": routing_width,
            "architecture": architecture,
            "query_strategy": query_strategy,
            "objective": objective,
            "negative_policy": negative_policy,
        },
        checkpoint,
    )
    return {
        "architecture": architecture,
        "routing_width": routing_width,
        "query_strategy": query_strategy,
        "train_domain": train_domain,
        "seed": seed,
        "shuffled_labels": shuffled_labels,
        "objective": objective,
        "negative_policy": negative_policy,
        "negatives_per_positive": negatives_per_positive,
        "negative_composition": dict(composition_totals),
        "hard_negative_rounds": mined_rounds,
        "hard_negative_steps": hard_negative_steps if mined_rounds else 0,
        "temperature": temperature,
        "margin": margin,
        "learning_rate": learning_rate,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "best_step": best_step,
        "training_seconds": training_seconds,
        "mean_training_loss": statistics.fmean(losses),
        "adapter_parameters": model.parameter_count,
        "adapter_bytes_fp32": model.parameter_count * 4,
        "routing_vector_bytes_fp32": routing_width * 4,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "validation": validation,
        "test": test,
    }


def _condition_aggregates(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs:
        key = (
            run["architecture"],
            run["routing_width"],
            run["query_strategy"],
            run["train_domain"],
            run["shuffled_labels"],
            run["objective"],
            run["negative_policy"],
        )
        grouped[key].append(run)
    output = []
    for key, values in sorted(grouped.items()):
        record = {
            "architecture": key[0],
            "routing_width": key[1],
            "query_strategy": key[2],
            "train_domain": key[3],
            "shuffled_labels": key[4],
            "objective": key[5],
            "negative_policy": key[6],
            "seeds": len(values),
            "adapter_parameters": values[0]["adapter_parameters"],
        }
        for dataset in ("combined", "hotpotqa", "qasper"):
            for metric in (
                "recall_at_3",
                "recall_at_8",
                "recall_at_16",
                "all_evidence_recall_at_3",
                "target_coverage_at_3",
                "mrr",
                "selected_fraction_at_3",
                "mean_selected_normalized_position",
                "score_position_correlation",
            ):
                samples = [run["test"]["aggregates"][dataset][metric] for run in values]
                record[f"{dataset}_{metric}_mean"] = statistics.fmean(samples)
                record[f"{dataset}_{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
                record[f"{dataset}_{metric}_ci95"] = (
                    2.776 * record[f"{dataset}_{metric}_std"] / math.sqrt(len(samples))
                    if len(samples) > 1
                    else 0.0
                )
        record["adapter_seconds_per_example"] = statistics.fmean(
            run["test"]["adapter_seconds_per_example"] for run in values
        )
        record["routing_seconds_per_example"] = statistics.fmean(
            run["test"]["routing_seconds_per_example"] for run in values
        )
        record["routing_index_bytes_mean"] = statistics.fmean(
            run["test"]["routing_index_bytes_mean"] for run in values
        )
        record["detail_kv_bytes_mean"] = statistics.fmean(
            run["test"]["detail_kv_bytes_mean"] for run in values
        )
        record["routing_index_fraction_of_detail_kv"] = (
            record["routing_index_bytes_mean"] / record["detail_kv_bytes_mean"]
        )
        output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    device = torch.device(args.device)
    features = _load_features(args.feature_dir)
    manifest = json.loads(
        (args.feature_dir / "feature_dataset_manifest.json").read_text(encoding="utf-8")
    )
    cost = {
        key: manifest[key]
        for key in (
            "feature_width",
            "native_kv_heads",
            "native_head_dim",
            "native_kv_dtype_bytes",
        )
    }
    runs = []
    for architecture in args.architectures:
        for routing_width in args.routing_widths:
            for query_strategy in args.query_strategies:
                for train_domain in args.train_domains:
                    for seed in args.seeds:
                        result = train_one(
                            features=features,
                            architecture=architecture,
                            routing_width=routing_width,
                            query_strategy=query_strategy,
                            train_domain=train_domain,
                            seed=seed,
                            device=device,
                            steps=args.steps,
                            learning_rate=args.learning_rate,
                            temperature=args.temperature,
                            objective=args.objective,
                            margin=args.margin,
                            negative_policy=args.negative_policy,
                            negatives_per_positive=args.negatives_per_positive,
                            hard_negative_steps=args.hard_negative_steps,
                            shuffled_labels=args.shuffled_labels,
                            checkpoint_dir=args.feature_dir / "checkpoints",
                            cost=cost,
                        )
                        runs.append(result)
                        print(
                            f"{architecture} d={routing_width} {query_strategy} "
                            f"{train_domain} seed={seed}: "
                            f"R3={result['test']['aggregates']['combined']['recall_at_3']:.3f}",
                            flush=True,
                        )
    baselines = {
        query: evaluate(None, features["test"], query, device, cost)
        for query in args.query_strategies
    }
    aggregates = _condition_aggregates(runs)
    artifact = {
        "runtime": runtime_metadata(),
        "feature_manifest": "feature_dataset_manifest.json",
        "device": str(device),
        "seeds": list(args.seeds),
        "architectures": list(args.architectures),
        "routing_widths": list(args.routing_widths),
        "query_strategies": list(args.query_strategies),
        "train_domains": list(args.train_domains),
        "shuffled_labels": args.shuffled_labels,
        "objective": args.objective,
        "temperature": args.temperature,
        "margin": args.margin,
        "negative_policy": args.negative_policy,
        "negatives_per_positive": args.negatives_per_positive,
        "hard_negative_steps": args.hard_negative_steps,
        "optimizer": "AdamW",
        "cost_model": cost,
        "baselines": baselines,
        "runs": runs,
        "aggregates": aggregates,
    }
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    (args.feature_dir / f"{args.stem}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.feature_dir / f"{args.stem}.csv", aggregates)
    return artifact


def _tuple(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--architectures", default="shared_linear,asymmetric_linear")
    parser.add_argument("--routing-widths", default="64,128")
    parser.add_argument("--query-strategies", default="last,question_exp_h2.0")
    parser.add_argument("--train-domains", default="joint")
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--objective", choices=("contrastive", "margin"), default="contrastive")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--negative-policy", choices=("exhaustive", "mixed", "mined"), default="exhaustive"
    )
    parser.add_argument("--negatives-per-positive", type=int, default=7)
    parser.add_argument("--hard-negative-steps", type=int, default=200)
    parser.add_argument("--shuffled-labels", action="store_true")
    parser.add_argument("--stem", default="linear_adapter_results")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "papers"
            / "shared"
            / "results"
            / "paper2_hf"
            / "routing"
            / "learned_adapter"
        ),
    )
    args = parser.parse_args()
    args.architectures = _tuple(args.architectures, str)
    args.routing_widths = _tuple(args.routing_widths, int)
    args.query_strategies = _tuple(args.query_strategies, str)
    args.train_domains = _tuple(args.train_domains, str)
    args.seeds = _tuple(args.seeds, int)
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"runs": len(result["runs"])}, indent=2))
