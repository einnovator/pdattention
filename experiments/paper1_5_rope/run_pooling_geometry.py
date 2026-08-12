"""Compare pooled RoPE attention geometry with semantic evidence retrieval."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_native_kv_benchmark as native  # noqa: E402

from common.recall_sparsity import DEFAULT_FRACTIONS, recall_sparsity_curve  # noqa: E402
from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    TIERS,
    environment_metadata,
    refresh_manifest,
    set_seed,
    write_csv,
    write_json,
)
from experiments.paper1_5_rope.learned_routing import (  # noqa: E402
    AsymmetricLinearRouter,
    cosine_scores,
    materialize_native_payload,
    native_attention_output,
    rank_candidate_ids,
    train_router,
    trainable_parameter_count,
)
from experiments.paper1_5_rope.pooling_geometry import (  # noqa: E402
    centered_rope_subgists,
    native_token_chunk_score,
    pearson_correlation,
    post_rope_mean,
    pre_rope_mean,
    qk_gist_score,
    spearman_correlation,
    topk_overlap,
)
from experiments.paper1_5_rope.run_learned_routing_geometry import (  # noqa: E402
    _attention_input,
    _split_heads,
    _tensor_digest,
)
from experiments.paper1_5_rope.run_qa_validation import _base_settings  # noqa: E402
from experiments.paper1_5_rope.run_retrieval_geometry_gate import _load_source  # noqa: E402
from pra_torch.positions import RotaryPositionEncoding  # noqa: E402


OUTPUT = RESULTS / "pooling_geometry"
FIGURES = REPO / "docs" / "papers" / "shared" / "figures"
GIST_COUNTS = (1, 2, 4, 8)
METHODS = (
    "post_rope_mean",
    "pre_rope_mean",
    "centered_g1",
    "centered_g2",
    "centered_g4",
    "centered_g8",
    "hidden_cosine",
    "learned_projection",
)
LABELS = {
    "post_rope_mean": "Post-RoPE mean",
    "pre_rope_mean": "Pre-RoPE mean",
    "centered_g1": "Centered G=1",
    "centered_g2": "Centered G=2",
    "centered_g4": "Centered G=4",
    "centered_g8": "Centered G=8",
    "hidden_cosine": "Hidden cosine",
    "learned_projection": "Learned 32-D",
}
COLORS = {
    "post_rope_mean": "#9C3D38",
    "pre_rope_mean": "#C47A35",
    "centered_g1": "#7C6AA6",
    "centered_g2": "#6657A3",
    "centered_g4": "#4E78A0",
    "centered_g8": "#2F6F9F",
    "hidden_cosine": "#777777",
    "learned_projection": "#2E7D55",
}


@dataclass
class ExtractedExample:
    """Compact frozen representations and precomputed geometric scores."""

    example_id: str
    query_hidden: torch.Tensor
    chunk_hidden: torch.Tensor
    positive_mask: torch.Tensor
    candidate_ids: list[str]
    evidence_ids: set[str]
    candidate_lengths: list[int]
    normalized_positions: list[float]
    native_scores: torch.Tensor
    fixed_scores: dict[str, torch.Tensor]
    detail_kv_bytes: int
    payload_parity_error: float
    fractional_centers: int
    total_centers: int


def _extract_examples(model, tokenizer, samples, device: str) -> list[ExtractedExample]:
    """Capture frozen state and evaluate all parameter-free pooling methods."""

    model.eval().requires_grad_(False)
    attention = model.blocks[-1].attn
    rope = attention.position_encoding
    if not isinstance(rope, RotaryPositionEncoding):
        raise TypeError("The pooling geometry experiment requires a RoPE checkpoint")
    output = []
    with torch.no_grad():
        for sample in samples:
            candidate_ids = [str(reference.uri) for reference in sample.references]
            token_rows = [
                list(tokenizer.encode(str(reference.metadata.get("text", ""))))
                for reference in sample.references
            ]
            if any(not row for row in token_rows):
                raise ValueError(f"Example {sample.id} contains an empty reference")
            source_length = sum(len(row) for row in token_rows)
            question_ids = list(tokenizer.encode(sample.question))
            query_hidden_tokens = _attention_input(
                model, question_ids, position_offset=source_length, device=device
            )
            query_hidden = query_hidden_tokens[:, -1, :]
            query_pre = _split_heads(attention, attention.q_proj(query_hidden[:, None, :]))
            query_position = torch.tensor(
                [source_length + len(question_ids) - 1], device=query_hidden.device
            )
            query_post = rope.apply_rotary(query_pre, query_position)

            chunk_hidden = []
            native_scores = []
            fixed_scores = defaultdict(list)
            candidate_lengths = []
            normalized_positions = []
            payload_by_id = {}
            detail_kv_bytes = 0
            fractional_centers = 0
            total_centers = 0
            cursor = 0
            for identity, token_ids in zip(candidate_ids, token_rows, strict=True):
                end = cursor + len(token_ids)
                hidden = _attention_input(model, token_ids, position_offset=cursor, device=device)
                raw_key = _split_heads(attention, attention.k_proj(hidden))
                value = _split_heads(attention, attention.v_proj(hidden))
                positions = torch.arange(cursor, end, device=hidden.device)
                positioned_key = rope.apply_rotary(raw_key, positions)
                chunk_hidden.append(hidden.mean(dim=1))
                native_scores.append(native_token_chunk_score(query_post, positioned_key))
                fixed_scores["post_rope_mean"].append(
                    qk_gist_score(query_post, post_rope_mean(positioned_key))
                )
                fixed_scores["pre_rope_mean"].append(
                    qk_gist_score(query_pre, pre_rope_mean(raw_key))
                )
                for count in GIST_COUNTS:
                    centered, centers = centered_rope_subgists(raw_key, positions, count, rope)
                    fixed_scores[f"centered_g{count}"].append(
                        qk_gist_score(query_post, centered)
                    )
                    fractional_centers += int(((centers % 1).abs() > 0).sum().item())
                    total_centers += centers.numel()
                payload_by_id[identity] = (positioned_key, value)
                detail_kv_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in (positioned_key, value)
                )
                candidate_lengths.append(len(token_ids))
                normalized_positions.append(
                    (cursor + end - 1) / 2.0 / max(source_length - 1, 1)
                )
                cursor = end

            fixed_ids = candidate_ids[: min(3, len(candidate_ids))]
            baseline_k, baseline_v = materialize_native_payload(payload_by_id, fixed_ids)
            baseline_output = native_attention_output(query_post, baseline_k, baseline_v)
            parity_error = 0.0
            for _method in METHODS:
                candidate_k, candidate_v = materialize_native_payload(payload_by_id, fixed_ids)
                candidate_output = native_attention_output(query_post, candidate_k, candidate_v)
                parity_error = max(
                    parity_error,
                    float((candidate_k - baseline_k).abs().max().cpu()),
                    float((candidate_v - baseline_v).abs().max().cpu()),
                    float((candidate_output - baseline_output).abs().max().cpu()),
                )
            evidence = set(str(value) for value in sample.target_reference_uris)
            if not evidence or not evidence <= set(candidate_ids):
                raise ValueError(f"Example {sample.id} has invalid evidence identities")
            output.append(
                ExtractedExample(
                    example_id=str(sample.id),
                    query_hidden=query_hidden[0].float().cpu(),
                    chunk_hidden=torch.cat(chunk_hidden, dim=0).float().cpu(),
                    positive_mask=torch.tensor(
                        [identity in evidence for identity in candidate_ids], dtype=torch.bool
                    ),
                    candidate_ids=candidate_ids,
                    evidence_ids=evidence,
                    candidate_lengths=candidate_lengths,
                    normalized_positions=normalized_positions,
                    native_scores=torch.cat(native_scores).float().cpu(),
                    fixed_scores={
                        method: torch.cat(values).float().cpu()
                        for method, values in fixed_scores.items()
                    },
                    detail_kv_bytes=detail_kv_bytes,
                    payload_parity_error=parity_error,
                    fractional_centers=fractional_centers,
                    total_centers=total_centers,
                )
            )
    return output


def _stack_hidden(examples: list[ExtractedExample], device: str):
    return (
        torch.stack([row.query_hidden for row in examples]).to(device),
        torch.stack([row.chunk_hidden for row in examples]).to(device),
        torch.stack([row.positive_mask for row in examples]).to(device),
    )


def _method_scores(
    method: str,
    examples: list[ExtractedExample],
    learned: AsymmetricLinearRouter,
    device: str,
) -> torch.Tensor:
    if method in examples[0].fixed_scores:
        return torch.stack([row.fixed_scores[method] for row in examples])
    query_hidden, chunk_hidden, _ = _stack_hidden(examples, device)
    with torch.no_grad():
        if method == "hidden_cosine":
            return cosine_scores(query_hidden, chunk_hidden).cpu()
        if method == "learned_projection":
            return learned(query_hidden, chunk_hidden).cpu()
    raise ValueError(f"Unknown pooling method: {method}")


def _evaluate(scores: torch.Tensor, examples: list[ExtractedExample]) -> tuple[dict, list[list[str]]]:
    candidate_ids = [row.candidate_ids for row in examples]
    rankings = rank_candidate_ids(scores, candidate_ids)
    lengths_by_rank = []
    for ranking, row in zip(rankings, examples, strict=True):
        length_by_id = dict(zip(row.candidate_ids, row.candidate_lengths, strict=True))
        lengths_by_rank.append([length_by_id[identity] for identity in ranking])
    report = recall_sparsity_curve(
        rankings,
        [row.evidence_ids for row in examples],
        fractions=DEFAULT_FRACTIONS,
        fixed_k=(1, 3, 8, 16),
        candidate_token_lengths=lengths_by_rank,
        require_complete_endpoint=True,
    )
    reciprocal_ranks = []
    for ranking, row in zip(rankings, examples, strict=True):
        first = min(index for index, identity in enumerate(ranking, start=1) if identity in row.evidence_ids)
        reciprocal_ranks.append(1.0 / first)
    report["mrr"] = statistics.fmean(reciprocal_ranks)
    return report, rankings


def _geometry_metrics(scores: torch.Tensor, examples: list[ExtractedExample]) -> dict:
    spearman = []
    top1 = []
    top3 = []
    position = []
    for row_scores, row in zip(scores.tolist(), examples, strict=True):
        native_scores = row.native_scores.tolist()
        spearman.append(spearman_correlation(row_scores, native_scores))
        top1.append(topk_overlap(row_scores, native_scores, 1))
        top3.append(topk_overlap(row_scores, native_scores, 3))
        position.append(pearson_correlation(row_scores, row.normalized_positions))
    return {
        "qk_spearman": statistics.fmean(spearman),
        "qk_top1_agreement": statistics.fmean(top1),
        "qk_top3_overlap": statistics.fmean(top3),
        "position_correlation": statistics.fmean(position),
        "absolute_position_correlation": statistics.fmean(abs(value) for value in position),
    }


def _summary_row(dataset, tier, seed, method, report, geometry, examples, model, router):
    by_fraction = {float(row["fraction"]): row for row in report["curve"]}
    row = {
        "dataset": dataset,
        "model_tier": tier,
        "seed": seed,
        "method": method,
        "examples": report["examples"],
        "gist_count": int(method.rsplit("g", 1)[1]) if method.startswith("centered_g") else 1,
        "mrr": report["mrr"],
        "f80": report["inverse"]["f80"],
        "f90": report["inverse"]["f90"],
        "auc_0_30": report["auc_0_30"],
        "r_at_3": report["fixed_k"]["3"]["recall"],
        "r_at_8": report["fixed_k"]["8"]["recall"],
        "router_parameters": trainable_parameter_count(router),
        "router_backbone_percent": 100.0
        * trainable_parameter_count(router)
        / sum(parameter.numel() for parameter in model.parameters()),
        "native_detail_kv_bytes": statistics.fmean(row.detail_kv_bytes for row in examples),
        "payload_parity_max_abs_error": max(row.payload_parity_error for row in examples),
        "fractional_center_fraction": sum(row.fractional_centers for row in examples)
        / max(sum(row.total_centers for row in examples), 1),
        **geometry,
    }
    for fraction in (0.05, 0.10, 0.20, 0.30):
        point = by_fraction[fraction]
        suffix = int(100 * fraction)
        row[f"r_at_{suffix}pct"] = point["recall"]
        row[f"any_r_at_{suffix}pct"] = point["any_evidence_recall"]
        row[f"all_r_at_{suffix}pct"] = point["all_evidence_recall"]
        row[f"kv_at_{suffix}pct"] = point["selected_kv_token_fraction"]
    return row


METRICS = (
    "qk_spearman",
    "qk_top1_agreement",
    "qk_top3_overlap",
    "position_correlation",
    "absolute_position_correlation",
    "mrr",
    "f80",
    "f90",
    "auc_0_30",
    "r_at_3",
    "r_at_8",
    "r_at_5pct",
    "r_at_10pct",
    "r_at_20pct",
    "r_at_30pct",
    "any_r_at_10pct",
    "all_r_at_10pct",
)


def _summarize(values: list[dict], identity: dict) -> dict:
    result = {**identity, "sample_count": len(values)}
    for metric in METRICS:
        observed = [float(row[metric]) for row in values]
        mean = statistics.fmean(observed)
        std = statistics.stdev(observed) if len(observed) > 1 else 0.0
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
        result[f"{metric}_ci95"] = 2.776 * std / math.sqrt(len(observed)) if len(observed) == 5 else 1.96 * std / math.sqrt(len(observed))
    result["payload_parity_max_abs_error"] = max(
        float(row["payload_parity_max_abs_error"]) for row in values
    )
    result["fractional_center_fraction"] = statistics.fmean(
        float(row["fractional_center_fraction"]) for row in values
    )
    return result


def _aggregate(per_seed: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in per_seed:
        groups[(row["dataset"], row["model_tier"], row["method"])].append(row)
    return [
        _summarize(
            values,
            {"dataset": key[0], "model_tier": key[1], "method": key[2], "seed_count": len(values)},
        )
        for key, values in sorted(groups.items())
    ]


def _publication_summary(per_seed: list[dict]) -> list[dict]:
    by_method_seed = defaultdict(list)
    for row in per_seed:
        by_method_seed[(row["method"], row["seed"])].append(row)
    seed_means = []
    for (method, seed), values in sorted(by_method_seed.items()):
        seed_means.append(
            {
                "method": method,
                "seed": seed,
                **{
                    metric: statistics.fmean(float(row[metric]) for row in values)
                    for metric in METRICS
                },
                "payload_parity_max_abs_error": max(
                    float(row["payload_parity_max_abs_error"]) for row in values
                ),
                "fractional_center_fraction": statistics.fmean(
                    float(row["fractional_center_fraction"]) for row in values
                ),
            }
        )
    output = []
    for method in METHODS:
        values = [row for row in seed_means if row["method"] == method]
        output.append(_summarize(values, {"method": method, "seed_count": len(values)}))
    return output


def _plot(publication: list[dict]) -> None:
    by_method = {row["method"]: row for row in publication}
    x = list(range(len(METHODS)))
    labels = [LABELS[method] for method in METHODS]
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.errorbar(
        x,
        [by_method[m]["qk_spearman_mean"] for m in METHODS],
        yerr=[by_method[m]["qk_spearman_ci95"] for m in METHODS],
        fmt="none",
        ecolor="#444444",
        capsize=3,
    )
    axis.scatter(x, [by_method[m]["qk_spearman_mean"] for m in METHODS], c=[COLORS[m] for m in METHODS], s=58)
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylabel(r"Native-QK fidelity $\rho_{QK}$")
    axis.set_ylim(-0.2, 1.02)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "rope_pooling_qk_fidelity.pdf")
    figure.savefig(OUTPUT / "rope_pooling_qk_fidelity.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 5.1))
    for method in METHODS:
        row = by_method[method]
        axis.errorbar(
            row["qk_spearman_mean"],
            row["auc_0_30_mean"],
            xerr=row["qk_spearman_ci95"],
            yerr=row["auc_0_30_ci95"],
            marker="o",
            color=COLORS[method],
            capsize=3,
            label=LABELS[method],
        )
    axis.set_xlabel(r"Native-QK fidelity $\rho_{QK}$")
    axis.set_ylabel(r"Semantic evidence AUC$_{0:30}$")
    axis.legend(frameon=True, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURES / "rope_pooling_geometry_vs_semantics.pdf")
    figure.savefig(OUTPUT / "rope_pooling_geometry_vs_semantics.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar(
        x,
        [by_method[m]["absolute_position_correlation_mean"] for m in METHODS],
        yerr=[by_method[m]["absolute_position_correlation_ci95"] for m in METHODS],
        color=[COLORS[m] for m in METHODS],
        capsize=3,
    )
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylabel(r"Absolute position correlation $|\rho_{pos}|$")
    axis.set_ylim(0, 1.0)
    figure.tight_layout()
    figure.savefig(FIGURES / "rope_pooling_position_bias.pdf")
    figure.savefig(OUTPUT / "rope_pooling_position_bias.png", dpi=220)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    metadata = environment_metadata()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_seed = []
    curves = []
    rankings = []
    training = []
    for dataset in args.datasets:
        base = _base_settings(dataset, False, args.max_examples)
        _tokenizer, _training_module, modules = native.DATASET_PREPARERS[dataset](base)
        module = modules[args.split_count]
        samples = list(module.dataset)
        train_indices = list(module.train_dataset.indices)
        heldout_indices = sorted(set(range(len(samples))) - set(train_indices))
        if args.smoke:
            train_indices = train_indices[: min(6, len(train_indices))]
            heldout_indices = heldout_indices[: min(4, len(heldout_indices))]
        for tier in args.tiers:
            for seed in args.seeds:
                set_seed(seed)
                model, tokenizer, _settings, tokenizer_fingerprint = _load_source(
                    dataset, tier, seed, args.device
                )
                before_digest = _tensor_digest(model)
                extracted = _extract_examples(model, tokenizer, samples, args.device)
                train_examples = [extracted[index] for index in train_indices]
                test_examples = [extracted[index] for index in heldout_indices]
                train_query, train_chunks, train_positive = _stack_hidden(train_examples, args.device)
                set_seed(100_000 + int(seed))
                learned = AsymmetricLinearRouter(model.cfg.d_model, args.routing_dim).to(args.device)
                history = train_router(
                    learned,
                    train_query,
                    train_chunks,
                    train_positive,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    margin=args.margin,
                )
                for method in METHODS:
                    scores = _method_scores(method, test_examples, learned, args.device)
                    report, method_rankings = _evaluate(scores, test_examples)
                    geometry = _geometry_metrics(scores, test_examples)
                    per_seed.append(
                        _summary_row(
                            dataset,
                            tier,
                            seed,
                            method,
                            report,
                            geometry,
                            test_examples,
                            model,
                            learned,
                        )
                    )
                    for point in report["curve"]:
                        curves.append(
                            {
                                "dataset": dataset,
                                "model_tier": tier,
                                "seed": seed,
                                "method": method,
                                **point,
                            }
                        )
                    for example, ranking in zip(test_examples, method_rankings, strict=True):
                        rankings.append(
                            {
                                "dataset": dataset,
                                "model_tier": tier,
                                "seed": seed,
                                "method": method,
                                "example_id": example.example_id,
                                "ranking": ranking,
                                "evidence_ids": sorted(example.evidence_ids),
                            }
                        )
                after_digest = _tensor_digest(model)
                training.append(
                    {
                        "dataset": dataset,
                        "model_tier": tier,
                        "seed": seed,
                        "tokenizer_fingerprint": tokenizer_fingerprint,
                        "train_examples": len(train_examples),
                        "heldout_examples": len(test_examples),
                        "initial_loss": history[0],
                        "final_loss": history[-1],
                        "backbone_frozen": before_digest == after_digest,
                    }
                )
                learned_row = per_seed[-1]
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "tier": tier,
                            "seed": seed,
                            "learned_auc": learned_row["auc_0_30"],
                            "learned_qk": learned_row["qk_spearman"],
                        }
                    ),
                    flush=True,
                )
                del model, learned, extracted, train_examples, test_examples
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()

    aggregate = _aggregate(per_seed)
    publication = _publication_summary(per_seed)
    write_csv(OUTPUT / "pooling_geometry_per_seed.csv", per_seed)
    write_csv(OUTPUT / "pooling_geometry_aggregate.csv", aggregate)
    write_csv(OUTPUT / "pooling_geometry_publication.csv", publication)
    write_csv(OUTPUT / "pooling_geometry_curves.csv", curves)
    write_csv(OUTPUT / "pooling_geometry_training.csv", training)
    artifact = OUTPUT / "pooling_geometry.json"
    write_json(
        artifact,
        {
            "metadata": metadata,
            "protocol": {
                "datasets": args.datasets,
                "tiers": args.tiers,
                "seeds": args.seeds,
                "split_count": args.split_count,
                "candidate_references": args.split_count - 1,
                "routing_dimension": args.routing_dim,
                "router_steps": args.steps,
                "gist_counts": list(GIST_COUNTS),
                "native_qk_target": "maximum token score after mean reduction across heads",
                "multi_gist_reducer": "maximum subgist score after mean reduction across heads",
                "short_chunk_policy": "G is capped at one non-empty subgist per token",
                "position_bias": "per-example Pearson score correlation with normalized chunk center",
                "qk_fidelity": "per-example Spearman candidate ranking correlation",
                "semantic_metric": "evidence-identity recall; any/all evidence retained separately",
                "backbone": "frozen Paper 1.5 RoPE checkpoint",
                "payload": "unchanged native token-level post-RoPE K and native V",
                "train_eval_split": "established datamodule 80% train; held-out complement",
            },
            "per_seed": per_seed,
            "aggregate": aggregate,
            "publication": publication,
            "training": training,
            "rankings": rankings,
        },
    )
    _plot(publication)
    refresh_manifest(
        metadata=metadata,
        pooling_geometry={
            "artifact": artifact.relative_to(REPO).as_posix(),
            "datasets": args.datasets,
            "tiers": args.tiers,
            "seeds": args.seeds,
            "figures": [
                (FIGURES / "rope_pooling_qk_fidelity.pdf").relative_to(REPO).as_posix(),
                (FIGURES / "rope_pooling_geometry_vs_semantics.pdf").relative_to(REPO).as_posix(),
                (FIGURES / "rope_pooling_position_bias.pdf").relative_to(REPO).as_posix(),
            ],
        },
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+", choices=("hotpotqa", "qasper"), default=["hotpotqa", "qasper"])
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=["tiny", "small"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--split-count", type=int, default=16)
    parser.add_argument("--routing-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
