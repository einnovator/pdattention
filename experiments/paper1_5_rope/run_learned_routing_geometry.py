"""Train and evaluate a tiny semantic router on frozen Paper 1.5 RoPE models."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_native_kv_benchmark as native  # noqa: E402

from common.recall_sparsity import DEFAULT_FRACTIONS, recall_sparsity_curve  # noqa: E402
from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    TIERS,
    environment_metadata,
    set_seed,
    write_csv,
    write_json,
)
from experiments.paper1_5_rope.learned_routing import (  # noqa: E402
    AsymmetricLinearRouter,
    cosine_scores,
    materialize_native_payload,
    rank_candidate_ids,
    shuffled_positive_mask,
    train_router,
    trainable_parameter_count,
)
from experiments.paper1_5_rope.run_qa_validation import _base_settings  # noqa: E402
from experiments.paper1_5_rope.run_retrieval_geometry_gate import _load_source  # noqa: E402


OUTPUT = RESULTS / "learned_routing"
FIGURE = REPO / "docs" / "papers" / "shared" / "figures" / "rope_learned_routing.pdf"
METHODS = (
    "post_rope_k",
    "pre_rope_k",
    "hidden_cosine",
    "learned_projection",
    "shuffled_projection",
)


@dataclass
class RoutingRepresentations:
    """Frozen final-layer routing inputs for examples with equal candidate count."""

    query_hidden: torch.Tensor
    chunk_hidden: torch.Tensor
    query_pre_k: torch.Tensor
    chunk_pre_k: torch.Tensor
    query_post_k: torch.Tensor
    chunk_post_k: torch.Tensor
    positive_mask: torch.Tensor
    candidate_ids: list[list[str]]
    evidence_ids: list[set[str]]
    candidate_lengths: list[list[int]]
    example_ids: list[str]
    detail_kv_bytes: list[int]
    payload_parity_error: float

    def subset(self, indices: list[int]) -> "RoutingRepresentations":
        index = torch.tensor(indices, dtype=torch.long, device=self.query_hidden.device)
        return RoutingRepresentations(
            query_hidden=self.query_hidden.index_select(0, index),
            chunk_hidden=self.chunk_hidden.index_select(0, index),
            query_pre_k=self.query_pre_k.index_select(0, index),
            chunk_pre_k=self.chunk_pre_k.index_select(0, index),
            query_post_k=self.query_post_k.index_select(0, index),
            chunk_post_k=self.chunk_post_k.index_select(0, index),
            positive_mask=self.positive_mask.index_select(0, index),
            candidate_ids=[self.candidate_ids[i] for i in indices],
            evidence_ids=[self.evidence_ids[i] for i in indices],
            candidate_lengths=[self.candidate_lengths[i] for i in indices],
            example_ids=[self.example_ids[i] for i in indices],
            detail_kv_bytes=[self.detail_kv_bytes[i] for i in indices],
            payload_parity_error=self.payload_parity_error,
        )


def _tensor_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _attention_input(
    model,
    token_ids: list[int],
    *,
    position_offset: int,
    device: str,
) -> torch.Tensor:
    """Capture the final block's normalized attention input ``[1,T,D]``."""

    captured = []
    handle = model.blocks[-1].ln1.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach())
    )
    try:
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            model(ids, use_pra_memory=False, position_offset=position_offset)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one final-layer attention-input capture")
    return captured[0]


def _split_heads(attention, tensor: torch.Tensor) -> torch.Tensor:
    return attention._split_heads(tensor)


def _extract_representations(model, tokenizer, samples, device: str) -> RoutingRepresentations:
    model.eval().requires_grad_(False)
    attention = model.blocks[-1].attn
    rows = []
    parity_error = 0.0
    for sample in samples:
        candidate_ids = [str(reference.uri) for reference in sample.references]
        reference_token_ids = [
            list(tokenizer.encode(str(reference.metadata.get("text", ""))))
            for reference in sample.references
        ]
        if any(not token_ids for token_ids in reference_token_ids):
            raise ValueError(f"Example {sample.id} contains an empty reference")
        source_ids = [token for row in reference_token_ids for token in row]
        question_ids = list(tokenizer.encode(sample.question))
        if not question_ids:
            raise ValueError(f"Example {sample.id} contains an empty query")
        query_hidden_tokens = _attention_input(
            model,
            question_ids,
            position_offset=len(source_ids),
            device=device,
        )
        query_hidden = query_hidden_tokens[:, -1, :]
        raw_query = _split_heads(attention, attention.q_proj(query_hidden[:, None, :]))
        query_position = torch.tensor(
            [len(source_ids) + len(question_ids) - 1], device=query_hidden.device
        )
        post_query, _ = attention.position_encoding.transform_qk(
            raw_query, raw_query, query_position
        )
        raw_query_flat = raw_query[:, :, 0, :].reshape(1, -1)
        post_query_flat = post_query[:, :, 0, :].reshape(1, -1)

        cursor = 0
        chunk_hidden = []
        chunk_pre_k = []
        chunk_post_k = []
        payload_by_id = {}
        lengths = []
        detail_bytes = 0
        for identity, token_ids in zip(candidate_ids, reference_token_ids, strict=True):
            end = cursor + len(token_ids)
            # Independent chunk encoding keeps the attention-input hidden state
            # invariant to a common RoPE offset. Only native post-RoPE K carries
            # the exact logical placement used by the positional control.
            hidden = _attention_input(
                model,
                token_ids,
                position_offset=cursor,
                device=device,
            )
            raw_key = _split_heads(attention, attention.k_proj(hidden))
            value = _split_heads(attention, attention.v_proj(hidden))
            positions = torch.arange(cursor, end, device=hidden.device)
            _, post_key = attention.position_encoding.transform_qk(
                raw_key, raw_key, positions
            )
            chunk_hidden.append(hidden.mean(dim=1))
            chunk_pre_k.append(
                raw_key.transpose(1, 2).reshape(1, end - cursor, -1).mean(dim=1)
            )
            chunk_post_k.append(
                post_key.transpose(1, 2).reshape(1, end - cursor, -1).mean(dim=1)
            )
            key_payload = post_key
            value_payload = value
            payload_by_id[identity] = (key_payload, value_payload)
            detail_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (key_payload, value_payload)
            )
            lengths.append(end - cursor)
            cursor = end

        fixed_ids = candidate_ids[: min(3, len(candidate_ids))]
        baseline_k, baseline_v = materialize_native_payload(payload_by_id, fixed_ids)
        for _method in METHODS:
            candidate_k, candidate_v = materialize_native_payload(payload_by_id, fixed_ids)
            parity_error = max(
                parity_error,
                float((candidate_k - baseline_k).abs().max().cpu()),
                float((candidate_v - baseline_v).abs().max().cpu()),
            )

        evidence = set(str(value) for value in sample.target_reference_uris)
        if not evidence or not evidence <= set(candidate_ids):
            raise ValueError(f"Example {sample.id} has invalid evidence identities")
        rows.append(
            {
                "query_hidden": query_hidden,
                "chunk_hidden": torch.cat(chunk_hidden, dim=0).unsqueeze(0),
                "query_pre_k": raw_query_flat,
                "chunk_pre_k": torch.cat(chunk_pre_k, dim=0).unsqueeze(0),
                "query_post_k": post_query_flat,
                "chunk_post_k": torch.cat(chunk_post_k, dim=0).unsqueeze(0),
                "positive_mask": torch.tensor(
                    [[identity in evidence for identity in candidate_ids]],
                    dtype=torch.bool,
                    device=query_hidden.device,
                ),
                "candidate_ids": candidate_ids,
                "evidence_ids": evidence,
                "candidate_lengths": lengths,
                "example_id": str(sample.id),
                "detail_kv_bytes": detail_bytes,
            }
        )
    return RoutingRepresentations(
        query_hidden=torch.cat([row["query_hidden"] for row in rows]),
        chunk_hidden=torch.cat([row["chunk_hidden"] for row in rows]),
        query_pre_k=torch.cat([row["query_pre_k"] for row in rows]),
        chunk_pre_k=torch.cat([row["chunk_pre_k"] for row in rows]),
        query_post_k=torch.cat([row["query_post_k"] for row in rows]),
        chunk_post_k=torch.cat([row["chunk_post_k"] for row in rows]),
        positive_mask=torch.cat([row["positive_mask"] for row in rows]),
        candidate_ids=[row["candidate_ids"] for row in rows],
        evidence_ids=[row["evidence_ids"] for row in rows],
        candidate_lengths=[row["candidate_lengths"] for row in rows],
        example_ids=[row["example_id"] for row in rows],
        detail_kv_bytes=[row["detail_kv_bytes"] for row in rows],
        payload_parity_error=parity_error,
    )


def _method_scores(
    method: str,
    representations: RoutingRepresentations,
    routers: dict[str, AsymmetricLinearRouter],
) -> torch.Tensor:
    if method == "post_rope_k":
        return cosine_scores(representations.query_post_k, representations.chunk_post_k)
    if method == "pre_rope_k":
        return cosine_scores(representations.query_pre_k, representations.chunk_pre_k)
    if method == "hidden_cosine":
        return cosine_scores(representations.query_hidden, representations.chunk_hidden)
    return routers[method](representations.query_hidden, representations.chunk_hidden)


def _evaluate_scores(
    scores: torch.Tensor,
    representations: RoutingRepresentations,
) -> tuple[dict, list[list[str]]]:
    rankings = rank_candidate_ids(scores, representations.candidate_ids)
    lengths_by_example = []
    for ranking, identities, lengths in zip(
        rankings,
        representations.candidate_ids,
        representations.candidate_lengths,
        strict=True,
    ):
        length_by_id = dict(zip(identities, lengths, strict=True))
        lengths_by_example.append([length_by_id[identity] for identity in ranking])
    result = recall_sparsity_curve(
        rankings,
        representations.evidence_ids,
        fractions=DEFAULT_FRACTIONS,
        fixed_k=(1, 3, 8, 16),
        candidate_token_lengths=lengths_by_example,
        require_complete_endpoint=True,
    )
    reciprocal_ranks = []
    for ranking, evidence in zip(rankings, representations.evidence_ids, strict=True):
        first = min(index for index, identity in enumerate(ranking, start=1) if identity in evidence)
        reciprocal_ranks.append(1.0 / first)
    result["mrr"] = statistics.fmean(reciprocal_ranks)
    return result, rankings


def _summary_row(dataset, tier, seed, method, result, *, accounting):
    by_fraction = {float(row["fraction"]): row for row in result["curve"]}
    row = {
        "dataset": dataset,
        "model_tier": tier,
        "seed": seed,
        "method": method,
        "examples": result["examples"],
        "mrr": result["mrr"],
        "f80": result["inverse"]["f80"],
        "f90": result["inverse"]["f90"],
        "auc_0_30": result["auc_0_30"],
        "r_at_3": result["fixed_k"]["3"]["recall"],
        "r_at_8": result["fixed_k"]["8"]["recall"],
        **accounting,
    }
    for fraction in (0.05, 0.10, 0.20, 0.30):
        point = by_fraction[fraction]
        suffix = int(100 * fraction)
        row[f"r_at_{suffix}pct"] = point["recall"]
        row[f"any_r_at_{suffix}pct"] = point["any_evidence_recall"]
        row[f"all_r_at_{suffix}pct"] = point["all_evidence_recall"]
        row[f"kv_at_{suffix}pct"] = point["selected_kv_token_fraction"]
    return row


def _aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model_tier"], row["method"])].append(row)
    metrics = (
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
        "any_r_at_5pct",
        "any_r_at_10pct",
        "any_r_at_20pct",
        "any_r_at_30pct",
        "all_r_at_5pct",
        "all_r_at_10pct",
        "all_r_at_20pct",
        "all_r_at_30pct",
    )
    output = []
    for identity, values in sorted(groups.items()):
        result = {
            "dataset": identity[0],
            "model_tier": identity[1],
            "method": identity[2],
            "seed_count": len({row["seed"] for row in values}),
            "examples_per_seed": values[0]["examples"],
        }
        for metric in metrics:
            observed = [float(row[metric]) for row in values if row.get(metric) is not None]
            result[f"{metric}_mean"] = statistics.fmean(observed) if observed else None
            result[f"{metric}_std"] = statistics.pstdev(observed) if observed else None
        for field in (
            "router_parameters",
            "router_backbone_percent",
            "routing_dimension",
            "native_hidden_index_bytes",
            "projected_index_bytes",
            "native_detail_kv_bytes",
            "payload_parity_max_abs_error",
        ):
            result[field] = statistics.fmean(float(row[field]) for row in values)
        output.append(result)
    return output


def _paired_deltas(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["dataset"], row["model_tier"], row["seed"], row["method"]): row
        for row in rows
    }
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for tier in sorted({row["model_tier"] for row in rows}):
            for comparison in ("hidden_cosine", "post_rope_k", "shuffled_projection"):
                deltas = defaultdict(list)
                for seed in sorted({row["seed"] for row in rows}):
                    learned = indexed.get((dataset, tier, seed, "learned_projection"))
                    baseline = indexed.get((dataset, tier, seed, comparison))
                    if learned is None or baseline is None:
                        continue
                    for metric in ("auc_0_30", "r_at_5pct", "r_at_10pct", "r_at_20pct", "r_at_30pct", "mrr"):
                        deltas[metric].append(float(learned[metric]) - float(baseline[metric]))
                row = {
                    "dataset": dataset,
                    "model_tier": tier,
                    "comparison": f"learned_minus_{comparison}",
                    "seed_count": len(next(iter(deltas.values()), [])),
                }
                for metric, values in deltas.items():
                    row[f"{metric}_delta_mean"] = statistics.fmean(values)
                    row[f"{metric}_delta_std"] = statistics.pstdev(values)
                    row[f"{metric}_positive_seeds"] = sum(value > 0 for value in values)
                output.append(row)
    return output


def _plot(curve_rows: list[dict], path: Path) -> None:
    styles = {
        "post_rope_k": ("#9C3D38", "o", "Post-RoPE Q/K"),
        "pre_rope_k": ("#C47A35", "s", "Pre-RoPE Q/K"),
        "hidden_cosine": ("#2F6F9F", "^", "Hidden cosine"),
        "learned_projection": ("#2E7D55", "D", "Learned hidden"),
        "shuffled_projection": ("#777777", "x", "Shuffled labels"),
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True, sharey=True)
    for axis, dataset, tier in zip(
        axes.flat,
        ("hotpotqa", "qasper", "hotpotqa", "qasper"),
        ("tiny", "tiny", "small", "small"),
        strict=True,
    ):
        for method, (color, marker, label) in styles.items():
            selected = [
                row
                for row in curve_rows
                if row["dataset"] == dataset
                and row["model_tier"] == tier
                and row["method"] == method
                and float(row["fraction"]) <= 0.30
            ]
            by_fraction = defaultdict(list)
            for row in selected:
                by_fraction[float(row["selected_chunk_fraction"])].append(
                    float(row["recall"])
                )
            fractions = sorted(by_fraction)
            axis.plot(
                [100 * value for value in fractions],
                [statistics.fmean(by_fraction[value]) for value in fractions],
                color=color,
                marker=marker,
                label=label,
            )
        axis.set_title(f"{'HotpotQA' if dataset == 'hotpotqa' else 'QASPER'}-derived, {tier}")
        axis.set_xlim(5, 35)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Selected candidate references (%)")
    axes[0, 0].set_ylabel("Evidence-identity recall")
    axes[1, 0].set_ylabel("Evidence-identity recall")
    axes[0, 1].legend(frameon=True, fontsize=8, loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    figure.savefig(OUTPUT / "learned_routing_recall_sparsity.png", dpi=220)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    metadata = environment_metadata()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_seed_rows = []
    curve_rows = []
    ranking_rows = []
    training_rows = []
    for dataset in args.datasets:
        base = _base_settings(dataset, False, args.max_examples)
        _prepared_tokenizer, _training_module, modules = native.DATASET_PREPARERS[dataset](base)
        module = modules[args.split_count]
        all_samples = list(module.dataset)
        train_indices = list(module.train_dataset.indices)
        heldout_indices = sorted(set(range(len(all_samples))) - set(train_indices))
        if args.smoke:
            train_indices = train_indices[: min(6, len(train_indices))]
            heldout_indices = heldout_indices[: min(4, len(heldout_indices))]
        for tier in args.tiers:
            for seed in args.seeds:
                set_seed(seed)
                model, tokenizer, _settings, tokenizer_fingerprint = _load_source(
                    dataset, tier, seed, args.device
                )
                model.eval().requires_grad_(False)
                before_digest = _tensor_digest(model)
                representations = _extract_representations(
                    model, tokenizer, all_samples, args.device
                )
                train_data = representations.subset(train_indices)
                test_data = representations.subset(heldout_indices)

                initialization_seed = 100_000 + int(seed)
                set_seed(initialization_seed)
                learned = AsymmetricLinearRouter(model.cfg.d_model, args.routing_dim).to(args.device)
                initial_state = copy.deepcopy(learned.state_dict())
                learned_history = train_router(
                    learned,
                    train_data.query_hidden,
                    train_data.chunk_hidden,
                    train_data.positive_mask,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    margin=args.margin,
                )
                shuffled = AsymmetricLinearRouter(model.cfg.d_model, args.routing_dim).to(args.device)
                shuffled.load_state_dict(initial_state)
                shuffled_labels = shuffled_positive_mask(train_data.positive_mask, seed)
                shuffled_history = train_router(
                    shuffled,
                    train_data.query_hidden,
                    train_data.chunk_hidden,
                    shuffled_labels,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    margin=args.margin,
                )
                routers = {
                    "learned_projection": learned,
                    "shuffled_projection": shuffled,
                }
                backbone_parameters = sum(parameter.numel() for parameter in model.parameters())
                router_parameters = trainable_parameter_count(learned)
                candidate_count = len(test_data.candidate_ids[0])
                element_size = test_data.query_hidden.element_size()
                accounting = {
                    "router_parameters": router_parameters,
                    "router_backbone_percent": 100.0 * router_parameters / backbone_parameters,
                    "routing_dimension": args.routing_dim,
                    "native_hidden_index_bytes": candidate_count * model.cfg.d_model * element_size,
                    "projected_index_bytes": candidate_count * args.routing_dim * element_size,
                    "native_detail_kv_bytes": statistics.fmean(test_data.detail_kv_bytes),
                    "payload_parity_max_abs_error": test_data.payload_parity_error,
                }
                for method in METHODS:
                    with torch.no_grad():
                        scores = _method_scores(method, test_data, routers)
                    result, rankings = _evaluate_scores(scores, test_data)
                    per_seed_rows.append(
                        _summary_row(
                            dataset,
                            tier,
                            seed,
                            method,
                            result,
                            accounting=accounting,
                        )
                    )
                    for point in result["curve"]:
                        curve_rows.append(
                            {
                                "dataset": dataset,
                                "model_tier": tier,
                                "seed": seed,
                                "method": method,
                                **point,
                            }
                        )
                    for example_id, ranking, evidence in zip(
                        test_data.example_ids,
                        rankings,
                        test_data.evidence_ids,
                        strict=True,
                    ):
                        ranking_rows.append(
                            {
                                "dataset": dataset,
                                "model_tier": tier,
                                "seed": seed,
                                "method": method,
                                "example_id": example_id,
                                "ranking": ranking,
                                "evidence_ids": sorted(evidence),
                            }
                        )
                after_digest = _tensor_digest(model)
                training_rows.append(
                    {
                        "dataset": dataset,
                        "model_tier": tier,
                        "seed": seed,
                        "tokenizer_fingerprint": tokenizer_fingerprint,
                        "train_examples": len(train_indices),
                        "heldout_examples": len(heldout_indices),
                        "learned_initial_loss": learned_history[0],
                        "learned_final_loss": learned_history[-1],
                        "shuffled_initial_loss": shuffled_history[0],
                        "shuffled_final_loss": shuffled_history[-1],
                        "backbone_frozen": before_digest == after_digest,
                    }
                )
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "tier": tier,
                            "seed": seed,
                            "learned_loss": learned_history[-1],
                            "shuffled_loss": shuffled_history[-1],
                        }
                    ),
                    flush=True,
                )

    aggregate = _aggregate(per_seed_rows)
    paired = _paired_deltas(per_seed_rows)
    write_csv(OUTPUT / "learned_routing_per_seed.csv", per_seed_rows)
    write_csv(OUTPUT / "learned_routing_aggregate.csv", aggregate)
    write_csv(OUTPUT / "learned_routing_paired.csv", paired)
    write_csv(OUTPUT / "learned_routing_curves.csv", curve_rows)
    write_csv(OUTPUT / "learned_routing_training.csv", training_rows)
    write_json(
        OUTPUT / "learned_routing_geometry.json",
        {
            "metadata": metadata,
            "protocol": {
                "datasets": args.datasets,
                "tiers": args.tiers,
                "seeds": args.seeds,
                "split_count": args.split_count,
                "candidate_references": args.split_count - 1,
                "routing_dimension": args.routing_dim,
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "margin": args.margin,
                "backbone": "frozen Paper 1.5 RoPE checkpoint",
                "routing_layer": "final attention-input hidden state",
                "candidate_encoding": "independent chunks at exact logical RoPE offsets",
                "payload": "unchanged native post-RoPE K and native V",
                "train_eval_split": "established datamodule 80% train; held-out complement",
            },
            "per_seed": per_seed_rows,
            "aggregate": aggregate,
            "paired": paired,
            "training": training_rows,
            "rankings": ranking_rows,
        },
    )
    _plot(curve_rows, FIGURE)
    return OUTPUT / "learned_routing_geometry.json"


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
