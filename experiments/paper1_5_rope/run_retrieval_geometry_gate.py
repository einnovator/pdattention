"""Run the final fixed-distance and context-granularity RoPE gate."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_native_kv_benchmark as native  # noqa: E402

from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    TIERS,
    environment_metadata,
    set_seed,
    write_csv,
    write_json,
)
from experiments.paper1_5_rope.run_qa_validation import _base_settings  # noqa: E402
from data.tokenizer import BPETokenizer, PRATokenizer  # noqa: E402
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402
from pra_torch.native_metrics import recovered_context_benefit  # noqa: E402
from pra_torch.pra_train import evaluate_reference_ablation  # noqa: E402


DISTANCE_DIR = RESULTS / "distance"
CONTEXT_DIR = RESULTS / "context_gate"
DISTANCES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
POLICY_LABELS = {"exact": "exact", "local": "local"}


def _checkpoint_dir(dataset: str, tier: str, seed: int) -> Path:
    return (
        REPO
        / "out"
        / "paper1_5_rope"
        / "validation"
        / dataset
        / tier
        / "rope"
        / f"seed-{seed}"
    )


def _load_source(dataset, tier, seed, device):
    """Load weights and the exact tokenizer that assigned their embedding IDs."""
    path = _checkpoint_dir(dataset, tier, seed) / "checkpoint.pt"
    checkpoint = torch.load(path, map_location=device)
    settings = dict(checkpoint["settings"])
    if int(checkpoint.get("step", 0)) != int(settings["steps"]):
        raise RuntimeError(f"Incomplete validation checkpoint: {path}")
    if checkpoint.get("tokenizer_type") == "BPETokenizer":
        tokenizer = BPETokenizer.from_json(checkpoint["tokenizer_json"])
        tokenizer_payload = checkpoint["tokenizer_json"]
    else:
        tokenizer = PRATokenizer.from_vocab(checkpoint["stoi"])
        tokenizer_payload = repr(sorted(checkpoint["stoi"].items()))
    cfg = PRAConfig(**checkpoint["cfg"])
    source = TinyPRAModel(cfg).to(device)
    source.load_state_dict(checkpoint["model"])
    source.eval()
    fingerprint = hashlib.sha256(tokenizer_payload.encode("utf-8")).hexdigest()
    return source, tokenizer, settings, fingerprint


def _bind_tokenizer(modules, tokenizer, max_seq_len):
    for module in modules.values():
        module.tokenizer = tokenizer
        module.collator = native.AnswerTokenCollator(
            tokenizer, max_seq_len=int(max_seq_len)
        )


def _baselines(source, tokenizer, module, settings, device):
    collator = native.AnswerTokenCollator(tokenizer, max_seq_len=settings["max_seq_len"])
    indices = native._subset_indices(module, "test")
    full = native._loader(
        Subset(native.FullContextDataset(module.dataset), indices),
        collator,
        batch_size=1,
        shuffle=False,
        seed=0,
    )
    tail = native._loader(
        module.test_dataset,
        collator,
        batch_size=1,
        shuffle=False,
        seed=0,
    )
    return {
        "sa_full": native._evaluate_model(
            source, full, device, condition="sa_full", tokenizer=tokenizer
        ),
        "sa_tail": native._evaluate_model(
            source, tail, device, condition="sa_tail", tokenizer=tokenizer
        ),
    }


def _converted(source, device, *, encoding_strategy, encoding_block_references):
    cfg = native._native_config(
        source,
        device,
        {
            "store_pre_position_keys": True,
            "retrieval_position_policy": "exact",
            "prompt_position_mode": "historical",
            "reference_position_mode": "global",
            "reference_encoding_strategy": encoding_strategy,
            "encoding_block_references": encoding_block_references,
            "max_gists_per_reference": 128,
            "top_k_references": 2,
            "top_k_chunks_per_reference": 1,
            "max_materialized_memory_tokens": 160,
            "context_safety_reserve_tokens": 4,
            "collect_routing_metrics": True,
            "collect_rank_diagnostics": True,
        },
    )
    return convert_sa_model_to_pra(source, cfg).to(device).eval()


def _set_policy(model, policy: str, distance: int | None) -> None:
    model.cfg.retrieval_position_policy = policy
    model.cfg.retrieval_position_distance = distance


def _flatten(
    result,
    *,
    metadata,
    dataset,
    tier,
    seed,
    stage,
    setting,
    split_count,
    encoding_block_references,
    position_policy,
    requested_distance,
    oracle_mode,
    baselines,
    tokenizer_fingerprint,
):
    rows = []
    for row in result["per_example"]:
        rcb = recovered_context_benefit(
            sa_full_loss=baselines["sa_full"]["loss"],
            sa_tail_loss=baselines["sa_tail"]["loss"],
            pra_loss=row["loss"],
        )
        rows.append(
            {
                "git_sha": metadata["git_sha"],
                "dataset": dataset,
                "model_tier": tier,
                "seed": seed,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "stage": stage,
                "setting": setting,
                "example_id": row["example_id"],
                "split_count": split_count,
                "encoding_block_references": encoding_block_references,
                "routing_chunk_tokens": row.get("retrieved_physical_kv_tokens"),
                "materialization_budget": 160,
                "k_storage_mode": "pre_position_deferred",
                "position_policy": position_policy,
                "requested_distance": requested_distance,
                "original_logical_distance": row.get("original_retrieval_distance"),
                "effective_distance": row.get("effective_retrieval_distance"),
                "distance_over_training_context": (
                    row.get("effective_retrieval_distance") / 256
                    if row.get("effective_retrieval_distance") is not None
                    else None
                ),
                "distance_over_model_operation_limit": (
                    row.get("effective_retrieval_distance")
                    / row["model_operation_limit"]
                    if row.get("effective_retrieval_distance") is not None
                    else None
                ),
                "oracle_mode": oracle_mode,
                "loss": row["loss"],
                "token_accuracy": row["token_accuracy"],
                "sa_full_loss": baselines["sa_full"]["loss"],
                "sa_tail_loss": baselines["sa_tail"]["loss"],
                "rcb": rcb,
                "retrieval_key_rmse_vs_exact": row.get("retrieval_key_rmse_vs_exact"),
                "memory_attention_mass": row.get("memory_attention_mass"),
                "memory_last_query_attention_mass": row.get(
                    "memory_last_query_attention_mass"
                ),
                "memory_attention_entropy": row.get("memory_attention_entropy"),
                "memory_attention_max_weight": row.get("memory_attention_max_weight"),
                "retrieval_logit_rmse_vs_exact": row.get(
                    "retrieval_logit_rmse_vs_exact"
                ),
                "retrieval_attention_l1_vs_exact": row.get(
                    "retrieval_attention_l1_vs_exact"
                ),
                "retrieval_top_token_agreement_vs_exact": row.get(
                    "retrieval_top_token_agreement_vs_exact"
                ),
                "recall_at_k": row.get("recall_at_k"),
                "routing_mrr": row.get("routing_mrr"),
                "retrieved_physical_kv_tokens": row.get("retrieved_physical_kv_tokens"),
                "maximum_native_operation": row.get("maximum_native_operation"),
                "model_operation_limit": row.get("model_operation_limit"),
                "native_limit_violations": row.get("native_limit_violations"),
            }
        )
    return rows


def _evaluate_conditions(
    model,
    module,
    tokenizer,
    device,
    policies,
    conditions,
    context,
):
    rows = []
    encoded_cache = {}
    for policy, distance in policies:
        _set_policy(model, policy, distance)
        for condition in conditions:
            result = evaluate_reference_ablation(
                model=model,
                loader=module.test_loader(),
                tokenizer=tokenizer,
                device=device,
                condition=condition,
                collect_per_example=True,
                encoded_entry_cache=encoded_cache,
            )
            rows.extend(
                _flatten(
                    result,
                    position_policy=(
                        f"{policy}_{distance}"
                        if policy in {"fixed", "clipped"}
                        else policy
                    ),
                    requested_distance=distance,
                    oracle_mode=condition,
                    **context,
                )
            )
    return rows


def _seed_balanced(rows):
    identity = (
        "dataset",
        "model_tier",
        "stage",
        "setting",
        "split_count",
        "encoding_block_references",
        "position_policy",
        "requested_distance",
        "oracle_mode",
    )
    metrics = (
        "loss",
        "token_accuracy",
        "rcb",
        "retrieval_key_rmse_vs_exact",
        "memory_attention_mass",
        "memory_last_query_attention_mass",
        "memory_attention_entropy",
        "memory_attention_max_weight",
        "retrieval_logit_rmse_vs_exact",
        "retrieval_attention_l1_vs_exact",
        "retrieval_top_token_agreement_vs_exact",
        "recall_at_k",
        "routing_mrr",
        "retrieved_physical_kv_tokens",
        "original_logical_distance",
        "effective_distance",
        "maximum_native_operation",
        "native_limit_violations",
    )
    seed_groups = defaultdict(list)
    for row in rows:
        seed_groups[tuple(row[key] for key in identity) + (row["seed"],)].append(row)
    seed_rows = []
    for key, values in sorted(seed_groups.items(), key=str):
        record = dict(zip(identity + ("seed",), key))
        record["example_count"] = len(values)
        for metric in metrics:
            observed = [
                float(row[metric])
                for row in values
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            if observed:
                record[metric] = statistics.fmean(observed)
        seed_rows.append(record)
    groups = defaultdict(list)
    for row in seed_rows:
        groups[tuple(row[key] for key in identity)].append(row)
    aggregate = []
    for key, values in sorted(groups.items(), key=str):
        record = dict(zip(identity, key))
        record["seed_count"] = len(values)
        record["example_count"] = sum(row["example_count"] for row in values)
        for metric in metrics:
            observed = [float(row[metric]) for row in values if row.get(metric) is not None]
            if observed:
                record[f"{metric}_mean"] = statistics.fmean(observed)
                record[f"{metric}_std"] = statistics.pstdev(observed)
                record[f"{metric}_median"] = statistics.median(observed)
        aggregate.append(record)
    return seed_rows, aggregate


def _best_fixed_distance(aggregate):
    candidates = defaultdict(list)
    for row in aggregate:
        if row["stage"] == "distance" and row["position_policy"].startswith("fixed_"):
            candidates[int(row["requested_distance"])].append(row["loss_mean"])
    return min(candidates, key=lambda distance: statistics.fmean(candidates[distance]))


def _paired_directions(seed_rows, stage, varying_field, candidate, baseline="local"):
    index = {}
    for row in seed_rows:
        if row["stage"] != stage or row["oracle_mode"] != "native_oracle":
            continue
        identity = (row["dataset"], row["model_tier"], row[varying_field], row["seed"])
        index.setdefault(identity, {})[row["position_policy"]] = row["loss"]
    deltas = [
        values[candidate] - values[baseline]
        for values in index.values()
        if candidate in values and baseline in values
    ]
    return {
        "comparison": f"{candidate}_minus_{baseline}",
        "pair_count": len(deltas),
        "mean_delta_loss": statistics.fmean(deltas) if deltas else None,
        "median_delta_loss": statistics.median(deltas) if deltas else None,
        "candidate_better_pairs": sum(delta < 0 for delta in deltas),
        "candidate_worse_pairs": sum(delta > 0 for delta in deltas),
    }


def _plots(aggregate, best_distance):
    colors = {"hotpotqa": "#245A8D", "qasper": "#A34832"}
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for dataset, color in colors.items():
        values = [
            row
            for row in aggregate
            if row["stage"] == "distance"
            and row["dataset"] == dataset
            and row["position_policy"].startswith("fixed_")
        ]
        values.sort(key=lambda row: row["requested_distance"])
        axes[0].plot(
            [row["requested_distance"] for row in values],
            [row["loss_mean"] for row in values],
            marker="o",
            color=color,
            label=dataset,
        )
        axes[0].fill_between(
            [row["requested_distance"] for row in values],
            [row["loss_mean"] - row["loss_std"] for row in values],
            [row["loss_mean"] + row["loss_std"] for row in values],
            color=color,
            alpha=0.12,
        )
        exact = next(
            row
            for row in aggregate
            if row["stage"] == "distance"
            and row["dataset"] == dataset
            and row["position_policy"] == "exact"
        )
        axes[0].axhline(
            exact["loss_mean"], color=color, linestyle=":", linewidth=1.1
        )
        axes[1].plot(
            [row["requested_distance"] for row in values],
            [row["memory_attention_mass_mean"] for row in values],
            marker="s",
            color=color,
            label=dataset,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.axvline(best_distance, color="#555555", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        axis.set_xlabel("Nearest-token effective distance D")
    axes[0].set_ylabel("Oracle answer-token loss")
    axes[1].set_ylabel("Attention mass on retrieved memory")
    figure.tight_layout()
    figure.savefig(DISTANCE_DIR / "rope_d_sweep.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))
    policy_order = ("local", "exact", f"fixed_{best_distance}")
    for column, (stage, field, label) in enumerate(
        (
            ("encoding_context", "encoding_block_references", "References per encoding block"),
            ("routing_chunk", "split_count", "Source split count (inverse unit size)"),
        )
    ):
        for policy, marker in zip(policy_order, ("o", "s", "^")):
            values = [
                row
                for row in aggregate
                if row["stage"] == stage
                and row["model_tier"] == "tiny"
                and row["position_policy"] == policy
                and row["oracle_mode"] == "native_oracle"
            ]
            grouped = defaultdict(list)
            for row in values:
                grouped[row[field]].append(row["loss_mean"])
            x = sorted(grouped)
            axes[column].plot(
                x,
                [statistics.fmean(grouped[value]) for value in x],
                marker=marker,
                label=policy.replace("_", " "),
            )
        axes[column].set_xlabel(label)
        axes[column].set_ylabel("Oracle answer-token loss")
        axes[column].grid(alpha=0.25)
        axes[column].legend(frameon=False, fontsize=8)
    values = [
        row
        for row in aggregate
        if row["stage"] in {"distance", "encoding_context", "routing_chunk"}
        and row["oracle_mode"] == "native_oracle"
    ]
    for dataset, color in colors.items():
        selected = [row for row in values if row["dataset"] == dataset]
        axes[2].scatter(
            [row["retrieval_key_rmse_vs_exact_mean"] for row in selected],
            [row["loss_mean"] for row in selected],
            s=18,
            alpha=0.65,
            color=color,
            label=dataset,
        )
    axes[2].set_xlabel("Retrieved-K RMSE vs exact phase")
    axes[2].set_ylabel("Oracle answer-token loss")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(CONTEXT_DIR / "rope_context_gate.png", dpi=180)
    plt.close(figure)


def run(args):
    metadata = environment_metadata()
    all_rows = []
    prepared = {}
    base_by_dataset = {}
    distance_policies = [("local", None), ("exact", None)] + [
        ("fixed", distance) for distance in DISTANCES
    ]

    for dataset in args.datasets:
        base = _base_settings(dataset, False, 12)
        base_by_dataset[dataset] = base
        prepared[dataset] = native.DATASET_PREPARERS[dataset](base)
        _preparation_tokenizer, _training_module, modules = prepared[dataset]
        native._assert_fixed_target_invariants(modules)
        for seed in args.seeds:
            set_seed(seed)
            source, tokenizer, train_settings, tokenizer_fingerprint = _load_source(
                dataset, "tiny", seed, args.device
            )
            _bind_tokenizer(modules, tokenizer, train_settings["max_seq_len"])
            baselines = _baselines(source, tokenizer, modules[5], train_settings, args.device)
            model = _converted(
                source,
                args.device,
                encoding_strategy="block_slice",
                encoding_block_references=4,
            )
            context = {
                "metadata": metadata,
                "dataset": dataset,
                "tier": "tiny",
                "seed": seed,
                "stage": "distance",
                "setting": "split5_block4_oracle",
                "split_count": 5,
                "encoding_block_references": 4,
                "baselines": baselines,
                "tokenizer_fingerprint": tokenizer_fingerprint,
            }
            all_rows.extend(
                _evaluate_conditions(
                    model,
                    modules[5],
                    tokenizer,
                    args.device,
                    distance_policies,
                    ("native_oracle",),
                    context,
                )
            )
            del model, source
            if args.device == "cuda":
                torch.cuda.empty_cache()
            print(f"distance {dataset} tiny seed={seed} complete", flush=True)

    _, provisional = _seed_balanced(all_rows)
    best_distance = _best_fixed_distance(provisional)
    gate_policies = (("local", None), ("exact", None), ("fixed", best_distance))
    print(f"preregistered sweep selected fixed D={best_distance}", flush=True)

    for dataset in args.datasets:
        base = base_by_dataset[dataset]
        _preparation_tokenizer, _training_module, modules = prepared[dataset]
        for tier in args.tiers:
            for seed in args.seeds:
                set_seed(seed)
                source, tokenizer, train_settings, tokenizer_fingerprint = _load_source(
                    dataset, tier, seed, args.device
                )
                _bind_tokenizer(modules, tokenizer, train_settings["max_seq_len"])
                baselines = _baselines(
                    source, tokenizer, modules[5], train_settings, args.device
                )
                for block_references in (1, 2, 4):
                    model = _converted(
                        source,
                        args.device,
                        encoding_strategy="block_slice",
                        encoding_block_references=block_references,
                    )
                    all_rows.extend(
                        _evaluate_conditions(
                            model,
                            modules[5],
                            tokenizer,
                            args.device,
                            gate_policies,
                            ("native_oracle", "native_all"),
                            {
                                "metadata": metadata,
                                "dataset": dataset,
                                "tier": tier,
                                "seed": seed,
                                "stage": "encoding_context",
                                "setting": f"split5_block{block_references}",
                                "split_count": 5,
                                "encoding_block_references": block_references,
                                "baselines": baselines,
                                "tokenizer_fingerprint": tokenizer_fingerprint,
                            },
                        )
                    )
                    del model
                for split_count in (16, 5, 2):
                    model = _converted(
                        source,
                        args.device,
                        encoding_strategy="native_slice",
                        encoding_block_references=128,
                    )
                    all_rows.extend(
                        _evaluate_conditions(
                            model,
                            modules[split_count],
                            tokenizer,
                            args.device,
                            gate_policies,
                            ("native_oracle",),
                            {
                                "metadata": metadata,
                                "dataset": dataset,
                                "tier": tier,
                                "seed": seed,
                                "stage": "routing_chunk",
                                "setting": f"native_slice_split{split_count}",
                                "split_count": split_count,
                                "encoding_block_references": 128,
                                "baselines": baselines,
                                "tokenizer_fingerprint": tokenizer_fingerprint,
                            },
                        )
                    )
                    del model
                model = _converted(
                    source,
                    args.device,
                    encoding_strategy="block_slice",
                    encoding_block_references=4,
                )
                all_rows.extend(
                    _evaluate_conditions(
                        model,
                        modules[5],
                        tokenizer,
                        args.device,
                        (
                            ("exact", None),
                            ("fixed", best_distance),
                            ("clipped", 64),
                            ("clipped", best_distance),
                            ("log_compressed", best_distance),
                        ),
                        (
                            "native_oracle",
                            "native_evidence_adjacent",
                            "native_evidence_irrelevant",
                            "native_all",
                            "valid",
                        ),
                        {
                            "metadata": metadata,
                            "dataset": dataset,
                            "tier": tier,
                            "seed": seed,
                            "stage": "confirmation",
                            "setting": "split5_block4",
                            "split_count": 5,
                            "encoding_block_references": 4,
                            "baselines": baselines,
                            "tokenizer_fingerprint": tokenizer_fingerprint,
                        },
                    )
                )
                del model, source
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                print(f"gate {dataset} {tier} seed={seed} complete", flush=True)

    seed_rows, aggregate = _seed_balanced(all_rows)
    summary = {
        "metadata": metadata,
        "protocol": {
            "seeds": list(args.seeds),
            "datasets": list(args.datasets),
            "tiers": list(args.tiers),
            "distance_candidates": list(DISTANCES),
            "selected_fixed_distance": best_distance,
            "distance_convention": "query minus nearest selected memory token",
            "training_context": 256,
            "oracle_first": True,
            "same_raw_k_v_within_policy_sweeps": True,
        },
        "paired_directions": {
            "encoding_exact_vs_local": _paired_directions(
                seed_rows, "encoding_context", "encoding_block_references", "exact"
            ),
            "encoding_fixed_vs_local": _paired_directions(
                seed_rows,
                "encoding_context",
                "encoding_block_references",
                f"fixed_{best_distance}",
            ),
            "chunk_exact_vs_local": _paired_directions(
                seed_rows, "routing_chunk", "split_count", "exact"
            ),
            "chunk_fixed_vs_local": _paired_directions(
                seed_rows,
                "routing_chunk",
                "split_count",
                f"fixed_{best_distance}",
            ),
        },
        "native_limit_violations": int(
            sum(row.get("native_limit_violations", 0) for row in all_rows)
        ),
        "expected_vs_observed": [
            {
                "hypothesis": "tiny_fragment_contextualization",
                "expected": "larger encoding blocks reduce the exact-versus-local loss gap",
                "observed": (
                    "oracle evidence is causally first and therefore unchanged; when all "
                    "neighboring K/V is retained, larger encoding groups sharply reduce loss "
                    "in the small tier and modestly in the tiny tier"
                ),
                "matches": "partly",
                "alternative_explanation": (
                    "the benchmark cannot change the first evidence span through later causal "
                    "context, so its oracle row does not identify an evidence-context effect"
                ),
            },
            {
                "hypothesis": "larger_retrieved_unit",
                "expected": "larger materialized evidence units improve task use",
                "observed": (
                    "exact-position loss generally falls from 16-way to 5-way and 2-way "
                    "partitions, with one tiny-QASPER plateau"
                ),
                "matches": "yes",
                "alternative_explanation": (
                    "unit size and included neighboring content change together in this probe"
                ),
            },
            {
                "hypothesis": "distance_mismatch",
                "expected": "a stable fixed or clipped distance beats exact source distance",
                "observed": (
                    "fixed D=64 and exact split 50/50 across 100 confirmation cells; clipped "
                    "D=64 has a tiny mean advantage but no task-scale gain, while remote fixed "
                    "distances can fail sharply"
                ),
                "matches": "no stable optimum",
                "alternative_explanation": (
                    "these short scratch-model probes may not identify the distance regime of "
                    "a pretrained long-context model"
                ),
            },
            {
                "hypothesis": "interaction",
                "expected": "larger context and nearer rebinding both improve loss",
                "observed": (
                    "context/composition effects are substantial, but fixed-distance effects "
                    "near the useful range are small and condition-dependent"
                ),
                "matches": "partly",
                "alternative_explanation": "composition dominates positional placement here",
            },
        ],
        "paper1_5_implication": (
            "Source-relative exact placement repairs reset geometry; contextualization and "
            "materialized composition dominate the remaining task behavior. Fixed virtual "
            "distance is a real but fragile control variable, not a demonstrated universal gain."
        ),
        "paper2_gate": (
            "Use post-RoPE exact source coordinates for continuous history. For independent "
            "references, begin with an oracle exact-versus-clipped policy check on the smallest "
            "pretrained model before enabling pre-RoPE rebinding."
        ),
        "adaptive_geometry_gate": "closed: fixed-D evidence does not support broad adaptive implementation",
        "paper1_5_frozen": True,
        "seed_aggregate": aggregate,
    }
    DISTANCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    distance_rows = [row for row in all_rows if row["stage"] == "distance"]
    context_rows = [row for row in all_rows if row["stage"] != "distance"]
    write_json(
        DISTANCE_DIR / "rope_d_sweep.json",
        {
            key: value
            for key, value in summary.items()
            if key != "seed_aggregate"
        }
        | {
            "seed_aggregate": [
                row for row in aggregate if row["stage"] == "distance"
            ],
            "rows": distance_rows,
        },
    )
    write_csv(DISTANCE_DIR / "rope_d_sweep.csv", distance_rows)
    write_csv(
        DISTANCE_DIR / "rope_d_sweep_aggregate.csv",
        [row for row in aggregate if row["stage"] == "distance"],
    )
    write_json(
        CONTEXT_DIR / "rope_context_gate.json",
        summary | {"rows": context_rows},
    )
    write_csv(CONTEXT_DIR / "rope_context_gate.csv", context_rows)
    write_csv(
        CONTEXT_DIR / "rope_encoding_context_sweep.csv",
        [row for row in context_rows if row["stage"] == "encoding_context"],
    )
    write_csv(
        CONTEXT_DIR / "rope_routing_chunk_sweep.csv",
        [row for row in context_rows if row["stage"] == "routing_chunk"],
    )
    write_csv(
        CONTEXT_DIR / "rope_position_policy_gate.csv",
        [row for row in context_rows if row["stage"] == "confirmation"],
    )
    write_csv(
        CONTEXT_DIR / "rope_composition_context_probe.csv",
        [
            row
            for row in context_rows
            if row["stage"] == "confirmation"
            and row["oracle_mode"]
            in {
                "native_oracle",
                "native_evidence_adjacent",
                "native_evidence_irrelevant",
                "native_all",
                "valid",
            }
        ],
    )
    write_csv(
        CONTEXT_DIR / "rope_context_gate_aggregate.csv",
        [row for row in aggregate if row["stage"] != "distance"],
    )
    write_json(CONTEXT_DIR / "rope_context_gate_summary.json", summary)
    _plots(aggregate, best_distance)
    return CONTEXT_DIR / "rope_context_gate_summary.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--datasets", nargs="+", choices=("hotpotqa", "qasper"), default=["hotpotqa", "qasper"])
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=["tiny", "small"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
