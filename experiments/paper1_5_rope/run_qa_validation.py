"""Run matched HotpotQA/QASPER positional and composition validation."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

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


VALIDATION = RESULTS / "validation"
MODES = ("absolute", "sinusoidal", "rope")
STAGES = ("reset", "offset", "offset_overlap")
COLORS = {"absolute": "#245A8D", "sinusoidal": "#327A5A", "rope": "#A34832"}


def _base_settings(dataset: str, smoke: bool, max_examples: int) -> dict:
    values = dict(native.DATASET_DEFAULTS[dataset])
    values.update(
        {
            "max_examples": 4 if smoke else max_examples,
            "train_examples": 32
            if smoke
            else (1_000 if dataset == "hotpotqa" else values["train_examples"]),
            "validation_examples": 4 if smoke else 32,
        }
    )
    return values


def _settings(base: dict, tier: str, mode: str, stage: str, smoke: bool) -> dict:
    overlap = 0.25 if stage == "offset_overlap" else 0.0
    return {
        **base,
        **TIERS[tier],
        "position_encoding": mode,
        "batch_size": 2 if smoke else (16 if tier == "tiny" else 8),
        "steps": 2 if smoke else TIERS[tier]["steps"],
        "pra_overrides": {
            "reference_encoding_strategy": "block_slice",
            "encoding_block_references": 1,
            "reference_position_mode": "local" if stage == "reset" else "global",
            "encoding_overlap_fraction": overlap,
            "max_gists_per_reference": 128,
            "top_k_references": 2,
            "top_k_chunks_per_reference": 1,
            "trigger_threshold": float("-inf"),
            "max_materialized_memory_tokens": 160,
            "context_safety_reserve_tokens": 4,
            "collect_routing_metrics": True,
            "collect_rank_diagnostics": True,
        },
    }


def _train_settings(base: dict, tier: str, mode: str, smoke: bool) -> dict:
    values = _settings(base, tier, mode, "reset", smoke)
    values.pop("pra_overrides")
    return values


def _flatten_rows(
    raw_rows: list[dict], *, metadata: dict, dataset: str, tier: str, mode: str, stage: str, seed: int
) -> list[dict]:
    output = []
    for row in raw_rows:
        if int(row["split_count"]) != 5:
            continue
        output.append(
            {
                "git_sha": metadata["git_sha"],
                "dataset": dataset,
                "seed": seed,
                "model_tier": tier,
                "position_mode": mode,
                "stage": stage,
                "k_storage_mode": "post_position",
                "logical_offset_policy": "reset" if stage == "reset" else "source_relative",
                "overlap_fraction": 0.25 if stage == "offset_overlap" else 0.0,
                **row,
            }
        )
    return output


def _aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_tier"], row["position_mode"], row["stage"], row["condition"])].append(row)
    output = []
    metrics = (
        "loss",
        "token_accuracy",
        "rcb_routed",
        "recall_at_k",
        "routing_mrr",
        "fraction_targets_covered_at_2",
        "num_selected_chunks",
        "retrieved_physical_kv_tokens",
        "logical_native_ratio",
        "maximum_native_operation",
        "native_limit_violations",
        "duplication_factor",
    )
    for identity, values in sorted(groups.items()):
        result = dict(zip(("model_tier", "position_mode", "stage", "condition"), identity))
        result["seed_count"] = len({row["seed"] for row in values})
        result["example_count"] = len(values)
        for metric in metrics:
            observed = [
                float(row[metric])
                for row in values
                if row.get(metric) is not None and isinstance(row.get(metric), (int, float))
            ]
            if observed:
                result[f"{metric}_mean"] = statistics.fmean(observed)
                result[f"{metric}_median"] = statistics.median(observed)
                result[f"{metric}_std"] = statistics.pstdev(observed)
        output.append(result)
    return output


def _composition(rows: list[dict], dataset: str, metadata: dict) -> list[dict]:
    selected = [row for row in rows if row["stage"] == "offset_overlap"]
    by_example = defaultdict(dict)
    for row in selected:
        by_example[(row["model_tier"], row["position_mode"], row["seed"], row["example_id"])][
            row["condition"]
        ] = row
    output = []
    for identity, conditions in sorted(by_example.items()):
        if not {"native_oracle", "native_all", "native_routed", "native_shuffled"} <= conditions.keys():
            continue
        oracle = conditions["native_oracle"]
        all_memory = conditions["native_all"]
        routed = conditions["native_routed"]
        shuffled = conditions["native_shuffled"]
        output.append(
            {
                "git_sha": metadata["git_sha"],
                "dataset": dataset,
                "model_tier": identity[0],
                "position_mode": identity[1],
                "seed": identity[2],
                "example_id": identity[3],
                "evidence_only_loss": oracle["loss"],
                "all_memory_loss": all_memory["loss"],
                "router_selected_loss": routed["loss"],
                "shuffled_selection_loss": shuffled["loss"],
                "all_minus_evidence_loss": all_memory["loss"] - oracle["loss"],
                "routed_minus_evidence_loss": routed["loss"] - oracle["loss"],
                "nominal_oracle_worse_than_routed": int(oracle["loss"] > routed["loss"]),
                "evidence_plus_one_irrelevant": "not_tested",
            }
        )
    return output


def _plot(rows: list[dict], dataset: str, path: Path) -> None:
    present_tiers = [tier for tier in ("tiny", "small") if any(row["model_tier"] == tier for row in rows)]
    present_modes = [mode for mode in MODES if any(row["position_mode"] == mode for row in rows)]
    figure, axes = plt.subplots(
        len(present_tiers), len(present_modes), figsize=(3.7 * len(present_modes), 3.2 * len(present_tiers)), squeeze=False
    )
    for row_index, tier in enumerate(present_tiers):
        for column, mode in enumerate(present_modes):
            axis = axes[row_index, column]
            for condition, marker in (("native_routed", "o"), ("native_oracle", "s")):
                means = []
                for stage in STAGES:
                    observed = [
                        row["loss"]
                        for row in rows
                        if row["model_tier"] == tier
                        and row["position_mode"] == mode
                        and row["stage"] == stage
                        and row["condition"] == condition
                    ]
                    means.append(statistics.fmean(observed))
                axis.plot(STAGES, means, marker=marker, color=COLORS[mode], label=condition.removeprefix("native_"))
            axis.set_title(f"{tier} / {mode}")
            axis.tick_params(axis="x", rotation=18)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("Answer-token loss")
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{dataset} positional validation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args) -> Path:
    metadata = environment_metadata()
    base = _base_settings(args.dataset, args.smoke, args.max_examples)
    preparer = native.DATASET_PREPARERS[args.dataset]
    tokenizer, training_module, all_modules = preparer(base)
    modules = {split: all_modules[split] for split in (2, 5)}
    training_rows = []
    rows = []
    for tier in args.tiers:
        for mode in args.position_modes:
            train_settings = _train_settings(base, tier, mode, args.smoke)
            for seed in args.seeds:
                set_seed(seed)
                run_dir = (
                    REPO
                    / "out"
                    / "paper1_5_rope"
                    / "validation"
                    / args.dataset
                    / tier
                    / mode
                    / f"seed-{seed}"
                )
                checkpoint = run_dir / "checkpoint.pt"
                reused = checkpoint.exists() and not args.force
                source, training = native.train_full_context_sa(
                    seed=seed,
                    tokenizer=tokenizer,
                    datamodule=training_module,
                    settings=train_settings,
                    run_dir=run_dir,
                    device=args.device,
                    force=args.force,
                )
                history = training["history"]
                training_rows.append(
                    {
                        "git_sha": metadata["git_sha"],
                        "dataset": args.dataset,
                        "seed": seed,
                        "model_tier": tier,
                        "position_mode": mode,
                        "parameter_count": sum(p.numel() for p in source.parameters()),
                        "training_examples": train_settings["train_examples"],
                        "training_tokens": train_settings["steps"]
                        * train_settings["batch_size"]
                        * train_settings["max_seq_len"],
                        "native_training_context": train_settings["max_seq_len"],
                        "optimizer": "AdamW",
                        "scheduler": "constant",
                        "batch_size": train_settings["batch_size"],
                        "steps": training["final_step"],
                        "final_train_loss": history[-1]["train_loss"] if history else None,
                        "validation_loss": history[-1]["validation_loss"] if history else None,
                        "checkpoint": training["checkpoint"],
                        "checkpoint_reused": reused,
                    }
                )
                for stage in STAGES:
                    eval_settings = _settings(base, tier, mode, stage, args.smoke)
                    _, raw = native.evaluate_seed(
                        source=source,
                        tokenizer=tokenizer,
                        modules=modules,
                        settings=eval_settings,
                        device=args.device,
                    )
                    rows.extend(
                        _flatten_rows(
                            raw,
                            metadata=metadata,
                            dataset=args.dataset,
                            tier=tier,
                            mode=mode,
                            stage=stage,
                            seed=seed,
                        )
                    )
                del source
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    aggregate = _aggregate(rows)
    composition = _composition(rows, args.dataset, metadata)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_position_validation"
    path = VALIDATION / f"{stem}.json"
    write_json(
        path,
        {
            "metadata": metadata,
            "smoke": args.smoke,
            "expectations_recorded_before_analysis": [
                {
                    "expected": "positional fidelity improves with source-relative offsets",
                    "reason": "grouped one-reference blocks otherwise reset their source coordinates",
                },
                {
                    "expected": "routing and composition remain independently limiting",
                    "reason": "correct K positions do not determine which chunk set is useful",
                },
                {
                    "expected": "overlap need not improve every seed or mechanism",
                    "reason": "additional left context and distractor composition interact",
                },
            ],
            "protocol": "controlled answer-code/reference probe, not unrestricted QA",
            "training": training_rows,
            "rows": rows,
            "aggregate": aggregate,
        },
    )
    write_csv(VALIDATION / f"{args.dataset}_training.csv", training_rows)
    write_csv(VALIDATION / f"{stem}.csv", rows)
    write_csv(VALIDATION / f"{stem}_aggregate.csv", aggregate)
    write_json(
        VALIDATION / f"composition_probe_{args.dataset}.json",
        {"metadata": metadata, "rows": composition},
    )
    write_csv(VALIDATION / f"composition_probe_{args.dataset}.csv", composition)
    _plot(rows, args.dataset, VALIDATION / f"{stem}.png")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hotpotqa", "qasper"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument("--position-modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
