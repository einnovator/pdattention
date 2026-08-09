"""Run resumable native-KV routing sensitivity experiments on trained SA checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402
from pra_torch.native_metrics import recovered_context_benefit  # noqa: E402
from pra_torch.pra_train import evaluate_reference_ablation  # noqa: E402
from data.tokenizer import BPETokenizer, PRATokenizer  # noqa: E402
from run_native_kv_benchmark import (  # noqa: E402
    AnswerTokenCollator,
    DATASET_DEFAULTS,
    DATASET_PREPARERS,
    SEEDS,
    _assert_fixed_target_invariants,
    _native_config,
)


RANK_CUTOFFS = (1, 2, 4, 8, 16, 32)
SINGLE_GIST_MODES = {"mean", "last", "ref_end"}


@dataclass(frozen=True)
class SweepConfig:
    """One explicit selector configuration evaluated without transport training."""

    stage: str
    top_k_references: int
    gist_mode: str = "mean"
    gists_per_chunk: int = 1
    gist_score_aggregation: str = "max"
    top_k_chunks_per_reference: int = 1
    search_strategy: str = "hierarchical"
    trigger_threshold: float = float("-inf")
    reference_encoding_strategy: str = "independent"
    encoding_block_references: int = 1
    encoding_overlap_fraction: float = 0.0
    reference_position_mode: str = "local"

    @property
    def config_id(self) -> str:
        base = (
            f"{self.stage}-k{self.top_k_references}-{self.gist_mode}"
            f"-g{self.gists_per_chunk}-{self.gist_score_aggregation}"
        )
        if self.stage != "fragmentation":
            return base
        overlap = str(self.encoding_overlap_fraction).replace(".", "p")
        return (
            f"{base}-{self.reference_encoding_strategy}"
            f"-b{self.encoding_block_references}-o{overlap}-{self.reference_position_mode}"
        )

    @property
    def encoding_key(self) -> tuple:
        """Group settings that can safely share already encoded cache entries."""
        if self.stage in {"topk", "gist"}:
            return ("native-token-kv",)
        return (
            self.reference_encoding_strategy,
            self.encoding_block_references,
            self.encoding_overlap_fraction,
            self.reference_position_mode,
        )

    def pra_overrides(self) -> dict[str, Any]:
        return {
            "top_k_references": self.top_k_references,
            "top_k_chunks_per_reference": self.top_k_chunks_per_reference,
            "search_strategy": self.search_strategy,
            "trigger_threshold": self.trigger_threshold,
            "gist_mode": self.gist_mode,
            "gists_per_chunk": self.gists_per_chunk,
            "gist_score_aggregation": self.gist_score_aggregation,
            "collect_rank_diagnostics": True,
            "reference_encoding_strategy": self.reference_encoding_strategy,
            "encoding_block_references": self.encoding_block_references,
            "encoding_overlap_fraction": self.encoding_overlap_fraction,
            "reference_position_mode": self.reference_position_mode,
        }


def _sweep_configs(
    stage: str,
    split_count: int,
    selected_top_k: int,
    gist_modes: list[str],
    gist_counts: list[int],
    fragmentation_variants: list[str],
) -> list[SweepConfig]:
    if stage == "topk":
        values = (1, 2, 4, 8) if split_count == 32 else (1, 2, 4, 8, 16)
        return [SweepConfig(stage=stage, top_k_references=value) for value in values]
    if stage == "gist":
        configs = [
            SweepConfig(stage=stage, top_k_references=selected_top_k, gist_mode=mode)
            for mode in ("mean", "last")
            if mode in gist_modes
        ]
        configs.extend(
            SweepConfig(
                stage=stage,
                top_k_references=selected_top_k,
                gist_mode=mode,
                gists_per_chunk=count,
            )
            for mode in ("prototype", "kmeans", "som", "hybrid")
            if mode in gist_modes
            for count in gist_counts
        )
        return configs
    if stage == "fragmentation":
        common = {"stage": stage, "top_k_references": selected_top_k}
        variants = {
            "independent": SweepConfig(**common),
            "block4": SweepConfig(
                **common,
                reference_encoding_strategy="block_slice",
                encoding_block_references=4,
            ),
            "block8": SweepConfig(
                **common,
                reference_encoding_strategy="block_slice",
                encoding_block_references=8,
            ),
            "block16": SweepConfig(
                **common,
                reference_encoding_strategy="block_slice",
                encoding_block_references=16,
            ),
            "native": SweepConfig(
                **common,
                reference_encoding_strategy="native_slice",
                encoding_block_references=256,
            ),
            "overlap005": SweepConfig(
                **common,
                reference_encoding_strategy="block_slice",
                encoding_block_references=8,
                encoding_overlap_fraction=0.05,
            ),
            "overlap010": SweepConfig(
                **common,
                reference_encoding_strategy="block_slice",
                encoding_block_references=8,
                encoding_overlap_fraction=0.10,
            ),
            "overlap020": SweepConfig(
                **common,
                reference_encoding_strategy="block_slice",
                encoding_block_references=8,
                encoding_overlap_fraction=0.20,
            ),
            "block8_global": SweepConfig(
                **common,
                reference_encoding_strategy="block_slice",
                encoding_block_references=8,
                reference_position_mode="global",
            ),
        }
        return [variants[name] for name in fragmentation_variants]
    raise ValueError(f"Unsupported stage: {stage}")


def _checkpoint_path(dataset: str, seed: int) -> Path:
    return REPO / "out" / "native_kv_benchmarks" / dataset / f"seed-{seed}" / "checkpoint.pt"


def _load_checkpoint(dataset: str, seed: int) -> dict:
    checkpoint_path = _checkpoint_path(dataset, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained SA checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def _checkpoint_tokenizer(dataset: str, seed: int):
    """Restore the exact per-seed tokenizer paired with a historical checkpoint."""
    checkpoint = _load_checkpoint(dataset, seed)
    if checkpoint.get("tokenizer_json"):
        return BPETokenizer.from_json(checkpoint["tokenizer_json"])
    return PRATokenizer.from_vocab(checkpoint["stoi"])


def _load_source(dataset: str, seed: int, tokenizer, settings: dict, device: str):
    checkpoint_path = _checkpoint_path(dataset, seed)
    checkpoint = _load_checkpoint(dataset, seed)
    if checkpoint.get("settings") != settings:
        raise ValueError(f"Checkpoint settings differ from the {dataset} benchmark defaults")
    source_cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=int(settings["d_model"]),
        n_heads=int(settings["n_heads"]),
        n_layers=int(settings["n_layers"]),
        d_ff=int(settings["d_ff"]),
        max_seq_len=int(settings["max_seq_len"]),
        dropout=0.0,
        model_variant="td_sa",
        device=device,
    )
    source = TinyPRAModel(source_cfg)
    source.load_state_dict(checkpoint["model"])
    return source.to(device).eval(), checkpoint_path


def _baseline(dataset: str, seed: int, split_count: int) -> dict[str, float]:
    path = REPO / "out" / "native_kv_benchmarks" / dataset / f"seed-{seed}" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(value for value in payload["results"] if int(value["split_count"]) == split_count)
    return {
        "sa_full_loss": float(row["sa_full_loss"]),
        "sa_tail_loss": float(row["sa_tail_loss"]),
        "native_oracle_loss": float(row["native_oracle_loss"]),
        "native_oracle_rcb": float(row["rcb_oracle"]),
    }


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _evaluate_config(
    *,
    model,
    datamodule,
    tokenizer,
    device: str,
    encoded_entry_cache: dict,
    config: SweepConfig,
    baseline: dict[str, float],
) -> dict:
    model.cfg.top_k_references = config.top_k_references
    model.cfg.top_k_chunks_per_reference = config.top_k_chunks_per_reference
    changed_gists = (
        model.cfg.gist_mode != config.gist_mode
        or model.cfg.gists_per_chunk != config.gists_per_chunk
    )
    model.cfg.gist_mode = config.gist_mode
    model.cfg.gists_per_chunk = config.gists_per_chunk
    model.cfg.gist_score_aggregation = config.gist_score_aggregation
    model.cfg.search_strategy = config.search_strategy
    model.cfg.trigger_threshold = config.trigger_threshold
    model.cfg.reference_encoding_strategy = config.reference_encoding_strategy
    model.cfg.encoding_block_references = config.encoding_block_references
    model.cfg.encoding_overlap_fraction = config.encoding_overlap_fraction
    model.cfg.reference_position_mode = config.reference_position_mode
    if changed_gists and encoded_entry_cache:
        from pra_torch.memory import PRASimpleMemoryCache

        cache = PRASimpleMemoryCache()
        for entry in encoded_entry_cache.values():
            cache.put(entry)
        model.rebuild_cache_routing_gists(cache, tokenizer=tokenizer)
    oracle_result = None
    if config.stage == "fragmentation":
        oracle_result = evaluate_reference_ablation(
            model=model,
            loader=datamodule.test_loader(),
            tokenizer=tokenizer,
            device=device,
            condition="native_oracle",
            collect_per_example=True,
            encoded_entry_cache=encoded_entry_cache,
        )
    result = evaluate_reference_ablation(
        model=model,
        loader=datamodule.test_loader(),
        tokenizer=tokenizer,
        device=device,
        condition="valid",
        collect_per_example=True,
        encoded_entry_cache=encoded_entry_cache,
    )
    rows = result["per_example"]
    routed_rcb = recovered_context_benefit(
        sa_full_loss=baseline["sa_full_loss"],
        sa_tail_loss=baseline["sa_tail_loss"],
        pra_loss=float(result["loss"]),
    )
    summary = {
        **baseline,
        "native_routed_loss": float(result["loss"]),
        "native_routed_accuracy": float(result["token_accuracy"]),
        "routed_rcb": routed_rcb,
        "duration_seconds": float(result["duration_seconds"]),
    }
    if oracle_result is not None:
        summary["native_oracle_loss"] = float(oracle_result["loss"])
        summary["native_oracle_accuracy"] = float(oracle_result["token_accuracy"])
        summary["native_oracle_rcb"] = recovered_context_benefit(
            sa_full_loss=baseline["sa_full_loss"],
            sa_tail_loss=baseline["sa_tail_loss"],
            pra_loss=float(oracle_result["loss"]),
        )
        summary["oracle_duration_seconds"] = float(oracle_result["duration_seconds"])
    for key in (
        "active_fraction",
        "active_unique_fraction",
        "retrieved_physical_kv_tokens",
        "retrieved_unique_source_tokens",
        "kv_transfer_bytes",
        "routing_latency",
        "attention_latency",
        "routing_mrr",
        "routing_score_margin",
        "gist_comparisons",
        "recall_at_k",
    ):
        summary[key] = _mean(rows, key)
    for cutoff in RANK_CUTOFFS:
        for prefix in (
            "reference_recall_at",
            "any_target_hit_at",
            "all_targets_hit_at",
            "fraction_targets_covered_at",
        ):
            key = f"{prefix}_{cutoff}"
            summary[key] = _mean(rows, key)
    return {
        "summary": summary,
        "per_example": [
            *(oracle_result["per_example"] if oracle_result is not None else []),
            *rows,
        ],
    }


def _write_seed_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _flatten_results(root: Path, stage: str, datasets: list[str], seeds: list[int]):
    payloads = []
    for path in root.glob("*/seed-*/*/split-*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload["stage"] == stage
            and payload["dataset"] in datasets
            and payload["seed"] in seeds
        ):
            routed_rows = [
                row for row in payload.get("per_example", []) if row.get("condition") == "valid"
            ]
            for key in (
                "unique_source_tokens",
                "encoded_tokens_including_overlap",
                "stored_kv_tokens_including_overlap",
                "duplication_factor",
                "chunk_overlap_fraction",
                "chunk_overlap_tokens",
            ):
                if payload["summary"].get(key) is None:
                    payload["summary"][key] = _mean(routed_rows, key)
            payloads.append(payload)
    return payloads


def _aggregate(payloads: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for payload in payloads:
        key = (payload["dataset"], payload["split_count"], payload["config"]["config_id"])
        groups[key].append(payload)
    rows = []
    for (dataset, split_count, config_id), values in sorted(groups.items()):
        config = values[0]["config"]
        row = {"dataset": dataset, "split_count": split_count, **config, "seeds": len(values)}
        keys = sorted({key for value in values for key in value["summary"]})
        for key in keys:
            numbers = [
                float(value["summary"][key])
                for value in values
                if value["summary"].get(key) is not None
            ]
            if numbers:
                row[f"{key}_mean"] = statistics.fmean(numbers)
                row[f"{key}_stddev"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(report_dir: Path, payloads: list[dict], aggregate: list[dict], manifest: dict):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "parameter_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    raw_runs = []
    ranking_rows = []
    for payload in payloads:
        base = {
            "dataset": payload["dataset"],
            "seed": payload["seed"],
            "split_count": payload["split_count"],
            **payload["config"],
        }
        for row in payload["per_example"]:
            raw_runs.append({**base, **row})
            for layer_id, candidates in row.get("candidate_rankings_by_layer", {}).items():
                target_uris = set(row.get("oracle_selected_reference_uris") or [])
                for candidate in candidates:
                    ranking_rows.append(
                        {
                            **base,
                            "example_id": row["example_id"],
                            "layer_id": layer_id,
                            "reference_uri": candidate["reference_uri"],
                            "reference_rank": candidate["reference_rank"],
                            "reference_score": candidate["reference_score"],
                            "is_target": candidate["reference_uri"] in target_uris,
                        }
                    )
    _write_csv(report_dir / "raw_runs.csv", raw_runs)
    _write_csv(report_dir / "raw_rankings.csv", ranking_rows)
    _write_csv(report_dir / "aggregate_pareto.csv", aggregate)
    stage = manifest["stage"]
    _write_csv(report_dir / f"aggregate_by_{stage}.csv", aggregate)
    report_payload = {"manifest": manifest, "aggregate": aggregate}
    (report_dir / "aggregate.json").write_text(
        json.dumps(report_payload, indent=2), encoding="utf-8"
    )
    _plot(report_dir, aggregate, stage)


def _plot(report_dir: Path, rows: list[dict], stage: str) -> None:
    if not rows:
        return
    datasets = sorted({row["dataset"] for row in rows})
    splits = sorted({int(row["split_count"]) for row in rows})
    figure, axes = plt.subplots(
        len(datasets),
        len(splits),
        figsize=(5.2 * len(splits), 4 * len(datasets)),
        squeeze=False,
    )
    for row_index, dataset in enumerate(datasets):
        for column_index, split_count in enumerate(splits):
            axis = axes[row_index][column_index]
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and int(row["split_count"]) == split_count
            ]
            if stage == "topk":
                selected.sort(key=lambda row: int(row["top_k_references"]))
                axis.plot(
                    [row["active_fraction_mean"] for row in selected],
                    [row["routed_rcb_mean"] for row in selected],
                    marker="o",
                )
                for row in selected:
                    axis.annotate(
                        f"k={row['top_k_references']}",
                        (row["active_fraction_mean"], row["routed_rcb_mean"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
                axis.set_xlabel("Active physical K/V fraction")
                axis.set_ylabel("Routed RCB")
            elif stage == "gist":
                labels = [
                    f"{row['gist_mode']}:{row['gists_per_chunk']}\n(n={row['seeds']})"
                    for row in selected
                ]
                colors = ["#2878B5" if int(row["seeds"]) > 1 else "#A8B0B8" for row in selected]
                axis.bar(
                    range(len(selected)),
                    [row["routing_mrr_mean"] for row in selected],
                    color=colors,
                )
                axis.set_xticks(range(len(selected)), labels, rotation=70, ha="right", fontsize=7)
                axis.set_ylabel("Routing MRR")
            else:
                labels = [
                    (
                        f"{row['reference_encoding_strategy']}"
                        f"\nb={row['encoding_block_references']}"
                        f", o={float(row['encoding_overlap_fraction']):.2f}"
                        f"\n{row['reference_position_mode']}, n={row['seeds']}"
                    )
                    for row in selected
                ]
                positions = list(range(len(selected)))
                width = 0.38
                axis.bar(
                    [value - width / 2 for value in positions],
                    [row["native_oracle_rcb_mean"] for row in selected],
                    width=width,
                    label="Oracle",
                    color="#2A7F62",
                )
                axis.bar(
                    [value + width / 2 for value in positions],
                    [row["routed_rcb_mean"] for row in selected],
                    width=width,
                    label="Routed",
                    color="#C85A3E",
                )
                axis.set_xticks(positions, labels, rotation=65, ha="right", fontsize=7)
                axis.set_ylabel("Recovered Context Benefit")
                axis.legend(fontsize=8)
            axis.set_title(f"{dataset}, split {split_count}")
            axis.grid(alpha=0.25)
    title = (
        f"PRA {stage} sensitivity, successive halving (seed count shown)"
        if stage in {"gist", "fragmentation"}
        else f"PRA {stage} sensitivity, five paired seeds"
    )
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(report_dir / f"{stage}_sensitivity.pdf", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = REPO / "out" / "pra_parameter_sensitivity"
    report_dir = REPO / "out" / "reports" / f"pra_parameter_sensitivity_{args.stage}"
    for dataset in args.datasets:
        settings = dict(DATASET_DEFAULTS[dataset])
        _generated_tokenizer, _training_module, modules = DATASET_PREPARERS[dataset](settings)
        modules = {split: modules[split] for split in args.splits}
        _assert_fixed_target_invariants(modules)
        for seed in args.seeds:
            tokenizer = _checkpoint_tokenizer(dataset, seed)
            for datamodule in modules.values():
                datamodule.tokenizer = tokenizer
                datamodule.collator = AnswerTokenCollator(
                    tokenizer, max_seq_len=int(settings["max_seq_len"])
                )
            source, checkpoint_path = _load_source(dataset, seed, tokenizer, settings, device)
            configs_by_encoding = defaultdict(list)
            for split_count in args.splits:
                for config in _sweep_configs(
                    args.stage,
                    split_count,
                    args.selected_top_k,
                    args.gist_modes,
                    args.gist_counts,
                    args.fragmentation_variants,
                ):
                    configs_by_encoding[(split_count, config.encoding_key)].append(config)
            for (split_count, _encoding_key), configs in configs_by_encoding.items():
                first = configs[0]
                converted = convert_sa_model_to_pra(
                    source,
                    _native_config(source, device, first.pra_overrides()),
                ).to(device).eval()
                encoded_entry_cache = {}
                baseline = _baseline(dataset, seed, split_count)
                for config in sorted(configs, key=lambda value: value.top_k_references):
                    result_path = (
                        root
                        / dataset
                        / f"seed-{seed}"
                        / config.config_id
                        / f"split-{split_count}.json"
                    )
                    if result_path.exists() and not args.force:
                        print(f"reuse {result_path.relative_to(REPO)}", flush=True)
                        continue
                    print(
                        f"dataset={dataset} seed={seed} split={split_count} "
                        f"config={config.config_id}",
                        flush=True,
                    )
                    evaluated = _evaluate_config(
                        model=converted,
                        datamodule=modules[split_count],
                        tokenizer=tokenizer,
                        device=device,
                        encoded_entry_cache=encoded_entry_cache,
                        config=config,
                        baseline=baseline,
                    )
                    payload = {
                        "stage": args.stage,
                        "dataset": dataset,
                        "seed": seed,
                        "split_count": split_count,
                        "checkpoint": str(checkpoint_path),
                        "device": device,
                        "config": {"config_id": config.config_id, **config.__dict__},
                        **evaluated,
                    }
                    _write_seed_result(result_path, payload)
                del converted
                if device == "cuda":
                    torch.cuda.empty_cache()
            del source
    payloads = _flatten_results(root, args.stage, args.datasets, args.seeds)
    aggregate = _aggregate(payloads)
    manifest = {
        "stage": args.stage,
        "datasets": args.datasets,
        "splits": args.splits,
        "seeds": args.seeds,
        "device": device,
        "selected_top_k": args.selected_top_k,
        "requested_gist_modes": args.gist_modes,
        "requested_gist_counts": args.gist_counts,
        "requested_fragmentation_variants": args.fragmentation_variants,
        "evaluated_config_ids": sorted({row["config_id"] for row in aggregate}),
        "protocol": "inference-only selector sweep over frozen five-seed SA checkpoints",
    }
    _write_artifacts(report_dir, payloads, aggregate, manifest)
    if args.publish:
        result_dir = REPO / "docs" / "papers" / "shared" / "results"
        figure_dir = REPO / "docs" / "papers" / "shared" / "figures"
        result_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            report_dir / "aggregate.json",
            result_dir / f"pra_parameter_sensitivity_{args.stage}.json",
        )
        shutil.copy2(
            report_dir / f"aggregate_by_{args.stage}.csv",
            result_dir / f"pra_parameter_sensitivity_{args.stage}.csv",
        )
        shutil.copy2(
            report_dir / f"{args.stage}_sensitivity.pdf",
            figure_dir / f"pra_parameter_sensitivity_{args.stage}.pdf",
        )
    return report_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("topk", "gist", "fragmentation"), required=True
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("hotpotqa", "qasper"),
        default=["hotpotqa", "qasper"],
    )
    parser.add_argument("--splits", nargs="+", type=int, choices=(32, 64), default=[32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--selected-top-k", type=int, default=2)
    parser.add_argument(
        "--gist-modes",
        nargs="+",
        choices=("mean", "last", "prototype", "kmeans", "som", "hybrid"),
        default=["mean", "last", "prototype", "kmeans", "som", "hybrid"],
    )
    parser.add_argument("--gist-counts", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--fragmentation-variants",
        nargs="+",
        choices=(
            "independent",
            "block4",
            "block8",
            "block16",
            "native",
            "overlap005",
            "overlap010",
            "overlap020",
            "block8_global",
        ),
        default=[
            "independent",
            "block4",
            "block8",
            "block16",
            "native",
            "overlap005",
            "overlap010",
            "overlap020",
            "block8_global",
        ],
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
