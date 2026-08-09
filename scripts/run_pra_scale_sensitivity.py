"""Run resumable native-KV model-capacity and meaningful-context scale probes.

Unlike the original fixed-63-unit partition sweep, this experiment grows both
the source and the number of independently addressable regions. References are
indexed separately but obtain K/V from one causal historical encode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data.datamodules import PRADataModule  # noqa: E402
from data.datasets import (  # noqa: E402
    HotpotQANativeKVFixedTargetDataset,
    QASPERNativeKVFixedTargetDataset,
)
from data.native_kv_benchmarks import (  # noqa: E402
    NativeKVBenchmarkExample,
    hotpotqa_native_kv_examples,
    load_qasper_papers,
    qasper_native_kv_examples,
    write_native_kv_benchmark,
)
from data.tokenizer import BPETokenizer  # noqa: E402
from pra_torch.model import convert_sa_model_to_pra  # noqa: E402
from pra_torch.native_metrics import recovered_context_benefit  # noqa: E402
from pra_torch.pra_train import evaluate_reference_ablation  # noqa: E402
from run_native_kv_benchmark import (  # noqa: E402
    AnswerTokenCollator,
    FullContextDataset,
    SEEDS,
    _evaluate_model,
    _loader,
    _native_config,
    _set_seed,
    _subset_indices,
    train_full_context_sa,
)


SCALE_SPLITS = (32, 64, 128, 256)
RANK_CUTOFFS = (1, 2, 4, 8, 16, 32)
DATASET_SETTINGS = {
    "hotpotqa": {
        "train_examples": 4_000,
        "max_examples": 64,
        "vocab_size": 8_000,
        "learning_rate": 7e-4,
    },
    "qasper": {
        "train_examples": 200,
        "max_examples": 64,
        "vocab_size": 8_000,
        "learning_rate": 7e-4,
    },
}
MODEL_TIERS = {
    "tiny": {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "batch_size": 8,
        "steps": 200,
    },
    "small": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "d_ff": 768,
        "batch_size": 4,
        "steps": 300,
    },
}
MAX_SEQ_LEN = 768
GENERATION_VERSION = "nativekv_context_scale_v1"


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _truncate_example(
    example: NativeKVBenchmarkExample, source_unit_count: int
) -> NativeKVBenchmarkExample:
    """Return a nested source prefix while retaining the same target and ID."""
    if len(example.source_units) < source_unit_count:
        raise ValueError(
            f"Example {example.id} has {len(example.source_units)} units; "
            f"{source_unit_count} required."
        )
    return replace(
        example,
        source_units=example.source_units[:source_unit_count],
        metadata={**example.metadata, "scale_source_unit_count": source_unit_count},
    )


def _dataset_spec(dataset: str):
    if dataset == "hotpotqa":
        from datasets import load_dataset

        train_rows = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="train",
            cache_dir=str(REPO / "out" / "hf_cache"),
        )
        eval_rows = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="validation",
            cache_dir=str(REPO / "out" / "hf_cache"),
        )
        converter = hotpotqa_native_kv_examples
        dataset_class = HotpotQANativeKVFixedTargetDataset
        train_seed, eval_seed = 31_337, 72_991
    elif dataset == "qasper":
        train_rows = load_qasper_papers(
            "train", cache_dir=REPO / "out" / "hf_cache" / "qasper"
        )
        eval_rows = load_qasper_papers(
            "validation", cache_dir=REPO / "out" / "hf_cache" / "qasper"
        )
        converter = qasper_native_kv_examples
        dataset_class = QASPERNativeKVFixedTargetDataset
        train_seed, eval_seed = 44_321, 82_811
    else:
        raise ValueError(f"Unsupported scale dataset: {dataset}")
    return train_rows, eval_rows, converter, dataset_class, train_seed, eval_seed


def _dataset_matches(
    root: Path,
    dataset_class,
    *,
    split_count: int,
    count: int,
    source_unit_count: int,
) -> bool:
    manifest_path = root / dataset_class.stage / "manifest.json"
    questions_path = root / dataset_class.stage / "questions.jsonl"
    if not manifest_path.exists() or not questions_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != {
        "dataset": dataset_class.stage.removesuffix("_nativekv_fixed_target"),
        "split_count": split_count,
        "example_count": count,
        "generation_version": GENERATION_VERSION,
    }:
        return False
    with questions_path.open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    return int(first["source_unit_count"]) == source_unit_count


def _write_dataset(
    root: Path,
    *,
    dataset: str,
    dataset_class,
    split_count: int,
    examples: list[NativeKVBenchmarkExample],
) -> None:
    if _dataset_matches(
        root,
        dataset_class,
        split_count=split_count,
        count=len(examples),
        source_unit_count=len(examples[0].source_units),
    ):
        return
    write_native_kv_benchmark(
        root,
        stage=dataset_class.stage,
        dataset_name=dataset,
        split_count=split_count,
        examples=examples,
        generation_version=GENERATION_VERSION,
    )


def prepare_scale_data(dataset: str):
    """Build one 255-unit training set and nested 63/127/255-unit evaluations."""
    settings = DATASET_SETTINGS[dataset]
    max_units = max(SCALE_SPLITS) - 1
    dataset_class = (
        HotpotQANativeKVFixedTargetDataset
        if dataset == "hotpotqa"
        else QASPERNativeKVFixedTargetDataset
    )
    train_seed, eval_seed = (
        (31_337, 72_991) if dataset == "hotpotqa" else (44_321, 82_811)
    )
    root = REPO / "out" / "pra_scale_data" / dataset
    training_root = root / "training-source-255"
    cached = _dataset_matches(
        training_root,
        dataset_class,
        split_count=2,
        count=int(settings["train_examples"]),
        source_unit_count=max_units,
    )
    cached = cached and all(
        _dataset_matches(
            root / f"source-{split_count - 1}-split-{split_count}",
            dataset_class,
            split_count=split_count,
            count=int(settings["max_examples"]),
            source_unit_count=split_count - 1,
        )
        for split_count in SCALE_SPLITS
    )
    if not cached:
        train_rows, eval_rows, converter, _, _, _ = _dataset_spec(dataset)
        training_examples = converter(
            train_rows,
            max_examples=int(settings["train_examples"]),
            seed=train_seed,
            source_unit_count=max_units,
        )
        evaluation_examples = converter(
            eval_rows,
            max_examples=int(settings["max_examples"]),
            seed=eval_seed,
            source_unit_count=max_units,
        )
        if len(training_examples) < int(settings["train_examples"]):
            raise ValueError(
                f"{dataset} produced {len(training_examples)} scale training examples; "
                f"{settings['train_examples']} requested."
            )
        if len(evaluation_examples) < int(settings["max_examples"]):
            raise ValueError(
                f"{dataset} produced {len(evaluation_examples)} scale examples; "
                f"{settings['max_examples']} requested."
            )
        _write_dataset(
            training_root,
            dataset=dataset,
            dataset_class=dataset_class,
            split_count=2,
            examples=training_examples,
        )
        for split_count in SCALE_SPLITS:
            source_units = split_count - 1
            _write_dataset(
                root / f"source-{source_units}-split-{split_count}",
                dataset=dataset,
                dataset_class=dataset_class,
                split_count=split_count,
                examples=[
                    _truncate_example(example, source_units)
                    for example in evaluation_examples
                ],
            )
    training_dataset = dataset_class(training_root)
    tokenizer_path = root / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.from_json(tokenizer_path.read_text(encoding="utf-8"))
    else:
        tokenizer = BPETokenizer.train(
            PRADataModule._corpus(training_dataset),
            vocab_size=int(settings["vocab_size"]),
            min_frequency=2,
        )
        tokenizer_path.write_text(tokenizer.to_json(), encoding="utf-8")
    training_module = PRADataModule(
        dataset_stage=dataset_class.stage,
        data_dir=str(training_root),
        max_examples=int(settings["train_examples"]),
        batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        shuffle=True,
        tokenizer=tokenizer,
        split_seed=train_seed,
    ).load()
    # The scale probe needs a stable quality signal, not a validation pass over
    # hundreds of near-identical answer-code examples at every checkpoint.
    validation_indices = _subset_indices(training_module, "val")[:64]
    training_module.val_dataset = Subset(training_module.dataset, validation_indices)
    modules = {}
    fixed_ids = None
    for split_count in SCALE_SPLITS:
        source_units = split_count - 1
        split_root = root / f"source-{source_units}-split-{split_count}"
        datamodule = PRADataModule(
            dataset_stage=dataset_class.stage,
            data_dir=str(split_root),
            max_examples=int(settings["max_examples"]),
            batch_size=1,
            max_seq_len=MAX_SEQ_LEN,
            shuffle=False,
            tokenizer=tokenizer,
            split_seed=train_seed,
        ).load()
        datamodule.collator = AnswerTokenCollator(tokenizer, max_seq_len=MAX_SEQ_LEN)
        datamodule.test_dataset = datamodule.dataset
        ids = [(sample.id, sample.answer) for sample in datamodule.dataset]
        if fixed_ids is not None and ids != fixed_ids:
            raise AssertionError("Scale variants changed example order, IDs, or answers.")
        fixed_ids = ids
        modules[split_count] = datamodule

    lengths = [
        len(tokenizer.encode(sample.metadata["row"]["source_text"] + sample.question))
        for sample in training_dataset
    ]
    for datamodule in modules.values():
        lengths.extend(
            len(tokenizer.encode(sample.metadata["row"]["source_text"] + sample.question))
            for sample in datamodule.dataset
        )
    if max(lengths, default=0) > MAX_SEQ_LEN:
        raise ValueError(
            f"{dataset} scale prompt needs {max(lengths)} tokens; limit is {MAX_SEQ_LEN}."
        )
    print(
        f"prepared {dataset}: train={len(training_dataset)} "
        f"eval={len(modules[min(modules)].dataset)} "
        f"max_tokens={max(lengths)}",
        flush=True,
    )
    return tokenizer, training_module, modules


def _baseline_results(model, tokenizer, datamodule, device: str) -> tuple[dict, dict]:
    collator = AnswerTokenCollator(tokenizer, max_seq_len=MAX_SEQ_LEN)
    evaluation_dataset = datamodule.test_dataset
    full_loader = _loader(
        FullContextDataset(evaluation_dataset),
        collator,
        batch_size=1,
        shuffle=False,
        seed=0,
    )
    tail_loader = _loader(
        evaluation_dataset,
        collator,
        batch_size=1,
        shuffle=False,
        seed=0,
    )
    return (
        _evaluate_model(model, full_loader, device, condition="sa_full", tokenizer=tokenizer),
        _evaluate_model(model, tail_loader, device, condition="sa_tail", tokenizer=tokenizer),
    )


def _summary_metrics(rows: list[dict]) -> dict[str, float | None]:
    keys = (
        "accessible_tokens",
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
        "all_targets_hit_at_1",
        "all_targets_hit_at_2",
        "all_targets_hit_at_4",
        "all_targets_hit_at_8",
        "all_targets_hit_at_16",
        "all_targets_hit_at_32",
        "fraction_targets_covered_at_1",
        "fraction_targets_covered_at_2",
        "fraction_targets_covered_at_4",
        "fraction_targets_covered_at_8",
        "fraction_targets_covered_at_16",
        "fraction_targets_covered_at_32",
        "peak_cuda_memory",
        "unique_source_tokens",
        "encoded_tokens_including_overlap",
        "stored_kv_tokens_including_overlap",
        "duplication_factor",
    )
    return {key: _mean(rows, key) for key in keys}


def evaluate_scale(
    *,
    source,
    tokenizer,
    datamodule,
    split_count: int,
    top_k_values: list[int],
    device: str,
) -> tuple[list[dict], list[dict]]:
    """Evaluate dense controls, native all/oracle, and a routed top-k frontier."""
    sa_full, sa_tail = _baseline_results(source, tokenizer, datamodule, device)
    cfg = _native_config(
        source,
        device,
        {
            "reference_encoding_strategy": "native_slice",
            "reference_position_mode": "global",
            "prompt_position_mode": "historical",
            # Aggregate rank/coverage metrics are computed from the complete
            # in-memory ranking. Persisting every candidate again for every k
            # makes a single seed hundreds of megabytes without adding evidence.
            "collect_rank_diagnostics": False,
            "collect_routing_metrics": True,
            "recursive_max_total_references": 512,
            "recursive_max_total_tokens": 65_536,
            "top_k_references": max(top_k_values),
        },
    )
    model = convert_sa_model_to_pra(source, cfg).to(device).eval()
    encoded_entry_cache: dict = {}
    loader = datamodule.test_loader()
    native_all = evaluate_reference_ablation(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        condition="native_all",
        collect_per_example=True,
        encoded_entry_cache=encoded_entry_cache,
    )
    native_oracle = evaluate_reference_ablation(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
        condition="native_oracle",
        collect_per_example=True,
        encoded_entry_cache=encoded_entry_cache,
    )
    oracle_rcb = recovered_context_benefit(
        sa_full_loss=sa_full["loss"],
        sa_tail_loss=sa_tail["loss"],
        pra_loss=native_oracle["loss"],
    )
    rows = []
    raw = []
    for top_k in top_k_values:
        model.cfg.top_k_references = top_k
        routed = evaluate_reference_ablation(
            model=model,
            loader=loader,
            tokenizer=tokenizer,
            device=device,
            condition="valid",
            collect_per_example=True,
            encoded_entry_cache=encoded_entry_cache,
        )
        row = {
            "split_count": split_count,
            "source_unit_count": split_count - 1,
            "top_k_references": top_k,
            "sa_full_loss": sa_full["loss"],
            "sa_full_accuracy": sa_full["token_accuracy"],
            "sa_tail_loss": sa_tail["loss"],
            "sa_tail_accuracy": sa_tail["token_accuracy"],
            "native_all_loss": native_all["loss"],
            "native_all_accuracy": native_all["token_accuracy"],
            "native_all_rcb": recovered_context_benefit(
                sa_full_loss=sa_full["loss"],
                sa_tail_loss=sa_tail["loss"],
                pra_loss=native_all["loss"],
            ),
            "native_oracle_loss": native_oracle["loss"],
            "native_oracle_accuracy": native_oracle["token_accuracy"],
            "native_oracle_rcb": oracle_rcb,
            "native_routed_loss": routed["loss"],
            "native_routed_accuracy": routed["token_accuracy"],
            "native_routed_rcb": recovered_context_benefit(
                sa_full_loss=sa_full["loss"],
                sa_tail_loss=sa_tail["loss"],
                pra_loss=routed["loss"],
            ),
            "transport_gap": native_all["loss"] - sa_full["loss"],
            "sparse_gap": native_oracle["loss"] - native_all["loss"],
            "routing_gap": routed["loss"] - native_oracle["loss"],
            "sa_full_duration_seconds": sa_full["duration_seconds"],
            "sa_tail_duration_seconds": sa_tail["duration_seconds"],
            "native_all_duration_seconds": native_all["duration_seconds"],
            "native_oracle_duration_seconds": native_oracle["duration_seconds"],
            "native_routed_duration_seconds": routed["duration_seconds"],
            "reference_encoding_strategy": cfg.reference_encoding_strategy,
            "reference_position_mode": cfg.reference_position_mode,
            "prompt_position_mode": cfg.prompt_position_mode,
            "gist_mode": cfg.gist_mode,
            "gists_per_chunk": cfg.gists_per_chunk,
            **_summary_metrics(routed["per_example"]),
        }
        rows.append(row)
        condition_results = [routed]
        if top_k == top_k_values[0]:
            condition_results = [native_all, native_oracle, routed]
        for condition_result in condition_results:
            for value in condition_result["per_example"]:
                compact_value = {
                    key: item
                    for key, item in value.items()
                    if key not in {"rank_diagnostics_by_layer", "candidate_rankings_by_layer"}
                }
                raw.append(
                    {
                        "split_count": split_count,
                        "source_unit_count": split_count - 1,
                        "top_k_references": top_k,
                        **compact_value,
                    }
                )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows, raw


def _aggregate(seed_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in seed_rows:
        key = (
            row["dataset"],
            row["model_tier"],
            int(row["split_count"]),
            int(row["top_k_references"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for key, values in sorted(grouped.items()):
        dataset, tier, split_count, top_k = key
        result = {
            "dataset": dataset,
            "model_tier": tier,
            "split_count": split_count,
            "source_unit_count": split_count - 1,
            "top_k_references": top_k,
            "seeds": len(values),
            "parameter_count": values[0]["parameter_count"],
        }
        numeric_keys = sorted(
            {
                name
                for value in values
                for name, item in value.items()
                if isinstance(item, (int, float))
                and name
                not in {
                    "seed",
                    "split_count",
                    "source_unit_count",
                    "top_k_references",
                    "parameter_count",
                }
            }
        )
        for name in numeric_keys:
            numbers = [
                float(value[name])
                for value in values
                if value.get(name) is not None and math.isfinite(float(value[name]))
            ]
            if numbers:
                result[f"{name}_mean"] = statistics.fmean(numbers)
                result[f"{name}_stddev"] = (
                    statistics.stdev(numbers) if len(numbers) > 1 else 0.0
                )
        aggregate.append(result)
    return aggregate


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_scale(report_dir: Path, aggregate: list[dict]) -> None:
    datasets = sorted({row["dataset"] for row in aggregate})
    colors = {"tiny": "#2878B5", "small": "#C85A3E"}
    tiers = [tier for tier in ("tiny", "small") if any(row["model_tier"] == tier for row in aggregate)]

    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for index, dataset in enumerate(datasets):
        axis = axes[0][index]
        for tier in tiers:
            rows = [
                row
                for row in aggregate
                if row["dataset"] == dataset
                and row["model_tier"] == tier
                and int(row["top_k_references"]) == 8
            ]
            rows.sort(key=lambda row: int(row["split_count"]))
            axis.plot(
                [row["accessible_tokens_mean"] for row in rows],
                [row["native_routed_rcb_mean"] for row in rows],
                marker="o",
                color=colors[tier],
                label=f"{tier}, routed k=8",
            )
            axis.plot(
                [row["accessible_tokens_mean"] for row in rows],
                [row["native_oracle_rcb_mean"] for row in rows],
                marker="s",
                linestyle="--",
                color=colors[tier],
                label=f"{tier}, oracle",
            )
        axis.axhline(0.9, color="#555555", linestyle=":", linewidth=1)
        axis.set_title(dataset)
        axis.set_xlabel("Accessible tokens")
        axis.set_ylabel("Recovered Context Benefit")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Native-KV quality as context and addressability grow (five seeds)")
    figure.tight_layout()
    figure.savefig(report_dir / "pra_scale_quality.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for index, dataset in enumerate(datasets):
        axis = axes[0][index]
        for tier in tiers:
            rows = [
                row
                for row in aggregate
                if row["dataset"] == dataset
                and row["model_tier"] == tier
                and int(row["top_k_references"]) == 8
            ]
            rows.sort(key=lambda row: int(row["split_count"]))
            axis.plot(
                [row["accessible_tokens_mean"] for row in rows],
                [row["active_fraction_mean"] for row in rows],
                marker="o",
                color=colors[tier],
                label=tier,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Accessible tokens")
        axis.set_ylabel("Active physical K/V fraction")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Native-KV sparsity as context grows, routed k=8 (five seeds)")
    figure.tight_layout()
    figure.savefig(report_dir / "pra_scale_sparsity.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for index, dataset in enumerate(datasets):
        axis = axes[0][index]
        for split_count, marker in ((64, "o"), (128, "s"), (256, "^")):
            rows = [
                row
                for row in aggregate
                if row["dataset"] == dataset
                and row["model_tier"] == "tiny"
                and int(row["split_count"]) == split_count
            ]
            rows.sort(key=lambda row: int(row["top_k_references"]))
            axis.plot(
                [row["active_fraction_mean"] for row in rows],
                [row["native_routed_rcb_mean"] for row in rows],
                marker=marker,
                label=f"{split_count} units",
            )
            for row in rows:
                axis.annotate(
                    f"k={row['top_k_references']}",
                    (row["active_fraction_mean"], row["native_routed_rcb_mean"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_title(dataset)
        axis.set_xlabel("Active physical K/V fraction")
        axis.set_ylabel("Routed RCB")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Retrieval-breadth frontier under meaningful context scaling")
    figure.tight_layout()
    figure.savefig(report_dir / "pra_scale_frontier.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for index, dataset in enumerate(datasets):
        axis = axes[0][index]
        rows = [
            row
            for row in aggregate
            if row["dataset"] == dataset
            and int(row["split_count"]) == 64
            and int(row["top_k_references"]) == 8
        ]
        rows.sort(key=lambda row: tiers.index(row["model_tier"]))
        positions = range(len(rows))
        axis.bar(
            [value - 0.18 for value in positions],
            [row["native_oracle_rcb_mean"] for row in rows],
            width=0.36,
            label="Oracle",
            color="#2A7F62",
        )
        axis.bar(
            [value + 0.18 for value in positions],
            [row["native_routed_rcb_mean"] for row in rows],
            width=0.36,
            label="Routed k=8",
            color="#C85A3E",
        )
        axis.set_xticks(list(positions), [row["model_tier"] for row in rows])
        axis.set_title(dataset)
        axis.set_ylabel("Recovered Context Benefit")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("SA backbone capacity sensitivity at 64 addressable units")
    figure.tight_layout()
    figure.savefig(report_dir / "pra_model_capacity.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 7), squeeze=False)
    for index, dataset in enumerate(datasets):
        for tier in tiers:
            rows = [
                row
                for row in aggregate
                if row["dataset"] == dataset
                and row["model_tier"] == tier
                and int(row["top_k_references"]) == 8
            ]
            rows.sort(key=lambda row: int(row["split_count"]))
            accessible = [row["accessible_tokens_mean"] for row in rows]
            axes[0][index].plot(
                accessible,
                [1_000 * row["routing_latency_mean"] for row in rows],
                marker="o",
                color=colors[tier],
                label=tier,
            )
            axes[1][index].plot(
                accessible,
                [row["kv_transfer_bytes_mean"] / 1_024 for row in rows],
                marker="o",
                color=colors[tier],
                label=tier,
            )
        axes[0][index].set_title(dataset)
        axes[0][index].set_ylabel("Routing latency (ms/example)")
        axes[1][index].set_ylabel("Retrieved K/V transfer (KiB/example)")
        axes[1][index].set_xlabel("Accessible tokens")
        for axis in (axes[0][index], axes[1][index]):
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.suptitle("Prototype routing and transfer cost, routed k=8 (five seeds)")
    figure.tight_layout()
    figure.savefig(report_dir / "pra_scale_cost.pdf", bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        print("warning: scale experiments are running without CUDA", flush=True)
    report_dir = REPO / "out" / "reports" / "pra_scale_sensitivity"
    result_root = REPO / "out" / "pra_scale_sensitivity"
    report_dir.mkdir(parents=True, exist_ok=True)
    training_runs = []
    for dataset in args.datasets:
        tokenizer, training_module, modules = prepare_scale_data(dataset)
        for datamodule in modules.values():
            datamodule.test_dataset = Subset(
                datamodule.dataset,
                range(min(args.eval_examples, len(datamodule.dataset))),
            )
        for tier in args.model_tiers:
            settings = {
                **DATASET_SETTINGS[dataset],
                **MODEL_TIERS[tier],
                "max_seq_len": MAX_SEQ_LEN,
                "scale_source_unit_count": 255,
                "generation_version": GENERATION_VERSION,
                "model_tier": tier,
            }
            training_module.batch_size = int(settings["batch_size"])
            for seed in args.seeds:
                _set_seed(seed)
                run_dir = result_root / dataset / tier / f"seed-{seed}"
                source, train_info = train_full_context_sa(
                    seed=seed,
                    tokenizer=tokenizer,
                    datamodule=training_module,
                    settings=settings,
                    run_dir=run_dir,
                    device=device,
                    force=args.force_train,
                )
                parameter_count = sum(parameter.numel() for parameter in source.parameters())
                training_runs.append(
                    {
                        "dataset": dataset,
                        "model_tier": tier,
                        "seed": seed,
                        "parameter_count": parameter_count,
                        **train_info,
                    }
                )
                seed_path = run_dir / "scale_result.json"
                requested_pairs = {
                    (int(split_count), int(top_k))
                    for split_count in args.splits
                    for top_k in args.top_k
                }
                existing_payload = (
                    json.loads(seed_path.read_text(encoding="utf-8"))
                    if seed_path.exists() and not args.force_eval
                    else None
                )
                existing_pairs = {
                    (int(row["split_count"]), int(row["top_k_references"]))
                    for row in (existing_payload or {}).get("results", [])
                }
                if (
                    existing_payload is not None
                    and int(existing_payload.get("eval_examples", -1)) == args.eval_examples
                    and requested_pairs.issubset(existing_pairs)
                ):
                    payload = existing_payload
                    print(f"reuse {seed_path.relative_to(REPO)}", flush=True)
                else:
                    rows = []
                    raw_rows = []
                    for split_count in args.splits:
                        print(
                            f"dataset={dataset} tier={tier} seed={seed} "
                            f"split={split_count}",
                            flush=True,
                        )
                        scale_rows, scale_raw = evaluate_scale(
                            source=source,
                            tokenizer=tokenizer,
                            datamodule=modules[split_count],
                            split_count=split_count,
                            top_k_values=args.top_k,
                            device=device,
                        )
                        rows.extend(scale_rows)
                        raw_rows.extend(scale_raw)
                        oracle_rcb = scale_rows[0].get("native_oracle_rcb")
                        if (
                            split_count >= 128
                            and (oracle_rcb is None or oracle_rcb < args.oracle_threshold)
                        ):
                            print(
                                f"stop scale-up for {dataset}/{tier}/seed-{seed}: "
                                f"split-{split_count} oracle RCB={oracle_rcb}",
                                flush=True,
                            )
                            break
                    payload = {
                        "dataset": dataset,
                        "model_tier": tier,
                        "seed": seed,
                        "parameter_count": parameter_count,
                        "eval_examples": args.eval_examples,
                        "settings": settings,
                        "results": rows,
                        "raw": raw_rows,
                    }
                    seed_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                del source
                if device == "cuda":
                    torch.cuda.empty_cache()

    # Separate staged invocations (full tiny frontier, selected small setting)
    # contribute to one report without forcing a Cartesian rerun.
    all_seed_rows = []
    all_raw_rows = []
    for dataset in args.datasets:
        for tier in args.model_tiers:
            for seed in args.seeds:
                seed_path = result_root / dataset / tier / f"seed-{seed}" / "scale_result.json"
                if not seed_path.exists():
                    continue
                payload = json.loads(seed_path.read_text(encoding="utf-8"))
                parameter_count = int(payload["parameter_count"])
                all_seed_rows.extend(
                    {
                        "dataset": dataset,
                        "model_tier": tier,
                        "seed": seed,
                        "parameter_count": parameter_count,
                        **row,
                    }
                    for row in payload["results"]
                )
                all_raw_rows.extend(
                    {
                        "dataset": dataset,
                        "model_tier": tier,
                        "seed": seed,
                        **row,
                    }
                    for row in payload["raw"]
                )
    aggregate = _aggregate(all_seed_rows)
    observed_top_k = {
        tier: sorted(
            {
                int(row["top_k_references"])
                for row in aggregate
                if row["model_tier"] == tier
            }
        )
        for tier in sorted({row["model_tier"] for row in aggregate})
    }
    manifest = {
        "protocol": "SA-only training followed by inference-only native historical K/V slicing",
        "datasets": args.datasets,
        "model_tiers": args.model_tiers,
        "seeds": args.seeds,
        "splits": args.splits,
        "requested_top_k_this_invocation": args.top_k,
        "evaluated_top_k_by_model_tier": observed_top_k,
        "search_protocol": "full top-k frontier on tiny; selected k=8 capacity confirmation on small",
        "source_units_by_split": {str(value): value - 1 for value in args.splits},
        "reference_encoding_strategy": "native_slice",
        "reference_position_mode": "global",
        "prompt_position_mode": "historical",
        "device": device,
        "generation_version": GENERATION_VERSION,
        "oracle_scale_up_threshold": args.oracle_threshold,
        "evaluation_examples_per_seed": args.eval_examples,
    }
    (report_dir / "scale_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (report_dir / "aggregate.json").write_text(
        json.dumps({"manifest": manifest, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    (report_dir / "training_runs.json").write_text(
        json.dumps(training_runs, indent=2), encoding="utf-8"
    )
    _write_csv(report_dir / "aggregate_by_scale.csv", aggregate)
    _write_csv(report_dir / "raw_runs.csv", all_raw_rows)
    _plot_scale(report_dir, aggregate)

    if args.publish:
        results_dir = REPO / "docs" / "papers" / "shared" / "results"
        figures_dir = REPO / "docs" / "papers" / "shared" / "figures"
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_dir / "aggregate.json", results_dir / "pra_scale_sensitivity.json")
        shutil.copy2(
            report_dir / "aggregate_by_scale.csv",
            results_dir / "pra_scale_sensitivity.csv",
        )
        for name in (
            "pra_scale_quality.pdf",
            "pra_scale_sparsity.pdf",
            "pra_scale_frontier.pdf",
            "pra_model_capacity.pdf",
            "pra_scale_cost.pdf",
        ):
            shutil.copy2(report_dir / name, figures_dir / name)
    return report_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=("hotpotqa", "qasper"), default=["hotpotqa", "qasper"]
    )
    parser.add_argument(
        "--model-tiers", nargs="+", choices=tuple(MODEL_TIERS), default=list(MODEL_TIERS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--splits", nargs="+", type=int, choices=SCALE_SPLITS, default=list(SCALE_SPLITS))
    parser.add_argument("--top-k", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--oracle-threshold", type=float, default=0.9)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
