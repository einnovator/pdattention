"""Reusable training and evaluation helpers for PRA experiment notebooks."""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from IPython.display import display

from data.datamodules import PRADataModule
from data.datasets import (
    dataset_class_for_stage,
    generate_wikitext_reference_dataset,
    generate_wikitext_reference_dataset_v2,
)
from data.language_modeling import WikiTextDataModule
from data.tokenizer import BPETokenizer
from pra_torch.cache_services import build_cache_from_metadata
from pra_torch.cli import build_pra_config, deep_update, load_config
from pra_torch.config import PRAConfig, TrainConfig
from pra_torch.eval import run_evaluation
from pra_torch.lm_train import evaluate_language_model, train_language_model
from pra_torch.model import TinyPRAModel
from pra_torch.pra_train import (
    evaluate_pra_model,
    evaluate_reference_ablation,
    train_pra_model,
)


@dataclass(frozen=True)
class NotebookRuntime:
    repo: Path
    seed: int
    device: str
    write_images_to_disk: bool
    python: str
    torch_version: str
    torch_cuda: str | None
    cuda_available: bool
    gpu: str | None
    compute_capability: tuple[int, int] | None

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["repo"] = str(self.repo)
        return values


@dataclass(frozen=True)
class ExperimentPolicy:
    d_model: int
    n_layers: int
    n_heads: int
    batch_size: int
    epochs: int
    learning_rate: float
    max_seq_len: int = 96
    estimated_optimizer_steps: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository containing both ``src`` and ``data``."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find PRA repository root from {current}")


def configure_notebook(
    *,
    repo: str | Path | None = None,
    seed: int = 7,
    write_images_to_disk: bool = False,
) -> NotebookRuntime:
    """Capture a reproducible notebook runtime and select CUDA when available."""
    repo_path = find_repo_root(repo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NotebookRuntime(
        repo=repo_path,
        seed=int(seed),
        device=device,
        write_images_to_disk=bool(write_images_to_disk),
        python=sys.version.split()[0],
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        compute_capability=(
            torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
        ),
    )


def inspect_dataset(
    dataset_stage: str,
    *,
    data_dir: str | Path,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Load one dataset stage and summarize its first sample and references."""
    dataset = dataset_class_for_stage(dataset_stage)(data_dir, max_examples=max_examples)
    if not dataset:
        raise ValueError(f"Dataset stage {dataset_stage!r} contains no questions")
    sample = dataset[0]
    table = dataset.build_reference_table(sample)
    return {
        "stage": dataset_stage,
        "dataset_name": dataset.dataset_name,
        "size": len(dataset),
        "sample_id": sample.id,
        "question": sample.question,
        "answer": sample.answer,
        "target_reference_ids": sample.target_reference_ids,
        "reference_tokens": [handle.token for handle in table.all()],
        "reference_uris": [handle.uri for handle in table.all()],
    }


def experiment_policy(size: int) -> ExperimentPolicy:
    """Choose a conservative capacity and update budget from dataset size."""
    size = int(size)
    if size <= 3:
        values = (32, 2, 4, min(max(size, 1), 2), 20, 1e-3)
    elif size <= 30:
        values = (32, 2, 4, 8, 40, 8e-4)
    elif size <= 300:
        values = (64, 3, 4, 16, 12, 5e-4)
    elif size <= 3_000:
        values = (96, 4, 4, 32, 5, 3e-4)
    else:
        values = (128, 4, 4, 32, 3, 3e-4)
    d_model, n_layers, n_heads, batch_size, epochs, learning_rate = values
    if size < 3:
        train_examples = size
    elif size == 3:
        train_examples = 1
    else:
        train_examples = max(1, int(size * 0.8))
    estimated_steps = math.ceil(train_examples / batch_size) * epochs
    return ExperimentPolicy(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        estimated_optimizer_steps=estimated_steps,
    )


def build_datamodule(
    dataset_stage: str,
    runtime: NotebookRuntime,
    *,
    data_dir: str | Path | None = None,
    max_examples: int | None = None,
    policy: ExperimentPolicy | None = None,
) -> tuple[PRADataModule, dict[str, Any]]:
    """Build deterministic dataset splits, tokenizer, collator, and loaders."""
    source_dir = Path(data_dir or runtime.repo / "data")
    profile = inspect_dataset(dataset_stage, data_dir=source_dir, max_examples=max_examples)
    policy = policy or experiment_policy(profile["size"])
    torch.manual_seed(runtime.seed)
    datamodule = PRADataModule(
        dataset_stage=dataset_stage,
        data_dir=source_dir,
        max_examples=max_examples,
        batch_size=policy.batch_size,
        max_seq_len=policy.max_seq_len,
        shuffle=True,
        num_workers=0,
        pin_memory=runtime.cuda_available,
    ).load()
    batch = next(iter(datamodule.train_loader()))
    summary = {
        **profile,
        "train_size": len(datamodule.train_dataset),
        "validation_size": len(datamodule.val_dataset),
        "test_size": len(datamodule.test_dataset),
        "vocab_size": datamodule.tokenizer.vocab_size,
        "batch_shape": tuple(batch["input_ids"].shape),
    }
    return datamodule, summary


def inspect_reference_tokenization(
    datamodule: PRADataModule,
    text: str = "Use <REF_1> before <REF_2>.",
) -> dict[str, Any]:
    """Verify that available reference handles survive a tokenizer round trip."""
    available = sorted(
        token for token in datamodule.tokenizer.stoi if token.startswith("<REF_")
    )
    if not available:
        raise ValueError("Dataset tokenizer contains no reference tokens")
    selected = available[:2]
    probe = text if len(selected) >= 2 else f"Use {selected[0]}."
    ids = datamodule.tokenizer.encode(probe)
    decoded = datamodule.tokenizer.decode(ids)
    return {
        "text": probe,
        "ids": ids,
        "decoded": decoded,
        "reference_token_ids": {
            token: datamodule.tokenizer.stoi[token] for token in selected
        },
        "round_trip_ok": decoded == probe,
    }


def build_model_config(
    datamodule: PRADataModule,
    policy: ExperimentPolicy,
    runtime: NotebookRuntime,
) -> PRAConfig:
    """Create a PRA architecture compatible with a prepared datamodule."""
    return PRAConfig(
        vocab_size=datamodule.tokenizer.vocab_size,
        max_seq_len=policy.max_seq_len,
        d_model=policy.d_model,
        n_heads=policy.n_heads,
        n_layers=policy.n_layers,
        pra_layer_ids=tuple(range(policy.n_layers)),
        dropout=0.0,
        batch_size=policy.batch_size,
        lr=policy.learning_rate,
        device=runtime.device,
    )


def run_single_batch_smoke(
    datamodule: PRADataModule,
    policy: ExperimentPolicy,
    runtime: NotebookRuntime,
) -> dict[str, Any]:
    """Exercise cache creation, forward propagation, and one optimizer update."""
    batch = next(iter(datamodule.train_loader()))
    cfg = build_model_config(datamodule, policy, runtime)
    model = TinyPRAModel(cfg).to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=policy.learning_rate)
    cache = build_cache_from_metadata(
        model, datamodule.tokenizer, batch["metadata"], runtime.device
    )
    input_ids = batch["input_ids"].to(runtime.device)
    labels = batch["labels"].to(runtime.device)
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "cache_entries": len(cache.entries),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "logits_shape": tuple(logits.shape),
        "device": next(model.parameters()).device.type,
    }


def train_dataset_workload(
    dataset_stage: str,
    runtime: NotebookRuntime,
    *,
    data_dir: str | Path | None = None,
    max_examples: int | None = None,
    policy: ExperimentPolicy | None = None,
) -> dict[str, Any]:
    """Train one dataset through the canonical PRA functional engine."""
    source_dir = Path(data_dir or runtime.repo / "data")
    profile = inspect_dataset(dataset_stage, data_dir=source_dir, max_examples=max_examples)
    policy = policy or experiment_policy(profile["size"])
    datamodule, loader_summary = build_datamodule(
        dataset_stage,
        runtime,
        data_dir=source_dir,
        max_examples=max_examples,
        policy=policy,
    )
    cfg = build_model_config(datamodule, policy, runtime)
    steps_per_epoch = max(
        1, math.ceil(len(datamodule.train_dataset) / policy.batch_size)
    )
    experiment_name = f"{dataset_stage}_size_{profile['size']}"
    train_config = TrainConfig(
        experiment_name=experiment_name,
        output_dir=str(runtime.repo / "out" / "notebook_datasets"),
        seed=runtime.seed,
        device=runtime.device,
        epochs=policy.epochs,
        batch_size=policy.batch_size,
        learning_rate=policy.learning_rate,
        weight_decay=0.0,
        eval_every_steps=steps_per_epoch,
        save_every_steps=max(steps_per_epoch * policy.epochs, 1),
        log_every_steps=max(steps_per_epoch // 5, 1),
        use_tensorboard=False,
        save_metric_plots=runtime.write_images_to_disk,
        mixed_precision=(
            runtime.device == "cuda"
            and runtime.compute_capability is not None
            and runtime.compute_capability[0] >= 7
        ),
        dataset_stage=dataset_stage,
        data_dir=str(source_dir),
        max_examples=max_examples,
        max_seq_len=policy.max_seq_len,
        shuffle=True,
    )
    result = train_pra_model(
        cfg=cfg, train_config=train_config, datamodule=datamodule
    )
    result.update(
        {
            "dataset_stage": dataset_stage,
            "size": profile["size"],
            "policy": policy,
            "cfg": cfg,
            "train_config": train_config,
            "datamodule": datamodule,
            "loader_summary": loader_summary,
            "parameter_count": sum(
                parameter.numel() for parameter in result["model"].parameters()
            ),
        }
    )
    print(
        f"stage={dataset_stage} size={profile['size']:,} "
        f"parameters={result['parameter_count']:,} epochs={policy.epochs} "
        f"optimizer_steps={result['global_step']} "
        f"test_loss={result['test_metrics'].get('loss', float('nan')):.4f} "
        f"test_accuracy={result['test_metrics'].get('answer_accuracy', 0.0):.2%}"
    )
    return result


def load_metric_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the persisted scalar metric history for a training result."""
    metrics_path = result["state"].run_dir / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))["records"]


def plot_training_progress(
    result: dict[str, Any],
    *,
    write_to_disk: bool | None = None,
) -> dict[str, Any]:
    """Display batch-wise and epoch-wise losses, optionally saving the figure."""
    records = load_metric_records(result)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    batch_records = by_split["train_batch"]
    axes[0].plot(
        [record.get("batch_step", record["step"]) for record in batch_records],
        [record["metrics"]["train_loss"] for record in batch_records],
        linewidth=1.2,
    )
    axes[0].set(title="Batch-wise training loss", xlabel="batch step", ylabel="loss")
    for split, metric, label in (
        ("train_epoch", "train_loss", "train"),
        ("val_epoch", "loss", "validation"),
    ):
        values = by_split[split]
        axes[1].plot(
            [record.get("epoch", record["step"]) for record in values],
            [record["metrics"][metric] for record in values],
            marker="o",
            label=label,
        )
    axes[1].set(title="Epoch-wise loss", xlabel="epoch", ylabel="loss")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle(
        f"PRA training: {result['dataset_stage']} ({result['size']:,} examples)"
    )
    fig.tight_layout()

    should_write = (
        result["state"].train_config.save_metric_plots
        if write_to_disk is None
        else write_to_disk
    )
    output_path = None
    if should_write:
        output_path = result["state"].run_dir / "notebook_training_progress.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"wrote {output_path}")
    display(fig)
    plt.close(fig)
    return {
        "train_batches": len(batch_records),
        "train_epochs": len(by_split["train_epoch"]),
        "validation_epochs": len(by_split["val_epoch"]),
        "metrics_path": result["state"].run_dir / "metrics.json",
        "image_path": output_path,
    }


def evaluate_trained_workload(
    result: dict[str, Any], *, max_new_tokens: int = 12
):
    """Evaluate a checkpoint against no-ref, context, RAG, and PRA variants."""
    cfg = result["cfg"]
    return run_evaluation(
        ckpt=str(result["state"].checkpoint.latest_path),
        device=result["state"].device,
        datamodule=result["datamodule"],
        max_new_tokens=max_new_tokens,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_vanilla_layers=cfg.n_vanilla_layers,
        n_mixed_layers=cfg.n_mixed_layers,
        dropout=cfg.dropout,
        pra_layer_ids=cfg.pra_layer_ids,
        top_k_references=cfg.top_k_references,
        top_k_chunks_per_reference=cfg.top_k_chunks_per_reference,
        trigger_threshold=cfg.trigger_threshold,
        memory_transport=cfg.memory_transport,
        memory_alpha=cfg.memory_alpha,
    )


def plot_evaluation(
    results,
    result: dict[str, Any],
    *,
    write_to_disk: bool | None = None,
) -> dict[str, float]:
    """Display evaluation loss and generated exact-match comparisons."""
    names = [item.name for item in results]
    losses = [item.lm_loss for item in results]
    accuracies = [item.answer_exact_match for item in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    axes[0].bar(names, losses, color=colors)
    axes[0].set(title="Language-model loss", ylabel="loss")
    axes[1].bar(names, accuracies, color=colors)
    axes[1].set(
        title="Generated answer exact match", ylabel="accuracy", ylim=(0, 1)
    )
    fig.suptitle(f"Evaluation: {result['dataset_stage']}")
    fig.tight_layout()
    should_write = (
        result["state"].train_config.save_metric_plots
        if write_to_disk is None
        else write_to_disk
    )
    if should_write:
        output_path = result["state"].run_dir / "notebook_evaluation.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"wrote {output_path}")
    display(fig)
    plt.close(fig)
    return {item.name: item.answer_exact_match for item in results}


def summarize_workload(result: dict[str, Any]) -> dict[str, Any]:
    """Return the compact cross-dataset comparison row for one run."""
    summary = {
        "dataset_stage": result["dataset_stage"],
        "size": result["size"],
        "parameters": result["parameter_count"],
        "epochs": result["policy"].epochs,
        "optimizer_steps": result["global_step"],
        "batch_steps": result["batch_step"],
        "train_loss": result["train_metrics"].get("train_loss"),
        "validation_loss": result["val_metrics"].get("loss"),
        "test_loss": result["test_metrics"].get("loss"),
        "test_perplexity": result["test_metrics"].get("perplexity"),
        "test_token_accuracy": result["test_metrics"].get("token_accuracy"),
        "test_answer_accuracy": result["test_metrics"].get("answer_accuracy"),
        "test_reference_accuracy": result["test_metrics"].get(
            "reference_retrieval_accuracy"
        ),
        "test_reference_top1_accuracy": result["test_metrics"].get(
            "reference_selection_top1_accuracy"
        ),
        "test_reference_topk_accuracy": result["test_metrics"].get(
            "reference_selection_topk_accuracy"
        ),
        "test_reference_mrr": result["test_metrics"].get("reference_selection_mrr"),
    }
    loader_summary = result.get("loader_summary", {})
    if "split_count" in loader_summary:
        summary["split_count"] = loader_summary["split_count"]
        summary["reference_count"] = loader_summary.get("reference_count")
    if result.get("initial_reference_metrics"):
        initial = result["initial_reference_metrics"]
        summary["initial_validation_loss"] = initial.get("loss")
        summary["initial_reference_top1_accuracy"] = initial.get(
            "reference_selection_top1_accuracy"
        )
        summary["initial_reference_topk_accuracy"] = initial.get(
            "reference_selection_topk_accuracy"
        )
        summary["initial_reference_mrr"] = initial.get("reference_selection_mrr")
    summary.update(result.get("timing_metrics", {}))
    if result.get("plain_test_after_reference"):
        summary["plain_test_after_reference_loss"] = result["plain_test_after_reference"].get("loss")
        summary["plain_test_after_reference_perplexity"] = result["plain_test_after_reference"].get("perplexity")
        summary["plain_test_after_reference_token_accuracy"] = result["plain_test_after_reference"].get("token_accuracy")
    if result.get("reference_ablations"):
        by_condition = {item["condition"]: item for item in result["reference_ablations"]}
        for condition, metrics in by_condition.items():
            summary[f"reference_{condition}_loss"] = metrics["loss"]
            summary[f"reference_{condition}_perplexity"] = metrics["perplexity"]
        if "valid" in by_condition and "disabled" in by_condition:
            summary["reference_disabled_loss_delta"] = by_condition["disabled"]["loss"] - by_condition["valid"]["loss"]
    return summary


def workflow_parity_report(result: dict[str, Any]) -> dict[str, Any]:
    """Check the metric/history contract shared with ``pra_standalone.ipynb``."""
    records = load_metric_records(result)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["split"]] += 1
    expected_batches = len(result["datamodule"].train_loader()) * result["policy"].epochs
    return {
        "uses_train_pra_model": True,
        "has_batch_timeseries": counts["train_batch"] == expected_batches,
        "has_epoch_timeseries": counts["train_epoch"] == result["policy"].epochs,
        "has_validation_timeseries": counts["val_epoch"] == result["policy"].epochs,
        "has_test_metrics": counts["test"] == 1,
        "batch_records": counts["train_batch"],
        "epoch_records": counts["train_epoch"],
        "checkpoint_exists": result["state"].checkpoint.latest_path.exists(),
        "cuda_used": result["state"].device.startswith("cuda"),
    }


def experiment_artifact_name(
    model_name: str,
    dataset_name: str,
    *qualifiers: str | int | None,
) -> str:
    """Build a stable filesystem name from model, dataset, and run qualifiers."""
    parts = [model_name, dataset_name, *(str(value) for value in qualifiers if value is not None)]
    return "_".join(
        filter(None, (re.sub(r"[^a-zA-Z0-9._-]+", "-", part).strip("-_").lower() for part in parts))
    )


def _save_training_report_plot(result: dict[str, Any], path: Path) -> None:
    records = load_metric_records(result)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    batches = by_split["train_batch"]
    axes[0].plot(
        [record.get("batch_step", record["step"]) for record in batches],
        [record["metrics"]["train_loss"] for record in batches],
        linewidth=1.2,
        color="#2563eb",
    )
    axes[0].set(title="Batch-wise training loss", xlabel="batch step", ylabel="loss")
    for split, metric, label, color in (
        ("train_epoch", "train_loss", "train", "#2563eb"),
        ("val_epoch", "loss", "validation", "#dc2626"),
    ):
        values = by_split[split]
        if values:
            axes[1].plot(
                [record.get("epoch", record["step"]) for record in values],
                [record["metrics"][metric] for record in values],
                marker="o",
                label=label,
                color=color,
            )
    axes[1].set(title="Epoch-wise loss", xlabel="epoch", ylabel="loss")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_evaluation_report_plot(results, path: Path) -> None:
    names = [item.name for item in results]
    losses = [item.lm_loss for item in results]
    accuracies = [item.answer_exact_match for item in results]
    colors = ["#2563eb", "#d97706", "#059669", "#dc2626"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, losses, color=colors[: len(names)])
    axes[0].set(title="Language-model loss", ylabel="loss")
    axes[1].bar(names, accuracies, color=colors[: len(names)])
    axes[1].set(title="Generated answer exact match", ylabel="accuracy", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _html_table(values: dict[str, Any]) -> str:
    rows = []
    for key, value in values.items():
        if isinstance(value, float):
            value = f"{value:.6g}"
        rows.append(f"<tr><th>{escape(str(key).replace('_', ' ').title())}</th><td>{escape(str(value))}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def generate_html_report(
    result: dict[str, Any],
    *,
    runtime: NotebookRuntime,
    evaluation_results=None,
    dataset_details: dict[str, Any] | None = None,
    model_name: str | None = None,
    qualifiers: tuple[str | int, ...] = (),
    report_root: str | Path | None = None,
) -> Path:
    """Write a browsable HTML experiment report and its data/figure assets."""
    model_name = model_name or getattr(result.get("train_config"), "experiment_name", "model")
    dataset_name = str(result.get("dataset_stage", "dataset"))
    run_name = experiment_artifact_name(model_name, dataset_name, *qualifiers)
    report_dir = Path(report_root or runtime.repo / "out" / "reports") / run_name
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    training_plot = assets_dir / "training_progress.png"
    _save_training_report_plot(result, training_plot)
    evaluation_plot = None
    if evaluation_results:
        evaluation_plot = assets_dir / "evaluation.png"
        _save_evaluation_report_plot(evaluation_results, evaluation_plot)

    metrics_source = result["state"].run_dir / "metrics.json"
    shutil.copy2(metrics_source, report_dir / "metrics.json")
    summary = summarize_workload(result)
    model_config = asdict(result["cfg"])
    train_config = asdict(result["train_config"])
    policy = result["policy"].as_dict() if hasattr(result["policy"], "as_dict") else asdict(result["policy"])
    evaluation_rows = [asdict(item) for item in evaluation_results] if evaluation_results else []
    payload = {
        "report_name": run_name,
        "generated_at": datetime.now().astimezone().isoformat(),
        "runtime": runtime.as_dict(),
        "dataset": dataset_details or result.get("loader_summary", {}),
        "policy": policy,
        "model": {"name": model_name, "parameter_count": result["parameter_count"], **model_config},
        "training": train_config,
        "results": summary,
        "evaluation": evaluation_rows,
        "reference_ablations": result.get("reference_ablations", []),
        "checkpoint": str(result["state"].checkpoint.latest_path),
    }
    (report_dir / "report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    evaluation_table = ""
    if evaluation_rows:
        headers = ["name", "lm_loss", "answer_exact_match", "expected_ref_hit", "cache_hit_ratio", "latency"]
        evaluation_table = "<table><tr>" + "".join(f"<th>{escape(key.replace('_', ' ').title())}</th>" for key in headers) + "</tr>"
        for row in evaluation_rows:
            evaluation_table += "<tr>" + "".join(f"<td>{escape(f'{row.get(key):.6g}' if isinstance(row.get(key), float) else str(row.get(key)))}</td>" for key in headers) + "</tr>"
        evaluation_table += "</table>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(run_name)} experiment report</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;color:#18202a;background:#f4f6f8}}main{{max-width:1180px;margin:auto;padding:32px}}
h1{{font-size:30px;margin:0 0 6px}}h2{{font-size:19px;margin:30px 0 12px;border-bottom:1px solid #d7dde5;padding-bottom:7px}}
.muted{{color:#5d6875}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}section{{background:white;border:1px solid #d7dde5;border-radius:6px;padding:18px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;border-bottom:1px solid #e5e9ef;padding:7px;vertical-align:top}}th{{color:#4b5563;width:42%}}
img{{display:block;width:100%;height:auto;border:1px solid #d7dde5;background:white}}code{{font-family:Consolas,monospace}}a{{color:#1d4ed8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style></head><body><main>
<h1>{escape(model_name)} on {escape(dataset_name)}</h1><p class="muted">Generated {escape(payload['generated_at'])} · {result['parameter_count']:,} parameters · checkpoint <code>{escape(payload['checkpoint'])}</code></p>
<div class="grid"><section><h2>Results</h2>{_html_table(summary)}</section><section><h2>Runtime</h2>{_html_table(runtime.as_dict())}</section></div>
<h2>Training Progress</h2><img src="assets/training_progress.png" alt="Training loss plots">
{f'<h2>Evaluation</h2><img src="assets/evaluation.png" alt="Evaluation plots">{evaluation_table}' if evaluation_plot else ''}
<div class="grid"><section><h2>Dataset</h2>{_html_table(payload['dataset'])}</section><section><h2>Policy</h2>{_html_table(policy)}</section></div>
<div class="grid"><section><h2>Model Configuration</h2>{_html_table(payload['model'])}</section><section><h2>Training Configuration</h2>{_html_table(train_config)}</section></div>
<h2>Artifacts</h2><p><a href="report.json">Structured report JSON</a> · <a href="metrics.json">Complete metric history</a></p>
</main></body></html>"""
    report_path = report_dir / "index.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"wrote HTML report: {report_path}")
    return report_path


def generate_split_count_html_report(
    results: list[dict[str, Any]],
    report_paths: list[str | Path],
    *,
    runtime: NotebookRuntime,
    report_name: str = "wikitext2_split_count_comparison",
) -> Path:
    """Aggregate paired-seed quality and timing for fixed WikiText split counts."""
    if len(results) != len(report_paths):
        raise ValueError("results and report_paths must describe the same runs")
    rows = []
    for result, report_path in zip(results, report_paths):
        summary = summarize_workload(result)
        ablations = {
            item["condition"]: item for item in result.get("reference_ablations", [])
        }
        rows.append(
            {
                "model": result["model_name"],
                "seed": int(result["train_config"].seed),
                "split_count": int(result["loader_summary"]["split_count"]),
                "reference_count": int(result["loader_summary"]["reference_count"]),
                "test_loss": summary.get("test_loss"),
                "test_perplexity": summary.get("test_perplexity"),
                "valid_loss": ablations.get("valid", {}).get("loss"),
                "disabled_loss": ablations.get("disabled", {}).get("loss"),
                "shuffled_loss": ablations.get("shuffled", {}).get("loss"),
                "disabled_loss_delta": (
                    ablations.get("disabled", {}).get("loss")
                    - ablations.get("valid", {}).get("loss")
                ),
                "shuffled_loss_delta": (
                    ablations.get("shuffled", {}).get("loss")
                    - ablations.get("valid", {}).get("loss")
                ),
                "train_seconds": summary.get("train_duration_seconds"),
                "validation_seconds": summary.get("validation_duration_seconds"),
                "processed_tokens": summary.get("processed_tokens"),
                "training_tokens_per_second": summary.get("training_tokens_per_second"),
                "reference_top1_accuracy": summary.get("test_reference_top1_accuracy"),
                "reference_mrr": summary.get("test_reference_mrr"),
                "report": f"../{Path(report_path).parent.name}/index.html",
            }
        )

    paired_model_seeds = validate_paired_model_seeds(rows)
    metric_names = (
        "test_loss",
        "test_perplexity",
        "valid_loss",
        "disabled_loss",
        "shuffled_loss",
        "disabled_loss_delta",
        "shuffled_loss_delta",
        "train_seconds",
        "validation_seconds",
        "processed_tokens",
        "training_tokens_per_second",
        "reference_top1_accuracy",
        "reference_mrr",
    )
    aggregate_rows = []
    group_keys = sorted({(row["model"], row["split_count"]) for row in rows})
    for model, split_count in group_keys:
        group = [
            row
            for row in rows
            if row["model"] == model and row["split_count"] == split_count
        ]
        metrics = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            metrics[metric] = {
                "mean": statistics.fmean(values),
                "stddev": statistics.pstdev(values),
                "values": values,
            }
        aggregate_rows.append(
            {
                "model": model,
                "split_count": split_count,
                "reference_count": group[0]["reference_count"],
                "seeds": sorted(row["seed"] for row in group),
                "metrics": metrics,
            }
        )

    report_dir = runtime.repo / "out" / "reports" / report_name
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(row["model"] for row in aggregate_rows))
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, color in zip(models, colors):
        model_rows = sorted(
            (row for row in aggregate_rows if row["model"] == model),
            key=lambda row: row["split_count"],
        )
        splits = [row["split_count"] for row in model_rows]
        axes[0].errorbar(
            splits,
            [row["metrics"]["test_loss"]["mean"] for row in model_rows],
            yerr=[row["metrics"]["test_loss"]["stddev"] for row in model_rows],
            marker="o",
            capsize=4,
            label=model,
            color=color,
        )
        axes[1].errorbar(
            splits,
            [row["metrics"]["train_seconds"]["mean"] for row in model_rows],
            yerr=[row["metrics"]["train_seconds"]["stddev"] for row in model_rows],
            marker="o",
            capsize=4,
            label=model,
            color=color,
        )
    axes[0].set(title="Loss by WikiText split count", xlabel="total split count", ylabel="test loss")
    axes[1].set(title="Training time by WikiText split count", xlabel="total split count", ylabel="seconds")
    for axis in axes:
        axis.set_xticks([2, 5])
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    figure_path = assets_dir / "split_count_comparison.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    display(fig)
    plt.close(fig)

    payload = {
        "report_name": report_name,
        "generated_at": datetime.now().astimezone().isoformat(),
        "runtime": runtime.as_dict(),
        "model_seeds": paired_model_seeds,
        "split_count_semantics": "total parts including the final prediction tail",
        "controls": {
            "shared_tokenizer": True,
            "shared_source_examples": True,
            "paired_model_seeds": True,
            "fixed_dataset_generation_seed": 1729,
            "fixed_dataset_split_seed": 1729,
            "shared_optimizer_step_budget": True,
            "shared_sequence_length": True,
            "equal_processed_token_budget": False,
            "target_span_changes_with_split_count": True,
        },
        "aggregates": aggregate_rows,
        "seed_runs": rows,
    }
    (report_dir / "report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    headers = ["model", "split_count", "seeds", "test_loss", "disabled_loss_delta", "shuffled_loss_delta", "reference_top1_accuracy", "train_seconds", "validation_seconds"]
    table = "<table><tr>" + "".join(
        f"<th>{escape(header.replace('_', ' ').title())}</th>" for header in headers
    ) + "</tr>"
    for row in aggregate_rows:
        values = {
            "model": row["model"],
            "split_count": row["split_count"],
            "seeds": row["seeds"],
            **{
                metric: f"{row['metrics'][metric]['mean']:.6g} +/- {row['metrics'][metric]['stddev']:.3g}"
                for metric in headers[3:]
            },
        }
        table += "<tr>" + "".join(
            f"<td>{escape(str(values[header]))}</td>" for header in headers
        ) + "</tr>"
    table += "</table>"
    links = "".join(
        f'<li><a href="{escape(row["report"])}">{escape(row["model"])} split {row["split_count"]}, seed {row["seed"]}</a></li>'
        for row in rows
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>WikiText-2 split-count comparison</title>
<style>body{{font-family:Segoe UI,Arial;margin:32px auto;max-width:1150px;color:#18202a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}code{{font-family:Consolas}}</style></head><body>
<h1>WikiText-2 Split-Count Comparison</h1>
<p>Population mean +/- standard deviation across paired model seeds 1, 7, 21, 42, and 87. Split count is the total number of document parts, including the final prediction tail. Thus split 2 exposes one reference and split 5 exposes four. Runs share source examples, dataset generation/split seeds, tokenizer, optimizer-step budget, and sequence length.</p>
{table}<h2>Quality And Execution Time</h2><img src="assets/split_count_comparison.png" alt="Split-count comparison">
<h2>Run Reports</h2><ul>{links}</ul><p><a href="report.json">Structured report JSON</a></p></body></html>"""
    path = report_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"wrote split-count HTML report: {path}")
    return path


def validate_paired_model_seeds(
    rows: list[dict[str, Any]], minimum_seed_count: int = 5
) -> list[int]:
    """Require the same minimum set of model seeds in every comparison group."""
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["split_count"]))].append(int(row["seed"]))
    if not grouped:
        raise ValueError("A seed comparison requires at least one model/split group")

    expected: set[int] | None = None
    for key, seeds in grouped.items():
        unique_seeds = set(seeds)
        if len(unique_seeds) != len(seeds):
            raise ValueError(f"Comparison group {key} contains duplicate model seeds")
        if len(unique_seeds) < minimum_seed_count:
            raise ValueError(
                f"Comparison group {key} requires at least {minimum_seed_count} model seeds"
            )
        if expected is None:
            expected = unique_seeds
        elif unique_seeds != expected:
            raise ValueError("All comparison groups must use the same paired model seeds")

    return sorted(expected or ())


def load_experiment_settings(runtime: NotebookRuntime, experiment_name: str) -> dict[str, Any]:
    """Resolve one notebook experiment and its named model from central YAML."""
    config_path = runtime.repo / "config" / "config.yml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = dict((raw.get("experiments") or {}).get(experiment_name) or {})
    if not experiment:
        raise KeyError(f"Unknown experiment: {experiment_name}")
    model_name = str(experiment["model_name"])
    resolved = load_config(str(config_path), model_name=model_name)
    for section in ("model", "pra", "resolver", "cache"):
        overrides = experiment.get(f"{section}_overrides")
        if overrides:
            deep_update(resolved.setdefault(section, {}), dict(overrides))
    return {"name": experiment_name, "model_name": model_name, "experiment": experiment, "resolved": resolved}


def policy_from_experiment(settings: dict[str, Any]) -> ExperimentPolicy:
    model = settings["resolved"]["model"]
    train = settings["experiment"]["train"]
    max_steps = int(train.get("max_steps") or 0)
    return ExperimentPolicy(
        d_model=int(model["d_model"]),
        n_layers=int(model["n_layers"]),
        n_heads=int(model["n_heads"]),
        batch_size=int(train["batch_size"]),
        epochs=int(train["epochs"]),
        learning_rate=float(train["learning_rate"]),
        max_seq_len=int(train.get("max_seq_len", model["max_seq_len"])),
        estimated_optimizer_steps=max_steps,
    )


def _named_pra_config(settings: dict[str, Any], tokenizer, runtime: NotebookRuntime) -> PRAConfig:
    policy = policy_from_experiment(settings)
    resolved = settings["resolved"]
    resolved["model"]["max_seq_len"] = policy.max_seq_len
    return build_pra_config(
        resolved,
        vocab_size=tokenizer.vocab_size,
        batch_size=policy.batch_size,
        lr=policy.learning_rate,
        steps=int(settings["experiment"]["train"].get("max_steps") or 0),
        device=runtime.device,
    )


def train_wikitext_language_experiment(
    runtime: NotebookRuntime,
    experiment_name: str,
    *,
    resume_from: str | Path | None = None,
    artifact_qualifiers: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Train a configured model variant on plain WikiText BPE token blocks."""
    settings = load_experiment_settings(runtime, experiment_name)
    dataset_cfg = settings["experiment"]["dataset"]
    policy = policy_from_experiment(settings)
    datamodule = WikiTextDataModule(
        data_dir=runtime.repo / "data",
        dataset_name=dataset_cfg["name"],
        vocab_size=int(dataset_cfg["vocab_size"]),
        seq_len=policy.max_seq_len,
        batch_size=policy.batch_size,
        max_train_documents=dataset_cfg.get("max_train_documents"),
        max_eval_documents=dataset_cfg.get("max_eval_documents"),
        max_train_blocks=dataset_cfg.get("max_train_blocks"),
        pin_memory=runtime.cuda_available,
    ).load()
    cfg = _named_pra_config(settings, datamodule.tokenizer, runtime)
    train_cfg = settings["experiment"]["train"]
    artifact = experiment_artifact_name(
        settings["model_name"],
        "wikitext2",
        settings["name"],
        *artifact_qualifiers,
        f"seed{runtime.seed}",
        f"steps{train_cfg.get('max_steps')}",
        f"seq{policy.max_seq_len}",
    )
    training_config = TrainConfig(
        experiment_name=artifact,
        output_dir=str(runtime.repo / "out" / "notebook_wikitext"),
        seed=runtime.seed,
        device=runtime.device,
        epochs=policy.epochs,
        max_steps=int(train_cfg.get("max_steps") or 0) or None,
        batch_size=policy.batch_size,
        learning_rate=policy.learning_rate,
        eval_every_steps=max(
            int(train_cfg.get("eval_every_steps") or train_cfg.get("max_steps") or 1), 1
        ),
        save_every_steps=max(int(train_cfg.get("max_steps") or 1), 1),
        log_every_steps=1,
        use_tensorboard=False,
        save_metric_plots=False,
        mixed_precision=False,
        resume_from=str(resume_from) if resume_from else None,
        dataset_stage="wikitext2",
        data_dir=str(runtime.repo / "data"),
        max_seq_len=policy.max_seq_len,
    )
    result = train_language_model(cfg=cfg, train_config=training_config, datamodule=datamodule)
    result.update(
        {
            "dataset_stage": "wikitext2",
            "size": len(datamodule.train_dataset),
            "policy": policy,
            "cfg": cfg,
            "train_config": training_config,
            "datamodule": datamodule,
            "model_name": settings["model_name"],
            "experiment_settings": settings,
            "parameter_count": sum(parameter.numel() for parameter in result["model"].parameters()),
            "loader_summary": {
                "dataset": dataset_cfg["name"],
                "references": 0,
                "tokenizer": "Tokenizer(BPE) + Whitespace",
                "vocab_size": datamodule.tokenizer.vocab_size,
                "train_blocks": len(datamodule.train_dataset),
                "validation_blocks": len(datamodule.val_dataset),
                "test_blocks": len(datamodule.test_dataset),
                "sequence_length": policy.max_seq_len,
            },
        }
    )
    return result


def prepare_wikitext_reference_experiment(
    runtime: NotebookRuntime,
    experiment_name: str,
    *,
    tokenizer: BPETokenizer | None = None,
):
    """Generate WikiText references and prepare a BPE PRA datamodule."""
    settings = load_experiment_settings(runtime, experiment_name)
    dataset_cfg = settings["experiment"]["dataset"]
    policy = policy_from_experiment(settings)
    dataset_stage = str(dataset_cfg.get("stage", "wikitext2_references"))
    generator = (
        generate_wikitext_reference_dataset_v2
        if dataset_stage == "wikitext2_references_v2"
        else generate_wikitext_reference_dataset
    )
    data_root = runtime.repo / "data"
    if dataset_cfg.get("split_count") is not None:
        data_root = data_root / "generated" / experiment_artifact_name(
            dataset_stage,
            f"split{dataset_cfg['split_count']}",
            f"seed{dataset_cfg.get('seed', runtime.seed)}",
            f"examples{dataset_cfg['max_examples']}",
        )
    generator(
        data_root,
        dataset_name=dataset_cfg["name"],
        max_examples=int(dataset_cfg["max_examples"]),
        max_reference_parts=int(dataset_cfg.get("max_reference_parts", 5)),
        split_count=(
            int(dataset_cfg["split_count"])
            if dataset_cfg.get("split_count") is not None
            else None
        ),
        seed=int(dataset_cfg.get("seed", runtime.seed)),
        cache_dir=runtime.repo / "data" / ".hf_cache",
    )
    dataset_cls = dataset_class_for_stage(dataset_stage)
    dataset = dataset_cls(data_root, max_examples=int(dataset_cfg["max_examples"]))
    corpus = PRADataModule._corpus(dataset)
    reference_count = (
        int(dataset_cfg["split_count"]) - 1
        if dataset_cfg.get("split_count") is not None
        else int(dataset_cfg.get("max_reference_parts", 5))
    )
    reference_tokens = [f"<REF_{index}>" for index in range(1, reference_count + 1)]
    tokenizer = tokenizer or BPETokenizer.train(
        corpus,
        vocab_size=int(dataset_cfg.get("bpe_vocab_size", 2_000)),
        reference_tokens=reference_tokens,
    )
    torch.manual_seed(runtime.seed)
    datamodule = PRADataModule(
        dataset_stage=dataset_stage,
        data_dir=data_root,
        max_examples=int(dataset_cfg["max_examples"]),
        batch_size=policy.batch_size,
        max_seq_len=policy.max_seq_len,
        shuffle=True,
        pin_memory=runtime.cuda_available,
        tokenizer=tokenizer,
        split_seed=int(dataset_cfg.get("split_seed", dataset_cfg.get("seed", 0))),
    ).load()
    return settings, policy, datamodule


def train_wikitext_reference_experiment(
    runtime: NotebookRuntime,
    experiment_name: str,
    *,
    initial_checkpoint: str | Path | None = None,
    tokenizer_checkpoint: str | Path | None = None,
    tokenizer: BPETokenizer | None = None,
    training_mode: str | None = None,
    artifact_qualifiers: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Train a configured SA or PRA model on reference-split WikiText."""
    checkpoint = torch.load(initial_checkpoint, map_location="cpu") if initial_checkpoint else None
    tokenizer_source = checkpoint
    if tokenizer_source is None and tokenizer_checkpoint:
        tokenizer_source = torch.load(tokenizer_checkpoint, map_location="cpu")
    pretrained_tokenizer = tokenizer or (
        BPETokenizer.from_json(tokenizer_source["tokenizer_json"])
        if tokenizer_source and tokenizer_source.get("tokenizer_json")
        else None
    )
    settings, policy, datamodule = prepare_wikitext_reference_experiment(
        runtime, experiment_name, tokenizer=pretrained_tokenizer
    )
    dataset_cfg = settings["experiment"]["dataset"]
    cfg = _named_pra_config(settings, datamodule.tokenizer, runtime)
    model = TinyPRAModel(cfg)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    training_mode = training_mode or settings["experiment"].get("training_mode")
    if training_mode is None:
        training_mode = (
            "frozen_refpath"
            if checkpoint and cfg.model_variant in {"td_pra", "tdx_pra"}
            else "joint"
            if checkpoint
            else "scratch"
        )
    if training_mode not in {"scratch", "frozen_refpath", "joint"}:
        raise ValueError(f"Unsupported reference training mode: {training_mode}")
    if training_mode in {"frozen_refpath", "joint"} and checkpoint is None:
        raise ValueError(f"{training_mode} requires an initial checkpoint")
    if cfg.model_variant in {"td_pra", "tdx_pra"}:
        for name, module in model.named_modules():
            if name.endswith("mem_o_proj"):
                torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    trainable_parameter_names = []
    if training_mode == "frozen_refpath":
        if cfg.model_variant not in {"td_pra", "tdx_pra"}:
            raise ValueError("frozen_refpath requires a PRA architecture")
        if cfg.memory_transport != "cross_attention":
            raise ValueError(
                "frozen_refpath is the historical cross-attention adaptation regime; "
                "native_kv introduces no reference-path parameters to optimize"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, module in model.named_modules():
            if name.endswith("mem_o_proj"):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        trainable_parameter_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
    else:
        trainable_parameter_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
    train_cfg = settings["experiment"]["train"]
    artifact = experiment_artifact_name(
        settings["model_name"],
        settings["experiment"]["dataset"].get("stage", "wikitext2_references"),
        settings["name"],
        *artifact_qualifiers,
        training_mode,
        f"seed{runtime.seed}",
        f"steps{train_cfg.get('max_steps')}",
        f"seq{policy.max_seq_len}",
    )
    training_config = TrainConfig(
        experiment_name=artifact,
        output_dir=str(runtime.repo / "out" / "notebook_wikitext_refs"),
        seed=runtime.seed,
        device=runtime.device,
        epochs=policy.epochs,
        max_steps=int(train_cfg.get("max_steps") or 0) or None,
        batch_size=policy.batch_size,
        learning_rate=policy.learning_rate,
        eval_every_steps=max(
            int(train_cfg.get("eval_every_steps") or train_cfg.get("max_steps") or 1), 1
        ),
        save_every_steps=max(int(train_cfg.get("max_steps") or 1), 1),
        log_every_steps=1,
        use_tensorboard=False,
        save_metric_plots=False,
        mixed_precision=False,
        dataset_stage=str(
            settings["experiment"]["dataset"].get("stage", "wikitext2_references")
        ),
        data_dir=str(datamodule.data_dir),
        max_examples=len(datamodule.dataset),
        max_seq_len=policy.max_seq_len,
    )
    model.to(runtime.device)
    initial_reference_metrics = evaluate_pra_model(
        model=model,
        loader=datamodule.val_loader(),
        tokenizer=datamodule.tokenizer,
        train_config=training_config,
        device=runtime.device,
        split="initial_val",
    )
    result = train_pra_model(
        cfg=cfg, train_config=training_config, datamodule=datamodule, model=model
    )
    result.update(
        {
            "dataset_stage": str(
                settings["experiment"]["dataset"].get("stage", "wikitext2_references")
            ),
            "size": len(datamodule.dataset),
            "policy": policy,
            "cfg": cfg,
            "train_config": training_config,
            "datamodule": datamodule,
            "model_name": settings["model_name"],
            "experiment_settings": settings,
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
            "tokenizer_checkpoint": str(tokenizer_checkpoint) if tokenizer_checkpoint else None,
            "initial_reference_metrics": initial_reference_metrics,
            "training_mode": training_mode,
            "trainable_parameter_names": trainable_parameter_names,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in result["model"].parameters() if parameter.requires_grad
            ),
            "parameter_count": sum(parameter.numel() for parameter in result["model"].parameters()),
            "loader_summary": {
                "dataset": settings["experiment"]["dataset"]["name"],
                "stage": str(
                    settings["experiment"]["dataset"].get("stage", "wikitext2_references")
                ),
                "generation_seed": settings["experiment"]["dataset"].get("seed", runtime.seed),
                "generation_version": (
                    "wikitext_refs_v2"
                    if settings["experiment"]["dataset"].get("stage") == "wikitext2_references_v2"
                    else "wikitext_refs_v1"
                ),
                "examples": len(datamodule.dataset),
                "generated_data_root": str(datamodule.data_dir),
                "split_count": dataset_cfg.get("split_count", "mixed"),
                "reference_count": (
                    int(dataset_cfg["split_count"]) - 1
                    if dataset_cfg.get("split_count") is not None
                    else "1-" + str(dataset_cfg.get("max_reference_parts", 5))
                ),
                "tokenizer": "Tokenizer(BPE) + Whitespace",
                "tokenizer_shared_across_split_counts": tokenizer is not None,
                "vocab_size": datamodule.tokenizer.vocab_size,
                "train_size": len(datamodule.train_dataset),
                "validation_size": len(datamodule.val_dataset),
                "test_size": len(datamodule.test_dataset),
            },
        }
    )
    return result


def evaluate_plain_after_reference(
    result: dict[str, Any],
    runtime: NotebookRuntime,
    plain_experiment_name: str,
) -> dict[str, float]:
    """Measure whether reference fine-tuning preserved ordinary WikiText modeling."""
    settings = load_experiment_settings(runtime, plain_experiment_name)
    dataset_cfg = settings["experiment"]["dataset"]
    policy = policy_from_experiment(settings)
    datamodule = WikiTextDataModule(
        data_dir=runtime.repo / "data",
        dataset_name=dataset_cfg["name"],
        vocab_size=int(dataset_cfg["vocab_size"]),
        seq_len=policy.max_seq_len,
        batch_size=policy.batch_size,
        max_train_documents=dataset_cfg.get("max_train_documents"),
        max_eval_documents=dataset_cfg.get("max_eval_documents"),
        max_train_blocks=dataset_cfg.get("max_train_blocks"),
        pin_memory=runtime.cuda_available,
        tokenizer=result["datamodule"].tokenizer,
    ).load()
    metrics = evaluate_language_model(
        model=result["model"],
        loader=datamodule.test_loader(),
        device=result["state"].device,
        split="plain_test_after_reference",
    )
    result["plain_test_after_reference"] = metrics
    return metrics


def evaluate_reference_conditions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate the complete reference-content ablation set on one test split."""
    values = [
        evaluate_reference_ablation(
            model=result["model"],
            loader=result["datamodule"].test_loader(),
            tokenizer=result["datamodule"].tokenizer,
            device=result["state"].device,
            condition=condition,
            resolver_config=result["train_config"].resolver_config,
            cache_config=result["train_config"].cache_config,
        )
        for condition in ("valid", "disabled", "shuffled", "irrelevant", "empty", "oracle")
    ]
    result["reference_ablations"] = values
    return values


def generate_aggregate_html_report(
    report_paths: list[str | Path],
    *,
    report_name: str,
    title: str,
    report_root: str | Path,
) -> Path:
    """Aggregate seed-level reports into one experiment report with mean and spread."""
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    report_dir = Path(report_root) / report_name
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    numeric_keys = sorted(
        set.intersection(
            *[
                {key for key, value in payload["results"].items() if isinstance(value, (int, float))}
                for payload in payloads
            ]
        )
    )
    aggregate = {}
    for key in numeric_keys:
        values = [float(payload["results"][key]) for payload in payloads]
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "stddev": statistics.pstdev(values),
            "values": values,
        }
    compact_metrics = [
        key
        for key in (
            "test_loss",
            "test_perplexity",
            "test_token_accuracy",
            "plain_test_after_reference_loss",
            "reference_valid_loss",
            "reference_disabled_loss",
            "reference_shuffled_loss",
            "train_duration_seconds",
            "validation_duration_seconds",
            "wall_clock_seconds",
            "processed_tokens",
        )
        if key in aggregate
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    quality_metrics = [key for key in compact_metrics if key.endswith("loss")]
    axes[0].bar(
        [key.replace("_", " ") for key in quality_metrics],
        [aggregate[key]["mean"] for key in quality_metrics],
        yerr=[aggregate[key]["stddev"] for key in quality_metrics],
        color="#2563eb",
        capsize=4,
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Loss metrics across seeds")
    timing_metrics = [key for key in compact_metrics if key.endswith("duration_seconds") or key == "wall_clock_seconds"]
    axes[1].bar(
        [key.replace("_seconds", "").replace("_", " ") for key in timing_metrics],
        [aggregate[key]["mean"] for key in timing_metrics],
        yerr=[aggregate[key]["stddev"] for key in timing_metrics],
        color="#059669",
        capsize=4,
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Execution time across seeds")
    fig.tight_layout()
    fig.savefig(assets_dir / "aggregate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    rows = {
        key: f"{aggregate[key]['mean']:.6g} +/- {aggregate[key]['stddev']:.3g}"
        for key in compact_metrics
    }
    payload = {
        "report_name": report_name,
        "title": title,
        "generated_at": datetime.now().astimezone().isoformat(),
        "seeds": [payload["runtime"]["seed"] for payload in payloads],
        "model": payloads[0]["model"],
        "dataset": payloads[0]["dataset"],
        "aggregate": aggregate,
        "source_reports": [str(Path(path).resolve()) for path in report_paths],
    }
    (report_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    source_links = "".join(
        f'<li><a href="{Path(path).resolve().as_uri()}">{escape(Path(path).parent.name)}</a></li>'
        for path in report_paths
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font-family:Segoe UI,Arial;margin:32px auto;max-width:1100px;color:#18202a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}code{{font-family:Consolas}}</style></head><body>
<h1>{escape(title)}</h1><p>Seeds: {escape(str(payload['seeds']))}. Values are population mean +/- standard deviation.</p>
{_html_table(rows)}<h2>Quality And Execution Time</h2><img src="assets/aggregate.png"><h2>Seed Reports</h2><ul>{source_links}</ul>
<p><a href="report.json">Structured aggregate JSON</a></p></body></html>"""
    path = report_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def generate_matrix_findings_report(
    aggregate_paths: list[str | Path],
    *,
    report_root: str | Path,
    report_name: str = "wikitext2_pra_experiment_findings",
) -> Path:
    """Create a six-experiment comparison report from aggregate report JSON files."""
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in aggregate_paths]
    report_dir = Path(report_root) / report_name
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    labels = [report["title"].replace("WikiText-2", "").strip(" -") for report in reports]
    test_losses = [report["aggregate"]["test_loss"]["mean"] for report in reports]
    test_stds = [report["aggregate"]["test_loss"]["stddev"] for report in reports]
    train_times = [report["aggregate"]["train_duration_seconds"]["mean"] for report in reports]
    val_times = [report["aggregate"]["validation_duration_seconds"]["mean"] for report in reports]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].bar(labels, test_losses, yerr=test_stds, color=["#2563eb"] * 3 + ["#dc2626"] * 3, capsize=4)
    axes[0].set_ylabel("test loss")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Matched WikiText-2 architecture matrix")
    x = range(len(labels))
    axes[1].bar(x, train_times, label="training", color="#059669")
    axes[1].bar(x, val_times, bottom=train_times, label="validation", color="#d97706")
    axes[1].set_xticks(list(x), labels, rotation=20)
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Mean synchronized execution time")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(assets_dir / "matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    rows = []
    for report, label in zip(reports, labels):
        aggregate = report["aggregate"]
        rows.append(
            {
                "experiment": label,
                "test_loss": aggregate["test_loss"]["mean"],
                "plain_after_ref": aggregate.get("plain_test_after_reference_loss", {}).get("mean"),
                "valid_ref_loss": aggregate.get("reference_valid_loss", {}).get("mean"),
                "disabled_ref_loss": aggregate.get("reference_disabled_loss", {}).get("mean"),
                "train_seconds": aggregate["train_duration_seconds"]["mean"],
                "validation_seconds": aggregate["validation_duration_seconds"]["mean"],
            }
        )
    by_name = {report["report_name"]: report for report in reports}

    def mean(report_name: str, metric: str) -> float:
        return float(by_name[report_name]["aggregate"][metric]["mean"])

    plain_names = {
        "SelfAttention": "final_td_sa_tiny_wikitext2_plain",
        "full PRAttention": "final_td_pra_tiny_wikitext2_plain",
        "hybrid PRAttention": "final_tdx_pra_tiny_wikitext2_plain",
    }
    reference_names = {
        "SelfAttention": "final_td_sa_tiny_wikitext2_references",
        "full PRAttention": "final_td_pra_tiny_wikitext2_references",
        "hybrid PRAttention": "final_tdx_pra_tiny_wikitext2_references",
    }
    retention_deltas = {
        architecture: mean(reference_names[architecture], "plain_test_after_reference_loss")
        - mean(plain_names[architecture], "test_loss")
        for architecture in plain_names
    }
    reference_benefits = {
        architecture: mean(reference_names[architecture], "reference_disabled_loss")
        - mean(reference_names[architecture], "reference_valid_loss")
        for architecture in ("full PRAttention", "hybrid PRAttention")
    }
    shuffled_penalties = {
        architecture: mean(reference_names[architecture], "reference_shuffled_loss")
        - mean(reference_names[architecture], "reference_valid_loss")
        for architecture in ("full PRAttention", "hybrid PRAttention")
    }
    findings = [
        (
            "Plain WikiText equivalence",
            "All three tiny architectures learned the plain language-modeling task at the matched token budget. "
            f"Mean test losses were {mean(plain_names['SelfAttention'], 'test_loss'):.4f} for SelfAttention, "
            f"{mean(plain_names['full PRAttention'], 'test_loss'):.4f} for full PRAttention, and "
            f"{mean(plain_names['hybrid PRAttention'], 'test_loss'):.4f} for hybrid PRAttention.",
        ),
        (
            "Plain-language retention",
            f"After reference-format fine-tuning, SelfAttention plain WikiText loss increased by {retention_deltas['SelfAttention']:.4f}. "
            f"Full and hybrid PRAttention changed by {retention_deltas['full PRAttention']:.4f} and "
            f"{retention_deltas['hybrid PRAttention']:.4f}, respectively, because phase 2 trained only their reference output projections.",
        ),
        (
            "Reference utility",
            f"Valid references reduced loss relative to disabled references by {reference_benefits['full PRAttention']:.4f} for full PRAttention "
            f"and {reference_benefits['hybrid PRAttention']:.4f} for the hybrid. SelfAttention has no reference path, so its valid/disabled delta is zero by construction.",
        ),
        (
            "Reference specificity",
            f"Shuffling reference documents increased loss by {shuffled_penalties['full PRAttention']:.4f} for full PRAttention and "
            f"{shuffled_penalties['hybrid PRAttention']:.4f} for the hybrid. The consistent but modest penalty shows content sensitivity while leaving room for stronger reference-use training.",
        ),
        (
            "Full versus hybrid PRAttention",
            f"Mean valid-reference loss was {mean(reference_names['full PRAttention'], 'reference_valid_loss'):.4f} for full PRAttention and "
            f"{mean(reference_names['hybrid PRAttention'], 'reference_valid_loss'):.4f} for hybrid PRAttention. "
            "Full PRAttention was modestly better in this two-seed tiny-model pilot.",
        ),
    ]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiments": rows,
        "findings": [{"title": title, "text": value} for title, value in findings],
        "retention_loss_deltas": retention_deltas,
        "valid_reference_loss_benefits": reference_benefits,
        "shuffled_reference_loss_penalties": shuffled_penalties,
        "aggregate_reports": [str(Path(path).resolve()) for path in aggregate_paths],
    }
    (report_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    headers = list(rows[0])
    table = "<table><tr>" + "".join(f"<th>{escape(key.replace('_', ' ').title())}</th>" for key in headers) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{escape(f'{row[key]:.5f}' if isinstance(row[key], float) else str(row[key]))}</td>" for key in headers) + "</tr>"
    table += "</table>"
    links = "".join(f'<li><a href="{Path(path).resolve().with_name("index.html").as_uri()}">{escape(report["title"])}</a></li>' for path, report in zip(aggregate_paths, reports))
    finding_html = "".join(
        f"<h3>{escape(title)}</h3><p>{escape(value)}</p>" for title, value in findings
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>WikiText-2 PRA experiment findings</title>
<style>body{{font-family:Segoe UI,Arial;margin:32px auto;max-width:1200px;color:#18202a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}</style></head><body>
<h1>WikiText-2 PRA Experiment Findings</h1><p>Two-seed matched-budget comparison of SelfAttention, full PRAttention, and hybrid architectures on plain and reference-conditioned continuation.</p>
<h2>Conclusions</h2>{finding_html}<h2>Measured Results</h2>{table}<img src="assets/matrix.png"><h2>Six Experiment Reports</h2><ul>{links}</ul><p><a href="report.json">Structured findings JSON</a></p></body></html>"""
    path = report_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
