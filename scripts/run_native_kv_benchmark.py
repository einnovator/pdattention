"""Train full-context controls and evaluate native-KV fixed-source benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data.collators import PRACollator  # noqa: E402
from data.datamodules import PRADataModule  # noqa: E402
from data.datasets import SyntheticNativeKVFixedTargetDataset  # noqa: E402
from data.native_kv_benchmarks import (  # noqa: E402
    NATIVE_KV_SPLIT_COUNTS,
    generate_synthetic_native_kv_dataset,
)
from data.tokenizer import PRATokenizer  # noqa: E402
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402
from pra_torch.pra_train import evaluate_reference_ablation  # noqa: E402


SEEDS = (1, 7, 21, 42, 87)
PAD_ID = 0
SYNTHETIC_GENERATION_VERSION = "synthetic_nativekv_fixed_target_v5"
DATASET_DEFAULTS = {
    "synthetic": {
        "max_examples": 64,
        "train_examples": 2_048,
        "max_seq_len": 192,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "batch_size": 32,
        "steps": 100,
        "learning_rate": 1e-3,
    }
}


class FullContextDataset(Dataset):
    """Expose the invariant historical source directly before the local tail."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        source = str(sample.metadata["row"]["source_text"])
        return replace(sample, question=f"{source}{sample.question}")


class AnswerTokenCollator(PRACollator):
    """Supervise only the first answer token from an answer-free local prompt."""

    def __call__(self, samples):
        batch = super().__call__(samples)
        input_rows = []
        label_rows = []
        attention_rows = []
        for sample in samples:
            prompt_ids = self.tokenizer.encode(sample.question)[: self.max_seq_len]
            answer_ids = self.tokenizer.encode(sample.answer.strip())
            if not prompt_ids or not answer_ids:
                raise ValueError(f"Benchmark sample {sample.id} has an empty prompt or answer")
            labels = torch.full((len(prompt_ids),), self.pad_token_id, dtype=torch.long)
            labels[-1] = int(answer_ids[0])
            input_rows.append(torch.tensor(prompt_ids, dtype=torch.long))
            label_rows.append(labels)
            attention_rows.append(torch.ones(len(prompt_ids), dtype=torch.long))
        batch["input_ids"] = self._pad(input_rows, self.pad_token_id)
        batch["labels"] = self._pad(label_rows, self.pad_token_id)
        batch["attention_mask"] = self._pad(attention_rows, 0)
        return batch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, collator, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _subset_indices(datamodule: PRADataModule, split: str) -> list[int]:
    subset = getattr(datamodule, f"{split}_dataset")
    return list(getattr(subset, "indices", range(len(subset))))


def _evaluate_model(model, loader, device: str) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    token_count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, use_pra_memory=False)
            valid = labels.ne(PAD_ID)
            loss_sum += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=PAD_ID,
                    reduction="sum",
                ).cpu()
            )
            correct += int((logits.argmax(dim=-1).eq(labels) & valid).sum().item())
            token_count += int(valid.sum().item())
    loss = loss_sum / max(token_count, 1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "token_accuracy": correct / max(token_count, 1),
        "tokens": token_count,
        "duration_seconds": time.perf_counter() - start,
    }


def _checkpoint_payload(model, tokenizer, step: int, settings: dict[str, Any]) -> dict:
    return {
        "model": model.state_dict(),
        "cfg": model.cfg.__dict__,
        "step": step,
        "stoi": dict(tokenizer.stoi),
        "tokenizer_type": type(tokenizer).__name__,
        "tokenizer_json": tokenizer.to_json() if hasattr(tokenizer, "to_json") else None,
        "settings": settings,
    }


def train_full_context_sa(
    *,
    seed: int,
    tokenizer,
    datamodule: PRADataModule,
    settings: dict[str, Any],
    run_dir: Path,
    device: str,
    force: bool,
) -> tuple[TinyPRAModel, dict[str, Any]]:
    """Train or resume the ordinary full-context model used by every condition."""

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    cfg = PRAConfig(
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
    model = TinyPRAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=0.0
    )
    start_step = 0
    if checkpoint_path.exists() and not force:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint.get("step", 0))
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])

    full_dataset = FullContextDataset(datamodule.dataset)
    collator = AnswerTokenCollator(tokenizer, max_seq_len=int(settings["max_seq_len"]))
    train_subset = Subset(full_dataset, _subset_indices(datamodule, "train"))
    val_subset = Subset(full_dataset, _subset_indices(datamodule, "val"))
    train_loader = _loader(
        train_subset,
        collator,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    val_loader = _loader(
        val_subset,
        collator,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        seed=seed,
    )
    iterator = iter(train_loader)
    history = []
    train_start = time.perf_counter()
    model.train()
    max_steps = int(settings["steps"])
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, use_pra_memory=False)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=PAD_ID,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == max_steps:
            validation = _evaluate_model(model, val_loader, device)
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": validation["loss"],
                "validation_token_accuracy": validation["token_accuracy"],
            }
            history.append(record)
            print(
                f"seed={seed} step={step}/{max_steps} "
                f"train={record['train_loss']:.4f} val={record['validation_loss']:.4f} "
                f"acc={record['validation_token_accuracy']:.3f}",
                flush=True,
            )
            model.train()
        if step % 250 == 0 or step == max_steps:
            payload = _checkpoint_payload(model, tokenizer, step, settings)
            payload["optimizer"] = optimizer.state_dict()
            torch.save(payload, checkpoint_path)

    return model.eval(), {
        "start_step": start_step,
        "final_step": max_steps,
        "train_seconds": time.perf_counter() - train_start,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def _prepare_synthetic(settings: dict[str, Any]):
    def ensure(root: Path, *, split_count: int, max_examples: int, seed: int) -> None:
        manifest_path = root / SyntheticNativeKVFixedTargetDataset.stage / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        expected = {
            "dataset": "synthetic_native_kv",
            "split_count": split_count,
            "example_count": max_examples,
            "generation_version": SYNTHETIC_GENERATION_VERSION,
        }
        if manifest != expected:
            generate_synthetic_native_kv_dataset(
                root,
                split_count=split_count,
                max_examples=max_examples,
                seed=seed,
            )

    training_root = REPO / "out" / "native_kv_data" / "synthetic" / "training"
    ensure(
        training_root,
        split_count=2,
        max_examples=int(settings["train_examples"]),
        seed=1729,
    )
    training_dataset = SyntheticNativeKVFixedTargetDataset(training_root)
    tokenizer = PRATokenizer(PRADataModule._corpus(training_dataset))
    training_module = PRADataModule(
        dataset_stage=SyntheticNativeKVFixedTargetDataset.stage,
        data_dir=str(training_root),
        max_examples=int(settings["train_examples"]),
        batch_size=int(settings["batch_size"]),
        max_seq_len=int(settings["max_seq_len"]),
        shuffle=True,
        tokenizer=tokenizer,
        split_seed=1729,
    ).load()
    roots = {}
    for split_count in NATIVE_KV_SPLIT_COUNTS:
        root = REPO / "out" / "native_kv_data" / "synthetic" / f"split-{split_count}"
        ensure(
            root,
            split_count=split_count,
            max_examples=int(settings["max_examples"]),
            seed=91_973,
        )
        roots[split_count] = root
    modules = {
        split_count: PRADataModule(
            dataset_stage=SyntheticNativeKVFixedTargetDataset.stage,
            data_dir=str(root),
            max_examples=int(settings["max_examples"]),
            batch_size=1,
            max_seq_len=int(settings["max_seq_len"]),
            shuffle=False,
            tokenizer=tokenizer,
            split_seed=1729,
        ).load()
        for split_count, root in roots.items()
    }
    for datamodule in modules.values():
        datamodule.collator = AnswerTokenCollator(
            tokenizer, max_seq_len=int(settings["max_seq_len"])
        )
        datamodule.test_dataset = datamodule.dataset
    return tokenizer, training_module, modules


def _native_config(source: TinyPRAModel, device: str) -> PRAConfig:
    return PRAConfig(
        vocab_size=source.cfg.vocab_size,
        d_model=source.cfg.d_model,
        n_heads=source.cfg.n_heads,
        n_layers=source.cfg.n_layers,
        d_ff=source.cfg.d_ff,
        max_seq_len=source.cfg.max_seq_len,
        dropout=0.0,
        model_variant="td_pra",
        memory_transport="native_kv",
        top_k_references=1,
        top_k_chunks_per_reference=1,
        trigger_threshold=float("-inf"),
        detail_materialization="selected_chunks",
        recursive_max_total_references=128,
        recursive_max_total_tokens=8_192,
        device=device,
    )


def evaluate_seed(
    *,
    source: TinyPRAModel,
    tokenizer,
    modules: dict[int, PRADataModule],
    settings: dict[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    training_module = modules[2]
    full_dataset = FullContextDataset(training_module.dataset)
    collator = AnswerTokenCollator(tokenizer, max_seq_len=int(settings["max_seq_len"]))
    test_indices = _subset_indices(training_module, "test")
    full_loader = _loader(
        Subset(full_dataset, test_indices), collator, batch_size=1, shuffle=False, seed=0
    )
    tail_loader = _loader(
        training_module.test_dataset, collator, batch_size=1, shuffle=False, seed=0
    )
    sa_full = _evaluate_model(source, full_loader, device)
    sa_tail = _evaluate_model(source, tail_loader, device)
    converted = convert_sa_model_to_pra(source, _native_config(source, device)).to(device).eval()

    rows = []
    for split_count, datamodule in modules.items():
        condition_results = {}
        for condition in (
            "native_all",
            "native_oracle",
            "valid",
            "native_shuffled",
            "native_disabled",
        ):
            result = evaluate_reference_ablation(
                model=converted,
                loader=datamodule.test_loader(),
                tokenizer=tokenizer,
                device=device,
                condition=condition,
            )
            condition_results[condition] = result
        all_result = condition_results["native_all"]
        oracle = condition_results["native_oracle"]
        shuffled = condition_results["native_shuffled"]
        routed = condition_results["valid"]
        disabled = condition_results["native_disabled"]
        row = {
            "split_count": split_count,
            "reference_count": split_count - 1,
            "sa_full_loss": sa_full["loss"],
            "sa_full_accuracy": sa_full["token_accuracy"],
            "sa_tail_loss": sa_tail["loss"],
            "sa_tail_accuracy": sa_tail["token_accuracy"],
            "native_all_loss": all_result["loss"],
            "native_all_accuracy": all_result["token_accuracy"],
            "native_oracle_loss": oracle["loss"],
            "native_oracle_accuracy": oracle["token_accuracy"],
            "native_routed_loss": routed["loss"],
            "native_routed_accuracy": routed["token_accuracy"],
            "native_shuffled_loss": shuffled["loss"],
            "native_shuffled_accuracy": shuffled["token_accuracy"],
            "native_disabled_loss": disabled["loss"],
            "native_disabled_accuracy": disabled["token_accuracy"],
            "transport_gap": all_result["loss"] - sa_full["loss"],
            "sparse_gap": oracle["loss"] - all_result["loss"],
            "memory_benefit": sa_tail["loss"] - oracle["loss"],
            "content_causality": shuffled["loss"] - oracle["loss"],
            "dependency_gain": sa_tail["loss"] - sa_full["loss"],
        }
        print(
            f"split={split_count:>2} full={row['sa_full_loss']:.4f} "
            f"tail={row['sa_tail_loss']:.4f} all={row['native_all_loss']:.4f} "
            f"oracle={row['native_oracle_loss']:.4f} routed={row['native_routed_loss']:.4f} "
            f"shuffled={row['native_shuffled_loss']:.4f}",
            flush=True,
        )
        rows.append(row)
    return rows


def _aggregate(seed_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    numeric_keys = [
        key
        for key, value in seed_payloads[0]["results"][0].items()
        if isinstance(value, (int, float)) and key not in {"split_count", "reference_count"}
    ]
    for split_count in NATIVE_KV_SPLIT_COUNTS:
        matching = [
            row
            for payload in seed_payloads
            for row in payload["results"]
            if int(row["split_count"]) == split_count
        ]
        aggregate = {"split_count": split_count, "reference_count": split_count - 1}
        for key in numeric_keys:
            values = [float(row[key]) for row in matching]
            aggregate[key] = {
                "mean": statistics.fmean(values),
                "stddev": statistics.pstdev(values),
                "values": values,
            }
        rows.append(aggregate)
    return rows


def _plot_report(dataset_name: str, aggregate: list[dict[str, Any]], output: Path) -> None:
    splits = [row["split_count"] for row in aggregate]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    losses = (
        ("SA full", "sa_full_loss", "#245A8D", "o"),
        ("SA tail", "sa_tail_loss", "#5F6B76", "o"),
        ("Native all", "native_all_loss", "#2D7A68", "s"),
        ("Native oracle", "native_oracle_loss", "#A66A18", "D"),
        ("Native routed", "native_routed_loss", "#8B5A9F", "^"),
        ("Native shuffled", "native_shuffled_loss", "#A33D3D", "x"),
    )
    for label, key, color, marker in losses:
        axes[0].errorbar(
            splits,
            [row[key]["mean"] for row in aggregate],
            yerr=[row[key]["stddev"] for row in aggregate],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.2,
            capsize=2,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(splits, [str(value) for value in splits])
    axes[0].set_xlabel("Total split count")
    axes[0].set_ylabel("Answer-token loss")
    axes[0].set_title("A. Quality by source partition")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)

    gaps = (
        ("Memory benefit", "memory_benefit", "#2D7A68"),
        ("Content causality", "content_causality", "#A66A18"),
        ("Transport gap", "transport_gap", "#A33D3D"),
    )
    for label, key, color in gaps:
        axes[1].errorbar(
            splits,
            [row[key]["mean"] for row in aggregate],
            yerr=[row[key]["stddev"] for row in aggregate],
            label=label,
            color=color,
            marker="o",
            linewidth=1.2,
            capsize=2,
        )
    axes[1].axhline(0.0, color="#30363D", linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(splits, [str(value) for value in splits])
    axes[1].set_xlabel("Total split count")
    axes[1].set_ylabel("Loss difference")
    axes[1].set_title("B. Preregistered native-KV gaps")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D8DEE4", linewidth=0.7)
        axis.set_axisbelow(True)
    figure.suptitle(f"{dataset_name} native-KV benchmark (five paired seeds)", fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", metadata={"Creator": Path(__file__).name})
    plt.close(figure)


def _write_report(dataset_name: str, seed_payloads: list[dict[str, Any]], publish: bool) -> Path:
    aggregate = _aggregate(seed_payloads)
    report_dir = REPO / "out" / "reports" / f"native_kv_{dataset_name}_5seed"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "seeds": [payload["seed"] for payload in seed_payloads],
        "split_counts": list(NATIVE_KV_SPLIT_COUNTS),
        "transport": "native_kv",
        "training": "full-context SelfAttention followed by weight-preserving PRA conversion",
        "seed_results": seed_payloads,
        "aggregate": aggregate,
    }
    report_json = report_dir / "report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figure_path = report_dir / "native_kv_split_scaling.pdf"
    _plot_report(dataset_name, aggregate, figure_path)
    html_rows = "".join(
        "<tr>"
        f"<td>{row['split_count']}</td>"
        f"<td>{row['sa_full_loss']['mean']:.4f} +/- {row['sa_full_loss']['stddev']:.4f}</td>"
        f"<td>{row['sa_tail_loss']['mean']:.4f} +/- {row['sa_tail_loss']['stddev']:.4f}</td>"
        f"<td>{row['native_all_loss']['mean']:.4f} +/- {row['native_all_loss']['stddev']:.4f}</td>"
        f"<td>{row['native_oracle_loss']['mean']:.4f} +/- {row['native_oracle_loss']['stddev']:.4f}</td>"
        f"<td>{row['native_shuffled_loss']['mean']:.4f} +/- {row['native_shuffled_loss']['stddev']:.4f}</td>"
        "</tr>"
        for row in aggregate
    )
    (report_dir / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Native-KV benchmark</title>"
        "<style>body{font-family:Segoe UI,Arial;margin:32px auto;max-width:1100px;color:#18202a}"
        "table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}"
        "embed{width:100%;height:520px}</style></head><body>"
        f"<h1>{escape(dataset_name)} native-KV benchmark</h1>"
        f"<p>Seeds: {escape(str(payload['seeds']))}. Lower loss is better.</p>"
        "<table><thead><tr><th>Splits</th><th>SA full</th><th>SA tail</th><th>Native all</th>"
        f"<th>Native oracle</th><th>Native shuffled</th></tr></thead><tbody>{html_rows}</tbody></table>"
        "<h2>Split scaling</h2><embed src='native_kv_split_scaling.pdf' type='application/pdf'>"
        "<p><a href='report.json'>Structured results</a></p></body></html>",
        encoding="utf-8",
    )
    if publish:
        result_dir = REPO / "docs" / "papers" / "shared" / "results"
        figure_dir = REPO / "docs" / "papers" / "shared" / "figures"
        result_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"native_kv_{dataset_name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        _plot_report(dataset_name, aggregate, figure_dir / f"native_kv_{dataset_name}.pdf")
    return report_dir / "index.html"


def run(args: argparse.Namespace) -> Path:
    defaults = dict(DATASET_DEFAULTS[args.dataset])
    for key in ("max_examples", "steps", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            defaults[key] = value
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, training_module, modules = _prepare_synthetic(defaults)
    seed_payloads = []
    for seed in args.seeds:
        _set_seed(seed)
        run_dir = REPO / "out" / "native_kv_benchmarks" / args.dataset / f"seed-{seed}"
        result_path = run_dir / "result.json"
        source, training = train_full_context_sa(
            seed=seed,
            tokenizer=tokenizer,
            datamodule=training_module,
            settings=defaults,
            run_dir=run_dir,
            device=device,
            force=args.force,
        )
        existing_payload = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        )
        can_reuse_result = (
            existing_payload is not None
            and training["start_step"] >= int(defaults["steps"])
            and existing_payload.get("generation_version") == SYNTHETIC_GENERATION_VERSION
            and existing_payload.get("settings") == defaults
            and not args.force
        )
        if can_reuse_result:
            payload = existing_payload
        else:
            payload = {
                "dataset": args.dataset,
                "generation_version": SYNTHETIC_GENERATION_VERSION,
                "seed": seed,
                "device": device,
                "settings": defaults,
                "training": training,
                "results": evaluate_seed(
                    source=source,
                    tokenizer=tokenizer,
                    modules=modules,
                    settings=defaults,
                    device=device,
                ),
            }
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        seed_payloads.append(payload)
    return _write_report(args.dataset, seed_payloads, args.publish)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="synthetic")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    report = run(parse_args())
    print(f"report: {report}")
