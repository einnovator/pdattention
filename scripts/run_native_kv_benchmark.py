"""Train full-context controls and evaluate native-KV fixed-source benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
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
from data.datasets import (  # noqa: E402
    HotpotQANativeKVFixedTargetDataset,
    QASPERNativeKVFixedTargetDataset,
    SyntheticNativeKVFixedTargetDataset,
)
from data.native_kv_benchmarks import (  # noqa: E402
    NATIVE_KV_SPLIT_COUNTS,
    generate_hotpotqa_native_kv_dataset,
    generate_qasper_native_kv_dataset,
    generate_synthetic_native_kv_dataset,
)
from data.tokenizer import BPETokenizer, PRATokenizer  # noqa: E402
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402
from pra_torch.native_metrics import derive_native_kv_metrics, finite_values  # noqa: E402
from pra_torch.pra_train import evaluate_reference_ablation  # noqa: E402


SEEDS = (1, 7, 21, 42, 87)
PAD_ID = 0
SYNTHETIC_GENERATION_VERSION = "synthetic_nativekv_fixed_target_v5"
HOTPOTQA_GENERATION_VERSION = "hotpotqa_nativekv_answer_code_v3"
QASPER_GENERATION_VERSION = "qasper_nativekv_answer_code_v2"
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
        "generation_version": SYNTHETIC_GENERATION_VERSION,
    },
    "hotpotqa": {
        "max_examples": 64,
        "train_examples": 4_000,
        "max_seq_len": 256,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "batch_size": 32,
        "steps": 150,
        "learning_rate": 7e-4,
        "vocab_size": 8_000,
        "generation_version": HOTPOTQA_GENERATION_VERSION,
    },
    "qasper": {
        "max_examples": 64,
        "train_examples": 200,
        "max_seq_len": 256,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "batch_size": 32,
        "steps": 150,
        "learning_rate": 7e-4,
        "vocab_size": 8_000,
        "generation_version": QASPER_GENERATION_VERSION,
    },
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


def _evaluate_model(model, loader, device: str, *, condition: str, tokenizer) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    token_count = 0
    per_example = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, use_pra_memory=False)
            valid = labels.ne(PAD_ID)
            for index, item in enumerate(batch["metadata"]):
                item_labels = labels[index : index + 1]
                item_logits = logits[index : index + 1]
                item_valid = item_labels.ne(PAD_ID)
                count = int(item_valid.sum().item())
                item_loss_sum = float(
                    F.cross_entropy(
                        item_logits.reshape(-1, item_logits.size(-1)),
                        item_labels.reshape(-1),
                        ignore_index=PAD_ID,
                        reduction="sum",
                    ).cpu()
                )
                item_correct = int(
                    (item_logits.argmax(dim=-1).eq(item_labels) & item_valid).sum().item()
                )
                loss_sum += item_loss_sum
                correct += item_correct
                token_count += count
                row = item["sample"].metadata.get("row", {})
                local_tokens = len(tokenizer.encode(str(row.get("prompt", item["question"]))))
                displaced_tokens = len(tokenizer.encode(str(row.get("source_text", ""))))
                accessible_tokens = local_tokens + displaced_tokens
                active_tokens = accessible_tokens if condition == "sa_full" else local_tokens
                per_example.append(
                    {
                        "example_id": str(item["id"]),
                        "condition": condition,
                        "loss": item_loss_sum / max(count, 1),
                        "perplexity": math.exp(min(item_loss_sum / max(count, 1), 20.0)),
                        "token_accuracy": item_correct / max(count, 1),
                        "target_tokens": count,
                        "local_tokens": local_tokens,
                        "displaced_tokens": displaced_tokens,
                        "accessible_tokens": accessible_tokens,
                        "retrieved_tokens": 0.0,
                        "active_tokens": active_tokens,
                        "active_fraction": active_tokens / max(accessible_tokens, 1),
                        "num_references": int(row.get("reference_count", 0)),
                        "num_chunks": int(row.get("reference_count", 0)),
                        "num_selected_chunks": 0,
                        "num_selected_references": 0,
                        "fixed_target_id": row.get("fixed_target_id"),
                    }
                )
    loss = loss_sum / max(token_count, 1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "token_accuracy": correct / max(token_count, 1),
        "tokens": token_count,
        "duration_seconds": time.perf_counter() - start,
        "per_example": per_example,
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
        if checkpoint.get("settings") == settings:
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
            validation = _evaluate_model(
                model,
                val_loader,
                device,
                condition="sa_full",
                tokenizer=tokenizer,
            )
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


def _prepare_hotpotqa(settings: dict[str, Any]):
    def ensure(
        root: Path,
        *,
        split_count: int,
        dataset_split: str,
        max_examples: int,
        seed: int,
    ) -> None:
        manifest_path = root / HotpotQANativeKVFixedTargetDataset.stage / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        expected = {
            "dataset": "hotpotqa",
            "split_count": split_count,
            "example_count": max_examples,
            "generation_version": HOTPOTQA_GENERATION_VERSION,
        }
        if manifest != expected:
            generate_hotpotqa_native_kv_dataset(
                root,
                split_count=split_count,
                dataset_split=dataset_split,
                max_examples=max_examples,
                seed=seed,
                cache_dir=REPO / "out" / "hf_cache",
            )

    training_root = REPO / "out" / "native_kv_data" / "hotpotqa" / "training"
    ensure(
        training_root,
        split_count=2,
        dataset_split="train",
        max_examples=int(settings["train_examples"]),
        seed=31_337,
    )
    training_dataset = HotpotQANativeKVFixedTargetDataset(training_root)
    corpus = PRADataModule._corpus(training_dataset)
    tokenizer = BPETokenizer.train(
        corpus,
        vocab_size=int(settings["vocab_size"]),
        min_frequency=2,
    )
    prompt_lengths = [
        len(tokenizer.encode(sample.metadata["row"]["source_text"] + sample.question))
        for sample in training_dataset
    ]
    if max(prompt_lengths, default=0) > int(settings["max_seq_len"]):
        raise ValueError(
            "HotpotQA full-context prompt exceeds max_seq_len: "
            f"max={max(prompt_lengths)}, limit={settings['max_seq_len']}"
        )
    training_module = PRADataModule(
        dataset_stage=HotpotQANativeKVFixedTargetDataset.stage,
        data_dir=str(training_root),
        max_examples=int(settings["train_examples"]),
        batch_size=int(settings["batch_size"]),
        max_seq_len=int(settings["max_seq_len"]),
        shuffle=True,
        tokenizer=tokenizer,
        split_seed=31_337,
    ).load()
    roots = {}
    for split_count in NATIVE_KV_SPLIT_COUNTS:
        root = REPO / "out" / "native_kv_data" / "hotpotqa" / f"split-{split_count}"
        ensure(
            root,
            split_count=split_count,
            dataset_split="validation",
            max_examples=int(settings["max_examples"]),
            seed=72_991,
        )
        roots[split_count] = root
    modules = {
        split_count: PRADataModule(
            dataset_stage=HotpotQANativeKVFixedTargetDataset.stage,
            data_dir=str(root),
            max_examples=int(settings["max_examples"]),
            batch_size=1,
            max_seq_len=int(settings["max_seq_len"]),
            shuffle=False,
            tokenizer=tokenizer,
            split_seed=31_337,
        ).load()
        for split_count, root in roots.items()
    }
    for datamodule in modules.values():
        datamodule.collator = AnswerTokenCollator(
            tokenizer, max_seq_len=int(settings["max_seq_len"])
        )
        datamodule.test_dataset = datamodule.dataset
    return tokenizer, training_module, modules


def _prepare_qasper(settings: dict[str, Any]):
    def ensure(
        root: Path,
        *,
        split_count: int,
        dataset_split: str,
        max_examples: int,
        seed: int,
    ) -> None:
        manifest_path = root / QASPERNativeKVFixedTargetDataset.stage / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        expected = {
            "dataset": "qasper",
            "split_count": split_count,
            "example_count": max_examples,
            "generation_version": QASPER_GENERATION_VERSION,
        }
        if manifest != expected:
            generate_qasper_native_kv_dataset(
                root,
                split_count=split_count,
                dataset_split=dataset_split,
                max_examples=max_examples,
                seed=seed,
                cache_dir=REPO / "out" / "hf_cache" / "qasper",
            )

    training_root = REPO / "out" / "native_kv_data" / "qasper" / "training"
    ensure(
        training_root,
        split_count=2,
        dataset_split="train",
        max_examples=int(settings["train_examples"]),
        seed=44_321,
    )
    training_dataset = QASPERNativeKVFixedTargetDataset(training_root)
    tokenizer = BPETokenizer.train(
        PRADataModule._corpus(training_dataset),
        vocab_size=int(settings["vocab_size"]),
        min_frequency=2,
    )
    prompt_lengths = [
        len(tokenizer.encode(sample.metadata["row"]["source_text"] + sample.question))
        for sample in training_dataset
    ]
    if max(prompt_lengths, default=0) > int(settings["max_seq_len"]):
        raise ValueError(
            "QASPER full-context prompt exceeds max_seq_len: "
            f"max={max(prompt_lengths)}, limit={settings['max_seq_len']}"
        )
    training_module = PRADataModule(
        dataset_stage=QASPERNativeKVFixedTargetDataset.stage,
        data_dir=str(training_root),
        max_examples=int(settings["train_examples"]),
        batch_size=int(settings["batch_size"]),
        max_seq_len=int(settings["max_seq_len"]),
        shuffle=True,
        tokenizer=tokenizer,
        split_seed=44_321,
    ).load()
    roots = {}
    for split_count in NATIVE_KV_SPLIT_COUNTS:
        root = REPO / "out" / "native_kv_data" / "qasper" / f"split-{split_count}"
        ensure(
            root,
            split_count=split_count,
            dataset_split="validation",
            max_examples=int(settings["max_examples"]),
            seed=82_811,
        )
        roots[split_count] = root
    modules = {
        split_count: PRADataModule(
            dataset_stage=QASPERNativeKVFixedTargetDataset.stage,
            data_dir=str(root),
            max_examples=int(settings["max_examples"]),
            batch_size=1,
            max_seq_len=int(settings["max_seq_len"]),
            shuffle=False,
            tokenizer=tokenizer,
            split_seed=44_321,
        ).load()
        for split_count, root in roots.items()
    }
    for datamodule in modules.values():
        datamodule.collator = AnswerTokenCollator(
            tokenizer, max_seq_len=int(settings["max_seq_len"])
        )
        datamodule.test_dataset = datamodule.dataset
    return tokenizer, training_module, modules


DATASET_PREPARERS = {
    "synthetic": _prepare_synthetic,
    "hotpotqa": _prepare_hotpotqa,
    "qasper": _prepare_qasper,
}


def _assert_fixed_target_invariants(modules: dict[int, PRADataModule]) -> None:
    """Fail fast if partitioning changes any evaluated source-tail target."""

    baseline = {
        sample.id: (
            sample.metadata["row"]["source_text"],
            sample.question,
            sample.answer,
            sample.metadata["row"]["fixed_target_id"],
        )
        for sample in modules[min(modules)].dataset
    }
    for split_count, datamodule in modules.items():
        current = {
            sample.id: (
                sample.metadata["row"]["source_text"],
                sample.question,
                sample.answer,
                sample.metadata["row"]["fixed_target_id"],
            )
            for sample in datamodule.dataset
        }
        if current != baseline:
            raise AssertionError(f"Fixed-target invariant failed for split {split_count}")


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
        collect_detailed_timing=True,
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
    sa_full = _evaluate_model(
        source, full_loader, device, condition="sa_full", tokenizer=tokenizer
    )
    sa_tail = _evaluate_model(
        source, tail_loader, device, condition="sa_tail", tokenizer=tokenizer
    )
    converted = convert_sa_model_to_pra(source, _native_config(source, device)).to(device).eval()

    rows = []
    raw_rows = []
    encoded_entry_cache = {}
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
                collect_per_example=True,
                encoded_entry_cache=encoded_entry_cache,
            )
            condition_results[condition] = result
        all_result = condition_results["native_all"]
        oracle = condition_results["native_oracle"]
        shuffled = condition_results["native_shuffled"]
        routed = condition_results["valid"]
        disabled = condition_results["native_disabled"]
        losses = {
            "sa_full": sa_full["loss"],
            "sa_tail": sa_tail["loss"],
            "native_all": all_result["loss"],
            "native_oracle": oracle["loss"],
            "native_routed": routed["loss"],
            "native_shuffled": shuffled["loss"],
            "native_disabled": disabled["loss"],
        }
        derived = derive_native_kv_metrics(losses)
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
            **derived,
        }
        condition_rows = {
            "sa_full": sa_full["per_example"],
            "sa_tail": sa_tail["per_example"],
            "native_all": all_result["per_example"],
            "native_oracle": oracle["per_example"],
            "native_routed": routed["per_example"],
            "native_shuffled": shuffled["per_example"],
            "native_disabled": disabled["per_example"],
        }
        for canonical_condition, values in condition_rows.items():
            for value in values:
                raw_rows.append(
                    {
                        **value,
                        "split_count": split_count,
                        "num_references": split_count - 1,
                        "num_chunks": (
                            value.get("num_chunks", 0)
                            if canonical_condition.startswith("native_")
                            else split_count - 1
                        ),
                        "condition": canonical_condition,
                        "transport_mode": "native_kv"
                        if canonical_condition.startswith("native_")
                        else "self_attention",
                        "routing_mode": canonical_condition.removeprefix("native_"),
                        "gist_mode": converted.cfg.gist_mode
                        if canonical_condition.startswith("native_")
                        else None,
                        "top_k_or_threshold": (
                            "all"
                            if canonical_condition == "native_all"
                            else "oracle"
                            if canonical_condition == "native_oracle"
                            else converted.cfg.top_k_references
                            if canonical_condition == "native_routed"
                            else None
                        ),
                    }
                )
        for metric_condition in ("all", "oracle", "routed", "shuffled"):
            values = condition_rows[f"native_{metric_condition}"]
            row[f"active_fraction_{metric_condition}"] = statistics.fmean(
                float(value["active_fraction"]) for value in values
            )
            row[f"active_tokens_{metric_condition}"] = statistics.fmean(
                float(value["active_tokens"]) for value in values
            )
        print(
            f"split={split_count:>2} full={row['sa_full_loss']:.4f} "
            f"tail={row['sa_tail_loss']:.4f} all={row['native_all_loss']:.4f} "
            f"oracle={row['native_oracle_loss']:.4f} routed={row['native_routed_loss']:.4f} "
            f"shuffled={row['native_shuffled_loss']:.4f}",
            flush=True,
        )
        rows.append(row)
    by_key = {
        (int(value["split_count"]), str(value["example_id"]), str(value["condition"])): value
        for value in raw_rows
    }
    example_ids = sorted({str(value["example_id"]) for value in raw_rows})
    for split_count in NATIVE_KV_SPLIT_COUNTS:
        for example_id in example_ids:
            available = {
                condition: by_key.get((split_count, example_id, condition))
                for condition in (
                    "sa_full",
                    "sa_tail",
                    "native_all",
                    "native_oracle",
                    "native_routed",
                    "native_shuffled",
                    "native_disabled",
                )
            }
            if all(available.values()):
                metrics = derive_native_kv_metrics(
                    {condition: value["loss"] for condition, value in available.items()}
                )
                for condition, value in available.items():
                    value.update(metrics)
    return rows, raw_rows


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
                "median": statistics.median(values),
                "ci95_low": statistics.fmean(values)
                - 1.96 * statistics.pstdev(values) / math.sqrt(len(values)),
                "ci95_high": statistics.fmean(values)
                + 1.96 * statistics.pstdev(values) / math.sqrt(len(values)),
                "values": values,
            }
        rows.append(aggregate)
    return rows


def _plot_report(
    dataset_name: str, aggregate: list[dict[str, Any]], output: Path, *, seed_count: int
) -> None:
    splits = [row["split_count"] for row in aggregate]
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.1))
    losses = (
        ("SA full", "sa_full_loss", "#245A8D", "o"),
        ("SA tail", "sa_tail_loss", "#5F6B76", "o"),
        ("Native all", "native_all_loss", "#2D7A68", "s"),
        ("Native oracle", "native_oracle_loss", "#A66A18", "D"),
        ("Native routed", "native_routed_loss", "#8B5A9F", "^"),
        ("Native shuffled", "native_shuffled_loss", "#A33D3D", "x"),
    )
    for label, key, color, marker in losses:
        axes[0, 0].errorbar(
            splits,
            [row[key]["mean"] for row in aggregate],
            yerr=[row[key]["stddev"] for row in aggregate],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.2,
            capsize=2,
        )
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(splits, [str(value) for value in splits])
    axes[0, 0].set_xlabel("Total split count")
    axes[0, 0].set_ylabel("Answer-token loss")
    axes[0, 0].set_title("A. Quality by source partition")
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)

    gaps = (
        ("Transport", "transport_gap", "#A33D3D"),
        ("Sparsification", "sparse_gap", "#A66A18"),
        ("Routing", "routing_gap", "#8B5A9F"),
    )
    for label, key, color in gaps:
        axes[1, 1].errorbar(
            splits,
            [row[key]["mean"] for row in aggregate],
            yerr=[row[key]["stddev"] for row in aggregate],
            label=label,
            color=color,
            marker="o",
            linewidth=1.2,
            capsize=2,
        )
    axes[1, 1].axhline(0.0, color="#30363D", linewidth=0.8)
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks(splits, [str(value) for value in splits])
    axes[1, 1].set_xlabel("Total split count")
    axes[1, 1].set_ylabel("Loss difference")
    axes[1, 1].set_title("D. Error decomposition")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for label, key, color in (
        ("All", "rcb_all", "#2D7A68"),
        ("Oracle", "rcb_oracle", "#A66A18"),
        ("Routed", "rcb_routed", "#8B5A9F"),
        ("Shuffled", "rcb_shuffled", "#A33D3D"),
    ):
        axes[0, 1].errorbar(
            splits,
            [row[key]["mean"] for row in aggregate],
            yerr=[row[key]["stddev"] for row in aggregate],
            label=label,
            color=color,
            marker="o",
            linewidth=1.2,
            capsize=2,
        )
    axes[0, 1].axhline(1.0, color="#30363D", linewidth=0.8)
    axes[0, 1].axhline(0.0, color="#8C959F", linewidth=0.7)
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xticks(splits, [str(value) for value in splits])
    axes[0, 1].set_xlabel("Total split count")
    axes[0, 1].set_ylabel("Recovered context benefit")
    axes[0, 1].set_title("B. Context benefit recovery")
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)

    for label, fraction_key, loss_key, color, marker in (
        ("All", "active_fraction_all", "native_all_loss", "#2D7A68", "s"),
        ("Oracle", "active_fraction_oracle", "native_oracle_loss", "#A66A18", "D"),
        ("Routed", "active_fraction_routed", "native_routed_loss", "#8B5A9F", "^"),
    ):
        axes[1, 0].plot(
            [row[fraction_key]["mean"] for row in aggregate],
            [row[loss_key]["mean"] - row["sa_full_loss"]["mean"] for row in aggregate],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.2,
        )
    axes[1, 0].axhline(0.0, color="#30363D", linewidth=0.8)
    axes[1, 0].set_xlabel("Active token K/V fraction")
    axes[1, 0].set_ylabel("Loss minus full SA")
    axes[1, 0].set_title("C. Quality-active-context frontier")
    axes[1, 0].legend(frameon=False, fontsize=7)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D8DEE4", linewidth=0.7)
        axis.set_axisbelow(True)
    seed_label = "five paired seeds" if seed_count == 5 else f"{seed_count} seed(s)"
    figure.suptitle(f"{dataset_name} native-KV benchmark ({seed_label})", fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", metadata={"Creator": Path(__file__).name})
    plt.close(figure)


def _plot_efficiency_report(
    dataset_name: str,
    aggregate: list[dict[str, Any]],
    dependency_strata: list[dict[str, Any]],
    output: Path,
    *,
    seed_count: int,
) -> None:
    """Plot RCB/active-KV scaling and the dependency-sensitive comparison."""

    splits = [row["split_count"] for row in aggregate]
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.1))
    series = (
        ("Oracle", "oracle", "#A66A18", "D"),
        ("Routed", "routed", "#8B5A9F", "^"),
        ("Shuffled", "shuffled", "#A33D3D", "x"),
    )
    for label, key, color, marker in series:
        axes[0, 0].plot(
            [row[f"active_fraction_{key}"]["mean"] for row in aggregate],
            [row[f"rcb_{key}"]["mean"] for row in aggregate],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.2,
        )
    axes[0, 0].axhline(1.0, color="#30363D", linewidth=0.8)
    axes[0, 0].set_xlabel("Active token K/V fraction")
    axes[0, 0].set_ylabel("Recovered context benefit")
    axes[0, 0].set_title("E. Benefit-active-context frontier")
    axes[0, 0].legend(frameon=False, fontsize=7)

    for label, key, color, marker in (
        ("All", "all", "#2D7A68", "s"),
        *series[:2],
    ):
        axes[0, 1].plot(
            splits,
            [row[f"active_tokens_{key}"]["mean"] for row in aggregate],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.2,
        )
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xticks(splits, [str(value) for value in splits])
    axes[0, 1].set_xlabel("Total split count")
    axes[0, 1].set_ylabel("Active token K/V")
    axes[0, 1].set_title("F. Active K/V by partition")
    axes[0, 1].legend(frameon=False, fontsize=7)

    for label, key, color, marker in (
        ("All", "all", "#2D7A68", "s"),
        *series[:2],
    ):
        axes[1, 0].plot(
            splits,
            [row[f"active_fraction_{key}"]["mean"] for row in aggregate],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.2,
        )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(splits, [str(value) for value in splits])
    axes[1, 0].set_xlabel("Total split count")
    axes[1, 0].set_ylabel("Active token K/V fraction")
    axes[1, 0].set_title("G. Active fraction by partition")
    axes[1, 0].legend(frameon=False, fontsize=7)

    conditions = ("oracle", "routed", "shuffled")
    overall = [
        statistics.fmean(row[f"rcb_{condition}"]["mean"] for row in aggregate)
        for condition in conditions
    ]
    high_by_condition = {
        row["condition"].removeprefix("native_"): row["rcb_mean"]
        for row in dependency_strata
        if row["dependency_stratum"] == "high"
    }
    high = [high_by_condition.get(condition, float("nan")) for condition in conditions]
    positions = list(range(len(conditions)))
    axes[1, 1].bar([value - 0.18 for value in positions], overall, 0.36, label="All")
    axes[1, 1].bar([value + 0.18 for value in positions], high, 0.36, label="High dependency")
    axes[1, 1].axhline(1.0, color="#30363D", linewidth=0.8)
    axes[1, 1].set_xticks(positions, [value.title() for value in conditions])
    axes[1, 1].set_ylabel("Recovered context benefit")
    axes[1, 1].set_title("H. Dependency-sensitive RCB")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D8DEE4", linewidth=0.7)
        axis.set_axisbelow(True)
    seed_label = "five paired seeds" if seed_count == 5 else f"{seed_count} seed(s)"
    figure.suptitle(
        f"{dataset_name} native-KV efficiency diagnostics ({seed_label})", fontsize=11
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", metadata={"Creator": Path(__file__).name})
    plt.close(figure)


def _write_report(dataset_name: str, seed_payloads: list[dict[str, Any]], publish: bool) -> Path:
    aggregate = _aggregate(seed_payloads)
    raw_rows = [row for payload in seed_payloads for row in payload.get("raw_results", [])]
    for row in raw_rows:
        if row.get("condition") in {"sa_full", "sa_tail"}:
            row["num_references"] = int(row["split_count"]) - 1
            row["num_chunks"] = int(row["split_count"]) - 1
    dependency_values = finite_values(
        row.get("dependency_gain")
        for row in raw_rows
        if row.get("condition") == "sa_full" and row.get("split_count") == 2
    )
    dependency_thresholds = None
    if len(dependency_values) >= 3:
        low_cut, high_cut = statistics.quantiles(dependency_values, n=3, method="inclusive")
        dependency_thresholds = {"low_medium": low_cut, "medium_high": high_cut}
        for row in raw_rows:
            gain = row.get("dependency_gain")
            if gain is None:
                row["dependency_stratum"] = "undefined"
            elif float(gain) <= low_cut:
                row["dependency_stratum"] = "low"
            elif float(gain) <= high_cut:
                row["dependency_stratum"] = "medium"
            else:
                row["dependency_stratum"] = "high"
    dependency_strata = []
    for stratum in ("low", "medium", "high"):
        for condition in ("native_oracle", "native_routed", "native_shuffled"):
            matching = [
                row
                for row in raw_rows
                if row.get("dependency_stratum") == stratum
                and row.get("condition") == condition
            ]
            values = finite_values(row.get(f"rcb_{condition.removeprefix('native_')}") for row in matching)
            if values:
                dependency_strata.append(
                    {
                        "dependency_stratum": stratum,
                        "condition": condition,
                        "rows": len(matching),
                        "examples": len({row["example_id"] for row in matching}),
                        "rcb_mean": statistics.fmean(values),
                        "rcb_median": statistics.median(values),
                    }
                )
    report_dir = REPO / "out" / "reports" / f"native_kv_{dataset_name}_5seed"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "seeds": [payload["seed"] for payload in seed_payloads],
        "split_counts": list(NATIVE_KV_SPLIT_COUNTS),
        "transport": "native_kv",
        "training": "full-context SelfAttention followed by weight-preserving PRA conversion",
        "prediction_target": "first answer token",
        "seed_results": [
            {key: value for key, value in seed_payload.items() if key != "raw_results"}
            for seed_payload in seed_payloads
        ],
        "aggregate": aggregate,
        "raw_results": raw_rows,
        "dependency_thresholds": dependency_thresholds,
        "dependency_strata": dependency_strata,
    }
    report_json = report_dir / "report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if raw_rows:
        fieldnames = sorted({key for row in raw_rows for key in row if not isinstance(row[key], (dict, list))})
        with (report_dir / "raw_runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(raw_rows)
    aggregate_rows = []
    for row in aggregate:
        flat = {"split_count": row["split_count"], "reference_count": row["reference_count"]}
        for key, value in row.items():
            if isinstance(value, dict) and "mean" in value:
                for statistic in ("mean", "stddev", "median", "ci95_low", "ci95_high"):
                    flat[f"{key}_{statistic}"] = value[statistic]
        aggregate_rows.append(flat)
    if aggregate_rows:
        with (report_dir / "aggregate_by_split.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)
    figure_path = report_dir / "native_kv_split_scaling.pdf"
    _plot_report(dataset_name, aggregate, figure_path, seed_count=len(seed_payloads))
    efficiency_path = report_dir / "native_kv_efficiency.pdf"
    _plot_efficiency_report(
        dataset_name,
        aggregate,
        dependency_strata,
        efficiency_path,
        seed_count=len(seed_payloads),
    )
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
        f"<p>Seeds: {escape(str(payload['seeds']))}. Target: first answer token. Lower loss is better.</p>"
        "<table><thead><tr><th>Splits</th><th>SA full</th><th>SA tail</th><th>Native all</th>"
        f"<th>Native oracle</th><th>Native shuffled</th></tr></thead><tbody>{html_rows}</tbody></table>"
        "<h2>Split scaling</h2><embed src='native_kv_split_scaling.pdf' type='application/pdf'>"
        "<h2>Efficiency diagnostics</h2><embed src='native_kv_efficiency.pdf' type='application/pdf'>"
        "<p><a href='report.json'>Structured results</a> | <a href='raw_runs.csv'>Raw rows</a> | "
        "<a href='aggregate_by_split.csv'>Aggregate CSV</a></p></body></html>",
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
        _plot_report(
            dataset_name,
            aggregate,
            figure_dir / f"native_kv_{dataset_name}.pdf",
            seed_count=len(seed_payloads),
        )
        _plot_efficiency_report(
            dataset_name,
            aggregate,
            dependency_strata,
            figure_dir / f"native_kv_{dataset_name}_efficiency.pdf",
            seed_count=len(seed_payloads),
        )
    return report_dir / "index.html"


def run(args: argparse.Namespace) -> Path:
    defaults = dict(DATASET_DEFAULTS[args.dataset])
    for key in ("max_examples", "steps", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            defaults[key] = value
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, training_module, modules = DATASET_PREPARERS[args.dataset](defaults)
    _assert_fixed_target_invariants(modules)
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
            and existing_payload.get("generation_version") == defaults["generation_version"]
            and existing_payload.get("settings") == defaults
            and existing_payload.get("raw_results")
            and not args.force
        )
        if can_reuse_result:
            payload = existing_payload
        else:
            results, raw_results = evaluate_seed(
                source=source,
                tokenizer=tokenizer,
                modules=modules,
                settings=defaults,
                device=device,
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            experiment_id = (
                f"native-kv-{args.dataset}-seed-{seed}-steps-{defaults['steps']}"
            )
            checkpoint_id = f"{args.dataset}/seed-{seed}/step-{defaults['steps']}"
            for row in raw_results:
                row.update(
                    {
                        "experiment_id": experiment_id,
                        "timestamp": timestamp,
                        "checkpoint_id": checkpoint_id,
                        "model_name": "td_sa_converted_native_kv"
                        if str(row["condition"]).startswith("native_")
                        else "td_sa",
                        "seed": seed,
                        "dataset": args.dataset,
                    }
                )
            payload = {
                "dataset": args.dataset,
                "generation_version": defaults["generation_version"],
                "seed": seed,
                "device": device,
                "settings": defaults,
                "training": training,
                "results": results,
                "raw_results": raw_results,
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
