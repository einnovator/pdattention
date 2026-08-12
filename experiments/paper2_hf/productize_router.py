"""Train and package one five-seed PRA-HF routing adapter release candidate."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from transformers import __version__ as transformers_version

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pra_hf import evaluate_router_features
from pra_hf.training import load_feature_rows, train_router


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _domain(rows: list[dict], dataset: str) -> list[dict]:
    return rows if dataset == "combined" else [row for row in rows if row["dataset"] == dataset]


def _aggregate(runs: list[dict]) -> dict:
    output = {}
    for dataset in ("qasper", "hotpotqa", "combined"):
        keys = tuple(runs[0]["test"][dataset]["summary"])
        output[dataset] = {}
        for key in keys:
            values = [run["test"][dataset]["summary"][key] for run in runs]
            numeric = [float(value) for value in values if value is not None]
            output[dataset][key] = {
                "mean": statistics.fmean(numeric) if numeric else None,
                "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
            }
    return output


def run(args) -> dict:
    train = load_feature_rows([args.feature_dir / "router_features_train.pt"])
    validation = load_feature_rows([args.feature_dir / "router_features_validation.pt"])
    test = load_feature_rows([args.feature_dir / "router_features_test.pt"])
    train_rows = _domain(train, args.train_domain)
    validation_rows = _domain(validation, args.train_domain)
    git_sha = _git_sha()
    runs = []
    candidates = []
    for seed in args.seeds:
        metadata = {
            "base_model": args.base_model,
            "base_model_revision": args.base_model_revision,
            "pra_version": "0.2.0rc1",
            "model_family": args.model_family,
            "routing_representation": "attention_input_hidden_state",
            "routing_layer": args.routing_layer,
            "chunk_tokens": args.chunk_tokens,
            "training_datasets": [args.train_domain.upper()],
            "dataset_version": "QASPER v0.3 validation",
            "feature_split_seed": args.feature_split_seed,
            "git_sha": git_sha,
            "training_hardware": (
                torch.cuda.get_device_name(args.device)
                if str(args.device).startswith("cuda")
                else str(args.device)
            ),
            "training_learning_rate": args.learning_rate,
            "training_objective": "multi-positive softmax",
            "training_torch": torch.__version__,
            "training_transformers": transformers_version,
            "license_note": "Router weights only; base-model and dataset licenses apply separately.",
        }
        router, training = train_router(
            train_rows,
            validation_rows,
            routing_width=args.routing_dim,
            query_strategy="last",
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=seed,
            device=args.device,
            metadata=metadata,
        )
        reports = {
            dataset: evaluate_router_features(
                router,
                _domain(test, dataset),
                query_strategy="last",
                device=args.device,
            )
            for dataset in ("qasper", "hotpotqa", "combined")
        }
        record = {
            "seed": seed,
            "training": training,
            "test": reports,
        }
        runs.append(record)
        candidates.append((training["validation"]["auc_0_30"], seed, router, record))
        print(
            f"seed={seed} validation_auc={training['validation']['auc_0_30']:.4f} "
            f"qasper_R20={reports['qasper']['summary']['R@20%']:.4f}",
            flush=True,
        )
    _, selected_seed, selected_router, selected_run = max(
        candidates, key=lambda row: (row[0], -row[1])
    )
    selected_router.metadata.update(
        {
            "selection_rule": "maximum QASPER validation AUC0-30; seed breaks ties ascending",
            "selected_seed": selected_seed,
            "training_seconds": selected_run["training"]["training_seconds"],
            "metrics": selected_run["test"]["qasper"]["summary"],
            "transfer_metrics": selected_run["test"]["hotpotqa"]["summary"],
        }
    )
    selected_router.save_pretrained(args.output_router)
    summary = {
        "protocol": "five-seed frozen-backbone asymmetric-linear router productization",
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "model_family": args.model_family,
        "git_sha": git_sha,
        "train_domain": args.train_domain,
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "test_examples": len(test),
        "routing_dim": args.routing_dim,
        "routing_layer": args.routing_layer,
        "chunk_tokens": args.chunk_tokens,
        "steps": args.steps,
        "seeds": list(args.seeds),
        "selected_seed": selected_seed,
        "router_directory": str(args.output_router.resolve().relative_to(ROOT)),
        "aggregates": _aggregate(runs),
        "selected_test": selected_run["test"],
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-router", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--model-family", choices=("qwen", "llama", "gemma3"), required=True)
    parser.add_argument("--routing-layer", type=int, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--routing-dim", type=int, default=128)
    parser.add_argument("--train-domain", choices=("qasper", "hotpotqa", "combined"), default="qasper")
    parser.add_argument("--feature-split-seed", type=int, default=20260811)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
