"""Train five routing seeds and compare them with parameter-free PRA routing."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pra_hf.evaluation import evaluate_router_features
from pra_hf.training import load_feature_rows, train_router


class GenericCosineRouter:
    """The no-adapter PRA baseline: normalized model-state cosine similarity."""

    def to(self, _device):
        return self

    def eval(self):
        return self

    @staticmethod
    def scores(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return F.normalize(query.float(), dim=-1) @ F.normalize(
            memory.float(), dim=-1
        ).transpose(0, 1)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _domain(rows: list[dict], dataset: str) -> list[dict]:
    return rows if dataset == "combined" else [row for row in rows if row["dataset"] == dataset]


def _reports(router, rows: list[dict], device: str) -> dict[str, Any]:
    return {
        dataset: evaluate_router_features(
            router,
            _domain(rows, dataset),
            query_strategy="last",
            device=device,
        )
        for dataset in ("qasper", "hotpotqa", "combined")
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in ("qasper", "hotpotqa", "combined"):
        output[dataset] = {}
        for metric in runs[0]["test"][dataset]["summary"]:
            values = [
                float(run["test"][dataset]["summary"][metric])
                for run in runs
                if run["test"][dataset]["summary"][metric] is not None
            ]
            output[dataset][metric] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    train = load_feature_rows([args.feature_dir / "router_features_train.pt"])
    validation = load_feature_rows([args.feature_dir / "router_features_validation.pt"])
    test = load_feature_rows([args.feature_dir / "router_features_test.pt"])
    manifest = json.loads(
        (args.feature_dir / "feature_dataset_manifest.json").read_text(encoding="utf-8")
    )
    baseline = _reports(GenericCosineRouter(), test, "cpu")
    candidates = []
    runs = []
    for seed in args.seeds:
        metadata = {
            "base_model": args.base_model,
            "base_model_revision": args.base_model_revision,
            "pra_version": "0.2.0rc1",
            "model_family": args.model_family,
            "routing_representation": manifest["feature_source"],
            "routing_layer": manifest["routing_layer"],
            "chunk_tokens": manifest["routing_chunk_tokens"],
            "encoding_block_tokens": manifest["encoding_block_tokens"],
            "training_datasets": ["QASPER", "HOTPOTQA"],
            "dataset_version": "QASPER v0.3 and HotpotQA distractor validation",
            "feature_split_seed": manifest["seed"],
            "git_sha": _git_sha(),
            "training_hardware": manifest["runtime"]["hardware"],
            "training_learning_rate": args.learning_rate,
            "training_objective": "multi-positive softmax",
            "training_torch": torch.__version__,
            "feature_engine": f"mlx-lm {manifest['runtime']['mlx_lm']}",
            "license_note": "Router weights only; base-model and dataset licenses apply separately.",
        }
        router, training = train_router(
            train,
            validation,
            routing_width=args.routing_dim,
            query_strategy="last",
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=seed,
            device=args.device,
            metadata=metadata,
        )
        reports = _reports(router, test, args.device)
        record = {"seed": seed, "training": training, "test": reports}
        runs.append(record)
        candidates.append(
            (training["validation"]["auc_0_30"], -seed, router, record)
        )
        print(
            f"seed={seed} validation_auc={training['validation']['auc_0_30']:.4f} "
            f"combined_R20={reports['combined']['summary']['R@20%']:.4f}",
            flush=True,
        )
    _, _, selected_router, selected = max(candidates, key=lambda row: (row[0], row[1]))
    selected_seed = int(selected["seed"])
    selected_router.metadata.update(
        {
            "selection_rule": "maximum combined validation AUC0-30; seed breaks ties ascending",
            "selected_seed": selected_seed,
            "training_seconds": selected["training"]["training_seconds"],
            "metrics": selected["test"]["combined"]["summary"],
            "dataset_metrics": {
                dataset: selected["test"][dataset]["summary"]
                for dataset in ("qasper", "hotpotqa")
            },
            "generic_baseline_metrics": {
                dataset: baseline[dataset]["summary"]
                for dataset in ("qasper", "hotpotqa", "combined")
            },
        }
    )
    args.output_router.mkdir(parents=True, exist_ok=True)
    selected_router.save_pretrained(args.output_router)
    deltas = {
        dataset: {
            metric: (
                selected["test"][dataset]["summary"][metric]
                - baseline[dataset]["summary"][metric]
            )
            for metric in ("R@10%", "R@20%", "R@3", "R@8", "MRR", "AUC0-30")
        }
        for dataset in ("qasper", "hotpotqa", "combined")
    }
    summary = {
        "protocol": "matched generic-cosine versus five-seed learned PRA routing",
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "model_family": args.model_family,
        "feature_manifest": str((args.feature_dir / "feature_dataset_manifest.json").resolve()),
        "training_examples": len(train),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "routing_dim": args.routing_dim,
        "steps": args.steps,
        "seeds": list(args.seeds),
        "selected_seed": selected_seed,
        "adapter_parameters": selected["training"]["adapter_parameters"],
        "generic_baseline": baseline,
        "selected_learned": selected["test"],
        "selected_minus_generic": deltas,
        "five_seed_aggregate": _aggregate(runs),
        "runs": runs,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_sha": _git_sha(),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-router", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--model-family", choices=("qwen", "llama", "gemma3"), required=True)
    parser.add_argument("--routing-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
