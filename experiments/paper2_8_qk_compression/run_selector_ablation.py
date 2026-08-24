"""Run matched combined-loss salience/bilinear selector ablations for Paper 2.8."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_gated_study import SEEDS, _sha256, _write_csv
from experiments.paper2_8_qk_compression.run_query_conditioned_study import (
    _aggregate,
    _evaluate_selector,
    _fit_selector,
    _restore_test_cases,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata


RESULTS = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
RANKS = (8, 16, 32)
M_VALUES = (4, 8)
OBJECTIVE = "combined"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _configurations(ranks: tuple[int, ...]) -> list[tuple[str, int, bool, bool]]:
    return [("salience_only", 8, True, False)] + [
        ("bilinear_only", rank, False, True) for rank in ranks
    ]


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = RESULTS / "query_conditioned/prepared_cache.pt"
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
    test_features = torch.load(args.test_features, map_location="cpu", weights_only=False)
    training_batch = {
        key: value.to(device) for key, value in prepared["training_batch"].items()
    }
    test_cases = _restore_test_cases(prepared["test_auxiliary"], test_features)
    row_path = args.output_dir / "per_example.csv"
    history_path = args.output_dir / "training_history.csv"
    if args.overwrite:
        row_path.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
    rows = _read(row_path)
    history = _read(history_path)
    completed = {
        (row["ablation"], int(row["rank"]), int(row["m"]), int(row["seed"]))
        for row in rows
    }
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for ablation, model_rank, use_salience, use_interaction in _configurations(args.ranks):
        reported_rank = 0 if ablation == "salience_only" else model_rank
        for m in M_VALUES:
            for seed in args.seeds:
                configuration = (ablation, reported_rank, m, seed)
                if configuration in completed:
                    print(f"[resume] {configuration}", flush=True)
                    continue
                print(f"[train] {configuration}", flush=True)
                selector, run_history, train_seconds = _fit_selector(
                    training_batch,
                    objective=OBJECTIVE,
                    rank=model_rank,
                    m=m,
                    seed=seed,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    device=device,
                    use_salience=use_salience,
                    use_interaction=use_interaction,
                )
                for record in run_history:
                    record.update(
                        {
                            "ablation": ablation,
                            "rank": reported_rank,
                            "m": m,
                            "seed": seed,
                            "train_seconds": train_seconds,
                        }
                    )
                run_rows = _evaluate_selector(
                    selector,
                    test_cases,
                    objective=OBJECTIVE,
                    rank=reported_rank,
                    m=m,
                    seed=seed,
                    function="top_r_mean",
                    head_reduction="mean",
                    device=device,
                )
                for record in run_rows:
                    record["ablation"] = ablation
                    record["train_seconds"] = train_seconds
                rows.extend(run_rows)
                history.extend(run_history)
                _write_csv(row_path, rows)
                _write_csv(history_path, history)
                if seed == args.seeds[0]:
                    torch.save(
                        {
                            "state_dict": {
                                name: value.detach().cpu()
                                for name, value in selector.state_dict().items()
                            },
                            "ablation": ablation,
                            "rank": reported_rank,
                            "model_rank": model_rank,
                            "m": m,
                            "seed": seed,
                            "objective": OBJECTIVE,
                            "steps": args.steps,
                            "parameter_count": sum(
                                parameter.numel() for parameter in selector.parameters()
                            ),
                        },
                        checkpoint_dir
                        / f"{ablation}_r{reported_rank}_m{m}_seed{seed}.pt",
                    )
                del selector
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    numeric = []
    for row in rows:
        numeric.append(
            {
                key: (
                    int(value)
                    if key in {"rank", "m", "seed", "parameter_count"}
                    else float(value)
                    if key
                    in {
                        "evidence_recall",
                        "evidence_precision",
                        "any_evidence",
                        "chain_completion",
                        "mrr",
                        "teacher_top4_overlap",
                        "spearman",
                        "kl",
                        "materialized_kv_tokens",
                        "active_memory_fraction",
                        "native_dots",
                        "selection_ms",
                    }
                    else value
                )
                for key, value in row.items()
            }
        )
    grouped = {}
    for ablation in {row["ablation"] for row in numeric}:
        ablation_rows = [row for row in numeric if row["ablation"] == ablation]
        for record in _aggregate(ablation_rows):
            record["ablation"] = ablation
            grouped[(record["dataset"], ablation, record["rank"], record["m"])] = record
    _write_csv(args.output_dir / "summary.csv", list(grouped.values()))
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "objective": OBJECTIVE,
        "ablations": [configuration[0] for configuration in _configurations(args.ranks)],
        "ranks": list(args.ranks),
        "m_values": list(M_VALUES),
        "seeds": list(args.seeds),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "backbone_frozen": True,
        "test_used_for_model_selection": False,
        "prepared_cache_sha256": _sha256(prepared_path),
        "test_feature_sha256": _sha256(args.test_features),
        "command": "python experiments/paper2_8_qk_compression/run_selector_ablation.py --device cuda",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"rows": len(rows), "summary_rows": len(grouped)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--ranks", type=lambda value: tuple(map(int, value.split(","))), default=RANKS
    )
    parser.add_argument(
        "--seeds", type=lambda value: tuple(map(int, value.split(","))), default=SEEDS
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS / "selector_ablation"
    )
    parser.add_argument(
        "--test-features", type=Path, default=RESULTS / "native_qk_features_test.pt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
