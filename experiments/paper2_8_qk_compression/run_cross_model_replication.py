"""Train and confirm Paper 2.8 low-rank routing on a Llama-family model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_8_qk_compression.run_confirmation import (
    CONFIRMATION_OFFSET,
    CONFIRMATION_PER_DATASET,
    ROUTING_BUDGET,
    _aggregate,
    _lexical_scores,
    _oracle_indices,
    _paired,
    _plot,
    _row,
)
from experiments.paper2_8_qk_compression.run_gated_study import (
    SEEDS,
    _case_tensors,
    _score_compact,
    _sha256,
    _write_csv,
)
from experiments.paper2_8_qk_compression.run_low_rank_frontier import (
    _fit_direct_router,
)
from experiments.paper2_8_qk_compression.run_query_conditioned_study import (
    _query_feature,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_hf.qk_compression import (
    QueryConditionedLandmarkSelector,
    kmeans_centroids,
    low_rank_response_scores,
    masked_mean_keys,
)


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
RESULT_ROOT = ROOT / "docs/papers/shared/results/paper2_8_qk_compression/cross_model_smollm2"


def _key_scale(features: list[dict]) -> float:
    square_sum, count = 0.0, 0
    for feature in features:
        keys = feature["local_pre_key"].float().flatten(2)
        valid = keys[feature["local_token_mask"]]
        square_sum += float(valid.square().sum())
        count += valid.numel()
    return math.sqrt(square_sum / max(count, 1))


def _training_cases(features: list[dict], scale: float) -> list[dict]:
    cases = []
    for feature in features:
        query, keys, mask, positives = _case_tensors(feature, torch.device("cpu"))
        teacher = _score_compact(
            query, keys, mask, function="top_r_mean", head_reduction="mean"
        )
        cases.append(
            {
                "keys": feature["local_pre_key"].float().flatten(2) / scale,
                "query": _query_feature(query[0]),
                "token_mask": mask,
                "positives": positives,
                "teacher": teacher,
            }
        )
    return cases


def _load_or_train(
    cases: list[dict], args: argparse.Namespace, scale: float, device: torch.device
) -> dict[int, list[tuple[QueryConditionedLandmarkSelector, dict, int]]]:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output: dict[int, list[tuple[QueryConditionedLandmarkSelector, dict, int]]] = {}
    for rank in (8, 16):
        output[rank] = []
        for seed in args.seeds:
            path = args.checkpoint_dir / f"smollm2_direct_lowrank_r{rank}_seed{seed}.pt"
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                selector = QueryConditionedLandmarkSelector(
                    cases[0]["query"].numel(),
                    feature_width=cases[0]["keys"].shape[-1],
                    rank=rank,
                    use_salience=False,
                    use_interaction=True,
                ).to(device)
                selector.load_state_dict(checkpoint["state_dict"])
                selector.eval()
            else:
                selector, history, seconds = _fit_direct_router(
                    cases,
                    rank=rank,
                    seed=seed,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    device=device,
                )
                checkpoint = {
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "rank": rank,
                    "seed": seed,
                    "native_key_rms_scale": scale,
                    "query_width": cases[0]["query"].numel(),
                    "key_width": cases[0]["keys"].shape[-1],
                    "state_dict": {k: v.cpu() for k, v in selector.state_dict().items()},
                    "history": history,
                    "train_seconds": seconds,
                }
                torch.save(checkpoint, path)
            output[rank].append((selector, checkpoint, seed))
    return output


@torch.no_grad()
def _scores(selector, checkpoint, feature, *, static: bool, device) -> torch.Tensor:
    keys = feature["local_pre_key"].to(device).float().flatten(2)
    mask = feature["local_token_mask"].to(device)
    projected = selector.feature_projection(
        keys / float(checkpoint["native_key_rms_scale"])
    )
    if static:
        projected, mask = kmeans_centroids(projected, mask, 8)
    query = selector.query_projection(
        _query_feature(feature["query_pre_query"]).to(device)
    )
    return low_rank_response_scores(query, projected, mask, top_r=4)[0].cpu()


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    validation = torch.load(args.validation_features, map_location="cpu", weights_only=False)
    confirmation = torch.load(args.confirmation_features, map_location="cpu", weights_only=False)
    if any("query_pre_query" not in row for row in validation + confirmation):
        raise ValueError("Cross-model features must include native query_pre_query tensors.")
    if {(r["dataset"], r["example_id"]) for r in validation} & {
        (r["dataset"], r["example_id"]) for r in confirmation
    }:
        raise RuntimeError("Validation and confirmation identities overlap.")
    scale = _key_scale(validation)
    selectors = _load_or_train(_training_cases(validation, scale), args, scale, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, local_files_only=True
    )
    examples = load_split_examples(
        args.cache_dir, args.examples_per_dataset, args.offset, args.dataset_seed
    )
    examples_by_id = {(row["dataset"], row["id"]): row for row in examples}
    rows = []
    for index, feature in enumerate(confirmation, start=1):
        query, keys, mask, _ = _case_tensors(feature, device)
        teacher = _score_compact(
            query, keys, mask, function="top_r_mean", head_reduction="mean"
        ).cpu()
        mean_keys = masked_mean_keys(keys, mask)
        mean_scores = _score_compact(
            query,
            mean_keys,
            torch.ones((len(mean_keys), 1), dtype=torch.bool, device=device),
            function="top_r_mean",
            head_reduction="mean",
        ).cpu()
        native_width = int(keys.shape[2] * keys.shape[3])
        rows.extend(
            (
                _row(feature, condition="full_k_teacher", seed=-1, scores=teacher,
                     index_bytes_per_chunk=32 * native_width * 4),
                _row(feature, condition="native_mean", seed=-1, scores=mean_scores,
                     index_bytes_per_chunk=native_width * 4),
            )
        )
        example = examples_by_id[(feature["dataset"], feature["example_id"])]
        for name, lexical in _lexical_scores(tokenizer, example, feature, mean_scores).items():
            rows.append(_row(feature, condition=name, seed=-1, scores=lexical))
        by_condition: dict[str, list[torch.Tensor]] = defaultdict(list)
        for rank, static, name, byte_count in (
            (16, False, "lowrank_r16", 32 * 16 * 4),
            (8, True, "lowrank_r8_kmeans", 8 * 8 * 4),
        ):
            for selector, checkpoint, seed in selectors[rank]:
                score = _scores(selector, checkpoint, feature, static=static, device=device)
                by_condition[name].append(score)
                rows.append(_row(feature, condition=name, seed=seed, scores=score,
                                 index_bytes_per_chunk=byte_count))
            rows.append(
                _row(
                    feature,
                    condition=f"{name}_ensemble",
                    seed=-1,
                    scores=torch.stack(by_condition[name]).mean(0),
                    index_bytes_per_chunk=byte_count,
                )
            )
        rows.append(
            _row(
                feature,
                condition="oracle_evidence",
                seed=-1,
                scores=teacher,
                selected=_oracle_indices(feature["local_positive_mask"], teacher, ROUTING_BUDGET),
            )
        )
        print(f"[SmolLM confirmation {index}/{len(confirmation)}] {feature['dataset']} {feature['example_id']}", flush=True)
    summary, paired = _aggregate(rows), _paired(rows, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_example.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "paired_effects.csv", paired)
    _plot(summary, args.output_dir)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_family": "llama",
        "tokenizer_replication": True,
        "validation_identities": len(validation),
        "confirmation_identities": len(confirmation),
        "seeds": list(args.seeds),
        "routing_budget_chunks": ROUTING_BUDGET,
        "native_key_rms_scale": scale,
        "feature_hashes": {
            path.name: _sha256(path)
            for path in (args.validation_features, args.confirmation_features)
        },
        "checkpoint_hashes": {
            path.name: _sha256(path) for path in sorted(args.checkpoint_dir.glob("*.pt"))
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offset", type=int, default=CONFIRMATION_OFFSET)
    parser.add_argument("--examples-per-dataset", type=int, default=CONFIRMATION_PER_DATASET)
    parser.add_argument("--dataset-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seeds", type=lambda value: tuple(map(int, value.split(","))), default=SEEDS)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=RESULT_ROOT / "checkpoints")
    parser.add_argument("--validation-features", type=Path, default=RESULT_ROOT / "native_qk_features_validation.pt")
    parser.add_argument("--confirmation-features", type=Path, default=RESULT_ROOT / "native_qk_features_confirmation.pt")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
