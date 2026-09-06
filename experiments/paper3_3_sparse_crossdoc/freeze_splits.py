"""Freeze disjoint Paper 3.3 MultiHop-RAG identities.

The Paper 3.2 final adapter cohort is treated as an external legacy holdout. It
is never assigned to Paper 3.3 train, validation, or test data, which keeps the
new oracle and learned-policy study independent of the inherited endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from experiments.rag_vs_pra.datasets import load_multihop_rag


SCHEMA_VERSION = "paper3.3-frozen-splits-v1"
DEFAULT_LEGACY_MANIFEST = Path(
    "docs/papers/shared/results/paper3_2_rag/crossdoc_adapter/"
    "qwen3_1_7b_rank8_five_seed/manifest.json"
)


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def legacy_evaluation_ids(manifest_path: Path) -> tuple[str, ...]:
    """Read the explicitly declared final Paper 3.2 evaluation identities."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = tuple(str(value) for value in payload.get("eval_example_ids", ()))
    if not values:
        raise ValueError(f"no eval_example_ids found in {manifest_path}")
    if len(values) != len(set(values)):
        raise ValueError("legacy evaluation identities are not unique")
    return values


def make_splits(
    example_ids: Sequence[str],
    *,
    excluded_ids: Sequence[str],
    seed: int,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> dict[str, tuple[str, ...]]:
    """Create stable disjoint assignments after removing legacy identities."""

    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("all split sizes must be positive")
    unique = tuple(dict.fromkeys(str(value) for value in example_ids))
    if len(unique) != len(example_ids):
        raise ValueError("dataset example identities are not unique")
    excluded = set(str(value) for value in excluded_ids)
    available = [value for value in unique if value not in excluded]
    required = train_size + validation_size + test_size
    if len(available) < required:
        raise ValueError(f"need {required} examples but only {len(available)} remain")
    random.Random(seed).shuffle(available)
    train_end = train_size
    validation_end = train_end + validation_size
    test_end = validation_end + test_size
    return {
        "train": tuple(sorted(available[:train_end])),
        "validation": tuple(sorted(available[train_end:validation_end])),
        "test": tuple(sorted(available[validation_end:test_end])),
    }


def build_manifest(
    *,
    example_ids: Sequence[str],
    excluded_ids: Sequence[str],
    dataset_metadata: Mapping[str, str],
    seed: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    legacy_manifest: Path,
) -> dict[str, object]:
    splits = make_splits(
        example_ids,
        excluded_ids=excluded_ids,
        seed=seed,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )
    assigned = set().union(*(set(values) for values in splits.values()))
    excluded = set(excluded_ids)
    overlap = assigned & excluded
    if overlap:
        raise RuntimeError(f"legacy evaluation leakage: {sorted(overlap)}")
    split_overlap = (
        (set(splits["train"]) & set(splits["validation"]))
        | (set(splits["train"]) & set(splits["test"]))
        | (set(splits["validation"]) & set(splits["test"]))
    )
    if split_overlap:
        raise RuntimeError(f"split overlap: {sorted(split_overlap)}")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dict(dataset_metadata),
        "split_seed": seed,
        "evaluation_seeds": [11, 23, 37, 71, 101],
        "legacy_paper3_2_manifest": legacy_manifest.as_posix(),
        "legacy_paper3_2_eval_ids": sorted(excluded),
        "legacy_eval_excluded_from_all_splits": True,
        "train_ids": list(splits["train"]),
        "validation_ids": list(splits["validation"]),
        "test_ids": list(splits["test"]),
        "counts": {key: len(value) for key, value in splits.items()},
        "git_commit": _git_commit(),
    }
    payload["split_digest"] = _digest(
        {
            "dataset_revision": dataset_metadata["dataset_revision"],
            "corpus_revision": dataset_metadata["corpus_revision"],
            "seed": seed,
            "legacy_ids": sorted(excluded),
            "splits": splits,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument("--seed", type=int, default=3301)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--validation-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, questions, metadata = load_multihop_rag(args.cache_dir)
    excluded = legacy_evaluation_ids(args.legacy_manifest)
    payload = build_manifest(
        example_ids=[question.example_id for question in questions],
        excluded_ids=excluded,
        dataset_metadata=metadata,
        seed=args.seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        legacy_manifest=args.legacy_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
