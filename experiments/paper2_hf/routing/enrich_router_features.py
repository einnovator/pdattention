"""Add deterministic lexical hardness metadata to cached frozen routing vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.precompute_router_features import lexical_chunk_scores
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples


def run(feature_dir: Path, cache_dir: Path, seed: int) -> dict:
    """Enrich existing tensors without running the frozen language model again."""
    manifest_path = feature_dir / "feature_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    enriched = {}
    for split, split_metadata in manifest["splits"].items():
        per_dataset = split_metadata["dataset_counts"]
        counts = set(int(value) for value in per_dataset.values())
        if len(counts) != 1:
            raise ValueError("The deterministic loader requires equal per-dataset split counts.")
        examples = load_split_examples(
            cache_dir,
            counts.pop(),
            int(split_metadata["offset"]),
            seed,
        )
        by_identity = {(row["dataset"], row["id"]): row for row in examples}
        tensor_path = feature_dir / split_metadata["path"]
        features = torch.load(tensor_path, weights_only=False)
        for feature in features:
            identity = (feature["dataset"], feature["example_id"])
            example = by_identity.get(identity)
            if example is None:
                raise KeyError(f"Missing source example for frozen feature {identity!r}.")
            feature["lexical_scores"] = lexical_chunk_scores(
                tokenizer,
                example["source"],
                example["question"],
                feature["chunk_spans"],
            )
        torch.save(features, tensor_path)
        enriched[split] = len(features)

    manifest.update(
        {
            "lexical_hardness": "question/chunk token-id set Jaccard",
            "native_kv_heads": int(config.num_key_value_heads),
            "native_head_dim": int(config.head_dim),
            "native_kv_dtype_bytes": 2,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"enriched_examples": enriched, "manifest": str(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "papers"
            / "shared"
            / "results"
            / "paper2_hf"
            / "routing"
            / "learned_adapter"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.feature_dir, args.cache_dir, args.seed), indent=2))
