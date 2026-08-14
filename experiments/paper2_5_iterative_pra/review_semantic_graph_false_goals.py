"""Decode and classify the validation q95 false terminals for Paper 2.5."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples


PLAUSIBLY_RELEVANT = {
    ("5a84c4bb5542991dd0999dec", 4),
    ("5ae6143f5542996de7b71b2d", 0),
    ("5abce85755429959677d6b3e", 3),
    ("1805.07133:b9ea841b817ba23281c95c7a769873b840dee8d5", 20),
    ("1605.03481:b85fc420eb2f77f6f14f375cc1fcc5155eb5c0a8", 12),
    ("1605.03481:b85fc420eb2f77f6f14f375cc1fcc5155eb5c0a8", 9),
    ("2002.03407:10d450960907091f13e0be55f40bcb96f44dd074", 5),
    ("1808.10245:da9c0637623885afaf023a319beee87898948fe9", 8),
    ("2001.02885:96b07373756d7854bccc3c12e8d41454ab8741f5", 5),
    ("2001.02885:96b07373756d7854bccc3c12e8d41454ab8741f5", 4),
    ("1810.12085:c2cb6c4500d9e02fc9a1bdffd22c3df69655189f", 6),
    ("1912.13337:e97186c51d4af490dba6faaf833d269c8256426c", 26),
    ("1910.14599:ee2c9bc24d70daa0c87e38e0558e09ab97feb4f2", 22),
    ("1910.14599:ee2c9bc24d70daa0c87e38e0558e09ab97feb4f2", 2),
    ("1910.14599:ee2c9bc24d70daa0c87e38e0558e09ab97feb4f2", 9),
    ("2004.04721:664b3eadc12c8dde309e8bbd59e9af961a433cde", 18),
}


def run(args: argparse.Namespace) -> dict:
    with args.goal_rows.open(newline="", encoding="utf-8") as stream:
        false_rows = [
            row
            for row in csv.DictReader(stream)
            if row["partition"] == "validation"
            and row["strategy"] == "best_first"
            and row["theta_goal_label"] == "q95"
            and float(row["false_goal"]) == 1.0
        ]
    examples = {
        row["id"]: row
        for row in load_split_examples(args.cache_dir, 16, 8, args.data_seed)
    }
    features = {
        row["example_id"]: row
        for row in torch.load(args.source_features, map_location="cpu", weights_only=False)
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    output = []
    keys = sorted(
        {(row["example_id"], int(row["terminal_parent"])) for row in false_rows}
    )
    for example_id, parent in keys:
        example = examples[example_id]
        feature = features[example_id]
        start, end = feature["parent_spans"][parent]
        source_ids = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        classification = (
            "non_oracle_but_plausibly_relevant"
            if (example_id, parent) in PLAUSIBLY_RELEVANT
            else "false_semantic_closure"
        )
        output.append(
            {
                "dataset": example["dataset"],
                "example_id": example_id,
                "terminal_parent": parent,
                "seed_count": sum(
                    row["example_id"] == example_id
                    and int(row["terminal_parent"]) == parent
                    for row in false_rows
                ),
                "question": example["question"],
                "answer": example["answer"],
                "terminal_text": tokenizer.decode(
                    source_ids[int(start) : int(end)], skip_special_tokens=True
                ),
                "classification": classification,
            }
        )
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)
    weighted = Counter()
    unique = Counter()
    for row in output:
        unique[row["classification"]] += 1
        weighted[row["classification"]] += row["seed_count"]
    summary = {"rows": len(false_rows), "unique": dict(unique), "weighted": dict(weighted)}
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary)
    return summary


def parse_args() -> argparse.Namespace:
    result_dir = (
        ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/semantic_graph_search"
    )
    native_dir = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/native_qk_closure"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument(
        "--goal-rows", type=Path, default=result_dir / "goal_calibration_rows.csv"
    )
    parser.add_argument(
        "--source-features",
        type=Path,
        default=native_dir / "native_qk_features_test.pt",
    )
    parser.add_argument("--output", type=Path, default=result_dir / "false_goal_review.csv")
    parser.add_argument(
        "--summary",
        type=Path,
        default=result_dir / "false_goal_review_summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
