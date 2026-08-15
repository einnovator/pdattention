"""Build a canonical blinded judge package for Gate-3 generated answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper2_hf.build_behavioral_judge_package import (
    PairSpec,
    add_controls,
    build_package,
)


def _index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    indexed = {}
    for row in rows:
        key = (row["dataset"], row["example_id"], row["condition"])
        if key in indexed:
            raise ValueError(f"duplicate generation identity: {key}")
        indexed[key] = row
    return indexed


def _pair(
    left: dict,
    right: dict,
    *,
    left_condition: str,
    right_condition: str,
    group: str,
    artifact: dict,
) -> PairSpec:
    if left["question"] != right["question"]:
        raise ValueError("paired generations do not share an exact question")
    if left["reference_answer"] != right["reference_answer"]:
        raise ValueError("paired generations do not share an exact reference answer")
    left_answer = str(left["generated_answer"]).strip()
    right_answer = str(right["generated_answer"]).strip()
    return PairSpec(
        source_example_id=left["example_id"],
        dataset=left["dataset"],
        task_type="question_answering",
        prompt=left["question"],
        left_condition=left_condition,
        right_condition=right_condition,
        left_answer=left_answer or "[No generated answer]",
        right_answer=right_answer or "[No generated answer]",
        group=group,
        model_id=artifact["model_id"],
        model_revision=artifact["model_revision"],
        generation_mode="greedy",
        generation_seed=11,
        pra_selected_fraction=right.get("selected_source_fraction"),
        pra_materialized_kv_fraction=right.get("active_kv_fraction"),
        metadata={
            "partition": left["partition"],
            "reference_answer": left["reference_answer"],
            "left_token_f1": left["token_f1"],
            "right_token_f1": right["token_f1"],
            "left_answer_blank": not bool(left_answer),
            "right_answer_blank": not bool(right_answer),
        },
    )


def pairs_from_artifact(artifact: dict) -> list[PairSpec]:
    """Create the five frozen pairwise comparisons with no metrics in blind items."""
    rows = artifact["rows"]
    heldout = [row for row in rows if row["phase"] == "heldout"]
    layer = [row for row in rows if row["phase"] == "layer_sweep"]
    heldout_index = _index(heldout)
    layer_index = _index(layer)
    pairs: list[PairSpec] = []
    for dataset, example_id in sorted({(row["dataset"], row["example_id"]) for row in heldout}):
        def held(condition: str) -> dict:
            return heldout_index[(dataset, example_id, condition)]

        pairs.extend(
            (
                _pair(held("one_shot"), held("graph_balanced"),
                      left_condition="one_shot", right_condition="graph_balanced",
                      group="gate3_balanced_vs_one_shot", artifact=artifact),
                _pair(held("one_shot"), held("graph_high"),
                      left_condition="one_shot", right_condition="graph_high",
                      group="gate3_high_vs_one_shot", artifact=artifact),
                _pair(held("native_bounded"), held("graph_balanced"),
                      left_condition="native_bounded", right_condition="graph_balanced",
                      group="gate3_balanced_vs_native", artifact=artifact),
                _pair(held("oracle_evidence"), held("graph_balanced"),
                      left_condition="oracle_evidence", right_condition="graph_balanced",
                      group="gate3_balanced_vs_oracle", artifact=artifact),
            )
        )

    selected = artifact["band_selection"]["selected_bands"]
    validation_ids = sorted({(row["dataset"], row["example_id"]) for row in layer})
    for dataset, example_id in validation_ids:
        selected_name = f"graph_balanced__{selected[dataset]}"
        pairs.append(
            _pair(
                layer_index[(dataset, example_id, "graph_balanced__all_28")],
                layer_index[(dataset, example_id, selected_name)],
                left_condition="graph_balanced_all_28",
                right_condition="graph_balanced_selected_band",
                group="gate3_selected_band_vs_all_layers",
                artifact=artifact,
            )
        )
    return pairs


def run(args: argparse.Namespace) -> tuple[dict, dict]:
    artifact = json.loads(args.results.read_text(encoding="utf-8"))
    pairs = add_controls(pairs_from_artifact(artifact))
    return build_package(
        pairs,
        args.output_dir,
        seed=2505,
        include_order_reversal=True,
        batch_size=args.batch_size,
        input_paths=[args.results],
        availability={
            "protocol": "existing validated Paper-2 behavioral-equivalence harness",
            "external_sota_judge": "response required before headline use",
            "retrieval_metrics_visible_to_judge": False,
            "reference_answer_visible_to_judge": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path("docs/papers/shared/results/paper2_5_iterative_pra/output_validation")
    parser.add_argument("--results", type=Path, default=default / "gate3_generation_results.json")
    parser.add_argument("--output-dir", type=Path, default=default / "behavioral_judge")
    parser.add_argument("--batch-size", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    items, truth = run(parse_args())
    print(json.dumps({"blind_items": len(items["items"]), "private_truth_items": len(truth["items"])}, indent=2))
