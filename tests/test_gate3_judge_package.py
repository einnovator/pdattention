from experiments.paper2_5_iterative_pra.build_gate3_judge_package import pairs_from_artifact
from experiments.paper2_hf.build_behavioral_judge_package import (
    add_controls,
    build_package,
    validate_package,
)


def _row(condition, phase="heldout"):
    return {
        "phase": phase,
        "partition": "test" if phase == "heldout" else "validation",
        "dataset": "musique",
        "example_id": "example-1",
        "condition": condition,
        "question": "Where is the answer?",
        "reference_answer": "Paris",
        "generated_answer": condition.replace("_", " "),
        "token_f1": 0.5,
        "selected_source_fraction": 0.25,
        "active_kv_fraction": 0.25,
    }


def _artifact():
    heldout = [
        _row(condition)
        for condition in (
            "native_bounded",
            "one_shot",
            "graph_sparse",
            "graph_balanced",
            "graph_high",
            "oracle_evidence",
            "native_full_context",
        )
    ]
    layer = [
        _row("graph_balanced__late_1", "layer_sweep"),
        _row("graph_balanced__all_28", "layer_sweep"),
    ]
    return {
        "model_id": "model",
        "model_revision": "revision",
        "band_selection": {"selected_bands": {"musique": "late_1"}},
        "rows": heldout + layer,
    }


def test_gate3_pairs_cover_frozen_comparisons_and_native_controls(tmp_path):
    pairs = pairs_from_artifact(_artifact())
    assert {pair.group for pair in pairs} == {
        "gate3_balanced_vs_one_shot",
        "gate3_high_vs_one_shot",
        "gate3_balanced_vs_native",
        "gate3_balanced_vs_oracle",
        "gate3_selected_band_vs_all_layers",
    }
    controlled = add_controls(pairs)
    assert len(controlled) == len(pairs) + 2
    items, truth = build_package(
        controlled,
        tmp_path,
        seed=2505,
        include_order_reversal=True,
        batch_size=40,
        input_paths=[],
        availability={},
    )
    validate_package(items, truth)
    assert len(items["items"]) == 2 * len(controlled)
    assert all("condition" not in str(row).casefold() for row in items["items"])
    assert all("reference_answer" not in row for row in items["items"])
