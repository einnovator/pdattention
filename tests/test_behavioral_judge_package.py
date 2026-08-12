import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.paper2_hf.build_behavioral_judge_package import (
    BLIND_ITEM_KEYS,
    JUDGE_PROMPT,
    REQUESTED_SCORES,
    PairSpec,
    _response_schema,
    _test_offset,
    add_controls,
    build_package,
    validate_package,
)


def _pair(example_id: str = "example-1") -> PairSpec:
    return PairSpec(
        source_example_id=example_id,
        dataset="fixture",
        task_type="qa",
        prompt="Question: Which city is the capital of France?\nAnswer:",
        left_condition="native_no_context",
        right_condition="pra_routed_frozen",
        left_answer="Paris is the capital of France.",
        right_answer="The capital of France is Paris.",
        group="native_no_context_vs_pra",
        model_id="fixture/model",
        model_revision="fixture-revision",
        generation_mode="greedy",
        pra_selected_fraction=0.1,
        pra_materialized_kv_fraction=0.06,
    )


def _build(output_dir: Path, pairs: list[PairSpec], *, reversal: bool = True):
    return build_package(
        pairs,
        output_dir,
        seed=1234,
        include_order_reversal=reversal,
        batch_size=2,
        input_paths=[],
        availability={"fixture": True},
    )


def test_randomization_and_reversals_are_deterministic(tmp_path: Path) -> None:
    pairs = [_pair("example-1"), _pair("example-2")]
    items_a, truth_a = _build(tmp_path / "a", pairs)
    items_b, truth_b = _build(tmp_path / "b", pairs)

    assert items_a == items_b
    assert truth_a["items"] == truth_b["items"]

    by_pair: dict[str, list[dict]] = {}
    for row in truth_a["items"]:
        by_pair.setdefault(row["pair_group_id"], []).append(row)
    assert all(len(rows) == 2 for rows in by_pair.values())
    for rows in by_pair.values():
        first, second = rows
        assert first["condition_a"] == second["condition_b"]
        assert first["condition_b"] == second["condition_a"]
        assert first["answer_a_sha256"] == second["answer_b_sha256"]
        assert first["answer_b_sha256"] == second["answer_a_sha256"]


def test_blind_items_have_only_opaque_fields(tmp_path: Path) -> None:
    items, truth = _build(tmp_path, [_pair()])

    assert all(set(row) == BLIND_ITEM_KEYS for row in items["items"])
    assert all(row["item_id"].startswith("judge_") for row in items["items"])
    structural_text = json.dumps(
        [{"item_id": row["item_id"], "keys": sorted(row)} for row in items["items"]]
    ).lower()
    for source_hint in ("native", "pra", "routed", "qwen", "hotpot", "qasper"):
        assert source_hint not in structural_text
    validate_package(items, truth)


def test_validation_rejects_leaked_ids_and_order_mismatch(tmp_path: Path) -> None:
    items, truth = _build(tmp_path, [_pair()], reversal=False)

    leaked = copy.deepcopy(items)
    leaked["items"][0]["item_id"] = "native_000001"
    leaked_truth = copy.deepcopy(truth)
    leaked_truth["items"][0]["item_id"] = "native_000001"
    with pytest.raises(ValueError, match="leaks a source hint"):
        validate_package(leaked, leaked_truth)

    reordered = copy.deepcopy(items)
    row = reordered["items"][0]
    row["answer_a"], row["answer_b"] = row["answer_b"], row["answer_a"]
    with pytest.raises(ValueError, match="order mismatch"):
        validate_package(reordered, truth)


def test_controls_are_calibrated_and_do_not_mutate_input() -> None:
    original = [_pair()]
    augmented = add_controls(original)

    assert len(original) == 1
    assert len(augmented) == 3
    by_group = {pair.group: pair for pair in augmented}
    identical = by_group["calibration_identical"]
    corrupted = by_group["calibration_corrupted"]
    assert identical.left_answer == identical.right_answer
    assert corrupted.left_answer != corrupted.right_answer
    assert corrupted.metadata["corruption"] in {
        "meaningful_truncation",
        "explicit_contradiction",
    }


def test_package_files_batches_and_response_schema(tmp_path: Path) -> None:
    items, _ = _build(tmp_path, [_pair("one"), _pair("two")])

    expected = {
        "behavioral_judge_prompt.txt",
        "behavioral_judge_items.json",
        "behavioral_judge_truth.json",
        "behavioral_judge_response.schema.json",
        "behavioral_judge_example_response.json",
        "README.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert len(list((tmp_path / "batches").glob("*.json"))) == 2
    assert json.loads((tmp_path / "behavioral_judge_items.json").read_text())["items"] == items["items"]
    reason = _response_schema()["properties"]["items"]["items"]["properties"]["reason"]
    assert reason["maxLength"] == 320
    assert "39" in reason["pattern"]
    for score_name in REQUESTED_SCORES:
        assert score_name in JUDGE_PROMPT
    assert "-100 means Answer A is much better" in JUDGE_PROMPT
    assert "+100 means Answer B is much better" in JUDGE_PROMPT


def test_legacy_artifact_uses_recorded_last14_test_offset() -> None:
    assert _test_offset({"protocol": "legacy descriptive protocol"}) == 16
    assert _test_offset({"split_metadata": {"test_offset": 24}}) == 24


def test_documented_script_invocation_resolves_repository_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "experiments/paper2_hf/build_behavioral_judge_package.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--include-order-reversal" in result.stdout
