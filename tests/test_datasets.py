from pathlib import Path

import pytest

from pra_core.datasets import load_dataset, load_documents, load_questions, load_references


STAGES = [
    "stage0_synthetic_memory",
    "stage1_hierarchical_synthetic",
    "stage2_code_repos",
    "stage3_wikipedia",
    "stage4_books",
    "stage5_technical_docs",
    "stage6_github_repos",
]


@pytest.mark.parametrize("stage", STAGES)
def test_stage_loads_common_example_format(stage):
    examples = load_dataset(stage, "data", max_examples=1)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.prompt
    assert ex.target.startswith(" ")
    assert ex.reference_table.all()
    assert ex.refs
    assert ex.summaries
    assert isinstance(ex.expected_ref_ids, list)
    assert isinstance(ex.expected_anchors, list)


def test_raw_stage_loaders_read_jsonl_rows():
    stage_path = Path("data") / "stage0_synthetic_memory"
    assert load_documents(stage_path)[0]["uri"] == "memory://animal/cat"
    assert load_references(stage_path)[0]["token"] == "<REF_1>"
    assert load_questions(stage_path)[0]["expected_ref_ids"] == [1]
