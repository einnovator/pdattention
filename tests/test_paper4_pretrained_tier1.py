import json

from experiments.paper4_training.run_pretrained_tier1 import (
    load_examples,
    pra_layers,
    split_examples,
)
from experiments.paper4_training.aggregate_pretrained_tier1 import aggregate


def test_pra_layers_follow_preregistered_spacing():
    assert pra_layers(16, 4) == (3, 7, 11, 15)
    assert pra_layers(2, 4) == (1,)


def test_wikitext_join_resolves_relevant_part_identity(tmp_path):
    documents = [
        {"uri": "memory://part-1", "text": "first"},
        {"uri": "memory://part-2", "text": "second evidence"},
    ]
    question = {
        "id": "q1",
        "prompt": "<REF_2> <REF_1> Continue:",
        "answer": "target",
        "reference_uris": ["memory://part-1", "memory://part-2"],
        "part_ids": [2, 1],
        "relevant_part_ids": [1],
    }
    for name, rows in (("documents.jsonl", documents), ("questions.jsonl", [question])):
        (tmp_path / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    examples = load_examples(tmp_path)
    assert examples[0].evidence == "second evidence"
    assert examples[0].distractor == "first"
    assert "<REF_" not in examples[0].prompt


def test_split_is_deterministic_and_disjoint(tmp_path):
    documents = [
        {"uri": f"memory://{index}", "text": f"text {index}"}
        for index in range(6)
    ]
    questions = [
        {
            "id": f"q{index}",
            "prompt": "Continue:",
            "answer": "target",
            "reference_uris": [f"memory://{index}"],
            "part_ids": [1],
            "relevant_part_ids": [1],
        }
        for index in range(6)
    ]
    for name, rows in (("documents.jsonl", documents), ("questions.jsonl", questions)):
        (tmp_path / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    examples = load_examples(tmp_path)
    train_a, validation_a = split_examples(examples, 11, 4, 2)
    train_b, validation_b = split_examples(examples, 11, 4, 2)
    assert train_a == train_b
    assert validation_a == validation_b
    assert {row.example_id for row in train_a}.isdisjoint(
        row.example_id for row in validation_a
    )


def test_five_seed_aggregate_requires_and_summarizes_all_seeds(tmp_path):
    paths = []
    for seed, delta in zip((11, 23, 37, 53, 71), (-0.2, -0.3, -0.1, -0.4, -0.2)):
        payload = {
            "model_id": "test/model",
            "split_seed": seed,
            "configuration": {
                "minimum_evidence_nll_gain": 0.05,
                "maximum_retention_nll_loss": 0.1,
                "minimum_causal_evidence_margin": 0.0,
            },
            "results": [
                {
                    "regime": "consumer_lora",
                    "evidence_nll_delta": delta,
                    "ordinary_retention_nll_delta": 0.01,
                    "evidence_vs_distractor_nll_margin": 0.2,
                }
            ],
        }
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = aggregate(paths)
    assert result["consumer_gate"] == "PASS"
    assert result["passing_regimes"] == ["consumer_lora"]
    assert result["regimes"][0]["seed_count"] == 5
    assert result["regimes"][0]["positive_causal_margin_seeds"] == 5
