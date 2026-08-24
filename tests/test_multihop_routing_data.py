import json

from pra_hf.multihop_routing_data import cohort_manifest, load_multihop_routing_examples


def _write_fixture(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "dataset": "2wikimultihopqa",
                    "example_id": "wiki-validation",
                    "split": "validation",
                    "question": "Who directed Alpha?",
                },
                {
                    "dataset": "musique",
                    "example_id": "music-test",
                    "split": "test",
                    "question": "Where was Beta born?",
                },
            )
        ),
        encoding="utf-8",
    )
    twowiki = tmp_path / "2wiki.json"
    twowiki.write_text(
        json.dumps(
            [
                {
                    "_id": "wiki-validation",
                    "question": "Who directed Alpha?",
                    "answer": "Ada",
                    "context": [
                        ["Alpha", ["Alpha was directed by Ada.", "It opened in 2001."]],
                        ["Noise", ["A distractor sentence."]],
                    ],
                    "supporting_facts": [["Alpha", 0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    musique = tmp_path / "musique.jsonl"
    musique.write_text(
        json.dumps(
            {
                "id": "music-test",
                "question": "Where was Beta born?",
                "answer": "Lisbon",
                "paragraphs": [
                    {
                        "title": "Beta",
                        "paragraph_text": "Beta was born in Lisbon.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Noise",
                        "paragraph_text": "A distractor paragraph.",
                        "is_supporting": False,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return annotations, twowiki, musique


def test_multihop_sources_preserve_unique_authored_evidence(tmp_path):
    paths = _write_fixture(tmp_path)
    examples = load_multihop_routing_examples(*paths)
    assert len(examples) == 2
    for example in examples:
        assert example.evidence
        assert all(example.source.count(text) == 1 for text in example.evidence)
        assert all("[doc=" in text for text in example.evidence)


def test_multihop_manifest_is_disjoint_and_stable(tmp_path):
    examples = load_multihop_routing_examples(*_write_fixture(tmp_path))
    first = cohort_manifest(examples)
    second = cohort_manifest(reversed(examples))
    assert first["identity_sha256"] == second["identity_sha256"]
    assert first["validation_test_identity_disjoint"] is True
    assert first["dataset_split_counts"]["2wikimultihopqa"]["validation"] == 1
    assert first["dataset_split_counts"]["musique"]["test"] == 1
