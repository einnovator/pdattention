from __future__ import annotations

from experiments.paper6_3_openvino.analyze_distractor_behavior import analyze


def test_forensics_marks_answer_aware_construction_and_paired_change() -> None:
    entry = {
        "dataset": "qasper",
        "example_id": "one",
        "question": "Which method improved recall?",
        "answer": "sparse routing",
        "selected_source": "The result discusses routing.",
        "distractor_source": (
            "Document: A\nSparse routing improved recall.\n\n"
            "Document: B\nUnrelated baseline details."
        ),
    }
    payload = {
        "schema_version": "source",
        "model_id": "model",
        "device": "GPU",
        "max_full_tokens": 100,
        "rows": [
            {
                "dataset": "qasper",
                "example_id": "one",
                "condition": "evidence_only",
                "token_f1": 0.0,
                "answer_containment": 0.0,
                "output_text": "unknown",
                "source_tokens": 20,
                "distractor_tokens": 0,
            },
            {
                "dataset": "qasper",
                "example_id": "one",
                "condition": "relevant_distractors_k1",
                "token_f1": 1.0,
                "answer_containment": 1.0,
                "output_text": "sparse routing",
                "source_tokens": 30,
                "distractor_tokens": 10,
            },
        ],
    }
    report = analyze(payload, {"entries": [entry]})

    assert report["construction_audit"]["relevance_terms"] == "question_plus_gold_answer"
    assert report["summaries"][0]["mean_f1_delta"] == 1.0
    assert report["summaries"][0]["answer_exact_in_added_fraction"] == 1.0
