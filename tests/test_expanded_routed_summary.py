from experiments.engine_serving.summarize_expanded_routed_cohort import summarize


def test_summary_keeps_sample_count_and_unique_identity_separate() -> None:
    base = {
        "condition": "routed_native",
        "evidence_recall_at_4": 0.5,
        "token_f1": 0.25,
        "gold_answer_logprob": -2.0,
        "routing_ms": 3.0,
        "index_build_ms": 4.0,
    }
    rows = summarize(
        [
            {
                "dataset": "qasper",
                "rows": [
                    {**base, "example_id": "a"},
                    {**base, "example_id": "a"},
                    {**base, "example_id": "b"},
                ],
            }
        ]
    )

    assert rows[0]["sampled_examples"] == 3
    assert rows[0]["unique_examples"] == 2
    assert rows[0]["evidence_recall_at_4"] == 0.5
