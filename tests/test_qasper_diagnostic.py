from experiments.paper2_hf.qa.run_qasper_diagnostic import (
    _answer_contained,
    _frozen_new_comparison,
    _rationale_consistent,
    _starts_with_polarity,
    _terminal_punctuation,
    apply_calibrator,
    classify_error,
    hotpot_relation_distance,
    select_confidence_gate,
)


def _row(**updates):
    row = {
        "generated_text": "Yes, the evidence supports it.",
        "reference_answer": "yes",
        "question": "Does the evidence support it?",
        "source": "The evidence supports it.",
        "evidence": ["The evidence supports it."],
        "hit_max_new_tokens": False,
    }
    row.update(updates)
    return row


def test_polarity_and_finish_helpers_cover_markdown_and_subword_truncation():
    assert _starts_with_polarity("- **No**, it does not.") == "no"
    assert _starts_with_polarity("The answer is no.") is None
    assert _terminal_punctuation("Mr. Tumnus.")
    assert not _terminal_punctuation("Lucy Pev")
    assert _rationale_consistent("Yes, the evidence agrees.", "yes")
    assert not _rationale_consistent("Yes. No, that is incorrect.", "yes")


def test_answer_containment_is_case_and_punctuation_insensitive():
    assert _answer_contained("The answer is Mr. Tumnus.", "Mr. Tumnus")
    assert not _answer_contained("The answer is Lucy Pevensie.", "Mr. Tumnus")


def test_taxonomy_prioritizes_correct_and_polarity_over_truncation():
    primary, flags = classify_error(_row(hit_max_new_tokens=True), "No.")
    assert primary == "correct_semantically_equivalent"
    assert "generation_truncation" in flags

    primary, flags = classify_error(
        _row(generated_text="No, the evidence does not", hit_max_new_tokens=True),
        "Yes.",
    )
    assert primary == "polarity_inversion"
    assert "generation_truncation" in flags


def test_taxonomy_detects_no_displacement_before_generic_truncation():
    row = _row(
        generated_text="The question asks about the evidence",
        reference_answer="target entity",
        hit_max_new_tokens=True,
    )
    primary, flags = classify_error(row, row["generated_text"])
    assert primary == "no_behavioral_displacement"
    assert "no_behavioral_displacement" in flags


def test_affine_calibrator_applies_learned_threshold():
    row = {"yes_minus_no_logprob": -0.25}
    assert apply_calibrator(row, {"scale": 1.0, "bias": 0.5}) == "yes"
    assert apply_calibrator(row, {"scale": 1.0, "bias": 0.0}) == "no"


def test_confidence_gate_uses_validation_accuracy_then_margin():
    rows = [
        {
            "example_id": "a",
            "reference_answer": "yes",
            "gate_alpha": 0.0,
            "routing_top1_top2_margin": 0.1,
            "constrained_polarity": "no",
            "gold_polarity_margin": -1.0,
        },
        {
            "example_id": "a",
            "reference_answer": "yes",
            "gate_alpha": 1.0,
            "routing_top1_top2_margin": 0.1,
            "constrained_polarity": "yes",
            "gold_polarity_margin": 1.0,
        },
        {
            "example_id": "b",
            "reference_answer": "no",
            "gate_alpha": 0.0,
            "routing_top1_top2_margin": 0.01,
            "constrained_polarity": "no",
            "gold_polarity_margin": 2.0,
        },
        {
            "example_id": "b",
            "reference_answer": "no",
            "gate_alpha": 1.0,
            "routing_top1_top2_margin": 0.01,
            "constrained_polarity": "yes",
            "gold_polarity_margin": -2.0,
        },
    ]
    selected = select_confidence_gate(rows)
    assert selected["validation_accuracy"] == 1.0
    assert selected["threshold"] == 0.1


def test_hotpot_relation_distance_uses_answer_and_evidence_tiers():
    row = {
        "dataset": "hotpotqa",
        "generated_text": "The evidence discusses Paris and France.",
        "reference_answer": "London",
        "question": "Which city is the target?",
        "evidence": ["London is the final target.", "Paris is in France."],
        "source": "London is the final target. Paris is in France.",
    }
    assert hotpot_relation_distance(row) == 2
    row["generated_text"] = "London"
    assert hotpot_relation_distance(row) == 0


def test_frozen_new_comparison_ignores_rows_without_old_generation():
    base = {
        "dataset": "qasper",
        "condition": "pra_routed_frozen",
        "generated_text": "Yes.",
        "old_8_token_text": "Yes, because",
        "eos_emitted": True,
        "hit_max_new_tokens": False,
        "answer_contained": True,
    }
    rows = [base, {**base, "condition": "derived", "old_8_token_text": None}]
    result = _frozen_new_comparison(rows)
    assert len(result) == 1
    assert result[0]["old_terminal_punctuation_rate"] == 0.0
    assert result[0]["new_eos_rate"] == 1.0
