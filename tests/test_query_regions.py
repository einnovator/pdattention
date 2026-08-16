from __future__ import annotations

import pytest

from pra_hf import PromptSegment, QueryRegionSelector, render_segments, token_offsets


@pytest.mark.parametrize(
    "prompt",
    (
        "QUESTION: Why did service X fail?\nCONTEXT:\n2026-01-01 ERROR timeout",
        "CONTEXT: deployment notes\nQUESTION: Why did service X fail?\nLOGS: ERROR timeout",
        "LOGS: ERROR timeout\nQUESTION: Why did service X fail?",
    ),
)
def test_structural_query_discovery_is_position_independent(prompt: str) -> None:
    selection = QueryRegionSelector().select(prompt, policy="structural")
    assert "Why did service X fail?" in selection.selected_text()[0]
    assert selection.policy == "structural"
    assert selection.confidence > 0.5


def test_explicit_span_is_authoritative() -> None:
    prompt = "context tokens QUESTION exact objective trailing payload"
    selection = QueryRegionSelector().select(prompt, query_spans=((2, 5),))
    assert selection.spans == ((2, 5),)
    assert selection.selected_text() == ("QUESTION exact objective",)
    assert selection.confidence == 1.0


def test_structured_segments_exclude_payload_and_url_candidates() -> None:
    segments = [
        PromptSegment("instruction", "Use only supplied evidence."),
        PromptSegment("query", "Which deployment caused the timeout?"),
        PromptSegment("context", "2026-01-01 ERROR timeout"),
        PromptSegment("references", "https://example.test/runbook"),
    ]
    selection = QueryRegionSelector().select(segments=segments, policy="structured")
    assert selection.selected_text() == ("Which deployment caused the timeout?",)
    assert selection.regions[0].role == "query"


def test_multi_region_preserves_instruction_and_question() -> None:
    segments = [
        PromptSegment("instruction", "Compare the two releases."),
        PromptSegment("context", "Release A succeeded; release B timed out."),
        PromptSegment("query", "Which release failed?"),
    ]
    selection = QueryRegionSelector(max_regions=2).select(
        segments=segments, policy="multi_region"
    )
    assert selection.selected_text() == (
        "Compare the two releases.",
        "Which release failed?",
    )


def test_session_state_uses_latest_query_not_whole_history() -> None:
    segments = [
        PromptSegment("history", "Earlier user asked about billing."),
        PromptSegment("query", "Why is the API timing out now?"),
        PromptSegment("tool_output", "ERROR upstream timeout"),
    ]
    selection = QueryRegionSelector().select(segments=segments, policy="session_state")
    assert selection.selected_text() == ("Why is the API timing out now?",)


def test_head_displacement_and_reinterpretation_change_the_cue() -> None:
    prompt = "QUESTION: Why did service X fail?\n" + "LOG filler\n" * 80
    selector = QueryRegionSelector(suffix_tokens=8)
    head = selector.select(prompt, policy="head")
    retried = selector.reinterpret(prompt, head)
    assert "QUESTION" not in head.selected_text()[0]
    assert "Why did service X fail?" in retried.selected_text()[0]
    assert retried.spans != head.spans


def test_segment_serialization_and_token_offsets_are_deterministic() -> None:
    segments = [PromptSegment("query", "alpha beta"), PromptSegment("context", "gamma")]
    first = render_segments(segments)
    second = render_segments(segments)
    assert first == second
    assert token_offsets(first[0]) == ((0, 5), (6, 10), (11, 16))
