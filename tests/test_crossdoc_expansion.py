from __future__ import annotations

from dataclasses import replace

import pytest

from pra_hf.crossdoc_expansion import (
    BuiltinCrossDocumentExpansionPolicy,
    CrossDocumentBudget,
    CrossDocumentDirection,
    CrossDocumentExpansionConfig,
    CrossDocumentExpansionMode,
    CrossDocumentExpansionRequest,
    CrossDocumentExpansionRuntime,
    CrossDocumentFailure,
    CrossDocumentGranularity,
    CrossDocumentPolicyEvidence,
    CrossDocumentPolicyQualification,
    CrossDocumentSelectedRecord,
    build_cross_document_expansion_policy,
    qualify_cross_document_policy,
    _target_interval,
)
from pra_hf.rag_evaluation import RAGChunk


def _chunk(document: str, ordinal: int, text: str, start: int) -> RAGChunk:
    return RAGChunk(
        chunk_id=f"{document}:chunk:{ordinal}",
        document_id=document,
        ordinal=ordinal,
        start=start,
        end=start + len(text),
        text=text,
        token_count=len(text.split()),
    )


def _fixture() -> tuple[CrossDocumentExpansionRequest, dict[str, RAGChunk]]:
    chunks = {
        "d1s": _chunk("d1", 0, "Alice discovered the amber key in Paris.", 0),
        "d1x": _chunk("d1", 1, "The expedition ended in winter.", 50),
        "d2s": _chunk("d2", 0, "The lunar vault was designed by Bob.", 0),
        "d2x": _chunk("d2", 1, "The amber key opens the lunar vault.", 50),
        "d3s": _chunk("d3", 0, "Carol maintains a garden in Rome.", 0),
        "d3x": _chunk("d3", 1, "A blue key opens the garden shed.", 50),
    }
    selected = (
        CrossDocumentSelectedRecord("pra://d1", 1, (chunks["d1s"],), 0.9),
        CrossDocumentSelectedRecord("pra://d2", 2, (chunks["d2s"],), 0.8),
        CrossDocumentSelectedRecord("pra://d3", 3, (chunks["d3s"],), 0.2),
    )
    request = CrossDocumentExpansionRequest(
        query="Which vault does Alice's amber key open?",
        selection_receipt_id="selection-1",
        selected_records=selected,
        candidate_chunks_by_record={
            "pra://d1": (chunks["d1s"], chunks["d1x"]),
            "pra://d2": (chunks["d2s"], chunks["d2x"]),
            "pra://d3": (chunks["d3s"], chunks["d3x"]),
        },
        budget=CrossDocumentBudget(top_k_per_pair=1, max_extra_tokens=16),
        authorized_record_uris=frozenset(("pra://d1", "pra://d2", "pra://d3")),
        gold_chunk_ids=frozenset((chunks["d2x"].chunk_id,)),
    )
    return request, chunks


def _run(
    request: CrossDocumentExpansionRequest,
    mode: CrossDocumentExpansionMode,
    **config: object,
):
    policy = BuiltinCrossDocumentExpansionPolicy(
        CrossDocumentExpansionConfig(mode=mode, budget=request.budget, **config)
    )
    return CrossDocumentExpansionRuntime().expand(request, policy)


def test_none_is_a_receipted_no_op() -> None:
    request, _ = _fixture()
    plan = _run(request, CrossDocumentExpansionMode.NONE)

    assert plan.spans == ()
    assert plan.receipt.policy_name == "none"
    assert plan.receipt.selection_receipt_id == "selection-1"
    assert plan.receipt.new_native_tokens == 0
    assert len(plan.receipt.receipt_id) == 64


def test_query_conditioned_lexical_expansion_finds_peer_evidence() -> None:
    request, chunks = _fixture()
    plan = _run(request, CrossDocumentExpansionMode.LEXICAL, query_conditioned=True)

    assert chunks["d2x"].chunk_id in {row.target_chunk_id for row in plan.spans}
    assert all(row.target_record_uri != row.source_record_uri for row in plan.spans)
    assert plan.receipt.deduplicated_tokens <= 16
    assert plan.receipt.requested_tokens >= plan.receipt.deduplicated_tokens


@pytest.mark.parametrize(
    ("direction", "expected_pairs"),
    [
        (CrossDocumentDirection.SYMMETRIC, 6),
        (CrossDocumentDirection.HIGHER_TO_LOWER, 3),
        (CrossDocumentDirection.LOWER_TO_HIGHER, 3),
        (CrossDocumentDirection.TOP_RECORD_HUB, 2),
    ],
)
def test_directionality_limits_ordered_record_pairs(
    direction: CrossDocumentDirection, expected_pairs: int
) -> None:
    request, _ = _fixture()
    policy = BuiltinCrossDocumentExpansionPolicy(
        CrossDocumentExpansionConfig(
            mode="lexical", direction=direction, budget=request.budget
        )
    )

    candidates = policy.propose(request)

    assert len({row.pair for row in candidates}) == expected_pairs


def test_query_selected_pair_threshold_filters_low_relevance_record() -> None:
    request, _ = _fixture()
    policy = BuiltinCrossDocumentExpansionPolicy(
        CrossDocumentExpansionConfig(
            mode="lexical",
            direction="query_selected_pair_only",
            minimum_pair_query_score=0.5,
            budget=request.budget,
        )
    )

    candidates = policy.propose(request)

    assert {row.source_record_uri for row in candidates} == {"pra://d1", "pra://d2"}
    assert {row.target_record_uri for row in candidates} == {"pra://d1", "pra://d2"}


def test_overlap_is_removed_and_resident_native_tokens_are_reused() -> None:
    request, chunks = _fixture()
    request = replace(
        request,
        resident_chunk_ids=frozenset((chunks["d2x"].chunk_id,)),
        budget=CrossDocumentBudget(top_k_per_pair=2, max_extra_tokens=30),
    )
    plan = _run(request, CrossDocumentExpansionMode.ORACLE)

    assert all(row.target_chunk_id != chunks["d2s"].chunk_id for row in plan.spans)
    gold = [row for row in plan.spans if row.target_chunk_id == chunks["d2x"].chunk_id]
    assert gold
    assert sum(row.reused_native_tokens for row in gold) == sum(row.token_count for row in gold)
    assert plan.receipt.reused_native_tokens > 0
    assert CrossDocumentFailure.DUPLICATE_EXPANSION.value in plan.receipt.failures


def test_runtime_clips_final_interval_to_exact_global_budget() -> None:
    request, _ = _fixture()
    request = replace(request, budget=CrossDocumentBudget(top_k_per_pair=2, max_extra_tokens=3))

    plan = _run(request, CrossDocumentExpansionMode.LEXICAL)

    assert sum(row.token_count for row in plan.spans) == 3
    assert plan.receipt.deduplicated_tokens == 3
    assert all(row.end > row.start and row.text for row in plan.spans)


def test_oracle_uses_request_local_gold_annotations() -> None:
    request, chunks = _fixture()
    plan = _run(request, CrossDocumentExpansionMode.ORACLE)

    assert plan.spans[0].target_chunk_id == chunks["d2x"].chunk_id
    assert all(row["gold_support"] for row in plan.receipt.candidates)
    selected_candidate = next(
        row for row in plan.receipt.candidates if row["target_chunk_id"] == chunks["d2x"].chunk_id
    )
    assert selected_candidate["gold_support"] is True


def test_unauthorized_policy_proposal_is_filtered_by_runtime() -> None:
    request, _ = _fixture()
    request = replace(request, authorized_record_uris=frozenset(("pra://d1", "pra://d2")))
    plan = _run(request, CrossDocumentExpansionMode.LEXICAL)

    assert all(row.target_record_uri != "pra://d3" for row in plan.spans)
    assert CrossDocumentFailure.UNAUTHORIZED_TARGET.value in plan.receipt.failures


def test_hybrid_rrf_persists_independent_channel_ranks() -> None:
    request, _ = _fixture()
    plan = _run(request, CrossDocumentExpansionMode.HYBRID_RRF)

    assert plan.receipt.candidates
    assert all(row["lexical_rank"] is not None for row in plan.receipt.candidates)
    assert all(row["dense_rank"] is not None for row in plan.receipt.candidates)
    assert {row["selection_reason"] for row in plan.receipt.candidates} == {
        "reciprocal_rank_fusion"
    }


def test_dense_policy_batches_documents_and_identifies_encoder_revision() -> None:
    request, _ = _fixture()

    class RecordingEncoder:
        identity = "fixture-dense@abc123"

        def __init__(self) -> None:
            self.query_calls = 0
            self.document_batches = []

        def encode_query(self, text):
            self.query_calls += 1
            return (1.0, float("amber" in text.casefold()))

        def encode_documents(self, texts):
            self.document_batches.append(tuple(texts))
            return tuple((1.0, float("amber" in text.casefold())) for text in texts)

    encoder = RecordingEncoder()
    policy = BuiltinCrossDocumentExpansionPolicy(
        CrossDocumentExpansionConfig(mode="dense", budget=request.budget),
        semantic_encoder=encoder,
    )

    candidates = policy.propose(request)

    assert candidates
    assert encoder.query_calls == 1 + len(request.selected_records)
    assert len(encoder.document_batches) == 1
    assert len(encoder.document_batches[0]) == 6
    assert policy.revision.endswith("+fixture-dense@abc123")


def test_experimental_modes_fail_closed_in_sdk_factory() -> None:
    with pytest.raises(ValueError, match="experiment/debug"):
        build_cross_document_expansion_policy(CrossDocumentExpansionConfig(mode="oracle"))
    policy = build_cross_document_expansion_policy(
        CrossDocumentExpansionConfig(mode="oracle"), allow_experimental=True
    )
    assert policy.name == "oracle"


def test_parameter_free_modes_require_a_passing_sdk_qualification() -> None:
    config = CrossDocumentExpansionConfig(mode="lexical")
    with pytest.raises(ValueError, match="qualification gate"):
        build_cross_document_expansion_policy(config)

    policy = build_cross_document_expansion_policy(
        config,
        qualification=CrossDocumentPolicyQualification("SDK_OPTIONAL", True, ()),
    )
    assert policy.name == "lexical"


def test_second_expansion_round_fails_closed_until_implemented() -> None:
    with pytest.raises(ValueError, match="iterative cross-document expansion is locked"):
        CrossDocumentExpansionConfig(mode="lexical", iteration_depth=2)


def test_custom_policy_requires_an_explicit_plugin() -> None:
    with pytest.raises(ValueError, match="custom_policy"):
        build_cross_document_expansion_policy(CrossDocumentExpansionConfig(mode="custom"))


def test_sdk_qualification_requires_powered_multiscale_evidence() -> None:
    weak = qualify_cross_document_policy(
        CrossDocumentPolicyEvidence(
            policy_name="hybrid_rrf",
            gap_recovered=0.49,
            extra_native_fraction=0.10,
            official_score_delta_ci=(0.0, 0.1),
            seed_count=5,
            model_sizes=("1.7B", "4B"),
            cross_family_validated=True,
        )
    )
    assert weak.status == "RESEARCH_ONLY"
    assert weak.sdk_exposable is False

    strong = qualify_cross_document_policy(
        CrossDocumentPolicyEvidence(
            policy_name="hybrid_rrf",
            gap_recovered=0.85,
            extra_native_fraction=0.08,
            official_score_delta_ci=(0.0, 0.1),
            seed_count=5,
            model_sizes=("1.7B", "4B"),
            cross_family_validated=True,
        )
    )
    assert strong.status == "SDK_STRONG_CANDIDATE"
    assert strong.sdk_exposable is True


def test_receipt_contains_no_source_text() -> None:
    request, _ = _fixture()
    receipt = _run(request, CrossDocumentExpansionMode.HYBRID_RRF).receipt.to_dict()
    serialized = str(receipt)

    assert "Alice discovered" not in serialized
    assert "amber key opens" not in serialized
    assert all("target_text_sha256" in row for row in receipt["candidates"])
    assert all("text_sha256" in row for row in receipt["chosen_spans"])


def test_logical_interval_selects_the_best_matching_sentence() -> None:
    chunk = _chunk(
        "d2",
        4,
        "Routine maintenance ended. The amber key opens the lunar vault. Weather stayed clear.",
        100,
    )

    span, text, tokens = _target_interval(
        chunk,
        "Alice amber key vault",
        CrossDocumentGranularity.LOGICAL_INTERVAL,
    )

    assert text == "The amber key opens the lunar vault."
    assert span[0] > chunk.start
    assert span[1] < chunk.end
    assert tokens == 7


def test_receipt_persists_extracted_keyterms() -> None:
    request, _ = _fixture()
    receipt = _run(request, CrossDocumentExpansionMode.ENTITY).receipt.to_dict()

    assert receipt["candidates"]
    assert any("amber" in row["keyterms"] for row in receipt["candidates"])
