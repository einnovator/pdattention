from collections import Counter

from data.full_pra_context_cases import OmissionStratum, full_pra_context_cases
from pra_hf.progressive_context import ContextAction
from pra_hf.typed_context import CompressorRegistry


def test_full_pra_fixture_is_balanced_partitioned_and_stratified():
    cases = full_pra_context_cases()

    assert len(cases) == 30
    assert Counter(case.case_class for case in cases) == {
        f"C{index}_{name}": 5
        for index, name in enumerate(("CONTINUE", "FULL", "MORE", "CURSOR", "SEARCH", "TOOL"))
    }
    assert Counter(case.partition for case in cases) == {"validation": 12, "test": 18}
    assert {case.omission_stratum for case in cases} == set(OmissionStratum)
    assert {case.expected_action for case in cases} == set(ContextAction)


def test_hidden_strata_keep_answer_out_of_declared_semantic_cue():
    for case in full_pra_context_cases():
        if case.omission_stratum in {
            OmissionStratum.SEMANTIC_OMISSION,
            OmissionStratum.OPAQUE_HIDDEN,
        }:
            assert case.expected_answer not in str(case.semantic_cue)


def test_compact_index_cannot_see_backing_only_answer():
    registry = CompressorRegistry()
    for case in full_pra_context_cases():
        compact = registry.compress(case.record_type, case.payload, unit_limit=3)
        visible = str(compact.compact_payload)
        if case.omission_stratum == OmissionStratum.COMPACT_EXPLICIT:
            assert case.expected_answer in visible
        else:
            assert case.expected_answer not in visible
