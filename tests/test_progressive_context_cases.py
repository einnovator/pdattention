import csv
from collections import Counter

from data.progressive_context_cases import ContextCaseClass, progressive_context_cases
from experiments.paper7_records.run_progressive_context_iteration import (
    Baseline,
    _paired_bootstrap,
    _read_csv,
)
from pra_hf.progressive_context import ContextAction


def test_progressive_context_fixture_is_balanced_and_functional():
    cases = progressive_context_cases()
    counts = Counter(case.case_class for case in cases)

    assert len(cases) == 30
    assert counts == {case_class: 5 for case_class in ContextCaseClass}
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.expected_action for case in cases} == set(ContextAction)


def test_progressive_context_fixture_declares_mechanism_arguments():
    for case in progressive_context_cases():
        if case.expected_action == ContextAction.MATERIALIZE_MORE:
            assert case.selector
            assert case.capabilities.partial_selectors
        elif case.expected_action == ContextAction.SEARCH_RECORD:
            assert case.search_query
            assert case.capabilities.searchable
        elif case.expected_action in {ContextAction.CURSOR_NEXT, ContextAction.CURSOR_QUERY}:
            assert case.cursor_collection
        elif case.expected_action == ContextAction.CALL_TOOL:
            assert case.tool_name and case.tool_payload


def test_progressive_csv_reload_preserves_seed_keys_for_postprocessing(tmp_path):
    path = tmp_path / "rows.csv"
    fields = ("case_id", "seed", "baseline", "task_success")
    rows = [
        {"case_id": "case-a", "seed": seed, "baseline": baseline.value,
         "task_success": int(baseline == Baseline.PRA_PLUS_MODEL)}
        for seed in (11, 23)
        for baseline in (Baseline.PRA_PLUS_MODEL, Baseline.COMPACT_ONLY)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    loaded = _read_csv(path)
    comparison = _paired_bootstrap(
        loaded, Baseline.PRA_PLUS_MODEL, Baseline.COMPACT_ONLY, iterations=100
    )

    assert {row["seed"] for row in loaded} == {11, 23}
    assert comparison["paired_case_delta"] == 1.0
