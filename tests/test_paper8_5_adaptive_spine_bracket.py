from experiments.paper8_5_agent_memory.adaptive_spine_bracket import recommend


def _episode(resolved: bool) -> dict:
    return {"official_resolved": resolved}


def _state(candidate_tokens: int | None = None, candidate_ok: bool = True) -> dict:
    cells = {
        "independent-n2-a__S01_persistent_full__r01": {
            "status": "complete",
            "cumulative_full_tokens": 1000,
            "cumulative_materialized_tokens": 1000,
            "official_resolved_count": 2,
            "episodes": {"e1": _episode(True), "e2": _episode(True)},
        }
    }
    if candidate_tokens is not None:
        cells["independent-n2-a__S03_completed_episode_spine-r4m2v2_tasks1__r01"] = {
            "status": "complete",
            "cumulative_full_tokens": 1000,
            "cumulative_materialized_tokens": candidate_tokens,
            "official_resolved_count": 2 if candidate_ok else 1,
            "episodes": {"e1": _episode(True), "e2": _episode(candidate_ok)},
        }
    return {"cells": cells}


def test_starts_at_r4_after_control() -> None:
    assert recommend(_state(), "independent-n2-a")["strategy_config_id"] == "r4m2v2_tasks1"


def test_r4_inside_target_stops() -> None:
    decision = recommend(_state(600), "independent-n2-a")
    assert decision["decision"] == "target_found"
    assert decision["point"]["failure_aware_saving_vs_persistent_full"] == 0.4


def test_low_saving_retires_more() -> None:
    decision = recommend(_state(800), "independent-n2-a")
    assert decision["strategy_config_id"] == "r2m2v2_tasks1"


def test_high_saving_retains_more() -> None:
    decision = recommend(_state(400), "independent-n2-a")
    assert decision["strategy_config_id"] == "r6m2v2_tasks1"


def test_lost_success_is_charged_zero_and_retains_more() -> None:
    decision = recommend(_state(600, candidate_ok=False), "independent-n2-a")
    assert decision["strategy_config_id"] == "r6m2v2_tasks1"
    assert decision["parent_point"]["failure_aware_saving_vs_persistent_full"] == 0.0
