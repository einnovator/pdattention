from experiments.paper8_5_agent_memory.persistent_spine_gate import evaluate


def _cell(*, tokens: int, calls: int, second_resolved: bool = True) -> dict:
    return {
        "status": "complete", "cumulative_full_tokens": 1000,
        "cumulative_materialized_tokens": tokens, "calls": calls,
        "official_resolved_count": 1 + int(second_resolved),
        "episodes": {
            "e1": {"official_resolved": True},
            "e2": {"official_resolved": second_resolved},
        },
    }


def _state(candidate_tokens: int = 600, candidate_calls: int = 8) -> dict:
    cells = {}
    for repeat in (1, 2):
        cells[f"seq__S01_persistent_full__r{repeat:02d}"] = _cell(
            tokens=1000, calls=10
        )
        cells[
            f"seq__S03_completed_episode_spine-r4m2v2_tasks1__r{repeat:02d}"
        ] = _cell(tokens=candidate_tokens, calls=candidate_calls)
    return {"cells": cells}


def test_repeat_gate_advances_lossless_target_point() -> None:
    result = evaluate(
        _state(), sequence_id="seq", strategy_config_id="r4m2v2_tasks1"
    )
    assert result["decision"] == "advance"
    assert result["aggregate_failure_aware_saving_vs_persistent_full"] == 0.4
    assert result["calls_delta_vs_persistent_full"] == -4


def test_repeat_gate_never_offsets_a_lost_full_success() -> None:
    state = _state(candidate_tokens=500)
    state["cells"][
        "seq__S03_completed_episode_spine-r4m2v2_tasks1__r02"
    ]["episodes"]["e2"]["official_resolved"] = False
    state["cells"][
        "seq__S03_completed_episode_spine-r4m2v2_tasks1__r02"
    ]["official_resolved_count"] = 1
    result = evaluate(
        state, sequence_id="seq", strategy_config_id="r4m2v2_tasks1"
    )
    assert result["decision"] == "stop_or_rebracket"
    assert result["lost_persistent_full_successes"] == 1
    assert result["aggregate_failure_aware_saving_vs_persistent_full"] == 0.0
