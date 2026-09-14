import json
from pathlib import Path

from experiments.paper8_5_agent_memory.run_persistent_spine_12h_plan import (
    _complete,
    _selected_cells,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = (
    ROOT / "experiments" / "paper8_5_agent_memory" / "configs" /
    "autonomous_persistent_spine_confirmation_v1.json"
)


def test_plan_selects_both_r4_repeats_without_neighbor_collision() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    cells = _selected_cells(
        spec, sequence_id="independent-n2-a",
        strategy_id="S03_completed_episode_spine",
        strategy_config_id="r4m2v2_tasks1",
    )
    assert [row["repeat"] for row in cells] == [1, 2]
    state = {"cells": {row["cell_id"]: {"status": "complete"} for row in cells}}
    assert _complete(state, cells)
    state["cells"][cells[1]["cell_id"]]["status"] = "infrastructure_error"
    assert not _complete(state, cells)
