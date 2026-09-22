from __future__ import annotations

import json
from pathlib import Path

from experiments.paper4_5_agent.prepare_common_model_bridge import prepare_matrix


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/paper4_5_agent/configs/pra/paper8_5_frozen_common_model_bridge_v1.json"
BENCHMARK = ROOT / "experiments/paper4_5_agent/benchmarks/swebench_verified_common_django3.json"


def test_common_model_bridge_is_frozen_and_dependency_gated() -> None:
    result = prepare_matrix(CONTRACT, BENCHMARK)
    assert result["model"]["id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert result["model"]["precision"] == "BF16"
    assert len(result["cells"]) == 50
    by_id = {row["cell_id"]: row for row in result["cells"]}
    tail = by_id["mlx:1:django__django-15277:tail90"]
    assert tail["requires"] == ["mlx:1:django__django-15277:pra100"]
    assert tail["admission_gate"] == "pra100_semantic_position_physical_lifecycle"
    persistent = by_id["mlx:persistent-n3:e0"]
    assert len(persistent["requires"]) == 3
    assert persistent["session_mode"] == "persistent"


def test_tail90_native_rule_forbids_post_cache_reencoding() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tail = next(row for row in contract["policies"] if row["policy_id"] == "TAIL90_CAUSAL_V1")
    assert "never encode" in tail["native_engine_rule"]
    assert "original resident K/V span" in tail["native_engine_rule"]
    assert tail["native_materialization_class"].startswith("strict_resident_subset")
