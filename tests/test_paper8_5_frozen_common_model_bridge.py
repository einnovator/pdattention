from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/paper8_5_agent_memory/configs/frozen_common_model_bridge_v1.json"
BENCHMARK = ROOT / "experiments/paper8_5_agent_memory/benchmarks/common_django3_v1.json"


def test_common_model_bridge_policy_and_task_identity_are_frozen() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    tasks = payload["task_cohort"]["ordered_instance_ids"]
    digest = hashlib.sha256(("\n".join(tasks) + "\n").encode()).hexdigest()
    assert digest == payload["task_cohort"]["canonical_ids_sha256"]
    assert tasks == benchmark["instance_ids"]
    assert digest == benchmark["canonical_ids_sha256"]
    assert payload["model"]["revision"] == payload["model"]["tokenizer_revision"]
    assert payload["generation"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_completion_tokens": 1024,
    }
    policies = {row["policy_id"]: row for row in payload["policies"]}
    assert set(policies) == {
        "TAIL90_CAUSAL_V1",
        "E0_BOUNDARY_FREE_ACTIVE_EPOCH_V1",
    }
    assert policies["E0_BOUNDARY_FREE_ACTIVE_EPOCH_V1"]["keep_completed_task_statements"]
    assert not payload["canonical_record_contract"]["agent_id_visible_to_policy"]
    assert not payload["canonical_record_contract"]["task_id_visible_to_policy"]
