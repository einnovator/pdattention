"""Freeze the common agent/model/task matrix before engine execution.

This is deliberately a preparation step, not a selector.  Paper 8.5 owns the
logical policy definitions.  Paper 4.5 imports their immutable contract and
creates task-major engine cells whose treatment arms are dependency-gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .benchmark import load_benchmark_card


DEFAULT_CONTRACT = Path(
    "experiments/paper4_5_agent/configs/pra/paper8_5_frozen_common_model_bridge_v1.json"
)
DEFAULT_BENCHMARK = Path(
    "experiments/paper4_5_agent/benchmarks/swebench_verified_common_django3.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy_by_id(contract: Mapping[str, Any], policy_id: str) -> Mapping[str, Any]:
    rows = [row for row in contract.get("policies", ()) if row.get("policy_id") == policy_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one frozen policy {policy_id!r}")
    return rows[0]


def validate_contract(contract: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "paper8.5.frozen-common-model-bridge.v1":
        raise ValueError("unexpected common-model bridge schema")
    model = contract.get("model") or {}
    if model.get("id") != "Qwen/Qwen2.5-Coder-7B-Instruct":
        raise ValueError("common bridge must use the frozen Qwen2.5-Coder-7B model")
    if model.get("revision") != model.get("tokenizer_revision"):
        raise ValueError("model and tokenizer revisions must match")
    if model.get("precision") != "BF16":
        raise ValueError("common bridge precision must remain BF16")
    frozen_tasks = contract.get("task_cohort") or {}
    if list(frozen_tasks.get("ordered_instance_ids") or ()) != list(card["instance_ids"]):
        raise ValueError("benchmark card and frozen policy contract use different tasks")
    if frozen_tasks.get("canonical_ids_sha256") != card.get("canonical_ids_sha256"):
        raise ValueError("benchmark card and frozen policy contract use different digests")

    tail = _policy_by_id(contract, "TAIL90_CAUSAL_V1")
    if float(tail.get("budget_fraction", 0.0)) != 0.90:
        raise ValueError("TAIL90 budget is not frozen at 90 percent")
    native_rule = str(tail.get("native_engine_rule", ""))
    if "never encode" not in native_rule or "resident K/V" not in native_rule:
        raise ValueError("TAIL90 lacks the strict resident-K/V materialization rule")

    epoch = _policy_by_id(contract, "E0_BOUNDARY_FREE_ACTIVE_EPOCH_V1")
    if epoch.get("session_mode") != "persistent":
        raise ValueError("E0 must remain a persistent-session policy")
    if epoch.get("keep_completed_task_statements") is not True:
        raise ValueError("E0 must preserve every genuine user instruction")
    forbidden = ("task ID", "episode marker", "agent ID")
    boundary_rule = str(epoch.get("boundary_rule", ""))
    if not all(value in boundary_rule for value in forbidden):
        raise ValueError("E0 must explicitly reject hidden task/agent boundaries")


def prepare_matrix(contract_path: Path, benchmark_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    card = load_benchmark_card(benchmark_path)
    validate_contract(contract, card)
    engines = list((contract.get("engines") or {}).get("primary_order") or ())
    engines += list((contract.get("engines") or {}).get("secondary_order") or ())
    tasks = list(card["instance_ids"])
    cells: list[dict[str, Any]] = []
    for engine in engines:
        for task_index, instance_id in enumerate(tasks, 1):
            stem = f"{engine}:{task_index}:{instance_id}"
            plain_id = f"{stem}:plain"
            pra100_id = f"{stem}:pra100"
            cells.extend((
                {
                    "cell_id": plain_id,
                    "engine": engine,
                    "agent": "mini-swe-agent@2.4.6",
                    "task_index": task_index,
                    "instance_id": instance_id,
                    "arm": "plain",
                    "requires": [],
                    "admission_gate": "official_resolution",
                },
                {
                    "cell_id": pra100_id,
                    "engine": engine,
                    "agent": "mini-swe-agent@2.4.6",
                    "task_index": task_index,
                    "instance_id": instance_id,
                    "arm": "pra100",
                    "requires": [plain_id],
                    "admission_gate": "plain_official_resolution",
                },
                {
                    "cell_id": f"{stem}:tail90",
                    "engine": engine,
                    "agent": "mini-swe-agent@2.4.6",
                    "task_index": task_index,
                    "instance_id": instance_id,
                    "arm": "TAIL90_CAUSAL_V1",
                    "requires": [pra100_id],
                    "admission_gate": "pra100_semantic_position_physical_lifecycle",
                },
            ))
        cells.append({
            "cell_id": f"{engine}:persistent-n3:e0",
            "engine": engine,
            "agent": "mini-swe-agent@2.4.6",
            "instance_ids": tasks,
            "session_mode": "persistent",
            "arm": "E0_BOUNDARY_FREE_ACTIVE_EPOCH_V1",
            "requires": [
                f"{engine}:{index}:{instance_id}:pra100"
                for index, instance_id in enumerate(tasks, 1)
            ],
            "admission_gate": "all_three_pra100_gates",
        })
    return {
        "schema_version": "paper4.5.common-model-bridge-matrix.v1",
        "contract_path": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "model": contract["model"],
        "generation": contract["generation"],
        "logical_ledger_rule": contract["engines"]["logical_ledger_rule"],
        "required_engine_metrics": [
            "selected_history_reencoded_tokens",
            "selected_kv_reference_bytes",
            "selected_kv_copy_bytes",
            "total_kv_copy_bytes",
            "consumer_temporary_bytes",
            "consumer_temporary_peak_bytes",
            "new_suffix_encoded_tokens",
            "resident_kv_bytes",
            "ttft_ms",
            "total_latency_ms",
            "calls_to_solution",
            "official_resolution",
        ],
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_matrix(args.contract, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cells": len(result["cells"]),
        "contract_sha256": result["contract_sha256"],
    }))


if __name__ == "__main__":
    main()
