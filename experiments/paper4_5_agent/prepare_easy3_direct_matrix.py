"""Prepare the locked task-major Paper 4.5 direct-engine matrix.

This command writes configuration receipts only. It never launches
mini-swe-agent, Docker, grading, or model generation. An optional endpoint
probe is limited to GET health/model metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.paper4_5_agent.build_easy_cohorts import ids_digest


EXPECTED_ARMS = (
    ("no_pra", False, 1.0, "none"),
    ("pra_100_no_adaptor", True, 1.0, "none"),
    ("pra_100_frozen_adaptor", True, 1.0, "frozen_bundle"),
    ("pra_090_no_adaptor", True, 0.9, "none"),
    ("pra_090_frozen_adaptor", True, 0.9, "frozen_bundle"),
)
REQUIRED_RETENTION_FIELDS = {
    "recent_completed_turns",
    "recent_records_per_turn",
    "recent_source_turns",
    "recent_progress_turns",
    "recent_mutation_turns",
    "recent_verification_turns",
    "large_record_chunk_tokens",
    "max_records_per_turn_before_chunking",
    "preserve_action_observation_pairs",
    "causal_bundle_round_up",
}
INCREMENTAL_ARTIFACTS = (
    "cell_manifest.json",
    "run_manifest.json",
    "request_telemetry.jsonl",
    "interaction_history.jsonl",
    "selection_fixture.jsonl",
    "results.jsonl",
    "official_result.json",
    "divergence.json",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _git_contains(root: Path, ancestor: str, head: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, head],
        cwd=root, capture_output=True, check=False,
    ).returncode == 0


def _validate_adaptor(root: Path, value: Mapping[str, Any]) -> list[str]:
    required = (
        "bundle_id", "bundle_revision", "path", "weights_sha256",
        "parameter_precision", "insertion_points",
    )
    missing = [name for name in required if not value.get(name)]
    if missing:
        return [
            "frozen Qwen3-Coder-30B adaptor is not immutable: missing "
            + ", ".join(missing)
        ]
    path = root / str(value["path"])
    if not path.is_file():
        return [f"frozen adaptor does not exist: {path}"]
    if _sha256(path) != str(value["weights_sha256"]):
        return ["frozen adaptor weights_sha256 does not match its file"]
    return []


def _validate_adaptor_contract(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("semantic_role") != "learned_routing_selection":
        errors.append("adaptor semantic role must be learned_routing_selection")
    if value.get("applies_at") != (
        "causal-record scoring before K/V subset materialization"
    ):
        errors.append("adaptor injection stage is not the locked selector stage")
    if value.get("certified_qwen3_coder_30b_memory_or_late_band_training_path") is not False:
        errors.append("uncertified memory/late-band path may not enter this matrix")
    for name in (
        "required_training_provenance", "required_evaluation_provenance",
        "required_injection_contract", "explicitly_not",
    ):
        if not value.get(name):
            errors.append(f"adaptor qualification omits {name}")
    return errors


def validate_config(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Fail on drift in cohort, order, factors, or engine qualification."""

    errors: list[str] = []
    card_path = root / str(config["benchmark_card"])
    card = _load(card_path)
    locked = [str(value) for value in config["locked_task_ids"]]
    if locked != list(card.get("instance_ids", ()))[:3]:
        errors.append("locked task IDs are not the first three baseline-success IDs")
    if ids_digest(locked) != config.get("locked_task_ids_sha256"):
        errors.append("locked task ID digest mismatch")
    observed_arms = tuple(
        (
            str(row.get("label")), bool(row.get("pra")),
            float(row.get("retention_fraction")), str(row.get("adaptor")),
        )
        for row in config.get("arms", ())
    )
    if observed_arms != EXPECTED_ARMS:
        errors.append("arm order or factor values differ from the locked five-arm matrix")
    execution = config.get("execution", {})
    if execution.get("gateway") is not False:
        errors.append("gateway must remain disabled")
    if execution.get("temperature") != 0 or execution.get("seed") != 0:
        errors.append("the matrix requires temperature=0 and seed=0")
    retention = config.get("retention_policy", {})
    if set(retention) != REQUIRED_RETENTION_FIELDS:
        errors.append("retention policy fields are incomplete or undeclared")
    errors.extend(_validate_adaptor_contract(config.get("adaptor_qualification") or {}))
    gate_path = root / str(execution.get("engine_gate"))
    gate = _load(gate_path)
    engine_name = str(execution.get("engine"))
    engine_gate = (gate.get("engine_qualification") or {}).get(engine_name) or {}
    engine_qualified = (
        gate.get("status") == "QUALIFIED"
        and engine_gate.get("status") == "QUALIFIED"
        and engine_gate.get("agent_campaign") == "enabled"
    )
    if not engine_qualified:
        errors.append(f"{engine_name} agent-history gate is not explicitly qualified")
    head = _git_head(root)
    source_commit = str(config.get("source_commit"))
    if head is None or not _git_contains(root, source_commit, head):
        errors.append(
            f"checkout HEAD {head!r} does not contain locked source {source_commit}"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "engine_qualified": engine_qualified,
        "engine_gate_status": engine_gate.get("status"),
        "engine_gate_sha256": _sha256(gate_path),
        "benchmark_card_sha256": _sha256(card_path),
        "git_head": head,
        "adaptor_blockers": _validate_adaptor(
            root, config.get("frozen_adaptor") or {},
        ),
        "adaptor_qualification": dict(config.get("adaptor_qualification") or {}),
    }


def _probe_endpoint(url: str) -> dict[str, Any]:
    """Read endpoint metadata only; never submit a generation request."""

    base = url.rstrip("/").removesuffix("/v1")
    observations: dict[str, Any] = {}
    for name, suffix in (("health", "/health"), ("models", "/v1/models")):
        try:
            with urllib.request.urlopen(base + suffix, timeout=10) as response:
                observations[name] = {
                    "status": response.status,
                    "payload": json.load(response),
                }
        except (OSError, urllib.error.URLError, ValueError) as error:
            observations[name] = {
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
            }
    observations["generation_requests_sent"] = 0
    return observations


def _task_directory(config: Mapping[str, Any], task_index: int, task_id: str) -> Path:
    return Path(str(config["output_root"])) / f"task_{task_index:02d}_{task_id}"


def _runner_argv(
    config: Mapping[str, Any], task_index: int, arm: Mapping[str, Any],
    cell_output: Path, endpoint: str | None,
) -> list[str] | None:
    if arm["adaptor"] == "frozen_bundle":
        return None
    execution = config["execution"]
    argv = [
        "python", "-m", "experiments.paper4_5_agent.runners.local_qwen_swebench",
        "--benchmark-card", str(config["benchmark_card"]),
        "--task-index", str(task_index),
        "--output", cell_output.as_posix(),
        "--run-id", f"easy3-t{task_index:02d}-{arm['label']}-seed0",
        "--base-url", endpoint or f"${{{execution['endpoint_environment']}}}",
        "--engine", str(execution["engine"]),
        "--engine-version", str(execution["engine_version"]),
        "--max-completion-tokens", str(execution["max_completion_tokens"]),
        "--sampling-seed", str(execution["seed"]),
        "--top-p", str(execution["top_p"]),
        "--prefix-caching", "--require-endpoint-preflight",
    ]
    if arm["pra"]:
        retention = config["retention_policy"]
        argv.extend([
            "--mode", "direct-native-pra",
            "--budget-fraction", str(arm["retention_fraction"]),
            "--recent-completed-turns", str(retention["recent_completed_turns"]),
            "--recent-records-per-turn", str(retention["recent_records_per_turn"]),
            "--recent-source-turns", str(retention["recent_source_turns"]),
            "--recent-progress-turns", str(retention["recent_progress_turns"]),
            "--recent-mutation-turns", str(retention["recent_mutation_turns"]),
            "--recent-verification-turns", str(retention["recent_verification_turns"]),
            "--large-record-chunk-tokens", str(retention["large_record_chunk_tokens"]),
            "--max-records-per-turn-before-chunking",
            str(retention["max_records_per_turn_before_chunking"]),
            "--preserve-action-observation-pairs", "--causal-bundle-round-up",
            "--selection-record", (cell_output / "selection_fixture.jsonl").as_posix(),
        ])
    return argv


def build_plan(
    config: Mapping[str, Any], root: Path, *, probe_endpoint: bool = False,
) -> dict[str, Any]:
    qualification = validate_config(config, root)
    endpoint_name = str(config["execution"]["endpoint_environment"])
    endpoint = os.environ.get(endpoint_name)
    endpoint_observation = (
        _probe_endpoint(endpoint) if probe_endpoint and endpoint else {
            "status": "not_probed",
            "reason": "endpoint environment is unset" if not endpoint else "probe disabled",
            "generation_requests_sent": 0,
        }
    )
    cells: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(config["locked_task_ids"], 1):
        task_dir = _task_directory(config, task_index, str(task_id))
        for arm in config["arms"]:
            cell_output = task_dir / "locked_easy3_v1" / str(arm["label"])
            blockers = list(qualification["errors"])
            if arm["adaptor"] == "frozen_bundle":
                blockers.extend(qualification["adaptor_blockers"])
                blockers.append(
                    "runner has no qualified Qwen3-Coder-30B frozen-adaptor injection path"
                )
            if not endpoint:
                blockers.append(f"{endpoint_name} is unset on the execution host")
            run_argv = _runner_argv(config, task_index, arm, cell_output, endpoint)
            cells.append({
                "cell_index": len(cells) + 1,
                "cell_id": f"task-{task_index:02d}-{arm['label']}",
                "task_index": task_index,
                "task_id": task_id,
                "arm": dict(arm),
                "locked_source_commit": config["source_commit"],
                "actual_git_head": qualification["git_head"],
                "seed": config["execution"]["seed"],
                "temperature": config["execution"]["temperature"],
                "model": config["execution"]["model"],
                "model_revision": config["execution"]["model_revision"],
                "selection_policy": config["execution"]["selection_policy"],
                "materialization_policy": config["execution"]["materialization_policy"],
                "retention_policy": dict(config["retention_policy"]) if arm["pra"] else None,
                "output_directory": cell_output.as_posix(),
                "expected_incremental_artifacts": list(INCREMENTAL_ARTIFACTS),
                "run_argv": run_argv,
                "qualification_argv": [*run_argv, "--preflight-only"] if run_argv else None,
                "status": "BLOCKED" if blockers else "CONFIG_QUALIFIED",
                "blockers": list(dict.fromkeys(blockers)),
            })
    return {
        "schema_version": config["schema_version"],
        "status": "CONFIG_QUALIFIED" if all(not row["blockers"] for row in cells) else "PARTIALLY_BLOCKED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "configured_agent_target": config["execution"]["agent_host"],
            "configured_model_engine_target": config["execution"][
                "model_engine_host"
            ],
        },
        "qualification": qualification,
        "locked_source_commit": config["source_commit"],
        "actual_git_head": qualification["git_head"],
        "endpoint_environment": endpoint_name,
        "endpoint_observation": endpoint_observation,
        "task_order": list(config["locked_task_ids"]),
        "arm_order": [row["label"] for row in config["arms"]],
        "cell_order": "task-major",
        "expected_cells": 15,
        "expensive_task_runs_started": 0,
        "metrics": config["metrics"],
        "cells": cells,
    }


def write_plan(plan: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for cell in plan["cells"]:
        cell_path = destination.parent / "cells" / str(cell["cell_id"]) / "cell_manifest.json"
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        cell_path.write_text(json.dumps(cell, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/paper4_5_agent/configs/campaigns/swebench_easy3_direct_matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-endpoint", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = _load(root / args.config)
    plan = build_plan(config, root, probe_endpoint=args.probe_endpoint)
    write_plan(plan, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
