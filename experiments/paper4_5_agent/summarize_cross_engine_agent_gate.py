"""Aggregate the frozen Paper 4.5 one-task-per-engine release gate.

This reducer keeps behavioral qualification separate from measurement
completeness.  Easy-14 is authorized only when every required engine has an
admissible plain run, exact PRA-100 behavior, a completed PRA-90 treatment,
and all four predeclared outcome families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_ENGINES = ("llama.cpp", "huggingface", "mlx", "sglang-mlx", "vllm")


def _measurement_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    plain = payload.get("plain") or {}
    pra100 = payload.get("pra_100") or {}
    pra90 = payload.get("pra_90") or {}
    gates = payload.get("gates") or {}
    return {
        "task_success_and_exact_100_trajectory": bool(
            plain.get("task_success") is True
            and pra100.get("task_success") is True
            and gates.get("pra_100_exact_action_trajectory") is True
        ),
        "realized_retention_at_90": bool(
            pra90.get("realized_retention_fraction") is not None
        ),
        "reencoding_and_kv_copy_bytes": bool(
            pra100.get("selected_history_reencoded_tokens") is not None
            and pra100.get("physical_kv_copy_bytes") is not None
            and pra100.get("total_kv_copy_bytes") is not None
            and pra90.get("selected_history_reencoded_tokens") is not None
            and pra90.get("physical_kv_copy_bytes") is not None
            and pra90.get("total_kv_copy_bytes") is not None
        ),
        "consumer_temporary_bytes_and_calls": bool(
            pra100.get("consumer_temporary_bytes") is not None
            and pra100.get("consumer_temporary_peak_bytes") is not None
            and pra90.get("consumer_temporary_bytes") is not None
            and pra90.get("consumer_temporary_peak_bytes") is not None
            and plain.get("calls_to_solution") is not None
            and pra100.get("calls_to_solution") is not None
            and pra90.get("calls_to_solution") is not None
        ),
    }


def _admission_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcome = payload.get("outcome") or {}
    plain = payload.get("plain_admission") or (
        (payload.get("conditions") or {}).get("plain") or {}
    ) or payload.get("frozen_task01_plain") or {}
    if not plain and "plain-admission" in str(payload.get("schema_version") or ""):
        plain = payload
    template = payload.get("chat_template") or {}
    resolved = plain.get("resolved")
    if resolved is None:
        resolved = plain.get("official_solve", plain.get("task_success"))
    if resolved is None:
        resolved = outcome.get(
            "plain_model_task_pair_admitted", outcome.get("official_resolved")
        )
    status = " ".join(
        str(value)
        for value in (
            plain.get("status"),
            plain.get("classification"),
            plain.get("failure_type"),
            plain.get("failure_stage"),
            outcome.get("reason"),
            outcome.get("exit_status"),
            payload.get("classification"),
        )
        if value
    )
    if resolved is None and status:
        resolved = (
            False
            if (
                "fail" in status
                or "no_progress" in status
                or "hardware_inadmissible" in status
            )
            else None
        )
    task = payload.get("task", payload.get("instance_id"))
    task_id = task.get("instance_id") if isinstance(task, Mapping) else task
    if task_id is None:
        task_id = plain.get("instance_id")
    model_calls = plain.get("model_calls", plain.get("calls_to_solution"))
    if model_calls is None:
        controls = plain.get("interface_controls") or ()
        model_calls = max(
            (
                int(row.get("completed_api_calls", row.get("api_calls", 0)))
                for row in controls
                if isinstance(row, Mapping)
            ),
            default=None,
        )
    if model_calls is None:
        model_calls = plain.get(
            "accepted_tool_actions",
            plain.get(
                "completed_tool_calls",
                plain.get("agent_actions_completed", plain.get("tool_calls_completed")),
            ),
        )
    if model_calls is None:
        model_calls = outcome.get("api_calls")
    termination = plain.get(
        "termination_reason", plain.get(
            "termination", plain.get(
                "failure_type", plain.get("failure_stage", outcome.get("exit_status"))
            )
        ),
    )
    pra100 = payload.get("pra_100") or (
        (payload.get("conditions") or {}).get("pra_100") or {}
    )
    pra90 = payload.get("pra_090") or payload.get("pra_90") or (
        (payload.get("conditions") or {}).get("pra_90") or {}
    )
    append_stable = (
        bool(
            template.get("two_turn_byte_prefix_stable") is True
            and template.get("two_turn_token_prefix_stable") is True
        )
        if template
        else None
    )
    capacity_failure = any(
        marker in status.lower()
        for marker in (
            "cuda_out_of_memory",
            "out of memory",
            "capacity",
            "memory",
        )
    )
    throughput_failure = any(
        marker in status.lower()
        for marker in (
            "hardware-throughput",
            "hardware throughput",
            "throughput",
        )
    )
    hardware_failure = capacity_failure or throughput_failure or any(
        marker in status.lower()
        for marker in (
            "hardware_inadmissible",
        )
    )
    return {
        "status": (
            "admission_succeeded_gate_pending"
            if resolved is True
            else (
                "admission_failed_hardware_capacity"
                if capacity_failure
                else "admission_failed_hardware_throughput"
                if throughput_failure or hardware_failure
                else "admission_failed_model_task_pair"
            )
        ),
        "artifact": path.as_posix(),
        "classification": (
            "plain_model_task_admitted_gate_pending"
            if resolved is True
            else (
                "plain_hardware_capacity_admission_failure_not_pra_failure"
                if capacity_failure
                else "plain_hardware_throughput_admission_failure_not_pra_failure"
                if throughput_failure or hardware_failure
                else "plain_model_task_admission_failure_not_pra_failure"
            )
        ),
        "behavioral_gate_complete": False,
        "measurement_gate_complete": False,
        "admission": {
            "model": payload.get("model"),
            "task": task_id,
            "plain_official_solve": resolved,
            "model_calls": model_calls,
            "termination_reason": termination,
            "append_stable_template_qualified": append_stable,
            "pra_100_run": bool(
                payload.get("pra_100_executed") is True
                or (
                    pra100.get("status")
                    and not str(pra100["status"]).startswith("not_run")
                )
            ),
            "pra_90_run": bool(
                payload.get("pra_90_executed") is True
                or (
                    pra90.get("status")
                    and not str(pra90["status"]).startswith("not_run")
                )
            ),
        },
    }


def summarize(
    gates_by_engine: Mapping[str, Path | None],
    admissions_by_engine: Mapping[str, Path | None] | None = None,
) -> dict[str, Any]:
    admissions_by_engine = admissions_by_engine or {}
    engines: dict[str, Any] = {}
    behavioral_complete = 0
    measurement_complete = 0
    for engine in REQUIRED_ENGINES:
        path = gates_by_engine.get(engine)
        if path is None or not path.is_file():
            admission_path = admissions_by_engine.get(engine)
            if admission_path is not None and admission_path.is_file():
                engines[engine] = _admission_summary(admission_path)
                continue
            engines[engine] = {
                "status": "pending",
                "artifact": None if path is None else path.as_posix(),
                "behavioral_gate_complete": False,
                "measurement_gate_complete": False,
            }
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = _measurement_checks(payload)
        behavior = bool((payload.get("gates") or {}).get("engine_task_gate_complete"))
        measured = behavior and all(checks.values())
        behavioral_complete += int(behavior)
        measurement_complete += int(measured)
        engines[engine] = {
            "status": "complete" if measured else (
                "behavioral_complete_measurement_pending" if behavior else "incomplete"
            ),
            "artifact": path.as_posix(),
            "classification": payload.get("classification"),
            "behavioral_gate_complete": behavior,
            "measurement_gate_complete": measured,
            "outcome_family_checks": checks,
        }

    required = len(REQUIRED_ENGINES)
    release = behavioral_complete == required and measurement_complete == required
    return {
        "schema_version": "paper4.5.cross-engine-agent-gate.v1",
        "required_engines": list(REQUIRED_ENGINES),
        "engines": engines,
        "behavioral_complete_engines": behavioral_complete,
        "measurement_complete_engines": measurement_complete,
        "required_engine_count": required,
        "cross_engine_agent_gate_complete": release,
        "easy14_expansion_allowed": release,
        "claim_boundary": (
            "A PRA-100 action divergence is an implementation failure. A PRA-90 "
            "divergence is a retention-quality result only after exact PRA-100 "
            "behavior. Missing telemetry is reported as missing, never as zero."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", action="append", default=[], metavar="ENGINE=PATH",
        help="Per-engine frozen-task gate artifact (repeatable)",
    )
    parser.add_argument(
        "--admission", action="append", default=[], metavar="ENGINE=PATH",
        help=(
            "Plain model/task admission artifact used when a completed engine "
            "gate does not yet exist (repeatable)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths: dict[str, Path] = {}
    for value in args.gate:
        engine, separator, path = value.partition("=")
        if not separator or engine not in REQUIRED_ENGINES:
            parser.error(f"invalid --gate {value!r}; expected one of {REQUIRED_ENGINES}")
        paths[engine] = Path(path)
    admissions: dict[str, Path] = {}
    for value in args.admission:
        engine, separator, path = value.partition("=")
        if not separator or engine not in REQUIRED_ENGINES:
            parser.error(
                f"invalid --admission {value!r}; expected one of {REQUIRED_ENGINES}"
            )
        admissions[engine] = Path(path)
    payload = summarize(paths, admissions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "easy14_expansion_allowed": payload["easy14_expansion_allowed"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
