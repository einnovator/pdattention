"""Reduce one frozen mini-swe-agent task into the Paper 4.5 100% -> 90% gate.

The reducer deliberately keeps task outcome, action parity, retention, K/V
transport, and consumer allocation separate.  A PRA-90 result is admissible
only after the matching plain and PRA-100 runs both solve and their ordered
assistant action trajectories are byte-exact after removing provider metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_trajectory(run_dir: Path, instance_id: str) -> Path | None:
    matches = sorted(run_dir.rglob(f"{instance_id}.traj.json"))
    return matches[0] if matches else None


def _action_trajectory(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = []
    for message in payload.get("messages") or ():
        if message.get("role") != "assistant":
            continue
        extra = message.get("extra") or {}
        if not extra.get("response"):
            continue
        trajectory.append({
            "content": str(message.get("content") or ""),
            "actions": extra.get("actions") or [],
        })
    return trajectory


def _submission(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str((payload.get("info") or {}).get("submission") or "")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_action_divergence(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]],
) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate), start=1):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate)) + 1
    return None


def _run_summary(run_dir: Path | None, instance_id: str) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    rows = _jsonl(run_dir / "results.jsonl")
    matches = [row for row in rows if str(row.get("instance_id")) == instance_id]
    if len(matches) != 1:
        return {
            "status": "pending" if not matches else "invalid",
            "reason": f"expected one result row for {instance_id}; found {len(matches)}",
            "run_directory": run_dir.as_posix(),
        }
    row = matches[0]
    trajectory_path = _find_trajectory(run_dir, instance_id)
    if trajectory_path is None:
        return {
            "status": "invalid",
            "reason": "result exists but trajectory is missing",
            "run_directory": run_dir.as_posix(),
        }
    actions = _action_trajectory(trajectory_path)
    telemetry = _jsonl(run_dir / "request_telemetry.jsonl")
    manifest_path = run_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {}
    )
    endpoint = manifest.get("gateway_preflight") or {}
    submission = _submission(trajectory_path)
    logical = sum(int(item.get("logical_input_tokens_estimate") or 0) for item in telemetry)
    physical = sum(int(item.get("physical_input_tokens_estimate") or 0) for item in telemetry)
    context_retention = [
        int(item.get("physical_input_tokens_estimate") or 0)
        / int(item.get("logical_input_tokens_estimate") or 1)
        for item in telemetry
        if int(item.get("logical_input_tokens_estimate") or 0) > 0
    ]
    history_retention = [
        float(item["realized_retention_fraction"])
        for item in telemetry if item.get("realized_retention_fraction") is not None
    ]
    return {
        "status": "complete",
        "run_directory": run_dir.as_posix(),
        "run_id": row.get("run_id"),
        "chat_template_profile": endpoint.get("chat_template_profile"),
        "chat_template_digest": endpoint.get("chat_template_digest"),
        "mode": row.get("mode"),
        "task_success": bool(row.get("resolved")),
        "grader_outcome": row.get("grader_outcome"),
        "termination_reason": row.get("termination_reason"),
        "calls_to_solution": row.get("model_call_count"),
        "tool_calls": row.get("tool_call_count"),
        "action_trajectory_length": len(actions),
        "action_trajectory_sha256": _digest(actions),
        "action_trajectory": actions,
        "patch_sha256": hashlib.sha256(submission.encode("utf-8")).hexdigest(),
        "patch_bytes": len(submission.encode("utf-8")),
        "engine_version": manifest.get("engine_version"),
        "engine_runtime_identity": endpoint.get("runtime_identity"),
        "exact_environment": manifest.get("exact_environment"),
        "configuration_differences": manifest.get("configuration_differences"),
        "requested_retention_fraction": row.get("context_budget_fraction"),
        "realized_retention_fraction": (
            physical / logical if logical else row.get("realized_retention_fraction")
        ),
        "realized_retention_fraction_min": (
            min(context_retention) if context_retention
            else row.get("realized_retention_fraction_min")
        ),
        "realized_retention_fraction_max": (
            max(context_retention) if context_retention
            else row.get("realized_retention_fraction_max")
        ),
        "engine_reported_history_kv_retention_fraction_min": (
            min(history_retention) if history_retention else None
        ),
        "engine_reported_history_kv_retention_fraction_max": (
            max(history_retention) if history_retention else None
        ),
        "selected_history_reencoded_tokens": (
            sum(int(item.get("selected_history_reencoded_tokens") or 0) for item in telemetry)
            if telemetry else row.get(
                "selected_history_reencoded_tokens",
                row.get("selected_text_reencoded_tokens"),
            )
        ),
        "physical_kv_copy_bytes": (
            (
                sum(int(item.get("physical_kv_copy_bytes") or 0) for item in telemetry)
                if any(item.get("physical_kv_copy_bytes") is not None for item in telemetry)
                else None
            )
            if telemetry else row.get("physical_kv_copy_bytes")
        ),
        "total_kv_copy_bytes": (
            (
                sum(int(item.get("total_kv_copy_bytes") or 0) for item in telemetry)
                if any(item.get("total_kv_copy_bytes") is not None for item in telemetry)
                else None
            )
            if telemetry else row.get("total_kv_copy_bytes")
        ),
        "canonical_suffix_graft_d2d_bytes": (
            (
                sum(
                    int(item.get("canonical_suffix_graft_d2d_bytes") or 0)
                    for item in telemetry
                )
                if any(
                    item.get("canonical_suffix_graft_d2d_bytes") is not None
                    for item in telemetry
                )
                else None
            )
            if telemetry else row.get("canonical_suffix_graft_d2d_bytes")
        ),
        "host_to_device_bytes": (
            (
                sum(int(item.get("host_to_device_bytes") or 0) for item in telemetry)
                if any(item.get("host_to_device_bytes") is not None for item in telemetry)
                else None
            )
            if telemetry else row.get("host_to_device_bytes")
        ),
        "consumer_temporary_bytes": (
            (
                sum(int(item.get("consumer_temporary_bytes") or 0) for item in telemetry)
                if any(item.get("consumer_temporary_bytes") is not None for item in telemetry)
                else None
            )
            if telemetry else row.get("consumer_temporary_bytes")
        ),
        "consumer_temporary_peak_bytes": (
            (
                max(int(item.get("consumer_temporary_peak_bytes") or 0) for item in telemetry)
                if any(item.get("consumer_temporary_peak_bytes") is not None for item in telemetry)
                else None
            )
            if telemetry else row.get("consumer_temporary_peak_bytes")
        ),
        "fused_attention_calls": (
            (
                sum(int(item.get("fused_attention_calls") or 0) for item in telemetry)
                if any(item.get("fused_attention_calls") is not None for item in telemetry)
                else None
            )
            if telemetry else row.get("fused_attention_calls")
        ),
        "logical_input_tokens_estimate": row.get("logical_input_tokens_estimate"),
        "physical_input_tokens_estimate": row.get("physical_input_tokens_estimate"),
        "wall_time_s": row.get("wall_time_s"),
        "trajectory_path": trajectory_path.as_posix(),
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
    }


def summarize(
    *, engine: str, instance_id: str, plain_dir: Path,
    pra100_dir: Path | None, pra90_dir: Path | None,
) -> dict[str, Any]:
    plain = _run_summary(plain_dir, instance_id)
    pra100 = _run_summary(pra100_dir, instance_id)
    pra90 = _run_summary(pra90_dir, instance_id)

    plain_complete = bool(plain and plain.get("status") == "complete")
    pra100_complete = bool(pra100 and pra100.get("status") == "complete")
    exact_100 = bool(
        plain_complete
        and pra100_complete
        and plain["action_trajectory_sha256"] == pra100["action_trajectory_sha256"]
    )
    exact_patch_100 = bool(
        plain_complete
        and pra100_complete
        and plain["patch_sha256"] == pra100["patch_sha256"]
    )
    plain_runtime_identity = (
        plain.get("engine_runtime_identity") if plain else None
    )
    pra100_runtime_identity = (
        pra100.get("engine_runtime_identity") if pra100 else None
    )
    runtime_identity_100_match = (
        None
        if plain_runtime_identity is None and pra100_runtime_identity is None
        else bool(
            plain_complete
            and pra100_complete
            and plain_runtime_identity
            and plain_runtime_identity == pra100_runtime_identity
        )
    )
    plain_template_digest = plain.get("chat_template_digest") if plain else None
    pra100_template_digest = pra100.get("chat_template_digest") if pra100 else None
    template_digest_100_match = (
        None
        if plain_template_digest is None and pra100_template_digest is None
        else bool(
            plain_complete
            and pra100_complete
            and plain_template_digest
            and plain_template_digest == pra100_template_digest
        )
    )
    parity_100 = bool(
        exact_100
        and exact_patch_100
        and runtime_identity_100_match is not False
        and template_digest_100_match is not False
        and plain.get("task_success") is True
        and pra100.get("task_success") is True
    )
    pra90_complete = bool(pra90 and pra90.get("status") == "complete")
    first_90_divergence = (
        _first_action_divergence(
            plain["action_trajectory"], pra90["action_trajectory"]
        )
        if plain_complete and pra90_complete else None
    )

    if plain_complete and plain.get("task_success") is not True:
        classification = "inadmissible_model_task_pair"
    elif pra100_complete and not parity_100:
        classification = "implementation_bug_at_100_percent"
    elif parity_100 and not pra90_complete:
        classification = "qualified_for_90_percent_execution"
    elif parity_100 and pra90_complete:
        classification = (
            "retention_quality_result_at_90_percent"
            if pra90.get("task_success") is not None
            else "invalid_90_percent_result"
        )
    else:
        classification = "pending"

    return {
        "schema_version": "paper4.5.frozen-agent-engine-gate.v1",
        "engine": engine,
        "instance_id": instance_id,
        "temperature": 0,
        "plain": plain,
        "pra_100": pra100,
        "pra_90": pra90,
        "gates": {
            "plain_task_success": (
                plain.get("task_success") if plain_complete else None
            ),
            "pra_100_task_success": (
                pra100.get("task_success") if pra100_complete else None
            ),
            "pra_100_exact_action_trajectory": exact_100 if pra100_complete else None,
            "pra_100_exact_patch": exact_patch_100 if pra100_complete else None,
            "pra_100_engine_runtime_identity_match": (
                runtime_identity_100_match if pra100_complete else None
            ),
            "pra_100_chat_template_digest_match": (
                template_digest_100_match if pra100_complete else None
            ),
            "pra_100_behavioral_parity": parity_100 if pra100_complete else None,
            "pra_90_execution_allowed": parity_100,
            "pra_90_first_action_divergence": first_90_divergence,
            "pra_90_patch_matches_plain": (
                bool(
                    plain_complete and pra90_complete
                    and plain["patch_sha256"] == pra90["patch_sha256"]
                )
                if pra90_complete else None
            ),
            "engine_task_gate_complete": bool(parity_100 and pra90_complete),
            # This reducer covers one engine only. The campaign-level reducer
            # must observe a completed gate for every qualified engine before
            # authorizing any Easy-14 expansion.
            "easy14_expansion_allowed": False,
            "easy14_expansion_blocker": (
                "requires completed 100%->90% gates for every qualified engine"
            ),
        },
        "classification": classification,
        "claim_boundary": (
            "A 100% divergence is an implementation failure. A 90% divergence is "
            "a selection/retention-quality result only after the 100% gate passes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--plain-dir", type=Path, required=True)
    parser.add_argument("--pra-100-dir", type=Path)
    parser.add_argument("--pra-90-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        engine=args.engine,
        instance_id=args.instance_id,
        plain_dir=args.plain_dir,
        pra100_dir=args.pra_100_dir,
        pra90_dir=args.pra_90_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"classification": payload["classification"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
