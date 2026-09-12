"""Reduce matched frozen replay artifacts into JSON and Markdown comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


_STUDY = "paper8_5_agent_memory_frozen_replay"


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _identity(artifact: Mapping[str, Any], source: str) -> dict[str, str]:
    if artifact.get("study") != _STUDY:
        raise ValueError(f"{source}: not a frozen replay artifact")
    configuration = artifact.get("run_configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"{source}: missing run_configuration")
    if artifact.get("run_configuration_digest") != _digest(configuration):
        raise ValueError(f"{source}: invalid run_configuration_digest")
    identity = {
        "instance_id": artifact.get("instance_id"),
        "trajectory_digest": configuration.get("trajectory_digest"),
        "model": artifact.get("model"),
        "tokenizer": artifact.get("tokenizer"),
    }
    if not all(isinstance(value, str) and value for value in identity.values()):
        raise ValueError(f"{source}: incomplete trajectory/model/tokenizer identity")
    if configuration.get("model") != identity["model"]:
        raise ValueError(f"{source}: top-level and configured model differ")
    if configuration.get("tokenizer") != identity["tokenizer"]:
        raise ValueError(f"{source}: top-level and configured tokenizer differ")
    if configuration.get("reference_replay_digest") != artifact.get(
        "reference_replay_digest"
    ):
        raise ValueError(f"{source}: inconsistent reference_replay_digest")
    return identity


def _validated_rows(
    artifact: Mapping[str, Any], source: str
) -> list[Mapping[str, Any]]:
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{source}: rows must be a list")
    attempted = _require_int(artifact.get("attempted_decisions"), f"{source}.attempted_decisions")
    completed = _require_int(artifact.get("completed_decisions"), f"{source}.completed_decisions")
    if attempted != len(rows):
        raise ValueError(f"{source}: attempted_decisions does not match rows")
    completed_rows = 0
    for offset, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or row.get("decision") != offset:
            raise ValueError(f"{source}: rows are not a contiguous decision prefix")
        if not isinstance(row.get("action_valid"), bool):
            raise ValueError(f"{source}: decision {offset} lacks action_valid")
        status = row.get("decision_status")
        if not isinstance(status, str):
            raise ValueError(f"{source}: decision {offset} lacks decision_status")
        completed_rows += int(status == "completed")
        for field in (
            "full_history_tokens",
            "selected_whole_record_tokens",
            "materialized_tokens",
            "requested_budget_tokens",
            "mandatory_overflow_tokens",
        ):
            value = _require_int(row.get(field), f"{source}.rows[{offset}].{field}")
            if value < 0:
                raise ValueError(f"{source}: decision {offset} has negative {field}")
        retention = _require_number(
            row.get("realized_record_retention_fraction"),
            f"{source}.rows[{offset}].realized_record_retention_fraction",
        )
        if not 0 <= retention <= 1:
            raise ValueError(f"{source}: decision {offset} has invalid retention")
    if completed != completed_rows:
        raise ValueError(f"{source}: completed_decisions does not match row statuses")
    return rows


def _is_transport_failure(row: Mapping[str, Any]) -> bool:
    status = str(row.get("decision_status", ""))
    return status.startswith("failed_transport")


def _first_divergence(
    rows: Sequence[Mapping[str, Any]],
    full_by_decision: Mapping[int, Mapping[str, Any]],
    predicate,
) -> int | None:
    return next(
        (
            int(row["decision"])
            for row in rows
            if predicate(row, full_by_decision[int(row["decision"])])
        ),
        None,
    )


def _canonical_shell_action(command: Any) -> str | None:
    """Apply only syntax-preserving cleanup used for a conservative metric."""

    if not isinstance(command, str):
        return None
    lines = [
        line.strip()
        for line in command.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


def _arm_summary(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    full_by_decision: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    attempted = len(rows)
    valid = sum(bool(row["action_valid"]) for row in rows)
    exact_content = sum(
        row.get("generated_content")
        == full_by_decision[int(row["decision"])].get("generated_content")
        for row in rows
    )
    reference_command_decisions = sum(
        full_by_decision[int(row["decision"])].get("generated_command") is not None
        for row in rows
    )
    exact_command = sum(
        full_by_decision[int(row["decision"])].get("generated_command") is not None
        and row.get("generated_command")
        == full_by_decision[int(row["decision"])].get("generated_command")
        for row in rows
    )
    equivalent_command = sum(
        full_by_decision[int(row["decision"])].get("generated_command") is not None
        and _canonical_shell_action(row.get("generated_command"))
        == _canonical_shell_action(
            full_by_decision[int(row["decision"])].get("generated_command")
        )
        for row in rows
    )
    retentions = [float(row["realized_record_retention_fraction"]) for row in rows]
    full_tokens = sum(int(row["full_history_tokens"]) for row in rows)
    selected_tokens = sum(int(row["selected_whole_record_tokens"]) for row in rows)
    materialized_tokens = sum(int(row["materialized_tokens"]) for row in rows)
    saving = full_tokens - selected_tokens
    transport_decisions = [
        int(row["decision"]) for row in rows if _is_transport_failure(row)
    ]
    format_invalid_decisions = [
        int(row["decision"])
        for row in rows
        if not row["action_valid"] and not _is_transport_failure(row)
    ]
    return {
        "policy": artifact.get("policy"),
        "treatment": _treatment_label(artifact),
        "decisions_attempted": attempted,
        "decisions_completed": int(artifact["completed_decisions"]),
        "valid_action_decisions": valid,
        "valid_action_rate": valid / attempted if attempted else None,
        "exact_content_decisions": exact_content,
        "exact_content_rate_vs_full": exact_content / attempted if attempted else None,
        "exact_command_decisions": exact_command,
        "reference_command_decisions": reference_command_decisions,
        "exact_command_rate_vs_full": (
            exact_command / reference_command_decisions
            if reference_command_decisions
            else None
        ),
        "conservative_action_equivalent_decisions": equivalent_command,
        "conservative_action_equivalent_rate_vs_full": (
            equivalent_command / reference_command_decisions
            if reference_command_decisions
            else None
        ),
        "first_content_divergence": _first_divergence(
            rows,
            full_by_decision,
            lambda row, full: row.get("generated_content") != full.get("generated_content"),
        ),
        "first_command_divergence": _first_divergence(
            rows,
            full_by_decision,
            lambda row, full: row.get("generated_command") != full.get("generated_command"),
        ),
        "first_conservative_action_divergence": _first_divergence(
            rows,
            full_by_decision,
            lambda row, full: (
                full.get("generated_command") is not None
                and _canonical_shell_action(row.get("generated_command"))
                != _canonical_shell_action(full.get("generated_command"))
            ),
        ),
        "first_action_validity_divergence": _first_divergence(
            rows,
            full_by_decision,
            lambda row, full: bool(row["action_valid"]) != bool(full["action_valid"]),
        ),
        "realized_retention": {
            "mean": mean(retentions) if retentions else None,
            "min": min(retentions) if retentions else None,
            "max": max(retentions) if retentions else None,
        },
        "cumulative_full_history_tokens": full_tokens,
        "cumulative_selected_whole_record_tokens": selected_tokens,
        "cumulative_materialized_tokens": materialized_tokens,
        "logical_token_saving_tokens": saving,
        "logical_token_saving_fraction": saving / full_tokens if full_tokens else None,
        "mandatory_overflow_tokens": sum(
            int(row["mandatory_overflow_tokens"]) for row in rows
        ),
        "unused_matched_budget_tokens": sum(
            max(
                0,
                int(row["requested_budget_tokens"])
                - int(row["selected_whole_record_tokens"]),
            )
            for row in rows
        ),
        "transport_failures": len(transport_decisions),
        "transport_failure_decisions": transport_decisions,
        "format_invalid_decisions": len(format_invalid_decisions),
        "format_invalid_decision_ids": format_invalid_decisions,
    }


def _treatment_label(artifact: Mapping[str, Any]) -> str:
    """Name an arm by policy plus its budget contract, not policy alone."""
    policy = str(artifact.get("policy"))
    if policy == "full":
        return "FULL"
    configuration = artifact.get("run_configuration", {})
    fraction = _require_number(
        artifact.get("budget_fraction", configuration.get("budget_fraction", 1.0)),
        "budget_fraction",
    )
    matched_digest = artifact.get("matched_budget_source_digest")
    if matched_digest:
        budget = f"matched-ceiling:{str(matched_digest)[:8]}"
    else:
        interpretation = configuration.get(
            "whole_turn_budget_interpretation", "retention_ceiling"
        )
        contract = "floor" if interpretation == "retention_floor_round_up" else "ceiling"
        budget = f"{100 * fraction:g}%-{contract}"
    seed = configuration.get("seed")
    suffix = f";seed={seed}" if seed is not None else ""
    return f"{policy}@{budget}{suffix}"


def reduce_replays(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("at least one frozen replay artifact is required")
    labels = list(sources or (f"artifact[{index}]" for index in range(len(artifacts))))
    if len(labels) != len(artifacts):
        raise ValueError("sources and artifacts must have the same length")
    identities = [_identity(artifact, source) for artifact, source in zip(artifacts, labels)]
    baseline_identity = identities[0]
    for identity, source in zip(identities[1:], labels[1:]):
        for field in ("instance_id", "trajectory_digest", "model", "tokenizer"):
            if identity[field] != baseline_identity[field]:
                raise ValueError(f"{source}: {field} differs from the comparison set")
    policies = [artifact.get("policy") for artifact in artifacts]
    if any(not isinstance(policy, str) or not policy for policy in policies):
        raise ValueError("every artifact must have a policy")
    treatments = [_treatment_label(artifact) for artifact in artifacts]
    if len(set(treatments)) != len(treatments):
        raise ValueError("comparison treatments must be unique")
    full_indexes = [index for index, policy in enumerate(policies) if policy == "full"]
    if len(full_indexes) != 1:
        raise ValueError("comparison requires exactly one full policy artifact")
    full_index = full_indexes[0]
    full = artifacts[full_index]
    full_digest = _digest(full)
    rows_by_artifact = [
        _validated_rows(artifact, source)
        for artifact, source in zip(artifacts, labels)
    ]
    full_rows = rows_by_artifact[full_index]
    full_by_decision = {int(row["decision"]): row for row in full_rows}
    for index, (artifact, rows, source) in enumerate(
        zip(artifacts, rows_by_artifact, labels)
    ):
        if index == full_index:
            continue
        if artifact.get("comparison_reference") != "contemporaneous_full_replay":
            raise ValueError(f"{source}: comparison reference is not contemporaneous FULL")
        if artifact.get("reference_replay_digest") != full_digest:
            raise ValueError(f"{source}: reference replay digest does not match FULL")
        for row in rows:
            decision = int(row["decision"])
            full_row = full_by_decision.get(decision)
            if full_row is None:
                raise ValueError(f"{source}: decision {decision} has no FULL reference")
            if row.get("trajectory_message_index") != full_row.get("trajectory_message_index"):
                raise ValueError(f"{source}: decision {decision} is not trajectory-aligned")
    arms = [
        _arm_summary(artifact, rows, full_by_decision)
        for artifact, rows in zip(artifacts, rows_by_artifact)
    ]
    return {
        "schema_version": 1,
        "study": "paper8_5_agent_memory_frozen_replay_comparison",
        "evidence_class": "next_action_not_autonomous_task_quality",
        "comparison_identity": {
            **baseline_identity,
            "full_replay_digest": full_digest,
        },
        "arm_count": len(arms),
        "arms": arms,
    }


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _decision(value: int | None) -> str:
    return "—" if value is None else str(value)


def render_markdown(summary: Mapping[str, Any]) -> str:
    identity = summary["comparison_identity"]
    lines = [
        "# Frozen replay comparison",
        "",
        (
            f"Instance `{identity['instance_id']}`; model `{identity['model']}`; "
            f"tokenizer `{identity['tokenizer']}`. Rates use attempted decisions "
            "as the denominator and compare directly with contemporaneous FULL."
        ),
        "",
        "| Policy | Tried/done | Valid | Exact content | Exact command | Conservative action equivalence | First content/command/action/validity divergence | Retention mean/min/max | Full/selected/materialized tokens | Logical saving | Overflow | Unused matched budget | Transport failures | Format-invalid |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        retention = arm["realized_retention"]
        retention_text = "/".join(
            _rate(retention[key]) for key in ("mean", "min", "max")
        )
        divergence = "/".join(_decision(arm[key]) for key in (
            "first_content_divergence",
            "first_command_divergence",
            "first_conservative_action_divergence",
            "first_action_validity_divergence",
        ))
        tokens = "/".join(str(arm[key]) for key in (
            "cumulative_full_history_tokens",
            "cumulative_selected_whole_record_tokens",
            "cumulative_materialized_tokens",
        ))
        saving = (
            f"{arm['logical_token_saving_tokens']} "
            f"({_rate(arm['logical_token_saving_fraction'])})"
        )
        policy = str(arm["treatment"]).replace("|", "\\|")
        lines.append(
            f"| {policy} | {arm['decisions_attempted']}/{arm['decisions_completed']} "
            f"| {_rate(arm['valid_action_rate'])} "
            f"| {_rate(arm['exact_content_rate_vs_full'])} "
            f"| {_rate(arm['exact_command_rate_vs_full'])} "
            f"| {_rate(arm['conservative_action_equivalent_rate_vs_full'])} "
            f"| {divergence} | {retention_text} | {tokens} | {saving} "
            f"| {arm['mandatory_overflow_tokens']} "
            f"| {arm['unused_matched_budget_tokens']} "
            f"| {arm['transport_failures']} | {arm['format_invalid_decisions']} |"
        )
    lines.extend(("", "`—` means no divergence or no defined denominator.", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in args.artifacts]
    summary = reduce_replays(artifacts, sources=[str(path) for path in args.artifacts])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "arm_count": summary["arm_count"],
        "output_json": str(args.output_json),
        "output_markdown": str(args.output_markdown),
    }, indent=2))


if __name__ == "__main__":
    main()
