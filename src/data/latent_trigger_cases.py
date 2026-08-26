"""Deterministic paired latent-trigger fixtures for Paper 7."""

from __future__ import annotations

from typing import Iterable

from pra_hf.context_records import RecordType
from pra_hf.context_recovery import TriggerCase, TriggerFamily


_DISTRACTOR_ACTIONS = (
    "continue_standard",
    "retry_unchanged",
    "defer_for_review",
)


def _text_payload(label: str, hidden_line: str, *, lines: int = 21) -> str:
    values = [f"observation_{index:02d}=normal {label}" for index in range(lines)]
    values[7] = hidden_line
    return "\n".join(values)


def _log_payload(label: str, hidden_line: str) -> dict[str, object]:
    events = [f"INFO observation_{index:02d}=normal {label}" for index in range(21)]
    events[7] = f"INFO {hidden_line}"
    return {"service": label, "events": events}


def _terminal_payload(label: str, hidden_line: str) -> dict[str, object]:
    lines = [f"check_{index:02d}: ok {label}" for index in range(21)]
    lines[7] = hidden_line
    return {
        "command": f"inspect {label}",
        "exit_status": 0,
        "working_directory": "/srv/agent",
        "stdout": "\n".join(lines),
        "stderr": "",
    }


def _db_payload(label: str, hidden_fields: dict[str, object]) -> dict[str, object]:
    rows = [{"row": index, "resource": label, "status": "normal"} for index in range(21)]
    rows[7] = {"row": 7, "resource": label, **hidden_fields}
    columns = sorted({key for row in rows for key in row})
    return {"columns": columns, "rows": rows}


def _graph_payload(label: str, hidden_node: dict[str, object]) -> dict[str, object]:
    nodes = [{"id": f"{label}-{index}", "state": "normal"} for index in range(12)]
    nodes[5] = {"id": f"{label}-5", **hidden_node}
    edges = [
        {"source": f"{label}-{index}", "target": f"{label}-{index + 1}", "type": "observes"}
        for index in range(11)
    ]
    return {"nodes": nodes, "edges": edges}


def _case(
    case_id: str,
    family: TriggerFamily,
    record_type: RecordType,
    payload: object,
    trigger: str,
    evidence: str,
    explicit_query: str,
    latent_query: str,
    required_action: str,
    misleading_action: str = "continue_standard",
) -> TriggerCase:
    actions = (misleading_action, required_action, *_DISTRACTOR_ACTIONS[1:])
    return TriggerCase(
        case_id=case_id,
        family=family,
        record_type=record_type,
        payload=payload,
        hidden_trigger=trigger,
        hidden_evidence=evidence,
        explicit_query=explicit_query,
        latent_query=latent_query,
        candidate_actions=actions,
        required_action=required_action,
        misleading_action=misleading_action,
    )


def latent_trigger_cases() -> tuple[TriggerCase, ...]:
    """Return one functionally consequential case per required trigger family."""

    return (
        _case(
            "entity-custodian", TriggerFamily.ENTITY, RecordType.GENERIC_TEXT,
            _text_payload("privacy_export", "custodian=Dr_Rhea_Malik required_action=contact_data_custodian"),
            "Dr_Rhea_Malik", "the named custodian must approve the export",
            "What action is required because Dr_Rhea_Malik is the custodian?",
            "Prepare this privacy export safely.", "contact_data_custodian",
        ),
        _case(
            "alias-bluebird", TriggerFamily.ALIAS, RecordType.RAG_RESULT,
            _text_payload("account_transfer", "alias=bluebird owner=regulated_account required_action=resolve_alias_owner"),
            "bluebird", "the alias resolves to a regulated account",
            "How should I handle the bluebird alias?",
            "Transfer the requested account safely.", "resolve_alias_owner",
        ),
        _case(
            "date-certificate", TriggerFamily.NUMBER_DATE, RecordType.DB_RESULT,
            _db_payload("tls_gateway", {"certificate_expiry": "2026-09-14", "required_action": "rotate_certificate"}),
            "2026-09-14", "the certificate expiry requires rotation before rollout",
            "What action follows from certificate expiry 2026-09-14?",
            "Roll out this gateway safely.", "rotate_certificate",
        ),
        _case(
            "rare-maintenance-token", TriggerFamily.RARE_STRING, RecordType.TERMINAL_OUTPUT,
            _terminal_payload("replica_set", "maintenance_token=ZXQ-441-M required_action=apply_maintenance_override"),
            "ZXQ-441-M", "the maintenance token authorizes a bounded override",
            "What should happen when token ZXQ-441-M is present?",
            "Restore this replica set safely.", "apply_maintenance_override",
        ),
        _case(
            "relation-atlas-vault", TriggerFamily.RELATION, RecordType.GRAPH_RESULT,
            _graph_payload("atlas", {"relation": "depends_on:vault", "required_action": "restart_vault_first"}),
            "depends_on:vault", "Atlas depends on Vault and Vault must restart first",
            "What action follows from depends_on:vault?",
            "Restart the Atlas service safely.", "restart_vault_first",
        ),
        _case(
            "action-quarantine", TriggerFamily.ACTION, RecordType.LOG_BLOCK,
            _log_payload("replica_health", "replica_divergence=confirmed required_action=quarantine_replica"),
            "quarantine_replica", "confirmed divergence requires quarantine",
            "Should I perform quarantine_replica?",
            "Recover the unhealthy replica safely.", "quarantine_replica",
        ),
        _case(
            "state-pending-deletion", TriggerFamily.STATE_TRANSITION, RecordType.DB_RESULT,
            _db_payload("customer_archive", {"state": "pending_deletion", "required_action": "cancel_deletion"}),
            "pending_deletion", "the archive is already pending deletion and must be preserved",
            "What action follows from pending_deletion state?",
            "Preserve this customer archive safely.", "cancel_deletion",
        ),
        _case(
            "permission-read-only", TriggerFamily.PERMISSION, RecordType.GENERIC_TEXT,
            _text_payload("billing_ledger", "authorization=read_only required_action=request_write_grant"),
            "read_only", "the current principal lacks write authority",
            "What action is needed under read_only authorization?",
            "Correct this billing ledger safely.", "request_write_grant",
        ),
        _case(
            "dependency-libcrypto", TriggerFamily.DEPENDENCY, RecordType.TERMINAL_OUTPUT,
            _terminal_payload("legacy_worker", "dependency=libcrypto-legacy required_action=install_compatibility_package"),
            "libcrypto-legacy", "the worker depends on a legacy compatibility package",
            "What action follows from dependency libcrypto-legacy?",
            "Deploy this legacy worker safely.", "install_compatibility_package",
        ),
        _case(
            "error-e7312", TriggerFamily.ERROR_CODE, RecordType.LOG_BLOCK,
            _log_payload("search_cluster", "code=E7312 required_action=rebuild_search_index"),
            "E7312", "error E7312 denotes index corruption",
            "What action resolves error code E7312?",
            "Restore search service safely.", "rebuild_search_index",
        ),
        _case(
            "backend-glacier", TriggerFamily.BACKEND_TYPE, RecordType.GENERIC_TEXT,
            _text_payload("stored_object", "storage_backend=glacier-legacy required_action=inspect_legacy_archive"),
            "glacier-legacy", "legacy archive storage requires inspection before migration",
            "What action is required for glacier-legacy?",
            "Migrate this stored object safely.", "inspect_legacy_archive",
        ),
        _case(
            "schema-decimal-string", TriggerFamily.SCHEMA_TYPE, RecordType.DB_RESULT,
            _db_payload("payment_import", {"amount_type": "decimal_string", "required_action": "parse_decimal_string"}),
            "decimal_string", "amounts are decimal strings rather than binary floats",
            "What action follows from amount type decimal_string?",
            "Import these payments safely.", "parse_decimal_string",
        ),
        _case(
            "threshold-latency", TriggerFamily.THRESHOLD_ANOMALY, RecordType.GENERIC_TEXT,
            _text_payload("checkout_api", "threshold_marker=SLO_BREACH_842 required_action=escalate_capacity"),
            "SLO_BREACH_842", "p99 latency exceeds the 500 ms action threshold",
            "What action follows from SLO_BREACH_842?",
            "Stabilize checkout performance safely.", "escalate_capacity",
        ),
    )


def benchmark_rows(cases: Iterable[TriggerCase] | None = None) -> list[dict[str, object]]:
    """Serialize public benchmark fields without losing explicit/latent pairing."""

    return [
        {
            "case_id": case.case_id,
            "family": case.family.value,
            "record_type": case.record_type.value,
            "hidden_trigger": case.hidden_trigger,
            "hidden_evidence": case.hidden_evidence,
            "explicit_query": case.explicit_query,
            "latent_query": case.latent_query,
            "candidate_actions": list(case.candidate_actions),
            "required_action": case.required_action,
            "misleading_action": case.misleading_action,
        }
        for case in (tuple(cases) if cases is not None else latent_trigger_cases())
    ]
