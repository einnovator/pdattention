"""Run deterministic adversarial checks against the typed execution boundary."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from data.agent_workflows import realistic_tool_catalog, workflow_executor, workflow_tasks
from pra_hf.agent_execution import ExecutionAuthorization, ToolCall


def _resource(resources, name: str):
    return next(resource for resource in resources if resource.name == name)


def run(output: Path) -> dict[str, object]:
    resources = realistic_tool_catalog()
    task = next(task for task in workflow_tasks() if task.task_id == "m4-user-3")
    update = _resource(resources, "update_user")
    delete = _resource(resources, "delete_user")
    executor = workflow_executor(resources, task)
    update_call = ToolCall(
        "update_user", {"user_id": "u17", "status": "reviewed"}
    )
    bound = ExecutionAuthorization(
        frozenset((update.uri,)),
        allow_writes=True,
        tenant_id=update.tenant_id,
        session_id="session-a",
        expires_at_unix=200.0,
    )

    cases = [
        (
            "valid_bound_call",
            executor,
            update_call,
            (update.uri,),
            bound,
            update.tenant_id,
            "session-a",
            100.0,
            "executed",
        ),
        (
            "cross_tenant_replay",
            executor,
            update_call,
            (update.uri,),
            bound,
            "tenant-b",
            "session-a",
            100.0,
            "tenant_not_authorized",
        ),
        (
            "cross_session_replay",
            executor,
            update_call,
            (update.uri,),
            bound,
            update.tenant_id,
            "session-b",
            100.0,
            "session_not_authorized",
        ),
        (
            "expired_grant",
            executor,
            update_call,
            (update.uri,),
            bound,
            update.tenant_id,
            "session-a",
            200.0,
            "authorization_expired",
        ),
        (
            "undisclosed_uri",
            executor,
            update_call,
            (),
            ExecutionAuthorization(frozenset((update.uri,)), allow_writes=True),
            None,
            None,
            100.0,
            "tool_not_disclosed",
        ),
        (
            "write_escalation",
            executor,
            update_call,
            (update.uri,),
            ExecutionAuthorization(frozenset((update.uri,))),
            None,
            None,
            100.0,
            "write_not_authorized",
        ),
        (
            "destructive_escalation",
            executor,
            ToolCall("delete_user", {"user_id": "u17"}),
            (delete.uri,),
            ExecutionAuthorization(
                frozenset((delete.uri,)), allow_writes=True, allow_destructive=False
            ),
            None,
            None,
            100.0,
            "destructive_not_authorized",
        ),
        (
            "unknown_argument_injection",
            executor,
            ToolCall(
                "update_user",
                {"user_id": "u17", "status": "reviewed", "admin": True},
            ),
            (update.uri,),
            ExecutionAuthorization(frozenset((update.uri,)), allow_writes=True),
            None,
            None,
            100.0,
            "unknown_argument",
        ),
        (
            "argument_type_confusion",
            executor,
            ToolCall("update_user", {"user_id": 17, "status": "reviewed"}),
            (update.uri,),
            ExecutionAuthorization(frozenset((update.uri,)), allow_writes=True),
            None,
            None,
            100.0,
            "argument_type_mismatch",
        ),
    ]

    revoked_resources = tuple(
        replace(resource, revoked=True) if resource.uri == update.uri else resource
        for resource in resources
    )
    revoked_executor = workflow_executor(revoked_resources, task)
    cases.append(
        (
            "revoked_resource",
            revoked_executor,
            update_call,
            (update.uri,),
            ExecutionAuthorization(frozenset((update.uri,)), allow_writes=True),
            None,
            None,
            100.0,
            "tool_revoked",
        )
    )

    rows = []
    for name, active_executor, call, selected, authorization, tenant, session, now, expected in cases:
        result = active_executor.execute(
            call,
            selected_uris=selected,
            authorization=authorization,
            tenant_id=tenant,
            session_id=session,
            now_unix=now,
            call_id=name,
        )
        rows.append(
            {
                "case": name,
                "expected_reason": expected,
                "observed_reason": result.reason,
                "executed": result.executed,
                "passed": result.reason == expected,
            }
        )

    attacks = [row for row in rows if row["case"] != "valid_bound_call"]
    payload = {
        "schema_version": "paper6.5-authorization-adversarial-v1",
        "experiment": "typed_execution_authorization_boundary",
        "evidence_tier": "DETERMINISTIC_MECHANISM_TEST",
        "rows": rows,
        "summary": {
            "checks": len(rows),
            "checks_passed": sum(bool(row["passed"]) for row in rows),
            "adversarial_attempts": len(attacks),
            "adversarial_attempts_blocked": sum(
                bool(row["passed"]) and not bool(row["executed"]) for row in attacks
            ),
            "valid_calls_executed": sum(
                bool(row["executed"]) for row in rows if row["case"] == "valid_bound_call"
            ),
        },
        "scope": (
            "Deterministic in-memory execution-boundary checks; not a penetration "
            "test or remote-provider security evaluation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args().output)
    print(json.dumps(result["summary"], indent=2))
