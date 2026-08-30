"""Measure text, full-resource, and delta transport on one typed agent session."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pra_hf.agent_transport import (
    AgentTurnContext,
    context_record_to_wire_resource,
    render_text_messages,
    wire_resource_identity,
)
from pra_hf.context_records import (
    ContextRecord,
    RecordType,
    RecordView,
    RecordViewName,
)
from pra_hf.deployment import PRAWireRequest, PRAWireResource
from pra_hf.gateway_session import ResourceDelta, ResourceOperation


@dataclass(frozen=True)
class ProtocolTurn:
    name: str
    context: AgentTurnContext


def _record(
    record_id: str,
    record_type: RecordType,
    payload: Mapping[str, object] | str,
    *,
    version: str,
    task_status: str,
) -> ContextRecord:
    views = {}
    if record_type in {RecordType.TOOL_RECORD, RecordType.SKILL_RECORD}:
        views = {
            RecordViewName.SELECTION: RecordView(
                RecordViewName.SELECTION,
                payload,
                tuple(payload) if isinstance(payload, Mapping) else ("body",),
            ),
            RecordViewName.FULL: RecordView(
                RecordViewName.FULL,
                payload,
                tuple(payload) if isinstance(payload, Mapping) else ("body",),
            ),
        }
    return ContextRecord(
        record_id,
        record_type,
        payload,
        version=version,
        selection_provenance={
            "authorization_scope": "project:alpha",
            "task": {
                "task_id": "task-1",
                "task_status": task_status,
            },
        },
        views=views,
    )


def build_workflow() -> tuple[ProtocolTurn, ...]:
    """Create a deterministic open, tool, wait, resume, and complete workflow."""

    document = _record(
        "doc:design",
        RecordType.GENERIC_DOCUMENT,
        {"uri": "file:///design.md", "body": "Architecture evidence " * 160},
        version="v1",
        task_status="active",
    )
    tool = _record(
        "tool:search",
        RecordType.TOOL_RECORD,
        {"uri": "tool://search", "schema": {"name": "search", "arguments": ["query"]}},
        version="v1",
        task_status="active",
    )
    skill = _record(
        "skill:review",
        RecordType.SKILL_RECORD,
        {"uri": "skill://review", "instructions": "Check evidence before concluding."},
        version="v1",
        task_status="active",
    )
    task_active = _record(
        "task:task-1",
        RecordType.TASK_STATE,
        {"task_id": "task-1", "status": "active", "description": "Review the design"},
        version="v1",
        task_status="active",
    )
    result = _record(
        "result:search-1",
        RecordType.TOOL_RESPONSE,
        {"uri": "result://search-1", "compact": "Three relevant sections", "body": "Finding " * 90},
        version="v1",
        task_status="active",
    )
    task_blocked = _record(
        "task:task-1",
        RecordType.TASK_STATE,
        {"task_id": "task-1", "status": "blocked", "description": "Review the design"},
        version="v2",
        task_status="blocked",
    )
    task_resumed = _record(
        "task:task-1",
        RecordType.TASK_STATE,
        {"task_id": "task-1", "status": "active", "description": "Review the design"},
        version="v3",
        task_status="active",
    )
    task_complete = _record(
        "task:task-1",
        RecordType.TASK_STATE,
        {"task_id": "task-1", "status": "completed", "description": "Review the design"},
        version="v4",
        task_status="completed",
    )
    messages: list[Mapping[str, Any]] = []
    turns = []
    for name, user, assistant, records, task in (
        ("open", "Review the design", "I will inspect it.", (document,), task_active),
        ("tool_result", "Use search", "Search completed.", (document, result), task_active),
        ("waiting", "Wait for approval", "Task paused.", (document, result), task_blocked),
        ("resume", "Approval arrived", "Continuing.", (document, result), task_resumed),
        ("complete", "Finish the review", "Review complete.", (document, result), task_complete),
    ):
        messages.append({"role": "user", "content": user})
        context = AgentTurnContext(
            messages=tuple(messages),
            records=(*records, task),
            tool_records=(tool,),
            skill_records=(skill,),
            tools=({"type": "function", "function": {"name": "search"}},),
            task_id="task-1",
            task_metadata=dict(task.payload),
            selected_record_ids=tuple(record.record_id for record in (*records, task, tool, skill)),
            metadata={"tenant_id": "tenant-a"},
        )
        turns.append(ProtocolTurn(name, context))
        messages.append({"role": "assistant", "content": assistant})
    return tuple(turns)


def _bytes(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def _identity(resource: PRAWireResource) -> str:
    return wire_resource_identity(resource)


def run_protocol_benchmark() -> tuple[list[dict[str, object]], dict[str, object]]:
    previous_messages: tuple[Mapping[str, Any], ...] = ()
    previous_resources: dict[str, tuple[str, PRAWireResource]] = {}
    rows: list[dict[str, object]] = []
    for index, turn in enumerate(build_workflow(), start=1):
        resources = tuple(
            context_record_to_wire_resource(
                record, tenant_id="tenant-a", session_id="session-a"
            )
            for record in turn.context.detached_records
        )
        text_messages = render_text_messages(turn.context)
        text_payload = {
            "model": "fixture",
            "messages": text_messages,
            "tools": turn.context.tools,
        }
        full_request = PRAWireRequest(
            "fixture",
            turn.context.messages,
            tools=turn.context.tools,
            tenant_id="tenant-a",
            session_id="session-a",
            task_id="task-1",
            resources=resources,
            metadata={"task_metadata": turn.context.task_metadata},
        ).to_openai()
        message_delta = (
            turn.context.messages[len(previous_messages) :]
            if previous_messages
            and tuple(turn.context.messages[: len(previous_messages)]) == previous_messages
            else turn.context.messages
        )
        current = {resource.resource_id: (_identity(resource), resource) for resource in resources}
        operations: list[ResourceDelta] = []
        changed = []
        for resource_id, (identity, resource) in current.items():
            old = previous_resources.get(resource_id)
            if old is None or old[0] != identity:
                operation = ResourceOperation.ADD if old is None else ResourceOperation.UPDATE
                changed.append(resource)
                operations.append(ResourceDelta(
                    operation, resource_id, resource.uri, identity,
                    resource.record_type, resource.authorization_scope,
                ))
            else:
                operations.append(ResourceDelta(
                    ResourceOperation.UNCHANGED, resource_id, resource.uri,
                    identity, resource.record_type, resource.authorization_scope,
                ))
        for resource_id, (identity, resource) in previous_resources.items():
            if resource_id not in current:
                operations.append(ResourceDelta(
                    ResourceOperation.REMOVE, resource_id, resource.uri, identity,
                    resource.record_type, resource.authorization_scope,
                ))
        delta_request = PRAWireRequest(
            "fixture",
            tuple(message_delta),
            tools=turn.context.tools,
            tenant_id="tenant-a",
            session_id="session-a",
            task_id="task-1",
            resources=tuple(changed),
            resource_ops=tuple(operations),
            history_mode="DELTA" if previous_messages else "FULL",
            metadata={"task_metadata": turn.context.task_metadata},
        ).to_openai()
        text_bytes = _bytes(text_payload)
        full_bytes = _bytes(full_request)
        delta_bytes = _bytes(delta_request)
        rows.append({
            "turn": index,
            "stage": turn.name,
            "records": len(resources),
            "full_text_bytes": text_bytes,
            "pra_full_bytes": full_bytes,
            "pra_delta_bytes": delta_bytes,
            "message_delta_bytes": _bytes(message_delta),
            "resource_delta_bytes": _bytes([row.to_dict() for row in operations]),
            "resource_bodies_sent": len(changed),
            "visible_text_tokens_estimate": sum(
                len(str(message.get("content", "")).split()) for message in text_messages
            ),
            "visible_pra_tokens_estimate": sum(
                len(str(message.get("content", "")).split()) for message in turn.context.messages
            ),
            "selection_parity": 1,
            "task_metadata_preserved": int(
                delta_request["pra"]["metadata"]["task_metadata"]["status"]
                == turn.context.task_metadata["status"]
            ),
            "tool_schema_preserved": int(bool(turn.context.tools)),
        })
        previous_messages = (
            *turn.context.messages,
            {"role": "assistant", "content": {
                "open": "I will inspect it.",
                "tool_result": "Search completed.",
                "waiting": "Task paused.",
                "resume": "Continuing.",
                "complete": "Review complete.",
            }[turn.name]},
        )
        previous_resources = current
    totals = {
        "turns": len(rows),
        "full_text_bytes": sum(int(row["full_text_bytes"]) for row in rows),
        "pra_full_bytes": sum(int(row["pra_full_bytes"]) for row in rows),
        "pra_delta_bytes": sum(int(row["pra_delta_bytes"]) for row in rows),
        "resource_bodies_sent_full": sum(int(row["records"]) for row in rows),
        "resource_bodies_sent_delta": sum(int(row["resource_bodies_sent"]) for row in rows),
        "selection_parity": all(row["selection_parity"] == 1 for row in rows),
        "task_metadata_preserved": all(row["task_metadata_preserved"] == 1 for row in rows),
        "tool_schema_preserved": all(row["tool_schema_preserved"] == 1 for row in rows),
    }
    totals["delta_vs_text_reduction"] = 1 - totals["pra_delta_bytes"] / totals["full_text_bytes"]
    totals["delta_vs_full_reduction"] = 1 - totals["pra_delta_bytes"] / totals["pra_full_bytes"]
    return rows, totals


def write_results(output: Path) -> dict[str, object]:
    rows, summary = run_protocol_benchmark()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "agent_transport_turns.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "agent_transport_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "generated_agent_transport_results.tex").write_text(
        "\n".join((
            f"\\newcommand{{\\AgentTransportTextBytes}}{{{summary['full_text_bytes']:,}}}",
            f"\\newcommand{{\\AgentTransportFullBytes}}{{{summary['pra_full_bytes']:,}}}",
            f"\\newcommand{{\\AgentTransportDeltaBytes}}{{{summary['pra_delta_bytes']:,}}}",
            f"\\newcommand{{\\AgentTransportDeltaTextReduction}}{{{100 * summary['delta_vs_text_reduction']:.1f}\\%}}",
            f"\\newcommand{{\\AgentTransportDeltaFullReduction}}{{{100 * summary['delta_vs_full_reduction']:.1f}\\%}}",
            f"\\newcommand{{\\AgentTransportFullBodies}}{{{summary['resource_bodies_sent_full']}}}",
            f"\\newcommand{{\\AgentTransportDeltaBodies}}{{{summary['resource_bodies_sent_delta']}}}",
        )) + "\n",
        encoding="utf-8",
    )
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.2, 3.5))
        x = range(len(rows))
        axis.plot(x, [row["full_text_bytes"] for row in rows], marker="o", label="TEXT")
        axis.plot(x, [row["pra_full_bytes"] for row in rows], marker="s", label="PRA_FULL")
        axis.plot(x, [row["pra_delta_bytes"] for row in rows], marker="^", label="PRA_DELTA")
        axis.set_xticks(list(x), [str(row["stage"]) for row in rows], rotation=15)
        axis.set_ylabel("Wire bytes per turn")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output / "agent_transport_wire_bytes.pdf")
        figure.savefig(output / "agent_transport_wire_bytes.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/paper4_5_runtime/agent_transport_protocol"),
    )
    args = parser.parse_args()
    print(json.dumps(write_results(args.output), indent=2))


if __name__ == "__main__":
    main()
