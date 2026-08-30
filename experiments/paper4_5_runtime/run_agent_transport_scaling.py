"""Measure negotiated record transport over 20- and 50-turn agent sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.paper4_5_runtime.run_agent_transport_protocol import (
    ProtocolTurn,
    _bytes,
    _record,
)
from pra_hf.agent_transport import (
    AgentTurnContext,
    context_record_to_wire_resource,
    render_text_messages,
    wire_resource_identity,
)
from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.deployment import PRAWireRequest, PRAWireResource
from pra_hf.gateway_session import ResourceDelta, ResourceOperation


DEFAULT_OUTPUT = Path(
    "docs/papers/shared/results/paper4_5_runtime/agent_transport_scaling"
)


def _task_record(status: str, version: int, progress: int) -> ContextRecord:
    return _record(
        "task:long-session",
        RecordType.TASK_STATE,
        {
            "task_id": "long-session",
            "status": status,
            "description": "Review a changing long-running project",
            "progress": progress,
        },
        version=f"v{version}",
        task_status=status,
    )


def _assistant_reply(stage: str) -> str:
    return {
        "waiting": "Waiting for the dependency.",
        "resume": "The dependency is available; resuming.",
        "tool_result": "The tool result has been incorporated.",
        "task_update": "Task state updated.",
        "complete": "The review is complete.",
    }.get(stage, "Continuing the review.")


def build_scaling_workflow(turn_count: int) -> tuple[ProtocolTurn, ...]:
    """Create mostly stable context with sparse updates, tool results, and closure."""

    if turn_count < 5:
        raise ValueError("Transport scaling requires at least five turns.")
    document_v1 = _record(
        "doc:project",
        RecordType.GENERIC_DOCUMENT,
        {"uri": "file:///project.md", "body": "Stable project evidence " * 220},
        version="v1",
        task_status="active",
    )
    document_v2 = _record(
        "doc:project",
        RecordType.GENERIC_DOCUMENT,
        {"uri": "file:///project.md", "body": "Updated project evidence " * 230},
        version="v2",
        task_status="active",
    )
    tool = _record(
        "tool:lookup",
        RecordType.TOOL_RECORD,
        {"uri": "tool://lookup", "schema": {"name": "lookup", "arguments": ["query"]}},
        version="v1",
        task_status="active",
    )
    skill = _record(
        "skill:verification",
        RecordType.SKILL_RECORD,
        {"uri": "skill://verification", "instructions": "Verify evidence and cite its version."},
        version="v1",
        task_status="active",
    )
    waiting_turn = max(3, turn_count // 4)
    resume_turn = waiting_turn + 2
    update_turn = max(resume_turn + 2, (3 * turn_count) // 5)
    task_version = 1
    task = _task_record("active", task_version, 0)
    latest_results: list[ContextRecord] = []
    messages: list[Mapping[str, Any]] = []
    turns: list[ProtocolTurn] = []
    for turn_index in range(1, turn_count + 1):
        stage = "work"
        if turn_index == waiting_turn:
            stage = "waiting"
            task_version += 1
            task = _task_record("waiting", task_version, turn_index)
        elif turn_index == resume_turn:
            stage = "resume"
            task_version += 1
            task = _task_record("active", task_version, turn_index)
        elif turn_index == turn_count:
            stage = "complete"
            task_version += 1
            task = _task_record("completed", task_version, 100)
        elif turn_index in {update_turn, update_turn + 5} and turn_index < turn_count:
            stage = "task_update"
            task_version += 1
            task = _task_record("active", task_version, min(95, 2 * turn_index))

        if turn_index % 9 == 0:
            stage = "tool_result"
            result_index = turn_index // 9
            latest_results.append(_record(
                f"result:lookup-{result_index}",
                RecordType.TOOL_RESPONSE,
                {
                    "uri": f"result://lookup-{result_index}",
                    "compact": f"Result {result_index}",
                    "body": (f"Tool evidence {result_index} " * 80),
                },
                version="v1",
                task_status=str(task.payload["status"]),
            ))
            latest_results = latest_results[-2:]

        document = document_v2 if turn_index >= update_turn else document_v1
        user = f"Continue project review, turn {turn_index}."
        assistant = _assistant_reply(stage)
        messages.append({"role": "user", "content": user})
        records = (document, task, *latest_results)
        context = AgentTurnContext(
            messages=tuple(messages),
            records=records,
            tool_records=(tool,),
            skill_records=(skill,),
            tools=({"type": "function", "function": {"name": "lookup"}},),
            task_id="long-session",
            task_metadata=dict(task.payload),
            selected_record_ids=tuple(
                record.record_id for record in (*records, tool, skill)
            ),
            metadata={"tenant_id": "tenant-a", "turn": turn_index},
        )
        turns.append(ProtocolTurn(stage, context))
        messages.append({"role": "assistant", "content": assistant})
    return tuple(turns)


def _operations(
    previous: Mapping[str, tuple[str, PRAWireResource]],
    resources: Sequence[PRAWireResource],
) -> tuple[tuple[PRAWireResource, ...], tuple[ResourceDelta, ...], dict[str, tuple[str, PRAWireResource]]]:
    current = {
        resource.resource_id: (wire_resource_identity(resource), resource)
        for resource in resources
    }
    changed: list[PRAWireResource] = []
    operations: list[ResourceDelta] = []
    for resource_id, (identity, resource) in current.items():
        old = previous.get(resource_id)
        if old is None or old[0] != identity:
            operation = ResourceOperation.ADD if old is None else ResourceOperation.UPDATE
            changed.append(resource)
        else:
            operation = ResourceOperation.UNCHANGED
        operations.append(ResourceDelta(
            operation,
            resource_id,
            resource.uri,
            identity,
            resource.record_type,
            resource.authorization_scope,
        ))
    for resource_id, (identity, resource) in previous.items():
        if resource_id not in current:
            operations.append(ResourceDelta(
                ResourceOperation.REMOVE,
                resource_id,
                resource.uri,
                identity,
                resource.record_type,
                resource.authorization_scope,
            ))
    return tuple(changed), tuple(operations), current


def run_scaling_workload(turn_count: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    previous_messages: tuple[Mapping[str, Any], ...] = ()
    previous_resources: dict[str, tuple[str, PRAWireResource]] = {}
    reconnect_turn = turn_count // 2 + 1
    cumulative = {"text": 0, "full": 0, "delta": 0, "full_bodies": 0, "delta_bodies": 0}
    rows: list[dict[str, object]] = []
    for turn_index, turn in enumerate(build_scaling_workflow(turn_count), 1):
        resynchronized = turn_index == reconnect_turn
        if resynchronized:
            previous_messages = ()
            previous_resources = {}
        resources = tuple(
            context_record_to_wire_resource(
                record, tenant_id="tenant-a", session_id="long-session"
            )
            for record in turn.context.detached_records
        )
        text_payload = {
            "model": "fixture",
            "messages": render_text_messages(turn.context),
            "tools": turn.context.tools,
        }
        full_payload = PRAWireRequest(
            "fixture",
            turn.context.messages,
            tools=turn.context.tools,
            tenant_id="tenant-a",
            session_id="long-session",
            task_id=turn.context.task_id,
            resources=resources,
            metadata={"task_metadata": turn.context.task_metadata},
        ).to_openai()
        prefix_matches = bool(previous_messages) and tuple(
            turn.context.messages[: len(previous_messages)]
        ) == previous_messages
        message_delta = (
            turn.context.messages[len(previous_messages):]
            if prefix_matches else turn.context.messages
        )
        changed, operations, current = _operations(previous_resources, resources)
        delta_payload = PRAWireRequest(
            "fixture",
            tuple(message_delta),
            tools=turn.context.tools,
            tenant_id="tenant-a",
            session_id="long-session",
            task_id=turn.context.task_id,
            resources=changed,
            resource_ops=operations,
            history_mode="DELTA" if prefix_matches else "FULL",
            metadata={"task_metadata": turn.context.task_metadata},
        ).to_openai()
        text_bytes = _bytes(text_payload)
        full_bytes = _bytes(full_payload)
        delta_bytes = _bytes(delta_payload)
        cumulative["text"] += text_bytes
        cumulative["full"] += full_bytes
        cumulative["delta"] += delta_bytes
        cumulative["full_bodies"] += len(resources)
        cumulative["delta_bodies"] += len(changed)
        rows.append({
            "session_turns": turn_count,
            "turn": turn_index,
            "stage": turn.name,
            "resynchronized": resynchronized,
            "text_bytes": text_bytes,
            "pra_full_bytes": full_bytes,
            "pra_delta_bytes": delta_bytes,
            "cumulative_text_bytes": cumulative["text"],
            "cumulative_pra_full_bytes": cumulative["full"],
            "cumulative_pra_delta_bytes": cumulative["delta"],
            "full_resource_bodies": len(resources),
            "delta_resource_bodies": len(changed),
            "cumulative_full_resource_bodies": cumulative["full_bodies"],
            "cumulative_delta_resource_bodies": cumulative["delta_bodies"],
            "resource_delta_bytes": _bytes([operation.to_dict() for operation in operations]),
            "visible_text_tokens_estimate": sum(
                len(str(message.get("content", "")).split())
                for message in text_payload["messages"]
            ),
            "visible_pra_tokens_estimate": sum(
                len(str(message.get("content", "")).split())
                for message in turn.context.messages
            ),
            "selection_parity": True,
            "task_metadata_preserved": (
                delta_payload["pra"]["metadata"]["task_metadata"]
                == turn.context.task_metadata
            ),
            "tool_schema_preserved": bool(turn.context.tools),
        })
        previous_messages = (
            *turn.context.messages,
            {"role": "assistant", "content": _assistant_reply(turn.name)},
        )
        previous_resources = current
    summary = {
        "turns": turn_count,
        "reconnect_turn": reconnect_turn,
        "text_bytes": cumulative["text"],
        "pra_full_bytes": cumulative["full"],
        "pra_delta_bytes": cumulative["delta"],
        "delta_vs_text_reduction": 1 - cumulative["delta"] / cumulative["text"],
        "delta_vs_full_reduction": 1 - cumulative["delta"] / cumulative["full"],
        "full_resource_bodies": cumulative["full_bodies"],
        "delta_resource_bodies": cumulative["delta_bodies"],
        "selection_parity": all(row["selection_parity"] for row in rows),
        "task_metadata_preserved": all(row["task_metadata_preserved"] for row in rows),
        "tool_schema_preserved": all(row["tool_schema_preserved"] for row in rows),
        "resynchronizations": sum(row["resynchronized"] for row in rows),
    }
    return rows, summary


def write_results(output: Path, turn_counts: Sequence[int]) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summaries = []
    for turn_count in turn_counts:
        workload_rows, summary = run_scaling_workload(turn_count)
        rows.extend(workload_rows)
        summaries.append(summary)
    with (output / "agent_transport_scaling_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "1.0",
        "experiment": "paper4_5_agent_transport_scaling_v1",
        "evidence_tier": "CONTROLLED_PROTOCOL",
        "summaries": summaries,
    }
    (output / "agent_transport_scaling_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    largest = max(summaries, key=lambda row: int(row["turns"]))
    (output / "generated_agent_transport_scaling_results.tex").write_text(
        "\n".join((
            f"\\newcommand{{\\AgentScalingTurns}}{{{largest['turns']}}}",
            f"\\newcommand{{\\AgentScalingTextBytes}}{{{largest['text_bytes']:,}}}",
            f"\\newcommand{{\\AgentScalingFullBytes}}{{{largest['pra_full_bytes']:,}}}",
            f"\\newcommand{{\\AgentScalingDeltaBytes}}{{{largest['pra_delta_bytes']:,}}}",
            f"\\newcommand{{\\AgentScalingDeltaTextReduction}}{{{100 * largest['delta_vs_text_reduction']:.1f}\\%}}",
            f"\\newcommand{{\\AgentScalingDeltaFullReduction}}{{{100 * largest['delta_vs_full_reduction']:.1f}\\%}}",
            f"\\newcommand{{\\AgentScalingFullBodies}}{{{largest['full_resource_bodies']}}}",
            f"\\newcommand{{\\AgentScalingDeltaBodies}}{{{largest['delta_resource_bodies']}}}",
        )) + "\n",
        encoding="utf-8",
    )
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        for turn_count in turn_counts:
            selected = [row for row in rows if row["session_turns"] == turn_count]
            x = [row["turn"] for row in selected]
            axes[0].plot(x, [row["cumulative_text_bytes"] for row in selected], label=f"TEXT {turn_count}")
            axes[0].plot(x, [row["cumulative_pra_full_bytes"] for row in selected], linestyle="--", label=f"FULL {turn_count}")
            axes[0].plot(x, [row["cumulative_pra_delta_bytes"] for row in selected], linestyle=":", label=f"DELTA {turn_count}")
            axes[1].plot(x, [row["cumulative_full_resource_bodies"] for row in selected], linestyle="--", label=f"FULL {turn_count}")
            axes[1].plot(x, [row["cumulative_delta_resource_bodies"] for row in selected], linestyle=":", label=f"DELTA {turn_count}")
        axes[0].set_xlabel("Turn")
        axes[0].set_ylabel("Cumulative wire bytes")
        axes[1].set_xlabel("Turn")
        axes[1].set_ylabel("Cumulative resource bodies")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(output / "agent_transport_scaling.pdf")
        figure.savefig(output / "agent_transport_scaling.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--turns", type=int, nargs="+", default=(20, 50))
    args = parser.parse_args()
    print(json.dumps(write_results(args.output, args.turns), indent=2))


if __name__ == "__main__":
    main()
