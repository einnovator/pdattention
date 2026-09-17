"""Audit cross-instruction edges in a boundary-free persistent session.

This is an evaluator-side diagnostic.  It reconstructs each request frontier
from ordinary user prompts, never from benchmark issue IDs, and reports which
older instruction epochs remain reachable and why.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .dag import (
    DagEdgeKind,
    _workspace_lineage,
    build_frontier_information_flow_dag,
)
from .multi_issue_session import compose_multi_issue_session
from .recordizer import recordize_replay_messages


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes") if isinstance(payload, Mapping) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("persistent prefix requires non-empty episodes")
    result = []
    for index, row in enumerate(episodes, 1):
        trajectory = row.get("trajectory") if isinstance(row, Mapping) else None
        if not isinstance(trajectory, Mapping):
            raise ValueError(f"prefix episode {index} has no trajectory")
        result.append(dict(trajectory))
    return result


def _load_episode_export(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = payload.get("trajectory") if isinstance(payload, Mapping) else None
    if not isinstance(trajectory, Mapping):
        raise ValueError("episode export has no trajectory")
    return dict(trajectory)


def _request_prefix_lengths(messages: list[Mapping[str, Any]]) -> tuple[int, ...]:
    return tuple(
        index for index, row in enumerate(messages)
        if str(row.get("role") or "") == "assistant"
    )


def audit_cross_epoch_edges(
    *,
    prefix_episodes: list[dict[str, Any]],
    active_episode: dict[str, Any],
    recent_user_prompts: int,
    protocol_exemplars: int,
) -> dict[str, Any]:
    messages = active_episode.get("messages")
    if not isinstance(messages, list):
        raise ValueError("active episode has no messages")
    request_rows: list[dict[str, Any]] = []
    all_unexplained: list[dict[str, Any]] = []
    for request_index, prefix_length in enumerate(
        _request_prefix_lengths(messages), 1
    ):
        active = dict(active_episode)
        active["messages"] = messages[:prefix_length]
        composed = compose_multi_issue_session(
            (*prefix_episodes, active),
            session_id="frontier-cross-epoch-audit",
            boundary_mode="boundary_free",
        )
        history = recordize_replay_messages(composed["messages"])
        dag = build_frontier_information_flow_dag(
            history,
            recent_user_prompts=recent_user_prompts,
            valid_protocol_exemplars=protocol_exemplars,
        )
        epoch_for_record = {
            record_id: epoch.epoch_index
            for epoch in dag.epochs
            for record_id in epoch.record_ids
        }
        records = history.record_by_id
        cross_rows: list[dict[str, Any]] = []
        unexplained: list[dict[str, Any]] = []
        for edge in dag.edges:
            source_epoch = epoch_for_record.get(edge.source_id)
            target_epoch = epoch_for_record.get(edge.target_id)
            if source_epoch == target_epoch:
                continue
            source_lineage = _workspace_lineage(records[edge.source_id])
            target_lineage = _workspace_lineage(records[edge.target_id])
            if edge.kind is DagEdgeKind.PROTOCOL_CONTROL:
                basis = "atomic_protocol_exemplar"
            elif edge.kind is DagEdgeKind.DECLARED_DEPENDENCY:
                basis = "declared_record_dependency"
            elif (
                edge.kind is DagEdgeKind.RESOURCE_FLOW
                and source_lineage is not None
                and source_lineage == target_lineage
            ):
                basis = "explicit_workspace_resource_lineage"
            else:
                basis = "unexplained_cross_epoch_edge"
            row = {
                "source_record_id": edge.source_id,
                "target_record_id": edge.target_id,
                "source_epoch": source_epoch,
                "target_epoch": target_epoch,
                "kind": edge.kind.value,
                "resource_id": edge.resource_id,
                "source_workspace_lineage_id": source_lineage,
                "target_workspace_lineage_id": target_lineage,
                "basis": basis,
            }
            cross_rows.append(row)
            if basis == "unexplained_cross_epoch_edge":
                unexplained.append(row)
                all_unexplained.append({"request_index": request_index, **row})
        live_by_epoch = Counter(
            epoch_for_record[record_id]
            for record_id in dag.live_ancestor_record_ids
        )
        records_by_epoch = Counter(epoch_for_record.values())
        retired_by_epoch = Counter(
            candidate.epoch_index for candidate in dag.retirement_candidates
        )
        request_rows.append({
            "request_index": request_index,
            "active_message_prefix_length": prefix_length,
            "instruction_epoch_count": len(dag.epochs),
            "frontier_epoch_indices": list(dag.frontier_epoch_indices),
            "record_count_by_epoch": dict(sorted(records_by_epoch.items())),
            "live_record_count_by_epoch": dict(sorted(live_by_epoch.items())),
            "retired_group_count_by_epoch": dict(sorted(retired_by_epoch.items())),
            "cross_epoch_edge_count": len(cross_rows),
            "cross_epoch_edge_kind_counts": dict(sorted(Counter(
                row["kind"] for row in cross_rows
            ).items())),
            "unexplained_cross_epoch_edge_count": len(unexplained),
            "cross_epoch_edges": cross_rows,
        })
    return {
        "schema_version": 1,
        "study": "paper8_5_boundary_free_cross_epoch_edge_audit",
        "policy_inputs": {
            "recent_user_prompts": recent_user_prompts,
            "protocol_exemplars": protocol_exemplars,
            "boundary_mode": "boundary_free",
            "evaluator_task_ids_visible_to_policy": False,
        },
        "completed_prefix_episode_count": len(prefix_episodes),
        "active_request_count": len(request_rows),
        "unexplained_cross_epoch_edge_count": len(all_unexplained),
        "unexplained_cross_epoch_edges": all_unexplained,
        "requests": request_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-prefix", type=Path, required=True)
    parser.add_argument("--episode-export", type=Path, required=True)
    parser.add_argument("--recent-user-prompts", type=int, default=1)
    parser.add_argument("--protocol-exemplars", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_cross_epoch_edges(
        prefix_episodes=_load_episodes(args.persistent_prefix),
        active_episode=_load_episode_export(args.episode_export),
        recent_user_prompts=args.recent_user_prompts,
        protocol_exemplars=args.protocol_exemplars,
    )
    result["inputs"] = {
        "persistent_prefix": str(args.persistent_prefix),
        "persistent_prefix_sha256": _sha256(args.persistent_prefix),
        "episode_export": str(args.episode_export),
        "episode_export_sha256": _sha256(args.episode_export),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "active_request_count": result["active_request_count"],
        "unexplained_cross_epoch_edge_count": result[
            "unexplained_cross_epoch_edge_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
