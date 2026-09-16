"""Compose audited persistent-session trajectories from independent episodes."""

from __future__ import annotations

import argparse
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .recordizer import annotate_minisweagent_messages, recordize_replay_messages


class SessionMode(str, Enum):
    FRESH_PER_ISSUE = "fresh_per_issue"
    PERSISTENT = "persistent"


class BoundaryMode(str, Enum):
    """How issue transitions are represented in the model/policy history."""

    EXPLICIT = "explicit"
    BOUNDARY_FREE = "boundary_free"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix(value: Any, episode_id: str) -> str:
    return f"{episode_id}:{value}"


def compose_multi_issue_session(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    source_paths: Sequence[Path] | None = None,
    session_id: str | None = None,
    boundary_mode: BoundaryMode | str = BoundaryMode.EXPLICIT,
) -> dict[str, Any]:
    """Join issue episodes while keeping all logical identities explicit.

    The workspaces may differ: the user-visible task record declares each
    transition, while record metadata binds every record to its episode and
    workspace scope. This models a developer retaining one chat/session while
    moving between issues, not applying unrelated SWE-bench patches to one
    incompatible base commit.
    """

    selected_boundary_mode = BoundaryMode(boundary_mode)
    if not trajectories:
        raise ValueError("at least one issue trajectory is required")
    paths = list(source_paths or ())
    if paths and len(paths) != len(trajectories):
        raise ValueError("source_paths must align one-to-one with trajectories")
    instance_ids = [str(row.get("instance_id") or "") for row in trajectories]
    if any(not value for value in instance_ids) or len(set(instance_ids)) != len(instance_ids):
        raise ValueError("issue trajectories require distinct non-empty instance IDs")

    combined: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    assistant_count = 0
    global_record_index = 0
    global_turn_index = 0
    for episode_index, (trajectory, instance_id) in enumerate(
        zip(trajectories, instance_ids), 1
    ):
        raw_messages = trajectory.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(f"{instance_id}: trajectory has no messages")
        annotated = annotate_minisweagent_messages(raw_messages)
        episode_id = f"episode-{episode_index:02d}"
        local_turn_ids: dict[str, str] = {}
        start_decision = assistant_count + 1
        kept = 0
        for message in annotated:
            role = str(message.get("role") or "")
            if role == "system" and episode_index > 1:
                continue
            copied = dict(message)
            metadata = dict(copied.get("metadata") or {})
            declaration = dict(metadata.get("pra_record") or {})
            record_metadata = dict(declaration.get("metadata") or {})
            if selected_boundary_mode is BoundaryMode.EXPLICIT:
                declaration["record_id"] = _prefix(
                    declaration["record_id"], episode_id
                )
                declaration["turn_id"] = _prefix(
                    declaration["turn_id"], episode_id
                )
                declaration["causal_group_id"] = _prefix(
                    declaration["causal_group_id"], episode_id
                )
                record_metadata.update({
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "episode_status": (
                        "active" if episode_index == len(trajectories) else "completed"
                    ),
                    "workspace_scope": instance_id,
                })
            else:
                # Resequence identities over the continuous transcript.  The
                # selector sees neither an episode namespace nor a workspace
                # transition.  The separate ``episodes`` ledger below remains
                # evaluator-only and never enters the request history.
                declaration["record_id"] = f"record-{global_record_index:06d}"
                global_record_index += 1
                local_turn_id = str(declaration["turn_id"])
                if declaration.get("primary_role") == "system":
                    global_turn_id = "system"
                elif episode_index == 1 and declaration.get("primary_role") == "task":
                    global_turn_id = "task"
                elif declaration.get("primary_role") == "task":
                    global_turn_id = f"input-{global_record_index - 1:06d}"
                    declaration["primary_role"] = "user_input"
                    declaration["semantic_roles"] = ["user_input"]
                else:
                    global_turn_id = local_turn_ids.get(local_turn_id, "")
                    if not global_turn_id:
                        global_turn_id = f"turn-{global_turn_index:06d}"
                        global_turn_index += 1
                        local_turn_ids[local_turn_id] = global_turn_id
                declaration["turn_id"] = global_turn_id
                declaration["causal_group_id"] = global_turn_id
                for key in (
                    "episode_id", "episode_index", "episode_status",
                    "workspace_scope",
                ):
                    record_metadata.pop(key, None)
            declaration["metadata"] = record_metadata
            metadata["pra_record"] = declaration
            if selected_boundary_mode is BoundaryMode.EXPLICIT:
                metadata["pra_episode"] = {
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "status": record_metadata["episode_status"],
                    "workspace_scope": instance_id,
                }
            else:
                metadata.pop("pra_episode", None)
            copied["metadata"] = metadata
            if role == "exit":
                # mini-swe-agent stores the observation produced by its final
                # submission command as a non-OpenAI ``exit`` role.  A later
                # issue in the same chat must see that command as completed,
                # so normalize the terminal observation to ``user`` while
                # retaining its verbatim payload in the FULL control.
                copied["role"] = "user"
                semantic_roles = list(declaration.get("semantic_roles") or ())
                for value in ("tool_observation", "finalization"):
                    if value not in semantic_roles:
                        semantic_roles.append(value)
                declaration["semantic_roles"] = semantic_roles
                metadata["pra_record"] = declaration
                copied["metadata"] = metadata
            if (
                selected_boundary_mode is BoundaryMode.EXPLICIT
                and declaration.get("primary_role") == "task"
                and episode_index > 1
            ):
                copied["content"] = (
                    f'<pra_episode_boundary id="{episode_id}" '
                    f'workspace="{instance_id}" prior_status="completed"/>\n'
                    + str(copied.get("content") or "")
                )
            combined.append(copied)
            kept += 1
            if copied["role"] == "assistant":
                assistant_count += 1
        episodes.append({
            "episode_id": episode_id,
            "episode_index": episode_index,
            "instance_id": instance_id,
            "status": "active" if episode_index == len(trajectories) else "completed",
            "workspace_scope": instance_id,
            "first_assistant_decision": start_decision,
            "last_assistant_decision": assistant_count,
            "assistant_decisions": assistant_count - start_decision + 1,
            "model_visible_messages": kept,
            "reference_exit_status": (trajectory.get("info") or {}).get("exit_status"),
            "reference_submission": bool((trajectory.get("info") or {}).get("submission")),
            "source": str(paths[episode_index - 1]) if paths else None,
            "source_sha256": _sha256(paths[episode_index - 1]) if paths else None,
        })

    history = recordize_replay_messages(combined)
    if len(history.records) != len(combined):
        raise AssertionError("composed session lost a declared record")
    return {
        "schema_version": 1,
        "study": "paper8_5_multi_issue_persistent_session",
        "instance_id": "persistent-session:" + "+".join(instance_ids),
        "issue_count": len(episodes),
        "session_mode": SessionMode.PERSISTENT.value,
        "boundary_mode": selected_boundary_mode.value,
        "session_id": session_id or "persistent:" + "+".join(instance_ids),
        "session_semantics": (
            "one model-visible conversation with evaluator-hidden issue boundaries; "
            "prior workspaces are not physically merged"
            if selected_boundary_mode is BoundaryMode.BOUNDARY_FREE
            else
            "one model-visible conversation across explicitly declared issue and "
            "workspace transitions; prior workspaces are not physically merged"
        ),
        "episodes": episodes,
        "reference_success": all(
            row["reference_exit_status"] == "Submitted" and row["reference_submission"]
            for row in episodes
        ),
        "messages": combined,
        "record_count": len(history.records),
        "turn_count": len(history.turns),
    }


def compose_issue_session_schedule(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    mode: SessionMode | str,
    source_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Build the same ordered cohort as fresh or persistent sessions."""

    selected_mode = SessionMode(mode)
    paths = list(source_paths or ())
    if paths and len(paths) != len(trajectories):
        raise ValueError("source_paths must align one-to-one with trajectories")
    if selected_mode is SessionMode.PERSISTENT:
        sessions = [compose_multi_issue_session(
            trajectories,
            source_paths=paths or None,
            session_id="paper8.5:persistent:locked-order-v1",
        )]
    else:
        sessions = []
        for index, trajectory in enumerate(trajectories, 1):
            session = compose_multi_issue_session(
                (trajectory,),
                source_paths=(paths[index - 1],) if paths else None,
                session_id=f"paper8.5:fresh:issue-{index:02d}",
            )
            session["session_mode"] = SessionMode.FRESH_PER_ISSUE.value
            session["global_issue_index"] = index
            sessions.append(session)
    return {
        "schema_version": 1,
        "study": "paper8_5_issue_session_schedule",
        "session_mode": selected_mode.value,
        "issue_count": len(trajectories),
        "session_count": len(sessions),
        "ordered_instance_ids": [
            str(trajectory.get("instance_id") or "") for trajectory in trajectories
        ],
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--session-mode",
        choices=[mode.value for mode in SessionMode],
        default=SessionMode.PERSISTENT.value,
    )
    args = parser.parse_args()
    paths = [path.resolve() for path in args.trajectory]
    trajectories = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if args.session_mode == SessionMode.PERSISTENT.value:
        result = compose_multi_issue_session(trajectories, source_paths=paths)
    else:
        result = compose_issue_session_schedule(
            trajectories, mode=args.session_mode, source_paths=paths
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "issue_count": result["issue_count"],
        "session_mode": result["session_mode"],
        "session_count": result.get("session_count", 1),
        "record_count": result.get("record_count"),
        "turn_count": result.get("turn_count"),
    }))


if __name__ == "__main__":
    main()
