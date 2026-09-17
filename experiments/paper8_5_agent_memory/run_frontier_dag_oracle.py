"""Compare recent-frontier information-flow retirement with structural oracles.

This is an engine-independent logical screen.  It does not claim autonomous
task accuracy.  Evaluator boundaries are used only to score the selector; the
information-flow DAG sees genuine user prompts, generic record metadata, tool
effects, and resources.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .dag import (
    FrontierRetirementConfidence,
    FrontierSimplificationMode,
    build_frontier_information_flow_dag,
    simplify_disconnected_frontier,
)
from .model import (
    AgentRecord,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
)
from .multi_issue_session import BoundaryMode, compose_multi_issue_session
from .recordizer import recordize_replay_messages
from .selectors import whitespace_tokens


TokenCounter = Callable[[str], int]


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _group_costs(
    history: CanonicalAgentHistory,
    count_tokens: TokenCounter,
) -> dict[str, int]:
    records = history.record_by_id
    return {
        turn.causal_group_id: sum(
            count_tokens(records[record_id].content) for record_id in turn.record_ids
        )
        for turn in history.turns
    }


def score_frontier_dag(
    history: CanonicalAgentHistory,
    *,
    oracle_excluded_groups: Iterable[str],
    recent_user_prompts: int,
    count_tokens: TokenCounter = whitespace_tokens,
) -> dict[str, Any]:
    """Score no-path predictions against an evaluator-only group oracle."""

    dag = build_frontier_information_flow_dag(
        history, recent_user_prompts=recent_user_prompts
    )
    costs = _group_costs(history, count_tokens)
    oracle = set(oracle_excluded_groups)
    heuristic = {row.causal_group_id for row in dag.retirement_candidates}
    certified = {
        row.causal_group_id for row in dag.retirement_candidates
        if row.confidence == FrontierRetirementConfidence.CERTIFIED
    }
    recent_epoch_only = {
        group_id
        for epoch in dag.epochs
        if epoch.epoch_index not in dag.frontier_epoch_indices
        for group_id in epoch.causal_group_ids
        if group_id in costs
    }

    def score(name: str, predicted: set[str]) -> dict[str, Any]:
        true_positive = predicted.intersection(oracle)
        false_positive = predicted.difference(oracle)
        false_negative = oracle.difference(predicted)
        predicted_tokens = sum(costs.get(row, 0) for row in predicted)
        oracle_tokens = sum(costs.get(row, 0) for row in oracle)
        true_positive_tokens = sum(costs.get(row, 0) for row in true_positive)
        return {
            "rule": name,
            "predicted_group_count": len(predicted),
            "oracle_group_count": len(oracle),
            "true_positive_group_count": len(true_positive),
            "false_positive_group_count": len(false_positive),
            "false_negative_group_count": len(false_negative),
            "token_weighted_precision": _fraction(
                true_positive_tokens, predicted_tokens
            ),
            "token_weighted_recall": _fraction(
                true_positive_tokens, oracle_tokens
            ),
            "false_positive_groups": sorted(false_positive),
            "false_negative_groups": sorted(false_negative),
        }

    simplifications = []
    for mode in FrontierSimplificationMode:
        for allow_heuristic in (False, True):
            plan = simplify_disconnected_frontier(
                history,
                dag,
                mode=mode,
                count_tokens=count_tokens,
                allow_heuristic=allow_heuristic,
            )
            simplifications.append({
                "mode": mode.value,
                "confidence_gate": "heuristic_allowed" if allow_heuristic else "certified_only",
                "retired_group_count": len(plan.retired_causal_group_ids),
                "full_tokens": plan.full_tokens,
                "materialized_tokens": plan.materialized_tokens,
                "saving_fraction": plan.saving_fraction,
                "replacement_count": len(plan.record_replacements),
            })
    return {
        "recent_user_prompts": recent_user_prompts,
        "instruction_epoch_count": len(dag.epochs),
        "frontier_epoch_indices": list(dag.frontier_epoch_indices),
        "edge_count": len(dag.edges),
        "candidate_count": len(dag.retirement_candidates),
        "unknown_effect_candidate_count": sum(
            row.unknown_effect for row in dag.retirement_candidates
        ),
        "comparisons": [
            score("recent_epoch_only_no_dag", recent_epoch_only),
            score("dag_no_path_certified_only", certified),
            score("dag_no_path_heuristic_allowed", heuristic),
        ],
        "simplifications": simplifications,
        "candidates": [asdict(row) for row in dag.retirement_candidates],
    }


def _episode_ranges(episodes: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int], ...]:
    rows = []
    start = 0
    for episode in episodes:
        count = int(episode["model_visible_messages"])
        rows.append((start, start + count))
        start += count
    return tuple(rows)


def _independent_oracle_groups(
    history: CanonicalAgentHistory,
    episodes: Sequence[Mapping[str, Any]],
    *,
    recent_user_prompts: int,
) -> set[str]:
    """Retire all old interaction groups for known independent workspaces."""

    ranges = _episode_ranges(episodes)
    cutoff = max(0, len(ranges) - recent_user_prompts)
    old_end = ranges[cutoff][0] if cutoff < len(ranges) else ranges[-1][1]
    records = history.record_by_id
    return {
        turn.causal_group_id
        for turn in history.turns
        if turn.record_ids
        and all(records[row].message_index < old_end for row in turn.record_ids)
    }


def analyze_independent_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    recent_user_prompts_values: Sequence[int] = (2, 3),
    count_tokens: TokenCounter = whitespace_tokens,
) -> dict[str, Any]:
    session = compose_multi_issue_session(
        trajectories,
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        session_id="paper8.5:frontier-dag-oracle",
    )
    history = recordize_replay_messages(session["messages"])
    results = []
    for recent in recent_user_prompts_values:
        oracle = _independent_oracle_groups(
            history, session["episodes"], recent_user_prompts=recent
        )
        results.append(score_frontier_dag(
            history,
            oracle_excluded_groups=oracle,
            recent_user_prompts=recent,
            count_tokens=count_tokens,
        ))
    return {
        "case": "real_independent_cross_workspace_chain",
        "issue_count": len(trajectories),
        "instance_ids": [str(row.get("instance_id") or "") for row in trajectories],
        "selector_visibility": (
            "boundary-free model transcript; evaluator episode ledger used only for scoring"
        ),
        "results": results,
    }


def _synthetic_chain(
    resources: Sequence[str],
    *,
    workspace_scopes: Sequence[str | None] | None = None,
    unknown_epochs: Iterable[int] = (),
) -> CanonicalAgentHistory:
    """Construct a typed chain with known resource dependencies."""

    records = [AgentRecord(
        "system", "system", "system", 0, "system", "agent protocol",
        AgentRecordRole.SYSTEM, (AgentRecordRole.SYSTEM,),
    )]
    turns = []
    scopes = tuple(workspace_scopes or (None,) * len(resources))
    unknown = set(unknown_epochs)
    message_index = 1
    for epoch, (resource, scope) in enumerate(zip(resources, scopes)):
        metadata = {"workspace_lineage_id": scope} if scope else {}
        instruction_id = f"instruction-{epoch}"
        instruction_role = AgentRecordRole.TASK if epoch == 0 else AgentRecordRole.USER_INPUT
        records.append(AgentRecord(
            instruction_id, instruction_id, instruction_id, message_index,
            "user", f"Work on {resource}", instruction_role, (instruction_role,),
            resource_ids=(resource,), metadata=metadata,
        ))
        message_index += 1
        group = f"turn-{epoch}"
        action_id = f"action-{epoch}"
        observation_id = f"observation-{epoch}"
        operation = "other" if epoch in unknown else "read"
        records.extend((
            AgentRecord(
                action_id, group, group, message_index, "assistant",
                "THOUGHT inspect state " * 8
                + f"\n```mswea_bash_command\ncat {resource}\n```",
                AgentRecordRole.ASSISTANT_ACTION,
                (AgentRecordRole.ASSISTANT_ACTION,),
                command=f"cat {resource}", resource_ids=(resource,),
                metadata={**metadata, "operation_kind": operation},
            ),
            AgentRecord(
                observation_id, group, group, message_index + 1, "user",
                "<returncode>0</returncode>\n<output>" + "evidence " * 40 + "</output>",
                AgentRecordRole.TOOL_OBSERVATION,
                (AgentRecordRole.TOOL_OBSERVATION,),
                command=f"cat {resource}", return_code=0,
                resource_ids=(resource,),
                metadata={**metadata, "operation_kind": operation, "output_complete": True},
            ),
        ))
        turns.append(AgentTurn(group, group, (action_id, observation_id), message_index, True))
        message_index += 2
    return CanonicalAgentHistory(tuple(records), tuple(turns))


def synthetic_oracle_cases(
    *, count_tokens: TokenCounter = whitespace_tokens,
) -> list[dict[str, Any]]:
    cases = []
    independent = _synthetic_chain([f"src/task_{index}.py" for index in range(6)])
    cases.append({
        "case": "independent_distinct_resources_unscoped",
        "result": score_frontier_dag(
            independent,
            oracle_excluded_groups={f"turn-{index}" for index in range(4)},
            recent_user_prompts=2,
            count_tokens=count_tokens,
        ),
    })
    dependent = _synthetic_chain([
        "src/shared.py", "src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/shared.py"
    ])
    cases.append({
        "case": "old_resource_reaches_current_task",
        "result": score_frontier_dag(
            dependent,
            oracle_excluded_groups={"turn-1", "turn-2", "turn-3"},
            recent_user_prompts=2,
            count_tokens=count_tokens,
        ),
    })
    scoped = _synthetic_chain(
        ["src/common.py"] * 6,
        workspace_scopes=[f"workspace-{index}" for index in range(6)],
    )
    cases.append({
        "case": "same_path_distinct_declared_workspaces",
        "result": score_frontier_dag(
            scoped,
            oracle_excluded_groups={f"turn-{index}" for index in range(4)},
            recent_user_prompts=2,
            count_tokens=count_tokens,
        ),
    })
    unknown = _synthetic_chain(
        [f"src/task_{index}.py" for index in range(6)],
        workspace_scopes=[f"workspace-{index}" for index in range(6)],
        unknown_epochs=(1,),
    )
    cases.append({
        "case": "unknown_effect_lowers_confidence",
        "result": score_frontier_dag(
            unknown,
            oracle_excluded_groups={f"turn-{index}" for index in range(4)},
            recent_user_prompts=2,
            count_tokens=count_tokens,
        ),
    })
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--append-episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision", default="unversioned")
    args = parser.parse_args()
    count_tokens = whitespace_tokens
    tokenizer_identity = "whitespace_v1_diagnostic"
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            revision=args.tokenizer_revision,
            local_files_only=Path(args.tokenizer).exists(),
        )
        count_tokens = lambda text: len(
            tokenizer.encode(text, add_special_tokens=False)
        )
        tokenizer_identity = f"{args.tokenizer}@{args.tokenizer_revision}"
    sources = []
    trajectories = []
    if args.prefix:
        payload = json.loads(args.prefix.read_text(encoding="utf-8"))
        trajectories.extend(row["trajectory"] for row in payload.get("episodes", ()))
        sources.append({"path": str(args.prefix.resolve()), "sha256": _sha256(args.prefix)})
    if args.append_episode:
        payload = json.loads(args.append_episode.read_text(encoding="utf-8"))
        trajectories.append(payload.get("trajectory", payload))
        sources.append({
            "path": str(args.append_episode.resolve()),
            "sha256": _sha256(args.append_episode),
        })
    result = {
        "schema_version": 1,
        "study": "paper8_5_recent_frontier_information_flow_oracle",
        "claim_scope": (
            "offline structural/oracle alignment and token opportunity; not autonomous quality"
        ),
        "tokenizer_identity": tokenizer_identity,
        "sources": sources,
        "synthetic_cases": synthetic_oracle_cases(count_tokens=count_tokens),
        "real_chain": (
            analyze_independent_trajectories(
                trajectories, count_tokens=count_tokens
            )
            if trajectories else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "real_issue_count": len(trajectories),
        "synthetic_case_count": len(result["synthetic_cases"]),
    }))


if __name__ == "__main__":
    main()
