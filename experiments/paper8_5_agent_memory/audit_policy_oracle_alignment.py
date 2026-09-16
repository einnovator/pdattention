"""Audit persistent-memory policies against an evaluator-only structural oracle.

This is deliberately an offline sanity check, not an autonomous quality run.
The selector sees the same boundary-free transcript used by the experiment.
Only the scorer uses the source-episode ledger to answer three basic questions:

* did the policy remove interaction detail from the active issue;
* how much removable prior-issue detail did it leave behind; and
* did it preserve a terminal closure turn for every pinned prior user prompt.

The independent-cross-workspace oracle retires every completed assistant/tool
group except the latest finalization group from each prior issue.  It is an
upper-bound structural oracle, not a claim of LLM behavioral irrelevance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import AgentMemoryBudget, AgentRecord, AgentRecordRole
from .multi_issue_session import BoundaryMode, compose_multi_issue_session
from .recordizer import recordize_replay_messages
from .selectors import (
    PersistentGlobalRetirementConfig,
    PersistentGlobalRetirementSelector,
    PersistentInstructionEpochRetirementConfig,
    PersistentInstructionEpochRetirementSelector,
    TokenCounter,
    whitespace_tokens,
)


@dataclass(frozen=True)
class PolicyAudit:
    policy: str
    selected_tokens: int
    full_tokens: int
    saving_fraction: float
    active_interaction_excluded_tokens: int
    active_interaction_excluded_fraction: float
    prior_interaction_excluded_tokens: int
    prior_interaction_excluded_fraction: float
    oracle_exclusion_precision: float
    oracle_exclusion_recall: float
    orphaned_prior_instruction_epochs: int
    retained_prior_finalization_epochs: int


def _episode_ranges(episodes: Iterable[Mapping[str, Any]]) -> tuple[tuple[int, int, str], ...]:
    ranges = []
    start = 0
    for episode in episodes:
        count = int(episode["model_visible_messages"])
        ranges.append((start, start + count, str(episode["instance_id"])))
        start += count
    return tuple(ranges)


def _episode_for(record: AgentRecord, ranges: tuple[tuple[int, int, str], ...]) -> str:
    for start, end, identity in ranges:
        if start <= record.message_index < end:
            return identity
    raise ValueError(f"record {record.record_id} lies outside the episode ledger")


def _is_interaction(record: AgentRecord) -> bool:
    return not any(record.has_role(role) for role in (
        AgentRecordRole.SYSTEM,
        AgentRecordRole.TASK,
        AgentRecordRole.USER_INPUT,
    ))


def _tokens(records: Iterable[AgentRecord], count_tokens: TokenCounter) -> int:
    return sum(count_tokens(record.content) for record in records)


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def audit_prefix(
    payload: Mapping[str, Any],
    *,
    count_tokens: TokenCounter = whitespace_tokens,
    tokenizer_identity: str = "whitespace_v1_diagnostic",
) -> dict[str, Any]:
    source_episodes = payload.get("episodes")
    if not isinstance(source_episodes, list) or len(source_episodes) < 2:
        raise ValueError("audit requires a persistent prefix with at least two episodes")
    trajectories = [episode["trajectory"] for episode in source_episodes]
    session = compose_multi_issue_session(
        trajectories,
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        session_id=str(payload.get("session_id") or "oracle-alignment-audit"),
    )
    history = recordize_replay_messages(session["messages"])
    ranges = _episode_ranges(session["episodes"])
    episode_ids = tuple(str(row["instance_id"]) for row in session["episodes"])
    active_episode = episode_ids[-1]
    records_by_episode = {
        identity: tuple(
            record for record in history.records
            if _episode_for(record, ranges) == identity
        )
        for identity in episode_ids
    }
    interaction_ids = {
        identity: {
            record.record_id for record in rows if _is_interaction(record)
        }
        for identity, rows in records_by_episode.items()
    }
    active_ids = interaction_ids[active_episode]
    prior_ids = set().union(*(interaction_ids[row] for row in episode_ids[:-1]))

    finalization_by_episode: dict[str, set[str]] = {}
    for identity in episode_ids[:-1]:
        finalization_turns = [
            turn for turn in history.turns
            if any(
                _episode_for(history.record_by_id[record_id], ranges) == identity
                and history.record_by_id[record_id].has_role(AgentRecordRole.FINALIZATION)
                for record_id in turn.record_ids
            )
        ]
        latest = finalization_turns[-1:] if finalization_turns else []
        finalization_by_episode[identity] = {
            record_id for turn in latest for record_id in turn.record_ids
        }
    closure_ids = set().union(*finalization_by_episode.values())
    oracle_excluded_ids = prior_ids.difference(closure_ids)

    selectors = {
        "GLOBAL_R4_M2_V2_P1": PersistentGlobalRetirementSelector(
            PersistentGlobalRetirementConfig(4, 2, 2, 1)
        ),
        "E0": PersistentInstructionEpochRetirementSelector(
            PersistentInstructionEpochRetirementConfig()
        ),
        "E0_F1": PersistentInstructionEpochRetirementSelector(
            PersistentInstructionEpochRetirementConfig(prior_finalization_turns=1)
        ),
        "E1": PersistentInstructionEpochRetirementSelector(
            PersistentInstructionEpochRetirementConfig(prior_full_epochs=1)
        ),
        "E2": PersistentInstructionEpochRetirementSelector(
            PersistentInstructionEpochRetirementConfig(prior_full_epochs=2)
        ),
    }
    costs = {
        record.record_id: count_tokens(record.content) for record in history.records
    }
    full_tokens = sum(costs.values())
    active_tokens = sum(costs[row] for row in active_ids)
    prior_tokens = sum(costs[row] for row in prior_ids)
    oracle_tokens = sum(costs[row] for row in oracle_excluded_ids)
    audits = []
    for name, selector in selectors.items():
        plan = selector.select(
            history=history,
            query="",
            budget=AgentMemoryBudget(max_tokens=max(1, full_tokens)),
            count_tokens=count_tokens,
        )
        selected = set(plan.selected_record_ids)
        excluded = set(costs).difference(selected)
        active_excluded = active_ids.intersection(excluded)
        prior_excluded = prior_ids.intersection(excluded)
        true_excluded = oracle_excluded_ids.intersection(excluded)
        retained_closures = sum(
            bool(record_ids) and record_ids.issubset(selected)
            for record_ids in finalization_by_episode.values()
        )
        orphaned = sum(
            not record_ids or not record_ids.issubset(selected)
            for record_ids in finalization_by_episode.values()
        )
        audits.append(PolicyAudit(
            policy=name,
            selected_tokens=plan.selected_tokens,
            full_tokens=full_tokens,
            saving_fraction=1.0 - _fraction(plan.selected_tokens, full_tokens),
            active_interaction_excluded_tokens=sum(costs[row] for row in active_excluded),
            active_interaction_excluded_fraction=_fraction(
                sum(costs[row] for row in active_excluded), active_tokens
            ),
            prior_interaction_excluded_tokens=sum(costs[row] for row in prior_excluded),
            prior_interaction_excluded_fraction=_fraction(
                sum(costs[row] for row in prior_excluded), prior_tokens
            ),
            oracle_exclusion_precision=_fraction(
                sum(costs[row] for row in true_excluded),
                sum(costs[row] for row in excluded),
            ),
            oracle_exclusion_recall=_fraction(
                sum(costs[row] for row in true_excluded), oracle_tokens
            ),
            orphaned_prior_instruction_epochs=orphaned,
            retained_prior_finalization_epochs=retained_closures,
        ))

    resource_sets = {
        identity: {
            resource
            for record in records_by_episode[identity]
            for resource in record.resource_ids
        }
        for identity in episode_ids
    }
    environment_sets = {
        identity: {
            str(record.metadata["environment_fingerprint"])
            for record in records_by_episode[identity]
            if record.metadata.get("environment_fingerprint")
        }
        for identity in episode_ids
    }
    active_resources = resource_sets[active_episode]
    active_environments = environment_sets[active_episode]
    overlap = []
    for identity in episode_ids[:-1]:
        raw = resource_sets[identity].intersection(active_resources)
        scoped = {
            (environment, resource)
            for environment in environment_sets[identity]
            for resource in resource_sets[identity]
        }.intersection({
            (environment, resource)
            for environment in active_environments
            for resource in active_resources
        })
        overlap.append({
            "prior_episode": identity,
            "raw_resource_overlap_count": len(raw),
            "raw_resource_overlap": sorted(raw),
            "environment_overlap_count": len(
                environment_sets[identity].intersection(active_environments)
            ),
            "scoped_resource_overlap_count": len(scoped),
        })

    return {
        "schema_version": 1,
        "study": "paper8_5_policy_oracle_alignment_audit",
        "selector_visibility": (
            "boundary-free transcript only; evaluator episode ledger is used only for scoring"
        ),
        "oracle_scope": (
            "independent cross-workspace prior interaction detail, retaining one "
            "terminal finalization causal group per pinned prior user instruction"
        ),
        "tokenizer": tokenizer_identity,
        "episode_ids": episode_ids,
        "active_episode": active_episode,
        "full_tokens": full_tokens,
        "active_interaction_tokens": active_tokens,
        "prior_interaction_tokens": prior_tokens,
        "oracle_excludable_tokens": oracle_tokens,
        "policies": [asdict(row) for row in audits],
        "resource_scope_diagnostic": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    args = parser.parse_args()
    payload = json.loads(args.prefix.read_text(encoding="utf-8"))
    count_tokens = whitespace_tokens
    tokenizer_identity = "whitespace_v1_diagnostic"
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            revision=args.tokenizer_revision,
            local_files_only=Path(args.tokenizer).exists(),
        )
        count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        tokenizer_identity = (
            f"{args.tokenizer}@{args.tokenizer_revision}"
            if args.tokenizer_revision else str(args.tokenizer)
        )
    result = audit_prefix(
        payload,
        count_tokens=count_tokens,
        tokenizer_identity=tokenizer_identity,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
