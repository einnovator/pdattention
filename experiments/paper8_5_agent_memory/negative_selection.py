"""Auditable negative-selection heuristics for coding-agent histories.

These rules retire complete action--observation groups.  They are empirical
heuristics, not proofs of behavioral irrelevance.  The stronger variants use
resource/version/span witnesses; the intentionally aggressive variants remain
visibly labelled so they cannot be mistaken for DAG-certified exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable, Mapping

from pra_hf.tool_semantics import OperationKind, ResourceAccess

from .model import (
    AgentMemoryBudget,
    AgentMemoryExclusion,
    AgentMemoryPlan,
    AgentRecord,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
)
from .miniswe_semantics import (
    classify_bash_operation,
    extract_resource_ids,
    normalize_resource as _normalize_resource,
    search_signature as _search_signature,
)
from .selectors import FullHistorySelector, TokenCounter, whitespace_tokens

BashOperation = OperationKind


class NegativeRule(str, Enum):
    H1_SEARCH_CONSUMED = "H1_SEARCH_CONSUMED"
    H1_ALL_BRANCHES_CONSUMED = "H1_ALL_BRANCHES_CONSUMED"
    H2A_WRITE_CURRENT_READ = "H2A_WRITE_CURRENT_READ"
    H2B_VERIFIED_WRITE = "H2B_VERIFIED_WRITE"
    H2_BARE_AGGRESSIVE = "H2_BARE_AGGRESSIVE"
    H3_READ_SUPERSEDED = "H3_READ_SUPERSEDED"
    H4_WORKING_SET = "H4_WORKING_SET"


NEGATIVE_POLICY_RULES: Mapping[str, tuple[NegativeRule, ...]] = {
    "task_aware_progress_spine_v4": (
        NegativeRule.H1_ALL_BRANCHES_CONSUMED,
        NegativeRule.H2A_WRITE_CURRENT_READ,
        NegativeRule.H2B_VERIFIED_WRITE,
        NegativeRule.H3_READ_SUPERSEDED,
    ),
    "h1_search_consumed": (NegativeRule.H1_SEARCH_CONSUMED,),
    "h1_all_branches_consumed_strict": (
        NegativeRule.H1_ALL_BRANCHES_CONSUMED,
    ),
    "h2a_write_current_read": (NegativeRule.H2A_WRITE_CURRENT_READ,),
    "h2b_verified_write": (NegativeRule.H2B_VERIFIED_WRITE,),
    "h2_bare_aggressive": (NegativeRule.H2_BARE_AGGRESSIVE,),
    "h3_read_superseded": (NegativeRule.H3_READ_SUPERSEDED,),
    "h4_working_set": (NegativeRule.H4_WORKING_SET,),
    "safe2_h1_h3": (
        NegativeRule.H1_SEARCH_CONSUMED,
        NegativeRule.H3_READ_SUPERSEDED,
    ),
    "h2a_h3": (
        NegativeRule.H2A_WRITE_CURRENT_READ,
        NegativeRule.H3_READ_SUPERSEDED,
    ),
    "h2b_h3": (
        NegativeRule.H2B_VERIFIED_WRITE,
        NegativeRule.H3_READ_SUPERSEDED,
    ),
    "safe3_h1_h3_h2b": (
        NegativeRule.H1_SEARCH_CONSUMED,
        NegativeRule.H3_READ_SUPERSEDED,
        NegativeRule.H2B_VERIFIED_WRITE,
    ),
    "all_h1_h2a_h2b_h3_h4": (
        NegativeRule.H1_SEARCH_CONSUMED,
        NegativeRule.H2A_WRITE_CURRENT_READ,
        NegativeRule.H2B_VERIFIED_WRITE,
        NegativeRule.H3_READ_SUPERSEDED,
        NegativeRule.H4_WORKING_SET,
    ),
}


@dataclass(frozen=True)
class NegativeSelectionConfig:
    rules: tuple[NegativeRule, ...]
    search_delay_turns: int = 0
    write_delay_turns: int = 1
    same_span_reads_to_keep: int = 1
    working_set_resources: int = 4
    protected_head_turns: int = 1
    protected_tail_turns: int = 2
    h2_require_version_match: bool = True
    h2b_allow_workspace_verification: bool = False

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("at least one negative-selection rule is required")
        if self.search_delay_turns < 0 or self.write_delay_turns < 0:
            raise ValueError("Kf and Kw delays cannot be negative")
        if self.same_span_reads_to_keep < 1:
            raise ValueError("Kr must be at least one")
        if self.working_set_resources < 1:
            raise ValueError("Kx must be at least one")
        if self.protected_head_turns < 0 or self.protected_tail_turns < 0:
            raise ValueError("protected head/tail counts cannot be negative")


@dataclass(frozen=True)
class TurnSemantics:
    turn: AgentTurn
    action: AgentRecord
    observations: tuple[AgentRecord, ...]
    operation: OperationKind
    accesses: tuple[ResourceAccess, ...]
    discovered_resources: tuple[str, ...]
    changed_resources: tuple[str, ...]
    successful: bool
    observation_complete: bool
    resource_scope: str | None = None

    @property
    def resources(self) -> tuple[str, ...]:
        values = [row.resource_id for row in self.accesses]
        values.extend(self.discovered_resources)
        return tuple(dict.fromkeys(values))


def _command_resources(command: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        normalized
        for value in extract_resource_ids(command, "")
        if (normalized := _normalize_resource(value))
    ))


def _declared_values(rows: Iterable[AgentRecord], key: str) -> tuple[str, ...]:
    values: list[str] = []
    for row in rows:
        raw = row.metadata.get(key)
        if isinstance(raw, (list, tuple)):
            values.extend(_normalize_resource(str(value)) for value in raw)
    return tuple(dict.fromkeys(value for value in values if value))


def _declared_operation(rows: Iterable[AgentRecord]) -> OperationKind:
    for row in rows:
        value = row.metadata.get("operation_kind")
        if value is not None:
            try:
                return OperationKind(str(value))
            except ValueError:
                return OperationKind.UNKNOWN
    return OperationKind.UNKNOWN


def _declared_accesses(rows: Iterable[AgentRecord]) -> tuple[ResourceAccess, ...]:
    accesses: list[ResourceAccess] = []
    for row in rows:
        value = row.metadata.get("resource_accesses")
        if not isinstance(value, (list, tuple)):
            continue
        for raw in value:
            if isinstance(raw, Mapping):
                try:
                    access = ResourceAccess.from_mapping(raw)
                except (KeyError, TypeError, ValueError):
                    continue
                accesses.append(replace(
                    access, resource_id=_normalize_resource(access.resource_id)
                ))
    # Observation declarations supersede duplicate action declarations.
    unique: dict[tuple[str, str, int | None, int | None, str | None], ResourceAccess] = {}
    for access in accesses:
        key = (
            access.resource_id,
            access.span_kind,
            access.span_start,
            access.span_end,
            access.signature,
        )
        unique[key] = access
    return tuple(unique.values())


def _runtime_resource_scope(rows: Iterable[AgentRecord]) -> str | None:
    """Return a stable non-task scope for resource identity.

    Relative paths such as ``setup.cfg`` recur across unrelated repositories.
    The instrumented runtime already supplies an environment fingerprint and
    cwd; omitting that scope creates false DAG edges between different
    workspaces.  Missing scope deliberately preserves the legacy identity and
    therefore cannot be used as cross-workspace proof.
    """

    for row in reversed(tuple(rows)):
        environment = row.metadata.get("environment_fingerprint")
        if not isinstance(environment, str) or not environment:
            continue
        cwd = row.metadata.get("cwd")
        return f"env={environment}|cwd={cwd if isinstance(cwd, str) else '?'}"
    return None


def _scoped_resource(resource: str, scope: str | None) -> str:
    return f"{scope}::{resource}" if scope else resource


def _unscoped_resource(resource: str) -> str:
    return resource.split("::", 1)[1] if "::" in resource else resource


def _turn_semantics(history: CanonicalAgentHistory) -> list[TurnSemantics]:
    records = history.record_by_id
    epochs: dict[str, int] = {}
    rows: list[TurnSemantics] = []
    for turn in history.turns:
        members = tuple(records[rid] for rid in turn.record_ids)
        action = next((row for row in members if row.has_role(
            AgentRecordRole.ASSISTANT_ACTION
        )), None)
        if action is None:
            continue
        observations = tuple(row for row in members if row.has_role(
            AgentRecordRole.TOOL_OBSERVATION
        ))
        observation_complete = bool(observations) and all(
            row.metadata.get("output_complete") is not False
            and row.metadata.get("output_truncated") is not True
            and row.metadata.get("timed_out") is not True
            for row in observations
        )
        declared_rows = (action, *observations)
        resource_scope = _runtime_resource_scope(declared_rows)
        operation = _declared_operation(declared_rows)
        accesses = [
            replace(
                access,
                resource_id=_scoped_resource(access.resource_id, resource_scope),
            )
            for access in _declared_accesses(declared_rows)
        ]
        resources = tuple(dict.fromkeys(
            access.resource_id for access in accesses
        )) or tuple(dict.fromkeys(
            _scoped_resource(_normalize_resource(value), resource_scope)
            for row in declared_rows for value in row.resource_ids
        ))
        if not accesses:
            accesses = [
                ResourceAccess(resource, f"inferred-epoch:{epochs.get(resource, 0)}")
                for resource in resources
            ]
        if not observation_complete and operation in {
            OperationKind.SEARCH_DISCOVERY, OperationKind.READ, OperationKind.DIFF,
        }:
            accesses = [replace(
                access,
                span_kind="unknown",
                span_start=None,
                span_end=None,
                signature=None,
            ) for access in accesses]
        changed_resources = [
            _scoped_resource(resource, resource_scope)
            for resource in _declared_values(declared_rows, "changed_resource_ids")
        ]
        if operation == OperationKind.WRITE:
            for resource in resources:
                epochs[resource] = epochs.get(resource, 0) + 1
        discovered = tuple(
            _scoped_resource(resource, resource_scope)
            for resource in _declared_values(declared_rows, "discovered_resource_ids")
        )
        rows.append(TurnSemantics(
            turn,
            action,
            observations,
            operation,
            tuple(accesses),
            discovered,
            tuple(changed_resources),
            bool(observations) and all(row.return_code in (None, 0) for row in observations),
            observation_complete,
            resource_scope,
        ))
    return rows


def _covers(newer: ResourceAccess, older: ResourceAccess) -> bool:
    if newer.resource_id != older.resource_id or newer.version != older.version:
        return False
    if newer.span_kind == "whole":
        return True
    if newer.span_kind == older.span_kind == "lines":
        assert newer.span_start is not None and newer.span_end is not None
        assert older.span_start is not None and older.span_end is not None
        return newer.span_start <= older.span_start and newer.span_end >= older.span_end
    if newer.span_kind == older.span_kind == "query":
        return newer.signature == older.signature
    return False


def _metadata_resource_ids(rows: Iterable[AgentRecord], key: str) -> set[str]:
    result: set[str] = set()
    for row in rows:
        value = row.metadata.get(key)
        if isinstance(value, (list, tuple)):
            result.update(_normalize_resource(str(item)) for item in value)
    return result


def _complete_successful_observation(semantics: TurnSemantics) -> bool:
    """Require a complete causal result and complete, non-truncated output."""

    return bool(
        semantics.turn.complete
        and semantics.successful
        and semantics.observations
        and all(
            row.metadata.get("output_complete") is True
            and row.metadata.get("timed_out") is not True
            and row.metadata.get("output_truncated") is not True
            for row in semantics.observations
        )
    )


def _candidate(
    *,
    history: CanonicalAgentHistory,
    semantics: TurnSemantics,
    rule: NegativeRule,
    classification: str,
    reason: str,
    resources: Iterable[str],
    witnesses: Iterable[str],
    count_tokens: TokenCounter,
) -> AgentMemoryExclusion:
    record_ids = semantics.turn.record_ids
    tokens = sum(count_tokens(history.record_by_id[rid].content) for rid in record_ids)
    resources = tuple(dict.fromkeys(resources))
    witness_ids = tuple(dict.fromkeys(witnesses))
    tombstone = (
        f"INACTIVE group={semantics.turn.causal_group_id} rule={rule.value} "
        f"resources={','.join(resources) or '-'} witnesses={','.join(witness_ids) or '-'}"
    )
    return AgentMemoryExclusion(
        semantics.turn.causal_group_id,
        record_ids,
        rule.value,
        classification,
        reason,
        resources,
        witness_ids,
        tombstone,
        tokens,
    )


def build_negative_exclusions(
    history: CanonicalAgentHistory,
    config: NegativeSelectionConfig,
    *,
    count_tokens: TokenCounter = whitespace_tokens,
) -> tuple[AgentMemoryExclusion, ...]:
    turns = _turn_semantics(history)
    records = history.record_by_id
    candidates: list[AgentMemoryExclusion] = []
    enabled = set(config.rules)

    if NegativeRule.H1_SEARCH_CONSUMED in enabled:
        for index, older in enumerate(turns):
            if (
                older.operation != BashOperation.SEARCH_DISCOVERY
                or not older.successful
                or not older.observation_complete
            ):
                continue
            discovered = set(older.discovered_resources)
            consumed = next((
                (later_index, later)
                for later_index, later in enumerate(turns[index + 1 :], start=index + 1)
                if later.operation in {BashOperation.READ, BashOperation.DIFF}
                and later.successful
                and later.observation_complete
                and discovered.intersection(later.resources)
            ), None)
            if consumed is None:
                continue
            consumed_index, witness = consumed
            # A concrete read establishes that the search result was consumed,
            # but the requested H1 transition also requires the trajectory to
            # move on to a non-discovery/non-read operation.  Without this
            # guard, a still-active search/read exploration chain would be
            # retired prematurely merely because its first candidate was
            # opened.
            transition = next((
                (later_index, later)
                for later_index, later in enumerate(
                    turns[consumed_index + 1 :], start=consumed_index + 1
                )
                if later.operation not in {
                    BashOperation.SEARCH_DISCOVERY,
                    BashOperation.READ,
                    BashOperation.DIFF,
                }
            ), None)
            if transition is None:
                continue
            transition_index, transition_witness = transition
            if len(turns) - 1 - transition_index < config.search_delay_turns:
                continue
            candidates.append(_candidate(
                history=history,
                semantics=older,
                rule=NegativeRule.H1_SEARCH_CONSUMED,
                classification="high_confidence_supersession_not_certificate",
                reason=(
                    "discovery output was consumed by a concrete downstream read "
                    "and followed by a non-read transition; "
                    f"Kf={config.search_delay_turns} completed later turns elapsed"
                ),
                resources=older.discovered_resources,
                witnesses=(
                    *older.turn.record_ids,
                    *witness.turn.record_ids,
                    *transition_witness.turn.record_ids,
                ),
                count_tokens=count_tokens,
            ))

    if NegativeRule.H1_ALL_BRANCHES_CONSUMED in enabled:
        for index, older in enumerate(turns):
            if (
                older.operation != BashOperation.SEARCH_DISCOVERY
                or not _complete_successful_observation(older)
            ):
                continue
            concrete = {
                resource for resource in older.discovered_resources
                if not _unscoped_resource(resource).startswith("search:")
            }
            if not concrete:
                continue
            consumers: dict[str, TurnSemantics] = {}
            consumer_indexes: dict[str, int] = {}
            for later_index, later in enumerate(turns[index + 1 :], start=index + 1):
                if (
                    later.operation not in {BashOperation.READ, BashOperation.DIFF}
                    or not _complete_successful_observation(later)
                ):
                    continue
                for resource in concrete.intersection(later.resources):
                    consumers.setdefault(resource, later)
                    consumer_indexes.setdefault(resource, later_index)
            if set(consumers) != concrete:
                continue
            final_consumer_index = max(consumer_indexes.values())
            transition = next((
                (later_index, later)
                for later_index, later in enumerate(
                    turns[final_consumer_index + 1 :],
                    start=final_consumer_index + 1,
                )
                if later.operation not in {
                    BashOperation.SEARCH_DISCOVERY,
                    BashOperation.READ,
                    BashOperation.DIFF,
                }
                and later.turn.complete
            ), None)
            if transition is None:
                continue
            transition_index, transition_turn = transition
            if len(turns) - 1 - transition_index < config.search_delay_turns:
                continue
            witnesses = [*older.turn.record_ids]
            for resource in sorted(concrete):
                witnesses.extend(consumers[resource].turn.record_ids)
            witnesses.extend(transition_turn.turn.record_ids)
            candidates.append(_candidate(
                history=history,
                semantics=older,
                rule=NegativeRule.H1_ALL_BRANCHES_CONSUMED,
                classification="strict_all_branch_supersession_not_certificate",
                reason=(
                    "every concrete discovered resource has a complete successful "
                    "read/diff, followed by a non-read transition; "
                    f"Kf={config.search_delay_turns} completed later turns elapsed"
                ),
                resources=older.discovered_resources,
                witnesses=witnesses,
                count_tokens=count_tokens,
            ))

    writes = [
        (index, row) for index, row in enumerate(turns)
        if row.operation == BashOperation.WRITE and row.successful
    ]
    if NegativeRule.H2A_WRITE_CURRENT_READ in enabled:
        for index, write in writes:
            if not write.changed_resources:
                continue
            for later in turns[index + 1 :]:
                shared = set(write.changed_resources).intersection(later.resources)
                if not shared:
                    continue
                if later.operation == BashOperation.WRITE:
                    break
                if (
                    later.operation not in {BashOperation.READ, BashOperation.DIFF}
                    or not later.successful
                    or not later.observation_complete
                ):
                    continue
                if config.h2_require_version_match:
                    write_versions = {row.resource_id: row.version for row in write.accesses}
                    later_versions = {row.resource_id: row.version for row in later.accesses}
                    shared = {
                        resource for resource in shared
                        if write_versions.get(resource) == later_versions.get(resource)
                        and not str(write_versions.get(resource, "")).startswith("inferred-")
                    }
                if not shared:
                    continue
                candidates.append(_candidate(
                    history=history,
                    semantics=write,
                    rule=NegativeRule.H2A_WRITE_CURRENT_READ,
                    classification="version_guarded_supersession_not_certificate",
                    reason="write payload is represented by a later current-version read/diff",
                    resources=shared,
                    witnesses=(*write.turn.record_ids, *later.turn.record_ids),
                    count_tokens=count_tokens,
                ))
                break

    if NegativeRule.H2B_VERIFIED_WRITE in enabled:
        for index, write in writes:
            if not write.changed_resources:
                continue
            for later in turns[index + 1 :]:
                if later.operation == BashOperation.WRITE and set(write.changed_resources).intersection(
                    later.resources
                ):
                    break
                if (
                    later.operation != BashOperation.VERIFY
                    or not later.successful
                    or not later.observation_complete
                ):
                    continue
                dependencies = set(later.resources)
                dependencies.update(
                    _scoped_resource(resource, later.resource_scope)
                    for resource in _metadata_resource_ids(
                        later.observations, "verification_resource_ids"
                    )
                )
                dependencies.update(
                    _scoped_resource(resource, later.resource_scope)
                    for resource in _metadata_resource_ids(
                        later.observations, "dependency_resource_ids"
                    )
                )
                shared = set(write.changed_resources).intersection(dependencies)
                if not shared and not config.h2b_allow_workspace_verification:
                    continue
                candidates.append(_candidate(
                    history=history,
                    semantics=write,
                    rule=NegativeRule.H2B_VERIFIED_WRITE,
                    classification="verified_state_consolidation_heuristic",
                    reason=(
                        "successful verification explicitly depends on the written resource"
                        if shared else
                        "successful workspace verification followed the write (relaxed arm)"
                    ),
                    resources=shared or write.changed_resources,
                    witnesses=(*write.turn.record_ids, *later.turn.record_ids),
                    count_tokens=count_tokens,
                ))
                break

    if NegativeRule.H2_BARE_AGGRESSIVE in enabled:
        for index, write in writes:
            if len(turns) - 1 - index < config.write_delay_turns:
                continue
            rewritten = any(
                later.operation == BashOperation.WRITE
                and set(write.changed_resources or write.resources).intersection(later.resources)
                for later in turns[index + 1 :]
            )
            if not rewritten:
                candidates.append(_candidate(
                    history=history,
                    semantics=write,
                    rule=NegativeRule.H2_BARE_AGGRESSIVE,
                    classification="aggressive_unsafe_ablation",
                    reason=(
                        "no later same-resource write was observed; this does not imply "
                        f"the mutation was correct (Kw={config.write_delay_turns})"
                    ),
                    resources=write.changed_resources or write.resources,
                    witnesses=write.turn.record_ids,
                    count_tokens=count_tokens,
                ))

    if NegativeRule.H3_READ_SUPERSEDED in enabled:
        reads = [
            row for row in turns
            if row.operation == BashOperation.READ
            and row.successful
            and row.observation_complete
        ]
        for index, older in enumerate(reads):
            if not older.accesses:
                continue
            witnesses: list[str] = []
            all_covered = True
            for access in older.accesses:
                covering = [
                    later for later in reads[index + 1 :]
                    if any(_covers(candidate, access) for candidate in later.accesses)
                ]
                if len(covering) < config.same_span_reads_to_keep:
                    all_covered = False
                    break
                for witness in covering[-config.same_span_reads_to_keep :]:
                    witnesses.extend(witness.turn.record_ids)
            if all_covered:
                candidates.append(_candidate(
                    history=history,
                    semantics=older,
                    rule=NegativeRule.H3_READ_SUPERSEDED,
                    classification="version_span_supersession_heuristic",
                    reason=(
                        "each observed resource/version/span is covered by at least "
                        f"Kr={config.same_span_reads_to_keep} newer reads"
                    ),
                    resources=older.resources,
                    witnesses=(*older.turn.record_ids, *witnesses),
                    count_tokens=count_tokens,
                ))

    if NegativeRule.H4_WORKING_SET in enabled:
        reads = [
            row for row in turns
            if row.operation == BashOperation.READ
            and row.successful
            and row.observation_complete
        ]
        active: list[str] = []
        for row in reversed(reads):
            for resource in reversed(row.resources):
                if resource not in active:
                    active.append(resource)
                if len(active) >= config.working_set_resources:
                    break
            if len(active) >= config.working_set_resources:
                break
        pinned = {
            resource
            for record in history.records
            if record.has_role(AgentRecordRole.TASK)
            or record.has_role(AgentRecordRole.USER_INPUT)
            or record.has_role(AgentRecordRole.MUTATION)
            or record.has_role(AgentRecordRole.ERROR_OR_REJECTION)
            for resource in record.resource_ids
        }
        pinned.update(_metadata_resource_ids(history.records, "dependency_resource_ids"))
        pinned = {_normalize_resource(resource) for resource in pinned}
        pinned_unscoped = {_unscoped_resource(resource) for resource in pinned}
        for older in reads:
            resources = set(older.resources)
            if (
                resources
                and resources.isdisjoint(active)
                and resources.isdisjoint(pinned)
                and {
                    _unscoped_resource(resource) for resource in resources
                }.isdisjoint(pinned_unscoped)
            ):
                candidates.append(_candidate(
                    history=history,
                    semantics=older,
                    rule=NegativeRule.H4_WORKING_SET,
                    classification="capacity_working_set_heuristic",
                    reason=(
                        f"resources fall outside the latest Kx={config.working_set_resources} "
                        "distinct reads and are not dependency-pinned"
                    ),
                    resources=resources,
                    witnesses=(),
                    count_tokens=count_tokens,
                ))

    complete = [turn for turn in history.turns if turn.complete]
    protected = {
        turn.causal_group_id for turn in complete[: config.protected_head_turns]
    }
    if config.protected_tail_turns:
        protected.update(
            turn.causal_group_id for turn in complete[-config.protected_tail_turns :]
        )
    protected.update(
        record.causal_group_id for record in history.records
        if record.has_role(AgentRecordRole.ERROR_OR_REJECTION)
        or record.has_role(AgentRecordRole.TASK)
        or record.has_role(AgentRecordRole.USER_INPUT)
    )
    # One group can satisfy more than one rule. Keep the first rule in the
    # predeclared config order so arm accounting remains mutually exclusive.
    priority = {rule.value: index for index, rule in enumerate(config.rules)}
    candidates.sort(key=lambda row: priority[row.rule_id])
    chosen: dict[str, AgentMemoryExclusion] = {}
    for row in candidates:
        if row.causal_group_id not in protected:
            chosen.setdefault(row.causal_group_id, row)
    return tuple(sorted(
        chosen.values(),
        key=lambda row: history.record_by_id[row.record_ids[0]].message_index,
    ))


class NegativeHeuristicSelector:
    """Exclude heuristic groups, then optionally apply a positive selector."""

    def __init__(
        self,
        config: NegativeSelectionConfig,
        fallback_selector: object | None = None,
    ) -> None:
        self.config = config
        self.fallback_selector = fallback_selector

    def select(
        self,
        *,
        history: CanonicalAgentHistory,
        query: str,
        budget: AgentMemoryBudget,
        count_tokens: Callable[[str], int] = whitespace_tokens,
    ) -> AgentMemoryPlan:
        exclusions = build_negative_exclusions(
            history, self.config, count_tokens=count_tokens
        )
        excluded_groups = {row.causal_group_id for row in exclusions}
        filtered = CanonicalAgentHistory(
            tuple(
                row for row in history.records
                if row.causal_group_id not in excluded_groups
            ),
            tuple(
                row for row in history.turns
                if row.causal_group_id not in excluded_groups
            ),
        )
        selector = self.fallback_selector or FullHistorySelector()
        plan = selector.select(
            history=filtered,
            query=query,
            budget=budget,
            count_tokens=count_tokens,
        )
        label = "+".join(rule.value.lower() for rule in self.config.rules)
        if self.fallback_selector is not None:
            label += f"+{plan.policy}"
        replacements: dict[str, object] = {}
        if self.fallback_selector is None:
            complete = [turn for turn in history.turns if turn.complete]
            head = complete[: self.config.protected_head_turns]
            head_ids = {turn.turn_id for turn in head}
            tail = [
                turn for turn in complete[-self.config.protected_tail_turns :]
                if turn.turn_id not in head_ids
            ] if self.config.protected_tail_turns else []
            protected_groups = {
                turn.causal_group_id for turn in (*head, *tail)
            }
            protected_groups.update(
                row.causal_group_id for row in history.records
                if row.has_role(AgentRecordRole.ERROR_OR_REJECTION)
                or row.has_role(AgentRecordRole.TASK)
                or row.has_role(AgentRecordRole.USER_INPUT)
            )
            mandatory_ids = {
                row.record_id for row in history.records
                if row.has_role(AgentRecordRole.SYSTEM)
                or row.has_role(AgentRecordRole.TASK)
                or row.has_role(AgentRecordRole.USER_INPUT)
                or row.causal_group_id in protected_groups
            }
            selected_middle = [
                turn for turn in filtered.turns
                if turn.turn_id not in head_ids
                and turn.causal_group_id not in {row.causal_group_id for row in tail}
            ]
            replacements.update({
                "mandatory_tokens": sum(
                    count_tokens(history.record_by_id[rid].content)
                    for rid in mandatory_ids
                ),
                "mandatory_overflow_tokens": max(
                    0,
                    sum(
                        count_tokens(history.record_by_id[rid].content)
                        for rid in mandatory_ids
                    ) - budget.max_tokens,
                ),
                "head_turns": len(head),
                "tail_turns": len(tail),
                "middle_candidate_turns": max(0, len(complete) - len(head) - len(tail)),
                "middle_selected_turns": len(selected_middle),
            })
        return replace(
            plan,
            policy=label,
            full_history_tokens=sum(count_tokens(row.content) for row in history.records),
            exclusions=exclusions,
            **replacements,
        )


def reacquired_excluded_resources(
    command: str | None,
    exclusions: Iterable[AgentMemoryExclusion],
) -> tuple[str, ...]:
    """Return hidden resources immediately reacquired by a read/search action."""

    operation = classify_bash_operation(command)
    if operation not in {BashOperation.SEARCH_DISCOVERY, BashOperation.READ, BashOperation.DIFF}:
        return ()
    requested = set(_command_resources(command))
    if operation == BashOperation.SEARCH_DISCOVERY:
        requested.add(_search_signature(command or ""))
    hidden = {
        _unscoped_resource(resource)
        for row in exclusions for resource in row.resource_ids
    }
    return tuple(sorted(requested.intersection(hidden)))
