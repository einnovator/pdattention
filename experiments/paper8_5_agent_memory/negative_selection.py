"""Auditable negative-selection heuristics for coding-agent histories.

These rules retire complete action--observation groups.  They are empirical
heuristics, not proofs of behavioral irrelevance.  The stronger variants use
resource/version/span witnesses; the intentionally aggressive variants remain
visibly labelled so they cannot be mistaken for DAG-certified exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import re
from typing import Callable, Iterable, Mapping

from .model import (
    AgentMemoryBudget,
    AgentMemoryExclusion,
    AgentMemoryPlan,
    AgentRecord,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
)
from .recordizer import extract_resource_ids
from .selectors import FullHistorySelector, TokenCounter, whitespace_tokens


class BashOperation(str, Enum):
    SEARCH_DISCOVERY = "search_discovery"
    READ = "read"
    WRITE = "write"
    DIFF = "diff"
    VERIFY = "verify"
    OTHER = "other"


class NegativeRule(str, Enum):
    H1_SEARCH_CONSUMED = "H1_SEARCH_CONSUMED"
    H2A_WRITE_CURRENT_READ = "H2A_WRITE_CURRENT_READ"
    H2B_VERIFIED_WRITE = "H2B_VERIFIED_WRITE"
    H2_BARE_AGGRESSIVE = "H2_BARE_AGGRESSIVE"
    H3_READ_SUPERSEDED = "H3_READ_SUPERSEDED"
    H4_WORKING_SET = "H4_WORKING_SET"


NEGATIVE_POLICY_RULES: Mapping[str, tuple[NegativeRule, ...]] = {
    "h1_search_consumed": (NegativeRule.H1_SEARCH_CONSUMED,),
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
class ResourceAccess:
    resource_id: str
    version: str
    span_kind: str
    span_start: int | None = None
    span_end: int | None = None
    signature: str | None = None


@dataclass(frozen=True)
class TurnSemantics:
    turn: AgentTurn
    action: AgentRecord
    observations: tuple[AgentRecord, ...]
    operation: BashOperation
    accesses: tuple[ResourceAccess, ...]
    discovered_resources: tuple[str, ...]
    changed_resources: tuple[str, ...]
    successful: bool
    observation_complete: bool

    @property
    def resources(self) -> tuple[str, ...]:
        values = [row.resource_id for row in self.accesses]
        values.extend(self.discovered_resources)
        return tuple(dict.fromkeys(values))


_SEARCH = re.compile(
    r"(?:^|[;&|]\s*)(?:find\b|fd\b|rg\s+--files\b|"
    r"(?:grep|rg)\b[^\n]*(?:\s-(?:[^\s]*l[^\s]*|files-with-matches)\b))"
)
_READ = re.compile(r"(?:^|[;&|]\s*)(?:cat|head|tail|less|more|sed\s+-n|grep|rg)\b")
_WRITE = re.compile(
    r"(?:apply_patch|sed\s+-i|perl\s+-pi|git\s+apply|patch\s+-p|"
    r"(?:write_text|write_bytes|open\([^)]*,\s*['\"]w)|>{1,2}\s*)"
)
_DIFF = re.compile(r"(?:^|[;&|]\s*)git\s+(?:diff|show)\b")
_VERIFY = re.compile(
    r"(?:^|[;&|]\s*)(?:pytest|tox|nox|python\s+-m\s+(?:pytest|unittest)|"
    r"make\s+(?:test|check|lint)|ruff|mypy|npm\s+test|cargo\s+test)\b"
)
_SED_SPAN = re.compile(r"sed\s+-n\s+['\"]?(\d+)\s*,\s*(\d+)p")
_HEAD_SPAN = re.compile(r"head(?:\s+-n)?\s+(\d+)\b")
_GREP_PATTERN = re.compile(r"(?:grep|rg)\s+(?:-[^\s]+\s+)*(['\"]?[^\s'\"]+['\"]?)")


def classify_bash_operation(command: str | None) -> BashOperation:
    command = command or ""
    if _WRITE.search(command):
        return BashOperation.WRITE
    if _DIFF.search(command):
        return BashOperation.DIFF
    if _VERIFY.search(command):
        return BashOperation.VERIFY
    if _SEARCH.search(command):
        return BashOperation.SEARCH_DISCOVERY
    if _READ.search(command):
        return BashOperation.READ
    return BashOperation.OTHER


def _normalize_resource(value: str) -> str:
    value = value.strip("'\"`[](){}:,;").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _command_resources(command: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        normalized
        for value in extract_resource_ids(command, "")
        if (normalized := _normalize_resource(value))
    ))


def _metadata_versions(
    observations: tuple[AgentRecord, ...], key: str
) -> dict[str, str]:
    for observation in reversed(observations):
        value = observation.metadata.get(key)
        if isinstance(value, Mapping):
            return {_normalize_resource(str(k)): str(v) for k, v in value.items()}
    return {}


def _span(command: str, resource: str) -> tuple[str, int | None, int | None, str | None]:
    if re.search(r"(?:^|[;&|]\s*)cat\b", command):
        return "whole", None, None, None
    if match := _SED_SPAN.search(command):
        return "lines", int(match.group(1)), int(match.group(2)), None
    if match := _HEAD_SPAN.search(command):
        return "lines", 1, int(match.group(1)), None
    if re.search(r"(?:^|[;&|]\s*)tail\b", command):
        digest = hashlib.sha256(command.encode()).hexdigest()[:12]
        return "query", None, None, f"tail:{resource}:{digest}"
    if re.search(r"(?:^|[;&|]\s*)(?:grep|rg)\b", command):
        match = _GREP_PATTERN.search(command)
        query = match.group(1).strip("'\"") if match else command
        digest = hashlib.sha256(query.encode()).hexdigest()[:12]
        return "query", None, None, f"grep:{resource}:{digest}"
    return "unknown", None, None, None


def _search_signature(command: str) -> str:
    normalized = " ".join(command.split())
    return "search:" + hashlib.sha256(normalized.encode()).hexdigest()[:12]


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
        operation = classify_bash_operation(action.command)
        resources = _command_resources(action.command)
        pre_versions = _metadata_versions(observations, "resource_version_fingerprints")
        post_versions = _metadata_versions(
            observations, "post_resource_version_fingerprints"
        )
        accesses: list[ResourceAccess] = []
        changed_resources: list[str] = []
        for resource in resources:
            if operation == BashOperation.WRITE:
                version = post_versions.get(resource)
                epochs[resource] = epochs.get(resource, 0) + 1
                before = pre_versions.get(resource)
                if version not in (None, "missing") and version != before:
                    changed_resources.append(resource)
            else:
                version = pre_versions.get(resource)
            version = version or f"inferred-epoch:{epochs.get(resource, 0)}"
            kind, start, end, signature = _span(action.command or "", resource)
            # An explicitly truncated tool result cannot establish whole-file
            # or complete query coverage.  Missing completeness metadata is
            # still usable in the labelled heuristic tier, but an observed
            # negative receipt must fail closed.
            if not observation_complete and operation in {
                BashOperation.SEARCH_DISCOVERY,
                BashOperation.READ,
                BashOperation.DIFF,
            }:
                kind, start, end, signature = "unknown", None, None, None
            accesses.append(ResourceAccess(resource, version, kind, start, end, signature))
        discovered = ()
        if operation == BashOperation.SEARCH_DISCOVERY:
            output_resources = (
                _normalize_resource(value)
                for observation in observations
                for value in extract_resource_ids(None, observation.content)
            )
            discovered = tuple(dict.fromkeys((
                _search_signature(action.command or ""),
                *(value for value in output_resources if value),
            )))
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
                dependencies.update(_metadata_resource_ids(
                    later.observations, "verification_resource_ids"
                ))
                dependencies.update(_metadata_resource_ids(
                    later.observations, "dependency_resource_ids"
                ))
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
            or record.has_role(AgentRecordRole.MUTATION)
            or record.has_role(AgentRecordRole.ERROR_OR_REJECTION)
            for resource in record.resource_ids
        }
        pinned.update(_metadata_resource_ids(history.records, "dependency_resource_ids"))
        pinned = {_normalize_resource(resource) for resource in pinned}
        for older in reads:
            resources = set(older.resources)
            if resources and resources.isdisjoint(active) and resources.isdisjoint(pinned):
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
            )
            mandatory_ids = {
                row.record_id for row in history.records
                if row.has_role(AgentRecordRole.SYSTEM)
                or row.has_role(AgentRecordRole.TASK)
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
    hidden = {resource for row in exclusions for resource in row.resource_ids}
    return tuple(sorted(requested.intersection(hidden)))
