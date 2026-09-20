"""Portable agent-history policy over declared record and resource metadata.

This module contains no agent names, tool names, shell syntax, or engine code.
Adapters declare operation/resource facts; the planner preserves causal groups,
replaces only eligible observation bodies with size-gated receipts, and may
then apply a conservative middle-history budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping

from .agent_history import (
    AgentRecord,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
    SemanticStatus,
)
from .mediation import HistorySelectionConfig, WireAgentMemoryPlan
from .tool_semantics import OperationKind, ResourceAccess


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class _TurnState:
    turn: AgentTurn
    action: AgentRecord
    observations: tuple[AgentRecord, ...]
    operation: OperationKind
    accesses: tuple[ResourceAccess, ...]
    discovered: tuple[str, ...]
    changed: tuple[str, ...]
    successful: bool
    complete_output: bool

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(access.resource_id for access in self.accesses))


def _values(records: Iterable[AgentRecord], key: str) -> tuple[str, ...]:
    result: list[str] = []
    for record in records:
        value = record.metadata.get(key)
        if isinstance(value, (list, tuple)):
            result.extend(str(row) for row in value)
    return tuple(dict.fromkeys(result))


def _operation(records: Iterable[AgentRecord]) -> OperationKind:
    for record in records:
        value = record.metadata.get("operation_kind")
        if value is not None:
            try:
                return OperationKind(str(value))
            except ValueError:
                return OperationKind.UNKNOWN
    return OperationKind.UNKNOWN


def _accesses(records: Iterable[AgentRecord]) -> tuple[ResourceAccess, ...]:
    result: dict[tuple[Any, ...], ResourceAccess] = {}
    for record in records:
        value = record.metadata.get("resource_accesses")
        if not isinstance(value, (list, tuple)):
            continue
        for row in value:
            if not isinstance(row, Mapping):
                continue
            try:
                access = ResourceAccess.from_mapping(row)
            except (KeyError, TypeError, ValueError):
                continue
            key = (
                access.resource_id, access.span_kind, access.span_start,
                access.span_end, access.signature,
            )
            result[key] = access
    return tuple(result.values())


def _states(history: CanonicalAgentHistory) -> list[_TurnState]:
    records = history.record_by_id
    result: list[_TurnState] = []
    for turn in history.turns:
        members = tuple(records[record_id] for record_id in turn.record_ids)
        action = next((
            row for row in members if row.has_role(AgentRecordRole.ASSISTANT_ACTION)
        ), None)
        if action is None:
            continue
        observations = tuple(
            row for row in members if row.has_role(AgentRecordRole.TOOL_OBSERVATION)
        )
        declarations = (action, *observations)
        result.append(_TurnState(
            turn=turn,
            action=action,
            observations=observations,
            operation=_operation(declarations),
            accesses=_accesses(declarations),
            discovered=_values(declarations, "discovered_resource_ids"),
            changed=_values(declarations, "changed_resource_ids"),
            successful=bool(observations) and all(
                row.return_code in (None, 0)
                and row.semantic_status not in {
                    SemanticStatus.FAILED,
                    SemanticStatus.PARTIAL,
                }
                for row in observations
            ),
            complete_output=bool(observations) and all(
                (row.complete is True or row.metadata.get("output_complete") is True)
                and row.metadata.get("timed_out") is not True
                and row.metadata.get("output_truncated") is not True
                for row in observations
            ),
        ))
    return result


def _covers(newer: ResourceAccess, older: ResourceAccess) -> bool:
    if (
        newer.resource_id != older.resource_id
        or newer.version in {"", "unknown"}
        or newer.version != older.version
    ):
        return False
    if newer.span_kind == "whole":
        return True
    if newer.span_kind == older.span_kind == "lines":
        if None in {newer.span_start, newer.span_end, older.span_start, older.span_end}:
            return False
        return bool(
            newer.span_start <= older.span_start and newer.span_end >= older.span_end
        )
    return bool(
        newer.span_kind == older.span_kind == "query"
        and newer.signature is not None
        and newer.signature == older.signature
    )


def _receipt(
    rule: str,
    state: _TurnState,
    resources: Iterable[str],
    *,
    presentation: str,
) -> str:
    named = ", ".join(sorted(set(resources))) or "none"
    code = state.observations[-1].return_code if state.observations else None
    code = code if code is not None else "unknown"
    if presentation == "metadata_only":
        return f"<returncode>{code}</returncode>\n<output></output>"
    if presentation != "semantic_receipt":
        raise ValueError(
            "history_selection.options.receipt_presentation must be "
            "metadata_only or semantic_receipt"
        )
    fact = {
        "H1": f"earlier discovery was consumed; resources: {named}",
        "H2A": f"mutation action retained; current read confirms: {named}",
        "H2B": f"mutation action retained; later verification passed for: {named}",
        "H3": f"older read superseded by retained current-version evidence: {named}",
    }[rule]
    return f"<returncode>{code}</returncode>\n<output>[PRA memory] {fact}</output>"


class PortableStateAuthorityPlanner:
    """Build the reusable task-aware-progress-spine-v4 wire plan.

    The tokenizer is injected by the hosting runtime.  Receipt compaction is
    therefore activated only when exact accounting proves a strict reduction.
    Unknown or incomplete effects fail closed and remain verbatim.
    """

    supported_policies = {"task-aware-progress-spine-v4"}

    def __init__(self, count_tokens: TokenCounter) -> None:
        self.count_tokens = count_tokens

    def build(
        self,
        history: CanonicalAgentHistory,
        config: HistorySelectionConfig,
    ) -> WireAgentMemoryPlan:
        if config.policy not in self.supported_policies:
            raise ValueError(f"unsupported portable agent-memory policy: {config.policy}")
        records = history.record_by_id
        states = _states(history)
        protected = {
            state.turn.causal_group_id for state in states[: config.head_turns]
        }
        if config.tail_turns:
            protected.update(
                state.turn.causal_group_id for state in states[-config.tail_turns :]
            )
        dependency_groups = {
            dependency
            for record in history.records
            for dependency in record.depends_on
        }
        protected.update(dependency_groups)
        protected.update(
            record.causal_group_id for record in history.records if record.depends_on
        )
        replacements: dict[str, str] = {}
        decisions: list[dict[str, Any]] = []
        presentation = str(config.options.get("receipt_presentation", "metadata_only"))

        def install(state: _TurnState, rule: str, resources: Iterable[str]) -> None:
            if state.turn.causal_group_id in protected or len(state.observations) != 1:
                return
            observation = state.observations[0]
            candidate = _receipt(rule, state, resources, presentation=presentation)
            source_tokens = self.count_tokens(observation.content)
            receipt_tokens = self.count_tokens(candidate)
            if receipt_tokens >= source_tokens:
                return
            replacements[observation.record_id] = candidate
            decisions.append({
                "rule": rule,
                "causal_group_id": state.turn.causal_group_id,
                "observation_record_id": observation.record_id,
                "source_tokens": source_tokens,
                "replacement_tokens": receipt_tokens,
                "saving_tokens": source_tokens - receipt_tokens,
            })

        # H1: every concrete search result is consumed by a later read/diff,
        # then the trajectory advances beyond discovery/read.
        for index, older in enumerate(states):
            if (
                older.operation != OperationKind.SEARCH_DISCOVERY
                or not older.successful or not older.complete_output
            ):
                continue
            discovered = {row for row in older.discovered if not row.startswith("search:")}
            if not discovered:
                continue
            consumed: set[str] = set()
            final_consumer = index
            for later_index, later in enumerate(states[index + 1 :], start=index + 1):
                if (
                    later.operation in {OperationKind.READ, OperationKind.DIFF}
                    and later.successful and later.complete_output
                ):
                    shared = discovered.intersection(later.resources)
                    if shared:
                        consumed.update(shared)
                        final_consumer = later_index
            advanced = any(
                later.turn.complete
                and later.operation not in {
                    OperationKind.SEARCH_DISCOVERY, OperationKind.READ, OperationKind.DIFF,
                }
                for later in states[final_consumer + 1 :]
            )
            if consumed == discovered and advanced:
                install(older, "H1", discovered)

        # H2: retain the action, compact a successful mutation result only when
        # a later current-version read or explicit verification dependency exists.
        for index, write in enumerate(states):
            if write.operation != OperationKind.WRITE or not write.successful or not write.changed:
                continue
            write_versions = {row.resource_id: row.version for row in write.accesses}
            for later in states[index + 1 :]:
                if later.operation == OperationKind.WRITE and set(write.changed).intersection(later.resources):
                    break
                shared = set(write.changed).intersection(later.resources)
                if later.operation in {OperationKind.READ, OperationKind.DIFF} and shared:
                    current = {row.resource_id: row.version for row in later.accesses}
                    matched = {
                        resource for resource in shared
                        if write_versions.get(resource) not in {None, "unknown"}
                        and write_versions.get(resource) == current.get(resource)
                    }
                    if matched and later.successful and later.complete_output:
                        install(write, "H2A", matched)
                        break
                if later.operation == OperationKind.VERIFY and later.successful and later.complete_output:
                    dependencies = set(later.resources)
                    dependencies.update(_values(later.observations, "verification_resource_ids"))
                    dependencies.update(_values(later.observations, "dependency_resource_ids"))
                    matched = set(write.changed).intersection(dependencies)
                    if matched:
                        install(write, "H2B", matched)
                        break

        # H3: an older complete read is compacted only when a retained later
        # read covers every same-version span.
        reads = [
            state for state in states
            if state.operation == OperationKind.READ
            and state.successful and state.complete_output and state.accesses
        ]
        keep = int(config.options.get("same_span_reads_to_keep", 1))
        for index, older in enumerate(reads):
            if all(
                len([
                    later for later in reads[index + 1 :]
                    if any(_covers(candidate, access) for candidate in later.accesses)
                ]) >= keep
                for access in older.accesses
            ):
                install(older, "H3", older.resources)

        selected = [record.record_id for record in history.records]
        full_tokens = sum(self.count_tokens(record.content) for record in history.records)
        materialized_tokens = sum(
            self.count_tokens(replacements.get(record.record_id, record.content))
            for record in history.records
        )
        requested_fraction = float(config.options.get("retention_fraction", 1.0))
        if not 0 < requested_fraction <= 1:
            raise ValueError("history_selection.options.retention_fraction must be in (0, 1]")
        target = max(1, math.ceil(full_tokens * requested_fraction))

        # Optional second stage: drop complete middle groups oldest-first after
        # state-authority receipts.  Protected head/tail and decision spine
        # records remain mandatory. This is explicitly a selection heuristic,
        # not a proof of irrelevance.
        dropped_groups: list[str] = []
        if materialized_tokens > target and bool(config.options.get("apply_middle_budget", True)):
            spine_roles = {
                AgentRecordRole.MUTATION,
                AgentRecordRole.VERIFICATION,
                AgentRecordRole.ERROR_OR_REJECTION,
                AgentRecordRole.FINALIZATION,
            }
            for state in states:
                group = state.turn.causal_group_id
                if group in protected:
                    continue
                members = tuple(records[record_id] for record_id in state.turn.record_ids)
                if any(any(member.has_role(role) for role in spine_roles) for member in members):
                    continue
                cost = sum(
                    self.count_tokens(replacements.get(member.record_id, member.content))
                    for member in members
                )
                if materialized_tokens - cost < target:
                    continue
                selected = [record_id for record_id in selected if record_id not in state.turn.record_ids]
                for record_id in state.turn.record_ids:
                    replacements.pop(record_id, None)
                materialized_tokens -= cost
                dropped_groups.append(group)

        return WireAgentMemoryPlan(
            schema_version=1,
            policy=config.policy,
            selected_record_ids=tuple(selected),
            record_replacements=replacements,
            source_history_digest=history.digest,
            decision_metadata={
                "planner": "portable_state_authority_v1",
                "token_accounting": "exact_injected_tokenizer",
                "full_tokens": full_tokens,
                "materialized_tokens": materialized_tokens,
                "requested_retention_fraction": requested_fraction,
                "realized_retention_fraction": (
                    materialized_tokens / full_tokens if full_tokens else 1.0
                ),
                "receipt_decisions": decisions,
                "receipt_presentation": presentation,
                "dropped_causal_group_ids": dropped_groups,
                "unknown_effect_policy": config.unknown_effect,
            },
        )


def tokenizer_counter(tokenizer: Any) -> TokenCounter:
    """Adapt a Hugging Face-compatible tokenizer to exact content counting."""

    def count(text: str) -> int:
        encoded = tokenizer(text, add_special_tokens=False)
        values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        return len(values)

    return count


__all__ = ["PortableStateAuthorityPlanner", "TokenCounter", "tokenizer_counter"]
