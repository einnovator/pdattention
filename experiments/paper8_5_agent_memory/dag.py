"""Conservative resource/effect DAG and negative-exclusion policy.

The DAG can certify operational obsolescence under declared assumptions. It
does not claim that removing text is behaviorally risk-free for an LLM: even a
duplicate changes positions and repetition. Missing provenance fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
import hashlib
from typing import Callable, Mapping

from .model import (
    AgentMemoryExclusion,
    AgentMemoryPlan,
    AgentRecord,
    AgentRecordRole,
    AgentTurn,
    CanonicalAgentHistory,
)
from pra_hf.tool_semantics import (
    DeclaredToolSemanticsProvider,
    EffectKind,
    EffectProvenance,
    ResourceEffect,
    ToolEffectAnalysis,
    ToolSemanticsProvider,
)
from .miniswe_semantics import (
    bash_semantics_are_certifiable,
    classify_bash_effect,
)


class DagEdgeKind(str, Enum):
    CAUSAL_RESULT = "causal_result"
    RESOURCE_FLOW = "resource_flow"
    INVALIDATES_STATE = "invalidates_state"
    SUPERSEDES_CURRENT_STATE = "supersedes_current_state"
    INSTRUCTION_CONTROL = "instruction_control"
    DECISION_FLOW = "decision_flow"
    DECLARED_DEPENDENCY = "declared_dependency"
    PROTOCOL_CONTROL = "protocol_control"
    WORKFLOW_CONTROL = "workflow_control"


class ExclusionClass(str, Enum):
    CERTIFIED_OPERATIONAL_DUPLICATE = "certified_operational_duplicate"
    HEURISTIC_EXACT_DUPLICATE = "heuristic_exact_duplicate"
    SUPERSEDED_CURRENT_STATE = "superseded_current_state_not_semantic_proof"
    INVALIDATED_BY_WRITE = "invalidated_by_write_not_semantic_proof"
    RECOVERABLE_COLD = "recoverable_cold_not_irrelevant"


@dataclass(frozen=True)
class ExclusionCertificate:
    rule_id: str
    proof_scope: str
    witness_record_ids: tuple[str, ...]
    resource_version_fingerprints: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class DagEdge:
    source_id: str
    target_id: str
    kind: DagEdgeKind
    resource_id: str | None = None


@dataclass(frozen=True)
class ExclusionCandidate:
    causal_group_id: str
    classification: ExclusionClass
    evidence_record_ids: tuple[str, ...]
    default_exclusion_eligible: bool
    reason: str
    certificate: ExclusionCertificate | None = None


@dataclass(frozen=True)
class AgentHistoryDag:
    effects: tuple[ResourceEffect, ...]
    edges: tuple[DagEdge, ...]
    exclusion_candidates: tuple[ExclusionCandidate, ...]

    @property
    def certified_excluded_groups(self) -> tuple[str, ...]:
        return tuple(
            row.causal_group_id
            for row in self.exclusion_candidates
            if row.default_exclusion_eligible and row.certificate is not None
        )


class FrontierRetirementConfidence(str, Enum):
    """Strength of a no-path retirement decision.

    ``CERTIFIED`` requires an application- or runtime-provided workspace
    lineage.  ``HEURISTIC`` means that the visible resource/effect graph has no
    path, but incomplete Bash semantics or an unscoped workspace prevents a
    proof.  The distinction is intentional: absence of an inferred edge is
    not evidence that an edge cannot exist.
    """

    CERTIFIED = "certified_disconnected"
    HEURISTIC = "heuristic_disconnected"


class FrontierSimplificationMode(str, Enum):
    """Model-visible realization for a disconnected causal group."""

    WHOLE_CAUSAL_GROUP = "whole_causal_group"
    OBSERVATION_PAYLOAD = "observation_payload"
    ACTION_PARAMETERS_AND_OBSERVATION = "action_parameters_and_observation"


@dataclass(frozen=True)
class InstructionEpoch:
    epoch_index: int
    instruction_record_id: str
    record_ids: tuple[str, ...]
    causal_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrontierRetirementCandidate:
    causal_group_id: str
    epoch_index: int
    record_ids: tuple[str, ...]
    confidence: FrontierRetirementConfidence
    resource_ids: tuple[str, ...]
    unknown_effect: bool
    reason: str


@dataclass(frozen=True)
class FrontierInformationFlowDag:
    """Forward information-flow DAG rooted at genuine user instructions."""

    epochs: tuple[InstructionEpoch, ...]
    edges: tuple[DagEdge, ...]
    frontier_epoch_indices: tuple[int, ...]
    frontier_record_ids: tuple[str, ...]
    live_ancestor_record_ids: tuple[str, ...]
    retirement_candidates: tuple[FrontierRetirementCandidate, ...]


@dataclass(frozen=True)
class FrontierSimplificationPlan:
    mode: FrontierSimplificationMode
    selected_record_ids: tuple[str, ...]
    record_replacements: tuple[tuple[str, str], ...]
    retired_causal_group_ids: tuple[str, ...]
    full_tokens: int
    materialized_tokens: int

    @property
    def saving_fraction(self) -> float:
        return (
            0.0
            if self.full_tokens == 0
            else 1.0 - self.materialized_tokens / self.full_tokens
        )


class MiniSweBashSemanticsProvider:
    """Special support for mini-swe-agent's single generic Bash tool.

    Regex inference is useful for candidate construction but never supplies a
    certified effect on its own. Arbitrary Bash fails closed as unknown.
    """

    def analyze(
        self,
        action: AgentRecord,
        observations: tuple[AgentRecord, ...],
    ) -> ToolEffectAnalysis:
        rows = (action, *observations)
        kind = classify_bash_effect(action.command)
        resources = tuple(dict.fromkeys(
            resource for row in rows for resource in row.resource_ids
        )) or ("resource:unknown",)
        effects = tuple(ResourceEffect(
            action.record_id,
            resource,
            kind,
            provenance=EffectProvenance.STATIC_HEURISTIC,
        ) for resource in resources)
        return ToolEffectAnalysis(
            tool_category="bash",
            effects=effects,
            provenance=EffectProvenance.STATIC_HEURISTIC,
            complete=False,
            unknown_barrier=kind == EffectKind.UNKNOWN,
        )


HarnessMetadataSemanticsProvider = DeclaredToolSemanticsProvider


class CompositeToolSemanticsProvider:
    """Prefer complete harness declarations, otherwise use Bash diagnostics."""

    def __init__(self) -> None:
        self.declared = HarnessMetadataSemanticsProvider()
        self.bash = MiniSweBashSemanticsProvider()

    def analyze(
        self,
        action: AgentRecord,
        observations: tuple[AgentRecord, ...],
    ) -> ToolEffectAnalysis:
        declared = self.declared.analyze(action, observations)
        return declared if not declared.unknown_barrier else self.bash.analyze(action, observations)


_INSTRUCTION_ROLES = (AgentRecordRole.TASK, AgentRecordRole.USER_INPUT)


def _workspace_lineage(record: AgentRecord) -> str | None:
    """Return only an explicit workspace identity, never cwd or image identity.

    A cwd such as ``/testbed`` and an environment/image fingerprint commonly
    recur in independent containers.  Treating either as a workspace lineage
    silently creates false cross-task resource dependencies.
    """

    for key in ("workspace_lineage_id", "workspace_scope", "resource_scope_id"):
        value = record.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _environment_scope(record: AgentRecord) -> str | None:
    value = record.metadata.get("environment_fingerprint")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _instruction_epochs(
    history: CanonicalAgentHistory,
) -> tuple[tuple[InstructionEpoch, ...], dict[str, int]]:
    ordered = sorted(history.records, key=lambda row: row.message_index)
    epoch_rows: list[dict[str, object]] = []
    epoch_for_record: dict[str, int] = {}
    active: dict[str, object] | None = None
    for record in ordered:
        if any(record.has_role(role) for role in _INSTRUCTION_ROLES):
            active = {
                "instruction": record.record_id,
                "records": [],
                "groups": [],
            }
            epoch_rows.append(active)
        if active is None:
            continue
        epoch_index = len(epoch_rows) - 1
        epoch_for_record[record.record_id] = epoch_index
        cast_records = active["records"]
        assert isinstance(cast_records, list)
        cast_records.append(record.record_id)
        if record.causal_group_id not in active["groups"]:
            cast_groups = active["groups"]
            assert isinstance(cast_groups, list)
            cast_groups.append(record.causal_group_id)
    return tuple(
        InstructionEpoch(
            index,
            str(row["instruction"]),
            tuple(str(value) for value in row["records"]),
            tuple(str(value) for value in row["groups"]),
        )
        for index, row in enumerate(epoch_rows)
    ), epoch_for_record


def _record_resources(record: AgentRecord) -> tuple[str, ...]:
    declared = record.metadata.get("dependency_resource_ids")
    dependencies = (
        tuple(str(value) for value in declared)
        if isinstance(declared, (list, tuple))
        else ()
    )
    discovered = record.metadata.get("discovered_resource_ids")
    discoveries = (
        tuple(str(value) for value in discovered)
        if isinstance(discovered, (list, tuple))
        else ()
    )
    return tuple(dict.fromkeys((*record.resource_ids, *dependencies, *discoveries)))


def _has_unknown_effect(record: AgentRecord) -> bool:
    if not record.has_role(AgentRecordRole.ASSISTANT_ACTION):
        return False
    operation = str(record.metadata.get("operation_kind") or "").lower()
    return operation in {"", "other", "unknown"}


def build_frontier_information_flow_dag(
    history: CanonicalAgentHistory,
    *,
    recent_user_prompts: int = 2,
    valid_protocol_exemplars: int = 0,
    valid_workflow_exemplars: int = 0,
) -> FrontierInformationFlowDag:
    """Build a boundary-free, forward information-flow graph.

    Every TASK/USER_INPUT record starts an instruction epoch.  The most recent
    ``M`` epochs form the live frontier.  Older causal groups become retirement
    candidates only when no directed path reaches that frontier.  The selector
    never receives evaluator task IDs or source-episode boundaries.

    Edges encode (1) instruction control within an epoch, (2) action/result and
    next-decision flow, (3) same-resource flow across epochs, and (4) explicit
    record dependencies supplied by a generic harness.  Unknown tool semantics
    lower confidence; they do not invent a proof of irrelevance.
    """

    if recent_user_prompts < 1:
        raise ValueError("recent_user_prompts must be positive")
    if valid_protocol_exemplars < 0:
        raise ValueError("valid_protocol_exemplars cannot be negative")
    if valid_workflow_exemplars < 0:
        raise ValueError("valid_workflow_exemplars cannot be negative")
    epochs, epoch_for_record = _instruction_epochs(history)
    if not epochs:
        return FrontierInformationFlowDag((), (), (), (), (), ())
    records = history.record_by_id
    edge_rows: list[DagEdge] = []
    epoch_lineages: dict[int, str] = {}
    epoch_environments: dict[int, str] = {}
    for epoch in epochs:
        lineages = {
            value
            for record_id in epoch.record_ids
            if (value := _workspace_lineage(records[record_id])) is not None
        }
        if len(lineages) == 1:
            epoch_lineages[epoch.epoch_index] = next(iter(lineages))
        values = {
            value
            for record_id in epoch.record_ids
            if (value := _environment_scope(records[record_id])) is not None
        }
        # An environment fingerprint is weaker than a workspace lineage, but
        # when stable within one instruction epoch it prevents obviously
        # unrelated containers/images from aliasing on /testbed or patch.txt.
        if len(values) == 1:
            epoch_environments[epoch.epoch_index] = next(iter(values))

    # Intra-epoch decision spine.  It deliberately stops at a new genuine user
    # prompt: task boundaries are inferred from visible protocol, never from an
    # evaluator's issue ID.
    for epoch in epochs:
        prior = epoch.instruction_record_id
        for turn in history.turns:
            turn_ids = tuple(
                record_id for record_id in turn.record_ids
                if epoch_for_record.get(record_id) == epoch.epoch_index
            )
            if not turn_ids:
                continue
            action = next((
                record_id for record_id in turn_ids
                if records[record_id].has_role(AgentRecordRole.ASSISTANT_ACTION)
            ), None)
            if action is None:
                continue
            edge_rows.append(DagEdge(
                prior,
                action,
                DagEdgeKind.INSTRUCTION_CONTROL
                if prior == epoch.instruction_record_id
                else DagEdgeKind.DECISION_FLOW,
            ))
            observations = tuple(
                record_id for record_id in turn_ids
                if records[record_id].has_role(AgentRecordRole.TOOL_OBSERVATION)
            )
            for observation in observations:
                edge_rows.append(DagEdge(
                    action, observation, DagEdgeKind.CAUSAL_RESULT
                ))
            prior = observations[-1] if observations else action

    # Resource flow across instruction epochs requires a real workspace
    # lineage declaration.  An environment/image fingerprint is not a
    # workspace identity: independent containers commonly share both it and a
    # cwd such as /testbed.  Without an explicit lineage, scope inferred
    # resources to the current instruction epoch.  This still models resource
    # flow within one issue, while preventing same-path coincidences from
    # manufacturing dependencies between unrelated issues.  The resulting
    # retirement remains HEURISTIC (and therefore opt-in) because missing
    # lineage is not a proof of independence.
    last_resource_record: dict[tuple[str, str], str] = {}
    for record in sorted(history.records, key=lambda row: row.message_index):
        epoch_index = epoch_for_record.get(record.record_id)
        lineage = _workspace_lineage(record) or (
            epoch_lineages.get(epoch_index) if epoch_index is not None else None
        )
        environment = _environment_scope(record) or (
            epoch_environments.get(epoch_index) if epoch_index is not None else None
        )
        scope = (
            f"workspace:{lineage}" if lineage is not None
            else (
                f"epoch:{epoch_index}:environment:{environment}"
                if environment is not None
                else f"epoch:{epoch_index}:workspace:unknown"
            )
        )
        for resource in _record_resources(record):
            key = (scope, resource)
            predecessor = last_resource_record.get(key)
            if predecessor is not None and predecessor != record.record_id:
                edge_rows.append(DagEdge(
                    predecessor,
                    record.record_id,
                    DagEdgeKind.RESOURCE_FLOW,
                    resource,
                ))
            last_resource_record[key] = record.record_id
        dependencies = record.metadata.get("dependency_record_ids")
        if isinstance(dependencies, (list, tuple)):
            for predecessor in dependencies:
                predecessor = str(predecessor)
                if predecessor in records and predecessor != record.record_id:
                    edge_rows.append(DagEdge(
                        predecessor,
                        record.record_id,
                        DagEdgeKind.DECLARED_DEPENDENCY,
                    ))

    frontier_epochs = tuple(
        epoch.epoch_index for epoch in epochs[-recent_user_prompts:]
    )

    # A valid protocol completion is durable control evidence, not task
    # content.  Retaining the latest P exemplars prevents a malformed recent
    # submission from becoming the only behavioral example even when the
    # system prompt still describes the protocol.  Validity is declared by the
    # harness adapter from the actual completion envelope; the generic DAG
    # never parses agent-specific command syntax.
    control_barrier_ids: set[str] = set()
    if valid_protocol_exemplars or valid_workflow_exemplars:
        valid_groups: list[tuple[str, str]] = []
        seen_groups: set[str] = set()
        for record in sorted(
            history.records, key=lambda row: row.message_index, reverse=True
        ):
            if not bool(record.metadata.get("protocol_completion_valid")):
                continue
            if record.causal_group_id in seen_groups:
                continue
            seen_groups.add(record.causal_group_id)
            valid_groups.append((record.causal_group_id, record.record_id))
            if len(valid_groups) >= max(
                valid_protocol_exemplars, valid_workflow_exemplars
            ):
                break
        frontier_instructions = tuple(
            epoch.instruction_record_id for epoch in epochs
            if epoch.epoch_index in frontier_epochs
        )
        for _, source_id in valid_groups[:valid_protocol_exemplars]:
            control_barrier_ids.add(source_id)
            for target_id in frontier_instructions:
                if source_id != target_id:
                    edge_rows.append(DagEdge(
                        source_id,
                        target_id,
                        DagEdgeKind.PROTOCOL_CONTROL,
                    ))

        # An aggressive one-prompt frontier may retain a valid submission
        # example yet still lose the workflow fact that a mutation must occur
        # before verification and submission.  W exemplars therefore pin a
        # minimal successful mutation -> verification -> completion spine from
        # the latest harness-certified epoch.  Evidence comes from generic
        # observation metadata: an actual changed-resource receipt, followed
        # by a successful observation of that post-mutation resource.  No task
        # ID, command name, or repository-specific rule is consulted.
        for _, protocol_id in valid_groups[:valid_workflow_exemplars]:
            protocol_epoch = epoch_for_record.get(protocol_id)
            if protocol_epoch is None:
                continue
            protocol_index = records[protocol_id].message_index
            mutation_rows = [
                records[record_id]
                for record_id in epochs[protocol_epoch].record_ids
                if records[record_id].message_index < protocol_index
                and records[record_id].has_role(AgentRecordRole.TOOL_OBSERVATION)
                and records[record_id].return_code == 0
                and bool(records[record_id].metadata.get("changed_resource_ids"))
            ]
            if not mutation_rows:
                continue
            mutation = mutation_rows[-1]
            changed = {
                str(value)
                for value in mutation.metadata.get("changed_resource_ids", ())
            }
            verification_rows = [
                records[record_id]
                for record_id in epochs[protocol_epoch].record_ids
                if mutation.message_index < records[record_id].message_index < protocol_index
                and records[record_id].has_role(AgentRecordRole.TOOL_OBSERVATION)
                and records[record_id].return_code == 0
                and changed.intersection(_record_resources(records[record_id]))
                and not records[record_id].metadata.get("changed_resource_ids")
            ]
            workflow_sources = [mutation]
            if verification_rows:
                workflow_sources.append(verification_rows[-1])
            workflow_sources.append(records[protocol_id])
            for source in workflow_sources:
                control_barrier_ids.add(source.record_id)
                for target_id in frontier_instructions:
                    if source.record_id != target_id:
                        edge_rows.append(DagEdge(
                            source.record_id,
                            target_id,
                            DagEdgeKind.WORKFLOW_CONTROL,
                        ))

    # Deduplicate before reachability so an action+observation pair declaring
    # the same resource does not inflate evidence counts.
    edge_map = {
        (row.source_id, row.target_id, row.kind, row.resource_id): row
        for row in edge_rows
    }
    edges = tuple(edge_map.values())
    frontier_ids = {
        record_id
        for epoch in epochs
        if epoch.epoch_index in frontier_epochs
        for record_id in epoch.record_ids
    }
    incoming: dict[str, set[str]] = {}
    for edge in edges:
        incoming.setdefault(edge.target_id, set()).add(edge.source_id)
    live = set(frontier_ids)
    pending = list(frontier_ids)
    while pending:
        target = pending.pop()
        # A protocol exemplar is retained as an atomic behavioral example.
        # Its own causal predecessors are task-specific work, not protocol
        # dependencies, so liveness must not flood backward through the whole
        # completed instruction epoch.
        if target in control_barrier_ids:
            continue
        for source in incoming.get(target, ()):
            if source not in live:
                live.add(source)
                pending.append(source)

    candidates: list[FrontierRetirementCandidate] = []
    turn_by_group = {turn.causal_group_id: turn for turn in history.turns}
    for epoch in epochs:
        if epoch.epoch_index in frontier_epochs:
            continue
        for group_id in epoch.causal_group_ids:
            turn = turn_by_group.get(group_id)
            if turn is None:
                # Instruction/system records are immutable and never candidates.
                continue
            group_ids = tuple(
                record_id for record_id in turn.record_ids
                if epoch_for_record.get(record_id) == epoch.epoch_index
            )
            if not group_ids or any(record_id in live for record_id in group_ids):
                continue
            rows = tuple(records[record_id] for record_id in group_ids)
            lineages = {
                value for row in rows
                if (value := _workspace_lineage(row)) is not None
            }
            if not lineages and epoch.epoch_index in epoch_lineages:
                lineages.add(epoch_lineages[epoch.epoch_index])
            explicit_lineage = len(lineages) == 1
            unknown = any(_has_unknown_effect(row) for row in rows)
            confidence = (
                FrontierRetirementConfidence.CERTIFIED
                if explicit_lineage and not unknown
                else FrontierRetirementConfidence.HEURISTIC
            )
            resources = tuple(dict.fromkeys(
                resource for row in rows for resource in _record_resources(row)
            ))
            candidates.append(FrontierRetirementCandidate(
                group_id,
                epoch.epoch_index,
                group_ids,
                confidence,
                resources,
                unknown,
                (
                    "no directed information-flow path reaches any record in "
                    f"the last {recent_user_prompts} user-instruction epochs"
                ),
            ))

    return FrontierInformationFlowDag(
        epochs,
        edges,
        frontier_epochs,
        tuple(sorted(frontier_ids, key=lambda value: records[value].message_index)),
        tuple(sorted(live, key=lambda value: records[value].message_index)),
        tuple(candidates),
    )


_ACTION_PARAMETERS_OMITTED = (
    "[PRA memory] Prior tool action completed; parameters omitted."
)


def simplify_disconnected_frontier(
    history: CanonicalAgentHistory,
    dag: FrontierInformationFlowDag,
    *,
    mode: FrontierSimplificationMode | str,
    count_tokens: Callable[[str], int],
    allow_heuristic: bool = True,
) -> FrontierSimplificationPlan:
    """Realize disconnected groups while preserving every user instruction.

    Whole-group retirement is protocol-safe.  Payload/parameter modes retain
    role alternation using compact ordinary-text stubs and are explicit causal
    ablations.  A replacement is accepted only when it is smaller than its
    source; otherwise the source remains byte-identical.
    """

    selected_mode = FrontierSimplificationMode(mode)
    records = history.record_by_id
    eligible = tuple(
        candidate for candidate in dag.retirement_candidates
        if allow_heuristic
        or candidate.confidence == FrontierRetirementConfidence.CERTIFIED
    )
    eligible_groups = {row.causal_group_id for row in eligible}
    selected: list[str] = []
    replacements: dict[str, str] = {}
    for record in sorted(history.records, key=lambda row: row.message_index):
        retired = record.causal_group_id in eligible_groups
        if retired and selected_mode == FrontierSimplificationMode.WHOLE_CAUSAL_GROUP:
            continue
        selected.append(record.record_id)
        if not retired:
            continue
        replacement: str | None = None
        if record.has_role(AgentRecordRole.TOOL_OBSERVATION):
            return_code = record.return_code if record.return_code is not None else "unknown"
            replacement = (
                f"<returncode>{return_code}</returncode>\n"
                "<output>[PRA memory] prior tool result omitted; "
                "no live dependency</output>"
            )
        elif (
            selected_mode
            == FrontierSimplificationMode.ACTION_PARAMETERS_AND_OBSERVATION
            and record.has_role(AgentRecordRole.ASSISTANT_ACTION)
        ):
            replacement = _ACTION_PARAMETERS_OMITTED
        if replacement is not None and count_tokens(replacement) < count_tokens(record.content):
            replacements[record.record_id] = replacement

    full_tokens = sum(count_tokens(row.content) for row in history.records)
    materialized_tokens = sum(
        count_tokens(replacements.get(record_id, records[record_id].content))
        for record_id in selected
    )
    return FrontierSimplificationPlan(
        selected_mode,
        tuple(selected),
        tuple(replacements.items()),
        tuple(row.causal_group_id for row in eligible),
        full_tokens,
        materialized_tokens,
    )


def _turn_records(
    turn: AgentTurn,
    records: Mapping[str, AgentRecord],
) -> tuple[AgentRecord, ...]:
    return tuple(records[record_id] for record_id in turn.record_ids)


def _operational_read_digest(
    turn: AgentTurn,
    records: Mapping[str, AgentRecord],
) -> str:
    """Hash the executed command and observations, excluding private reasoning.

    Two Bash turns can be the same witnessed operation even when the model's
    prose before the command differs.  The certificate remains deliberately
    operational: it proves duplicate external evidence, not that deleting the
    older reasoning text is behaviorally invisible to an LLM.
    """

    rows = _turn_records(turn, records)
    action = next(
        (row for row in rows if row.has_role(AgentRecordRole.ASSISTANT_ACTION)),
        None,
    )
    observations = tuple(
        row for row in rows if row.has_role(AgentRecordRole.TOOL_OBSERVATION)
    )
    content = "\x1e".join((
        f"command\x1f{action.command if action is not None else ''}",
        *(f"observation\x1f{row.content}" for row in observations),
    ))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _runtime_identity(
    rows: tuple[AgentRecord, ...],
) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    """Return a certificate identity only when the harness captured it fully."""

    observation = next(
        (row for row in reversed(rows) if row.has_role(AgentRecordRole.TOOL_OBSERVATION)),
        None,
    )
    if observation is None or observation.metadata.get("output_complete") is not True:
        return None
    cwd = observation.metadata.get("cwd")
    environment = observation.metadata.get("environment_fingerprint")
    versions = observation.metadata.get("resource_version_fingerprints")
    if not isinstance(cwd, str) or not isinstance(environment, str):
        return None
    if not isinstance(versions, Mapping) or not versions:
        return None
    normalized = tuple(sorted((str(key), str(value)) for key, value in versions.items()))
    return cwd, environment, normalized


def build_resource_effect_dag(
    history: CanonicalAgentHistory,
    semantics: ToolSemanticsProvider | None = None,
) -> AgentHistoryDag:
    """Infer conservative resource flows from a mini-swe-agent trajectory."""

    records = history.record_by_id
    effects: list[ResourceEffect] = []
    edges: list[DagEdge] = []
    candidates: list[ExclusionCandidate] = []
    last_resource_record: dict[str, str] = {}
    last_read_record: dict[str, str] = {}
    seen_read_bundle: dict[str, AgentTurn] = {}
    semantics = semantics or CompositeToolSemanticsProvider()

    for turn in history.turns:
        rows = _turn_records(turn, records)
        if not rows:
            continue
        action = next(
            (row for row in rows if row.has_role(AgentRecordRole.ASSISTANT_ACTION)),
            None,
        )
        observations = tuple(
            row for row in rows if row.has_role(AgentRecordRole.TOOL_OBSERVATION)
        )
        if action is None:
            continue
        for observation in observations:
            edges.append(DagEdge(
                action.record_id,
                observation.record_id,
                DagEdgeKind.CAUSAL_RESULT,
            ))

        analysis = semantics.analyze(action, observations)
        effects.extend(analysis.effects)
        effects_by_resource = {
            effect.resource_id: effect for effect in analysis.effects
        }
        if not effects_by_resource:
            effects_by_resource["resource:unknown"] = ResourceEffect(
                action.record_id,
                "resource:unknown",
                EffectKind.UNKNOWN,
                provenance=EffectProvenance.UNKNOWN,
            )
        evidence_id = observations[-1].record_id if observations else action.record_id
        for resource, effect in effects_by_resource.items():
            kind = effect.kind
            predecessor = last_resource_record.get(resource)
            if predecessor is not None:
                edge_kind = (
                    DagEdgeKind.INVALIDATES_STATE
                    if kind == EffectKind.WRITE
                    else DagEdgeKind.SUPERSEDES_CURRENT_STATE
                    if kind == EffectKind.READ
                    else DagEdgeKind.RESOURCE_FLOW
                )
                edges.append(DagEdge(predecessor, action.record_id, edge_kind, resource))
            if kind == EffectKind.WRITE and resource in last_read_record:
                candidates.append(ExclusionCandidate(
                    records[last_read_record[resource]].causal_group_id,
                    ExclusionClass.INVALIDATED_BY_WRITE,
                    (last_read_record[resource], action.record_id),
                    False,
                    "a write invalidates prior current-state evidence but does not prove "
                    "that its historical facts are irrelevant",
                ))
            if kind == EffectKind.READ:
                if resource in last_read_record:
                    candidates.append(ExclusionCandidate(
                        records[last_read_record[resource]].causal_group_id,
                        ExclusionClass.SUPERSEDED_CURRENT_STATE,
                        (last_read_record[resource], evidence_id),
                        False,
                        "a newer read supersedes current-state authority but may omit "
                        "independent facts from the older observation",
                    ))
                last_read_record[resource] = evidence_id
            last_resource_record[resource] = evidence_id

        bundle_kinds = {effect.kind for effect in analysis.effects}
        read_only_bundle = bool(bundle_kinds) and bundle_kinds <= {
            EffectKind.READ,
            EffectKind.PURE,
        }
        if read_only_bundle and turn.complete:
            digest = _operational_read_digest(turn, records)
            older = seen_read_bundle.get(digest)
            if older is not None:
                older_rows = _turn_records(older, records)
                older_identity = _runtime_identity(older_rows)
                current_identity = _runtime_identity(rows)
                trusted_provenance = analysis.provenance in {
                    EffectProvenance.DECLARED_TOOL_SEMANTICS,
                    EffectProvenance.RUNTIME_TRACED,
                }
                certified = (
                    older_identity is not None
                    and older_identity == current_identity
                    and trusted_provenance
                    and analysis.complete
                    and not analysis.unknown_barrier
                )
                certificate = None
                if certified:
                    certificate = ExclusionCertificate(
                        rule_id="TRACE_EXACT_OPERATION_RESULT_V1",
                        proof_scope="operational_duplicate_not_llm_behavioral_equivalence",
                        witness_record_ids=(*older.record_ids, *turn.record_ids),
                        resource_version_fingerprints=older_identity[2],
                        assumptions=(
                            f"cwd={older_identity[0]}",
                            f"environment_fingerprint={older_identity[1]}",
                            "complete output and return code were captured",
                        ),
                    )
                candidates.append(ExclusionCandidate(
                    older.causal_group_id,
                    ExclusionClass.CERTIFIED_OPERATIONAL_DUPLICATE
                    if certified else ExclusionClass.HEURISTIC_EXACT_DUPLICATE,
                    (*older.record_ids, *turn.record_ids),
                    certified,
                    "the executed read-only command and complete observation are "
                    "byte-identical (assistant reasoning may differ); "
                    + (
                        "runtime identity also matches"
                        if certified else
                        "runtime identity is missing, so this remains a heuristic"
                    ),
                    certificate,
                ))
            seen_read_bundle[digest] = turn

    # A group may have several candidate explanations. Preserve the strongest
    # safe proof once and otherwise retain all diagnostic classifications.
    deduplicated: dict[tuple[str, ExclusionClass], ExclusionCandidate] = {}
    for candidate in candidates:
        deduplicated[(candidate.causal_group_id, candidate.classification)] = candidate
    return AgentHistoryDag(
        tuple(effects),
        tuple(edges),
        tuple(deduplicated.values()),
    )


def exclude_certified_groups(
    history: CanonicalAgentHistory,
    dag: AgentHistoryDag | None = None,
    *,
    protected_causal_group_ids: tuple[str, ...] = (),
) -> CanonicalAgentHistory:
    """Remove only groups with a certificate under the operational model."""

    dag = dag or build_resource_effect_dag(history)
    excluded = set(dag.certified_excluded_groups).difference(
        protected_causal_group_ids
    )
    records = tuple(
        record for record in history.records
        if record.causal_group_id not in excluded
    )
    turns = tuple(
        turn for turn in history.turns
        if turn.causal_group_id not in excluded
    )
    return CanonicalAgentHistory(records, turns)


class DagCertifiedExclusionSelector:
    """Apply certified negative exclusion before an optional positive selector."""

    def __init__(
        self,
        fallback_selector: object | None = None,
        *,
        protected_head_turns: int = 1,
        protected_tail_turns: int = 2,
    ) -> None:
        self.fallback_selector = fallback_selector
        self.protected_head_turns = protected_head_turns
        self.protected_tail_turns = protected_tail_turns

    def select(
        self,
        *,
        history: CanonicalAgentHistory,
        query: str,
        budget: object,
        count_tokens: Callable[[str], int],
    ):
        from .selectors import FullHistorySelector

        complete = [turn for turn in history.turns if turn.complete]
        protected_turns = list(complete[: self.protected_head_turns])
        if self.protected_tail_turns:
            protected_turns.extend(complete[-self.protected_tail_turns :])
        protected = {
            turn.causal_group_id
            for turn in protected_turns
        }
        for record in history.records:
            if (
                record.has_role(AgentRecordRole.ERROR_OR_REJECTION)
                or record.has_role(AgentRecordRole.SYSTEM)
                or record.has_role(AgentRecordRole.TASK)
                or record.has_role(AgentRecordRole.USER_INPUT)
            ):
                protected.add(record.causal_group_id)
        for role in (AgentRecordRole.MUTATION, AgentRecordRole.VERIFICATION):
            matching = [record for record in history.records if record.has_role(role)]
            if matching:
                protected.add(matching[-1].causal_group_id)
        dag = build_resource_effect_dag(history)
        filtered = exclude_certified_groups(
            history, dag,
            protected_causal_group_ids=tuple(protected),
        )
        selector = self.fallback_selector or FullHistorySelector()
        plan = selector.select(
            history=filtered,
            query=query,
            budget=budget,
            count_tokens=count_tokens,
        )
        original_tokens = sum(count_tokens(record.content) for record in history.records)
        records = history.record_by_id
        turns = {
            turn.causal_group_id: turn for turn in history.turns
        }
        exclusions = []
        for candidate in dag.exclusion_candidates:
            certificate = candidate.certificate
            if (
                not candidate.default_exclusion_eligible
                or certificate is None
                or candidate.causal_group_id in protected
            ):
                continue
            turn = turns.get(candidate.causal_group_id)
            if turn is None:
                continue
            resources = tuple(dict.fromkeys(
                resource
                for record_id in turn.record_ids
                for resource in records[record_id].resource_ids
            ))
            excluded_tokens = sum(
                count_tokens(records[record_id].content)
                for record_id in turn.record_ids
            )
            exclusions.append(AgentMemoryExclusion(
                causal_group_id=candidate.causal_group_id,
                record_ids=turn.record_ids,
                rule_id=certificate.rule_id,
                classification=candidate.classification.value,
                reason=candidate.reason,
                resource_ids=resources,
                witness_record_ids=certificate.witness_record_ids,
                tombstone=(
                    "INACTIVE group=" + candidate.causal_group_id
                    + " rule=" + certificate.rule_id
                    + " scope=" + certificate.proof_scope
                ),
                excluded_tokens=excluded_tokens,
            ))
        policy = (
            "dag_certified_exclusion"
            if self.fallback_selector is None
            else f"dag_certified_exclusion+{plan.policy}"
        )
        return replace(
            plan,
            policy=policy,
            full_history_tokens=original_tokens,
            exclusions=tuple(exclusions),
        )


class FrontierDagRetirementSelector:
    """Autonomous whole-group policy backed by recent-frontier reachability.

    This selector intentionally implements only the protocol-safe whole causal
    group realization.  Observation-only and action-parameter stubs remain
    separate materialization ablations and must not be conflated with logical
    group selection.
    """

    def __init__(
        self,
        *,
        recent_user_prompts: int = 2,
        allow_heuristic: bool = False,
        valid_protocol_exemplars: int = 0,
        valid_workflow_exemplars: int = 0,
    ) -> None:
        if recent_user_prompts < 1:
            raise ValueError("recent_user_prompts must be positive")
        self.recent_user_prompts = recent_user_prompts
        self.allow_heuristic = allow_heuristic
        if valid_protocol_exemplars < 0:
            raise ValueError("valid_protocol_exemplars cannot be negative")
        self.valid_protocol_exemplars = valid_protocol_exemplars
        if valid_workflow_exemplars < 0:
            raise ValueError("valid_workflow_exemplars cannot be negative")
        self.valid_workflow_exemplars = valid_workflow_exemplars

    def select(
        self,
        *,
        history: CanonicalAgentHistory,
        query: str,
        budget: object,
        count_tokens: Callable[[str], int],
    ):
        del query
        dag = build_frontier_information_flow_dag(
            history,
            recent_user_prompts=self.recent_user_prompts,
            valid_protocol_exemplars=self.valid_protocol_exemplars,
            valid_workflow_exemplars=self.valid_workflow_exemplars,
        )
        eligible = tuple(
            row for row in dag.retirement_candidates
            if self.allow_heuristic
            or row.confidence == FrontierRetirementConfidence.CERTIFIED
        )
        excluded_groups = {row.causal_group_id for row in eligible}
        selected_records = tuple(
            row for row in sorted(history.records, key=lambda value: value.message_index)
            if row.causal_group_id not in excluded_groups
        )
        selected_ids = tuple(row.record_id for row in selected_records)
        selected_groups = tuple(dict.fromkeys(
            row.causal_group_id for row in selected_records
        ))
        full_tokens = sum(count_tokens(row.content) for row in history.records)
        selected_tokens = sum(count_tokens(row.content) for row in selected_records)
        requested = int(getattr(budget, "max_tokens", full_tokens))
        exclusions = tuple(AgentMemoryExclusion(
            causal_group_id=row.causal_group_id,
            record_ids=row.record_ids,
            rule_id=f"FRONTIER_NO_PATH_M{self.recent_user_prompts}_V1",
            classification=row.confidence.value,
            reason=row.reason,
            resource_ids=row.resource_ids,
            witness_record_ids=dag.frontier_record_ids,
            tombstone=(
                f"INACTIVE group={row.causal_group_id} "
                f"rule=FRONTIER_NO_PATH_M{self.recent_user_prompts}_V1 "
                f"confidence={row.confidence.value}"
            ),
            excluded_tokens=sum(
                count_tokens(history.record_by_id[record_id].content)
                for record_id in row.record_ids
            ),
        ) for row in eligible)
        return AgentMemoryPlan(
            policy=(
                f"frontier_dag_m{self.recent_user_prompts}_"
                + ("heuristic" if self.allow_heuristic else "certified")
                + f"_p{self.valid_protocol_exemplars}"
                + (
                    f"_w{self.valid_workflow_exemplars}"
                    if self.valid_workflow_exemplars else ""
                )
            ),
            selected_record_ids=selected_ids,
            selected_causal_group_ids=selected_groups,
            selection_reasons=tuple(
                (record_id, "recent_frontier_or_live_ancestor")
                for record_id in selected_ids
            ),
            full_history_tokens=full_tokens,
            selected_tokens=selected_tokens,
            requested_budget_tokens=requested,
            mandatory_tokens=selected_tokens,
            mandatory_overflow_tokens=max(0, selected_tokens - requested),
            head_turns=0,
            tail_turns=len(dag.frontier_epoch_indices),
            middle_candidate_turns=len(history.turns),
            middle_selected_turns=sum(
                turn.causal_group_id not in excluded_groups for turn in history.turns
            ),
            exclusions=exclusions,
        )
