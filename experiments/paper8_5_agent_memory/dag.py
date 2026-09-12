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
import re
from typing import Callable, Mapping, Protocol

from .model import AgentRecord, AgentRecordRole, AgentTurn, CanonicalAgentHistory


class EffectKind(str, Enum):
    READ = "read"
    WRITE = "write"
    VERIFY = "verify"
    PURE = "pure"
    UNKNOWN = "unknown"


class EffectProvenance(str, Enum):
    DECLARED_TOOL_SEMANTICS = "declared_tool_semantics"
    RUNTIME_TRACED = "runtime_traced"
    STATIC_HEURISTIC = "static_heuristic"
    UNKNOWN = "unknown"


class DagEdgeKind(str, Enum):
    CAUSAL_RESULT = "causal_result"
    RESOURCE_FLOW = "resource_flow"
    INVALIDATES_STATE = "invalidates_state"
    SUPERSEDES_CURRENT_STATE = "supersedes_current_state"


class ExclusionClass(str, Enum):
    CERTIFIED_OPERATIONAL_DUPLICATE = "certified_operational_duplicate"
    HEURISTIC_EXACT_DUPLICATE = "heuristic_exact_duplicate"
    SUPERSEDED_CURRENT_STATE = "superseded_current_state_not_semantic_proof"
    INVALIDATED_BY_WRITE = "invalidated_by_write_not_semantic_proof"
    RECOVERABLE_COLD = "recoverable_cold_not_irrelevant"


@dataclass(frozen=True)
class ResourceEffect:
    record_id: str
    resource_id: str
    kind: EffectKind
    resource_version_fingerprint: str | None = None
    provenance: EffectProvenance = EffectProvenance.STATIC_HEURISTIC


@dataclass(frozen=True)
class ToolEffectAnalysis:
    tool_category: str
    effects: tuple[ResourceEffect, ...]
    provenance: EffectProvenance
    complete: bool
    unknown_barrier: bool


class ToolSemanticsProvider(Protocol):
    """Harness-side adapter from tool semantics to generic resource effects."""

    def analyze(
        self,
        action: AgentRecord,
        observations: tuple[AgentRecord, ...],
    ) -> ToolEffectAnalysis: ...


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


_READ_ONLY = re.compile(
    r"^\s*(?:cat|head|tail|sed\s+-n|rg|grep|find|ls|pwd|git\s+(?:status|diff|show))\b"
)
_WRITE = re.compile(
    r"(?:apply_patch|sed\s+-i|perl\s+-pi|git\s+apply|patch\s+-p|"
    r"(?:write_text|open\([^)]*,\s*['\"]w)|(?:^|[;&|]\s*)rm\b|>{1,2}\s*)"
)
_VERIFY = re.compile(
    r"^\s*(?:pytest|tox|nox|python\s+-m\s+(?:pytest|unittest)|"
    r"make\s+(?:test|check|lint)|ruff|mypy|npm\s+test|cargo\s+test)\b"
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
        kind = _effect_kind(action.command)
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


class HarnessMetadataSemanticsProvider:
    """Consume generic effect metadata emitted by an agent/tool harness."""

    def analyze(
        self,
        action: AgentRecord,
        observations: tuple[AgentRecord, ...],
    ) -> ToolEffectAnalysis:
        declaration = next((
            row.metadata.get("tool_semantics") for row in reversed(observations)
            if isinstance(row.metadata.get("tool_semantics"), Mapping)
        ), None)
        if declaration is None:
            return ToolEffectAnalysis(
                "unknown", (), EffectProvenance.UNKNOWN, False, True,
            )
        try:
            provenance = EffectProvenance(str(declaration["provenance"]))
            complete = bool(declaration["complete"])
            effects = tuple(ResourceEffect(
                action.record_id,
                str(row["resource_id"]),
                EffectKind(str(row["kind"])),
                str(row["resource_version_fingerprint"])
                if row.get("resource_version_fingerprint") is not None else None,
                provenance,
            ) for row in declaration.get("effects", ()))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid harness tool_semantics declaration") from error
        trusted = provenance in {
            EffectProvenance.DECLARED_TOOL_SEMANTICS,
            EffectProvenance.RUNTIME_TRACED,
        }
        return ToolEffectAnalysis(
            str(declaration.get("category", "generic")),
            effects,
            provenance,
            complete,
            not (trusted and complete and effects),
        )


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


def _effect_kind(command: str | None) -> EffectKind:
    if not command:
        return EffectKind.UNKNOWN
    if _WRITE.search(command):
        return EffectKind.WRITE
    if _VERIFY.search(command):
        return EffectKind.VERIFY
    if _READ_ONLY.search(command):
        return EffectKind.READ
    if command.strip() in {"true", ":"}:
        return EffectKind.PURE
    return EffectKind.UNKNOWN


def _turn_records(
    turn: AgentTurn,
    records: Mapping[str, AgentRecord],
) -> tuple[AgentRecord, ...]:
    return tuple(records[record_id] for record_id in turn.record_ids)


def _turn_digest(turn: AgentTurn, records: Mapping[str, AgentRecord]) -> str:
    content = "\x1e".join(
        f"{record.role}\x1f{record.content}" for record in _turn_records(turn, records)
    )
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
            digest = _turn_digest(turn, records)
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
                        rule_id="TRACE_EXACT_READ_BUNDLE_V1",
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
                    "the complete read-only action and observation are byte-identical; "
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
            if record.has_role(AgentRecordRole.ERROR_OR_REJECTION):
                protected.add(record.causal_group_id)
        for role in (AgentRecordRole.MUTATION, AgentRecordRole.VERIFICATION):
            matching = [record for record in history.records if record.has_role(role)]
            if matching:
                protected.add(matching[-1].causal_group_id)
        filtered = exclude_certified_groups(
            history,
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
        policy = (
            "dag_certified_exclusion"
            if self.fallback_selector is None
            else f"dag_certified_exclusion+{plan.policy}"
        )
        return replace(plan, policy=policy, full_history_tokens=original_tokens)
