"""Latent trigger recovery policies over typed adaptive-context records.

The module keeps hypothesis generation, address search, exact materialization,
and final action choice as separate stages. It does not add a record type or a
second backing store; all reads pass through :class:`AdaptiveContextRuntime`.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

from .adaptive_context_runtime import (
    AdaptiveContextRuntime,
    CursorAction,
    CursorOperation,
    MaterializationEvent,
    MaterializationResult,
)
from .context_records import RecordType, RecordViewName
from .context_store import RecordAccessDenied, RecordScope


class TriggerFamily(str, Enum):
    """Functional hidden-evidence families in the paired benchmark."""

    ENTITY = "entity"
    ALIAS = "alias"
    NUMBER_DATE = "number_date"
    RARE_STRING = "rare_string"
    RELATION = "relation"
    ACTION = "action"
    STATE_TRANSITION = "state_transition"
    PERMISSION = "permission"
    DEPENDENCY = "dependency"
    ERROR_CODE = "error_code"
    BACKEND_TYPE = "backend_type"
    SCHEMA_TYPE = "schema_type"
    THRESHOLD_ANOMALY = "threshold_anomaly"


class RecoveryPolicy(str, Enum):
    """Policies compared by the Paper 7 latent-trigger benchmark."""

    COMPACT_ONLY = "compact_only"
    CCR_EXPLICIT = "ccr_explicit"
    CCR_PROACTIVE = "ccr_proactive"
    CCR_MIXED = "ccr_mixed"
    GENERIC_PROACTIVE = "generic_proactive"
    ACTION_CONDITIONED = "action_conditioned"
    MULTI_HYPOTHESIS = "multi_hypothesis"
    FULL_CONTEXT = "full_context"


@dataclass(frozen=True)
class TriggerCase:
    """One explicit/latent pair whose hidden evidence changes the action."""

    case_id: str
    family: TriggerFamily | str
    record_type: RecordType | str
    payload: object
    hidden_trigger: str
    hidden_evidence: str
    explicit_query: str
    latent_query: str
    candidate_actions: tuple[str, ...]
    required_action: str
    misleading_action: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", TriggerFamily(self.family))
        object.__setattr__(self, "record_type", RecordType(self.record_type))
        object.__setattr__(self, "candidate_actions", tuple(self.candidate_actions))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.case_id or not self.hidden_trigger or not self.hidden_evidence:
            raise ValueError("case_id, hidden_trigger, and hidden_evidence are required.")
        if self.required_action not in self.candidate_actions:
            raise ValueError("required_action must be in candidate_actions.")
        if self.misleading_action not in self.candidate_actions:
            raise ValueError("misleading_action must be in candidate_actions.")
        if self.explicit_query == self.latent_query:
            raise ValueError("Explicit and latent queries must differ.")
        if self.hidden_trigger.casefold() not in self.explicit_query.casefold():
            raise ValueError("Explicit query must name the hidden trigger.")
        if self.hidden_trigger.casefold() in self.latent_query.casefold():
            raise ValueError("Latent query must omit the hidden trigger.")


@dataclass(frozen=True)
class RecoveryHandle:
    """Stable model-visible marker resolving one exact scoped original."""

    marker: str
    record_id: str
    scope_fingerprint: str
    content_hash: str


class CCRStyleBaseline:
    """Architectural reversible-compression baseline with exact local recovery."""

    def __init__(self, runtime: AdaptiveContextRuntime) -> None:
        self.runtime = runtime
        self._handles: dict[str, RecoveryHandle] = {}

    def register(self, record_id: str) -> RecoveryHandle:
        record = self.runtime.records[record_id]
        digest = hashlib.sha256(
            f"{record_id}\0{record.backing.content_hash}".encode("utf-8")
        ).hexdigest()[:24]
        handle = RecoveryHandle(
            marker=f"ccr://{self.runtime.scope.fingerprint}/{digest}",
            record_id=record_id,
            scope_fingerprint=self.runtime.scope.fingerprint,
            content_hash=record.backing.content_hash,
        )
        self._handles[handle.marker] = handle
        return handle

    def retrieve(
        self,
        marker: str,
        *,
        scope: RecordScope | None = None,
        level: RecordViewName | str = RecordViewName.FULL,
    ) -> MaterializationResult:
        handle = self._handles.get(marker)
        caller_scope = scope or self.runtime.scope
        if handle is None:
            raise KeyError(marker)
        if caller_scope.fingerprint != handle.scope_fingerprint:
            raise RecordAccessDenied(marker)
        result = self.runtime.retrieve_record(handle.record_id, level=level, scope=caller_scope)
        record = self.runtime.records[handle.record_id]
        if record.backing.content_hash != handle.content_hash:
            raise ValueError("CCR handle content hash no longer matches the record.")
        return result


@dataclass(frozen=True)
class Probe:
    """One bounded hypothesis rendered as an address-search query."""

    text: str
    hypothesis: str
    action: str | None = None
    source: str = "deterministic"


@dataclass(frozen=True)
class ProbeBatchResult:
    """Discovery/materialization result for one bounded probe batch."""

    probes: tuple[Probe, ...]
    touched_record_ids: tuple[str, ...]
    materializations: tuple[MaterializationResult, ...]
    expected_record_retrieved: bool
    expected_trigger_materialized: bool
    false_positive_expansions: int
    search_latency_seconds: float
    materialized_bytes: int

    @property
    def expansion_precision(self) -> float:
        if not self.materializations:
            return 0.0
        useful = int(self.expected_trigger_materialized)
        return useful / len(self.materializations)


@dataclass(frozen=True)
class RecoveryMetrics:
    """Stage-separated metrics for one policy/case/model presentation."""

    hypothesis_recall: bool
    address_recall: bool
    trigger_materialized: bool
    final_action_correct: bool
    expansion_count: int
    false_positive_expansions: int
    materialized_bytes: int
    model_calls: int
    retrieval_tool_calls: int
    proactive_expansions: int

    @property
    def tr_address(self) -> bool:
        return self.hypothesis_recall and self.address_recall

    @property
    def tr_materialize(self) -> bool:
        return self.trigger_materialized

    @property
    def tr_action(self) -> bool:
        return self.final_action_correct

    @property
    def expansion_precision(self) -> float:
        if self.expansion_count == 0:
            return 0.0
        return (self.expansion_count - self.false_positive_expansions) / self.expansion_count


def build_action_probe(goal: str, action: str) -> Probe:
    """Build the fixed action-conditioned probe used by the mechanism control."""

    action_text = action.replace("_", " ")
    return Probe(
        text=(
            f"Goal: {goal} Candidate action: {action_text} (action_id={action}). "
            "Find hidden evidence relevant to whether this action is required."
        ),
        hypothesis=f"evidence may require {action_text}",
        action=action,
        source="action_template",
    )


def action_conditioned_probes(
    goal: str,
    candidate_actions: Sequence[str],
    *,
    limit: int,
) -> tuple[Probe, ...]:
    """Probe at most ``limit`` candidate actions in caller-provided rank order."""

    if limit <= 0:
        raise ValueError("Probe limit must be positive.")
    return tuple(build_action_probe(goal, action) for action in candidate_actions[:limit])


def hypothesis_probes(
    goal: str,
    hypotheses: Sequence[str],
    *,
    limit: int,
    source: str = "model",
) -> tuple[Probe, ...]:
    """Convert a bounded hypothesis list into retrieval-oriented probes."""

    if limit <= 0:
        raise ValueError("Hypothesis limit must be positive.")
    return tuple(
        Probe(
            text=f"Goal: {goal} Possible hidden consideration: {hypothesis}",
            hypothesis=str(hypothesis),
            source=source,
        )
        for hypothesis in tuple(hypotheses)[:limit]
        if str(hypothesis).strip()
    )


class LatentRecoveryEngine:
    """Runs bounded probes against retrieval-only views and exact backing state."""

    def __init__(self, runtime: AdaptiveContextRuntime) -> None:
        self.runtime = runtime

    def execute_probes(
        self,
        probes: Sequence[Probe],
        *,
        expected_record_id: str,
        hidden_trigger: str,
        address_kinds: Sequence[str] | None = None,
        per_probe_k: int = 1,
        materialize: bool = True,
    ) -> ProbeBatchResult:
        if per_probe_k <= 0:
            raise ValueError("per_probe_k must be positive.")
        started = time.perf_counter()
        touched: list[str] = []
        for probe in probes:
            for record in self.runtime.search_records(
                probe.text,
                top_k=per_probe_k,
                address_kinds=address_kinds,
            ):
                if record.record_id not in touched:
                    touched.append(record.record_id)
        search_latency = time.perf_counter() - started
        expansions: list[MaterializationResult] = []
        if materialize:
            for record_id in touched:
                expansions.append(self.runtime.proactive_materialize(
                    MaterializationEvent(record_id),
                    reason="latent-trigger probe",
                ))
        trigger_visible = any(
            hidden_trigger.casefold() in _payload_text(result.payload).casefold()
            for result in expansions
            if result.record_id == expected_record_id
        )
        false_positives = sum(result.record_id != expected_record_id for result in expansions)
        return ProbeBatchResult(
            tuple(probes),
            tuple(touched),
            tuple(expansions),
            expected_record_id in touched,
            trigger_visible,
            false_positives,
            search_latency,
            sum(result.payload_bytes for result in expansions),
        )


def parse_cursor_action(
    value: Mapping[str, object],
    *,
    cursor_id: str,
    allowed_operations: Sequence[CursorOperation | str] | None = None,
) -> CursorAction:
    """Parse an untrusted model decision while pinning the authorized cursor."""

    operation = CursorOperation(str(value.get("operation", "")))
    allowed = {
        CursorOperation(item)
        for item in (allowed_operations or tuple(CursorOperation))
    }
    if operation not in allowed:
        raise ValueError(f"Cursor operation is not allowed in this task: {operation.value}")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise ValueError("Cursor action arguments must be a mapping.")
    return CursorAction(cursor_id, operation, arguments)


def trigger_case_fingerprint(cases: Sequence[TriggerCase]) -> str:
    """Stable benchmark fingerprint used by determinism tests and manifests."""

    serializable = [
        {
            "case_id": case.case_id,
            "family": case.family.value,
            "record_type": case.record_type.value,
            "hidden_trigger": case.hidden_trigger,
            "explicit_query": case.explicit_query,
            "latent_query": case.latent_query,
            "candidate_actions": case.candidate_actions,
            "required_action": case.required_action,
        }
        for case in cases
    ]
    payload = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hypothesis_matches_trigger(hypotheses: Sequence[str], case: TriggerCase) -> bool:
    """Score whether a generated hypothesis names the trigger or required action."""

    needles = {
        case.hidden_trigger.casefold(),
        case.required_action.casefold(),
        case.required_action.replace("_", " ").casefold(),
    }
    return any(
        needle in hypothesis.casefold()
        for hypothesis in hypotheses
        for needle in needles
    )


def _payload_text(payload: object) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


HypothesisGenerator = Callable[[str, Sequence[str], int], Sequence[str]]
