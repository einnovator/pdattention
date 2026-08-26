"""Two-phase typed capability disclosure without decoder callbacks.

Phase A activates each candidate's deterministic selection view. After the
model chooses a stable record identity, the co-located runtime activates that
record's complete full view for Phase B. This is a local view transition: it
does not rerun semantic discovery or require an agent callback. An opaque
remote model provider can still require a second model request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .context_records import (
    ContextRecord,
    RecordAtomicity,
    RecordType,
    RecordViewName,
    serialize_record,
)


CAPABILITY_TYPES = frozenset({RecordType.TOOL_DEFINITION, RecordType.SKILL})


def bounded_candidate_ids(record_ids: Sequence[str], max_candidates: int) -> tuple[str, ...]:
    """Deduplicate a ranked identity list and enforce the hard palette cap."""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")
    return tuple(dict.fromkeys(record_ids))[:max_candidates]


def minimum_candidate_budget(
    target_record_id: str,
    ranked_record_ids: Sequence[str],
    candidate_budgets: Sequence[int],
) -> int | None:
    """Return the smallest evaluated K that contains the target, if any."""

    ranked = tuple(dict.fromkeys(ranked_record_ids))
    budgets = tuple(sorted(set(int(value) for value in candidate_budgets)))
    if any(value <= 0 for value in budgets):
        raise ValueError("candidate_budgets must contain only positive values.")
    return next(
        (budget for budget in budgets if target_record_id in ranked[:budget]),
        None,
    )


@dataclass(frozen=True)
class CapabilityTransition:
    """Auditable runtime transition from a known selection to its full view."""

    record_id: str
    from_view: RecordViewName | str = RecordViewName.SELECTION
    to_view: RecordViewName | str = RecordViewName.FULL

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_view", RecordViewName(self.from_view))
        object.__setattr__(self, "to_view", RecordViewName(self.to_view))
        if not self.record_id:
            raise ValueError("CapabilityTransition requires record_id.")
        if self.from_view == self.to_view:
            raise ValueError("CapabilityTransition must change the named view.")
        if self.from_view != RecordViewName.SELECTION or self.to_view != RecordViewName.FULL:
            raise ValueError("Paper 6.5 supports only selection-to-full transitions.")


@dataclass(frozen=True)
class CapabilityViewMaterialization:
    """Complete record-view payload admitted during one explicit phase."""

    phase: str
    view: RecordViewName
    record_ids: tuple[str, ...]
    serialized_payload: str
    serialized_tokens: int
    serialized_bytes: int
    native_kv_bytes: int
    atomicity_violations: int = 0


@dataclass(frozen=True)
class ProgressiveDisclosureCost:
    """Token accounting for full-all and selection-to-full disclosure."""

    candidate_count: int
    all_candidate_full_tokens: int
    phase_a_selection_tokens: int
    phase_b_selected_full_tokens: int
    progressive_tokens: int
    full_candidate_tokens_avoided: int
    tokens_saved: int
    token_savings_fraction: float
    disclosure_ratio: float
    total_disclosure_ratio: float


@dataclass(frozen=True)
class ProgressiveKVCost:
    """Active native-K/V accounting derived from measured view token counts."""

    native_kv_bytes_per_token: int
    full_all_active_bytes: int
    progressive_active_bytes: int
    bytes_saved: int
    savings_fraction: float


@dataclass(frozen=True)
class CapabilityChoiceAccounting:
    """Retrieval, conditional choice, and end-to-end choice decomposition."""

    examples: int
    target_in_palette: int
    correct_choices: int
    retrieval_recall: float
    conditional_choice_accuracy: float
    end_to_end_choice_accuracy: float


def capability_choice_accounting(
    rows: Sequence[tuple[bool, bool]],
) -> CapabilityChoiceAccounting:
    """Aggregate ``(target_in_palette, choice_correct)`` observations.

    A correct choice outside the palette violates the bounded-choice protocol
    and is rejected instead of silently inflating end-to-end accuracy.
    """

    values = tuple((bool(target), bool(correct)) for target, correct in rows)
    if any(correct and not target for target, correct in values):
        raise ValueError("A bounded capability choice cannot be correct outside its palette.")
    examples = len(values)
    target_count = sum(target for target, _ in values)
    correct_count = sum(correct for _, correct in values)
    return CapabilityChoiceAccounting(
        examples=examples,
        target_in_palette=target_count,
        correct_choices=correct_count,
        retrieval_recall=target_count / max(examples, 1),
        conditional_choice_accuracy=correct_count / max(target_count, 1),
        end_to_end_choice_accuracy=correct_count / max(examples, 1),
    )


def materialize_capability_views(
    records: Sequence[ContextRecord],
    *,
    view: RecordViewName | str,
    phase: str,
    token_counter: Callable[[str], int],
    native_kv_bytes_per_token: int = 0,
) -> CapabilityViewMaterialization:
    """Materialize whole named views for an already bounded capability set."""

    view_name = RecordViewName(view)
    for record in records:
        if record.record_type not in CAPABILITY_TYPES:
            raise ValueError("Capability palettes accept only TOOL_DEFINITION or SKILL records.")
        if record.policy.atomicity != RecordAtomicity.RECORD:
            raise ValueError("Capability views must preserve record atomicity.")
        record.materialize(view_name)
    payload = "\n".join(serialize_record(record, view=view_name) for record in records)
    tokens = token_counter(payload) if payload else 0
    return CapabilityViewMaterialization(
        phase=phase,
        view=view_name,
        record_ids=tuple(record.record_id for record in records),
        serialized_payload=payload,
        serialized_tokens=tokens,
        serialized_bytes=len(payload.encode("utf-8")),
        native_kv_bytes=tokens * native_kv_bytes_per_token,
    )


def transition_selected_capability(
    records: Sequence[ContextRecord],
    transition: CapabilityTransition,
    *,
    token_counter: Callable[[str], int],
    native_kv_bytes_per_token: int = 0,
) -> CapabilityViewMaterialization:
    """Materialize only the selected full record for Phase B."""

    by_id = {record.record_id: record for record in records}
    if transition.record_id not in by_id:
        raise ValueError("Selected capability is not present in the Phase-A palette.")
    record = by_id[transition.record_id]
    if transition.from_view not in record.views or transition.to_view not in record.views:
        raise ValueError("Capability transition references a missing named view.")
    if record.policy.selected_view != transition.to_view:
        raise ValueError("Capability transition violates the selected-view policy.")
    return materialize_capability_views(
        (record,),
        view=transition.to_view,
        phase="B",
        token_counter=token_counter,
        native_kv_bytes_per_token=native_kv_bytes_per_token,
    )


def disclosure_cost(
    records: Sequence[ContextRecord],
    *,
    selected_record_id: str,
    token_counter: Callable[[str], int],
) -> ProgressiveDisclosureCost:
    """Compare two-phase exposure with all candidate full records."""

    full = materialize_capability_views(
        records, view=RecordViewName.FULL, phase="full_all", token_counter=token_counter
    )
    selection = materialize_capability_views(
        records, view=RecordViewName.SELECTION, phase="A", token_counter=token_counter
    )
    selected = transition_selected_capability(
        records,
        CapabilityTransition(selected_record_id),
        token_counter=token_counter,
    )
    denominator = max(full.serialized_tokens, 1)
    progressive_tokens = selection.serialized_tokens + selected.serialized_tokens
    tokens_saved = full.serialized_tokens - progressive_tokens
    return ProgressiveDisclosureCost(
        candidate_count=len(records),
        all_candidate_full_tokens=full.serialized_tokens,
        phase_a_selection_tokens=selection.serialized_tokens,
        phase_b_selected_full_tokens=selected.serialized_tokens,
        progressive_tokens=progressive_tokens,
        full_candidate_tokens_avoided=max(
            full.serialized_tokens - selected.serialized_tokens, 0
        ),
        tokens_saved=tokens_saved,
        token_savings_fraction=tokens_saved / denominator,
        disclosure_ratio=selection.serialized_tokens / denominator,
        total_disclosure_ratio=progressive_tokens / denominator,
    )


def native_kv_cost(
    disclosure: ProgressiveDisclosureCost,
    *,
    native_kv_bytes_per_token: int,
) -> ProgressiveKVCost:
    """Convert token exposure into active native-K/V bytes without cache claims."""

    if native_kv_bytes_per_token < 0:
        raise ValueError("native_kv_bytes_per_token must be non-negative.")
    full_bytes = disclosure.all_candidate_full_tokens * native_kv_bytes_per_token
    progressive_bytes = disclosure.progressive_tokens * native_kv_bytes_per_token
    saved = full_bytes - progressive_bytes
    return ProgressiveKVCost(
        native_kv_bytes_per_token=native_kv_bytes_per_token,
        full_all_active_bytes=full_bytes,
        progressive_active_bytes=progressive_bytes,
        bytes_saved=saved,
        savings_fraction=saved / max(full_bytes, 1),
    )
