"""Model-visible realizations of H1/H2 negative-selection evidence.

The negative heuristics normally remove an entire action--observation group.
This module provides a deliberately more conservative experimental arm: keep
the original assistant action verbatim and replace only its observation body
with a compact deterministic receipt.  The receipt is built exclusively from
records already present in the frozen-history prefix. It is emitted only when
the experiment's exact token counter proves it is strictly smaller than the
source observation; otherwise the complete original group is retained.

This is a materialization treatment, not a certificate that an LLM will behave
identically.  Logical full-record tokens and materialized receipt tokens remain
separate in the returned accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from .materialization import (
    MaterializationMode,
    MaterializedMemoryPlan,
    MaterializedRecord,
)
from .model import (
    AgentMemoryExclusion,
    AgentMemoryPlan,
    AgentRecord,
    AgentRecordRole,
    CanonicalAgentHistory,
)
from .selectors import TokenCounter, whitespace_tokens


class NegativeRealizationMode(str, Enum):
    """How a negative-selection candidate is exposed to the model."""

    DROP = "drop"
    OBSERVATION_RECEIPT = "observation_receipt"


class ReceiptKind(str, Enum):
    RESOURCE_MAP = "resource_map"
    MUTATION = "mutation"


class ReceiptAbstentionReason(str, Enum):
    MALFORMED_OR_PREFIX_INVALID = "malformed_or_prefix_invalid"
    NOT_SMALLER_THAN_SOURCE_OBSERVATION = "not_smaller_than_source_observation"


@dataclass(frozen=True)
class ModelVisibleReceipt:
    causal_group_id: str
    observation_record_id: str
    source_record_ids: tuple[str, ...]
    retained_action_record_ids: tuple[str, ...]
    rule_id: str
    kind: ReceiptKind
    content: str
    source_observation_tokens: int
    receipt_tokens: int
    retained_action_tokens: int
    witness_record_ids: tuple[str, ...]

    @property
    def observation_token_saving(self) -> int:
        return self.source_observation_tokens - self.receipt_tokens


@dataclass(frozen=True)
class ReceiptAbstention:
    causal_group_id: str
    source_record_ids: tuple[str, ...]
    rule_id: str
    reason: ReceiptAbstentionReason
    source_observation_tokens: int | None
    candidate_receipt_tokens: int | None


@dataclass(frozen=True)
class NegativeReceiptRealization:
    materialized: MaterializedMemoryPlan
    receipts: tuple[ModelVisibleReceipt, ...]
    dropped_exclusions: tuple[AgentMemoryExclusion, ...]
    abstentions: tuple[ReceiptAbstention, ...]

    @property
    def fail_closed_group_ids(self) -> tuple[str, ...]:
        return tuple(row.causal_group_id for row in self.abstentions)

    @property
    def receipt_tokens(self) -> int:
        return sum(row.receipt_tokens for row in self.receipts)

    @property
    def receipt_source_observation_tokens(self) -> int:
        return sum(row.source_observation_tokens for row in self.receipts)

    @property
    def retained_action_tokens(self) -> int:
        return sum(row.retained_action_tokens for row in self.receipts)

    @property
    def receipt_token_saving(self) -> int:
        return self.receipt_source_observation_tokens - self.receipt_tokens

    @property
    def abstained_source_observation_tokens(self) -> int:
        return sum(
            row.source_observation_tokens or 0 for row in self.abstentions
        )

    @property
    def abstained_candidate_receipt_tokens(self) -> int:
        return sum(row.candidate_receipt_tokens or 0 for row in self.abstentions)


def _receipt_kind(rule_id: str) -> ReceiptKind | None:
    if rule_id.startswith("H1_"):
        return ReceiptKind.RESOURCE_MAP
    if rule_id.startswith("H2"):
        return ReceiptKind.MUTATION
    return None


def _string_metadata(record: AgentRecord, key: str) -> str | None:
    value = record.metadata.get(key)
    return value if isinstance(value, str) and value else None


def _versions(record: AgentRecord, resources: tuple[str, ...]) -> dict[str, str]:
    value = record.metadata.get("post_resource_version_fingerprints")
    if not isinstance(value, Mapping):
        return {}
    wanted = set(resources)
    return {
        str(resource): str(version)
        for resource, version in sorted(value.items(), key=lambda row: str(row[0]))
        if str(resource) in wanted and version is not None
    }


def _receipt_payload(
    *,
    exclusion: AgentMemoryExclusion,
    observation: AgentRecord,
    kind: ReceiptKind,
) -> dict[str, object]:
    # Resource IDs are produced from the current history prefix by the
    # negative selector. Raw observation contents and any future reference
    # action are deliberately unavailable to this renderer.
    concrete_resources = tuple(
        sorted(
            resource for resource in exclusion.resource_ids
            if not resource.startswith("search:")
        )
    )
    common: dict[str, object] = {
        "resources": concrete_resources,
        "return_code": (
            observation.return_code
            if observation.return_code is not None else "unknown"
        ),
        "output_complete": (
            observation.metadata.get("output_complete")
            if isinstance(observation.metadata.get("output_complete"), bool)
            else "unknown"
        ),
    }
    cwd = _string_metadata(observation, "cwd")
    if cwd is not None:
        common["cwd"] = cwd
    if kind == ReceiptKind.RESOURCE_MAP:
        common["state"] = "discovery_consumed"
        common["detail"] = "full discovery listing elided; mapped resources remain named"
    else:
        common["state"] = (
            "mutation_represented_by_current_read"
            if exclusion.rule_id == "H2A_WRITE_CURRENT_READ"
            else "mutation_followed_by_successful_verification"
            if exclusion.rule_id == "H2B_VERIFIED_WRITE"
            else "mutation_payload_elided_by_aggressive_rule"
        )
        versions = _versions(observation, concrete_resources)
        if versions:
            common["post_resource_versions"] = versions
        common["detail"] = "original mutation action retained; tool-output payload elided"
    return common


def _compact_value(value: object) -> str:
    """Render trusted runtime metadata on one deterministic bounded line."""

    return " ".join(str(value).split())[:240]


def _render_receipt(payload: dict[str, object]) -> str:
    resources = tuple(str(value) for value in payload["resources"])
    versions = payload.get("post_resource_versions")
    version_map = versions if isinstance(versions, Mapping) else {}
    resource_text = ", ".join(
        f"{resource}@{version_map[resource]}"
        if resource in version_map else resource
        for resource in resources
    ) or "none"
    state = str(payload["state"])
    if state == "discovery_consumed":
        fact = f"search consumed; mapped resources: {resource_text}"
    elif state == "mutation_represented_by_current_read":
        fact = f"mutation retained; current read confirms: {resource_text}"
    elif state == "mutation_followed_by_successful_verification":
        fact = f"mutation retained; verification passed for: {resource_text}"
    else:
        fact = f"mutation retained; output elided by aggressive rule: {resource_text}"
    completeness = payload["output_complete"]
    if completeness is True:
        fact += "; output complete"
    elif completeness == "unknown":
        fact += "; completeness unknown"
    else:
        fact += "; output incomplete"
    if cwd := payload.get("cwd"):
        fact += f"; cwd: {_compact_value(cwd)}"
    return (
        f"<returncode>{payload['return_code']}</returncode>\n"
        f"<output>[PRA memory] {fact}</output>"
    )


def realize_negative_receipts(
    history: CanonicalAgentHistory,
    plan: AgentMemoryPlan,
    materialized: MaterializedMemoryPlan,
    *,
    count_tokens: TokenCounter = whitespace_tokens,
) -> NegativeReceiptRealization:
    """Replace eligible H1/H2 observations with prefix-derived receipts.

    H1/H2 groups must be one assistant action plus one tool observation.  An
    eligible but malformed group is reinserted whole (fail closed).  Other
    negative rules retain their normal drop realization.  Existing
    within-record materialization of selected records is preserved.
    """

    source = history.record_by_id
    rows_by_id = {row.record_id: row for row in materialized.records}
    receipts: list[ModelVisibleReceipt] = []
    dropped: list[AgentMemoryExclusion] = []
    abstentions: list[ReceiptAbstention] = []

    def reinsert_whole(group_records: tuple[AgentRecord, ...]) -> None:
        for record in group_records:
            tokens = count_tokens(record.content)
            rows_by_id[record.record_id] = MaterializedRecord(
                record.record_id,
                record.content,
                MaterializationMode.WHOLE_RECORD,
                tokens,
                tokens,
            )

    for exclusion in plan.exclusions:
        kind = _receipt_kind(exclusion.rule_id)
        if kind is None:
            dropped.append(exclusion)
            continue
        group_records = tuple(
            source[record_id]
            for record_id in exclusion.record_ids
            if record_id in source
        )
        actions = tuple(
            row for row in group_records
            if row.has_role(AgentRecordRole.ASSISTANT_ACTION)
        )
        observations = tuple(
            row for row in group_records
            if row.has_role(AgentRecordRole.TOOL_OBSERVATION)
        )
        witness_ids = tuple(dict.fromkeys(exclusion.witness_record_ids))
        prefix_ids = set(source)
        structurally_valid = (
            len(group_records) == len(exclusion.record_ids)
            and len(actions) == 1
            and len(observations) == 1
            and set(witness_ids) <= prefix_ids
        )
        if not structurally_valid:
            abstentions.append(ReceiptAbstention(
                exclusion.causal_group_id,
                exclusion.record_ids,
                exclusion.rule_id,
                ReceiptAbstentionReason.MALFORMED_OR_PREFIX_INVALID,
                None,
                None,
            ))
            reinsert_whole(group_records)
            continue

        action = actions[0]
        observation = observations[0]
        action_tokens = count_tokens(action.content)
        observation_tokens = count_tokens(observation.content)
        payload = _receipt_payload(
            exclusion=exclusion,
            observation=observation,
            kind=kind,
        )
        content = _render_receipt(payload)
        receipt_tokens = count_tokens(content)
        if receipt_tokens >= observation_tokens:
            abstentions.append(ReceiptAbstention(
                exclusion.causal_group_id,
                exclusion.record_ids,
                exclusion.rule_id,
                ReceiptAbstentionReason.NOT_SMALLER_THAN_SOURCE_OBSERVATION,
                observation_tokens,
                receipt_tokens,
            ))
            reinsert_whole(group_records)
            continue
        rows_by_id[action.record_id] = MaterializedRecord(
            action.record_id,
            action.content,
            MaterializationMode.WHOLE_RECORD,
            action_tokens,
            action_tokens,
        )
        rows_by_id[observation.record_id] = MaterializedRecord(
            observation.record_id,
            content,
            MaterializationMode.TOOL_NEGATIVE_RECEIPT,
            observation_tokens,
            receipt_tokens,
        )
        receipts.append(ModelVisibleReceipt(
            exclusion.causal_group_id,
            observation.record_id,
            exclusion.record_ids,
            (action.record_id,),
            exclusion.rule_id,
            kind,
            content,
            observation_tokens,
            receipt_tokens,
            action_tokens,
            witness_ids,
        ))

    ordered_rows = tuple(sorted(
        rows_by_id.values(), key=lambda row: source[row.record_id].message_index
    ))
    selected_ids = tuple(row.record_id for row in ordered_rows)
    selected_groups = tuple(dict.fromkeys(
        source[record_id].causal_group_id for record_id in selected_ids
    ))
    original_selected_tokens = sum(
        count_tokens(source[record_id].content) for record_id in selected_ids
    )
    selection_reasons = dict(plan.selection_reasons)
    for receipt in receipts:
        for record_id in receipt.source_record_ids:
            selection_reasons[record_id] = (
                f"model_visible_{receipt.kind.value}_receipt:{receipt.rule_id}"
            )
    for abstention in abstentions:
        group = next(
            row for row in plan.exclusions
            if row.causal_group_id == abstention.causal_group_id
        )
        for record_id in group.record_ids:
            selection_reasons[record_id] = (
                f"receipt_fail_closed:{abstention.reason.value}"
            )
    derived_plan = replace(
        plan,
        policy=f"{plan.policy}+observation_receipt",
        selected_record_ids=selected_ids,
        selected_causal_group_ids=selected_groups,
        selection_reasons=tuple(
            (record_id, selection_reasons.get(record_id, "selected"))
            for record_id in selected_ids
        ),
        selected_tokens=original_selected_tokens,
    )
    realized = MaterializedMemoryPlan(
        logical_plan=derived_plan,
        records=ordered_rows,
        full_selected_tokens=sum(row.original_tokens for row in ordered_rows),
        materialized_tokens=sum(row.materialized_tokens for row in ordered_rows),
    )
    return NegativeReceiptRealization(
        realized,
        tuple(receipts),
        tuple(dropped),
        tuple(abstentions),
    )
