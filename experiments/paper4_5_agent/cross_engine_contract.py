"""Fail-closed contract for Paper 4.5 cross-engine agent experiments."""

from __future__ import annotations

from typing import Any, Mapping


class CrossEngineContractError(ValueError):
    pass


IDENTITY_FIELDS = (
    "agent_id",
    "agent_revision",
    "model_id",
    "model_revision",
    "precision",
    "tokenizer_id",
    "tokenizer_revision",
    "chat_template_digest",
    "task_id",
    "task_snapshot_digest",
    "temperature",
    "top_p",
    "seed",
    "max_completion_tokens",
)

PLAN_FIELDS = (
    "history_digest",
    "plan_digest",
    "selected_record_ids",
    "full_logical_tokens",
    "selected_logical_tokens",
)


def _required(row: Mapping[str, Any], field: str, engine: str) -> Any:
    value = row.get(field)
    if value is None or value == "" or value == []:
        raise CrossEngineContractError(f"{engine}: missing required field {field}")
    return tuple(value) if field == "selected_record_ids" else value


def validate_cross_engine_cells(
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate that engines realize one identical logical experiment.

    Each cell has ``identity`` and ``logical_plan`` objects.  Engine identity
    and physical metrics deliberately live outside those objects.  Physical
    page rounding is allowed; changing logical tokens or selected IDs is not.
    """

    if len(cells) < 2:
        raise CrossEngineContractError("at least two engine cells are required")
    normalized: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    for engine, cell in cells.items():
        identity = cell.get("identity")
        plan = cell.get("logical_plan")
        if not isinstance(identity, Mapping) or not isinstance(plan, Mapping):
            raise CrossEngineContractError(
                f"{engine}: identity and logical_plan objects are required"
            )
        normalized[engine] = (
            tuple(_required(identity, field, engine) for field in IDENTITY_FIELDS),
            tuple(_required(plan, field, engine) for field in PLAN_FIELDS),
        )

    reference_engine = next(iter(normalized))
    reference_identity, reference_plan = normalized[reference_engine]
    failures: list[str] = []
    for engine, (identity, plan) in normalized.items():
        for fields, expected, actual in (
            (IDENTITY_FIELDS, reference_identity, identity),
            (PLAN_FIELDS, reference_plan, plan),
        ):
            for field, before, after in zip(fields, expected, actual):
                if before != after:
                    failures.append(
                        f"{engine}.{field}={after!r}; "
                        f"{reference_engine}.{field}={before!r}"
                    )
    if failures:
        raise CrossEngineContractError(
            "confounded cross-engine experiment: " + "; ".join(failures)
        )

    full_tokens = int(reference_plan[PLAN_FIELDS.index("full_logical_tokens")])
    selected_tokens = int(
        reference_plan[PLAN_FIELDS.index("selected_logical_tokens")]
    )
    if selected_tokens > full_tokens:
        raise CrossEngineContractError(
            "selected_logical_tokens exceeds full_logical_tokens"
        )
    return {
        "qualified": True,
        "engines": list(cells),
        "reference_engine": reference_engine,
        "full_logical_tokens": full_tokens,
        "selected_logical_tokens": selected_tokens,
        "logical_saving_fraction": (
            0.0 if full_tokens == 0 else 1.0 - selected_tokens / full_tokens
        ),
        "claim_boundary": (
            "Logical saving is identical by construction. Compare only physical "
            "rounding, copies, temporary bytes, residence, and latency by engine."
        ),
    }

