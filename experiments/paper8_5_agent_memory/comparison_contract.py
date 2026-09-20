"""Fail-closed identity checks for agent and engine comparisons.

Paper 8.5 compares memory policies across agents.  Paper 4.5 compares physical
realizations of a frozen logical plan across engines.  These are different
experimental axes: an agent may generate a different history, but an engine
must not change the history or selection plan it is asked to realize.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence


class ComparisonAxis(str, Enum):
    AGENT = "agent"
    ENGINE = "engine"


class ComparisonContractError(ValueError):
    """Raised when a proposed comparison changes more than its declared axis."""


@dataclass(frozen=True)
class ExperimentIdentity:
    """Coordinates that determine an agent-memory experiment.

    Revisions and digests are required strings rather than optional labels so
    that aliases such as ``qwen3-coder:30b`` cannot silently move.
    """

    agent_id: str
    agent_revision: str
    engine_id: str
    engine_revision: str
    model_id: str
    model_revision: str
    observed_model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_digest: str
    task_ids: tuple[str, ...]
    task_snapshot_digest: str
    policy_id: str
    policy_revision: str
    policy_parameters_digest: str
    materialization_mode: str
    temperature: float
    top_p: float
    seed: int
    max_completion_tokens: int
    context_limit: int
    native_agent_compaction: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        missing = [
            key for key, value in asdict(self).items()
            if key != "native_agent_compaction" and value in (None, "", ())
        ]
        if missing:
            raise ComparisonContractError(
                "experiment identity has unresolved fields: " + ", ".join(missing)
            )
        if "whitespace" in self.tokenizer_id.lower():
            raise ComparisonContractError(
                "diagnostic whitespace tokenization is inadmissible for a "
                "controlled agent or engine comparison"
            )
        if self.native_agent_compaction:
            raise ComparisonContractError(
                "agent-native compaction must be disabled in controlled comparisons"
            )


@dataclass(frozen=True)
class FrozenPlanObservation:
    """Logical accounting emitted before an engine consumes a selected plan."""

    history_digest: str
    plan_digest: str
    selected_record_ids: tuple[str, ...]
    full_logical_tokens: int
    selected_logical_tokens: int
    tokenizer_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selected_record_ids", tuple(self.selected_record_ids)
        )
        if self.full_logical_tokens < 0 or self.selected_logical_tokens < 0:
            raise ComparisonContractError("logical token counts cannot be negative")
        if self.selected_logical_tokens > self.full_logical_tokens:
            raise ComparisonContractError(
                "selected logical tokens cannot exceed full logical tokens"
            )

    @property
    def logical_saving_fraction(self) -> float:
        if self.full_logical_tokens == 0:
            return 0.0
        return 1.0 - self.selected_logical_tokens / self.full_logical_tokens


_AGENT_AXIS_FIELDS = frozenset({"agent_id", "agent_revision"})
_ENGINE_AXIS_FIELDS = frozenset({"engine_id", "engine_revision"})


def comparison_mismatches(
    left: ExperimentIdentity,
    right: ExperimentIdentity,
    *,
    axis: ComparisonAxis | str,
) -> dict[str, tuple[object, object]]:
    """Return non-axis identity changes that confound a comparison."""

    axis = ComparisonAxis(axis)
    allowed = _AGENT_AXIS_FIELDS if axis is ComparisonAxis.AGENT else _ENGINE_AXIS_FIELDS
    left_values = asdict(left)
    right_values = asdict(right)
    return {
        key: (left_values[key], right_values[key])
        for key in left_values
        if key not in allowed and left_values[key] != right_values[key]
    }


def validate_controlled_comparison(
    left: ExperimentIdentity,
    right: ExperimentIdentity,
    *,
    axis: ComparisonAxis | str,
) -> None:
    """Reject model, tokenizer, task, policy, or sampling drift.

    Agent comparisons keep the engine fixed.  Engine comparisons keep the
    agent fixed.  Both keep model weights, tokenizer, tasks, policy parameters,
    materialization semantics, and generation parameters fixed.
    """

    axis = ComparisonAxis(axis)
    mismatches = comparison_mismatches(left, right, axis=axis)
    if mismatches:
        detail = "; ".join(
            f"{key}: {before!r} != {after!r}"
            for key, (before, after) in sorted(mismatches.items())
        )
        raise ComparisonContractError(
            f"confounded cross-{axis.value} comparison: {detail}"
        )


def validate_frozen_plan_across_engines(
    observations: Mapping[str, FrozenPlanObservation],
) -> FrozenPlanObservation:
    """Require identical logical plans before comparing physical engines.

    Page/block rounding, resident bytes, copy bytes, and latency may differ by
    engine.  Logical full tokens, selected tokens, record identities, and plan
    digests may not.  A discrepancy is a failed experiment, not an engine
    savings result.
    """

    if len(observations) < 2:
        raise ComparisonContractError("cross-engine validation requires two engines")
    iterator = iter(observations.items())
    reference_engine, reference = next(iterator)
    fields: Sequence[str] = (
        "history_digest",
        "plan_digest",
        "selected_record_ids",
        "full_logical_tokens",
        "selected_logical_tokens",
        "tokenizer_revision",
    )
    failures: list[str] = []
    for engine, candidate in iterator:
        for field in fields:
            expected = getattr(reference, field)
            actual = getattr(candidate, field)
            if expected != actual:
                failures.append(
                    f"{engine}.{field}={actual!r}; "
                    f"{reference_engine}.{field}={expected!r}"
                )
    if failures:
        raise ComparisonContractError(
            "engine changed the frozen logical experiment: " + "; ".join(failures)
        )
    return reference
