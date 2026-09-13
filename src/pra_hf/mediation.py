"""Shared request-mediation contract for external gateways and embedded runtimes.

This module owns placement, mode resolution, record inference, shadow/active
feature states, and cross-hop idempotency.  It intentionally does not contain
agent- or tool-specific syntax and it does not touch model-native K/V.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Protocol

from .agent_history import AgentRecordizer, OpenAIRecordizer, RecordizationResult
from .deployment import PRAEngineCapabilities, PRAGatewayMode, PRAWireRequest


class MediationLocation(str, Enum):
    NONE = "none"
    EXTERNAL = "external"
    EMBEDDED = "embedded"


class MediationFeatureMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ACTIVE = "active"


class InferenceFailurePolicy(str, Enum):
    FULL_PASSTHROUGH = "full_passthrough"
    ERROR = "error"


class CapabilityDowngradePolicy(str, Enum):
    ERROR = "error"
    SELECTED_CONTEXT = "selected_context"


class DoubleMediationPolicy(str, Enum):
    SKIP_IDENTICAL = "skip_identical"
    REJECT = "reject"


@dataclass(frozen=True)
class RecordInferenceConfig:
    enabled: bool = True
    profile: str = "openai-standard-v1"
    require_exact_metadata: bool = False

    @classmethod
    def from_value(cls, value: object) -> "RecordInferenceConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("record_inference must be a boolean or mapping")


@dataclass(frozen=True)
class HistorySelectionConfig:
    mode: MediationFeatureMode | str = MediationFeatureMode.OFF
    policy: str = "full"
    head_turns: int = 2
    tail_turns: int = 4
    unknown_effect: str = "retain_full"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", MediationFeatureMode(self.mode))
        object.__setattr__(self, "options", dict(self.options))
        if self.head_turns < 0 or self.tail_turns < 0:
            raise ValueError("history head/tail turns cannot be negative")
        if self.unknown_effect not in {"retain_full", "error"}:
            raise ValueError("unknown_effect must be retain_full or error")

    @classmethod
    def from_value(cls, value: object) -> "HistorySelectionConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(mode=MediationFeatureMode.ACTIVE, policy=value)
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("history_selection must be a policy name or mapping")


@dataclass(frozen=True)
class ResultCompactionConfig:
    mode: MediationFeatureMode | str = MediationFeatureMode.OFF
    size_gate: str = "tokenizer_exact"
    preserve_exact_backing: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", MediationFeatureMode(self.mode))
        object.__setattr__(self, "options", dict(self.options))
        if self.size_gate not in {"tokenizer_exact", "bytes", "off"}:
            raise ValueError("unsupported result-compaction size gate")

    @classmethod
    def from_value(cls, value: object) -> "ResultCompactionConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(mode=value)
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("result_compaction must be a mode or mapping")


@dataclass(frozen=True)
class ToolDisclosureConfig:
    mode: MediationFeatureMode | str = MediationFeatureMode.OFF
    profile: str = "planning"
    max_tools: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", MediationFeatureMode(self.mode))
        if self.max_tools <= 0:
            raise ValueError("max_tools must be positive")

    @classmethod
    def from_value(cls, value: object) -> "ToolDisclosureConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(mode=value)
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("tool_disclosure must be a mode or mapping")


@dataclass(frozen=True)
class MediationIdempotencyConfig:
    prevent_double_mediation: bool = True
    on_conflict: DoubleMediationPolicy | str = DoubleMediationPolicy.SKIP_IDENTICAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "on_conflict", DoubleMediationPolicy(self.on_conflict))


@dataclass(frozen=True)
class PRAMediationConfig:
    location: MediationLocation | str = MediationLocation.NONE
    mode: str = "auto"
    strict_typed_contract: bool = True
    inference_failure: InferenceFailurePolicy | str = InferenceFailurePolicy.FULL_PASSTHROUGH
    capability_downgrade: CapabilityDowngradePolicy | str = CapabilityDowngradePolicy.ERROR
    record_inference: RecordInferenceConfig = field(default_factory=RecordInferenceConfig)
    history_selection: HistorySelectionConfig = field(default_factory=HistorySelectionConfig)
    result_compaction: ResultCompactionConfig = field(default_factory=ResultCompactionConfig)
    tool_disclosure: ToolDisclosureConfig = field(default_factory=ToolDisclosureConfig)
    idempotency: MediationIdempotencyConfig = field(default_factory=MediationIdempotencyConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", MediationLocation(self.location))
        object.__setattr__(self, "inference_failure", InferenceFailurePolicy(self.inference_failure))
        object.__setattr__(self, "capability_downgrade", CapabilityDowngradePolicy(self.capability_downgrade))
        if self.mode not in {"auto", *(mode.value for mode in PRAGatewayMode)}:
            raise ValueError("mediation mode must be auto or G00/G10/G01/G11")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "PRAMediationConfig") -> "PRAMediationConfig":
        if isinstance(value, cls):
            return value
        data = dict(value)
        # Accept the compact proposal spelling while normalizing it to the
        # explicit product contract.
        if "strict" in data and "strict_typed_contract" not in data:
            data["strict_typed_contract"] = bool(data.pop("strict"))
        if "infer_records" in data and "record_inference" not in data:
            data["record_inference"] = data.pop("infer_records")
        if "prevent_double_mediation" in data and "idempotency" not in data:
            data["idempotency"] = {
                "prevent_double_mediation": bool(data.pop("prevent_double_mediation"))
            }
        data["record_inference"] = RecordInferenceConfig.from_value(
            data.get("record_inference", True)
        )
        data["history_selection"] = HistorySelectionConfig.from_value(
            data.get("history_selection", {"mode": "off", "policy": "full"})
        )
        data["result_compaction"] = ResultCompactionConfig.from_value(
            data.get("result_compaction", "off")
        )
        data["tool_disclosure"] = ToolDisclosureConfig.from_value(
            data.get("tool_disclosure", "off")
        )
        idempotency = data.get("idempotency", {})
        data["idempotency"] = (
            idempotency
            if isinstance(idempotency, MediationIdempotencyConfig)
            else MediationIdempotencyConfig(**dict(idempotency))
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["location"] = self.location.value
        value["inference_failure"] = self.inference_failure.value
        value["capability_downgrade"] = self.capability_downgrade.value
        value["history_selection"]["mode"] = self.history_selection.mode.value
        value["result_compaction"]["mode"] = self.result_compaction.mode.value
        value["tool_disclosure"]["mode"] = self.tool_disclosure.mode.value
        value["idempotency"]["on_conflict"] = self.idempotency.on_conflict.value
        return value

    @property
    def policy_digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MediationStamp:
    schema_version: int
    location: str
    gateway_mode: str
    request_digest: str
    policy_digest: str
    plan_digest: str
    hop_count: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MediationStamp":
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MediationConflictError(RuntimeError):
    pass


class RecordInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WireAgentMemoryPlan:
    """Portable logical plan produced once and realized at either location."""

    schema_version: int
    policy: str
    selected_record_ids: tuple[str, ...]
    record_replacements: Mapping[str, str] = field(default_factory=dict)
    selected_tool_names: tuple[str, ...] = ()
    source_history_digest: str | None = None
    decision_metadata: Mapping[str, Any] = field(default_factory=dict)
    declared_plan_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported agent-memory plan schema")
        object.__setattr__(
            self, "selected_record_ids", tuple(dict.fromkeys(self.selected_record_ids))
        )
        object.__setattr__(self, "record_replacements", {
            str(key): str(value) for key, value in self.record_replacements.items()
        })
        object.__setattr__(
            self, "selected_tool_names", tuple(dict.fromkeys(self.selected_tool_names))
        )
        object.__setattr__(self, "decision_metadata", dict(self.decision_metadata))
        if self.declared_plan_digest is not None and self.declared_plan_digest != self.digest:
            raise ValueError("agent-memory plan digest does not match its contents")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WireAgentMemoryPlan":
        data = dict(value)
        data["selected_record_ids"] = tuple(data.get("selected_record_ids", ()))
        data["selected_tool_names"] = tuple(data.get("selected_tool_names", ()))
        if "plan_digest" in data and "declared_plan_digest" not in data:
            data["declared_plan_digest"] = data.pop("plan_digest")
        return cls(**data)

    @property
    def digest(self) -> str:
        value = {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "selected_record_ids": self.selected_record_ids,
            "record_replacements": dict(self.record_replacements),
            "selected_tool_names": self.selected_tool_names,
            "source_history_digest": self.source_history_digest,
            "decision_metadata": dict(self.decision_metadata),
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "selected_record_ids": list(self.selected_record_ids),
            "record_replacements": dict(self.record_replacements),
            "selected_tool_names": list(self.selected_tool_names),
            "source_history_digest": self.source_history_digest,
            "decision_metadata": dict(self.decision_metadata),
            "plan_digest": self.digest,
        }


class AgentMemoryPlanBuilder(Protocol):
    """Portable policy producer injected by a runtime with an exact tokenizer."""

    def build(
        self,
        history: Any,
        config: HistorySelectionConfig,
    ) -> WireAgentMemoryPlan: ...


@dataclass(frozen=True)
class PreparedMediation:
    request: PRAWireRequest
    resolved_mode: PRAGatewayMode
    recordization: RecordizationResult | None
    applied: bool
    skipped_existing: bool
    trace: Mapping[str, Any]


def _request_digest(request: PRAWireRequest) -> str:
    metadata = dict(request.metadata)
    metadata.pop("mediation_stamp", None)
    if metadata.pop("agent_records_generated_by_mediator", False):
        metadata.pop("agent_records", None)
    if metadata.pop("agent_memory_plan_generated_by_mediator", False):
        metadata.pop("agent_memory_plan", None)
    payload = request.to_dict()
    payload["metadata"] = metadata
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _has_typed_input(request: PRAWireRequest) -> bool:
    if request.resources or request.resource_ops:
        return True
    if request.metadata.get("agent_records"):
        return True
    return any(
        isinstance(message.get("pra_record"), Mapping)
        or (
            isinstance(message.get("metadata"), Mapping)
            and (
                isinstance(message["metadata"].get("pra_record"), Mapping)
                or isinstance(message["metadata"].get("pra"), Mapping)
            )
        )
        for message in request.messages
    )


def _tool_semantics_by_name(request: PRAWireRequest) -> dict[str, Mapping[str, Any]]:
    declarations: dict[str, Mapping[str, Any]] = {}
    for tool in request.tools:
        function = tool.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = str(function.get("name") or tool.get("name") or "")
        value = tool.get("x-pra-semantics", function.get("x-pra-semantics"))
        if name and isinstance(value, Mapping):
            declarations[name] = dict(value)
    return declarations


def resolve_auto_gateway_mode(
    request: PRAWireRequest,
    capabilities: PRAEngineCapabilities,
    *,
    inference_exact: bool,
    config: PRAMediationConfig,
) -> PRAGatewayMode:
    if config.location == MediationLocation.NONE:
        return PRAGatewayMode.G00_PASS_THROUGH
    typed = _has_typed_input(request)
    if typed and capabilities.logical_refs:
        return PRAGatewayMode.G11_MEDIATION
    if typed:
        if config.capability_downgrade == CapabilityDowngradePolicy.SELECTED_CONTEXT:
            return PRAGatewayMode.G10_TEXT_FALLBACK
        raise RecordInferenceError("typed mediation requires a logical-record engine")
    if capabilities.logical_refs and config.record_inference.enabled and inference_exact:
        return PRAGatewayMode.G01_UPGRADE
    return PRAGatewayMode.G00_PASS_THROUGH


class RequestMediator:
    """Normalize and stamp one request identically at either mediation location."""

    def __init__(
        self,
        config: PRAMediationConfig | Mapping[str, Any],
        *,
        recordizer: AgentRecordizer | None = None,
        plan_builder: AgentMemoryPlanBuilder | None = None,
    ) -> None:
        self.config = PRAMediationConfig.from_mapping(config)
        self.recordizer = recordizer or OpenAIRecordizer()
        self.plan_builder = plan_builder

    def prepare(
        self,
        request: PRAWireRequest,
        capabilities: PRAEngineCapabilities,
        *,
        plan_digest: str | None = None,
    ) -> PreparedMediation:
        if self.config.location == MediationLocation.NONE:
            return PreparedMediation(
                request,
                PRAGatewayMode.G00_PASS_THROUGH,
                None,
                False,
                False,
                {"status": "mediation_disabled", "resolved_mode": "G00"},
            )
        digest = _request_digest(request)
        plan_value = request.metadata.get("agent_memory_plan")
        plan = (
            WireAgentMemoryPlan.from_mapping(plan_value)
            if isinstance(plan_value, Mapping) else None
        )
        effective_plan_digest = plan_digest or (plan.digest if plan is not None else "full")
        existing_value = request.metadata.get("mediation_stamp")
        if isinstance(existing_value, Mapping) and self.config.idempotency.prevent_double_mediation:
            existing = MediationStamp.from_mapping(existing_value)
            identical = (
                existing.request_digest == digest
                and existing.policy_digest == self.config.policy_digest
                and (
                    existing.plan_digest == effective_plan_digest
                    or (plan is None and self.plan_builder is not None)
                )
            )
            if identical and self.config.idempotency.on_conflict == DoubleMediationPolicy.SKIP_IDENTICAL:
                return PreparedMediation(
                    request,
                    PRAGatewayMode(existing.gateway_mode),
                    None,
                    False,
                    True,
                    {"status": "already_mediated_identical", "stamp": existing.to_dict()},
                )
            raise MediationConflictError("request already contains an active mediation stamp")

        recordization = None
        inference_exact = _has_typed_input(request)
        if self.config.record_inference.enabled:
            recordization = self.recordizer.recordize(
                request.messages,
                request_metadata={
                    **request.metadata,
                    "session_id": request.session_id,
                    "tool_semantics_by_name": _tool_semantics_by_name(request),
                },
            )
            inference_exact = recordization.exact
            if not inference_exact and self.config.inference_failure == InferenceFailurePolicy.ERROR:
                raise RecordInferenceError(
                    "record inference was ambiguous: " + ", ".join(recordization.ambiguity_reasons)
                )

        generated_plan = False
        needs_plan = (
            self.config.history_selection.policy != "full"
            and self.config.history_selection.mode
            in {MediationFeatureMode.ACTIVE, MediationFeatureMode.SHADOW}
        )
        if plan is None and needs_plan and recordization is not None and self.plan_builder is not None:
            plan = self.plan_builder.build(
                recordization.history,
                self.config.history_selection,
            )
            effective_plan_digest = plan.digest
            generated_plan = True

        transformed = request
        mediation_trace: dict[str, Any] = {}
        if recordization is not None:
            transformed, mediation_trace = self._realize_logical_features(
                request, recordization, plan
            )
        resolved = (
            resolve_auto_gateway_mode(
                request, capabilities, inference_exact=inference_exact, config=self.config,
            )
            if self.config.mode == "auto"
            else PRAGatewayMode(self.config.mode)
        )
        # Shadow features never mutate model-visible input.  A full/pass-through
        # decision is nevertheless stamped so a later hop cannot silently run
        # a second active policy under a different contract.
        stamp = MediationStamp(
            schema_version=1,
            location=self.config.location.value,
            gateway_mode=resolved.value,
            request_digest=_request_digest(transformed),
            policy_digest=self.config.policy_digest,
            plan_digest=effective_plan_digest,
        )
        prepared = replace(
            transformed,
            metadata={**transformed.metadata, "mediation_stamp": stamp.to_dict()},
        )
        return PreparedMediation(
            prepared,
            resolved,
            recordization,
            True,
            False,
            {
                "status": "prepared",
                "requested_mode": self.config.mode,
                "resolved_mode": resolved.value,
                "location": self.config.location.value,
                "recordization_source": recordization.source if recordization else "typed",
                "recordization_exact": recordization.exact if recordization else True,
                "ambiguity_reasons": list(recordization.ambiguity_reasons) if recordization else [],
                "history_selection_mode": self.config.history_selection.mode.value,
                "result_compaction_mode": self.config.result_compaction.mode.value,
                "tool_disclosure_mode": self.config.tool_disclosure.mode.value,
                "agent_memory_plan_source": (
                    "generated" if generated_plan else "request" if plan is not None else "full"
                ),
                **mediation_trace,
                "stamp": stamp.to_dict(),
            },
        )

    def _realize_logical_features(
        self,
        request: PRAWireRequest,
        recordization: RecordizationResult,
        plan: WireAgentMemoryPlan | None,
    ) -> tuple[PRAWireRequest, dict[str, Any]]:
        history = recordization.history
        history_value = [
            {
                "record_id": row.record_id,
                "turn_id": row.turn_id,
                "causal_group_id": row.causal_group_id,
                "message_index": row.message_index,
                "primary_role": row.primary_role.value,
                "semantic_roles": [role.value for role in row.semantic_roles],
                "resource_ids": list(row.resource_ids),
                "metadata": dict(row.metadata),
            }
            for row in history.records
        ]
        metadata = {
            **request.metadata,
            "agent_records": history_value,
            "agent_records_generated_by_mediator": "agent_records" not in request.metadata,
        }
        if plan is not None:
            metadata["agent_memory_plan"] = plan.to_dict()
            metadata["agent_memory_plan_generated_by_mediator"] = (
                "agent_memory_plan" not in request.metadata
            )
        messages = tuple(dict(row) for row in request.messages)
        tools = request.tools
        selected_ids = tuple(row.record_id for row in history.records)
        replacements = 0

        selection_mode = self.config.history_selection.mode
        compaction_mode = self.config.result_compaction.mode
        disclosure_mode = self.config.tool_disclosure.mode
        if selection_mode == MediationFeatureMode.ACTIVE:
            if plan is None and self.config.history_selection.policy != "full":
                raise RecordInferenceError(
                    "active history selection requires a frozen agent_memory_plan"
                )
            if plan is not None:
                known = history.record_by_id
                if (
                    plan.source_history_digest is not None
                    and plan.source_history_digest != history.digest
                ):
                    raise RecordInferenceError(
                        "agent-memory plan was produced for a different history"
                    )
                unknown = set(plan.selected_record_ids) - set(known)
                if unknown:
                    raise RecordInferenceError(
                        "agent-memory plan selects unknown records: " + ", ".join(sorted(unknown))
                    )
                unknown_replacements = set(plan.record_replacements) - set(known)
                if unknown_replacements:
                    raise RecordInferenceError(
                        "agent-memory plan replaces unknown records: "
                        + ", ".join(sorted(unknown_replacements))
                    )
                unselected_replacements = set(plan.record_replacements) - set(
                    plan.selected_record_ids
                )
                if unselected_replacements:
                    raise RecordInferenceError(
                        "agent-memory replacements must remain selected: "
                        + ", ".join(sorted(unselected_replacements))
                    )
                selected = set(plan.selected_record_ids)
                mandatory = {
                    row.record_id for row in history.records
                    if row.primary_role.value in {"system", "task"}
                }
                if history.records:
                    mandatory.add(history.records[-1].record_id)
                if not mandatory <= selected:
                    raise RecordInferenceError("agent-memory plan removes mandatory task/current state")
                selected_ids = tuple(
                    row.record_id for row in history.records if row.record_id in selected
                )
                selected_messages = []
                for record in history.records:
                    if record.record_id not in selected:
                        continue
                    message = dict(request.messages[record.message_index])
                    if record.record_id in plan.record_replacements:
                        message["content"] = plan.record_replacements[record.record_id]
                        replacements += 1
                    selected_messages.append(message)
                self._validate_selected_messages(selected_messages)
                messages = tuple(selected_messages)
        elif selection_mode == MediationFeatureMode.SHADOW and plan is not None:
            selected_ids = plan.selected_record_ids

        if compaction_mode == MediationFeatureMode.ACTIVE:
            if plan is None:
                raise RecordInferenceError("active result compaction requires a frozen agent_memory_plan")
            # Replacements are applied above when selection is active.  A
            # compaction-only plan retains all records and replaces observation
            # bodies without changing causal membership.
            if selection_mode != MediationFeatureMode.ACTIVE:
                known = history.record_by_id
                if (
                    plan.source_history_digest is not None
                    and plan.source_history_digest != history.digest
                ):
                    raise RecordInferenceError(
                        "agent-memory plan was produced for a different history"
                    )
                unknown_replacements = set(plan.record_replacements) - set(known)
                if unknown_replacements:
                    raise RecordInferenceError(
                        "agent-memory plan replaces unknown records: "
                        + ", ".join(sorted(unknown_replacements))
                    )
                values = []
                for record in history.records:
                    message = dict(request.messages[record.message_index])
                    if record.record_id in plan.record_replacements:
                        if record.record_id not in known:
                            raise RecordInferenceError("replacement targets an unknown record")
                        message["content"] = plan.record_replacements[record.record_id]
                        replacements += 1
                    values.append(message)
                messages = tuple(values)

        if disclosure_mode == MediationFeatureMode.ACTIVE:
            if plan is None:
                raise RecordInferenceError("active tool disclosure requires a frozen agent_memory_plan")
            allowed = set(plan.selected_tool_names)
            if allowed:
                tools = tuple(tool for tool in request.tools if self._tool_name(tool) in allowed)

        return replace(request, messages=messages, tools=tools, metadata=metadata), {
            "agent_record_count": len(history.records),
            "selected_agent_record_ids": list(selected_ids),
            "model_visible_message_changed": messages != request.messages,
            "model_visible_tool_set_changed": tools != request.tools,
            "record_replacement_count": replacements,
            "agent_memory_plan_digest": plan.digest if plan is not None else "full",
        }

    @staticmethod
    def _tool_name(tool: Mapping[str, Any]) -> str:
        function = tool.get("function")
        if isinstance(function, Mapping):
            return str(function.get("name", ""))
        return str(tool.get("name", tool.get("type", "")))

    @staticmethod
    def _validate_selected_messages(messages: list[Mapping[str, Any]]) -> None:
        if not messages:
            raise RecordInferenceError("agent-memory plan produced an empty request")
        pending: set[str] = set()
        previous_role = None
        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            if role == previous_role == "assistant":
                raise RecordInferenceError("selection produced adjacent assistant messages")
            if role == "assistant":
                calls = message.get("tool_calls")
                if isinstance(calls, (list, tuple)):
                    pending.update(
                        str(row["id"]) for row in calls
                        if isinstance(row, Mapping) and row.get("id")
                    )
            elif role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                message_metadata = message.get("metadata")
                declaration = (
                    message_metadata.get("pra_record")
                    if isinstance(message_metadata, Mapping) else None
                )
                explicitly_paired = bool(
                    isinstance(declaration, Mapping)
                    and declaration.get("causal_group_id")
                    and declaration.get("primary_role") == "tool_observation"
                )
                if call_id not in pending and not explicitly_paired:
                    raise RecordInferenceError(
                        f"selected tool result {index} has no retained action"
                    )
                if call_id in pending:
                    pending.remove(call_id)
            previous_role = role


__all__ = [
    "CapabilityDowngradePolicy",
    "AgentMemoryPlanBuilder",
    "DoubleMediationPolicy",
    "HistorySelectionConfig",
    "InferenceFailurePolicy",
    "MediationConflictError",
    "MediationFeatureMode",
    "MediationIdempotencyConfig",
    "MediationLocation",
    "MediationStamp",
    "PRAMediationConfig",
    "PreparedMediation",
    "RecordInferenceConfig",
    "RecordInferenceError",
    "RequestMediator",
    "ResultCompactionConfig",
    "ToolDisclosureConfig",
    "WireAgentMemoryPlan",
    "resolve_auto_gateway_mode",
]
