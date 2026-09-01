"""Safe, typed execution primitives for Paper 6.5 agent experiments.

Discovery identifies a resource and model generation proposes a call.  This
module owns the boundary after both: it parses the proposal, checks the
selected identity and argument schema, applies host authorization, executes
only registered in-memory handlers, and preserves the result as a typed
observation resource.  It never invokes an external service.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from pra_hf.agent_resources import AgentResource, SideEffectClass, resource_uri


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    """A model-proposed call before host authorization.

    ``name`` is the prompt-visible function name. ``arguments`` is the parsed
    JSON object and remains untrusted until :class:`SafeToolExecutor` validates
    it against the selected resource schema.
    """

    name: str
    arguments: Mapping[str, object]
    raw_text: str = ""


@dataclass(frozen=True)
class ExecutionAuthorization:
    """Request-scoped authority independent of discovery confidence.

    ``tenant_id`` and ``session_id`` optionally bind a grant to the caller's
    execution scope.  ``expires_at_unix`` bounds replay in long-running agent
    sessions.  They are optional so local fixtures and older SDK callers keep
    their URI-only behavior while production gateways can enforce all three.
    """

    allowed_uris: frozenset[str]
    allow_writes: bool = False
    allow_destructive: bool = False
    tenant_id: str | None = None
    session_id: str | None = None
    expires_at_unix: float | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    """Auditable result of validating and optionally executing one call."""

    accepted: bool
    executed: bool
    reason: str
    resource_uri: str | None
    call: ToolCall | None
    observation: AgentResource | None = None
    output: Mapping[str, object] = field(default_factory=dict)


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse the first Qwen/OpenAI-style ``<tool_call>`` JSON payload."""

    match = _TOOL_CALL.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments, raw_text=match.group(0))


def resource_tool_schema(resource: AgentResource) -> dict[str, object]:
    """Convert one typed tool resource into an OpenAI-compatible function schema."""

    try:
        source = json.loads(resource.content) if resource.content else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool resource has invalid JSON content: {resource.uri}") from exc
    parameters = source.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        raise ValueError(f"Tool parameters must be an object: {resource.uri}")
    if "type" not in parameters:
        parameters = {"type": "object", **parameters}
    return {
        "type": "function",
        "function": {
            "name": resource.name,
            "description": resource.description,
            "parameters": parameters,
        },
    }


def _validate_arguments(resource: AgentResource, arguments: Mapping[str, object]) -> str | None:
    schema = resource_tool_schema(resource)["function"]["parameters"]
    properties = schema.get("properties", {})
    required = schema.get("required", ())
    if not isinstance(properties, dict) or not isinstance(required, (list, tuple)):
        return "invalid_resource_schema"
    missing = [name for name in required if name not in arguments]
    if missing:
        return "missing_required_argument"
    if any(name not in properties for name in arguments):
        return "unknown_argument"
    for name, value in arguments.items():
        expected = properties[name].get("type") if isinstance(properties[name], dict) else None
        if expected == "string" and not isinstance(value, str):
            return "argument_type_mismatch"
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return "argument_type_mismatch"
        if expected == "boolean" and not isinstance(value, bool):
            return "argument_type_mismatch"
    return None


class SafeToolExecutor:
    """Execute registered pure handlers after identity, schema, and risk checks.

    Handlers receive validated arguments plus immutable prior observations and
    return a JSON-compatible mapping.  The executor cannot authorize itself:
    callers must provide an :class:`ExecutionAuthorization` for every call.
    """

    def __init__(
        self,
        resources: Sequence[AgentResource],
        handlers: Mapping[str, Callable[[Mapping[str, object], Sequence[AgentResource]], Mapping[str, object]]],
    ) -> None:
        self.by_uri = {resource.uri: resource for resource in resources}
        self.by_name = {resource.name: resource for resource in resources}
        self.handlers = dict(handlers)
        if len(self.by_name) != len(tuple(resources)):
            raise ValueError("Tool names must be unique inside an execution registry.")

    def execute(
        self,
        call: ToolCall | None,
        *,
        selected_uris: Sequence[str],
        authorization: ExecutionAuthorization,
        prior_observations: Sequence[AgentResource] = (),
        call_id: str,
        tenant_id: str | None = None,
        session_id: str | None = None,
        now_unix: float | None = None,
    ) -> ToolExecutionResult:
        """Validate and execute one proposed call against the disclosed set."""

        if call is None:
            return ToolExecutionResult(False, False, "malformed_call", None, None)
        resource = self.by_name.get(call.name)
        if resource is None:
            return ToolExecutionResult(False, False, "unknown_tool", None, call)
        if authorization.tenant_id is not None:
            if tenant_id != authorization.tenant_id:
                return ToolExecutionResult(
                    False, False, "tenant_not_authorized", resource.uri, call
                )
            if resource.tenant_id != authorization.tenant_id:
                return ToolExecutionResult(
                    False, False, "resource_tenant_mismatch", resource.uri, call
                )
        if (
            authorization.session_id is not None
            and session_id != authorization.session_id
        ):
            return ToolExecutionResult(
                False, False, "session_not_authorized", resource.uri, call
            )
        if authorization.expires_at_unix is not None:
            current_time = time.time() if now_unix is None else now_unix
            if current_time >= authorization.expires_at_unix:
                return ToolExecutionResult(
                    False, False, "authorization_expired", resource.uri, call
                )
        if resource.uri not in selected_uris:
            return ToolExecutionResult(False, False, "tool_not_disclosed", resource.uri, call)
        if resource.uri not in authorization.allowed_uris:
            return ToolExecutionResult(False, False, "tool_not_authorized", resource.uri, call)
        if resource.revoked:
            return ToolExecutionResult(False, False, "tool_revoked", resource.uri, call)
        if resource.side_effect_class == SideEffectClass.WRITE and not authorization.allow_writes:
            return ToolExecutionResult(False, False, "write_not_authorized", resource.uri, call)
        if (
            resource.side_effect_class == SideEffectClass.DESTRUCTIVE
            and not authorization.allow_destructive
        ):
            return ToolExecutionResult(False, False, "destructive_not_authorized", resource.uri, call)
        invalid = _validate_arguments(resource, call.arguments)
        if invalid is not None:
            return ToolExecutionResult(False, False, invalid, resource.uri, call)
        handler = self.handlers.get(resource.uri)
        if handler is None:
            return ToolExecutionResult(False, False, "handler_not_registered", resource.uri, call)
        output = dict(handler(call.arguments, tuple(prior_observations)))
        observation = AgentResource(
            uri=resource_uri("observation", resource.namespace, call_id, "v1"),
            kind="observation",
            namespace=resource.namespace,
            name=call_id,
            version="v1",
            description=f"Typed result produced by {resource.name}",
            content=json.dumps(output, sort_keys=True, separators=(",", ":")),
            side_effect_class=SideEffectClass.NONE,
            tenant_id=resource.tenant_id,
            metadata={
                "producer_tool_uri": resource.uri,
                "call_id": call_id,
                "schema": resource.metadata.get("returns", {}),
                "provenance": "paper6_5_safe_executor",
            },
        )
        return ToolExecutionResult(
            True, True, "executed", resource.uri, call, observation, output
        )
