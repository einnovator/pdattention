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
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from pra_hf.agent_resources import AgentResource, SideEffectClass, resource_uri


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_SERIALIZED_TOOL_DECISION = re.compile(
    r"I executed the tool decision named\s+"
    r"(?P<name>\"(?:\\.|[^\"\\])*\")\s+"
    r"with these arguments:\s*(?P<arguments>\{.*\})\.\s*$",
    re.DOTALL,
)
_QWEN_FUNCTION_DECISION = re.compile(
    r"(?:<tool_call>\s*)?"
    r"<function=(?P<name>[A-Za-z_][A-Za-z0-9_.:-]*)>\s*"
    r"(?P<body>.*?)\s*</function>\s*(?:</tool_call>)?\s*$",
    re.DOTALL,
)
_QWEN_FUNCTION_PARAMETER = re.compile(
    r"<parameter=(?P<name>[A-Za-z_][A-Za-z0-9_.:-]*)>\s*"
    r"(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)
_TERMINAL_SERIALIZED_TOOL_INTENT = re.compile(
    r"I executed the tool decision named\s+\"[^\"\r\n]*\"\s+"
    r"with these arguments:\s*\{.*\}\}?\.\s*$",
    re.DOTALL,
)
_TERMINAL_QWEN_TOOL_INTENT = re.compile(
    r"(?:<tool_call>\s*)?<function=[^>]+>.*?</function>\s*"
    r"(?:</tool_call>)?\s*$",
    re.DOTALL,
)
_TERMINAL_XML_TOOL_INTENT = re.compile(r"<tool_call>.*</tool_call>\s*$", re.DOTALL)


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
    """Request-scoped authority independent of discovery confidence."""

    allowed_uris: frozenset[str]
    allow_writes: bool = False
    allow_destructive: bool = False


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
    """Parse one explicit provider-neutral tool decision.

    Native calls are projected into the canonical ``<tool_call>`` envelope.
    Some OpenAI-compatible model servers occasionally serialize the durable
    assistant-action projection as ordinary text instead of returning a
    ``message.tool_calls`` object.  Accept that representation only when the
    exact PRA-owned sentence terminates the response.  Anchoring the marker at
    end-of-response prevents quoted examples or later prose from silently
    crossing the host's execution boundary.
    """

    match = _TOOL_CALL.search(text)
    if match is not None:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        name = payload.get("name")
        arguments = payload.get("arguments")
        raw_text = match.group(0)
    else:
        serialized = _SERIALIZED_TOOL_DECISION.search(text)
        if serialized is not None:
            try:
                name = json.loads(serialized.group("name"))
                arguments = json.loads(serialized.group("arguments"))
            except json.JSONDecodeError:
                return None
            raw_text = serialized.group(0)
        else:
            function = _QWEN_FUNCTION_DECISION.search(text)
            if function is None:
                return None
            name = function.group("name")
            arguments = {}
            body = function.group("body")
            consumed = []
            for parameter in _QWEN_FUNCTION_PARAMETER.finditer(body):
                argument_name = parameter.group("name")
                if argument_name in arguments:
                    return None
                raw_value = parameter.group("value").strip()
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError:
                    value = raw_value
                arguments[argument_name] = value
                consumed.append(parameter.span())
            cursor = 0
            for start, end in consumed:
                if body[cursor:start].strip():
                    return None
                cursor = end
            if body[cursor:].strip():
                return None
            raw_text = function.group(0)
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments, raw_text=raw_text)


def has_terminal_tool_intent(text: str) -> bool:
    """Return whether a terminal response is an attempted tool decision.

    This deliberately does not repair or execute malformed output.  It lets an
    agent distinguish a malformed explicit action from an ordinary final
    answer so the former can become a visible rejection observation and be
    retried.  Requiring a known terminal marker avoids treating explanatory
    prose or quoted mid-response examples as executable intent.
    """

    return any(
        pattern.search(text) is not None
        for pattern in (
            _TERMINAL_XML_TOOL_INTENT,
            _TERMINAL_SERIALIZED_TOOL_INTENT,
            _TERMINAL_QWEN_TOOL_INTENT,
        )
    )


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
    ) -> ToolExecutionResult:
        """Validate and execute one proposed call against the disclosed set."""

        if call is None:
            return ToolExecutionResult(False, False, "malformed_call", None, None)
        resource = self.by_name.get(call.name)
        if resource is None:
            return ToolExecutionResult(False, False, "unknown_tool", None, call)
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
        try:
            output = dict(handler(call.arguments, tuple(prior_observations)))
            reason = "executed"
        except Exception as error:
            # Tool failures are observations the model can recover from, not
            # agent-process failures.  The call passed authorization and was
            # invoked, so preserve it as executed while recording semantic
            # failure separately from transport completion.
            output = {
                "error_type": type(error).__name__,
                "error": str(error),
                "tool": call.name,
                "arguments": dict(call.arguments),
            }
            reason = "execution_error"
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
            True, True, reason, resource.uri, call, observation, output
        )
