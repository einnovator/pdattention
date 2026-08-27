"""Create typed, semantically enriched tool records from ordinary callables."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import re
import types
import typing
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Iterable, Mapping, get_args, get_origin, get_type_hints

from pra_hf.agent_resources import AgentResource, SideEffectClass, normalize_text, resource_uri, terms
from pra_hf.semantic_resource_discovery import CanonicalConceptMap


_EMPTY = inspect.Signature.empty
_GENERIC_TERMS = frozenset({
    "a", "an", "and", "any", "as", "at", "be", "by", "data", "for", "from",
    "get", "in", "into", "is", "it", "of", "object", "on", "or", "record", "return",
    "returns", "that", "the", "this", "to", "tool", "use", "value", "with",
})
_OPERATION_CANONICAL = {
    "archive": "archive", "create": "create", "delete": "delete", "export": "export",
    "extract": "inspect", "get": "get", "inspect": "inspect", "notify": "notify",
    "read": "read", "search": "search",
    "update": "update", "validate": "validate",
    "reactivate": "update", "activate": "update", "modify": "update", "change": "update",
    "set": "update", "retrieve": "get", "fetch": "get", "read": "read", "find": "search",
    "locate": "search", "lookup": "search", "notify": "notify", "inform": "notify",
    "remove": "delete", "purge": "delete", "validate": "validate", "check": "validate",
}


@dataclass(frozen=True)
class ParameterSchema:
    """Provider-independent description of one callable parameter."""

    name: str
    type_name: str
    required: bool
    description: str = ""
    kind: str = "POSITIONAL_OR_KEYWORD"
    has_default: bool = False
    default: object = None
    json_schema: Mapping[str, object] = field(default_factory=dict)
    compatible_type_id: str = "unknown"


@dataclass(frozen=True)
class ReturnSchema:
    """Provider-independent callable result schema."""

    type_name: str = "unknown"
    description: str = ""
    json_schema: Mapping[str, object] = field(default_factory=dict)
    compatible_type_id: str = "unknown"


@dataclass(frozen=True)
class ToolSchema:
    """Canonical callable schema with provider export adapters."""

    inputs: tuple[ParameterSchema, ...] = ()
    output: ReturnSchema = field(default_factory=ReturnSchema)

    def parameters_json_schema(self) -> dict[str, object]:
        properties = {
            row.name: {
                **dict(row.json_schema),
                **({"description": row.description} if row.description else {}),
            }
            for row in self.inputs
        }
        return {
            "type": "object",
            "properties": properties,
            "required": [row.name for row in self.inputs if row.required],
            "additionalProperties": any(row.kind == "VAR_KEYWORD" for row in self.inputs),
        }

    def to_json_schema(self) -> dict[str, object]:
        return {
            "inputs": self.parameters_json_schema(),
            "output": {
                **dict(self.output.json_schema),
                **({"description": self.output.description} if self.output.description else {}),
            },
        }

    def to_openai_tool(self, name: str, description: str) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": self.parameters_json_schema(),
            },
        }

    def to_anthropic_tool(self, name: str, description: str) -> dict[str, object]:
        return {
            "name": name,
            "description": description,
            "input_schema": self.parameters_json_schema(),
        }


@dataclass(frozen=True)
class TypeSchema:
    """Cached normalized form of one Python type expression."""

    identity: str
    type_name: str
    json_schema: Mapping[str, object]
    compatible_type_id: str
    module_fingerprint: str


def _qualified_type_identity(annotation: object) -> str:
    origin = get_origin(annotation)
    if origin is not None:
        args = ",".join(_qualified_type_identity(value) for value in get_args(annotation))
        return f"{_qualified_type_identity(origin)}[{args}]"
    module = getattr(annotation, "__module__", "typing")
    name = getattr(annotation, "__qualname__", getattr(annotation, "_name", repr(annotation)))
    return f"{module}.{name}"


def _module_fingerprint(annotation: object) -> str:
    module = getattr(get_origin(annotation) or annotation, "__module__", "typing")
    root = module.split(".", 1)[0]
    try:
        version = importlib_metadata.version(root)
    except importlib_metadata.PackageNotFoundError:
        version = "stdlib-or-local"
    return f"{module}@{version}"


class PythonTypeSchemaCache:
    """Stable schema cache for standard, structured, and optional Python types."""

    def __init__(self) -> None:
        self._cache: dict[str, TypeSchema] = {}

    def resolve(self, annotation: object) -> TypeSchema:
        if annotation is _EMPTY or annotation is Any:
            annotation = Any
        identity = _qualified_type_identity(annotation)
        if identity not in self._cache:
            schema, type_name, compatible = self._build(annotation)
            self._cache[identity] = TypeSchema(
                identity=identity,
                type_name=type_name,
                json_schema=schema,
                compatible_type_id=compatible,
                module_fingerprint=_module_fingerprint(annotation),
            )
        return self._cache[identity]

    def _build(self, annotation: object) -> tuple[dict[str, object], str, str]:
        if annotation is Any or annotation is _EMPTY:
            return {}, "unknown", "unknown"
        if annotation in {None, type(None)}:
            return {"type": "null"}, "None", "none"
        primitives = {
            str: ("string", "str"), int: ("integer", "int"), float: ("number", "float"),
            bool: ("boolean", "bool"),
        }
        if annotation in primitives:
            json_type, name = primitives[annotation]
            return {"type": json_type}, name, name
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is typing.Annotated:
            return self._build(args[0])
        if origin is typing.Literal:
            values = list(args)
            value_types = {type(value) for value in values}
            schema = {"enum": values}
            if len(value_types) == 1 and next(iter(value_types)) in primitives:
                schema["type"] = primitives[next(iter(value_types))][0]
            return schema, f"Literal[{', '.join(map(repr, values))}]", "literal:" + "|".join(map(str, values))
        if origin in {typing.Union, types.UnionType}:
            members = [self.resolve(value) for value in args]
            non_null = [row for row in members if row.compatible_type_id != "none"]
            optional = len(non_null) != len(members)
            name = " | ".join(row.type_name for row in non_null) + (" | None" if optional else "")
            schema = {"anyOf": [dict(row.json_schema) for row in members]}
            compatible = "union:" + "|".join(sorted(row.compatible_type_id for row in members))
            return schema, name, compatible
        if origin in {list, set, frozenset}:
            item = self.resolve(args[0] if args else Any)
            return {"type": "array", "items": dict(item.json_schema)}, f"{origin.__name__}[{item.type_name}]", f"list:{item.compatible_type_id}"
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                item = self.resolve(args[0])
                return {"type": "array", "items": dict(item.json_schema)}, f"tuple[{item.type_name}, ...]", f"tuple:{item.compatible_type_id}"
            members = [self.resolve(value) for value in args]
            return {
                "type": "array", "prefixItems": [dict(row.json_schema) for row in members],
                "minItems": len(members), "maxItems": len(members),
            }, f"tuple[{', '.join(row.type_name for row in members)}]", "tuple:" + "|".join(row.compatible_type_id for row in members)
        if origin is dict:
            key = self.resolve(args[0] if args else str)
            value = self.resolve(args[1] if len(args) > 1 else Any)
            return {"type": "object", "additionalProperties": dict(value.json_schema)}, f"dict[{key.type_name}, {value.type_name}]", f"dict:{key.compatible_type_id}:{value.compatible_type_id}"
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            values = [row.value for row in annotation]
            return {"enum": values}, annotation.__name__, _qualified_type_identity(annotation)
        if dataclasses.is_dataclass(annotation):
            hints = get_type_hints(annotation)
            properties = {}
            required = []
            for row in dataclasses.fields(annotation):
                child = self.resolve(hints.get(row.name, row.type))
                properties[row.name] = dict(child.json_schema)
                if row.default is dataclasses.MISSING and row.default_factory is dataclasses.MISSING:
                    required.append(row.name)
            return {"type": "object", "properties": properties, "required": required}, annotation.__name__, _qualified_type_identity(annotation)
        if typing.is_typeddict(annotation):
            hints = get_type_hints(annotation)
            required_keys = set(getattr(annotation, "__required_keys__", ()))
            properties = {name: dict(self.resolve(value).json_schema) for name, value in hints.items()}
            return {"type": "object", "properties": properties, "required": sorted(required_keys)}, annotation.__name__, _qualified_type_identity(annotation)
        if inspect.isclass(annotation) and hasattr(annotation, "model_json_schema"):
            return dict(annotation.model_json_schema()), annotation.__name__, _qualified_type_identity(annotation)
        if inspect.isclass(annotation) and hasattr(annotation, "schema") and hasattr(annotation, "__fields__"):
            return dict(annotation.schema()), annotation.__name__, _qualified_type_identity(annotation)
        annotations = getattr(annotation, "__annotations__", {})
        if inspect.isclass(annotation) and annotations:
            properties = {name: dict(self.resolve(value).json_schema) for name, value in get_type_hints(annotation).items()}
            return {"type": "object", "properties": properties, "required": sorted(properties)}, annotation.__name__, _qualified_type_identity(annotation)
        name = getattr(annotation, "__name__", normalize_text(str(annotation)) or "unknown")
        return {}, name, _qualified_type_identity(annotation)

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "entries": [asdict(self._cache[key]) for key in sorted(self._cache)],
        }


@dataclass(frozen=True)
class ParsedDocstring:
    """Descriptions recovered without requiring one docstring convention."""

    summary: str = ""
    parameters: Mapping[str, str] = field(default_factory=dict)
    returns: str = ""


def parse_docstring(docstring: str | None) -> ParsedDocstring:
    """Parse common Google, NumPy, and Sphinx parameter/return descriptions."""

    if not docstring:
        return ParsedDocstring()
    lines = inspect.cleandoc(docstring).splitlines()
    summary = next((line.strip() for line in lines if line.strip()), "")
    parameters: dict[str, str] = {}
    returns = ""
    section = ""
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        lower = stripped.casefold()
        sphinx = re.match(r":param\s+([^:]+):\s*(.*)", stripped)
        if sphinx:
            parameters[sphinx.group(1).split()[-1]] = sphinx.group(2).strip()
        elif lower.startswith(":return:") or lower.startswith(":returns:"):
            returns = stripped.split(":", 2)[-1].strip()
        elif lower in {"args:", "arguments:", "parameters:"}:
            section = "parameters"
        elif lower in {"returns:", "return:"}:
            section = "returns"
        elif index + 1 < len(lines) and set(lines[index + 1].strip()) == {"-"}:
            if lower == "parameters":
                section = "numpy_parameters"
            elif lower in {"returns", "return"}:
                section = "numpy_returns"
            index += 1
        elif section == "parameters":
            match = re.match(r"([*\w][\w*]*)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if match:
                parameters[match.group(1).lstrip("*")] = match.group(2).strip()
        elif section == "numpy_parameters" and stripped and not lines[index].startswith(" "):
            name = stripped.split(":", 1)[0].strip().lstrip("*")
            description = lines[index + 1].strip() if index + 1 < len(lines) else ""
            parameters[name] = description
        elif section in {"returns", "numpy_returns"} and stripped:
            if section == "numpy_returns" and index + 1 < len(lines) and lines[index + 1].startswith(" "):
                returns = lines[index + 1].strip()
            elif ":" not in stripped:
                returns = stripped
            section = ""
        index += 1
    return ParsedDocstring(summary=summary, parameters=parameters, returns=returns)


@dataclass(frozen=True)
class KeywordEvidence:
    """One automatic semantic term with retained origin and weight."""

    term: str
    source: str
    weight: float
    surface: str


@dataclass(frozen=True)
class ToolRecord:
    """Provider-independent record extracted from one callable without execution."""

    name: str
    qualified_name: str
    module: str
    description: str
    signature: str
    schema: ToolSchema
    is_async: bool
    accepts_varargs: bool
    accepts_varkw: bool
    namespace: str
    tenant_id: str
    version: str
    aliases: tuple[str, ...] = ()
    manual_tags: frozenset[str] = frozenset()
    auto_tags: frozenset[str] = frozenset()
    keywords: frozenset[str] = frozenset()
    keyword_evidence: tuple[KeywordEvidence, ...] = ()
    operation_concepts: frozenset[str] = frozenset()
    object_concepts: frozenset[str] = frozenset()
    field_provenance: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_agent_resource(self) -> AgentResource:
        content = {
            "name": self.name,
            "parameters": self.schema.parameters_json_schema(),
            "returns": {
                **dict(self.schema.output.json_schema),
                **({"description": self.schema.output.description} if self.schema.output.description else {}),
            },
        }
        consumes = tuple(sorted({row.compatible_type_id for row in self.schema.inputs if row.compatible_type_id != "unknown"}))
        produces = () if self.schema.output.compatible_type_id in {"unknown", "none"} else (self.schema.output.compatible_type_id,)
        operation = next(iter(sorted(self.operation_concepts)), None)
        declared_side_effect = self.metadata.get("side_effect_class")
        inferred_side_effect = (
            SideEffectClass(declared_side_effect)
            if declared_side_effect is not None
            else SideEffectClass.DESTRUCTIVE
            if operation == "delete"
            else SideEffectClass.NONE
        )
        return AgentResource(
            uri=resource_uri("tool", self.namespace, self.qualified_name, self.version),
            kind="tool",
            namespace=self.namespace,
            name=self.name,
            version=self.version,
            description=self.description,
            content=json.dumps(content, sort_keys=True),
            aliases=self.aliases,
            # This conservative hint controls disclosure only. The host still
            # owns authorization and must not trust inferred metadata to execute.
            side_effect_class=inferred_side_effect,
            tenant_id=self.tenant_id,
            metadata={
                **dict(self.metadata),
                "qualified_name": self.qualified_name,
                "module": self.module,
                "signature": self.signature,
                "is_async": self.is_async,
                "tags": tuple(sorted(self.manual_tags)),
                "auto_tags": tuple(sorted(self.auto_tags)),
                "keywords": tuple(sorted(self.keywords)),
                "keyword_evidence": tuple(asdict(row) for row in self.keyword_evidence),
                "operation_kind": operation,
                "object_types": tuple(sorted(self.object_concepts)),
                "auto_field_provenance": dict(self.field_provenance),
                "consumes": consumes,
                "produces": produces,
                "auto_enrichment": True,
                "side_effect_provenance": (
                    "host_declared" if declared_side_effect is not None
                    else "auto_name_conservative"
                ),
            },
        )


def _informative(text: str) -> tuple[str, ...]:
    return tuple(token for token in terms(text) if len(token) > 1 and token not in _GENERIC_TERMS)


def _keyword_evidence(
    name: str,
    parsed: ParsedDocstring,
    schema: ToolSchema,
    concept_map: CanonicalConceptMap | None,
) -> tuple[KeywordEvidence, ...]:
    evidence: dict[tuple[str, str], KeywordEvidence] = {}

    def add(values: Iterable[str], source: str, weight: float) -> None:
        for surface in values:
            for token in _informative(surface):
                key = (token, source)
                evidence[key] = KeywordEvidence(token, source, weight, surface)

    add((name,), "function_name", 1.0)
    add((row.name for row in schema.inputs), "parameter_name", 0.82)
    add((row.type_name for row in schema.inputs), "parameter_type", 0.78)
    add((schema.output.type_name,), "return_type", 0.80)
    add((parsed.summary, *parsed.parameters.values(), parsed.returns), "docstring", 0.55)
    for token in terms(name):
        canonical = _OPERATION_CANONICAL.get(token)
        if canonical:
            key = (canonical, "operation_canonical")
            weight = 0.45 if canonical in {"get", "read"} else 0.92
            evidence[key] = KeywordEvidence(canonical, "operation_canonical", weight, token)
    if concept_map is not None:
        source = " ".join((name, parsed.summary, *parsed.parameters.values(), parsed.returns))
        for match in concept_map.match(source, language="en"):
            add((match.canonical,), f"dictionary:{match.source}", 0.62)
    return tuple(sorted(evidence.values(), key=lambda row: (-row.weight, row.term, row.source)))


def tool_record_from_callable(
    function: Callable[..., object],
    *,
    namespace: str | None = None,
    tenant_id: str = "default",
    version: str = "v1",
    aliases: Iterable[str] = (),
    manual_tags: Iterable[str] = (),
    metadata: Mapping[str, object] | None = None,
    concept_map: CanonicalConceptMap | None = None,
    type_cache: PythonTypeSchemaCache | None = None,
) -> ToolRecord:
    """Inspect an ordinary Python callable and return a typed tool record."""

    if not callable(function):
        raise TypeError("tool_record_from_callable expects a callable.")
    cache = type_cache or PythonTypeSchemaCache()
    signature = inspect.signature(function)
    parsed = parse_docstring(inspect.getdoc(function))
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = {}
    inputs = []
    for name, parameter in signature.parameters.items():
        annotation = hints.get(name, parameter.annotation)
        resolved = cache.resolve(annotation)
        has_default = parameter.default is not _EMPTY
        inputs.append(ParameterSchema(
            name=name,
            type_name=resolved.type_name,
            required=not has_default and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD},
            description=parsed.parameters.get(name, ""),
            kind=parameter.kind.name,
            has_default=has_default,
            default=None if not has_default else parameter.default,
            json_schema=resolved.json_schema,
            compatible_type_id=resolved.compatible_type_id,
        ))
    return_type = cache.resolve(hints.get("return", signature.return_annotation))
    schema = ToolSchema(tuple(inputs), ReturnSchema(
        type_name=return_type.type_name,
        description=parsed.returns,
        json_schema=return_type.json_schema,
        compatible_type_id=return_type.compatible_type_id,
    ))
    name = getattr(function, "__name__", function.__class__.__name__)
    qualified_name = getattr(function, "__qualname__", name)
    module = getattr(function, "__module__", "__main__")
    evidence = _keyword_evidence(name, parsed, schema, concept_map)
    keywords = frozenset(row.term for row in evidence)
    operations = frozenset(
        _OPERATION_CANONICAL[token]
        for token in terms(name)
        if token in _OPERATION_CANONICAL
    )
    name_tokens = _informative(name)
    objects = frozenset(
        token
        for token in name_tokens
        if token not in operations and token not in _OPERATION_CANONICAL
    )
    auto_tags = frozenset((*operations, *objects, *(row.compatible_type_id for row in schema.inputs))) - {"unknown"}
    field_provenance = {
        "keywords": tuple(sorted(f"{row.term}:{row.source}" for row in evidence)),
        "operation_concepts": tuple(sorted(f"{value}:function_name" for value in operations)),
        "object_concepts": tuple(sorted(f"{value}:function_name" for value in objects)),
        "auto_tags": tuple(sorted(f"{value}:derived" for value in auto_tags)),
        "embedding_fields": ("name", "docstring", "parameter_schema", "return_schema", "module_namespace"),
    }
    return ToolRecord(
        name=name,
        qualified_name=qualified_name,
        module=module,
        description=parsed.summary,
        signature=str(signature),
        schema=schema,
        is_async=inspect.iscoroutinefunction(function),
        accepts_varargs=any(row.kind == "VAR_POSITIONAL" for row in inputs),
        accepts_varkw=any(row.kind == "VAR_KEYWORD" for row in inputs),
        namespace=namespace or module,
        tenant_id=tenant_id,
        version=version,
        aliases=tuple(dict.fromkeys(aliases)),
        manual_tags=frozenset(normalize_text(value) for value in manual_tags if value),
        auto_tags=auto_tags,
        keywords=keywords,
        keyword_evidence=evidence,
        operation_concepts=operations,
        object_concepts=objects,
        field_provenance=field_provenance,
        metadata=dict(metadata or {}),
    )
