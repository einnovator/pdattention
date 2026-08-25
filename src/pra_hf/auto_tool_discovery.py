"""Auditable zero-configuration semantic views for callable-derived tools.

The runtime creates one :class:`ToolRecord` and derives every policy view from
that immutable record. Policies may hide evidence sources for an ablation, but
they never regenerate or relabel the underlying callable metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .agent_resources import terms
from .semantic_resource_discovery import CanonicalConceptMap
from .tool_records import ToolRecord


class AutoEvidenceSource(str, Enum):
    """Callable fields that may contribute to automatic discovery."""

    FUNCTION_NAME = "function_name"
    DOCSTRING = "docstring"
    PARAMETER_NAME = "parameter_name"
    PARAMETER_DESCRIPTION = "parameter_description"
    RETURN_DESCRIPTION = "return_description"
    TYPE_SCHEMA = "type_schema"
    MODULE_NAMESPACE = "module_namespace"
    DICTIONARY_EXPANSION = "dictionary_expansion"
    INFERRED_OPERATION = "inferred_operation"
    INFERRED_OBJECT = "inferred_object"
    AUTO_TAG = "auto_tag"
    EMBEDDING_FIELD = "embedding_field"


_GENERIC = frozenset({
    "a", "an", "and", "as", "by", "data", "for", "from", "in", "is", "it",
    "of", "on", "or", "record", "return", "returns", "the", "to", "value", "with",
})

_SOURCE_WEIGHTS = {
    AutoEvidenceSource.FUNCTION_NAME: 1.00,
    AutoEvidenceSource.PARAMETER_NAME: 0.82,
    AutoEvidenceSource.PARAMETER_DESCRIPTION: 0.62,
    AutoEvidenceSource.DOCSTRING: 0.58,
    AutoEvidenceSource.RETURN_DESCRIPTION: 0.62,
    AutoEvidenceSource.TYPE_SCHEMA: 0.76,
    AutoEvidenceSource.MODULE_NAMESPACE: 0.30,
    AutoEvidenceSource.DICTIONARY_EXPANSION: 0.68,
    AutoEvidenceSource.INFERRED_OPERATION: 0.90,
    AutoEvidenceSource.INFERRED_OBJECT: 0.86,
    AutoEvidenceSource.AUTO_TAG: 0.72,
    AutoEvidenceSource.EMBEDDING_FIELD: 0.50,
}


def _informative(value: str) -> tuple[str, ...]:
    return tuple(token for token in terms(value) if len(token) > 1 and token not in _GENERIC)


@dataclass(frozen=True)
class AutoSemanticEvidence:
    """One normalized semantic term and the callable field that produced it."""

    term: str
    source: AutoEvidenceSource
    surface: str
    weight: float


@dataclass(frozen=True)
class AutoToolSemanticView:
    """Frozen, provider-independent semantic projection of one tool record."""

    name: str
    uri_name: str
    raw_text: str
    evidence: tuple[AutoSemanticEvidence, ...]
    operations: frozenset[str]
    objects: frozenset[str]
    auto_tags: frozenset[str]
    embedding_fields: tuple[tuple[str, str], ...]

    def weighted_terms(
        self,
        *,
        sources: Iterable[AutoEvidenceSource] | None = None,
    ) -> dict[str, float]:
        """Return maximum provenance weight per term for selected sources."""

        allowed = None if sources is None else frozenset(AutoEvidenceSource(value) for value in sources)
        output: dict[str, float] = {}
        for row in self.evidence:
            if allowed is not None and row.source not in allowed:
                continue
            output[row.term] = max(output.get(row.term, 0.0), row.weight)
        return output

    def embedding_text(self, representation: str) -> str | tuple[str, ...]:
        """Return one of the frozen automatic embedding representations."""

        fields = dict(self.embedding_fields)
        if representation == "description":
            return fields["description"]
        if representation == "name_description":
            return f"{fields['name']}. {fields['description']}"
        if representation == "structured_card":
            return "\n".join(
                f"{label.replace('_', ' ').title()}: {fields[label]}"
                for label in ("description", "operation", "object", "inputs", "outputs", "module")
            )
        if representation == "multi_vector":
            return (
                f"{fields['name']}. {fields['description']}",
                f"operation {fields['operation']}; object {fields['object']}",
                f"inputs {fields['inputs']}; outputs {fields['outputs']}; module {fields['module']}",
            )
        raise ValueError(f"Unknown automatic embedding representation: {representation}")


def automatic_semantic_view(
    record: ToolRecord,
    *,
    concepts: CanonicalConceptMap | None = None,
) -> AutoToolSemanticView:
    """Derive one immutable, provenance-bearing semantic view from a record."""

    rows: dict[tuple[str, AutoEvidenceSource], AutoSemanticEvidence] = {}

    def add(values: Iterable[str], source: AutoEvidenceSource) -> None:
        for surface in values:
            for token in _informative(surface):
                key = (token, source)
                rows[key] = AutoSemanticEvidence(token, source, surface, _SOURCE_WEIGHTS[source])

    add((record.name,), AutoEvidenceSource.FUNCTION_NAME)
    add((record.description,), AutoEvidenceSource.DOCSTRING)
    add((row.name for row in record.schema.inputs), AutoEvidenceSource.PARAMETER_NAME)
    add((row.description for row in record.schema.inputs), AutoEvidenceSource.PARAMETER_DESCRIPTION)
    add((record.schema.output.description,), AutoEvidenceSource.RETURN_DESCRIPTION)
    add(
        (*[row.type_name for row in record.schema.inputs], record.schema.output.type_name),
        AutoEvidenceSource.TYPE_SCHEMA,
    )
    add((record.module, record.namespace), AutoEvidenceSource.MODULE_NAMESPACE)

    callable_text = " ".join(
        (
            record.name,
            record.description,
            *(row.name for row in record.schema.inputs),
            *(row.description for row in record.schema.inputs),
            *(row.type_name for row in record.schema.inputs),
            record.schema.output.description,
            record.schema.output.type_name,
            record.module,
            record.namespace,
        )
    )
    operations = set(record.operation_concepts)
    objects = set(record.object_concepts)
    if concepts is not None:
        for match in concepts.match(callable_text, language="en"):
            add((match.canonical,), AutoEvidenceSource.DICTIONARY_EXPANSION)
            (operations if match.kind == "operation" else objects).add(match.canonical)
    add(operations, AutoEvidenceSource.INFERRED_OPERATION)
    add(objects, AutoEvidenceSource.INFERRED_OBJECT)

    type_tags = {
        row.compatible_type_id for row in record.schema.inputs
        if row.compatible_type_id not in {"unknown", "none"}
    }
    if record.schema.output.compatible_type_id not in {"unknown", "none"}:
        type_tags.add(record.schema.output.compatible_type_id)
    auto_tags = frozenset({*record.auto_tags, *operations, *objects, *type_tags})
    add(auto_tags, AutoEvidenceSource.AUTO_TAG)

    raw_text = " ".join(
        value for value in (
            record.name.replace("_", " "), record.description,
            " ".join(row.name for row in record.schema.inputs),
            " ".join(row.description for row in record.schema.inputs),
            record.schema.output.description,
            " ".join(row.type_name for row in record.schema.inputs),
            record.schema.output.type_name,
        ) if value
    )
    embedding_fields = (
        ("name", record.name.replace("_", " ")),
        ("description", record.description),
        ("operation", " ".join(sorted(operations))),
        ("object", " ".join(sorted(objects))),
        ("inputs", " ".join(f"{row.name} {row.type_name}" for row in record.schema.inputs)),
        ("outputs", f"{record.schema.output.type_name} {record.schema.output.description}"),
        ("module", f"{record.module} {record.namespace}"),
    )
    for _, value in embedding_fields:
        add((value,), AutoEvidenceSource.EMBEDDING_FIELD)
    return AutoToolSemanticView(
        name=record.name,
        uri_name=record.qualified_name,
        raw_text=raw_text,
        evidence=tuple(sorted(rows.values(), key=lambda row: (row.source.value, row.term, row.surface))),
        operations=frozenset(operations),
        objects=frozenset(objects),
        auto_tags=auto_tags,
        embedding_fields=embedding_fields,
    )


def weighted_keyword_score(
    query: str,
    view: AutoToolSemanticView,
    *,
    sources: Iterable[AutoEvidenceSource],
    concepts: CanonicalConceptMap | None = None,
    language: str = "en",
    expand_query: bool = False,
) -> float:
    """Score query-term coverage under one explicitly isolated source policy."""

    return weighted_keyword_scores(
        query,
        (view,),
        sources=sources,
        concepts=concepts,
        language=language,
        expand_query=expand_query,
    )[0]


def weighted_keyword_scores(
    query: str,
    views: Iterable[AutoToolSemanticView],
    *,
    sources: Iterable[AutoEvidenceSource],
    concepts: CanonicalConceptMap | None = None,
    language: str = "en",
    expand_query: bool = False,
) -> tuple[float, ...]:
    """Score aligned views while normalizing query concepts only once."""

    query_terms = automatic_query_terms(
        query,
        concepts=concepts,
        language=language,
        expand=expand_query,
    )
    if not query_terms:
        return tuple(0.0 for _ in views)
    output = []
    for view in views:
        values = view.weighted_terms(sources=sources)
        output.append(sum(values.get(term, 0.0) for term in query_terms) / len(query_terms))
    return tuple(output)


def automatic_query_terms(
    query: str,
    *,
    concepts: CanonicalConceptMap | None = None,
    language: str = "en",
    expand: bool = False,
) -> frozenset[str]:
    """Normalize one query with the same rules used by automatic keywords."""

    values = set(_informative(query))
    if expand and concepts is not None:
        values.update(row.canonical for row in concepts.match(query, language=language))
    return frozenset(values)


def inferred_concept_score(
    query: str,
    view: AutoToolSemanticView,
    concepts: CanonicalConceptMap,
    *,
    language: str = "en",
) -> float:
    """Match weighted inferred operation/object candidates without hard labels."""

    return inferred_concept_scores(query, (view,), concepts, language=language)[0]


def inferred_concept_scores(
    query: str,
    views: Iterable[AutoToolSemanticView],
    concepts: CanonicalConceptMap,
    *,
    language: str = "en",
) -> tuple[float, ...]:
    """Score aligned inferred concept views from one query expansion."""

    query_concepts = concepts.concepts(query, language=language)
    output = []
    for view in views:
        operation = max((query_concepts["operation"].get(value, 0.0) for value in view.operations), default=0.0)
        objects = max((query_concepts["object"].get(value, 0.0) for value in view.objects), default=0.0)
        output.append(0.65 * operation + 0.35 * objects)
    return tuple(output)


def auto_tag_score(
    query: str,
    view: AutoToolSemanticView,
    concepts: CanonicalConceptMap,
    *,
    language: str = "en",
) -> float:
    """Score only inferred auto-tags; manual resource tags are never consulted."""

    return auto_tag_scores(query, (view,), concepts, language=language)[0]


def auto_tag_scores(
    query: str,
    views: Iterable[AutoToolSemanticView],
    concepts: CanonicalConceptMap,
    *,
    language: str = "en",
) -> tuple[float, ...]:
    """Score aligned automatic tags from one query concept expansion."""

    values = concepts.concepts(query, language=language)
    query_tags = set(values["operation"]) | set(values["object"])
    if not query_tags:
        return tuple(0.0 for _ in views)
    return tuple(
        len(query_tags & set(view.auto_tags)) / len(query_tags)
        for view in views
    )


def evidence_provenance_counts(views: Iterable[AutoToolSemanticView]) -> dict[str, int]:
    """Count retained evidence rows by source for the experiment manifest."""

    counts = {source.value: 0 for source in AutoEvidenceSource}
    for view in views:
        for row in view.evidence:
            counts[row.source.value] += 1
    return counts
