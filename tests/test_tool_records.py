"""Tests for automatic callable ingestion and normalized Python schemas."""

from __future__ import annotations

import dataclasses
import enum
from typing import Literal, Optional, TypedDict, Union

import pytest

from data.python_tool_ingestion_cases import PAPER6_5_TOOL_CALLABLES, delete_user, get_user, update_user
from pra_hf.tool_records import (
    PythonTypeSchemaCache,
    parse_docstring,
    tool_record_from_callable,
)


class Mode(enum.Enum):
    FAST = "fast"
    SAFE = "safe"


@dataclasses.dataclass
class Payload:
    name: str
    count: int = 1


class Result(TypedDict):
    ok: bool
    value: str


def test_type_cache_supports_standard_composites_and_stable_manifest() -> None:
    cache = PythonTypeSchemaCache()
    annotations = (
        str,
        list[int],
        tuple[str, ...],
        dict[str, float],
        Optional[str],
        Union[int, str],
        Literal["a", "b"],
        Mode,
        Payload,
        Result,
    )

    first = [cache.resolve(annotation) for annotation in annotations]
    second = [cache.resolve(annotation) for annotation in annotations]

    assert first == second
    assert cache.resolve(Optional[str]).json_schema["anyOf"]
    assert cache.resolve(Literal["a", "b"]).json_schema["enum"] == ["a", "b"]
    assert cache.resolve(Mode).json_schema["enum"] == ["fast", "safe"]
    assert cache.resolve(Payload).json_schema["required"] == ["name"]
    assert cache.resolve(Result).json_schema["required"] == ["ok", "value"]
    assert len(cache.to_manifest()["entries"]) >= len(annotations)


def test_pydantic_adapter_is_optional() -> None:
    pydantic = pytest.importorskip("pydantic")

    class Model(pydantic.BaseModel):
        name: str

    schema = PythonTypeSchemaCache().resolve(Model)
    assert schema.json_schema["properties"]["name"]["type"] == "string"


def test_google_numpy_and_sphinx_docstrings_are_parsed() -> None:
    google = parse_docstring("""Summary.\n\nArgs:\n    item: Item description.\n\nReturns:\n    Result text.""")
    numpy = parse_docstring("""Summary.\n\nParameters\n----------\nitem : str\n    Item description.\n\nReturns\n-------\nstr\n    Result text.""")
    sphinx = parse_docstring("""Summary.\n\n:param item: Item description.\n:return: Result text.""")

    assert google.parameters["item"] == "Item description."
    assert numpy.parameters["item"] == "Item description."
    assert numpy.returns == "Result text."
    assert sphinx.parameters["item"] == "Item description."
    assert sphinx.returns == "Result text."


def test_callable_ingestion_preserves_signature_schema_and_metadata_separation() -> None:
    record = tool_record_from_callable(update_user, namespace="paper6_5", manual_tags=("curated",))
    resource = record.to_agent_resource()

    assert record.name == "update_user"
    assert record.description == "Change the status associated with a user account."
    assert [row.name for row in record.schema.inputs] == ["user_id", "status"]
    assert all(row.required for row in record.schema.inputs)
    assert record.operation_concepts == frozenset({"update"})
    assert "user" in record.object_concepts
    assert resource.tags == frozenset({"curated"})
    assert "user" in resource.auto_tags
    assert resource.metadata["auto_enrichment"] is True


def test_generic_retrieval_verbs_still_define_operations_and_destructive_names_are_suppressed() -> None:
    get_record = tool_record_from_callable(get_user)
    delete_resource = tool_record_from_callable(delete_user).to_agent_resource()

    assert get_record.operation_concepts == frozenset({"get"})
    assert get_record.object_concepts == frozenset({"user"})
    get_evidence = next(row for row in get_record.keyword_evidence if row.term == "get")
    assert get_evidence.weight < 0.5
    assert delete_resource.side_effect_class.value == "destructive"
    assert delete_resource.metadata["side_effect_provenance"] == "auto_name_conservative"


def test_zero_metadata_fixture_contains_eighteen_inspectable_callables() -> None:
    records = [tool_record_from_callable(row, namespace="paper6_5_auto") for row in PAPER6_5_TOOL_CALLABLES]

    assert len(records) == 18
    assert len({row.name for row in records}) == 18
    assert all(row.signature and row.module and row.qualified_name for row in records)


def test_unknown_descriptions_are_not_invented_and_variadics_are_retained() -> None:
    def plain(value, *items: int, **options: str):
        return value

    record = tool_record_from_callable(plain)

    assert record.description == ""
    assert record.accepts_varargs
    assert record.accepts_varkw
    assert record.schema.inputs[0].type_name == "unknown"
