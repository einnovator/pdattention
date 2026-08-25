"""Integration tests for the Paper 6.5 zero-config tool path."""

from __future__ import annotations

from data.python_tool_ingestion_cases import get_user, search_user, update_user
from experiments.paper6_5_tools.run_union_jit_ollama import _generic_schema_chunk
from pra_hf.agent_execution import resource_tool_schema
from pra_hf.context_records import materialize_authoritative_slice, tool_catalog_slice_records
from pra_hf.tool_records import tool_record_from_callable
from pra_hf.union_discovery import ToolDiscoveryPolicy, discover_candidate_set


def test_callable_union_slice_materializes_every_selected_schema_whole() -> None:
    resources = tuple(
        tool_record_from_callable(function, namespace="test", tenant_id="tenant-a").to_agent_resource()
        for function in (search_user, get_user, update_user)
    )
    scores = {
        "lexical": {resources[0].uri: 0.9, resources[1].uri: 0.8, resources[2].uri: 0.1},
        "tags": {resources[0].uri: 0.7, resources[1].uri: 0.9, resources[2].uri: 0.2},
    }
    candidates = discover_candidate_set(
        "find and retrieve a user",
        resources,
        scores,
        ToolDiscoveryPolicy(max_candidates=2, embedding=False, dictionary=False),
    )
    parent, children = tool_catalog_slice_records(
        candidates, resources, slice_id="slice:test", child_view="full"
    )
    required_bytes = sum(row.size_bytes for row in children) + len(children) - 1
    result = materialize_authoritative_slice(
        parent,
        children,
        max_bytes=required_bytes,
        token_counter=lambda value: len(value.split()),
        native_kv_bytes_per_token=128,
    )

    assert result.status == "materialized"
    assert result.materialized_record_ids == candidates.candidate_uris
    assert result.record_coverage == 1.0
    assert result.partial_record_count == 0
    assert result.atomicity_violations == 0
    assert result.upstream_selection_preserved
    by_uri = {resource.uri: resource for resource in resources}
    assert all(
        child.payload["schema"] == resource_tool_schema(by_uri[child.record_id])
        for child in children
    )


def test_generic_internal_chunk_router_can_drop_schema_structure() -> None:
    resource = {
        "schema": {
            "function": {
                "name": "export_document",
                "description": "Export a document as an artifact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string", "description": "Document identifier."},
                        "format": {"type": "string", "enum": ["pdf", "html"]},
                    },
                    "required": ["document_id", "format"],
                },
            }
        }
    }

    assert _generic_schema_chunk("Export the document as an artifact", resource) == "description"
    assert _generic_schema_chunk("Which document_id and format fields are required?", resource) == "parameters"
