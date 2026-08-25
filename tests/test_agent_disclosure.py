"""Contracts for Paper 6.5 discovery/disclosure separation and graph provenance."""

from __future__ import annotations

from data.agent_workflows import realistic_tool_catalog
from pra_hf.agent_disclosure import (
    AgentResourcePolicy,
    DisclosureMode,
    ToolCapabilityGraph,
    ToolDisclosurePolicy,
    disclosure_policy_for_profile,
)
from pra_hf.agent_resources import DiscoveryMode


def _uri(resources, name):
    return next(resource.uri for resource in resources if resource.name == name)


def test_sdk_policy_keeps_discovery_channel_and_disclosure_breadth_orthogonal():
    policy = AgentResourcePolicy()
    assert policy.discovery.mode == DiscoveryMode.ADAPTIVE
    assert policy.disclosure.mode == DisclosureMode.PLANNING
    explicit_broad = AgentResourcePolicy(
        discovery=policy.discovery.__class__("explicit", strict=True),
        disclosure=disclosure_policy_for_profile("broad", max_tools=7),
    )
    assert explicit_broad.discovery.hint().strict
    assert explicit_broad.disclosure.max_tools == 7


def test_capability_graph_retains_directional_schema_provenance():
    resources = realistic_tool_catalog()
    graph = ToolCapabilityGraph(resources)
    search = _uri(resources, "search_user")
    get = _uri(resources, "get_user")
    edges = [
        edge
        for edge in graph.outgoing[search]
        if edge.target_uri == get and edge.edge_type == "output_to_input"
    ]
    assert edges
    assert edges[0].directed
    assert "schema:user id" in edges[0].provenance


def test_planning_profile_expands_schema_chain_and_suppresses_destructive_neighbor():
    resources = realistic_tool_catalog()
    graph = ToolCapabilityGraph(resources)
    root = _uri(resources, "search_user")
    trace = graph.disclose(
        (root,),
        ToolDisclosurePolicy(
            "planning",
            max_tools=10,
            family_k=0,
            tag_k=0,
            schema_successor_k=6,
            schema_predecessor_k=2,
            schema_depth=3,
        ),
    )
    names = {graph.by_uri[uri].name for uri in trace.disclosed_uris}
    assert "search_user" in names
    assert "validate_user" in names
    assert "update_user" in names
    assert "delete_user" not in names
    assert any(row.source == "schema_successor" for row in trace.provenance)


def test_minimal_profile_discloses_only_the_direct_root():
    resources = realistic_tool_catalog()
    graph = ToolCapabilityGraph(resources)
    root = _uri(resources, "search_document")
    trace = graph.disclose((root,), disclosure_policy_for_profile("minimal"))
    assert trace.disclosed_uris == (root,)
    assert trace.graph_expansions == 0
