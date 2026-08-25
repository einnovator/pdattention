"""Contracts for the Paper 6.5 M5 policy ladder."""

from __future__ import annotations

from data.agent_workflows import realistic_tool_catalog, workflow_tasks
from experiments.paper6_5_tools.run_m5_disclosure import POLICIES, _set_metrics, policy_trace
from pra_hf.agent_disclosure import ToolCapabilityGraph
from pra_hf.agent_resources import PersistentResourceIndex


def test_m5_declares_all_required_fixed_and_oracle_policies():
    assert POLICIES == tuple(f"p{index}_{name}" for index, name in enumerate((
        "all_eager", "direct_top1", "direct_topk", "family_category", "tag_keyword",
        "schema_graph", "combined_graph", "reactive_jit", "speculative_planning",
        "oracle_capabilities",
    )))


def test_oracle_and_reactive_have_equal_total_budget_but_different_initial_coverage():
    resources = realistic_tool_catalog()
    graph = ToolCapabilityGraph(resources)
    index = PersistentResourceIndex(resources)
    task = next(task for task in workflow_tasks() if len(task.steps) == 4)
    reactive, reactive_total = policy_trace("p7_reactive_jit", graph, index, task)
    oracle, oracle_total = policy_trace("p9_oracle_capabilities", graph, index, task)
    assert len(reactive_total) == len(oracle_total) == len(task.steps)
    assert len(reactive.disclosed_uris) == 1
    assert len(oracle.disclosed_uris) == len(task.steps)


def test_policies_report_graded_relevance_and_unsafe_exposure():
    resources = realistic_tool_catalog()
    graph = ToolCapabilityGraph(resources)
    index = PersistentResourceIndex(resources)
    task = next(task for task in workflow_tasks() if len(task.steps) == 3)
    all_tools, total = policy_trace("p0_all_eager", graph, index, task)
    safe_graph, safe_total = policy_trace("p6_combined_graph", graph, index, task)
    all_metrics = _set_metrics(task, graph, all_tools, total)
    safe_metrics = _set_metrics(task, graph, safe_graph, safe_total)
    assert all_metrics["unsafe_exposure_count"] >= 1
    assert safe_metrics["unsafe_exposure_count"] == 0
    assert 0.0 <= safe_metrics["useful_precision_at_k"] <= 1.0
