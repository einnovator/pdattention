"""Non-model contracts for the Paper 6.5 pretrained bridge runner."""

from __future__ import annotations

from data.agent_workflows import realistic_tool_catalog, workflow_tasks
from experiments.paper6_5_tools.run_m2_m4_pretrained import (
    M2_CONDITIONS,
    M4_CONDITIONS,
    _m2_disclosure,
    _resources_for_step,
    call_matches,
)
from pra_hf.agent_execution import ToolCall


def test_call_metric_separates_identity_from_arguments():
    step = workflow_tasks()[0].steps[0]
    assert call_matches(ToolCall(step.tool_name, step.arguments), step) == (True, True)
    assert call_matches(ToolCall(step.tool_name, {"user_id": "wrong"}), step) == (True, False)
    assert call_matches(None, step) == (False, False)


def test_m2_controls_preserve_or_remove_the_target_as_declared():
    resources = realistic_tool_catalog()
    task = workflow_tasks()[0]
    target = next(resource for resource in resources if resource.name == task.steps[0].tool_name)
    disclosed = {
        condition: _m2_disclosure(condition, resources, target, 11)
        for condition in M2_CONDITIONS
    }
    assert disclosed["selected"] == disclosed["oracle"] == (target,)
    assert not disclosed["empty"]
    assert target not in disclosed["shuffled"]
    assert target not in disclosed["irrelevant"]
    assert {resource.uri for resource in disclosed["eager"]} == {
        resource.uri for resource in resources
    }


def test_m4_reactive_disclosure_advances_while_no_refresh_stays_fixed():
    resources = realistic_tool_catalog()
    task = next(task for task in workflow_tasks() if len(task.steps) == 3)
    reactive_names = [
        _resources_for_step("reactive_jit", resources, task, index, 11)[0].name
        for index in range(len(task.steps))
    ]
    fixed_names = [
        _resources_for_step("no_refresh", resources, task, index, 11)[0].name
        for index in range(len(task.steps))
    ]
    assert tuple(reactive_names) == task.required_tools
    assert len(set(fixed_names)) == 1
    assert set(M4_CONDITIONS) == {"reactive_jit", "eager_required", "no_refresh"}
