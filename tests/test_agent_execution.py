"""Contracts for typed, host-authorized Paper 6.5 tool execution."""

from __future__ import annotations

from data.agent_workflows import realistic_tool_catalog, workflow_executor, workflow_tasks
from pra_hf.agent_execution import (
    ExecutionAuthorization,
    ToolCall,
    parse_tool_call,
    resource_tool_schema,
)


def _resource(name: str):
    return next(resource for resource in realistic_tool_catalog() if resource.name == name)


def test_openai_schema_and_qwen_tool_call_parser_preserve_arguments():
    resource = _resource("update_user")
    schema = resource_tool_schema(resource)
    assert schema["function"]["name"] == "update_user"
    assert schema["function"]["parameters"]["required"] == ["user_id", "status"]
    call = parse_tool_call(
        '<tool_call>\n{"name":"update_user","arguments":{"user_id":"u17","status":"reviewed"}}\n</tool_call>'
    )
    assert call is not None
    assert call.name == "update_user"
    assert call.arguments == {"user_id": "u17", "status": "reviewed"}


def test_executor_requires_disclosure_and_independent_write_authorization():
    resources = realistic_tool_catalog()
    task = next(task for task in workflow_tasks() if task.task_id == "m4-user-3")
    executor = workflow_executor(resources, task)
    update = _resource("update_user")
    call = ToolCall("update_user", {"user_id": "u17", "status": "reviewed"})
    denied_disclosure = executor.execute(
        call,
        selected_uris=(),
        authorization=ExecutionAuthorization(frozenset((update.uri,)), allow_writes=True),
        call_id="call-1",
    )
    denied_write = executor.execute(
        call,
        selected_uris=(update.uri,),
        authorization=ExecutionAuthorization(frozenset((update.uri,))),
        call_id="call-2",
    )
    accepted = executor.execute(
        call,
        selected_uris=(update.uri,),
        authorization=ExecutionAuthorization(frozenset((update.uri,)), allow_writes=True),
        call_id="call-3",
    )
    assert denied_disclosure.reason == "tool_not_disclosed"
    assert denied_write.reason == "write_not_authorized"
    assert accepted.executed
    assert accepted.output == {"changed": True}
    assert accepted.observation is not None
    assert accepted.observation.kind == "observation"
    assert accepted.observation.metadata["producer_tool_uri"] == update.uri


def test_executor_rejects_destructive_call_without_separate_authority():
    resources = realistic_tool_catalog()
    task = workflow_tasks()[0]
    executor = workflow_executor(resources, task)
    delete = _resource("delete_user")
    result = executor.execute(
        ToolCall("delete_user", {"user_id": "u17"}),
        selected_uris=(delete.uri,),
        authorization=ExecutionAuthorization(
            frozenset((delete.uri,)), allow_writes=True, allow_destructive=False
        ),
        call_id="call-delete",
    )
    assert not result.executed
    assert result.reason == "destructive_not_authorized"


def test_workflow_catalog_covers_one_through_five_step_composition():
    horizons = {len(task.steps) for task in workflow_tasks()}
    assert {1, 3, 4, 5}.issubset(horizons)
    catalog_names = {resource.name for resource in realistic_tool_catalog()}
    for task in workflow_tasks():
        assert set(task.required_tools) <= catalog_names
        assert task.unsafe_tools.isdisjoint(task.required_tools)

