"""Contracts for typed, host-authorized Paper 6.5 tool execution."""

from __future__ import annotations

from data.agent_workflows import realistic_tool_catalog, workflow_executor, workflow_tasks
from pra_hf.agent_execution import (
    ExecutionAuthorization,
    SafeToolExecutor,
    ToolCall,
    has_terminal_tool_intent,
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


def test_tool_call_parser_accepts_only_terminal_durable_action_projection():
    text = (
        "I will inspect the exact definition next.\n"
        'I executed the tool decision named "read_file" with these arguments: '
        '{"path":"src/a.py","start_line":10,"end_line":20}.'
    )
    call = parse_tool_call(text)
    assert call is not None
    assert call.name == "read_file"
    assert call.arguments == {
        "path": "src/a.py",
        "start_line": 10,
        "end_line": 20,
    }

    assert parse_tool_call(text + " This is only a quoted example.") is None
    assert parse_tool_call(
        'I executed the tool decision named "read_file" with these arguments: [].'
    ) is None


def test_tool_call_parser_accepts_terminal_qwen_function_block():
    text = (
        "I will inspect the relevant files.\n"
        "<function=run_command>\n"
        "<parameter=command>\n"
        'find /testbed -name "*.py" | head -20\n'
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    call = parse_tool_call(text)
    assert call is not None
    assert call.name == "run_command"
    assert call.arguments == {
        "command": 'find /testbed -name "*.py" | head -20'
    }

    typed = parse_tool_call(
        "<tool_call><function=read_file>"
        "<parameter=path>src/a.py</parameter>"
        "<parameter=start_line>10</parameter>"
        "<parameter=end_line>20</parameter>"
        "</function></tool_call>"
    )
    assert typed is not None
    assert typed.arguments == {
        "path": "src/a.py",
        "start_line": 10,
        "end_line": 20,
    }

    assert parse_tool_call(text + " quoted only") is None
    assert parse_tool_call(
        "<function=run_command>unexpected"
        "<parameter=command>pwd</parameter></function>"
    ) is None


def test_terminal_tool_intent_detects_malformed_action_without_repairing_it():
    malformed = (
        'I executed the tool decision named "run_command" with these arguments: '
        '{"command":"git diff HEAD"}}.'
    )
    assert parse_tool_call(malformed) is None
    assert has_terminal_tool_intent(malformed)

    malformed_qwen = (
        "I will inspect next.\n<function=read_file>"
        "<parameter=path>src/a.py</parameter>trailing</function></tool_call>"
    )
    assert parse_tool_call(malformed_qwen) is None
    assert has_terminal_tool_intent(malformed_qwen)

    quoted = malformed + " This was only an example."
    assert not has_terminal_tool_intent(quoted)
    assert not has_terminal_tool_intent("The final answer mentions tool calls in prose.")


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


def test_executor_records_handler_failure_as_recoverable_observation():
    resource = _resource("update_user")

    def fail(_arguments, _observations):
        raise ValueError("exact text not found")

    executor = SafeToolExecutor((resource,), {resource.uri: fail})
    result = executor.execute(
        ToolCall("update_user", {"user_id": "u17", "status": "reviewed"}),
        selected_uris=(resource.uri,),
        authorization=ExecutionAuthorization(
            frozenset((resource.uri,)), allow_writes=True
        ),
        call_id="call-failed-edit",
    )

    assert result.accepted
    assert result.executed
    assert result.reason == "execution_error"
    assert result.output["error_type"] == "ValueError"
    assert result.output["error"] == "exact text not found"
    assert result.observation is not None


def test_workflow_catalog_covers_one_through_five_step_composition():
    horizons = {len(task.steps) for task in workflow_tasks()}
    assert {1, 3, 4, 5}.issubset(horizons)
    catalog_names = {resource.name for resource in realistic_tool_catalog()}
    for task in workflow_tasks():
        assert set(task.required_tools) <= catalog_names
        assert task.unsafe_tools.isdisjoint(task.required_tools)
