from __future__ import annotations

from pra_hf.tui import AgentShell

from test_pra_agent import _agent


def test_shell_dispatches_task_and_status_commands(tmp_path) -> None:
    agent = _agent(tmp_path)
    agent.start_session("shell", task_description="Root")
    output = []
    shell = AgentShell(agent, output.append)

    assert shell.dispatch('/task new "Inspect repository"')
    assert shell.dispatch("/tasks")
    assert shell.dispatch("/sessions")
    assert shell.dispatch("/tips")
    assert shell.dispatch("/status")
    assert not shell.dispatch("/exit")
    assert any("task-2" in line for line in output)
    assert any("session=shell" in line for line in output)
    assert any("/models before /model use" in line for line in output)
