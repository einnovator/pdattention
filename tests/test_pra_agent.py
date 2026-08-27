from __future__ import annotations

from pra_hf import (
    AgentConfig,
    CapabilitySDK,
    ContextPolicy,
    InMemorySessionService,
    PRAAgent,
    PRAAgentConfig,
    PRARuntime,
    PRARuntimeConfig,
    Skill,
    Tool,
    Toolset,
)


class _Backend:
    name = "test"

    def __init__(self) -> None:
        self.calls = 0
        self.prompts = []

    def add_reference(self, reference, *, text=None, uri=None):
        return {"reference": reference}

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls == 1:
            return '<tool_call>{"name":"lookup","arguments":{"value":"alpha"}}</tool_call>'
        return "The result is ALPHA."

    def inspect(self):
        return {"backend": self.name}


def _agent(tmp_path) -> PRAAgent:
    def lookup(value: str) -> dict[str, str]:
        """Uppercase one value."""

        return {"value": value.upper()}

    toolset = Toolset((Tool(lookup, tenant_id="tenant-a"),))
    sdk = CapabilitySDK(AgentConfig(
        tools=toolset.records,
        skills=(Skill(
            name="alpha-review",
            description="Review alpha lookups.",
            when_to_use="Use for alpha lookup requests.",
            instructions="Verify the uppercase result before answering.",
            tenant_id="tenant-a",
        ),),
        tenant_id="tenant-a",
    ))
    runtime = PRARuntime(
        config=PRARuntimeConfig(),
        backend=_Backend(),
        capability_sdk=sdk,
        executor=toolset.executor(),
        session_service=InMemorySessionService(),
        context_policy=ContextPolicy(local_store=tmp_path, persistent_store=False),
    )
    return PRAAgent(
        runtime,
        config=PRAAgentConfig(user_id="user-a", tenant_id="tenant-a"),
        toolset=toolset,
    )


def test_agent_persists_task_messages_and_compact_tool_result(tmp_path) -> None:
    agent = _agent(tmp_path)
    state = agent.start_session("session-a", task_description="Inspect alpha")

    assert state.active_task_id == "task-1"
    turn = agent.run_turn("Look up alpha")

    assert turn.text == "The result is ALPHA."
    assert turn.tool_executions[0].execution.executed
    assert turn.disclosed_skill_uris
    assert "Verify the uppercase result" in agent.runtime.backend.prompts[0]
    assert len(turn.session.records) == 3
    assert {row.payload.get("role") for row in turn.session.records if isinstance(row.payload, dict)} >= {"user", "assistant"}
    assert agent.runtime.inspect()["logical_sessions"]["session-a"]["records"] == 3


def test_agent_releases_physical_state_and_resumes_logical_session(tmp_path) -> None:
    agent = _agent(tmp_path)
    agent.start_session("session-a", task_description="First task")
    agent.create_task("Second task", task_id="task-2")
    agent.close()

    resumed = agent.start_session("session-a", resume=True)
    assert resumed.active_task_id == "task-2"
    assert len(resumed.tasks.tasks) == 2
