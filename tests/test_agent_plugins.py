"""Agent-family adapters preserve typed identity and explicit fallback semantics."""

from __future__ import annotations

import json

import pytest

from pra_hf.agent_plugins import (
    DeepSeekHarnessPRAAdapter,
    PiCodingAgentPRAAdapter,
    PRAAgentPluginConfig,
)
from pra_hf.deployment import PRAEngineCapabilities, PRAEngineResult
from pra_hf.gateway import PRACapabilityError, PRAGateway


class _OrdinaryEngine:
    def __init__(self) -> None:
        self.last_request = None

    def capabilities(self) -> PRAEngineCapabilities:
        return PRAEngineCapabilities(adapter="ordinary-test")

    def prepare_session(self, request):
        return request.session_id

    def generate(self, request) -> PRAEngineResult:
        self.last_request = request
        return PRAEngineResult("ok")

    def stream(self, request):
        raise NotImplementedError

    def close_session(self, session_id):
        return None


def test_deepseek_bridge_deduplicates_durable_tool_results_and_preserves_task() -> None:
    adapter = DeepSeekHarnessPRAAdapter(
        PRAAgentPluginConfig("deepseek-chat"),
        session_id="session-a",
        task_id="task-a",
    )
    event = {
        "type": "tool/result",
        "id": "event-7",
        "toolName": "read_file",
        "result": {"content": [{"type": "text", "text": "exact output"}]},
    }

    adapter.ingest_events([event, event])
    request = adapter.request([{"role": "user", "content": "Use the output"}])

    assert len(request.resources) == 1
    assert request.resources[0].text == "exact output"
    assert request.resources[0].metadata["task_id"] == "task-a"
    assert request.session_id == "session-a"
    assert "Tensor" not in json.dumps(request.to_dict())


def test_pi_bridge_accepts_rpc_tool_completion_and_message_end() -> None:
    adapter = PiCodingAgentPRAAdapter(PRAAgentPluginConfig("qwen"))
    recognized = adapter.ingest_events([
        {
            "type": "tool_execution_end",
            "toolCallId": "call-1",
            "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "first"}]},
            "isError": False,
        },
        {
            "type": "message_end",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-2",
                "toolName": "read",
                "content": [{"type": "text", "text": "second"}],
            },
        },
    ])

    assert [row.text for row in recognized] == ["first", "second"]
    assert all(row.record_type == "tool_result" for row in recognized)


def test_agent_bridge_uses_explicit_g10_fallback_with_ordinary_engine() -> None:
    engine = _OrdinaryEngine()
    gateway = PRAGateway(engine, mode="G10")
    adapter = PiCodingAgentPRAAdapter(PRAAgentPluginConfig("ordinary"))
    adapter.ingest_event({
        "type": "tool_execution_end",
        "toolCallId": "call-3",
        "toolName": "search",
        "result": {"content": [{"type": "text", "text": "bounded evidence"}]},
        "isError": False,
    })

    result = adapter.generate(gateway, [{"role": "user", "content": "Answer"}])

    assert result.text == "ok"
    assert engine.last_request.resources == ()
    assert "bounded evidence" in engine.last_request.messages[0]["content"]
    assert result.trace[1]["native_kv"] is False


def test_native_required_agent_request_cannot_silently_downgrade() -> None:
    engine = _OrdinaryEngine()
    gateway = PRAGateway(engine, mode="G10")
    adapter = DeepSeekHarnessPRAAdapter(
        PRAAgentPluginConfig(
            "deepseek-chat", allow_text_fallback=False, require_native_pra=True
        )
    )
    adapter.ingest_event({"type": "attachment", "id": "a", "content": "evidence"})

    with pytest.raises(PRACapabilityError):
        adapter.generate(gateway, [{"role": "user", "content": "Answer"}])
