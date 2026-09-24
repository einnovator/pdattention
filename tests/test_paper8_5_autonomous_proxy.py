from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import tarfile
import threading
import urllib.error
import urllib.request

import pytest

from pra_hf.agent_history import OpenAIRecordizer

from experiments.paper8_5_agent_memory.autonomous_proxy import (
    AutonomousSelectionConfig,
    AutonomousSelectionProxy,
    join_instrumentation_sidecars,
    summarize_openai_tool_receipts,
    transform_autonomous_payload,
)
from experiments.paper8_5_agent_memory.run_autonomous_swebench import (
    _execution_environment,
    _official_report,
    build_agent_command,
    build_grader_command,
    export_persistent_episode,
    load_persistent_prefix,
    load_locked_task,
    summarize_trace,
)
from experiments.paper8_5_agent_memory.auxiliary_workspace_state import (
    AUXILIARY_WORKSPACE_STATE_LABEL,
    create_auxiliary_workspace_state_prediction,
)
from experiments.paper8_5_agent_memory.serve_agent_transfer_proxy import (
    build_config as build_agent_transfer_config,
    build_parser as build_agent_transfer_parser,
)
from experiments.paper8_5_agent_memory.typed_tool_semantics import (
    DEFAULT_TOOL_SEMANTICS,
)
from experiments.paper8_5_agent_memory.negative_receipts import NegativeRealizationMode
from experiments.paper8_5_agent_memory.materialization import MaterializationMode


def test_agent_transfer_proxy_binds_exact_tokenizer_identity():
    args = build_agent_transfer_parser().parse_args([
        "--upstream", "http://127.0.0.1:9/v1",
        "--trace", "trace.jsonl",
        "--task-id", "task-1",
        "--tokenizer", "/models/tokenizer",
        "--tokenizer-revision", "revision-1",
    ])

    config = build_agent_transfer_config(
        args, tokenizer_identity="/models/tokenizer@revision-1",
    )

    assert config.tokenizer_identity == "/models/tokenizer@revision-1"


def test_default_typed_tool_semantics_use_portable_resource_arguments():
    assert "filePath" in DEFAULT_TOOL_SEMANTICS["read"]["resource_arguments"]
    assert DEFAULT_TOOL_SEMANTICS["read_file"]["operation_kind"] == "read"
    assert DEFAULT_TOOL_SEMANTICS["search_text"]["resource_arguments"] == ["path"]
    assert DEFAULT_TOOL_SEMANTICS["replace_text"]["operation_kind"] == "write"


def test_proxy_joins_generic_execution_receipt_without_changing_tool_text(
    tmp_path: Path,
):
    proxy = AutonomousSelectionProxy(
        "http://127.0.0.1:9/v1",
        config=AutonomousSelectionConfig(
            policy="full",
            input_protocol="openai_tools",
            expected_model="locked-model",
            task_id="task-1",
        ),
        trace_path=tmp_path / "trace.jsonl",
    )
    endpoint = proxy.start()
    receipt = {
        "schema_version": 1,
        "session_id": "task-1",
        "tool_call_id": "call-1",
        "action_record_id": "action-1",
        "observation_record_ids": ["observation-1"],
        "tool_category": "shell",
        "operation_kind": "unknown",
        "transport_status": "completed",
        "semantic_status": "failed",
        "return_code": 2,
        "error_kind": "nonzero_exit",
        "result_complete": True,
        "effect_trace_complete": False,
        "provenance": "runtime_traced",
        "resources": [],
        "receipt_digest": "frozen-receipt",
    }
    request = urllib.request.Request(
        endpoint + "/pra/execution-receipts",
        data=json.dumps(receipt).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 202
        payload, joined = proxy._attach_execution_receipts({
            "messages": [{
                "role": "tool", "tool_call_id": "call-1", "content": "failed",
            }],
        })
    finally:
        proxy.close()

    assert joined == 1
    assert payload["messages"][0]["content"] == "failed"
    assert payload["messages"][0]["metadata"][
        "pra_execution_receipt"
    ] == receipt


def test_generic_execution_receipt_versions_reach_canonical_dag_metadata():
    receipt = {
        "schema_version": 1,
        "session_id": "task-1",
        "tool_call_id": "call-1",
        "action_record_id": "action-1",
        "observation_record_ids": ["observation-1"],
        "tool_category": "filesystem",
        "operation_kind": "write",
        "transport_status": "completed",
        "semantic_status": "succeeded",
        "result_complete": True,
        "effect_trace_complete": True,
        "provenance": "runtime_traced",
        "resources": [{
            "resource_id": "file:///testbed/foo.py",
            "kind": "write",
            "version_before": "sha256:before",
            "version_after": "sha256:after",
        }],
        "receipt_digest": "frozen-receipt",
    }
    result = OpenAIRecordizer().recordize([
        {"role": "user", "content": "Fix it."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "edit", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "done",
            "metadata": {"pra_execution_receipt": receipt},
        },
    ])

    observation = result.history.records[-1]
    assert observation.metadata["resource_version_fingerprints"] == {
        "file:///testbed/foo.py": "sha256:before",
    }
    assert observation.metadata["post_resource_version_fingerprints"] == {
        "file:///testbed/foo.py": "sha256:after",
    }


def test_agent_transfer_proxy_exposes_frozen_policy_parameters(tmp_path: Path):
    args = build_agent_transfer_parser().parse_args([
        "--upstream", "http://model/v1",
        "--trace", str(tmp_path / "trace.jsonl"),
        "--task-id", "task-1",
        "--policy", "frontier_dag_retirement",
        "--boundary-mode", "boundary_free",
        "--protected-head-turns", "2",
        "--protected-tail-turns", "4",
        "--frontier-recent-user-prompts", "2",
        "--frontier-protocol-exemplars", "1",
        "--frontier-allow-heuristic",
        "--max-completion-tokens", "2048",
        "--upstream-qualification-path", "/v1/models",
        "--upstream-connect-attempts", "3",
        "--upstream-connect-retry-seconds", "0.25",
        "--upstream-curl-executable", "/usr/bin/curl",
    ])
    config = build_agent_transfer_config(args)

    assert config.frontier_recent_user_prompts == 2
    assert config.protected_head_turns == 2
    assert config.protected_tail_turns == 4
    assert config.frontier_protocol_exemplars == 1
    assert config.frontier_allow_heuristic is True
    assert config.max_completion_tokens == 2048
    assert config.boundary_mode.value == "boundary_free"
    assert config.tool_semantics_by_name["glob"]["operation_kind"] == (
        "search_discovery"
    )
    assert args.upstream_qualification_path == "/v1/models"
    assert args.upstream_connect_attempts == 3
    assert args.upstream_connect_retry_seconds == 0.25
    assert args.upstream_curl_executable == "/usr/bin/curl"


def test_agent_transfer_proxy_loads_agent_declared_tool_semantics(tmp_path: Path):
    declaration = tmp_path / "tools.json"
    declaration.write_text(json.dumps({
        "tool_semantics_by_name": {
            "file_editor": {
                "category": "filesystem",
                "operation_argument": "command",
                "operation_map": {"view": "read", "str_replace": "write"},
                "resource_arguments": ["path"],
            },
        },
    }))
    args = build_agent_transfer_parser().parse_args([
        "--upstream", "http://model/v1",
        "--trace", str(tmp_path / "trace.jsonl"),
        "--task-id", "task-1",
        "--tool-semantics-json", str(declaration),
    ])

    config = build_agent_transfer_config(args)

    assert set(config.tool_semantics_by_name) == {"file_editor"}
    assert config.tool_semantics_by_name["file_editor"]["operation_argument"] == (
        "command"
    )


def _messages() -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": "Use one bash command.", "name": "control"},
        {"role": "user", "content": "Fix the issue."},
    ]
    for command, output in (
        ("cat a.py", "a"),
        ("cat b.py", "b"),
        ("cat c.py", "c"),
        ("cat d.py", "d"),
    ):
        messages.extend((
            {
                "role": "assistant",
                "content": f"THOUGHT: inspect\n```mswea_bash_command\n{command}\n```",
            },
            {
                "role": "user",
                "content": f"<returncode>0</returncode>\n<output>{output}</output>",
            },
        ))
    return messages


def _payload() -> dict:
    return {
        "model": "locked-model",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "messages": _messages(),
        "response_format": {"type": "text"},
    }


def _openai_tool_payload() -> dict:
    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the issue."},
    ]
    for index in range(4):
        call_id = f"call-{index}"
        messages.extend((
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": f"src/f{index}.py"}),
                    },
                }],
            },
            {
                "role": "tool",
                "name": "read",
                "tool_call_id": call_id,
                "content": (f"source {index} " * 80).strip(),
            },
        ))
    return {
        "model": "locked-model",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": True,
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }],
    }


def _completed_episode(instance_id: str = "repo__old-1") -> dict:
    return {
        "instance_id": instance_id,
        "info": {"exit_status": "Submitted", "submission": "diff --git a/a b/a"},
        "messages": [
            {"role": "system", "content": "Use one bash command."},
            {"role": "user", "content": f"Fix {instance_id}."},
            {
                "role": "assistant",
                "content": "```mswea_bash_command\ncat old.py\n```",
            },
            {
                "role": "user",
                "content": "<returncode>0</returncode>\n<output>old evidence</output>",
            },
            {
                "role": "assistant",
                "content": (
                    "```mswea_bash_command\n"
                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
                ),
            },
            {"role": "exit", "content": "diff --git a/a b/a"},
        ],
    }


def test_full_is_an_exact_message_and_payload_control():
    source = _payload()
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="full", expected_model="locked-model", task_id="task-1"
        ),
    )
    assert result.payload == source
    assert result.payload is not source
    assert result.payload["messages"] is not source["messages"]
    assert result.trace["full_tokens"] == result.trace["selected_tokens"]
    assert result.trace["selected_tokens"] == result.trace["materialized_tokens"]
    assert result.trace["excluded_causal_group_count"] == 0


def test_openai_tool_full_is_exact_and_accepts_buffered_streaming():
    source = _openai_tool_payload()
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="full",
            input_protocol="openai_tools",
            expected_model="locked-model",
            task_id="task-1",
            tool_semantics_by_name={
                "read": {"category": "filesystem", "operation_kind": "read"},
            },
        ),
    )

    assert result.payload == source
    assert result.trace["recordization"]["source"] == "openai_standard"
    assert result.trace["recordization"]["ambiguity_reasons"] == []
    assert result.trace["instrumentation_sidecar_join"]["status"] == (
        "missing_or_invalid_execution_receipts"
    )
    assert result.trace["selection_abstained_for_sidecar"] is False


def test_openai_tool_dag_policy_requires_complete_execution_receipts():
    source = _openai_tool_payload()
    missing = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            input_protocol="openai_tools",
            expected_model="locked-model",
            task_id="task-1",
            boundary_mode="boundary_free",
        ),
    )
    assert missing.trace["selection_abstained_for_sidecar"] is True

    for message in source["messages"]:
        if message["role"] != "tool":
            continue
        call_id = message["tool_call_id"]
        message["metadata"] = {"pra_execution_receipt": {
            "schema_version": 1,
            "session_id": "task-1",
            "tool_call_id": call_id,
            "action_record_id": "action-" + call_id,
            "observation_record_ids": ["observation-" + call_id],
            "tool_category": "filesystem",
            "operation_kind": "read",
            "transport_status": "completed",
            "semantic_status": "succeeded",
            "result_complete": True,
            "effect_trace_complete": True,
            "provenance": "runtime_traced",
            "resources": [],
            "receipt_digest": "receipt-" + call_id,
        }}
    complete = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            input_protocol="openai_tools",
            expected_model="locked-model",
            task_id="task-1",
            boundary_mode="boundary_free",
        ),
    )
    assert complete.trace["instrumentation_sidecar_join"]["status"] == "exact"
    assert complete.trace["selection_abstained_for_sidecar"] is False


def test_openai_tool_receipt_summary_rejects_mismatched_call_identity():
    assert summarize_openai_tool_receipts([{
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "result",
        "metadata": {"pra_execution_receipt": {
            "tool_call_id": "call-other",
        }},
    }]) == {
        "status": "missing_or_invalid_execution_receipts",
        "tool_results": 1,
        "receipts": 0,
        "joined": 0,
        "missing_tool_call_ids": [],
        "invalid_tool_call_ids": ["call-1"],
    }


def test_openai_tool_selection_preserves_native_pairing_fields():
    source = _openai_tool_payload()
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="head_tail_recency",
            budget_fraction=0.55,
            input_protocol="openai_tools",
            expected_model="locked-model",
            task_id="task-1",
            tool_semantics_by_name={
                "read": {"category": "filesystem", "operation_kind": "read"},
            },
        ),
    )

    assert len(result.payload["messages"]) < len(source["messages"])
    assistant_calls = {
        call["id"]
        for message in result.payload["messages"]
        if message["role"] == "assistant"
        for call in message["tool_calls"]
    }
    tool_results = {
        message["tool_call_id"]
        for message in result.payload["messages"]
        if message["role"] == "tool"
    }
    assert assistant_calls == tool_results
    assert all(
        "name" in message
        for message in result.payload["messages"]
        if message["role"] == "tool"
    )


def test_persistent_full_prepends_completed_episode_without_leaking_metadata():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="full",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
        ),
        prior_episodes=(_completed_episode(),),
    )

    contents = [row["content"] for row in result.payload["messages"]]
    assert any("old evidence" in value for value in contents)
    assert any("repo__current-2" in value for value in contents)
    assert all(set(row) <= {"role", "content"} for row in result.payload["messages"])
    assert result.trace["persistent_prefix_applied"] is True
    assert result.trace["prior_episode_count"] == 1
    assert result.trace["episode_index"] == 2
    assert result.trace["full_tokens"] == result.trace["materialized_tokens"]


def test_persistent_active_episode_retires_all_completed_episode_detail():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_active_episode",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
        ),
        prior_episodes=(_completed_episode(),),
    )

    contents = [row["content"] for row in result.payload["messages"]]
    assert not any("old evidence" in value for value in contents)
    assert any("repo__current-2" in value for value in contents)
    assert result.trace["selected_tokens"] < result.trace["full_tokens"]


def test_persistent_oracle_addback_restores_one_complete_excluded_turn():
    config = AutonomousSelectionConfig(
        policy="persistent_episode_retirement",
        expected_model="locked-model",
        task_id="repo__current-2",
        session_id="session-locked",
        episode_index=2,
        completed_recent_turns=1,
        completed_mutation_turns=0,
        completed_verification_turns=0,
    )
    candidate = transform_autonomous_payload(
        _payload(), config, prior_episodes=(_completed_episode(),),
    )
    exclusion = candidate.plan.exclusions[0]
    restored = transform_autonomous_payload(
        _payload(), config, prior_episodes=(_completed_episode(),),
        oracle_addback_causal_group_ids=(exclusion.causal_group_id,),
    )

    assert "old evidence" not in "\n".join(
        row["content"] for row in candidate.payload["messages"]
    )
    assert "old evidence" in "\n".join(
        row["content"] for row in restored.payload["messages"]
    )
    assert restored.trace["oracle_addback"]["restored_causal_group_ids"] == (
        exclusion.causal_group_id,
    )
    assert restored.trace["oracle_addback"]["restored_tokens"] > 0
    assert len(restored.plan.exclusions) == len(candidate.plan.exclusions) - 1


def test_instruction_epoch_oracle_addback_restores_retired_turn():
    config = AutonomousSelectionConfig(
        policy="persistent_instruction_epoch_retirement",
        boundary_mode="boundary_free",
        expected_model="locked-model",
        task_id="repo__current-3",
        session_id="session-locked",
        episode_index=3,
        completed_recent_turns=0,
        completed_mutation_turns=0,
        completed_verification_turns=0,
        completed_protocol_turns=0,
        completed_instruction_epochs=1,
    )
    prior = (
        _completed_episode("repo__old-1"),
        _completed_episode("repo__old-2"),
    )
    candidate = transform_autonomous_payload(
        _payload(), config, prior_episodes=prior,
    )
    exclusion = candidate.plan.exclusions[0]
    restored = transform_autonomous_payload(
        _payload(), config, prior_episodes=prior,
        oracle_addback_causal_group_ids=(exclusion.causal_group_id,),
    )

    assert restored.trace["oracle_addback"]["restored_causal_group_ids"] == (
        exclusion.causal_group_id,
    )
    assert restored.trace["oracle_addback"]["restored_tokens"] > 0
    assert len(restored.plan.exclusions) == len(candidate.plan.exclusions) - 1


def test_persistent_protocol_floor_keeps_clean_action_observation_exemplar():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_episode_retirement",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
            completed_recent_turns=1,
            completed_mutation_turns=0,
            completed_verification_turns=0,
            completed_protocol_turns=1,
        ),
        prior_episodes=(_completed_episode(),),
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "old evidence" in visible
    assert any(
        reason == "completed_episode_progress_spine"
        for _, reason in result.plan.selection_reasons
    )


def test_boundary_free_global_policy_receives_no_episode_signal():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_global_retirement",
            boundary_mode="boundary_free",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
            completed_recent_turns=1,
            completed_mutation_turns=0,
            completed_verification_turns=0,
            completed_protocol_turns=0,
        ),
        prior_episodes=(_completed_episode(),),
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "pra_episode_boundary" not in visible
    assert "old evidence" not in visible
    assert "Fix repo__old-1." in visible
    assert "Fix the issue." in visible
    assert all("episode-" not in row for row in result.plan.selected_record_ids)
    assert all("episode" not in reason for _, reason in result.plan.selection_reasons)
    assert result.trace["wire_plan"]["decision_metadata"] == {
        "instruction_floor": "all_user_instructions"
    }


def test_instruction_epoch_policy_keeps_current_history_without_episode_signal():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            boundary_mode="boundary_free",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
            completed_recent_turns=0,
            completed_mutation_turns=0,
            completed_verification_turns=0,
            completed_protocol_turns=0,
        ),
        prior_episodes=(_completed_episode(),),
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "Fix repo__old-1." in visible
    assert "old evidence" not in visible
    assert "Fix the issue." in visible
    for resource in ("a.py", "b.py", "c.py", "d.py"):
        assert f"cat {resource}" in visible
    assert result.plan.policy == "persistent_instruction_epoch_retirement"
    assert all("episode" not in reason for _, reason in result.plan.selection_reasons)


def test_instruction_epoch_policy_can_keep_latest_prior_epoch_whole():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            boundary_mode="boundary_free",
            expected_model="locked-model",
            task_id="repo__current-3",
            session_id="session-locked",
            episode_index=3,
            completed_recent_turns=0,
            completed_mutation_turns=0,
            completed_verification_turns=0,
            completed_protocol_turns=0,
            completed_instruction_epochs=1,
        ),
        prior_episodes=(
            _completed_episode("repo__old-1"),
            _completed_episode("repo__old-2"),
        ),
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "Fix repo__old-1." in visible
    assert "Fix repo__old-2." in visible
    assert "Fix the issue." in visible
    assert visible.count("old evidence") == 1
    assert "recent_complete_instruction_epoch" in dict(
        result.plan.selection_reasons
    ).values()


def test_instruction_epoch_compact_finalization_closes_prompt_without_submission_sentinel():
    prior = _completed_episode()
    prior["messages"][-1]["content"] = (
        "diff --git a/old.py b/old.py\n"
        "--- a/old.py\n+++ b/old.py\n@@ -1 +1 @@\n-old\n+new"
    )
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            boundary_mode="boundary_free",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
            completed_recent_turns=0,
            completed_mutation_turns=0,
            completed_verification_turns=0,
            completed_protocol_turns=0,
            completed_finalization_turns=1,
            compact_completed_finalizations=True,
        ),
        prior_episodes=(prior,),
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "Fix repo__old-1." in visible
    assert "Prior instruction completed" in visible
    assert "Closure recorded" in visible
    assert "old evidence" not in visible
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in visible
    assert "```mswea_bash_command\ntrue\n```" not in visible
    assert result.trace["prior_finalization_receipt_count"] == 1
    assert result.trace["prior_finalization_receipt_token_saving"] > 0
    assert result.trace["materialized_tokens"] < result.trace["selected_tokens"]
    roles = [row["role"] for row in result.payload["messages"]]
    assert not any(left == right == "assistant" for left, right in zip(roles, roles[1:]))


def test_compact_finalization_requires_instruction_epoch_finalization_floor():
    with pytest.raises(ValueError, match="positive finalization floor"):
        AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            boundary_mode="boundary_free",
            compact_completed_finalizations=True,
        )
    with pytest.raises(ValueError, match="instruction-epoch retirement"):
        AutonomousSelectionConfig(
            policy="head_tail_recency",
            completed_finalization_turns=1,
            compact_completed_finalizations=True,
        )


def test_atomic_closed_epoch_retirement_keeps_only_active_instruction_and_protocol():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            boundary_mode="boundary_free",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
            completed_recent_turns=0,
            completed_mutation_turns=0,
            completed_verification_turns=0,
            completed_protocol_turns=0,
            completed_finalization_turns=0,
            retire_closed_instructions=True,
            keep_completed_task_statements=False,
        ),
        prior_episodes=(_completed_episode(),),
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "Fix repo__old-1." not in visible
    assert "old evidence" not in visible
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in visible
    assert "Fix the issue." in visible
    assert result.trace["retire_closed_instructions"] is True
    assert result.trace["wire_plan"]["decision_metadata"] == {
        "instruction_floor": "newest_user_instruction",
        "prior_instruction_retirement": "terminal_epoch_atomic",
    }


def test_closed_epoch_retirement_honors_all_user_instruction_floor():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            boundary_mode="boundary_free",
            expected_model="locked-model",
            task_id="repo__current-2",
            session_id="session-locked",
            episode_index=2,
            completed_recent_turns=0,
            completed_mutation_turns=0,
            completed_verification_turns=0,
            completed_protocol_turns=0,
            completed_finalization_turns=0,
            retire_closed_instructions=True,
            keep_completed_task_statements=True,
        ),
        prior_episodes=(_completed_episode(),),
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "Fix repo__old-1." in visible
    assert "old evidence" not in visible
    assert "Fix the issue." in visible
    assert result.trace["keep_completed_task_statements"] is True
    assert result.trace["wire_plan"]["decision_metadata"] == {
        "instruction_floor": "all_user_instructions",
    }


def test_boundary_free_global_policy_rejects_explicit_composition():
    with pytest.raises(ValueError, match="requires boundary_free"):
        AutonomousSelectionConfig(policy="persistent_global_retirement")
    with pytest.raises(ValueError, match="requires boundary_free"):
        AutonomousSelectionConfig(policy="persistent_instruction_epoch_retirement")
    with pytest.raises(ValueError, match="requires boundary_free"):
        AutonomousSelectionConfig(policy="frontier_dag_retirement")


def test_frontier_dag_policy_requires_full_pre_retirement_budget():
    with pytest.raises(ValueError, match="100% pre-retirement budget"):
        AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            boundary_mode="boundary_free",
            budget_fraction=0.9,
        )


def test_frontier_dag_policy_rejects_negative_protocol_exemplar_floor():
    with pytest.raises(ValueError, match="protocol_exemplars cannot be negative"):
        AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            boundary_mode="boundary_free",
            frontier_protocol_exemplars=-1,
        )


def test_frontier_dag_policy_rejects_negative_workflow_exemplar_floor():
    with pytest.raises(ValueError, match="workflow_exemplars cannot be negative"):
        AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            boundary_mode="boundary_free",
            frontier_workflow_exemplars=-1,
        )


def test_persistent_prefix_count_must_match_episode_index():
    with pytest.raises(ValueError, match="episode count"):
        transform_autonomous_payload(
            _payload(),
            AutonomousSelectionConfig(
                policy="full",
                expected_model="locked-model",
                task_id="repo__current-2",
                session_id="session-locked",
                episode_index=3,
            ),
            prior_episodes=(_completed_episode(),),
        )


def test_persistent_episode_export_round_trips_with_identity(tmp_path):
    agent_output = tmp_path / "agent"
    agent_output.mkdir()
    trajectory = _completed_episode("repo__issue-1")
    trajectory_path = agent_output / "repo__issue-1.traj.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    instrumentation_root = tmp_path / "instrumentation"
    instrumentation_root.mkdir()
    episode_path = tmp_path / "episode.json"

    exported = export_persistent_episode(
        agent_output=agent_output,
        instrumentation_root=instrumentation_root,
        instance_id="repo__issue-1",
        output=episode_path,
    )
    prefix_path = tmp_path / "prefix.json"
    prefix_path.write_text(json.dumps({
        "schema_version": 1,
        "session_id": "session-1",
        "episodes": [exported],
    }), encoding="utf-8")

    episodes, identity = load_persistent_prefix(prefix_path)
    assert episodes == [exported["trajectory"]]
    assert identity["instance_ids"] == ["repo__issue-1"]
    assert identity["session_id"] == "session-1"
    assert identity["episode_count"] == 1
    assert identity["sha256"]


def test_full_structured_observation_preserves_all_records_but_compacts_payload():
    source = _payload()
    source["messages"][3]["content"] = (
        "<returncode>0</returncode>\n<output>\n"
        + "\n".join(
            ["noise"] * 300
            + ["def target_function(value):", "    return value"]
            + ["noise"] * 300
        )
        + "\n</output>"
    )
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="full_structured_observation",
            budget_fraction=1.0,
            materialization_mode=MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
            materialization_threshold_tokens=64,
            expected_model="locked-model",
            task_id="task-1",
        ),
    )

    assert result.trace["selected_message_count"] == result.trace["full_message_count"]
    assert result.trace["excluded_causal_group_count"] == 0
    assert result.trace["selected_tokens"] == result.trace["full_tokens"]
    assert result.trace["materialized_tokens"] < result.trace["selected_tokens"]
    assert "def target_function" in result.payload["messages"][3]["content"]
    assert "<elided_lines>" in result.payload["messages"][3]["content"]


def test_full_structured_observation_rejects_selection_or_wrong_materializer():
    with pytest.raises(ValueError, match="100% logical-history"):
        AutonomousSelectionConfig(
            policy="full_structured_observation",
            budget_fraction=.9,
            materialization_mode=MaterializationMode.TOOL_STRUCTURED_EVIDENCE,
        )
    with pytest.raises(ValueError, match="requires tool_structured_evidence"):
        AutonomousSelectionConfig(policy="full_structured_observation")


def test_official_report_falls_back_to_exact_per_instance_artifact(tmp_path):
    run_id = "locked-run"
    instance_id = "django__django-15277"
    report = (
        tmp_path / "logs" / "run_evaluation" / run_id / "model" /
        instance_id / "report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({instance_id: {
        "patch_successfully_applied": True,
        "resolved": True,
    }}), encoding="utf-8")

    result = _official_report(tmp_path, run_id, instance_id)

    assert result["resolved"] is True
    assert result["score"] == 1.0
    assert result["error"] is False
    assert result["raw_report_kind"] == "per_instance"


def test_matched_token_tail_is_a_strict_autonomous_text_ceiling_without_sidecars():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="matched_token_tail",
            budget_fraction=.70,
            expected_model="locked-model",
            task_id="task-1",
        ),
    )

    assert result.trace["policy"] == "matched_token_tail"
    assert result.trace["protected_head_turns"] == 1
    assert result.trace["protected_tail_turns"] == 1
    assert result.trace["matched_tail_boundary_compaction"] == "legacy"
    assert result.trace["selection_reasons"]
    assert result.trace["selection_abstained_for_sidecar"] is False
    assert result.trace["budget_interpretation"] == (
        "strict_materialized_token_ceiling_with_mandatory_overflow"
    )
    assert result.trace["budget_satisfied"] is True
    assert result.trace["materialized_tokens"] <= result.trace["requested_budget_tokens"]
    assert result.trace["whole_turn_budget_undershoot_tokens"] > 0
    assert result.trace["materialized_budget_unused_tokens"] > 0
    assert result.trace["materialized_budget_overshoot_tokens"] == 0
    assert result.trace["selected_message_count"] < result.trace["full_message_count"]
    assert [row["content"] for row in result.payload["messages"][:2]] == [
        row["content"] for row in _payload()["messages"][:2]
    ]


def test_matched_token_tail_reports_unavoidable_immutable_prompt_overflow():
    source = _payload()
    source["messages"] = source["messages"][:2]
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="matched_token_tail",
            budget_fraction=.50,
            expected_model="locked-model",
        ),
    )

    assert result.payload == source
    assert result.trace["exact_request_passthrough"] is True
    assert result.trace["mandatory_budget_overflow_tokens"] > 0
    assert result.trace["unexplained_materialized_budget_overflow_tokens"] == 0
    assert result.trace["materialized_budget_unused_tokens"] == 0
    assert result.trace["materialized_budget_overshoot_tokens"] > 0
    assert result.trace["budget_satisfied"] is True


def test_matched_token_tail_preserves_standalone_format_error_recovery():
    source = _payload()
    source["messages"] = source["messages"][:2] + [{
        "role": "user",
        "content": (
            "Your previous response reached the output token limit. "
            "Respond concisely with exactly one action."
        ),
    }]
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="matched_token_tail",
            budget_fraction=.95,
            expected_model="locked-model",
            task_id="task-1",
        ),
    )

    assert result.payload == source
    assert result.trace["selected_message_count"] == 3
    assert result.trace["exact_request_passthrough"] is True
    assert result.trace["mandatory_budget_overflow_tokens"] > 0


def test_matched_token_tail_preserves_complete_current_turn_as_mandatory_overflow():
    source = _payload()
    source["messages"] = source["messages"][:4]
    source["messages"][3]["content"] = (
        "<returncode>0</returncode>\n<output>"
        + " ".join(f"small{index}" for index in range(100))
        + "</output>"
    )
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="matched_token_tail",
            budget_fraction=.95,
            expected_model="locked-model",
            task_id="task-1",
        ),
    )

    assert result.payload == source
    assert result.trace["selected_message_count"] == 4
    assert result.trace["exact_request_passthrough"] is True
    assert result.trace["mandatory_budget_overflow_tokens"] > 0
    assert result.trace["unexplained_materialized_budget_overflow_tokens"] == 0
    assert result.trace["budget_satisfied"] is True


def test_abstaining_negative_policy_is_an_exact_request_noop():
    source = _payload()
    result = transform_autonomous_payload(
        source,
        AutonomousSelectionConfig(
            policy="h2b_verified_write",
            expected_model="locked-model",
            require_exact_sidecars=False,
        ),
    )
    assert result.payload == source
    assert result.trace["exact_request_passthrough"] is True
    assert result.trace["request_input_sha256"] == result.trace[
        "selected_messages_sha256"
    ]


def test_negative_arm_preserves_prompt_and_current_turn_and_reports_exclusions():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="h4_working_set",
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
            working_set_resources=2,
            task_id="task-1",
            require_exact_sidecars=False,
        ),
    )
    selected = result.payload["messages"]
    assert selected[0]["role"] == "system"
    assert selected[1]["role"] == "user"
    assert selected[-1]["content"].endswith("<output>d</output>")
    assert "cat a.py" not in "\n".join(row["content"] for row in selected)
    assert result.trace["excluded_causal_group_count"] == 2
    assert result.trace["full_tokens"] > result.trace["selected_tokens"]
    assert result.trace["observation_metadata_coverage"] == {
        "observation_records": 4,
        "complete_status_records": 0,
        "resource_version_records": 0,
    }


def _write_receipt(
    root: Path,
    step: int,
    command: str,
    metadata: dict,
) -> None:
    session = root / "container-session"
    session.mkdir(parents=True, exist_ok=True)
    (session / f"execution_{step:04d}.json").write_text(json.dumps({
        "schema_version": 1,
        "step": step,
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "observation_metadata": metadata,
    }))


def _write_workspace_checkpoint(
    root: Path,
    *,
    step: int,
    patch: bytes,
    complete: bool = True,
    workspace_stable: bool = True,
    index_patch: bytes = b"",
    command: str | None = None,
) -> None:
    session = root / "container-session"
    session.mkdir(parents=True, exist_ok=True)
    worktree_path = session / f"decision_{step:04d}.worktree.patch"
    index_path = session / f"decision_{step:04d}.index.patch"
    archive_path = session / f"decision_{step:04d}.untracked.tar.gz"
    worktree_path.write_bytes(patch)
    index_path.write_bytes(index_patch)
    with tarfile.open(archive_path, "w:gz"):
        pass
    command = command or f"cat foo.py {step}"
    command_digest = hashlib.sha256(command.encode()).hexdigest()
    fingerprint = f"workspace-{step}"
    decision = {
        "schema_version": 1,
        "scope": "git_HEAD_plus_tracked_diff_plus_nonignored_untracked_files",
        "step": step,
        "command_sha256": command_digest,
        "workspace_version_fingerprint": fingerprint,
        "index_patch": index_path.name,
        "index_patch_sha256": hashlib.sha256(index_patch).hexdigest(),
        "index_capture_complete": complete,
        "worktree_patch": worktree_path.name,
        "worktree_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "worktree_capture_complete": complete,
        "untracked_archive": archive_path.name,
        "untracked_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "untracked_capture_complete": complete,
        "checkpoint_complete": complete,
    }
    (session / f"decision_{step:04d}.json").write_text(json.dumps(decision))
    post_fingerprint = fingerprint if workspace_stable else f"workspace-{step}-changed"
    execution = {
        "schema_version": 1,
        "step": step,
        "command_sha256": command_digest,
        "pre_state": {
            "complete": True,
            "workspace_version_fingerprint": fingerprint,
        },
        "post_state": {
            "complete": True,
            "workspace_version_fingerprint": post_fingerprint,
        },
    }
    (session / f"execution_{step:04d}.json").write_text(json.dumps(execution))


def _write_primary_predictions(path: Path, patch: str = "source excerpt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "org__repo-1": {
            "model_name_or_path": "openai/locked-model",
            "instance_id": "org__repo-1",
            "model_patch": patch,
        }
    }))


def _write_submitted_trajectory(
    primary_predictions: Path,
    *,
    command: str,
    submission: str = "source excerpt",
) -> None:
    trajectory = primary_predictions.parent / "org__repo-1" / "org__repo-1.traj.json"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text(json.dumps({
        "instance_id": "org__repo-1",
        "trajectory_format": "mini-swe-agent-1",
        "info": {"exit_status": "Submitted"},
        "messages": [
            {
                "role": "assistant",
                "content": f"```mswea_bash_command\n{command}\n```",
            },
            {
                "role": "exit",
                "content": submission,
                "extra": {
                    "exit_status": "Submitted",
                    "submission": submission,
                },
            },
        ],
    }))


def test_auxiliary_workspace_state_copies_verified_final_patch(tmp_path):
    patch = (
        b"diff --git a/foo.py b/foo.py\n"
        b"--- a/foo.py\n"
        b"+++ b/foo.py\n"
        b"@@ -1 +1 @@\n-old\n+new\n"
    )
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(instrumentation, step=0, patch=patch)
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary)

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "available"
    assert outcome["outcome_label"] == AUXILIARY_WORKSPACE_STATE_LABEL
    assert outcome["evidence_role"] == "auxiliary_only"
    assert outcome["replaces_official_submission"] is False
    assert outcome["checkpoint_step"] == 0
    assert outcome["last_action_workspace_stable"] is True
    assert outcome["primary_submission_looks_like_git_diff"] is False
    assert Path(outcome["patch"]).read_bytes() == patch
    auxiliary_predictions = json.loads(Path(outcome["predictions"]).read_text())
    assert auxiliary_predictions["org__repo-1"]["model_patch"].encode() == patch
    assert json.loads(primary.read_text())["org__repo-1"]["model_patch"] == "source excerpt"


def test_auxiliary_workspace_state_accepts_certified_terminal_read_pipeline(tmp_path):
    command = (
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
        "cat /testbed/foo.py | sed -n '1,5p'"
    )
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(
        instrumentation,
        step=0,
        patch=b"diff --git a/foo.py b/foo.py\n",
        command=command,
    )
    (instrumentation / "container-session" / "execution_0000.json").unlink()
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary)
    _write_submitted_trajectory(primary, command=command)

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "available"
    assert outcome["final_state_certificate"] == (
        "submitted_read_only_terminal_checkpoint"
    )
    assert outcome["decision_receipt_count"] == 1
    assert outcome["execution_receipt_count"] == 0
    assert outcome["last_action_workspace_stable"] is False
    assert outcome["terminal_command_programs"] == ["cat", "sed"]
    assert outcome["terminal_submission_matches_primary_prediction"] is True


def test_auxiliary_workspace_state_accepts_output_only_echo_submission_chain(tmp_path):
    command = (
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
        "echo '=== FINAL PATCH ===' && echo 'File: foo.py' && echo 'done'"
    )
    submission = "=== FINAL PATCH ===\nFile: foo.py\ndone\n"
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(
        instrumentation,
        step=0,
        patch=b"diff --git a/foo.py b/foo.py\n",
        command=command,
    )
    (instrumentation / "container-session" / "execution_0000.json").unlink()
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary, patch=submission)
    _write_submitted_trajectory(
        primary, command=command, submission=submission,
    )

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "available"
    assert outcome["final_state_certificate"] == (
        "submitted_read_only_terminal_checkpoint"
    )
    assert outcome["terminal_command_programs"] == ["echo", "echo", "echo"]


@pytest.mark.parametrize(
    "command",
    (
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat foo.py && rm foo.py",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat foo.py; cat bar.py",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat foo.py | tee copied.py",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat foo.py > copied.py",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && python3 -c 'print(1)'",
    ),
)
def test_auxiliary_workspace_state_rejects_uncertified_terminal_shell(
    tmp_path, command
):
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(
        instrumentation,
        step=0,
        patch=b"diff --git a/foo.py b/foo.py\n",
        command=command,
    )
    (instrumentation / "container-session" / "execution_0000.json").unlink()
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary)
    _write_submitted_trajectory(primary, command=command)

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "unavailable"
    assert outcome["reason"] == "terminal_command_not_certified_read_only"


@pytest.mark.parametrize(
    ("trajectory_command", "submission", "reason"),
    (
        (
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat other.py",
            "source excerpt",
            "terminal_command_digest_mismatch",
        ),
        (
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat foo.py",
            "different excerpt",
            "terminal_submission_digest_mismatch",
        ),
    ),
)
def test_auxiliary_workspace_state_binds_terminal_command_and_submission(
    tmp_path, trajectory_command, submission, reason
):
    checkpoint_command = (
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat foo.py"
    )
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(
        instrumentation,
        step=0,
        patch=b"diff --git a/foo.py b/foo.py\n",
        command=checkpoint_command,
    )
    (instrumentation / "container-session" / "execution_0000.json").unlink()
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary)
    _write_submitted_trajectory(
        primary,
        command=trajectory_command,
        submission=submission,
    )

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "unavailable"
    assert outcome["reason"] == reason


def test_auxiliary_workspace_state_does_not_fall_back_from_incomplete_last_checkpoint(
    tmp_path,
):
    patch = b"diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n"
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(instrumentation, step=0, patch=patch)
    _write_workspace_checkpoint(
        instrumentation, step=1, patch=patch, complete=False
    )
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary)

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "unavailable"
    assert outcome["reason"] == "last_checkpoint_incomplete"
    assert not (tmp_path / "run" / "auxiliary_workspace_state_preds.json").exists()
    assert not (tmp_path / "run" / "auxiliary_workspace_state.patch").exists()


@pytest.mark.parametrize(
    ("workspace_stable", "index_patch", "reason"),
    (
        (False, b"", "final_action_changed_or_unverified_workspace"),
        (True, b"diff --git a/bar.py b/bar.py\n", "staged_and_unstaged_composition_unsupported"),
    ),
)
def test_auxiliary_workspace_state_rejects_unsafe_final_state(
    tmp_path, workspace_stable, index_patch, reason
):
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(
        instrumentation,
        step=0,
        patch=b"diff --git a/foo.py b/foo.py\n",
        workspace_stable=workspace_stable,
        index_patch=index_patch,
    )
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary)

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "unavailable"
    assert outcome["reason"] == reason


def test_auxiliary_workspace_state_rejects_checkpoint_digest_mismatch(tmp_path):
    instrumentation = tmp_path / "instrumentation"
    _write_workspace_checkpoint(
        instrumentation,
        step=0,
        patch=b"diff --git a/foo.py b/foo.py\n",
    )
    source = (
        instrumentation / "container-session" / "decision_0000.worktree.patch"
    )
    source.write_bytes(source.read_bytes() + b"tampered\n")
    primary = tmp_path / "agent" / "preds.json"
    _write_primary_predictions(primary)

    outcome = create_auxiliary_workspace_state_prediction(
        instrumentation_root=instrumentation,
        output=tmp_path / "run",
        primary_predictions=primary,
        instance_id="org__repo-1",
    )

    assert outcome["status"] == "unavailable"
    assert outcome["reason"] == "checkpoint_digest_mismatch"
    assert outcome["official_grading_completed"] is False


def test_sidecar_join_enables_guarded_h2_without_leaking_extra(tmp_path):
    commands = ("echo fixed > foo.py", "cat foo.py")
    messages = [
        {"role": "system", "content": "Use bash."},
        {"role": "user", "content": "Fix foo.py."},
    ]
    for command, output in zip(commands, ("", "fixed")):
        messages.extend((
            {
                "role": "assistant",
                "content": f"```mswea_bash_command\n{command}\n```",
            },
            {
                "role": "user",
                "content": f"<returncode>0</returncode>\n<output>{output}</output>",
            },
        ))
    messages[2]["content"] = (
        "Illustrative payload:\n```python\nprint('not the action')\n```\n"
        f"```mswea_bash_command\n{commands[0]}\n```"
    )
    common = {
        "return_code": 0,
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    _write_receipt(tmp_path, 0, commands[0], {
        **common,
        "post_resource_version_fingerprints": {"foo.py": "v2"},
    })
    _write_receipt(tmp_path, 1, commands[1], {
        **common,
        "resource_version_fingerprints": {"foo.py": "v2"},
    })
    payload = {
        "model": "locked-model",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "messages": messages,
    }
    result = transform_autonomous_payload(
        payload,
        AutonomousSelectionConfig(
            policy="h2a_write_current_read",
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
        ),
        instrumentation_root=tmp_path,
    )
    assert result.trace["instrumentation_sidecar_join"]["status"] == "exact"
    assert result.trace["excluded_causal_group_count"] == 1
    assert "echo fixed" not in "\n".join(
        row["content"] for row in result.payload["messages"]
    )
    assert all("extra" not in row for row in result.payload["messages"])
    assert payload["messages"] == messages


def test_sidecar_join_accepts_intercepted_terminal_submission(tmp_path):
    command = "cat foo.py"
    submission = (
        "git diff -- foo.py > patch.txt && "
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
    )
    messages = [
        {"role": "system", "content": "Use bash."},
        {"role": "user", "content": "Fix foo.py."},
        {
            "role": "assistant",
            "content": f"```mswea_bash_command\n{command}\n```",
        },
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>source</output>",
        },
        {
            "role": "assistant",
            "content": f"```mswea_bash_command\n{submission}\n```",
        },
        {"role": "exit", "content": "diff --git a/foo.py b/foo.py\n"},
    ]
    _write_receipt(tmp_path, 0, command, {
        "return_code": 0,
        "output_complete": True,
        "resource_version_fingerprints": {"foo.py": "v1"},
    })

    enriched, audit = join_instrumentation_sidecars(messages, tmp_path)

    assert audit == {
        "status": "exact_terminal_submission",
        "commands": 2,
        "receipts": 1,
        "joined": 1,
    }
    assert enriched[3]["extra"]["return_code"] == 0
    assert "extra" not in enriched[5]


def test_sidecar_join_accepts_miniswe_xml_action_protocol(tmp_path):
    command = "sed -n '1,80p' foo.py"
    messages = [
        {"role": "system", "content": "Use one XML-wrapped Bash action."},
        {"role": "user", "content": "Fix foo.py."},
        {
            "role": "assistant",
            "content": (
                "THOUGHT: inspect the source\n"
                f"<mswea_bash_command>{command}</mswea_bash_command>"
            ),
        },
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>source</output>",
        },
    ]
    _write_receipt(tmp_path, 0, command, {
        "return_code": 0,
        "output_complete": True,
        "resource_version_fingerprints": {"foo.py": "v1"},
    })

    enriched, audit = join_instrumentation_sidecars(messages, tmp_path)

    assert audit == {
        "status": "exact",
        "commands": 1,
        "receipts": 1,
        "joined": 1,
    }
    assert enriched[3]["extra"]["return_code"] == 0


def test_sidecar_join_rejects_mixed_or_multiple_miniswe_actions(tmp_path):
    messages = [
        {"role": "system", "content": "Use one action."},
        {"role": "user", "content": "Fix foo.py."},
        {
            "role": "assistant",
            "content": (
                "```mswea_bash_command\ncat foo.py\n```\n"
                "<mswea_bash_command>cat foo.py</mswea_bash_command>"
            ),
        },
        {"role": "user", "content": "<returncode>0</returncode>"},
    ]

    _, audit = join_instrumentation_sidecars(messages, tmp_path)

    assert audit["status"] == "assistant_command_unparseable"
    assert audit["joined"] == 0


def test_sidecar_join_does_not_forgive_nonterminal_missing_receipt(tmp_path):
    messages = [
        {"role": "system", "content": "Use bash."},
        {"role": "user", "content": "Fix foo.py."},
        {
            "role": "assistant",
            "content": "```mswea_bash_command\ncat foo.py\n```",
        },
        {"role": "user", "content": "<returncode>0</returncode>"},
        {
            "role": "assistant",
            "content": "```mswea_bash_command\necho not-a-submission\n```",
        },
        {"role": "exit", "content": "done"},
    ]
    _write_receipt(tmp_path, 0, "cat foo.py", {
        "return_code": 0,
        "output_complete": True,
    })

    _, audit = join_instrumentation_sidecars(messages, tmp_path)

    assert audit["status"] == "missing_or_mismatched_receipts"
    assert audit["joined"] == 0


def test_progress_spine_v4_keeps_action_and_receipts_superseded_observation(tmp_path):
    commands = ("cat foo.py", "cat foo.py", "true")
    large_old = "\n".join(f"old source line {index}" for index in range(200))
    messages = [
        {"role": "system", "content": "Use bash."},
        {"role": "user", "content": "Fix foo.py."},
    ]
    for command, output in zip(commands, (large_old, "current source", "done")):
        messages.extend((
            {"role": "assistant", "content": f"```mswea_bash_command\n{command}\n```"},
            {"role": "user", "content": f"<returncode>0</returncode>\n<output>{output}</output>"},
        ))
    common = {
        "return_code": 0,
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
    }
    _write_receipt(tmp_path, 0, commands[0], {
        **common, "resource_version_fingerprints": {"foo.py": "v1"},
    })
    _write_receipt(tmp_path, 1, commands[1], {
        **common, "resource_version_fingerprints": {"foo.py": "v1"},
    })
    _write_receipt(tmp_path, 2, commands[2], common)
    payload = {"model": "locked-model", "messages": messages}

    result = transform_autonomous_payload(
        payload,
        AutonomousSelectionConfig(
            policy="task_aware_progress_spine_v4",
            negative_realization=NegativeRealizationMode.OBSERVATION_RECEIPT,
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
        ),
        instrumentation_root=tmp_path,
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "```mswea_bash_command\ncat foo.py\n```" in visible
    assert "old source line 199" not in visible
    assert "older read superseded" in visible
    assert result.trace["receipt_count"] == 1
    assert result.trace["receipt_token_saving"] > 0
    assert result.trace["recordization"] == {
        "source": "typed",
        "explicit_records": len(messages),
        "inferred_records": 0,
        "ambiguity_reasons": [],
    }


def test_progress_spine_v4_supports_metadata_only_protocol_stub(tmp_path):
    commands = ("cat foo.py", "cat foo.py", "true")
    messages = [{"role": "system", "content": "Use bash."},
                {"role": "user", "content": "Fix foo.py."}]
    for command, output in zip(commands, ("old " * 300, "current", "done")):
        messages.extend((
            {"role": "assistant", "content": f"```mswea_bash_command\n{command}\n```"},
            {"role": "user", "content": f"<returncode>0</returncode>\n<output>{output}</output>"},
        ))
    common = {"return_code": 0, "output_complete": True,
              "timed_out": False, "output_truncated": False}
    _write_receipt(tmp_path, 0, commands[0], {
        **common, "resource_version_fingerprints": {"foo.py": "v1"},
    })
    _write_receipt(tmp_path, 1, commands[1], {
        **common, "resource_version_fingerprints": {"foo.py": "v1"},
    })
    _write_receipt(tmp_path, 2, commands[2], common)

    result = transform_autonomous_payload(
        {"model": "locked-model", "messages": messages},
        AutonomousSelectionConfig(
            policy="task_aware_progress_spine_v4",
            negative_realization=NegativeRealizationMode.PROTOCOL_STUB,
            expected_model="locked-model", protected_head_turns=0,
            protected_tail_turns=1,
        ),
        instrumentation_root=tmp_path,
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "[PRA memory]" not in visible
    assert "<returncode>0</returncode>\n<output></output>" in visible
    assert result.trace["receipt_count"] == 1


def test_dag_certified_autonomous_policy_uses_only_runtime_bound_duplicates(tmp_path):
    commands = ("cat foo.py", "cat foo.py", "true")
    old_output = "same source line\n" * 200
    messages = [
        {"role": "system", "content": "Use bash."},
        {"role": "user", "content": "Fix foo.py."},
    ]
    for index, (command, output) in enumerate(zip(
        commands, (old_output, old_output, "done")
    )):
        messages.extend((
            {
                "role": "assistant",
                "content": (
                    f"THOUGHT: read attempt {index}\n"
                    f"```mswea_bash_command\n{command}\n```"
                ),
            },
            {
                "role": "user",
                "content": f"<returncode>0</returncode>\n<output>{output}</output>",
            },
        ))
    read_metadata = {
        "return_code": 0,
        "cwd": "/workspace/repo",
        "environment_fingerprint": "env-sha256",
        "resource_version_fingerprints": {"foo.py": "file-sha256"},
        "output_complete": True,
        "timed_out": False,
        "output_truncated": False,
        "tool_semantics": {
            "category": "filesystem_read",
            "provenance": "runtime_traced",
            "complete": True,
            "effects": [{
                "kind": "read",
                "resource_id": "foo.py",
                "resource_version_fingerprint": "file-sha256",
            }],
        },
    }
    _write_receipt(tmp_path, 0, commands[0], read_metadata)
    _write_receipt(tmp_path, 1, commands[1], read_metadata)
    _write_receipt(tmp_path, 2, commands[2], {
        "return_code": 0, "output_complete": True,
        "timed_out": False, "output_truncated": False,
    })

    result = transform_autonomous_payload(
        {"model": "locked-model", "messages": messages},
        AutonomousSelectionConfig(
            policy="dag_certified_exclusion",
            negative_realization=NegativeRealizationMode.PROTOCOL_STUB,
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
        ),
        instrumentation_root=tmp_path,
    )

    visible = "\n".join(row["content"] for row in result.payload["messages"])
    assert "read attempt 0" in visible
    assert "read attempt 1" in visible
    assert visible.count(old_output.rstrip()) == 1
    assert "<returncode>0</returncode>\n<output></output>" in visible
    assert result.trace["excluded_causal_group_count"] == 1
    assert result.trace["receipt_count"] == 1
    assert result.trace["exclusions"][0]["rule_id"] == (
        "TRACE_EXACT_OPERATION_RESULT_V1"
    )

    budgeted = transform_autonomous_payload(
        {"model": "locked-model", "messages": messages},
        AutonomousSelectionConfig(
            policy="dag_certified_progress_spine",
            budget_fraction=0.9,
            negative_realization=NegativeRealizationMode.PROTOCOL_STUB,
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
        ),
        instrumentation_root=tmp_path,
    )
    assert budgeted.trace["budget_interpretation"] == (
        "certified_exclusion_then_retention_floor"
    )
    assert budgeted.trace["budget_satisfied"] is True
    assert budgeted.trace["certified_exclusion_underfill_tokens"] > 0
    assert budgeted.trace["unexplained_floor_underfill_tokens"] == 0
    assert budgeted.trace["excluded_causal_group_count"] == 1


def test_head_tail_recency_uses_retention_floor_without_dag_rules():
    result = transform_autonomous_payload(
        _payload(),
        AutonomousSelectionConfig(
            policy="head_tail_recency", budget_fraction=.6,
            protected_head_turns=1, protected_tail_turns=1,
            expected_model="locked-model", require_exact_sidecars=False,
        ),
    )
    assert result.trace["budget_interpretation"] == "retention_floor_round_up"
    assert result.trace["excluded_causal_group_count"] == 0
    assert result.trace["selected_tokens"] >= result.trace["requested_budget_tokens"]


def test_head_tail_progress_spine_keeps_middle_mutation_bundle():
    messages = [
        {"role": "system", "content": "Use one Bash command per turn."},
        {"role": "user", "content": "Fix src/example.py."},
    ]
    commands = (
        "grep -n target src/example.py",
        "nano src/example.py",
        "vi src/example.py",
        "sed -i 's/old/new/' src/example.py",
        "git diff -- src/example.py > patch.txt",
        "cat patch.txt",
        "echo waiting",
        "cat patch.txt",
    )
    for index, command in enumerate(commands):
        messages.extend((
            {
                "role": "assistant",
                "content": (
                    "THOUGHT: continue\n```mswea_bash_command\n"
                    f"{command}\n```"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<returncode>0</returncode>\n"
                    f"<output>observation {index} " + "detail " * 40 + "</output>"
                ),
            },
        ))

    result = transform_autonomous_payload(
        {"model": "locked-model", "messages": messages},
        AutonomousSelectionConfig(
            policy="head_tail_progress_spine",
            budget_fraction=.6,
            protected_head_turns=2,
            protected_tail_turns=2,
            expected_model="locked-model",
            require_exact_sidecars=False,
        ),
    )

    assert "turn:t0003" in result.trace["selected_causal_group_ids"]
    mutation_ids = {"m8", "m9"}
    assert mutation_ids <= set(result.trace["selected_record_ids"])
    assert result.trace["selection_reasons"]["m8"] == (
        "role_floor:mutation"
    )
    assert result.trace["selection_reasons"]["m9"] == (
        "role_floor:mutation"
    )
    assert result.trace["budget_interpretation"] == "retention_floor_round_up"


def test_sidecar_sequence_mismatch_fails_closed(tmp_path):
    payload = _payload()
    for step, command in enumerate(("cat a.py", "cat WRONG.py", "cat c.py", "cat d.py")):
        _write_receipt(tmp_path, step, command, {
            "return_code": 0,
            "output_complete": True,
            "resource_version_fingerprints": {command.split()[-1]: "v1"},
        })
    result = transform_autonomous_payload(
        payload,
        AutonomousSelectionConfig(
            policy="h3_read_superseded",
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
        ),
        instrumentation_root=tmp_path,
    )
    assert result.trace["instrumentation_sidecar_join"]["status"] == (
        "missing_or_mismatched_receipts"
    )
    assert result.trace["instrumentation_sidecar_join"]["joined"] == 0
    assert result.trace["selection_abstained_for_sidecar"] is True
    assert result.trace["excluded_causal_group_count"] == 0


class _Upstream:
    def __init__(self, statuses=None) -> None:
        self.requests: list[dict] = []
        self.client_ports: list[int] = []
        self.qualification_requests = 0
        self.statuses = list(statuses or ())
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802
                outer.client_ports.append(self.client_address[1])
                outer.qualification_requests += 1
                body = b'{"object":"list","data":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                outer.client_ports.append(self.client_address[1])
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(json.loads(body))
                response = json.dumps({
                    "id": "response-1",
                    "pra": {
                        "selected_history_reencoded_tokens": 0,
                        "selected_history_kv_copy_bytes": 0,
                    },
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "```mswea_bash_command\ncat a.py\n```",
                        }
                    }],
                }).encode()
                status = outer.statuses.pop(0) if outer.statuses else 200
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_proxy_normalizes_completion_limit_to_portable_wire_key(tmp_path):
    upstream = _Upstream()
    proxy = AutonomousSelectionProxy(
        upstream.url,
        config=AutonomousSelectionConfig(
            policy="full",
            expected_model="locked-model",
            fill_missing_generation_parameters=True,
            max_completion_tokens=1024,
            max_calls=1,
        ),
        trace_path=tmp_path / "trace.jsonl",
    )
    payload = _payload()
    payload["max_completion_tokens"] = 1024
    url = proxy.start()
    try:
        assert _post(f"{url}/chat/completions", payload)[0] == 200
    finally:
        proxy.close()
        upstream.close()

    assert upstream.requests[0]["max_tokens"] == 1024
    assert "max_completion_tokens" not in upstream.requests[0]


def test_proxy_optionally_captures_unmodified_first_request(tmp_path):
    upstream = _Upstream()
    capture = tmp_path / "first-request.json"
    proxy = AutonomousSelectionProxy(
        upstream.url,
        config=AutonomousSelectionConfig(
            policy="full", expected_model="locked-model", max_calls=1,
        ),
        trace_path=tmp_path / "trace.jsonl",
        first_request_capture_path=capture,
        request_content_normalizations=((r"Message time: [^\n]+", "Message time: FROZEN"),),
    )
    payload = _payload()
    payload["messages"][-1]["content"] += "\nMessage time: 2026-09-23T00:46:17Z"
    url = proxy.start()
    try:
        assert _post(f"{url}/chat/completions", payload)[0] == 200
    finally:
        proxy.close()
        upstream.close()

    assert json.loads(capture.read_text()) == payload
    assert upstream.requests[0]["messages"][-1]["content"].endswith(
        "Message time: FROZEN"
    )


def test_proxy_forwards_ordinary_selected_text_and_logs_reacquisition(tmp_path):
    upstream = _Upstream()
    trace = tmp_path / "trace.jsonl"
    proxy = AutonomousSelectionProxy(
        upstream.url,
        config=AutonomousSelectionConfig(
            policy="h4_working_set",
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
            working_set_resources=2,
            max_calls=1,
            task_id="task-1",
            require_exact_sidecars=False,
        ),
        trace_path=trace,
    )
    url = proxy.start()
    try:
        status, response = _post(f"{url}/chat/completions", _payload())
        assert status == 200
        assert response["id"] == "response-1"
        forwarded = upstream.requests[0]
        assert "pra" not in forwarded
        assert all(set(row) == {"role", "content"} for row in forwarded["messages"])
        assert "cat a.py" not in "\n".join(row["content"] for row in forwarded["messages"])
        row = json.loads(trace.read_text().strip())
        assert row["assistant_content_sha256"] == hashlib.sha256(
            b"```mswea_bash_command\ncat a.py\n```"
        ).hexdigest()
        assert row["response_sha256_scope"] == (
            "raw_http_body_includes_volatile_response_metadata"
        )
        assert len(row["request_message_content_sha256"]) == len(_payload()["messages"])
        assert row["reacquired_excluded_resources"] == ["a.py"]
        assert row["reacquisition_proxy_for_false_exclusion"] is True

        status, error = _post(f"{url}/chat/completions", _payload())
        assert status == 429
        assert error["error"] == "max_model_calls_exceeded"
        assert len(upstream.requests) == 1
    finally:
        proxy.close()
        upstream.close()


def test_proxy_delivers_exact_wire_plan_to_native_builder(tmp_path):
    upstream = _Upstream()
    trace = tmp_path / "trace.jsonl"
    builder_calls = []

    def native_builder(payload, **kwargs):
        builder_calls.append((payload, kwargs))
        transformed = dict(payload)
        transformed["native_bridge_marker"] = True
        return transformed

    proxy = AutonomousSelectionProxy(
        upstream.url,
        config=AutonomousSelectionConfig(
            policy="full", expected_model="locked-model", max_calls=1,
            task_id="task-1", session_id="session-1",
        ),
        trace_path=trace,
        native_request_builder=native_builder,
    )
    url = proxy.start()
    try:
        assert _post(f"{url}/chat/completions", _payload())[0] == 200
        assert upstream.requests[0]["native_bridge_marker"] is True
        logical_payload, kwargs = builder_calls[0]
        assert logical_payload["messages"] == _payload()["messages"]
        assert kwargs["session_id"] == "session-1"
        assert kwargs["wire_plan"]["selected_record_ids"] == [
            f"m{index}" for index in range(len(_payload()["messages"]))
        ]
        assert kwargs["record_message_indices"] == {
            f"m{index}": index for index in range(len(_payload()["messages"]))
        }
        assert set(kwargs["mandatory_message_indices"]).issubset(
            kwargs["record_message_indices"].values()
        )
        row = json.loads(trace.read_text(encoding="utf-8").strip())
        assert row["native_pra_delivery"] is True
        assert row["engine_pra_metrics"] == {
            "selected_history_reencoded_tokens": 0,
            "selected_history_kv_copy_bytes": 0,
        }
        assert proxy.health()["kv_metrics_available"] is True
        assert proxy.health()["successful_request_count"] == 1
    finally:
        proxy.close()
        upstream.close()


def test_native_builder_sequence_advances_only_after_success(tmp_path):
    upstream = _Upstream(statuses=[500, 200, 200])
    trace = tmp_path / "trace.jsonl"
    builder_indexes = []

    def native_builder(payload, **kwargs):
        builder_indexes.append(kwargs["request_index"])
        return dict(payload)

    proxy = AutonomousSelectionProxy(
        upstream.url,
        config=AutonomousSelectionConfig(
            policy="full", expected_model="locked-model", max_calls=3,
            task_id="task-1", session_id="session-1",
        ),
        trace_path=trace,
        native_request_builder=native_builder,
    )
    url = proxy.start()
    try:
        assert _post(f"{url}/chat/completions", _payload())[0] == 500
        assert _post(f"{url}/chat/completions", _payload())[0] == 200
        assert _post(f"{url}/chat/completions", _payload())[0] == 200
        assert builder_indexes == [1, 1, 2]
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        assert [row["request_index"] for row in rows] == [1, 2, 3]
        assert [row["native_request_index"] for row in rows] == [1, 1, 2]
        assert proxy.health()["successful_request_count"] == 2
    finally:
        proxy.close()
        upstream.close()


def test_proxy_charges_the_next_observation_to_a_reacquisition(tmp_path):
    upstream = _Upstream()
    trace = tmp_path / "trace.jsonl"
    proxy = AutonomousSelectionProxy(
        upstream.url,
        config=AutonomousSelectionConfig(
            policy="h4_working_set",
            expected_model="locked-model",
            protected_head_turns=0,
            protected_tail_turns=1,
            working_set_resources=2,
            max_calls=2,
            require_exact_sidecars=False,
        ),
        trace_path=trace,
    )
    url = proxy.start()
    try:
        first = _payload()
        assert _post(f"{url}/chat/completions", first)[0] == 200
        second = _payload()
        second["messages"].extend((
            {
                "role": "assistant",
                "content": "```mswea_bash_command\ncat a.py\n```",
            },
            {
                "role": "user",
                "content": "<returncode>0</returncode>\n<output>one two three</output>",
            },
        ))
        assert _post(f"{url}/chat/completions", second)[0] == 200
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        assert rows[1]["previous_reacquisition_count"] == 1
        assert rows[1]["previous_reacquisition_resources"] == ["a.py"]
        assert rows[1]["reacquired_observation_tokens_from_previous_action"] > 0
    finally:
        proxy.close()
        upstream.close()


def test_proxy_logs_transport_failure_at_the_reserved_request_index(tmp_path):
    trace = tmp_path / "trace.jsonl"
    proxy = AutonomousSelectionProxy(
        "http://127.0.0.1:1/v1",
        config=AutonomousSelectionConfig(
            policy="full", expected_model="locked-model", max_calls=1,
        ),
        trace_path=trace,
        timeout_seconds=1,
    )
    url = proxy.start()
    try:
        status, error = _post(f"{url}/chat/completions", _payload())
        assert status == 502
        assert error["error"] == "selection_upstream_unavailable"
        row = json.loads(trace.read_text(encoding="utf-8").strip())
        assert row["request_index"] == 1
        assert row["upstream_status"] == 0
        assert row["upstream_transport_error"] == "URLError"
        assert row["assistant_command_sha256"] is None
        assert proxy.upstream_failed is True
        assert proxy.upstream_failure_detail["error_type"] == "URLError"
        assert proxy.health()["upstream_failed"] is True
        summary = summarize_trace(trace)
        assert summary["upstream_error_calls"] == 1
        assert summary["upstream_transport_error_calls"] == 1
    finally:
        proxy.close()


def test_proxy_qualifies_and_reuses_connection_without_replaying_posts(tmp_path):
    upstream = _Upstream()
    trace = tmp_path / "trace.jsonl"
    proxy = AutonomousSelectionProxy(
        upstream.url,
        config=AutonomousSelectionConfig(
            policy="full", expected_model="locked-model", max_calls=2,
        ),
        trace_path=trace,
        upstream_qualification_path="/api/version",
        upstream_connect_attempts=3,
        upstream_connect_retry_seconds=0,
    )
    url = proxy.start()
    try:
        assert _post(f"{url}/chat/completions", _payload())[0] == 200
        assert _post(f"{url}/chat/completions", _payload())[0] == 200
        assert upstream.qualification_requests == 1
        assert len(upstream.requests) == 2
        assert len(set(upstream.client_ports)) == 1
        assert len(trace.read_text(encoding="utf-8").splitlines()) == 2
    finally:
        proxy.close()
        upstream.close()


def test_proxy_curl_transport_sends_each_model_request_once(tmp_path):
    calls = []

    class Completed:
        returncode = 0
        stderr = b""
        stdout = json.dumps({
            "id": "response-1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "```mswea_bash_command\ncat a.py\n```",
                }
            }],
        }).encode() + b"\n200"

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    trace = tmp_path / "trace.jsonl"
    proxy = AutonomousSelectionProxy(
        "http://engine.test:11435/v1",
        config=AutonomousSelectionConfig(
            policy="full", expected_model="locked-model", max_calls=1,
        ),
        trace_path=trace,
        upstream_curl_executable="/usr/bin/curl",
        upstream_connect_attempts=4,
        upstream_connect_retry_seconds=2.2,
        curl_runner=runner,
    )
    url = proxy.start()
    try:
        assert _post(f"{url}/chat/completions", _payload())[0] == 200
        assert len(calls) == 1
        command, kwargs = calls[0]
        assert command.count("--data-binary") == 1
        assert command[command.index("--retry") + 1] == "3"
        assert "--retry-connrefused" in command
        assert command[command.index("--retry-delay") + 1] == "3"
        assert "--retry-all-errors" not in command
        assert command[-1] == "http://engine.test:11435/v1/chat/completions"
        assert json.loads(kwargs["input"])["model"] == "locked-model"
        assert kwargs["check"] is False
        assert len(trace.read_text(encoding="utf-8").splitlines()) == 1
    finally:
        proxy.close()


def test_locked_task_selection_and_agent_command_are_single_task(tmp_path):
    ids = ["org__repo-1", "org__repo-2"]
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps({
        "dataset": "org/dataset",
        "split": "test",
        "expected_count": 2,
        "canonical_ids_sha256": digest,
        "instance_ids": ids,
    }))
    _, selected, index = load_locked_task(
        card_path, task_index=2, instance_id=None
    )
    assert (selected, index) == ("org__repo-2", 2)
    with pytest.raises(ValueError, match="exactly one"):
        load_locked_task(card_path, task_index=None, instance_id=None)

    args = argparse.Namespace(
        split="test",
        served_model="locked-model",
        scaffold="swebench_backticks.yaml",
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_completion_tokens=128,
        upstream_timeout_seconds=1200,
        docker_executable=None,
        instrument_observations=True,
        instrumentation_output_root=tmp_path / "instrumentation",
        max_calls=12,
        require_unified_diff_submission=True,
    )
    command = build_agent_command(
        args,
        proxy_base_url="http://127.0.0.1:1234/v1",
        instance_id=selected,
        agent_output=tmp_path / "agent",
    )
    joined = " ".join(str(row) for row in command)
    assert r"(org__repo\-2)" in joined
    assert "model.model_kwargs.temperature=0.0" in joined
    assert "model.model_kwargs.seed=0" in joined
    assert "model.model_kwargs.timeout=1200" in joined
    assert "agent.step_limit=12" in joined
    assert "environment.pull_timeout=900" in joined
    assert "InstrumentedDockerEnvironment" in joined
    assert "environment.require_unified_diff_submission=true" in joined
    assert (
        "environment.image=docker.io/swebench/"
        "sweb.eval.x86_64.org_1776_repo-2:latest" in joined
    )


def test_auxiliary_grader_command_uses_separate_predictions_and_run_id(tmp_path):
    args = argparse.Namespace(split="test", run_id="primary", grader_workers=1)
    predictions = tmp_path / "auxiliary_workspace_state_preds.json"
    command = build_grader_command(
        args,
        dataset="org/dataset",
        instance_id="org__repo-1",
        predictions=predictions,
        output=tmp_path,
        run_id="primary-aux-workspace-state",
    )

    assert command[command.index("-p") + 1] == str(predictions)
    assert command[command.index("--run_id") + 1] == (
        "primary-aux-workspace-state"
    )
    assert command[command.index("--cache_level") + 1] == "instance"
    assert command[2] == (
        "experiments.paper8_5_agent_memory.swebench_grader_entrypoint"
    )


def test_execution_environment_binds_repository_and_src_layout(tmp_path):
    args = argparse.Namespace(
        pythonpath=[], docker_executable=None, docker_platform=None,
    )

    environment = _execution_environment(args)
    python_paths = environment["PYTHONPATH"].split(os.pathsep)
    repository = str(Path(__file__).resolve().parents[1])

    assert python_paths[:2] == [repository, str(Path(repository) / "src")]


def test_summary_counts_actions_reacquisition_and_repeated_categories(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = []
    for operation, resources, reacquired in (
        ("read", ["a.py"], []),
        ("read", ["a.py"], ["a.py"]),
        ("search_discovery", [], []),
        ("search_discovery", [], []),
        ("verify", ["tests/test_a.py"], []),
    ):
        rows.append({
            "full_tokens": 100,
            "selected_tokens": 90,
            "materialized_tokens": 80,
            "assistant_command_sha256": "same" if not resources else operation,
            "assistant_operation": operation,
            "assistant_resource_ids": resources,
            "assistant_is_search": operation == "search_discovery",
            "assistant_is_read": operation == "read",
            "assistant_is_test": operation == "verify",
            "reacquisition_count": len(reacquired),
            "reacquired_observation_tokens_from_previous_action": (
                17 if reacquired else 0
            ),
            "excluded_causal_group_count": 1,
            "excluded_tokens": 10,
            "budget_satisfied": True,
            "upstream_status": 200,
            "reported_completion_tokens": 7,
            "generation": {"max_completion_tokens": 8},
        })
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = summarize_trace(trace)
    assert summary["calls"] == 5
    assert summary["actions"] == 5
    assert summary["repeated_same_operation_resource_counts"] == {
        "search": 1, "read": 1, "test": 0
    }
    assert summary["reacquisition_events"] == 1
    assert summary["reacquired_observation_tokens"] == 17
    assert summary["cumulative_logical_retention_fraction"] == pytest.approx(0.9)
    assert summary["reported_usage_coverage_calls"] == 5
    assert summary["cumulative_reported_completion_tokens"] == 35
    assert summary["maximum_reported_completion_tokens"] == 7
    assert summary["completion_limit_coverage_calls"] == 5
    assert summary["completion_limit_violation_calls"] == 0
    assert summary["completion_limit_respected"] is True
    assert summary["cumulative_materialized_plus_reported_completion_tokens"] == 435


def test_summary_rejects_reported_completion_limit_overshoot(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({
        "reported_completion_tokens": 1025,
        "generation": {"max_completion_tokens": 1024},
    }) + "\n")

    summary = summarize_trace(trace)

    assert summary["completion_limit_coverage_calls"] == 1
    assert summary["completion_limit_violation_calls"] == 1
    assert summary["completion_limit_respected"] is False


def test_summary_aggregates_native_engine_metrics_without_imputing_missing_values(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "full_tokens": 100,
            "selected_tokens": 100,
            "materialized_tokens": 100,
            "budget_satisfied": True,
            "upstream_status": 200,
            "native_pra_delivery": True,
            "engine_pra_metrics": {
                "native_kv": True,
                "full_retention": True,
                "realized_retention_fraction": 1.0,
                "selected_kv_tokens": 80,
                "wire_tokens": 20,
                "selected_history_reencoded_tokens": 0,
                "selected_text_reencoded_tokens": 0,
                "physical_kv_copy_bytes": 0,
                "total_kv_copy_bytes": 0,
                "host_to_device_bytes": 0,
                "source_bootstrap": True,
                "source_bootstrap_tokens": 100,
                "source_bootstrap_cached_tokens": 0,
                "source_bootstrap_evaluated_tokens": 100,
                "consumer_temporary_bytes": 128,
                "consumer_temporary_peak_bytes": 96,
            },
        },
        {
            "full_tokens": 200,
            "selected_tokens": 100,
            "materialized_tokens": 100,
            "budget_satisfied": True,
            "upstream_status": 200,
            "native_pra_delivery": True,
            "engine_pra_metrics": {
                "native_kv": True,
                "full_retention": False,
                "realized_retention_fraction": 0.5,
                "selected_kv_tokens": 90,
                "wire_tokens": 10,
                "selected_history_reencoded_tokens": 0,
                "physical_kv_copy_bytes": 0,
                "total_kv_copy_bytes": 16,
                "consumer_temporary_bytes": 256,
                "consumer_temporary_peak_bytes": 192,
            },
        },
        {
            "full_tokens": 50,
            "selected_tokens": 50,
            "materialized_tokens": 50,
            "budget_satisfied": True,
            "upstream_status": 200,
            "native_pra_delivery": False,
            "engine_pra_metrics": {},
        },
    ]
    trace.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    summary = summarize_trace(trace)

    assert summary["schema_version"] == 2
    assert summary["native_pra_delivery_calls"] == 2
    assert summary["engine_pra_metrics_coverage_calls"] == 2
    assert summary["native_kv_calls"] == 2
    assert summary["full_retention_calls"] == 1
    assert summary["mean_realized_retention_fraction"] == pytest.approx(0.75)
    assert summary["minimum_realized_retention_fraction"] == pytest.approx(0.5)
    assert summary["maximum_realized_retention_fraction"] == pytest.approx(1.0)
    assert summary["cumulative_selected_kv_tokens"] == 170
    assert summary["cumulative_wire_tokens"] == 30
    assert summary["cumulative_selected_history_reencoded_tokens"] == 0
    assert summary["cumulative_physical_kv_copy_bytes"] == 0
    assert summary["cumulative_total_kv_copy_bytes"] == 16
    assert summary["cumulative_host_to_device_bytes"] == 0
    assert summary["source_bootstrap_calls"] == 1
    assert summary["source_bootstrap_coverage_calls"] == 1
    assert summary["cumulative_source_bootstrap_tokens"] == 100
    assert summary["cumulative_source_bootstrap_cached_tokens"] == 0
    assert summary["cumulative_source_bootstrap_evaluated_tokens"] == 100
    assert summary["cumulative_consumer_temporary_bytes"] == 384
    assert summary["maximum_consumer_temporary_peak_bytes"] == 192
