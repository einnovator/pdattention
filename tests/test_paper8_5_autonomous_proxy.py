from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

from experiments.paper8_5_agent_memory.autonomous_proxy import (
    AutonomousSelectionConfig,
    AutonomousSelectionProxy,
    transform_autonomous_payload,
)
from experiments.paper8_5_agent_memory.run_autonomous_swebench import (
    build_agent_command,
    load_locked_task,
    summarize_trace,
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
    def __init__(self) -> None:
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = b'{"object":"list","data":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append(json.loads(body))
                response = json.dumps({
                    "id": "response-1",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "```mswea_bash_command\ncat a.py\n```",
                        }
                    }],
                }).encode()
                self.send_response(200)
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
        assert row["reacquired_excluded_resources"] == ["a.py"]
        assert row["reacquisition_proxy_for_false_exclusion"] is True

        status, error = _post(f"{url}/chat/completions", _payload())
        assert status == 429
        assert error["error"] == "max_model_calls_exceeded"
        assert len(upstream.requests) == 1
    finally:
        proxy.close()
        upstream.close()


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
        docker_executable=None,
        instrument_observations=True,
        instrumentation_output_root=tmp_path / "instrumentation",
        max_calls=12,
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
    assert "agent.step_limit=12" in joined
    assert "InstrumentedDockerEnvironment" in joined
    assert (
        "environment.image=docker.io/swebench/"
        "sweb.eval.x86_64.org_1776_repo-2:latest" in joined
    )


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
            "excluded_causal_group_count": 1,
            "excluded_tokens": 10,
            "budget_satisfied": True,
            "upstream_status": 200,
        })
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = summarize_trace(trace)
    assert summary["calls"] == 5
    assert summary["actions"] == 5
    assert summary["repeated_same_operation_resource_counts"] == {
        "search": 1, "read": 1, "test": 0
    }
    assert summary["reacquisition_events"] == 1
    assert summary["cumulative_logical_retention_fraction"] == pytest.approx(0.9)
