"""Coding-agent campaign schemas, isolation, and paired analysis."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import yaml

from experiments.agents.adapters import FixtureAgentAdapter, command_adapter, load_command_manifest
from experiments.agents.adapters.command import _parse_json_lines
from experiments.agents.cli import build_parser
from experiments.agents.analysis import paired_comparison, summarize
from experiments.agents.catalog import load_catalog
from experiments.agents.runner import (
    _read_fixture_text,
    _write_text_artifact,
    external_plan,
    import_harbor_job,
    load_runs,
    run_manifest,
)
from experiments.agents.schema import (
    AgentEngineMatrix,
    BenchmarkManifest,
    CodingAgentRun,
    HardwareManifest,
    PRAProfile,
    PRAMode,
    ProtocolManifest,
    ResourceMetrics,
)


ROOT = Path(__file__).parents[1] / "experiments" / "agents"


def test_agent_catalog_has_required_audited_families() -> None:
    catalog = load_catalog(ROOT / "agent_catalog.yaml")
    slugs = {agent.slug for agent in catalog.agents}
    assert {"claude-code", "codex", "gemini-cli", "opencode", "pi", "openhands", "deepseek-harness", "aider", "swe-agent"} <= slugs
    deepseek = next(agent for agent in catalog.agents if agent.slug == "deepseek-harness")
    assert any("not official DeepSeek" in value for value in deepseek.limitations)


def test_support_and_machine_manifests_are_strict_and_complete() -> None:
    matrix = AgentEngineMatrix.model_validate(yaml.safe_load((ROOT / "engine_matrix.yaml").read_text()))
    hardware = HardwareManifest.model_validate(yaml.safe_load((ROOT / "hardware_manifest.yaml").read_text()))
    protocols = ProtocolManifest.model_validate(yaml.safe_load((ROOT / "protocol_matrix.yaml").read_text()))
    assert set(matrix.agents) == {agent.slug for agent in load_catalog(ROOT / "agent_catalog.yaml").agents}
    assert {host.host for host in hardware.hosts} >= {"192.168.1.86", "192.168.1.95", "192.168.1.102"}
    assert {row.protocol for row in protocols.protocols} >= {"openai-chat-completions", "openai-responses"}


def test_verified_command_specs_are_ephemeral_and_machine_safe() -> None:
    commands = {row.slug: row for row in load_command_manifest(ROOT / "agent_commands.yaml").commands}
    assert {"codex", "gemini-cli", "opencode", "pi"} <= set(commands)
    assert "--ephemeral" in commands["codex"].run_args
    assert "--no-session" in commands["pi"].run_args
    assert all(row.external_sandbox_required for row in commands.values())
    assert command_adapter("opencode", ROOT / "agent_commands.yaml").executable == "opencode"
    with pytest.raises(RuntimeError, match="not verified"):
        command_adapter("openhands", ROOT / "agent_commands.yaml")


def test_codex_jsonl_usage_and_nested_actions_are_normalized_once() -> None:
    usage, behavior = _parse_json_lines("\n".join((
        '{"type":"item.completed","item":{"type":"file_change","changes":[{"path":"a"}]}}',
        '{"type":"item.completed","item":{"type":"command_execution","command":"python -m pytest"}}',
        '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":80,"output_tokens":9}}',
    )))
    assert usage == {"input_tokens": 100, "output_tokens": 9, "cached_input_tokens": 80}
    assert behavior["model_calls"] == behavior["turns"] == 1
    assert behavior["tool_calls"] == 2
    assert behavior["shell_calls"] == behavior["file_writes"] == behavior["tests"] == 1


def test_opencode_and_pi_usage_and_tools_are_normalized() -> None:
    usage, behavior = _parse_json_lines("\n".join((
        '{"type":"tool_use","part":{"type":"tool","tool":"write"}}',
        '{"type":"step_finish","part":{"tokens":{"input":100,"output":9,"cache":{"read":20}}}}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"toolCall","name":"bash"}],"usage":{"input":50,"output":7,"cacheRead":10}}}',
    )))
    assert usage == {"input_tokens": 150, "output_tokens": 16, "cached_input_tokens": 30}
    assert behavior["model_calls"] == behavior["turns"] == 2
    assert behavior["tool_calls"] == 2
    assert behavior["file_writes"] == behavior["shell_calls"] == 1


def test_external_agent_arguments_preserve_codex_provider_configuration() -> None:
    args = build_parser().parse_args([
        "smoke-agent", "--agent", "codex", "--model", "qwen3-14b-pra",
        "--output", "out", "--agent-arg=-c",
        "--agent-arg=model_provider=\"pra_gateway\"",
    ])
    adapter = command_adapter("codex", ROOT / "agent_commands.yaml", extra_args=tuple(args.agent_arg))
    adapter.configure_provider(model=args.model, environment={"OPENAI_API_KEY": "secret"})
    adapter.cleanup()
    assert adapter.extra_args == ("-c", 'model_provider="pra_gateway"')
    assert adapter.provider["model"] == "qwen3-14b-pra"
    assert adapter.provider["environment"]["OPENAI_API_KEY"] == "secret"


@pytest.mark.parametrize("name", [
    "fixture_gateway_smoke.yaml",
    "terminal_bench_smoke.yaml", "terminal_bench_pilot.yaml", "terminal_bench_main.yaml",
    "swebench_lite_smoke.yaml", "swebench_lite_main.yaml", "swebench_verified_main.yaml",
])
def test_frozen_manifest_is_valid(name: str) -> None:
    manifest = BenchmarkManifest.load(ROOT / "manifests" / name)
    assert manifest.task_ids
    assert len(manifest.task_ids) == len(set(manifest.task_ids))


def test_terminal_bench_plan_filters_every_frozen_task() -> None:
    manifest = BenchmarkManifest.load(ROOT / "manifests" / "terminal_bench_smoke.yaml")
    command = external_plan(manifest, agent="opencode", model="model", condition="no-pra")["official_harness_command"]
    selected = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--include-task-name"
    ]
    assert selected == [f"terminal-bench/{task_id}" for task_id in manifest.task_ids]
    assert command[command.index("--agent-kwarg") + 1] == "version=1.18.26"


def test_swebench_plan_filters_every_frozen_task_and_timeout() -> None:
    manifest = BenchmarkManifest.load(ROOT / "manifests" / "swebench_lite_smoke.yaml")
    command = external_plan(manifest, agent="pi", model="model", condition="selected-balanced")["official_harness_command"]
    selected = [command[index + 1] for index, value in enumerate(command) if value == "-i"]
    assert selected == list(manifest.task_ids)
    assert command[command.index("--timeout") + 1] == str(manifest.timeout_seconds)


def test_fixture_campaign_writes_valid_isolated_results(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.load(ROOT / "manifests" / "fixture_smoke.yaml")
    rows = run_manifest(
        manifest, FixtureAgentAdapter(), output=tmp_path,
        agent="fixture-agent", engine="fixture", model="fixture-model",
        pra_mode=PRAMode.NONE, pra_profile=PRAProfile.NONE,
    )
    restored = load_runs(tmp_path / "runs.jsonl")
    assert len(rows) == len(restored) == 4
    assert all(row.outcome.success for row in restored)
    assert all(row.metadata["harness_validation_only"] for row in restored)
    assert summarize(restored)["none:none"]["task_success_rate"] == 1.0


def test_fixture_text_accepts_bom_text_and_rejects_invalid_bytes(tmp_path: Path) -> None:
    utf16 = tmp_path / "utf16.txt"
    utf16.write_text("PRA\n", encoding="utf-16")
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xffbroken")
    assert _read_fixture_text(utf16) == "PRA\n"
    assert _read_fixture_text(invalid) is None


def test_large_raw_transcript_is_compressed(tmp_path: Path) -> None:
    path = _write_text_artifact(tmp_path / "run.stdout", "abcd", compress_at=4)
    assert path.name == "run.stdout.gz"
    assert gzip.open(path, "rt", encoding="utf-8").read() == "abcd"


def test_official_harbor_result_is_normalized_without_regrading(tmp_path: Path) -> None:
    job = tmp_path / "harbor-job"
    trial = job / "task__abc"
    trajectory = trial / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(json.dumps({"steps": [
        {"source": "agent", "llm_call_count": 1, "metrics": {"prompt_tokens": 12},
         "tool_calls": [{"function_name": "write", "arguments": {}}]},
    ]}), encoding="utf-8")
    verifier_dir = trial / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "ctrf.json").write_text(json.dumps({
        "results": {"summary": {"tests": 3, "passed": 3}}
    }), encoding="utf-8")
    (trial / "result.json").write_text(json.dumps({
        "id": "run-1", "task_name": "terminal-bench/filter-js-from-html",
        "task_checksum": "sha256", "started_at": "2026-09-02T00:00:00Z",
        "finished_at": "2026-09-02T00:00:02Z", "agent_execution": {
            "started_at": "2026-09-02T00:00:00Z", "finished_at": "2026-09-02T00:00:01Z",
        }, "verifier": {"started_at": "2026-09-02T00:00:01", "finished_at": "2026-09-02T00:00:02"},
        "agent_info": {"name": "opencode", "version": "1.18.26"},
        "agent_result": {"n_input_tokens": 12, "n_output_tokens": 3, "n_cache_tokens": 2},
        "verifier_result": {"rewards": {"reward": 1.0}}, "config": {"job_id": "job-1"},
        "verifier_environment_mode": "shared", "exception_info": None,
    }), encoding="utf-8")
    manifest = BenchmarkManifest.load(ROOT / "manifests" / "terminal_bench_smoke.yaml")
    rows = import_harbor_job(
        job, manifest, output=job, engine="ollama", engine_version="0.32.7",
        host="nvidia+mac", hardware={"engine_chip": "Apple M4 Pro"},
        model="qwen3:14b", quantization="Q4_K_M",
        pra_mode=PRAMode.NONE, pra_profile=PRAProfile.NONE,
        connection="gateway", protocol="openai-chat-completions",
    )
    assert len(rows) == 1
    assert rows[0].outcome.success
    assert rows[0].outcome.tests_passed == rows[0].outcome.tests_total == 3
    assert rows[0].behavior.file_writes == 1
    assert rows[0].tokens.max_context_tokens == 12
    assert rows[0].timings.task_wall_ms == 2000
    assert rows[0].identity.engine_version == "0.32.7"
    assert rows[0].identity.hardware["engine_chip"] == "Apple M4 Pro"
    assert rows[0].identity.quantization == "Q4_K_M"
    assert load_runs(job / "runs.jsonl") == rows


def test_harbor_import_reads_pi_jsonl_behavior(tmp_path: Path) -> None:
    job = tmp_path / "harbor-job"
    trial = job / "query-optimize__abc"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    events = [
        {"type": "message_end", "message": {
            "role": "assistant", "usage": {"input": 120},
        }},
        {"type": "tool_execution_start", "toolName": "read"},
        {"type": "tool_execution_start", "toolName": "write"},
    ]
    (agent / "pi.txt").write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    (trial / "result.json").write_text(json.dumps({
        "id": "run-pi", "task_name": "terminal-bench/query-optimize",
        "started_at": "2026-09-02T00:00:00Z",
        "finished_at": "2026-09-02T00:00:02Z",
        "agent_info": {"name": "pi", "version": "0.73.1"},
        "agent_result": {"n_input_tokens": 120, "n_output_tokens": 4},
        "verifier_result": {"rewards": {"reward": 0.0}},
        "exception_info": None,
    }), encoding="utf-8")
    manifest = BenchmarkManifest.load(ROOT / "manifests" / "terminal_bench_smoke.yaml")
    rows = import_harbor_job(
        job, manifest, output=job, engine="ollama", host="nvidia",
        model="qwen3:14b", pra_mode=PRAMode.NONE, pra_profile=PRAProfile.NONE,
        connection="gateway", protocol="openai-chat-completions",
    )

    assert rows[0].behavior.turns == rows[0].behavior.model_calls == 1
    assert rows[0].behavior.tool_calls == 2
    assert rows[0].behavior.file_reads == rows[0].behavior.file_writes == 1
    assert rows[0].tokens.max_context_tokens == 120
    assert Path(rows[0].artifacts["agent_log"]).parts[-2:] == ("agent", "pi.txt")


def test_non_native_result_cannot_claim_native_memory(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.load(ROOT / "manifests" / "fixture_smoke.yaml")
    row = run_manifest(
        manifest, FixtureAgentAdapter(), output=tmp_path,
        agent="fixture-agent", engine="fixture", model="fixture-model",
        pra_mode=PRAMode.NONE, pra_profile=PRAProfile.NONE,
    )[0]
    value = row.model_dump()
    value["resources"] = ResourceMetrics(native_resources=1, native_active_bytes=16).model_dump()
    with pytest.raises(ValueError, match="non-native"):
        CodingAgentRun.model_validate(value)


def test_paired_analysis_keeps_task_identity(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.load(ROOT / "manifests" / "fixture_smoke.yaml")
    baseline = run_manifest(
        manifest, FixtureAgentAdapter(), output=tmp_path / "base",
        agent="fixture-agent", engine="fixture", model="fixture-model",
        pra_mode=PRAMode.NONE, pra_profile=PRAProfile.NONE,
    )
    selected = run_manifest(
        manifest, FixtureAgentAdapter(), output=tmp_path / "selected",
        agent="fixture-agent", engine="fixture", model="fixture-model",
        pra_mode=PRAMode.SELECTED_CONTEXT, pra_profile=PRAProfile.BALANCED,
    )
    comparison = paired_comparison(baseline, selected)
    assert comparison["pairs"] == 4
    assert comparison["wins"] == comparison["losses"] == 0
    assert comparison["mcnemar_exact_p"] == 1.0
