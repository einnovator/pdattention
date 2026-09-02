"""Coding-agent campaign schemas, isolation, and paired analysis."""

from __future__ import annotations

import gzip
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
