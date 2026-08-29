from __future__ import annotations

import json

import pytest

from pra_hf.agent_profiles import AgentProfileRegistry, load_mcp_config


def test_agent_profile_default_resolution_and_cli_override(tmp_path) -> None:
    config = tmp_path / "agents.yaml"
    config.write_text(
        """version: 1
default_profile: work
profiles:
  work:
    model: org/base
    runtime:
      mode: gateway
      engine: vllm
      endpoint: http://host:8000
    pra:
      profile: BALANCED
    tools:
      approval: ask
      candidates: 12
    generation:
      max_new_tokens: 2048
""",
        encoding="utf-8",
    )

    profile, sources = AgentProfileRegistry().resolve(
        config_path=config,
        overrides={"model": "org/override", "pra": {"profile": "ECONOMY"}},
    )

    assert profile.name == "work"
    assert profile.model == "org/override"
    assert profile.runtime_mode == "gateway"
    assert profile.runtime.endpoint == "http://host:8000"
    assert profile.pra == "ECONOMY"
    assert profile.tools.candidates == 12
    assert profile.max_new_tokens == 2048
    assert str(config.resolve()) in sources


def test_credentials_are_references_and_never_rendered(tmp_path) -> None:
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text("token: top-secret\n", encoding="utf-8")
    config = tmp_path / "agents.yaml"
    config.write_text(
        f"version: 1\ndefault_profile: remote\nprofiles:\n  remote:\n    credentials:\n      file: {credentials.as_posix()}\n",
        encoding="utf-8",
    )

    profile = AgentProfileRegistry().load(config_path=config).resolve()
    rendered = json.dumps(profile.redacted_dict())

    assert "top-secret" not in rendered
    assert "referenced, not loaded" in rendered


def test_mcp_external_then_inline_merge(tmp_path) -> None:
    mcp = tmp_path / "mcp.json"
    mcp.write_text(json.dumps({"servers": {"one": {"command": "one"}}}), encoding="utf-8")
    config = tmp_path / "agents.yaml"
    config.write_text(
        f"""version: 1
default_profile: test
profiles:
  test:
    mcp:
      file: {mcp.as_posix()}
      servers:
        two:
          command: two
""",
        encoding="utf-8",
    )

    profile = AgentProfileRegistry().load(config_path=config).resolve()
    value = load_mcp_config(profile)

    assert set(value["servers"]) == {"one", "two"}


def test_unknown_agent_config_major_version_fails(tmp_path) -> None:
    config = tmp_path / "agents.yaml"
    config.write_text("version: 2\nprofiles: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported agent configuration version"):
        AgentProfileRegistry().load(config_path=config)

