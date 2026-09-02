from __future__ import annotations

import json

from click.testing import CliRunner

from pra_hf.cli import cli
from pra_registry.cli import RegistryClient, main


def test_registry_is_available_from_both_entry_points() -> None:
    runner = CliRunner()
    assert runner.invoke(main, ["--help"]).exit_code == 0
    result = runner.invoke(cli, ["registry", "--help"])
    assert result.exit_code == 0 and "import-hf" in result.output and "resolve" in result.output


def test_remote_status_and_lists_use_registry_url(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(RegistryClient, "request", lambda self, method, path, body=None: seen.append((self.url, method, path, body)) or {"status": "ok"})
    runner = CliRunner()
    result = runner.invoke(cli, ["registry", "status", "--registry-url", "http://registry.test", "--json"])
    assert result.exit_code == 0 and json.loads(result.output)["status"] == "ok"
    assert seen == [("http://registry.test", "GET", "/health", None)]


def test_resolve_sends_only_supplied_constraints(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(RegistryClient, "request", lambda self, method, path, body=None: seen.append(body) or {"selected_bundle": {"id": "b"}})
    result = CliRunner().invoke(cli, [
        "registry", "resolve", "org/model", "--engine", "vllm",
        "--registry-url", "http://registry.test", "--json",
    ])
    assert result.exit_code == 0
    assert seen == [{"model": "org/model", "engine": "vllm"}]


def test_import_hf_runs_on_registry_service(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(RegistryClient, "request", lambda self, method, path, body=None: seen.append((method, path, body)) or {"bundle": {"id": "b"}})
    result = CliRunner().invoke(cli, [
        "registry", "import-hf", "EInnovator/pra-model", "--revision", "abc",
        "--registry-url", "http://registry.test", "--json",
    ])
    assert result.exit_code == 0
    assert seen == [("POST", "/v1/import/huggingface", {"repo_id": "EInnovator/pra-model", "revision": "abc"})]
