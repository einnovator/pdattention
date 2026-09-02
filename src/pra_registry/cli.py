"""Standalone and ``pra``-integrated command line for PRA Registry."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import click
import yaml


class RegistryClient:
    """Small remote client that keeps the main PRA CLI dependency-light."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 30.0) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise click.UsageError("Registry URL must use http:// or https://")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json", "User-Agent": "pra-registry-cli/1"}
        payload = None
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(f"{self.url}{path}", data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8"))
            except Exception:
                detail = {"error": {"detail": str(error)}}
            raise click.ClickException(f"Registry API {error.code}: {detail.get('error', detail).get('detail', error)}") from error
        except urllib.error.URLError as error:
            raise click.ClickException(f"Cannot reach PRA Registry at {self.url}: {error.reason}") from error


def _settings(config: Path | None, url: str | None, token: str | None) -> tuple[str, str | None]:
    value: dict[str, Any] = {}
    if config:
        loaded = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        value = dict(loaded.get("registry", loaded))
    resolved_url = url or os.environ.get("PRA_REGISTRY_URL") or value.get("url") or "http://127.0.0.1:9200"
    resolved_token = token or os.environ.get("PRA_REGISTRY_TOKEN")
    return str(resolved_url), resolved_token


def _emit(value: Any, json_output: bool, yaml_output: bool) -> None:
    if json_output and yaml_output:
        raise click.UsageError("Choose only one of --json or --yaml")
    if json_output:
        click.echo(json.dumps(value, indent=2, default=str))
    elif yaml_output:
        click.echo(yaml.safe_dump(value, sort_keys=False))
    else:
        click.echo(yaml.safe_dump(value, sort_keys=False).rstrip())


def output_options(function):
    function = click.option("--yaml", "yaml_output", is_flag=True, help="Emit YAML.")(function)
    function = click.option("--json", "json_output", is_flag=True, help="Emit JSON.")(function)
    return function


def connection_options(function):
    for decorator in reversed((
        click.option("--registry-url", metavar="URL", help="Registry base URL."),
        click.option("--token", envvar="PRA_REGISTRY_TOKEN", help="Bearer token; prefer the environment variable."),
        click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
    )):
        function = decorator(function)
    return function


def _client(config: Path | None, registry_url: str | None, token: str | None) -> RegistryClient:
    url, resolved_token = _settings(config, registry_url, token)
    return RegistryClient(url, resolved_token)


@click.group("registry")
def registry_cli() -> None:
    """Query or run the open PRA metadata and desired-state registry."""


@registry_cli.command("status")
@connection_options
@output_options
def status(registry_url, token, config, json_output, yaml_output):
    """Check registry protocol, database, and health."""
    _emit(_client(config, registry_url, token).request("GET", "/health"), json_output, yaml_output)


def _list_command(name: str, endpoint: str, help_text: str):
    @registry_cli.command(name)
    @connection_options
    @click.option("--limit", default=50, type=click.IntRange(1, 500), show_default=True)
    @click.option("--offset", default=0, type=click.IntRange(0), show_default=True)
    @output_options
    def command(registry_url, token, config, limit, offset, json_output, yaml_output):
        query = urllib.parse.urlencode({"limit": limit, "offset": offset})
        _emit(_client(config, registry_url, token).request("GET", f"{endpoint}?{query}"), json_output, yaml_output)
    command.help = help_text
    return command


_list_command("models", "/v1/models", "List registered immutable model identities.")
_list_command("bundles", "/v1/bundles", "List PRA bundles and artifact references.")
_list_command("profiles", "/v1/profiles", "List versioned PRA profiles.")
_list_command("qualifications", "/v1/qualifications", "List immutable qualification evidence.")
_list_command("deployments", "/v1/deployments", "List desired deployment state.")


@registry_cli.command("resolve")
@click.argument("model")
@click.option("--model-revision")
@click.option("--engine")
@click.option("--engine-version")
@click.option("--trust")
@connection_options
@output_options
def resolve(model, model_revision, engine, engine_version, trust, registry_url, token, config, json_output, yaml_output):
    """Resolve one model to a deterministic immutable PRA bundle."""
    body = {key: value for key, value in {
        "model": model, "model_revision": model_revision, "engine": engine,
        "engine_version": engine_version, "trust": trust,
    }.items() if value is not None}
    _emit(_client(config, registry_url, token).request("POST", "/v1/resolve/bundle", body), json_output, yaml_output)


def _import_pair(client: RegistryClient, model: Any, bundle: Any) -> dict[str, Any]:
    model_body = model.model_dump(mode="json")
    bundle_body = bundle.model_dump(mode="json")
    try:
        model_result = client.request("POST", "/v1/models", model_body)
    except click.ClickException as error:
        if "already exists" not in str(error):
            raise
        model_result = client.request("GET", f"/v1/models/{urllib.parse.quote(model.id, safe='')}")
    bundle_result = client.request("POST", "/v1/bundles", bundle_body)
    return {"model": model_result, "bundle": bundle_result}


@registry_cli.command("import-hf")
@click.argument("repo_id")
@click.option("--revision")
@connection_options
@output_options
def import_hf(repo_id, revision, registry_url, token, config, json_output, yaml_output):
    """Import bundle metadata from an immutable Hugging Face revision."""
    body = {"repo_id": repo_id, **({"revision": revision} if revision else {})}
    _emit(_client(config, registry_url, token).request("POST", "/v1/import/huggingface", body), json_output, yaml_output)


@registry_cli.command("sync-hf-collection")
@click.argument("collection")
@connection_options
@output_options
def sync_hf_collection(collection, registry_url, token, config, json_output, yaml_output):
    """Import every PRA model bundle in a Hugging Face Collection."""
    body = {"collection": collection}
    _emit(_client(config, registry_url, token).request("POST", "/v1/sync/huggingface-collection", body), json_output, yaml_output)


@registry_cli.command("serve")
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--host")
@click.option("--port", type=click.IntRange(1, 65535))
def serve(config, host, port):
    """Start the standalone headless PRA Registry service."""
    try:
        import uvicorn
        from .api import create_registry_app
        from .config import RegistryConfig
    except ImportError as error:
        raise click.ClickException("Install the 'registry' optional dependency") from error
    settings = RegistryConfig.load(config)
    if host is not None:
        settings.host = host
    if port is not None:
        settings.port = port
    settings.validate_binding()
    uvicorn.run(create_registry_app(settings), host=settings.host, port=settings.port)


@click.group(name="pra-registry", context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Standalone PRA Registry command."""


for command_name, command in registry_cli.commands.items():
    main.add_command(command, command_name)


if __name__ == "__main__":
    main()
