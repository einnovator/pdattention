"""Remote CLI for the open PRA engine-management protocol."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import click
import yaml

from .product_config import dump_data, pra_home


def _emit(value: Any, json_output: bool, yaml_output: bool) -> None:
    format_name = "json" if json_output else "yaml" if yaml_output else "human"
    click.echo(dump_data(value, format_name))


def _output_options(function):
    function = click.option("--yaml", "yaml_output", is_flag=True)(function)
    function = click.option("--json", "json_output", is_flag=True)(function)
    return function


def _target_options(function):
    decorators = (
        click.argument("target", required=False),
        click.option("--management-url", metavar="URL"),
        click.option("--token", envvar="PRA_MANAGEMENT_TOKEN", help="Bearer token; prefer its environment variable."),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


class ManagementClient:
    """Small dependency-free client used by CLI and external automation tests."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json", "User-Agent": "pra-cli/management-1"}
        payload = None
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}", data=payload, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                detail = {"error": {"detail": str(error)}}
            message = detail.get("error", detail).get("detail", str(error))
            raise click.ClickException(f"Management API {error.code}: {message}") from error
        except urllib.error.URLError as error:
            raise click.ClickException(f"Cannot reach PRA management API at {self.url}: {error.reason}") from error

    def get(self, path: str) -> Any:
        return self.request("GET", path)


def _connections_path() -> Path:
    return pra_home() / "engines" / "connections.json"


def _connections() -> dict[str, Any]:
    path = _connections_path()
    if not path.exists():
        return {"version": 1, "default": None, "connections": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_connections(value: Mapping[str, Any]) -> None:
    path = _connections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="connections-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolved_url(target: str | None, management_url: str | None) -> str:
    candidate = management_url or target
    registry = _connections()
    if candidate is None:
        candidate = registry.get("default")
    if candidate in registry.get("connections", {}):
        candidate = registry["connections"][candidate]["url"]
    if not candidate:
        raise click.UsageError("Provide URL/connection name or --management-url.")
    parsed = urllib.parse.urlparse(str(candidate))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise click.UsageError("Management URL must use http:// or https://.")
    return str(candidate).rstrip("/")


def _client(target: str | None, management_url: str | None, token: str | None) -> ManagementClient:
    return ManagementClient(_resolved_url(target, management_url), token)


@click.group("engine")
def engine_cli() -> None:
    """Connect to and manage one local or remote PRA engine API."""


@engine_cli.command("connect")
@click.argument("url")
@click.option("--name", default="default", show_default=True)
@click.option("--token-env", help="Environment variable holding the bearer token; the secret is not stored.")
@_output_options
def connect(url: str, name: str, token_env: str | None, json_output: bool, yaml_output: bool) -> None:
    """Validate and remember a management URL without storing credentials."""

    token = os.environ.get(token_env) if token_env else None
    client = ManagementClient(_resolved_url(url, None), token)
    health = client.get("/v1/pra/health")
    registry = _connections()
    registry["connections"][name] = {"url": client.url, "token_env": token_env}
    registry["default"] = name
    _write_connections(registry)
    _emit({"name": name, "url": client.url, "health": health, "stored_secret": False}, json_output, yaml_output)


def _registered_token(
    target: str | None,
    explicit: str | None,
    management_url: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    # Never send a saved connection's credential to a one-off URL. A token for
    # that URL must be supplied explicitly through --token/the environment.
    if management_url is not None or (target or "").startswith(("http://", "https://")):
        return None
    registry = _connections()
    name = target if target in registry.get("connections", {}) else registry.get("default")
    row = registry.get("connections", {}).get(name or "", {})
    return os.environ.get(row.get("token_env")) if row.get("token_env") else None


def _get_command(name: str, path: str, help_text: str):
    @engine_cli.command(name)
    @_target_options
    @_output_options
    def command(target, management_url, token, json_output, yaml_output):
        """Query one management API endpoint."""

        resolved_token = _registered_token(target, token, management_url)
        _emit(_client(target, management_url, resolved_token).get(path), json_output, yaml_output)

    command.help = help_text
    return command


_get_command("health", "/v1/pra/health", "Check protocol and local engine health.")
_get_command("config", "/v1/pra/config", "Show effective and desired configuration state.")
_get_command("storage", "/v1/pra/storage", "Show tier residency, quotas, and lifecycle counters.")
_get_command("sessions", "/v1/pra/sessions", "List privacy-safe session summaries.")
_get_command("resources", "/v1/pra/resources", "List privacy-safe resource summaries.")
_get_command("models", "/v1/pra/models", "List loaded model identities.")
_get_command("profiles", "/v1/pra/profiles", "List effective PRA profiles.")
_get_command("capabilities", "/v1/pra/capabilities", "Show qualified local engine capabilities.")
_get_command("audit", "/v1/pra/audit", "Show recent local management audit events.")
_get_command("registry-status", "/v1/pra/registry", "Show Registry registration and heartbeat state.")


@engine_cli.command("model")
@click.argument("target")
@click.argument("runtime_model_id")
@click.option("--management-url", metavar="URL")
@click.option("--token", envvar="PRA_MANAGEMENT_TOKEN", help="Bearer token; prefer its environment variable.")
@_output_options
def model(target, management_url, token, runtime_model_id, json_output, yaml_output) -> None:
    """Show one loaded model by its engine-local runtime identity."""

    client = _client(target, management_url, _registered_token(target, token, management_url))
    path = "/v1/pra/models/" + urllib.parse.quote(runtime_model_id, safe="")
    _emit(client.get(path), json_output, yaml_output)


def _require_dynamic_capability(client: ManagementClient, capability: str) -> None:
    capabilities = client.get("/v1/pra/capabilities")
    if not capabilities.get(capability, False):
        raise click.ClickException(f"The target engine does not support {capability.replace('_', ' ')}.")


@engine_cli.command("load-model")
@click.argument("target")
@click.argument("runtime_model_id")
@click.argument("model_id")
@click.option("--management-url", metavar="URL")
@click.option("--token", envvar="PRA_MANAGEMENT_TOKEN", help="Bearer token; prefer its environment variable.")
@click.option("--revision")
@click.option("--bundle")
@click.option("--profile")
@click.option("--execution-mode", default="selected-context", show_default=True)
@click.option("--parameter", "parameters", multiple=True, metavar="KEY=VALUE")
@_output_options
def load_model(target, management_url, token, runtime_model_id, model_id, revision, bundle, profile, execution_mode, parameters, json_output, yaml_output) -> None:
    """Load one model when the target engine supports dynamic residency."""

    client = _client(target, management_url, _registered_token(target, token, management_url))
    _require_dynamic_capability(client, "dynamic_model_load")
    parsed_parameters = {}
    for value in parameters:
        if "=" not in value:
            raise click.UsageError("--parameter must use KEY=VALUE.")
        key, item = value.split("=", 1)
        parsed_parameters[key] = item
    body = {
        key: value for key, value in {
            "runtime_model_id": runtime_model_id,
            "model_id": model_id,
            "revision": revision,
            "bundle": bundle,
            "profile": profile,
            "execution_mode": execution_mode,
            "parameters": parsed_parameters,
        }.items() if value is not None and value != ""
    }
    _emit(client.request("POST", "/v1/pra/actions/load-model", body), json_output, yaml_output)


@engine_cli.command("unload-model")
@click.argument("target")
@click.argument("runtime_model_id")
@click.option("--management-url", metavar="URL")
@click.option("--token", envvar="PRA_MANAGEMENT_TOKEN", help="Bearer token; prefer its environment variable.")
@click.option("--force", is_flag=True, help="Allow engine-defined forced unload behavior.")
@_output_options
def unload_model(target, management_url, token, runtime_model_id, force, json_output, yaml_output) -> None:
    """Unload one runtime model and release its model-native state."""

    client = _client(target, management_url, _registered_token(target, token, management_url))
    _require_dynamic_capability(client, "dynamic_model_unload")
    _emit(client.request("POST", "/v1/pra/actions/unload-model", {
        "runtime_model_id": runtime_model_id, "force": force,
    }), json_output, yaml_output)


@engine_cli.command("register")
@_target_options
@_output_options
def register(target, management_url, token, json_output, yaml_output) -> None:
    """Retry this engine's configured Registry registration immediately."""

    client = _client(target, management_url, _registered_token(target, token, management_url))
    _emit(client.request("POST", "/v1/pra/registry/register", {}), json_output, yaml_output)


@engine_cli.command("inspect")
@_target_options
@_output_options
def inspect(target, management_url, token, json_output, yaml_output) -> None:
    """Inspect engine identity, capabilities, state, and observability links."""

    client = _client(target, management_url, _registered_token(target, token, management_url))
    value = {
        "info": client.get("/v1/pra/info"),
        "capabilities": client.get("/v1/pra/capabilities"),
        "state": client.get("/v1/pra/state"),
        "observability": client.get("/v1/pra/observability"),
    }
    _emit(value, json_output, yaml_output)


@engine_cli.command("patch-config")
@_target_options
@click.option("--patch", "patch_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@_output_options
def patch_config(target, management_url, token, patch_path, json_output, yaml_output) -> None:
    """Apply a bounded YAML/JSON configuration patch."""

    value = yaml.safe_load(patch_path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise click.UsageError("Patch file must contain a mapping.")
    client = _client(target, management_url, _registered_token(target, token, management_url))
    _emit(client.request("PATCH", "/v1/pra/config", value), json_output, yaml_output)


@engine_cli.command("action")
@click.argument("action", type=click.Choice([
    "prefetch", "evict", "promote", "demote", "reload-profile", "reload-bundle", "maintenance"
]))
@_target_options
@click.option("--resource-id")
@click.option("--profile")
@click.option("--bundle")
@click.option("--tenant-id")
@click.option("--idempotency-key")
@_output_options
def action(action, target, management_url, token, resource_id, profile, bundle, tenant_id, idempotency_key, json_output, yaml_output) -> None:
    """Run one bounded engine-supported local management action."""

    body = {
        key: value for key, value in {
            "resource_id": resource_id, "profile": profile, "bundle": bundle,
            "tenant_id": tenant_id, "idempotency_key": idempotency_key,
        }.items() if value not in {None, ""}
    }
    client = _client(target, management_url, _registered_token(target, token, management_url))
    _emit(client.request("POST", f"/v1/pra/actions/{action}", body), json_output, yaml_output)


@engine_cli.command("serve")
@click.option("--engine", default="hf", show_default=True)
@click.option("--engine-version")
@click.option("--model", "models", multiple=True, help="Loaded model ID; repeat for multi-model engines.")
@click.option("--runtime-model-id", "runtime_model_ids", multiple=True, help="Engine-local model alias paired with each --model.")
@click.option("--revision")
@click.option("--inference-url")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--host")
@click.option("--port", type=click.IntRange(1, 65535))
@click.option("--auth-mode", type=click.Choice(["none", "static_bearer", "jwt_oidc", "mtls"]))
@click.option("--token-env")
@click.option("--metrics-url")
@click.option("--trace-backend-url")
@click.option("--grafana-url")
@click.option("--tls-certfile", type=click.Path(exists=True, dir_okay=False))
@click.option("--tls-keyfile", type=click.Path(exists=True, dir_okay=False))
@click.option("--tls-ca-certs", type=click.Path(exists=True, dir_okay=False))
@click.option("--registry-url")
@click.option("--registry-token-env", default="PRA_REGISTRY_TOKEN", show_default=True)
@click.option("--registry-instance-id")
@click.option("--registry-instance-name")
@click.option(
    "--registry-instance-host",
    help="Externally reachable host advertised to the Registry.",
)
@click.option(
    "--registry-management-url",
    help="Externally reachable management API URL advertised to the Registry.",
)
@click.option("--registry-required", is_flag=True)
def serve(engine, engine_version, models, runtime_model_ids, revision, inference_url, config_path, host, port, auth_mode, token_env, metrics_url, trace_backend_url, grafana_url, tls_certfile, tls_keyfile, tls_ca_certs, registry_url, registry_token_env, registry_instance_id, registry_instance_name, registry_instance_host, registry_management_url, registry_required) -> None:
    """Start an explicitly enabled local management sidecar on a separate port."""

    from .management import (
        LoadedModel,
        ManagementAPIConfig,
        ManagementProvider,
        PRAProfileSummary,
        serve_management_api,
    )

    capabilities: Mapping[str, Any] = {}
    if runtime_model_ids and len(runtime_model_ids) != len(models):
        raise click.UsageError("Repeat --runtime-model-id once for every --model.")
    primary_model = models[0] if models else None
    try:
        from .runtime_providers import RuntimeConfig, RuntimeProviderRegistry
        capabilities = RuntimeProviderRegistry.default().resolve(engine).capabilities(
            RuntimeConfig(engine=engine, model=primary_model, revision=revision, endpoint=inference_url)
        ).to_dict()
    except KeyError:
        capabilities = {"text_fallback": True, "integration_level": "E0"}
    loaded_models = [
        LoadedModel(
            runtime_model_id=(runtime_model_ids[index] if runtime_model_ids else ("default" if len(models) == 1 else model_id)),
            model_id=model_id,
            revision=revision,
        )
        for index, model_id in enumerate(models)
    ]
    if len(loaded_models) > 1:
        capabilities = {**capabilities, "multi_model": True, "max_loaded_models": len(loaded_models)}
    profiles = [PRAProfileSummary(name="BALANCED", source="management-sidecar")]
    provider = ManagementProvider(
        engine=engine,
        engine_version=engine_version,
        capabilities=capabilities,
        models=loaded_models,
        profiles=profiles,
        effective_config={
            "engine": engine, "model": primary_model, "models": list(models), "revision": revision,
            "inference_url": inference_url, "profile": "BALANCED",
        },
        observability={
            "metrics_url": metrics_url,
            "trace_backend_url": trace_backend_url,
            "grafana_url": grafana_url,
        },
    )
    values: dict[str, Any] = {}
    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise click.UsageError("Management config must contain a mapping.")
        values = dict(loaded.get("management_api", loaded))
    values["enabled"] = True
    if host is not None:
        values["host"] = host
    if port is not None:
        values["port"] = port
    auth_values = dict(values.get("auth") or {})
    if auth_mode is not None:
        auth_values["mode"] = auth_mode
    if token_env is not None:
        auth_values["token_env"] = token_env
    if auth_values:
        values["auth"] = auth_values
    for key, value in {
        "metrics_url": metrics_url,
        "trace_backend_url": trace_backend_url,
        "grafana_url": grafana_url,
        "tls_certfile": tls_certfile,
        "tls_keyfile": tls_keyfile,
        "tls_ca_certs": tls_ca_certs,
    }.items():
        if value is not None:
            values[key] = value
    settings = ManagementAPIConfig.from_mapping(values)
    if registry_url:
        from .registry_registration import RegistryClientAuth, RuntimeInstanceIdentity, RuntimeRegistryConfig
        settings.registry = RuntimeRegistryConfig(
            enabled=True, url=registry_url, required=registry_required,
            auth=RegistryClientAuth(type="bearer", token_env=registry_token_env),
            instance=RuntimeInstanceIdentity(
                id=registry_instance_id, name=registry_instance_name,
                host=registry_instance_host,
                management_url=registry_management_url,
                inference_url=inference_url,
            ),
        )
    click.echo(f"PRA management API ({engine}) on http://{settings.host}:{settings.port}")
    click.echo(f"OpenAPI: http://{settings.host}:{settings.port}/openapi.json")
    click.echo(f"Swagger: http://{settings.host}:{settings.port}/docs")
    serve_management_api(provider, settings)
