"""CLI entrypoint shared by the ``pra`` and ``pra-hf`` command trees."""

from __future__ import annotations

import os
from pathlib import Path

import click
import yaml

from .bundle import BundleResolver
from .deployment import HuggingFaceEngineAdapter, OpenAICompatibleEngineAdapter
from .engine_profiles import EngineType
from .gateway import FallbackInjectionPolicy, PRAGateway, serve_gateway
from .session_service import LocalSessionService
from .observability import Observability, load_observability_config
from .management_cli import ManagementClient, _emit


GATEWAY_MODE_ALIASES = {
    "passthrough": "G00",
    "selected-context": "G10",
    "upgrade": "G01",
    "typed-transport": "G11",
    "g00": "G00",
    "g10": "G10",
    "g01": "G01",
    "g11": "G11",
}


def resolve_gateway_mode(value: str) -> str:
    """Map public deployment names and legacy research codes to wire modes."""

    try:
        return GATEWAY_MODE_ALIASES[value.lower()]
    except KeyError as error:
        choices = ", ".join(GATEWAY_MODE_ALIASES)
        raise click.BadParameter(f"expected one of: {choices}") from error


@click.group("gateway")
def gateway_cli() -> None:
    """Run and inspect the standalone PRA mediation gateway."""


@gateway_cli.command("serve")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="YAML gateway configuration.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8080, type=int, show_default=True)
@click.option(
    "--mode",
    type=str,
    metavar="passthrough|selected-context|upgrade|typed-transport",
    default="passthrough",
    show_default=True,
)
@click.option(
    "--backend",
    type=click.Choice(
        ["openai", "sglang", "freetoken", "vllm", "ollama", "llama_cpp", "mlx", "custom", "huggingface"]
    ),
    default="openai",
    show_default=True,
)
@click.option("--backend-url")
@click.option("--model")
@click.option("-a", "--pra-bundle", default="auto", show_default=True)
@click.option("-p", "--profile", default="balanced", show_default=True)
@click.option("--pra-level", type=click.Choice(["auto", "E0", "E1", "E2", "E3"]), default="auto", hidden=True)
@click.option("--research", is_flag=True, hidden=True, help="Show internal protocol labels.")
@click.option(
    "--prefix-cache-mode",
    type=click.Choice(["auto", "unknown", "stateless", "automatic_prefix_cache", "explicit_prefix_handle", "session_state"]),
    default="auto",
    show_default=True,
)
@click.option("--session-state/--no-session-state", default=None)
@click.option("--incremental-messages/--full-messages", default=None)
@click.option("--resource-delta/--full-resources", default=None)
@click.option("--cache-affinity/--no-cache-affinity", default=None)
@click.option(
    "--fallback-injection",
    type=click.Choice([policy.value for policy in FallbackInjectionPolicy]),
    default=FallbackInjectionPolicy.BEFORE_CURRENT_USER.value,
    show_default=True,
)
@click.option("--sessions-dir", type=click.Path(path_type=str))
@click.option("--observability", type=click.Path(exists=True, dir_okay=False))
@click.option("--otel", is_flag=True, help="Enable OpenTelemetry tracing explicitly.")
@click.option("--otel-endpoint")
@click.option("--prometheus", is_flag=True, help="Enable the Prometheus endpoint explicitly.")
@click.option("--prometheus-port", type=click.IntRange(min=1, max=65535))
@click.option("--management-api/--no-management-api", default=None, help="Explicitly enable the separate gateway management listener.")
@click.option("--management-host", help="Management bind address; defaults to 127.0.0.1.")
@click.option("--management-port", type=click.IntRange(min=1, max=65535), help="Management port; defaults to 9150.")
@click.option(
    "--management-auth-mode",
    type=click.Choice(["none", "static_bearer", "jwt_oidc", "mtls"]),
    default=None,
    help="Management authentication; defaults to loopback-only no-auth.",
)
@click.option("--management-token-env", help="Environment variable containing the management bearer token.")
@click.option("--management-metrics-url")
@click.option("--management-trace-url")
@click.option("--management-grafana-url")
@click.option("--registry-url", help="Explicitly register this gateway with PRA Registry.")
@click.option("--registry-token-env", help="Environment variable containing the Registry token; defaults to PRA_REGISTRY_TOKEN with --registry-url.")
@click.option("--registry-instance-id", help="Stable Registry identity; otherwise it is persisted locally.")
@click.option("--registry-instance-name", help="Human-readable managed gateway name.")
@click.option("--registry-required", is_flag=True, help="Fail startup when initial registration fails.")
def gateway_serve(
    config_path, host, port, mode, backend, backend_url, model, pra_bundle, profile, pra_level, research, prefix_cache_mode,
    session_state, incremental_messages, resource_delta, cache_affinity,
    fallback_injection, sessions_dir, observability, otel, otel_endpoint,
    prometheus, prometheus_port,
    management_api, management_host, management_port, management_auth_mode,
    management_token_env, management_metrics_url, management_trace_url,
    management_grafana_url, registry_url, registry_token_env,
    registry_instance_id, registry_instance_name, registry_required,
) -> None:
    """Serve logical PRA and OpenAI-compatible HTTP endpoints."""

    resolved_mode = resolve_gateway_mode(mode)
    overrides = {}
    if otel or otel_endpoint or prometheus or prometheus_port:
        overrides["enabled"] = True
    if otel or otel_endpoint:
        overrides["otel"] = {
            "enabled": True,
            **({"endpoint": otel_endpoint} if otel_endpoint else {}),
        }
    if prometheus or prometheus_port:
        overrides["prometheus"] = {
            "enabled": True,
            **({"port": prometheus_port} if prometheus_port else {}),
        }
    telemetry = Observability(
        load_observability_config(
            observability, overrides=overrides, service="gateway"
        ),
        start_server=True,
    )

    bundle = None
    bundle_source = None
    if model:
        resolution = BundleResolver().resolve(
            pra_bundle, model=model, engine="hf" if backend == "huggingface" else backend
        )
        bundle = resolution.bundle
        bundle_source = resolution.source

    if backend == "huggingface":
        if not model:
            raise click.UsageError("--model is required for the Hugging Face backend.")
        from .model import PRAForCausalLM

        model_kwargs = {}
        if bundle is not None:
            for name, path in bundle.selected_learned_adapters(profile).items():
                adapter_type = str(bundle.learned_adapters[name].get("type", name))
                if adapter_type in {"routing", "router"}:
                    model_kwargs["routing_adapter"] = path
                elif adapter_type in {"memory", "memory_adapter", "consumer"}:
                    model_kwargs["memory_adapter"] = path
        adapter = HuggingFaceEngineAdapter(
            PRAForCausalLM.from_pretrained(model, **model_kwargs), observability=telemetry
        )
    else:
        if not backend_url:
            raise click.UsageError("--backend-url is required for remote engines.")
        engine_type = EngineType.OPENAI_GENERIC if backend == "openai" else EngineType(backend)
        adapter = OpenAICompatibleEngineAdapter(
            backend_url,
            name=backend,
            engine_type=engine_type,
            pra_level=pra_level,
            prefix_cache_mode=prefix_cache_mode,
            session_state=session_state,
            incremental_messages=incremental_messages,
            resource_delta=resource_delta,
            cache_affinity=cache_affinity,
            observability=telemetry,
        )
    management_requested = any((
        management_api is not None, config_path is not None,
        os.environ.get("PRA_GATEWAY_MANAGEMENT_ENABLED") is not None,
        registry_url, os.environ.get("PRA_GATEWAY_REGISTRY_URL"),
    ))
    management_settings = None
    management_provider = None
    gateway_adapter = adapter
    gateway_observability = telemetry
    if management_requested:
        from .gateway_management import (
            GatewayManagementAPIConfig,
            GatewayManagementProvider,
            GatewayMetricRecorder,
            GatewayPolicy,
            GatewayUpstreamRouter,
            UpstreamCreate,
        )

        raw = _gateway_management_yaml(config_path)
        management_settings = GatewayManagementAPIConfig.from_mapping(raw)
        update = {}
        if registry_url:
            update["enabled"] = True
        if management_api is not None:
            update["enabled"] = management_api
        if management_host is not None:
            update["host"] = management_host
        if management_port is not None:
            update["port"] = management_port
        if management_auth_mode is not None or management_token_env is not None:
            update["auth"] = management_settings.auth.model_copy(update={
                **({"mode": management_auth_mode} if management_auth_mode is not None else {}),
                **({"token_env": management_token_env} if management_token_env is not None else {}),
            })
        for field, value in (
            ("metrics_url", management_metrics_url), ("trace_backend_url", management_trace_url),
            ("grafana_url", management_grafana_url),
        ):
            if value is not None:
                update[field] = value
        if any((registry_url, management_settings.registry.enabled, registry_instance_id, registry_instance_name, registry_required)):
            registry_auth = management_settings.registry.auth.model_copy(update={
                **(
                    {"type": "bearer", "token_env": registry_token_env or "PRA_REGISTRY_TOKEN"}
                    if registry_url or registry_token_env else {}
                ),
            })
            registry_instance = management_settings.registry.instance.model_copy(update={
                **({"instance_id": registry_instance_id} if registry_instance_id else {}),
                **({"name": registry_instance_name} if registry_instance_name else {}),
                "management_url": f"http://{management_host or management_settings.host}:{management_port or management_settings.port}",
                "inference_url": f"http://{host}:{port}",
            })
            update["registry"] = management_settings.registry.model_copy(update={
                "enabled": True,
                **({"url": registry_url} if registry_url else {}),
                "required": registry_required,
                "auth": registry_auth,
                "instance": registry_instance,
            })
        management_settings = management_settings.model_copy(update=update)
        if management_settings.enabled or management_settings.registry.enabled:
            initial = UpstreamCreate(
                upstream_id="default", name=backend, base_url=backend_url or "embedded://huggingface",
                provider=backend, inference_api_type="embedded" if backend == "huggingface" else "openai-compatible",
                models=tuple([model] if model else ()), priority=0,
            )
            router = GatewayUpstreamRouter(initial, adapter, GatewayPolicy(default_upstream_id="default"))
            gateway_adapter = router
            gateway_observability = GatewayMetricRecorder(telemetry)
        else:
            management_settings = None

    gateway = PRAGateway(
        gateway_adapter,
        mode=resolved_mode,
        session_service=LocalSessionService(sessions_dir) if sessions_dir else None,
        fallback_injection=fallback_injection,
        observability=gateway_observability,
        bundle_source=bundle_source,
        default_profile=profile,
    )
    management_server = None
    if management_settings is not None:
        from .gateway_management import start_gateway_management_api
        management_provider = GatewayManagementProvider(
            gateway, gateway_adapter, management_settings, gateway_observability,
            policy_loader=(
                (lambda: _gateway_management_yaml(config_path).get("policy", {}))
                if config_path is not None else None
            ),
        )
        management_server = start_gateway_management_api(management_provider, management_settings)
    capabilities = adapter.capabilities()
    selected_enabled = resolved_mode in {"G10", "G11"}
    typed_enabled = resolved_mode == "G11"
    native_available = bool(capabilities.native_kv)
    click.echo(f"PRA gateway on http://{host}:{port} -> {capabilities.adapter}/{capabilities.engine_type.value}")
    click.echo("Existing OpenAI-compatible clients: supported")
    click.echo(f"Selected Context: {'enabled' if selected_enabled else 'disabled'}")
    click.echo(f"Typed resource transport: {'enabled' if typed_enabled else 'disabled'}")
    click.echo(f"Native Memory: {'delegated to backend' if native_available else 'not advertised by backend'}")
    click.echo(f"Backend native handshake: {'present' if native_available else 'absent'}")
    click.echo(f"Effective mode: {'Typed Transport' if typed_enabled else 'Selected Context' if selected_enabled else 'Pass-through'}")
    if research:
        click.echo(
            f"Internal protocol: {resolved_mode}, {capabilities.integration_level.value}, "
            f"{capabilities.prefix_cache_mode.value}"
        )
    if management_settings is not None and management_settings.enabled:
        click.echo(
            f"Gateway Management API: http://{management_settings.host}:{management_settings.port}/v1/pra/gateway/info"
        )
    try:
        serve_gateway(gateway, host=host, port=port)
    finally:
        if management_server is not None:
            from .gateway_management import stop_gateway_management_api

            stop_gateway_management_api(management_server)
        telemetry.close()


def _gateway_management_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gateway = dict(value.get("gateway", value))
    return dict(gateway.get("management_api", {}))


def _gateway_remote_options(function):
    function = click.option("--token", envvar="PRA_GATEWAY_MANAGEMENT_TOKEN", help="Gateway management bearer token.")(function)
    function = click.option("--management-url", default="http://127.0.0.1:9150", show_default=True)(function)
    function = click.option("--yaml", "yaml_output", is_flag=True)(function)
    function = click.option("--json", "json_output", is_flag=True)(function)
    return function


def _gateway_client(management_url: str, token: str | None) -> ManagementClient:
    return ManagementClient(management_url, token)


def _get_gateway_command(name: str, endpoint: str, help_text: str) -> None:
    @gateway_cli.command(name, help=help_text)
    @_gateway_remote_options
    def command(management_url, token, json_output, yaml_output):
        _emit(_gateway_client(management_url, token).get(f"/v1/pra/gateway/{endpoint}"), json_output, yaml_output)


_get_gateway_command("health", "health", "Check gateway protocol and health.")
_get_gateway_command("upstreams", "upstreams", "List configured upstream inference endpoints.")
_get_gateway_command("sessions", "sessions", "List privacy-safe gateway session summaries.")
_get_gateway_command("transport", "transport", "Show wire, delta, fallback, and reuse counters.")
_get_gateway_command("config", "config", "Show effective gateway and policy configuration.")
_get_gateway_command("registry-status", "registry", "Show Registry registration and heartbeat state.")


@gateway_cli.command("register")
@_gateway_remote_options
def gateway_register(management_url, token, json_output, yaml_output) -> None:
    """Retry this gateway's configured Registry registration immediately."""

    client = _gateway_client(management_url, token)
    _emit(
        client.request("POST", "/v1/pra/gateway/registry/register", {}),
        json_output, yaml_output,
    )


@gateway_cli.command("inspect")
@_gateway_remote_options
def gateway_inspect(management_url, token, json_output, yaml_output) -> None:
    """Inspect gateway identity, capabilities, state, and observability."""
    client = _gateway_client(management_url, token)
    prefix = "/v1/pra/gateway"
    _emit({
        "info": client.get(f"{prefix}/info"), "capabilities": client.get(f"{prefix}/capabilities"),
        "state": client.get(f"{prefix}/state"), "observability": client.get(f"{prefix}/observability"),
    }, json_output, yaml_output)


def _gateway_action(name: str, path: str, argument: str) -> None:
    help_text = (
        "Refresh one upstream capability handshake."
        if name == "renegotiate"
        else "Invalidate one gateway session so its next turn fully resynchronizes."
    )

    @gateway_cli.command(name, help=help_text)
    @click.argument(argument)
    @click.option("--reason", required=True)
    @_gateway_remote_options
    def command(**values):
        target = values.pop(argument)
        reason = values.pop("reason")
        client = _gateway_client(values.pop("management_url"), values.pop("token"))
        result = client.request("POST", f"/v1/pra/gateway/actions/{path}/{target}", {"reason": reason})
        _emit(result, values.pop("json_output"), values.pop("yaml_output"))


_gateway_action("renegotiate", "renegotiate", "upstream")
_gateway_action("resync", "resync-session", "session")
