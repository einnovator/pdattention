"""CLI entrypoint shared by the ``pra`` and ``pra-hf`` command trees."""

from __future__ import annotations

import click

from .bundle import BundleResolver
from .deployment import HuggingFaceEngineAdapter, OpenAICompatibleEngineAdapter
from .engine_profiles import EngineType
from .gateway import FallbackInjectionPolicy, PRAGateway, serve_gateway
from .session_service import LocalSessionService
from .observability import Observability, load_observability_config


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


def _apply_gateway_config(gateway: PRAGateway, values):
    """Apply only gateway settings that affect subsequent real requests."""

    from .management import ManagementAPIError

    unsupported = sorted(set(values) - {"profile"})
    if unsupported:
        raise ManagementAPIError(
            501,
            "CONFIG_FIELD_NOT_SUPPORTED",
            "This gateway can update only the default profile while running.",
            unsupported_fields=unsupported,
        )
    if "profile" in values:
        gateway.default_profile = str(values["profile"])
    return values


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
@click.option("--management-api", is_flag=True, help="Enable the separate PRA management listener.")
@click.option("--management-host", default="127.0.0.1", show_default=True)
@click.option("--management-port", type=click.IntRange(min=1, max=65535), default=9101, show_default=True)
@click.option(
    "--management-auth-mode",
    type=click.Choice(["none", "static_bearer", "jwt_oidc", "mtls"]),
    default="none",
    show_default=True,
)
@click.option("--management-token-env", default="PRA_MANAGEMENT_TOKEN", show_default=True)
@click.option("--management-metrics-url")
@click.option("--management-trace-url")
@click.option("--management-grafana-url")
def gateway_serve(
    host, port, mode, backend, backend_url, model, pra_bundle, profile, pra_level, research, prefix_cache_mode,
    session_state, incremental_messages, resource_delta, cache_affinity,
    fallback_injection, sessions_dir, observability, otel, otel_endpoint,
    prometheus, prometheus_port,
    management_api, management_host, management_port, management_auth_mode,
    management_token_env, management_metrics_url, management_trace_url,
    management_grafana_url,
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
    gateway = PRAGateway(
        adapter,
        mode=resolved_mode,
        session_service=LocalSessionService(sessions_dir) if sessions_dir else None,
        fallback_injection=fallback_injection,
        observability=telemetry,
        bundle_source=bundle_source,
        default_profile=profile,
    )
    management_server = None
    if management_api:
        from .management import (
            LoadedModel,
            ManagementAPIConfig,
            ManagementAuthConfig,
            ManagementProvider,
            PRAProfileSummary,
            start_management_api,
        )

        management_settings = ManagementAPIConfig(
            enabled=True,
            host=management_host,
            port=management_port,
            auth=ManagementAuthConfig(
                mode=management_auth_mode,
                token_env=management_token_env,
            ),
            metrics_url=management_metrics_url,
            trace_backend_url=management_trace_url,
            grafana_url=management_grafana_url,
        )
        management_provider = ManagementProvider(
            engine=backend,
            capabilities=adapter.capabilities().to_dict(),
            models=[] if model is None else [
                LoadedModel(
                    model_id=model,
                    pra_bundle_id=bundle_source,
                    profile=profile,
                    execution_mode=resolved_mode,
                )
            ],
            profiles=[PRAProfileSummary(name=profile, source="gateway")],
            effective_config={
                "engine": backend,
                "model": model,
                "profile": profile,
                "execution_mode": resolved_mode,
                "inference_url": backend_url,
            },
            storage_manager=getattr(adapter, "storage_manager", None),
            session_source=gateway.sessions,
            observability={
                "otel": {"enabled": bool(otel or otel_endpoint)},
                "prometheus": {"enabled": bool(prometheus or prometheus_port)},
            },
            config_patch_handler=lambda values: _apply_gateway_config(gateway, values),
        )
        management_server = start_management_api(
            management_provider, management_settings
        )
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
    if management_server is not None:
        click.echo(
            f"Management API: http://{management_host}:{management_port}/v1/pra/info"
        )
    try:
        serve_gateway(gateway, host=host, port=port)
    finally:
        if management_server is not None:
            from .management import stop_management_api

            stop_management_api(management_server)
        telemetry.close()
