"""CLI entrypoint shared by the ``pra`` and ``pra-hf`` command trees."""

from __future__ import annotations

import click

from .deployment import HuggingFaceEngineAdapter, OpenAICompatibleEngineAdapter
from .engine_profiles import EngineType
from .gateway import FallbackInjectionPolicy, PRAGateway, serve_gateway
from .session_service import LocalSessionService


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
def gateway_serve(
    host, port, mode, backend, backend_url, model, pra_level, research, prefix_cache_mode,
    session_state, incremental_messages, resource_delta, cache_affinity,
    fallback_injection, sessions_dir,
) -> None:
    """Serve logical PRA and OpenAI-compatible HTTP endpoints."""

    resolved_mode = resolve_gateway_mode(mode)

    if backend == "huggingface":
        if not model:
            raise click.UsageError("--model is required for the Hugging Face backend.")
        from .model import PRAForCausalLM

        adapter = HuggingFaceEngineAdapter(PRAForCausalLM.from_pretrained(model))
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
        )
    gateway = PRAGateway(
        adapter,
        mode=resolved_mode,
        session_service=LocalSessionService(sessions_dir) if sessions_dir else None,
        fallback_injection=fallback_injection,
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
    serve_gateway(gateway, host=host, port=port)
