"""CLI entrypoint shared by the ``pra`` and ``pra-hf`` command trees."""

from __future__ import annotations

import click

from .deployment import HuggingFaceEngineAdapter, OpenAICompatibleEngineAdapter
from .gateway import PRAGateway, serve_gateway


@click.group("gateway")
def gateway_cli() -> None:
    """Run and inspect the standalone PRA mediation gateway."""


@gateway_cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8080, type=int, show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["G00", "G10", "G01", "G11"], case_sensitive=False),
    default="G00",
    show_default=True,
)
@click.option(
    "--backend",
    type=click.Choice(
        ["openai", "sglang", "freetoken", "vllm", "ollama", "huggingface"]
    ),
    default="openai",
    show_default=True,
)
@click.option("--backend-url")
@click.option("--model")
def gateway_serve(host, port, mode, backend, backend_url, model) -> None:
    """Serve logical PRA and OpenAI-compatible HTTP endpoints."""

    if backend == "huggingface":
        if not model:
            raise click.UsageError("--model is required for the Hugging Face backend.")
        from .model import PRAForCausalLM

        adapter = HuggingFaceEngineAdapter(PRAForCausalLM.from_pretrained(model))
    else:
        if not backend_url:
            raise click.UsageError("--backend-url is required for remote engines.")
        adapter = OpenAICompatibleEngineAdapter(backend_url, name=backend)
    gateway = PRAGateway(adapter, mode=mode.upper())
    capabilities = adapter.capabilities()
    click.echo(
        f"PRA gateway {mode.upper()} on http://{host}:{port} "
        f"-> {capabilities.adapter} ({capabilities.integration_level.value})"
    )
    serve_gateway(gateway, host=host, port=port)
