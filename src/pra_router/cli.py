"""Commands for the open PRA Router Controller and reference router."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import click
import yaml

from .adapters import adapter_for
from .controller import RouterController
from .reference import ReferenceRouter, ReferenceRouterConfig, create_reference_router_app
from .registry import RegistryRouterSource


def _emit(value: Any, *, json_output: bool = False) -> None:
    click.echo(json.dumps(value, indent=2, default=str) if json_output else yaml.safe_dump(value, sort_keys=False).rstrip())


def _get(url: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{url.rstrip('/')}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _controller(registry_url: str, token: str | None) -> RouterController:
    source = RegistryRouterSource(registry_url, token)
    kinds = ("litellm", "agentgateway", "kubernetes-gaie", "pra-reference", "bifrost")
    return RouterController(source, {kind: adapter_for(kind) for kind in kinds})


@click.group("router")
def router_cli() -> None:
    """Manage PRA routes or run the small reference routing data plane."""


@router_cli.command("serve")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=9400, type=click.IntRange(1, 65535), show_default=True)
@click.option("--reload-token-env", default="PRA_ROUTER_RELOAD_TOKEN", show_default=True)
def serve(config_path: Path, host: str, port: int, reload_token_env: str) -> None:
    """Run the standalone reference router from last-good YAML/JSON config."""

    try:
        import uvicorn
    except ImportError as error:
        raise click.ClickException("Install the 'router' optional dependency") from error
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = raw.get("router", raw)
    router = ReferenceRouter(ReferenceRouterConfig.model_validate(config))
    uvicorn.run(create_reference_router_app(router, reload_token=os.environ.get(reload_token_env)), host=host, port=port)


@router_cli.command("inspect")
@click.argument("router_id", required=False)
@click.option("--url", default="http://127.0.0.1:9400", show_default=True)
@click.option("--registry-url", default="http://127.0.0.1:9200", show_default=True)
@click.option("--token", envvar="PRA_REGISTRY_TOKEN")
@click.option("--json", "json_output", is_flag=True)
def inspect(router_id: str | None, url: str, registry_url: str, token: str | None, json_output: bool) -> None:
    """Inspect a live reference router or Registry-managed router drift."""

    value = asyncio.run(_controller(registry_url, token).inspect(router_id)) if router_id else _get(url, "/v1/router/info")
    _emit(value, json_output=json_output)


@router_cli.command("routes")
@click.option("--url", default="http://127.0.0.1:9400", show_default=True)
@click.option("--json", "json_output", is_flag=True)
def routes(url: str, json_output: bool) -> None:
    """List logical routes and eligible backends from a live reference router."""

    _emit(_get(url, "/v1/router/routes"), json_output=json_output)


@router_cli.command("preview")
@click.argument("router_id")
@click.option("--registry-url", default="http://127.0.0.1:9200", show_default=True)
@click.option("--token", envvar="PRA_REGISTRY_TOKEN")
@click.option("--json", "json_output", is_flag=True)
def preview(router_id: str, registry_url: str, token: str | None, json_output: bool) -> None:
    """Compile and diff Registry intent without changing router state."""

    value = asyncio.run(_controller(registry_url, token).preview(router_id))
    _emit(value.model_dump(mode="json"), json_output=json_output)


@router_cli.command("reconcile")
@click.argument("router_id", required=False)
@click.option("--all", "all_routers", is_flag=True, help="Reconcile every registered router.")
@click.option("--registry-url", default="http://127.0.0.1:9200", show_default=True)
@click.option("--token", envvar="PRA_REGISTRY_TOKEN")
@click.option("--confirm", is_flag=True, help="Confirm applying the displayed desired state.")
@click.option("--json", "json_output", is_flag=True)
def reconcile(
    router_id: str | None, all_routers: bool, registry_url: str, token: str | None,
    confirm: bool, json_output: bool,
) -> None:
    """Apply desired routes and verify observed router state."""

    if bool(router_id) == bool(all_routers):
        raise click.UsageError("Provide ROUTER_ID or --all")
    if not confirm:
        raise click.UsageError("Reconciliation changes router state; pass --confirm after reviewing preview")
    controller = _controller(registry_url, token)
    result = asyncio.run(controller.reconcile_all() if all_routers else controller.reconcile(str(router_id)))
    if isinstance(result, list):
        value = [item.model_dump(mode="json") for item in result]
    else:
        value = result.model_dump(mode="json")
    _emit(value, json_output=json_output)


@router_cli.command("controller")
@click.option("--registry-url", default="http://127.0.0.1:9200", show_default=True)
@click.option("--token", envvar="PRA_REGISTRY_TOKEN")
@click.option("--interval", default=10.0, type=click.FloatRange(min=1), show_default=True)
@click.option("--once", is_flag=True, help="Reconcile once and exit.")
def controller(registry_url: str, token: str | None, interval: float, once: bool) -> None:
    """Continuously reconcile all Registry-managed router instances."""

    async def run() -> None:
        manager = _controller(registry_url, token)
        while True:
            results = await manager.reconcile_all()
            _emit([result.model_dump(mode="json") for result in results])
            if once:
                return
            await asyncio.sleep(interval)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        return


@click.group(name="pra-router", context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Standalone PRA Router command."""


for command_name, command in router_cli.commands.items():
    main.add_command(command, command_name)


if __name__ == "__main__":
    main()
