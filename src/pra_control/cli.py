"""Command-line entry point for the Enterprise PRA Control Plane."""

from __future__ import annotations

import asyncio
import getpass
import json
import os
from pathlib import Path
from typing import Any

import click

from .config import ControlPlaneConfig
from .domain import CallerContext, ControlError, domain_payload
from .rbac import ROLE_PERMISSIONS, Role


@click.group("control")
def control_cli() -> None:
    """Run the eInnovator PRA fleet and governance Control Plane."""


@control_cli.command("serve")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="YAML Control Plane configuration.")
@click.option("--host", help="Override the configured bind address.")
@click.option("--port", type=click.IntRange(1, 65535), help="Override the configured TCP port.")
@click.option("--public-url", help="Browser-visible URL used for SSO callbacks.")
@click.option("--reload", is_flag=True, help="Reload the development server after source changes.")
def serve(config_path: Path | None, host: str | None, port: int | None, public_url: str | None, reload: bool) -> None:
    """Start the authenticated Control Plane backend and web application."""
    overrides: dict[str, Any] = {}
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if public_url is not None:
        overrides["public_url"] = public_url
    config = ControlPlaneConfig.load(config_path, overrides=overrides)
    config.validate_security()
    try:
        import uvicorn
    except ImportError as error:
        raise click.ClickException("Install the 'control-plane' optional dependency.") from error
    os.environ["PRA_CONTROL_CONFIG"] = str(config_path or "")
    uvicorn.run(
        "pra_control.server:app" if reload else _factory(config),
        host=config.host, port=config.port, reload=reload,
        factory=reload,
    )


@control_cli.command("mcp")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="YAML Control Plane configuration.")
@click.option("--transport", type=click.Choice(["stdio", "http"]), default="stdio", show_default=True)
@click.option("--host", help="Override the MCP HTTP bind address.")
@click.option("--port", type=click.IntRange(1, 65535), help="Override the MCP HTTP port.")
def mcp(config_path: Path | None, transport: str, host: str | None, port: int | None) -> None:
    """Start the manager-backed MCP server over stdio or streamable HTTP."""
    config = ControlPlaneConfig.load(config_path)
    config.validate_security()
    if not config.mcp.enabled:
        raise click.ClickException("MCP is disabled; set control_plane.mcp.enabled=true")
    from .app import ControlRuntime
    from .mcp import build_http_app, stdio_presentation

    runtime = ControlRuntime(config)
    if transport == "stdio":
        if not config.mcp.transports.stdio.enabled:
            raise click.ClickException("MCP stdio transport is disabled")
        _, server = stdio_presentation(runtime.manager, config)
        server.run(transport="stdio")
        return
    if not config.mcp.transports.http.enabled:
        raise click.ClickException("MCP HTTP transport is disabled")
    try:
        import uvicorn
    except ImportError as error:
        raise click.ClickException("Install the 'control-plane' optional dependency.") from error
    settings = config.mcp.transports.http
    uvicorn.run(build_http_app(runtime.manager, config), host=host or settings.host, port=port or settings.port)


def _common_options(function):
    function = click.option("--auth-profile", help="Named service identity from control_plane.auth_profiles.")(function)
    function = click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="YAML Control Plane configuration.")(function)
    return function


def _embedded_caller(config: ControlPlaneConfig, profile_name: str | None) -> CallerContext:
    if profile_name:
        from .mcp import caller_from_profile
        try:
            profile = config.auth_profiles[profile_name]
        except KeyError as error:
            raise click.ClickException(f"Unknown auth profile: {profile_name}") from error
        return caller_from_profile(profile, transport="cli", supplied_token=profile.token())
    permissions = {permission.value for permission in ROLE_PERMISSIONS[Role.VIEWER]}
    return CallerContext(
        subject=f"os:{getpass.getuser()}", roles=[Role.VIEWER.value], permissions=permissions,
        auth_source="local_os", transport="cli",
    )


async def _embedded(config: ControlPlaneConfig, profile_name: str | None, operation):
    from .fleet import FleetService
    from .managers import ControlManager
    from .persistence import ControlStore
    store = ControlStore(config.database_url)
    fleet = FleetService(config, store)
    manager = ControlManager.build(config, store, fleet)
    try:
        return await operation(manager, _embedded_caller(config, profile_name))
    finally:
        await fleet.close()


def _run_embedded(config: ControlPlaneConfig, profile_name: str | None, operation):
    """Run one manager operation and present domain failures as concise CLI errors."""
    try:
        return asyncio.run(_embedded(config, profile_name, operation))
    except ControlError as error:
        raise click.ClickException(str(error)) from error


def _emit(value: Any) -> None:
    click.echo(json.dumps(domain_payload(value), indent=2, sort_keys=True))


@control_cli.command("fleet")
@_common_options
def fleet(config_path: Path | None, auth_profile: str | None) -> None:
    """List fleet state through the embedded manager."""
    config = ControlPlaneConfig.load(config_path)
    _emit(_run_embedded(config, auth_profile, lambda manager, caller: manager.fleet.list(caller)))


@control_cli.command("inspect")
@click.argument("instance_id")
@click.option("--section", default="summary", show_default=True)
@_common_options
def inspect(instance_id: str, section: str, config_path: Path | None, auth_profile: str | None) -> None:
    """Inspect one engine without routing through REST."""
    config = ControlPlaneConfig.load(config_path)
    _emit(_run_embedded(config, auth_profile, lambda manager, caller: manager.fleet.inspect(caller, instance_id, section)))


@control_cli.command("context")
@click.argument("task")
@click.option("--repository")
@_common_options
def context(task: str, repository: str | None, config_path: Path | None, auth_profile: str | None) -> None:
    """Assemble deterministic task context through the manager."""
    config = ControlPlaneConfig.load(config_path)
    _emit(_run_embedded(config, auth_profile, lambda manager, caller: manager.context.assemble(caller, task=task, repository=repository)))


@control_cli.command("plan")
@click.argument("action")
@click.argument("target")
@click.option("--values", default="{}", help="Requested change as a JSON object.")
@click.option("--idempotency-key")
@_common_options
def plan(action: str, target: str, values: str, idempotency_key: str | None, config_path: Path | None, auth_profile: str | None) -> None:
    """Create a durable action plan without applying it."""
    config = ControlPlaneConfig.load(config_path)
    try:
        payload = json.loads(values)
        if not isinstance(payload, dict):
            raise ValueError("values must be an object")
    except (json.JSONDecodeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _emit(_run_embedded(config, auth_profile, lambda manager, caller: manager.actions.plan(caller, action, target, payload, idempotency_key=idempotency_key)))


@control_cli.command("apply")
@click.argument("plan_id")
@click.option("--reason", required=True)
@click.option("--confirm", is_flag=True)
@click.option("--idempotency-key")
@_common_options
def apply(plan_id: str, reason: str, confirm: bool, idempotency_key: str | None, config_path: Path | None, auth_profile: str | None) -> None:
    """Apply a durable plan using manager authorization and central audit."""
    config = ControlPlaneConfig.load(config_path)
    _emit(_run_embedded(config, auth_profile, lambda manager, caller: manager.actions.apply(
        caller, plan_id, confirmation=confirm, reason=reason, idempotency_key=idempotency_key,
    )))


def _factory(config: ControlPlaneConfig):
    from .app import create_app
    return create_app(config)


def main() -> None:
    control_cli(prog_name="pra-control")


if __name__ == "__main__":
    main()
