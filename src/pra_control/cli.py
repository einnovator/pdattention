"""Command-line entry point for the Enterprise PRA Control Plane."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click

from .config import ControlPlaneConfig


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


def _factory(config: ControlPlaneConfig):
    from .app import create_app
    return create_app(config)


def main() -> None:
    control_cli(prog_name="pra-control")


if __name__ == "__main__":
    main()
