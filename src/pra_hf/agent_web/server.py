"""Uvicorn entry point kept inside the optional web dependency boundary."""

from __future__ import annotations

import argparse
from pathlib import Path


def run_server(
    *,
    host: str,
    port: int,
    profile: str | None = None,
    pra_override: str | None = None,
    config_path: str | None = None,
) -> None:
    import uvicorn

    from .app import AgentWebService, create_app
    from ..agent_profiles import AgentProfileRegistry

    registry = AgentProfileRegistry()
    if profile:
        document = registry.load(config_path=config_path)
        if profile not in document.profiles:
            raise ValueError(f"Unknown agent profile: {profile}")
    app = create_app(
        service=AgentWebService(
            registry=registry,
            config_path=config_path,
            default_profile=profile,
            pra_override=pra_override,
        )
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--profile")
    parser.add_argument("--pra")
    parser.add_argument("--config")
    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        profile=args.profile,
        pra_override=args.pra,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
