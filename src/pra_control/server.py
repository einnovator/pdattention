"""Environment-loaded ASGI factory used by development reload and containers."""

from __future__ import annotations

import os

from .app import create_app
from .config import ControlPlaneConfig


def app():
    path = os.environ.get("PRA_CONTROL_CONFIG") or None
    config = ControlPlaneConfig.load(path)
    config.validate_security()
    return create_app(config)
