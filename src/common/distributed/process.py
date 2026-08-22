"""Isolated local-process transport."""

from __future__ import annotations

from .local import LocalTransport


class ProcessTransport(LocalTransport):
    """Semantic marker for jobs that must not execute in the coordinator process."""
