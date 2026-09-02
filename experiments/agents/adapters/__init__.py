"""Coding-agent adapter implementations."""

from .base import AgentExecution, AgentTask, CodingAgentAdapter
from .command import CommandAgentAdapter
from .fixture import FixtureAgentAdapter
from .registry import command_adapter, command_spec, load_command_manifest

__all__ = [
    "AgentExecution", "AgentTask", "CodingAgentAdapter", "CommandAgentAdapter",
    "FixtureAgentAdapter", "command_adapter", "command_spec", "load_command_manifest",
]
