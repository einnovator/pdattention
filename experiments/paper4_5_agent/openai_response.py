"""OpenAI response semantics shared by direct PRA engine test servers."""

from __future__ import annotations

from typing import Any, Mapping

from pra_hf.deployment import PRAEngineResult, PRAWireRequest


def completion_finish_reason(
    request: PRAWireRequest,
    result: PRAEngineResult,
) -> str:
    """Distinguish semantic stop from a provider output-length ceiling.

    Transport completion is not agent completion.  A response that consumed
    the declared completion allowance must be exposed as ``length`` so an
    agent can run its ordinary format-recovery path instead of interpreting a
    truncated action as a semantic stop.
    """

    explicit = result.raw.get("finish_reason")
    if explicit in {"stop", "length", "tool_calls", "content_filter"}:
        return str(explicit)
    usage = result.raw.get("usage")
    completion_tokens = None
    if isinstance(usage, Mapping) and usage.get("completion_tokens") is not None:
        completion_tokens = int(usage["completion_tokens"])
    if (
        completion_tokens is not None
        and completion_tokens >= request.resolved_max_new_tokens
    ):
        return "length"
    return "stop"


def usage_dict(result: PRAEngineResult) -> dict[str, Any]:
    usage = result.raw.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}
