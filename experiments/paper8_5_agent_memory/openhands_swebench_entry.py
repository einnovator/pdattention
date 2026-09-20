"""Container entry point for a condenser-free OpenHands SWE-bench control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from pydantic import SecretStr

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.llm.utils import telemetry as openhands_telemetry
from openhands.tools.preset.default import get_default_tools


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def _install_openai_usage_compatibility() -> None:
    """Treat absent provider-specific cache counters as zero.

    OpenHands 1.49.2 checks ``model_fields_set`` before reading LiteLLM's
    ``cache_creation_tokens`` extension. Some OpenAI-compatible responses mark
    that extra field as present even though the wrapper exposes no attribute,
    aborting a valid completion in telemetry. This guard changes accounting
    only; it does not modify the request or model response.
    """

    original = openhands_telemetry.Telemetry._cache_buckets

    def safe_cache_buckets(usage: Any) -> tuple[int, int]:
        try:
            return original(usage)
        except AttributeError as error:
            if "cache_creation_tokens" not in str(error):
                raise
            details = getattr(usage, "prompt_tokens_details", None)
            return (
                int(getattr(details, "cached_tokens", 0) or 0),
                int(getattr(details, "cache_creation_tokens", 0) or 0),
            )

    openhands_telemetry.Telemetry._cache_buckets = staticmethod(safe_cache_buckets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="openai/qwen3-coder:30b")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    os.chdir("/testbed")
    _install_openai_usage_compatibility()
    llm = LLM(
        usage_id="agent",
        model=args.model,
        model_canonical_name="qwen3-coder:30b",
        base_url=args.base_url,
        api_key=SecretStr("dummy"),
        api_mode="chat",
        native_tool_calling=True,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_input_tokens=131072,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=None,
        enable_encrypted_reasoning=False,
        caching_prompt=False,
        stream=False,
    )
    agent = Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False, enable_sub_agents=False),
        system_prompt_kwargs={"cli_mode": True},
        condenser=None,
    )

    event_count = 0

    def emit(event: Any) -> None:
        nonlocal event_count
        event_count += 1
        print(json.dumps({
            "type": type(event).__name__,
            "event": _jsonable(event),
        }, sort_keys=True), flush=True)

    try:
        conversation = Conversation(
            agent=agent,
            workspace="/testbed",
            callbacks=[emit],
            persistence_dir="/tmp/paper85-openhands-conversation",
            max_iteration_per_run=args.max_iterations,
            visualizer=None,
        )
        conversation.send_message(prompt)
        conversation.run()
        print(json.dumps({
            "type": "paper85_run_summary",
            "event_count": event_count,
            "execution_status": str(conversation.state.execution_status),
            "conversation_id": str(conversation.state.id),
            "condenser": None,
        }, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        print(json.dumps({
            "type": "paper85_run_error",
            "event_count": event_count,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
