"""Container entry point for a condenser-free OpenHands SWE-bench control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any
import urllib.error
import urllib.request

from pydantic import SecretStr

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.llm.utils import telemetry as openhands_telemetry
from openhands.tools.preset.default import get_default_tools

from execution_receipts import build_execution_receipt, capture_execution_snapshot


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


class _ExecutionReceiptBridge:
    """OpenHands event adapter for the generic PRA receipt sidechannel."""

    def __init__(self, *, workspace: Path, endpoint: str, session_id: str) -> None:
        self.workspace = workspace
        self.endpoint = endpoint
        self.session_id = session_id
        self.pending: dict[str, dict[str, Any]] = {}
        self.emitted = 0
        self.delivered = 0
        self.failed_delivery = 0

    @staticmethod
    def _operation(action: dict[str, Any]) -> tuple[str, str, dict[str, str], bool]:
        kind = str(action.get("kind") or "")
        if kind == "FileEditorAction":
            operation = {
                "view": "read",
                "create": "write",
                "str_replace": "write",
                "insert": "write",
                "undo_edit": "write",
            }.get(str(action.get("command") or ""), "unknown")
            path = action.get("path")
            resources = (
                {f"file://{Path(str(path)).resolve()}": operation}
                if path and operation in {"read", "write"} else {}
            )
            return "filesystem", operation, resources, bool(resources)
        if kind == "TaskTrackerAction":
            return "agent_progress", "progress", {}, False
        if kind == "FinishAction":
            return "agent_protocol", "finalization", {}, True
        if kind == "ThinkAction":
            return "agent_progress", "progress", {}, True
        if kind == "TerminalAction":
            return "shell", "unknown", {}, False
        return "unknown", "unknown", {}, False

    def observe(self, row_type: str, event: Any) -> dict[str, Any] | None:
        if not isinstance(event, dict):
            return None
        if row_type == "ActionEvent":
            action_id = str(event.get("id") or "")
            action = event.get("action")
            action = action if isinstance(action, dict) else {}
            if not action_id:
                return None
            category, operation, resources, scope_complete = self._operation(action)
            self.pending[action_id] = {
                "tool_call_id": str(event.get("tool_call_id") or ""),
                "category": category,
                "operation": operation,
                "resources": resources,
                "scope_complete": scope_complete,
                "pre": capture_execution_snapshot(self.workspace, resources),
            }
            return None
        if row_type != "ObservationEvent":
            return None
        action_id = str(event.get("action_id") or "")
        pending = self.pending.pop(action_id, None)
        observation = event.get("observation")
        if pending is None or not isinstance(observation, dict):
            return None
        timed_out = bool(observation.get("timeout"))
        exit_code = observation.get("exit_code")
        failed = bool(observation.get("is_error")) or (
            isinstance(exit_code, int) and exit_code != 0
        ) or timed_out
        content = json.dumps(observation.get("content"), sort_keys=True, default=str)
        result_complete = not timed_out and "<response clipped>" not in content
        receipt = build_execution_receipt(
            action_record_id=action_id,
            observation_record_ids=(str(event.get("id") or ""),),
            tool_category=pending["category"],
            operation_kind=pending["operation"],
            transport_status="timed_out" if timed_out else "completed",
            semantic_status="failed" if failed else "succeeded",
            result_complete=result_complete,
            effect_scope_complete=bool(pending["scope_complete"]),
            pre=pending["pre"],
            post=capture_execution_snapshot(
                self.workspace, pending["resources"]
            ),
            resource_kinds=pending["resources"],
            return_code=(int(exit_code) if isinstance(exit_code, int) else None),
            error_kind=(
                "timeout" if timed_out else
                "tool_error" if bool(observation.get("is_error")) else
                "nonzero_exit" if isinstance(exit_code, int) and exit_code != 0 else
                None
            ),
            cwd=(
                str(observation.get("metadata", {}).get("working_dir"))
                if isinstance(observation.get("metadata"), dict) else None
            ),
            session_id=self.session_id,
            tool_call_id=pending["tool_call_id"],
        )
        self.emitted += 1
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(receipt).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 202:
                    raise OSError(f"receipt endpoint returned HTTP {response.status}")
            self.delivered += 1
            receipt["sidechannel_delivery"] = "accepted"
        except (OSError, urllib.error.URLError) as error:
            self.failed_delivery += 1
            receipt["sidechannel_delivery"] = "failed"
            receipt["sidechannel_error"] = str(error)
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="openai/qwen3-coder:30b")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument("--request-retries", type=int, default=0)
    parser.add_argument(
        "--native-tool-calling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use provider-native OpenAI tool_calls. Disable this for providers "
            "that require OpenHands' prompt-mocked, fail-closed tool protocol."
        ),
    )
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    os.chdir("/testbed")
    _install_openai_usage_compatibility()
    llm = LLM(
        usage_id="agent",
        model=args.model,
        model_canonical_name=args.model,
        base_url=args.base_url,
        api_key=SecretStr("dummy"),
        api_mode="chat",
        native_tool_calling=args.native_tool_calling,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_input_tokens=131072,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=None,
        enable_encrypted_reasoning=False,
        caching_prompt=False,
        stream=False,
        timeout=args.request_timeout_seconds,
        num_retries=args.request_retries,
    )
    agent = Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False, enable_sub_agents=False),
        system_prompt_kwargs={"cli_mode": True},
        condenser=None,
    )

    event_count = 0
    receipt_bridge = _ExecutionReceiptBridge(
        workspace=Path("/testbed"),
        endpoint=args.base_url.rstrip("/") + "/pra/execution-receipts",
        session_id=args.session_id,
    )

    def emit(event: Any) -> None:
        nonlocal event_count
        event_count += 1
        event_value = _jsonable(event)
        row_type = type(event).__name__
        print(json.dumps({
            "type": row_type,
            "event": event_value,
        }, sort_keys=True), flush=True)
        receipt = receipt_bridge.observe(row_type, event_value)
        if receipt is not None:
            print(json.dumps({
                "type": "paper85_execution_receipt",
                "receipt": receipt,
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
            "execution_receipts_emitted": receipt_bridge.emitted,
            "execution_receipts_delivered": receipt_bridge.delivered,
            "execution_receipt_delivery_failures": receipt_bridge.failed_delivery,
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
