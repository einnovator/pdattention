"""Serve the Paper 8.5 logical proxy for a standard OpenAI-tool agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading

from .autonomous_proxy import AutonomousSelectionConfig, AutonomousSelectionProxy


PI_TOOL_SEMANTICS = {
    "read": {"category": "filesystem", "operation_kind": "read"},
    "grep": {"category": "filesystem", "operation_kind": "read"},
    "find": {"category": "filesystem", "operation_kind": "search_discovery"},
    "ls": {"category": "filesystem", "operation_kind": "search_discovery"},
    "edit": {"category": "filesystem", "operation_kind": "write"},
    "write": {"category": "filesystem", "operation_kind": "write"},
    # Arbitrary shell is deliberately an unknown barrier unless execution
    # middleware supplies complete resource/effect evidence.
    "bash": {"category": "shell", "operation_kind": "unknown"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--policy", default="full")
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18185)
    parser.add_argument("--max-calls", type=int, default=60)
    args = parser.parse_args()

    config = AutonomousSelectionConfig(
        policy=args.policy,
        budget_fraction=args.budget_fraction,
        input_protocol="openai_tools",
        tool_semantics_by_name=PI_TOOL_SEMANTICS,
        fill_missing_generation_parameters=True,
        expected_model=args.model,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_completion_tokens=1024,
        max_calls=args.max_calls,
        task_id=args.task_id,
    )
    proxy = AutonomousSelectionProxy(
        args.upstream,
        config=config,
        trace_path=Path(args.trace),
        timeout_seconds=3600,
    )
    endpoint = proxy.start(args.host, args.port)
    print(json.dumps({
        "endpoint": endpoint,
        "policy": args.policy,
        "input_protocol": "openai_tools",
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_completion_tokens": 1024,
        },
    }), flush=True)

    stopped = threading.Event()

    def stop(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stopped.wait()
    proxy.close()


if __name__ == "__main__":
    main()
