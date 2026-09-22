"""Serve the Paper 8.5 logical proxy for a standard OpenAI-tool agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading

from .autonomous_proxy import AutonomousSelectionConfig, AutonomousSelectionProxy
from .run_autonomous_swebench import _exact_token_counter


DEFAULT_TOOL_SEMANTICS = {
    "read": {
        "category": "filesystem",
        "operation_kind": "read",
        "resource_arguments": ["path", "file_path"],
    },
    "grep": {"category": "filesystem", "operation_kind": "read"},
    "find": {"category": "filesystem", "operation_kind": "search_discovery"},
    "glob": {"category": "filesystem", "operation_kind": "search_discovery"},
    "ls": {"category": "filesystem", "operation_kind": "search_discovery"},
    "edit": {
        "category": "filesystem",
        "operation_kind": "write",
        "resource_arguments": ["path", "file_path"],
    },
    "write": {
        "category": "filesystem",
        "operation_kind": "write",
        "resource_arguments": ["path", "file_path"],
    },
    "apply_patch": {
        "category": "filesystem",
        "operation_kind": "write",
        "resource_arguments": ["path", "file_path"],
    },
    # Arbitrary shell is deliberately an unknown barrier unless execution
    # middleware supplies complete resource/effect evidence.
    "bash": {"category": "shell", "operation_kind": "unknown"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--policy", default="full")
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument(
        "--tokenizer",
        help=(
            "Exact tokenizer name or local path. If omitted, retain the "
            "legacy whitespace diagnostic and label it explicitly."
        ),
    )
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--allow-whitespace-tokenizer", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18185)
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--upstream-qualification-path")
    parser.add_argument("--upstream-connect-attempts", type=int, default=1)
    parser.add_argument("--upstream-connect-retry-seconds", type=float, default=1.0)
    parser.add_argument("--upstream-curl-executable")
    parser.add_argument("--tool-semantics-json", type=Path)
    parser.add_argument("--boundary-mode", default="boundary_free")
    parser.add_argument("--protected-head-turns", type=int, default=1)
    parser.add_argument("--protected-tail-turns", type=int, default=1)
    parser.add_argument(
        "--matched-tail-boundary-compaction",
        choices=("legacy", "declared_safe", "disabled"),
        default="legacy",
        help=(
            "Whether matched-tail may rewrite an oversized boundary tool "
            "observation. Cross-agent runs should use declared_safe or disabled."
        ),
    )
    parser.add_argument("--completed-recent-turns", type=int, default=1)
    parser.add_argument("--completed-mutation-turns", type=int, default=1)
    parser.add_argument("--completed-verification-turns", type=int, default=1)
    parser.add_argument("--completed-protocol-turns", type=int, default=0)
    parser.add_argument("--completed-finalization-turns", type=int, default=0)
    parser.add_argument("--completed-instruction-epochs", type=int, default=0)
    parser.add_argument(
        "--compact-completed-finalizations", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--retire-closed-instructions", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--frontier-recent-user-prompts", type=int, default=2)
    parser.add_argument("--frontier-protocol-exemplars", type=int, default=0)
    parser.add_argument("--frontier-workflow-exemplars", type=int, default=0)
    parser.add_argument(
        "--frontier-allow-heuristic", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--keep-completed-task-statements", action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _tool_semantics(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {name: dict(value) for name, value in DEFAULT_TOOL_SEMANTICS.items()}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tool_semantics_by_name" in payload:
        payload = payload["tool_semantics_by_name"]
    if not isinstance(payload, dict):
        raise ValueError("tool semantics JSON must be an object keyed by tool name")
    if not all(isinstance(name, str) and isinstance(value, dict)
               for name, value in payload.items()):
        raise ValueError("every tool semantics entry must be an object")
    return {name: dict(value) for name, value in payload.items()}


def build_config(
    args: argparse.Namespace, *, tokenizer_identity: str | None = None,
) -> AutonomousSelectionConfig:
    return AutonomousSelectionConfig(
        policy=args.policy,
        budget_fraction=args.budget_fraction,
        protected_head_turns=args.protected_head_turns,
        protected_tail_turns=args.protected_tail_turns,
        matched_tail_boundary_compaction=args.matched_tail_boundary_compaction,
        input_protocol="openai_tools",
        tool_semantics_by_name=_tool_semantics(args.tool_semantics_json),
        fill_missing_generation_parameters=True,
        expected_model=args.model,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_completion_tokens=args.max_completion_tokens,
        tokenizer_identity=(
            tokenizer_identity or "whitespace_v1_diagnostic_only"
        ),
        max_calls=args.max_calls,
        task_id=args.task_id,
        completed_recent_turns=args.completed_recent_turns,
        completed_mutation_turns=args.completed_mutation_turns,
        completed_verification_turns=args.completed_verification_turns,
        completed_protocol_turns=args.completed_protocol_turns,
        completed_finalization_turns=args.completed_finalization_turns,
        compact_completed_finalizations=args.compact_completed_finalizations,
        retire_closed_instructions=args.retire_closed_instructions,
        completed_instruction_epochs=args.completed_instruction_epochs,
        frontier_recent_user_prompts=args.frontier_recent_user_prompts,
        frontier_protocol_exemplars=args.frontier_protocol_exemplars,
        frontier_workflow_exemplars=args.frontier_workflow_exemplars,
        frontier_allow_heuristic=args.frontier_allow_heuristic,
        keep_completed_task_statements=args.keep_completed_task_statements,
        boundary_mode=args.boundary_mode,
    )


def main() -> None:
    args = build_parser().parse_args()

    tokenizer_name = args.tokenizer or "whitespace"
    tokenizer_revision = args.tokenizer_revision or (
        "diagnostic-only" if tokenizer_name == "whitespace" else "main"
    )
    count_tokens, tokenizer_identity = _exact_token_counter(
        tokenizer_name,
        tokenizer_revision,
        allow_whitespace=(
            args.allow_whitespace_tokenizer or args.tokenizer is None
        ),
    )
    config = build_config(args, tokenizer_identity=tokenizer_identity)
    proxy = AutonomousSelectionProxy(
        args.upstream,
        config=config,
        trace_path=Path(args.trace),
        count_tokens=count_tokens,
        timeout_seconds=3600,
        upstream_qualification_path=args.upstream_qualification_path,
        upstream_connect_attempts=args.upstream_connect_attempts,
        upstream_connect_retry_seconds=args.upstream_connect_retry_seconds,
        upstream_curl_executable=args.upstream_curl_executable,
    )
    endpoint = proxy.start(args.host, args.port)
    print(json.dumps({
        "endpoint": endpoint,
        "policy": args.policy,
        "input_protocol": "openai_tools",
        "policy_parameters": {
            "boundary_mode": config.boundary_mode.value,
            "protected_head_turns": config.protected_head_turns,
            "protected_tail_turns": config.protected_tail_turns,
            "matched_tail_boundary_compaction": (
                config.matched_tail_boundary_compaction
            ),
            "completed_recent_turns": config.completed_recent_turns,
            "completed_mutation_turns": config.completed_mutation_turns,
            "completed_verification_turns": config.completed_verification_turns,
            "completed_protocol_turns": config.completed_protocol_turns,
            "completed_instruction_epochs": config.completed_instruction_epochs,
            "compact_completed_finalizations": config.compact_completed_finalizations,
            "frontier_recent_user_prompts": config.frontier_recent_user_prompts,
            "frontier_protocol_exemplars": config.frontier_protocol_exemplars,
            "frontier_workflow_exemplars": config.frontier_workflow_exemplars,
            "frontier_allow_heuristic": config.frontier_allow_heuristic,
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_completion_tokens": config.max_completion_tokens,
        },
        "tokenizer": tokenizer_identity,
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
