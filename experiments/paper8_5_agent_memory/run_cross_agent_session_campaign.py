"""Run one agent through a boundary-free N-task Paper 8.5 session.

The runner keeps policy parameters transverse to agents.  Agent-specific code
is limited to native session continuation and event normalization; the same
OpenAI-tool selection proxy consumes every model request.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Callable

from .autonomous_proxy import AutonomousSelectionConfig, AutonomousSelectionProxy
from .multi_issue_session import BoundaryMode
from .run_autonomous_swebench import _exact_token_counter
from .run_kilo_swebench import build_parser as kilo_parser, run as run_kilo
from .run_openhands_swebench import (
    build_parser as openhands_parser,
    run as run_openhands,
)
from .run_opencode_swebench import (
    build_parser as opencode_parser,
    run as run_opencode,
)
from .run_pi_swebench import build_parser as pi_parser, run as run_pi
from .serve_agent_transfer_proxy import _tool_semantics


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _aggregate_predictions(output: Path, episodes: list[dict[str, Any]]) -> Path:
    """Create the single immutable prediction file consumed by SWE-bench.

    Each agent runner writes one episode-local mapping.  Session campaigns must
    be graded as one declared cohort, so fail closed on missing, malformed or
    duplicate predictions rather than silently grading a partial session.
    """
    predictions: dict[str, Any] = {}
    for episode in episodes:
        instance_id = str(episode["instance_id"])
        path = Path(str(episode["output"])) / "preds.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {instance_id}:
            raise ValueError(
                f"episode prediction identity mismatch for {instance_id}: {path}"
            )
        if instance_id in predictions:
            raise ValueError(f"duplicate prediction identity: {instance_id}")
        predictions[instance_id] = value[instance_id]
    path = output / "preds.json"
    _write_json(path, predictions)
    return path


def _tasks(path: Path, count: int, repository: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("tasks")
    if not isinstance(rows, list) or count < 1 or count > len(rows):
        raise ValueError("task registry/count is invalid")
    result = []
    for row in rows[:count]:
        instance_id = str(row.get("instance_id") or "")
        reference = Path(str(row.get("reference_artifact") or ""))
        if not reference.is_absolute():
            reference = repository / reference
        if not instance_id or not reference.is_file():
            raise ValueError(f"invalid task registry row: {row!r}")
        result.append({
            "instance_id": instance_id,
            "reference_trajectory": str(reference.resolve()),
        })
    return result


def _policy(args: argparse.Namespace, tokenizer_identity: str) -> AutonomousSelectionConfig:
    values: dict[str, Any] = {
        "policy": args.policy,
        "budget_fraction": args.budget_fraction,
        "protected_head_turns": args.protected_head_turns,
        "protected_tail_turns": args.protected_tail_turns,
        "matched_tail_boundary_compaction": "declared_safe",
        "expected_model": args.model,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_calls": args.max_calls * args.task_count,
        "max_completion_tokens": 1024,
        "tokenizer_identity": tokenizer_identity,
        "task_id": args.session_id,
        "session_id": args.session_id,
        "input_protocol": "openai_tools",
        "tool_semantics_by_name": _tool_semantics(args.tool_semantics),
        "fill_missing_generation_parameters": True,
        "boundary_mode": BoundaryMode.BOUNDARY_FREE,
        "frontier_recent_user_prompts": args.frontier_recent_user_prompts,
        "frontier_protocol_exemplars": args.frontier_protocol_exemplars,
        "frontier_allow_heuristic": args.frontier_allow_heuristic,
        "keep_completed_task_statements": True,
    }
    return AutonomousSelectionConfig(**values)


def _agent_arguments(
    args: argparse.Namespace,
    task: dict[str, str],
    output: Path,
    session_store: Path,
    continuation_id: str | None,
) -> tuple[Callable[[argparse.Namespace], Path], argparse.Namespace]:
    common = [
        "--instance-id", task["instance_id"],
        "--reference-trajectory", task["reference_trajectory"],
        "--output", str(output),
        "--arm", args.arm,
        "--model-revision", args.model_revision,
        "--ollama-tags-url", args.ollama_tags_url,
        "--docker", args.docker,
        "--agent-timeout-seconds", str(args.agent_timeout_seconds),
    ]
    if args.agent == "openhands":
        values = [
            *common,
            "--base-url", args.container_proxy_url,
            "--model", f"openai/{args.model}",
            "--served-model", args.model,
            "--tokenizer", args.tokenizer,
            "--tokenizer-revision", args.tokenizer_revision,
            "--session-store", str(session_store),
            "--max-iterations", str(args.max_calls),
        ]
        if continuation_id:
            values += ["--conversation-id", continuation_id]
        return run_openhands, openhands_parser().parse_args(values)
    if args.agent == "opencode":
        values = [
            *common,
            "--model-config", args.model_config,
            "--model", f"pra/{args.model}",
            "--served-model", args.model,
            "--tokenizer", args.tokenizer,
            "--tokenizer-revision", args.tokenizer_revision,
            "--session-store", str(session_store),
        ]
        if continuation_id:
            values += ["--session-id", continuation_id]
        return run_opencode, opencode_parser().parse_args(values)
    if args.agent == "pi":
        values = [
            *common,
            "--model-config", args.model_config,
            "--model", args.model,
            "--served-model", args.model,
            "--provider", "paper85-proxy",
            "--session-store", str(session_store),
        ]
        if continuation_id:
            values.append("--continue-session")
        return run_pi, pi_parser().parse_args(values)
    if args.agent == "kilo":
        values = [
            *common,
            "--model-config", args.model_config,
            "--model", f"openai-compatible/{args.model}",
            "--served-model", args.model,
            "--session-store", str(session_store),
        ]
        if continuation_id:
            values.append("--continue-session")
        return run_kilo, kilo_parser().parse_args(values)
    raise ValueError(f"unsupported agent: {args.agent}")


def _continuation_id(agent: str, manifest: dict[str, Any], ordinal: int) -> str:
    if agent == "openhands":
        value = manifest["native_session"]["observed_conversation_id"]
    elif agent == "opencode":
        value = manifest["native_session"]["observed_session_id"]
    else:
        value = f"dedicated-store-episode-{ordinal}"
    if not isinstance(value, str) or not value:
        raise RuntimeError("agent emitted no native continuation identity")
    return value


def _prepare_resume(
    args: argparse.Namespace,
    *,
    output: Path,
    tasks: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], str | None, int, int]:
    """Copy a failed partial campaign into a new immutable retry directory."""

    if not args.resume_from:
        return [], None, 0, 0
    source = Path(args.resume_from).resolve()
    if source == output or not source.is_dir():
        raise ValueError("--resume-from must name a different campaign directory")
    state_path = source / "campaign_state.json"
    if not state_path.is_file():
        raise ValueError("resume source has no campaign_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_parameters = {
        "budget_fraction": args.budget_fraction,
        "protected_head_turns": args.protected_head_turns,
        "protected_tail_turns": args.protected_tail_turns,
        "frontier_recent_user_prompts": args.frontier_recent_user_prompts,
        "frontier_protocol_exemplars": args.frontier_protocol_exemplars,
        "frontier_allow_heuristic": args.frontier_allow_heuristic,
    }
    identity = {
        "agent": args.agent,
        "arm": args.arm,
        "session_id": args.session_id,
        "task_count_declared": len(tasks),
        "task_order": [row["instance_id"] for row in tasks],
        "policy": args.policy,
        "policy_parameters": expected_parameters,
    }
    mismatches = {
        key: {"expected": value, "observed": state.get(key)}
        for key, value in identity.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume campaign identity mismatch: {mismatches}")
    completed = int(state.get("task_count_completed") or 0)
    source_episodes = state.get("episodes") or []
    if completed < 1 or completed >= len(tasks) or len(source_episodes) != completed:
        raise ValueError("resume source has no valid incomplete task prefix")

    episodes: list[dict[str, Any]] = []
    for ordinal, row in enumerate(source_episodes, 1):
        expected_id = tasks[ordinal - 1]["instance_id"]
        if row.get("ordinal") != ordinal or row.get("instance_id") != expected_id:
            raise ValueError("resume episode prefix identity mismatch")
        source_episode = source / f"episode-{ordinal:02d}"
        target_episode = output / f"episode-{ordinal:02d}"
        manifest_path = source_episode / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"resume episode lacks manifest: {source_episode}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_model = str(
            manifest.get("served_model") or manifest.get("model") or ""
        )
        if observed_model.split("/", 1)[-1] != args.model:
            raise ValueError(
                f"resume model mismatch: expected {args.model!r}, "
                f"observed {observed_model!r}"
            )
        if str(manifest.get("model_revision") or "") != args.model_revision:
            raise ValueError("resume model revision mismatch")
        shutil.copytree(source_episode, target_episode)
        copied = dict(row)
        copied["output"] = str(target_episode)
        episodes.append(copied)

    source_store = source / "native_session"
    if not source_store.is_dir():
        raise ValueError("resume source has no native session store")
    shutil.copytree(source_store, output / "native_session")
    source_trace = source / "proxy_trace.jsonl"
    if not source_trace.is_file():
        raise ValueError("resume source has no proxy trace")
    shutil.copy2(source_trace, output / "proxy_trace.jsonl")
    trace_rows = [
        json.loads(line)
        for line in source_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(row.get("request_index") or 0) for row in trace_rows]
    if indices != list(range(1, len(trace_rows) + 1)):
        raise ValueError("resume trace request indices are not contiguous")
    successful = sum(
        200 <= int(row.get("upstream_status") or 0) < 300 for row in trace_rows
    )
    continuation_id = str(source_episodes[-1].get("continuation_id") or "")
    if not continuation_id:
        raise ValueError("resume source has no continuation identity")
    return episodes, continuation_id, len(trace_rows), successful


def run(args: argparse.Namespace) -> Path:
    if args.agent != "openhands" and not args.model_config:
        raise ValueError(f"--model-config is required for {args.agent}")
    repository = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = _tasks(Path(args.task_registry).resolve(), args.task_count, repository)
    episodes, continuation_id, initial_requests, initial_successes = _prepare_resume(
        args, output=output, tasks=tasks
    )
    session_store = output / "native_session"
    count_tokens, tokenizer_identity = _exact_token_counter(
        args.tokenizer, args.tokenizer_revision, allow_whitespace=False
    )
    config = _policy(args, tokenizer_identity)
    trace = output / "proxy_trace.jsonl"
    proxy = AutonomousSelectionProxy(
        args.upstream,
        config=config,
        trace_path=trace,
        count_tokens=count_tokens,
        timeout_seconds=args.upstream_timeout_seconds,
        initial_request_count=initial_requests,
        initial_successful_request_count=initial_successes,
    )
    proxy.start("0.0.0.0", args.proxy_port)
    started = datetime.now(timezone.utc)
    error: BaseException | None = None
    try:
        for ordinal, task in enumerate(tasks[len(episodes):], len(episodes) + 1):
            episode = output / f"episode-{ordinal:02d}"
            runner, runner_args = _agent_arguments(
                args, task, episode, session_store, continuation_id
            )
            runner(runner_args)
            manifest = json.loads(
                (episode / "run_manifest.json").read_text(encoding="utf-8")
            )
            continuation_id = _continuation_id(args.agent, manifest, ordinal)
            episodes.append({
                "ordinal": ordinal,
                "instance_id": task["instance_id"],
                "output": str(episode),
                "continuation_id": continuation_id,
                "patch_bytes": manifest.get("patch_bytes"),
                "event_summary": manifest.get("event_summary"),
                "native_completion_observed": manifest.get(
                    "native_completion_observed", True
                ),
            })
    except BaseException as exc:
        error = exc
    finally:
        proxy.close()
    finished = datetime.now(timezone.utc)
    predictions_path: Path | None = None
    if error is None:
        try:
            predictions_path = _aggregate_predictions(output, episodes)
        except BaseException as exc:
            error = exc
    rows = [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if trace.exists() else []
    full_tokens = sum(int(row.get("full_tokens") or 0) for row in rows)
    selected_tokens = sum(int(row.get("materialized_tokens") or 0) for row in rows)
    state = {
        "schema_version": 1,
        "status": "failed" if error else "execution_complete_pending_grade",
        "agent": args.agent,
        "arm": args.arm,
        "session_id": args.session_id,
        "task_count_declared": len(tasks),
        "task_count_completed": len(episodes),
        "task_order": [row["instance_id"] for row in tasks],
        "policy": args.policy,
        "policy_parameters": {
            "budget_fraction": args.budget_fraction,
            "protected_head_turns": args.protected_head_turns,
            "protected_tail_turns": args.protected_tail_turns,
            "frontier_recent_user_prompts": args.frontier_recent_user_prompts,
            "frontier_protocol_exemplars": args.frontier_protocol_exemplars,
            "frontier_allow_heuristic": args.frontier_allow_heuristic,
        },
        "request_count": len(rows),
        "cumulative_full_history_tokens": full_tokens,
        "cumulative_materialized_history_tokens": selected_tokens,
        "own_saving_fraction": (
            1.0 - selected_tokens / full_tokens if full_tokens else 0.0
        ),
        "episodes": episodes,
        "autonomous_completion": all(
            row.get("native_completion_observed") is not False
            for row in episodes
        ),
        "predictions": None if predictions_path is None else str(predictions_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "error_type": None if error is None else type(error).__name__,
        "error_detail": None if error is None else str(error),
        "resumed_from": (
            None if not args.resume_from else str(Path(args.resume_from).resolve())
        ),
    }
    _write_json(output / "campaign_state.json", state)
    if error is not None:
        raise RuntimeError(
            f"cross-agent session campaign failed: {type(error).__name__}: {error}"
        ) from error
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("openhands", "pi", "kilo", "opencode"), required=True)
    parser.add_argument("--task-registry", required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--ollama-tags-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--tool-semantics", type=Path)
    parser.add_argument("--policy", default="full")
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--protected-head-turns", type=int, default=2)
    parser.add_argument("--protected-tail-turns", type=int, default=4)
    parser.add_argument("--frontier-recent-user-prompts", type=int, default=2)
    parser.add_argument("--frontier-protocol-exemplars", type=int, default=1)
    parser.add_argument("--frontier-allow-heuristic", action="store_true")
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--proxy-port", type=int, default=18185)
    parser.add_argument("--container-proxy-url", default="http://host.docker.internal:18185/v1")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--agent-timeout-seconds", type=int, default=3600)
    parser.add_argument("--upstream-timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--resume-from",
        help="copy and continue a validated incomplete campaign in a new output directory",
    )
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
