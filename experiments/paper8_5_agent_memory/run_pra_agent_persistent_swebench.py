"""Run several isolated SWE-bench issues in one persistent PRA Agent session.

Each issue receives a fresh official workspace container, while the agent,
model conversation, typed record identities, and history policy remain one
continuous session.  This is the boundary-free N-task experiment: task
prompts are ordinary user messages, not policy-side reset signals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Sequence

from pra_hf import (
    AgentConfig,
    CapabilitySDK,
    LocalSessionService,
    NegotiatedRemoteBackend,
    PRAAgent,
    PRAAgentConfig,
    PRARuntime,
    PRARuntimeConfig,
)
from pra_hf.agent_transport import TEXT_TOOL_OBSERVATION_PROJECTION

from .pra_agent_policy import (
    PRAAgentInstructionEpochSelector,
    PRAAgentMatchedTailSelector,
)
from .run_autonomous_swebench import _exact_token_counter
from .run_pi_swebench import (
    _sha256,
    _write_json,
    load_locked_task,
    observed_model_identity,
    swebench_image,
    task_prompt,
)
from .run_pra_agent_swebench import (
    DockerWorkspaceTools,
    PRA_SWE_AGENT_BEHAVIOR,
    SWEInspectionBudgetGuard,
    _durable_tool_events,
    _run,
)


def _task_spec(value: str) -> tuple[str, Path]:
    """Parse one INSTANCE_ID=REFERENCE_TRAJECTORY declaration."""

    instance_id, separator, trajectory = value.partition("=")
    if not separator or not instance_id.strip() or not trajectory.strip():
        raise argparse.ArgumentTypeError(
            "task must be INSTANCE_ID=REFERENCE_TRAJECTORY"
        )
    return instance_id.strip(), Path(trajectory).expanduser().resolve()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _saving_fraction(full_tokens: int, materialized_tokens: int) -> float:
    if full_tokens < 0 or materialized_tokens < 0:
        raise ValueError("token counts cannot be negative")
    return 0.0 if not full_tokens else 1.0 - materialized_tokens / full_tokens


def _initial_task_description(boundary_mode: str, prompt: str) -> str | None:
    """Seed task-graph state only for the explicitly bounded control.

    A boundary-free session represents every issue solely as an ordinary user
    message.  Keeping the first issue as an active task-graph node would leak a
    stale ``Active task: task-1`` control label into all later episodes.
    """

    if boundary_mode == "boundary_free":
        return None
    if boundary_mode == "explicit_task":
        return prompt
    raise ValueError(f"unknown boundary mode: {boundary_mode}")


def _load_task_registry(
    path: Path,
    *,
    repository: Path,
    task_count: int | None = None,
) -> list[tuple[str, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("task registry must contain a non-empty tasks array")
    tasks: list[tuple[str, Path]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("task registry rows must be objects")
        instance_id = str(row.get("instance_id") or "")
        reference_text = str(row.get("reference_artifact") or "")
        if not instance_id or not reference_text:
            raise ValueError("task registry row lacks identity or reference")
        reference = Path(reference_text)
        if not reference.is_absolute():
            reference = repository / reference
        tasks.append((instance_id, reference.resolve()))
    if len({row[0] for row in tasks}) != len(tasks):
        raise ValueError("task registry contains duplicate instance identities")
    if task_count is not None:
        if task_count < 1 or task_count > len(tasks):
            raise ValueError("task_count is outside the task registry")
        tasks = tasks[:task_count]
    return tasks


def _start_container(
    docker: str,
    instance_id: str,
    *,
    timeout: int,
    output: Path,
    ordinal: int,
) -> str:
    container = f"paper85-pra-persistent-{ordinal:02d}-{_slug(instance_id)}-{int(time.time())}"
    start = _run((
        docker,
        "run",
        "-d",
        "--name",
        container,
        "--platform",
        "linux/amd64",
        "-w",
        "/testbed",
        swebench_image(instance_id),
        "sleep",
        "2h",
    ), timeout=timeout)
    (output / "container_start.stdout.log").write_bytes(start.stdout)
    (output / "container_start.stderr.log").write_bytes(start.stderr)
    if start.returncode:
        raise RuntimeError(
            f"task {ordinal} container failed to start: {start.returncode}"
        )
    return container


def _tool_events(turn: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if turn is None:
        return rows
    for index, execution in enumerate(turn.tool_executions, start=1):
        result = execution.execution
        rows.append({
            "index": index,
            "accepted": result.accepted,
            "executed": result.executed,
            "reason": result.reason,
            "resource_uri": result.resource_uri,
            "tool_name": None if result.call is None else result.call.name,
            "arguments": (
                None if result.call is None else dict(result.call.arguments)
            ),
            "output": dict(result.output),
            "record": execution.record.compact_view(),
        })
    return rows


def run(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parents[2]
    if args.task and args.task_registry:
        raise ValueError("Use repeated --task or --task-registry, not both.")
    declared_tasks = list(args.task or ())
    if args.task_registry:
        declared_tasks = _load_task_registry(
            Path(args.task_registry).resolve(),
            repository=repository,
            task_count=args.task_count,
        )
    elif args.task_count is not None:
        raise ValueError("--task-count requires --task-registry")
    if not declared_tasks:
        raise ValueError("At least one --task or --task-registry is required.")
    benchmark = Path(args.benchmark_card).resolve()
    tasks: list[tuple[str, Path, int, str]] = []
    for instance_id, trajectory in declared_tasks:
        _, locked_id, task_index = load_locked_task(benchmark, instance_id)
        if not trajectory.is_file():
            raise FileNotFoundError(trajectory)
        tasks.append((locked_id, trajectory, task_index, task_prompt(
            trajectory, locked_id
        )))

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    episodes_root = output / "episodes"
    episodes_root.mkdir()
    _write_json(output / "session_plan.json", {
        "schema_version": 1,
        "boundary_mode": args.boundary_mode,
        "tasks": [
            {
                "ordinal": index,
                "instance_id": instance_id,
                "reference_trajectory": str(trajectory),
                "reference_trajectory_sha256": _sha256(trajectory.read_bytes()),
            }
            for index, (instance_id, trajectory, _, _) in enumerate(tasks, 1)
        ],
    })

    model_identity = observed_model_identity(args)
    count_tokens, tokenizer_identity = _exact_token_counter(
        args.tokenizer,
        args.tokenizer_revision,
        allow_whitespace=(
            args.allow_whitespace_tokenizer or args.tokenizer == "whitespace"
        ),
    )
    retention_fraction = (
        args.retention_fraction
        if args.history_policy == "matched_token_tail"
        else 1.0
    )
    if args.history_policy == "instruction_epoch":
        history_selector = PRAAgentInstructionEpochSelector(
            count_tokens=count_tokens,
            tokenizer_identity=tokenizer_identity,
            prior_full_epochs=args.prior_full_epochs,
            prior_recent_turns=args.prior_recent_turns,
            prior_mutation_turns=args.prior_mutation_turns,
            prior_verification_turns=args.prior_verification_turns,
            prior_protocol_turns=args.prior_protocol_turns,
            prior_finalization_turns=args.prior_finalization_turns,
        )
    else:
        history_selector = PRAAgentMatchedTailSelector(
            retention_fraction=retention_fraction,
            count_tokens=count_tokens,
            tokenizer_identity=tokenizer_identity,
            protected_head_turns=args.protected_head_turns,
            protected_tail_turns=args.protected_tail_turns,
        )

    tenant_id = "paper8-5"
    workspace = DockerWorkspaceTools(args.docker, "unbound")
    tools = workspace.toolset(tenant_id)
    capabilities = CapabilitySDK(AgentConfig(
        tools=tools.records,
        namespace="pra-agent",
        tenant_id=tenant_id,
        max_candidates=len(tools.records),
    ))
    backend = NegotiatedRemoteBackend(
        args.endpoint,
        args.model,
        transport="text",
        timeout_seconds=args.model_timeout_seconds,
        curl_executable=args.curl_executable,
        openai_fields={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
    )
    runtime = PRARuntime(
        config=PRARuntimeConfig(),
        backend=backend,
        capability_sdk=capabilities,
        executor=tools.executor(),
        session_service=LocalSessionService(output / "live_session"),
    )
    agent = PRAAgent(
        runtime,
        config=PRAAgentConfig(
            user_id="paper8-5",
            tenant_id=tenant_id,
            task_scope="session",
            context_records=args.context_records,
            tool_candidates=len(tools.records),
            max_tool_rounds=args.max_tool_rounds,
            allow_writes=True,
            max_new_tokens=args.max_completion_tokens,
            behavior_instructions=PRA_SWE_AGENT_BEHAVIOR,
        ),
        toolset=tools,
        history_selector=history_selector,
        tool_call_guard=SWEInspectionBudgetGuard(workspace),
        completion_guard=lambda _state, _text, _executions: (
            workspace.completion_rejection()
        ),
    )

    session_started = datetime.now(timezone.utc)
    predictions: dict[str, dict[str, str]] = {}
    episode_manifests: list[dict[str, object]] = []
    prefix_full_tokens = 0
    prefix_materialized_tokens = 0
    error: Exception | None = None
    active_container: str | None = None
    try:
        for ordinal, (instance_id, trajectory, task_index, prompt) in enumerate(
            tasks, 1
        ):
            episode = episodes_root / f"{ordinal:02d}-{_slug(instance_id)}"
            episode.mkdir()
            (episode / "task_prompt.txt").write_text(
                prompt + "\n", encoding="utf-8"
            )
            active_container = _start_container(
                args.docker,
                instance_id,
                timeout=args.image_timeout_seconds,
                output=episode,
                ordinal=ordinal,
            )
            workspace.bind_container(active_container)
            if ordinal == 1:
                agent.start_session(
                    args.session_id or f"paper85-pra-persistent-{int(time.time())}",
                    task_description=_initial_task_description(
                        args.boundary_mode, prompt
                    ),
                )
                if (
                    args.boundary_mode == "boundary_free"
                    and agent.state.active_task_id is not None
                ):
                    raise AssertionError(
                        "boundary-free session unexpectedly has an active task"
                    )
            elif args.boundary_mode == "explicit_task":
                agent.create_task(
                    prompt,
                    task_id=f"issue-{ordinal:02d}-{_slug(instance_id)}",
                    activate=True,
                )

            trace_start = len(history_selector.traces)
            durable_record_start = len(agent.state.records)
            episode_started = datetime.now(timezone.utc)
            turn = None
            episode_error: Exception | None = None
            try:
                turn = agent.run_turn(prompt)
            except Exception as observed:
                episode_error = observed
            episode_finished = datetime.now(timezone.utc)
            traces = history_selector.traces[trace_start:]
            durable_records = tuple(agent.state.records[durable_record_start:])

            status = _run((
                args.docker, "exec", "-w", "/testbed", active_container,
                "git", "status", "--short",
            ))
            patch = _run((
                args.docker, "exec", "-w", "/testbed", active_container,
                "git", "diff", "--binary", "--",
            ))
            (episode / "workspace_status.txt").write_bytes(status.stdout)
            (episode / "model.patch").write_bytes(patch.stdout)
            if turn is not None:
                (episode / "final_response.txt").write_text(
                    turn.text, encoding="utf-8"
                )
            events = _tool_events(turn)
            (episode / "tool_events.jsonl").write_text(
                "".join(json.dumps(row, default=str) + "\n" for row in events),
                encoding="utf-8",
            )
            durable_events = _durable_tool_events(durable_records)
            (episode / "durable_tool_events.jsonl").write_text(
                "".join(
                    json.dumps(row, default=str) + "\n"
                    for row in durable_events
                ),
                encoding="utf-8",
            )
            _write_json(episode / "selection_trace.json", {
                "schema_version": 1,
                "requests": traces,
            })
            full_tokens = sum(int(row["full_history_tokens"]) for row in traces)
            materialized_tokens = sum(
                int(row["materialized_history_tokens"]) for row in traces
            )
            prefix_full_tokens += full_tokens
            prefix_materialized_tokens += materialized_tokens
            episode_manifest: dict[str, object] = {
                "schema_version": 1,
                "ordinal": ordinal,
                "instance_id": instance_id,
                "task_index": task_index,
                "reference_trajectory": str(trajectory),
                "container": active_container,
                "started_at": episode_started.isoformat(),
                "finished_at": episode_finished.isoformat(),
                "elapsed_seconds": (
                    episode_finished - episode_started
                ).total_seconds(),
                "request_count": len(traces),
                "tool_event_count": len(durable_events),
                "returned_turn_tool_event_count": len(events),
                "patch_bytes": len(patch.stdout),
                "patch_sha256": _sha256(patch.stdout),
                "cumulative_full_history_tokens": full_tokens,
                "cumulative_materialized_history_tokens": materialized_tokens,
                "own_saving_fraction": _saving_fraction(
                    full_tokens, materialized_tokens
                ),
                "cumulative_prefix_full_history_tokens": prefix_full_tokens,
                "cumulative_prefix_materialized_history_tokens": (
                    prefix_materialized_tokens
                ),
                "cumulative_prefix_own_saving_fraction": _saving_fraction(
                    prefix_full_tokens, prefix_materialized_tokens
                ),
                "official_resolution": None,
                "error_type": (
                    None if episode_error is None
                    else type(episode_error).__name__
                ),
                "error_detail": (
                    None if episode_error is None else str(episode_error)
                ),
            }
            _write_json(episode / "run_manifest.json", episode_manifest)
            episode_manifests.append(episode_manifest)
            predictions[instance_id] = {
                "model_name_or_path": (
                    f"pra-agent-persistent-{args.model}-{args.history_policy}"
                ),
                "model_patch": patch.stdout.decode("utf-8", errors="replace"),
                "instance_id": instance_id,
            }
            if (
                args.boundary_mode == "explicit_task"
                and agent.state.active_task_id is not None
            ):
                agent.complete_task(
                    agent.state.active_task_id,
                    result_ref=f"episode:{ordinal:02d}:model.patch",
                )
            _run((args.docker, "rm", "-f", active_container))
            active_container = None
            if episode_error is not None:
                raise episode_error
    except Exception as observed:
        error = observed
        (output / "agent_error.txt").write_text(
            f"{type(observed).__name__}: {observed}\n", encoding="utf-8"
        )
    finally:
        if agent.session is not None:
            agent.export_session(output / "session.json")
        agent.close()
        if active_container is not None:
            _run((args.docker, "rm", "-f", active_container))

    session_finished = datetime.now(timezone.utc)
    _write_json(output / "preds.json", predictions)
    _write_json(output / "selection_trace.json", {
        "schema_version": 1,
        "requests": history_selector.traces,
    })
    total_full = sum(
        int(row["full_history_tokens"]) for row in history_selector.traces
    )
    total_materialized = sum(
        int(row["materialized_history_tokens"])
        for row in history_selector.traces
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper8_5_persistent_cross_agent_transfer",
        "agent": "pra-agent",
        "text_tool_observation_projection": TEXT_TOOL_OBSERVATION_PROJECTION,
        "pre_mutation_inspection_budget": 8,
        "session_persistence": "atomic_local_json_per_record",
        "behavior_instructions": PRA_SWE_AGENT_BEHAVIOR,
        "behavior_instructions_sha256": _sha256(
            PRA_SWE_AGENT_BEHAVIOR.encode("utf-8")
        ),
        "session_id": args.session_id,
        "task_count_declared": len(tasks),
        "task_count_completed": len(episode_manifests),
        "boundary_mode": args.boundary_mode,
        "task_graph_mode": (
            "disabled" if args.boundary_mode == "boundary_free"
            else "explicit_task_events"
        ),
        "task_order": [row[0] for row in tasks],
        "model": args.model,
        "served_model": args.served_model or args.model,
        "model_revision": args.model_revision,
        "observed_model_identity": model_identity,
        "endpoint": args.endpoint,
        "history_policy": args.history_policy,
        "retention_fraction": (
            retention_fraction
            if args.history_policy != "instruction_epoch"
            else None
        ),
        "protected_head_turns": args.protected_head_turns,
        "protected_tail_turns": args.protected_tail_turns,
        "prior_full_epochs": args.prior_full_epochs,
        "prior_recent_turns": args.prior_recent_turns,
        "prior_mutation_turns": args.prior_mutation_turns,
        "prior_verification_turns": args.prior_verification_turns,
        "prior_protocol_turns": args.prior_protocol_turns,
        "prior_finalization_turns": args.prior_finalization_turns,
        "tokenizer": tokenizer_identity,
        "tokenizer_revision": args.tokenizer_revision,
        "started_at": session_started.isoformat(),
        "finished_at": session_finished.isoformat(),
        "elapsed_seconds": (session_finished - session_started).total_seconds(),
        "request_count": len(history_selector.traces),
        "tool_event_count": sum(
            int(row["tool_event_count"]) for row in episode_manifests
        ),
        "cumulative_full_history_tokens": total_full,
        "cumulative_materialized_history_tokens": total_materialized,
        "own_saving_fraction": _saving_fraction(total_full, total_materialized),
        "episodes": episode_manifests,
        "transport": backend.inspect(),
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_completion_tokens": args.max_completion_tokens,
        },
        "error_type": None if error is None else type(error).__name__,
        "error_detail": None if error is None else str(error),
    }
    _write_json(output / "run_manifest.json", manifest)
    if error is not None:
        raise RuntimeError(
            f"persistent PRA Agent failed: {type(error).__name__}: {error}"
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        action="append",
        type=_task_spec,
        help="Repeat INSTANCE_ID=REFERENCE_TRAJECTORY in session order.",
    )
    parser.add_argument("--task-registry")
    parser.add_argument(
        "--task-count",
        type=int,
        help="Run the first N locked registry tasks in canonical order.",
    )
    parser.add_argument(
        "--benchmark-card",
        default=str(
            Path(__file__).with_name("benchmarks")
            / "easy14_baseline_success14.json"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--boundary-mode",
        choices=("boundary_free", "explicit_task"),
        default="boundary_free",
        help=(
            "boundary_free exposes only ordinary user prompts; explicit_task "
            "also mutates the PRA task graph and is a separately labelled control"
        ),
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--served-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--ollama-tags-url")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--max-tool-rounds", type=int, default=80)
    parser.add_argument("--context-records", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-timeout-seconds", type=int, default=3600)
    parser.add_argument("--image-timeout-seconds", type=int, default=900)
    parser.add_argument("--curl-executable")
    parser.add_argument(
        "--history-policy",
        choices=("full", "matched_token_tail", "instruction_epoch"),
        default="full",
    )
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--protected-head-turns", type=int, default=2)
    parser.add_argument("--protected-tail-turns", type=int, default=4)
    parser.add_argument("--prior-full-epochs", type=int, default=2)
    parser.add_argument("--prior-recent-turns", type=int, default=0)
    parser.add_argument("--prior-mutation-turns", type=int, default=0)
    parser.add_argument("--prior-verification-turns", type=int, default=0)
    parser.add_argument("--prior-protocol-turns", type=int, default=0)
    parser.add_argument("--prior-finalization-turns", type=int, default=0)
    parser.add_argument("--tokenizer", default="whitespace")
    parser.add_argument("--tokenizer-revision", default="diagnostic")
    parser.add_argument("--allow-whitespace-tokenizer", action="store_true")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
