import json
from pathlib import Path

from experiments.paper8_5_agent_memory.run_cross_agent_session_campaign import (
    _aggregate_predictions,
    _agent_arguments,
    _policy,
    _prepare_resume,
    _tasks,
    build_parser,
)


def test_aggregate_predictions_preserves_all_episode_identities(tmp_path: Path) -> None:
    import json

    episodes = []
    for index in range(3):
        instance_id = f"org__task-{index}"
        output = tmp_path / f"episode-{index}"
        output.mkdir()
        (output / "preds.json").write_text(
            json.dumps({instance_id: {"instance_id": instance_id, "model_patch": "x"}}),
            encoding="utf-8",
        )
        episodes.append({"instance_id": instance_id, "output": str(output)})
    path = _aggregate_predictions(tmp_path, episodes)
    assert list(json.loads(path.read_text(encoding="utf-8"))) == [
        "org__task-0", "org__task-1", "org__task-2",
    ]


def test_aggregate_predictions_rejects_partial_identity(tmp_path: Path) -> None:
    import json
    import pytest

    output = tmp_path / "episode"
    output.mkdir()
    (output / "preds.json").write_text(
        json.dumps({"wrong__task": {"model_patch": "x"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        _aggregate_predictions(
            tmp_path,
            [{"instance_id": "org__task", "output": str(output)}],
        )


def _args(tmp_path: Path, agent: str):
    return build_parser().parse_args([
        "--agent", agent,
        "--task-registry", str(tmp_path / "tasks.json"),
        "--task-count", "3",
        "--output", str(tmp_path / "output"),
        "--arm", "RECENT_FRONTIER_M2_P1",
        "--session-id", "session-1",
        "--upstream", "http://127.0.0.1:11435/v1",
        "--ollama-tags-url", "http://127.0.0.1:11435/api/tags",
        "--model", "qwen3-coder:30b-ctx131k",
        "--model-revision", "revision",
        "--tokenizer", "tokenizer",
        "--tokenizer-revision", "tokenizer-revision",
        "--model-config", str(tmp_path / "model.json"),
        "--policy", "frontier_dag_retirement",
        "--frontier-allow-heuristic",
    ])


def test_boundary_free_policy_budget_is_total_session_calls(tmp_path: Path) -> None:
    args = _args(tmp_path, "opencode")
    config = _policy(args, "tokenizer@revision")
    assert config.boundary_mode.value == "boundary_free"
    assert config.frontier_recent_user_prompts == 2
    assert config.frontier_protocol_exemplars == 1
    assert config.max_calls == 240


def test_pi_continuation_is_native_and_task_id_is_not_passed(tmp_path: Path) -> None:
    args = _args(tmp_path, "pi")
    task = {"instance_id": "django__django-15277", "reference_trajectory": "t.json"}
    _, first = _agent_arguments(
        args, task, tmp_path / "episode-1", tmp_path / "session", None
    )
    _, later = _agent_arguments(
        args, task, tmp_path / "episode-2", tmp_path / "session", "store-episode-1"
    )
    assert first.continue_session is False
    assert later.continue_session is True
    assert Path(first.settings_config).name == "pi_controlled_settings_v1.json"
    assert first.settings_config == later.settings_config
    assert not hasattr(later, "task_id")


def test_registry_uses_prefix_without_exposing_boundaries(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    paths = []
    rows = []
    for index in range(3):
        path = repository / f"task-{index}.json"
        path.write_text("{}", encoding="utf-8")
        paths.append(path)
        rows.append({
            "instance_id": f"org__task-{index}",
            "reference_artifact": path.name,
        })
    registry = tmp_path / "tasks.json"
    import json
    registry.write_text(json.dumps({"tasks": rows}), encoding="utf-8")
    tasks = _tasks(registry, 2, repository)
    assert [row["instance_id"] for row in tasks] == [
        "org__task-0", "org__task-1",
    ]
    assert "task_id" not in tasks[0]


def test_prepare_resume_copies_completed_prefix_and_preserves_counters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed"
    source.mkdir()
    episode = source / "episode-01"
    episode.mkdir()
    (episode / "run_manifest.json").write_text(json.dumps({
        "model": "qwen3-coder:30b-ctx131k",
        "model_revision": "revision",
    }), encoding="utf-8")
    (episode / "preds.json").write_text("{}", encoding="utf-8")
    (source / "native_session").mkdir()
    (source / "native_session" / "state.json").write_text("{}", encoding="utf-8")
    trace_rows = [
        {"request_index": 1, "upstream_status": 200},
        {"request_index": 2, "upstream_status": 500},
    ]
    (source / "proxy_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows),
        encoding="utf-8",
    )
    parameters = {
        "budget_fraction": 1.0,
        "protected_head_turns": 2,
        "protected_tail_turns": 4,
        "frontier_recent_user_prompts": 2,
        "frontier_protocol_exemplars": 1,
        "frontier_allow_heuristic": False,
    }
    (source / "campaign_state.json").write_text(json.dumps({
        "agent": "pi",
        "arm": "FULL",
        "session_id": "session",
        "task_count_declared": 3,
        "task_count_completed": 1,
        "task_order": ["t1", "t2", "t3"],
        "policy": "full",
        "policy_parameters": parameters,
        "episodes": [{
            "ordinal": 1,
            "instance_id": "t1",
            "output": str(episode),
            "continuation_id": "dedicated-store-episode-1",
        }],
    }), encoding="utf-8")
    output = tmp_path / "retry"
    output.mkdir()
    args = build_parser().parse_args([
        "--agent", "pi", "--task-registry", "tasks.json",
        "--task-count", "3", "--output", str(output), "--arm", "FULL",
        "--session-id", "session", "--upstream", "http://model/v1",
        "--ollama-tags-url", "http://model/api/tags",
        "--model", "qwen3-coder:30b-ctx131k", "--model-revision", "revision",
        "--tokenizer", "tokenizer", "--tokenizer-revision", "tokenizer-revision",
        "--model-config", "model.json", "--resume-from", str(source),
    ])
    tasks = [{"instance_id": row} for row in ("t1", "t2", "t3")]

    episodes, continuation_id, requests, successes = _prepare_resume(
        args, output=output, tasks=tasks
    )

    assert [row["instance_id"] for row in episodes] == ["t1"]
    assert Path(episodes[0]["output"]).parent == output
    assert continuation_id == "dedicated-store-episode-1"
    assert (output / "native_session" / "state.json").is_file()
    assert requests == 2
    assert successes == 1


def test_prepare_resume_carries_only_exact_full_prefix_into_selective_arm(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full-prefix"
    source.mkdir()
    episode = source / "episode-01"
    episode.mkdir()
    (episode / "run_manifest.json").write_text(json.dumps({
        "model": "qwen3-coder:30b-ctx131k",
        "model_revision": "revision",
    }), encoding="utf-8")
    (episode / "preds.json").write_text("{}", encoding="utf-8")
    (source / "native_session").mkdir()
    (source / "native_session" / "state.json").write_text("{}", encoding="utf-8")
    full_row = {
        "request_index": 1,
        "policy": "full",
        "exact_request_passthrough": True,
        "full_tokens": 100,
        "materialized_tokens": 100,
        "upstream_status": 200,
    }
    (source / "proxy_trace.jsonl").write_text(
        json.dumps(full_row) + "\n", encoding="utf-8"
    )
    (source / "campaign_state.json").write_text(json.dumps({
        "agent": "pi",
        "arm": "FULL",
        "session_id": "session-1",
        "task_count_declared": 3,
        "task_count_completed": 1,
        "task_order": ["t1", "t2", "t3"],
        "policy": "full",
        "episodes": [{
            "ordinal": 1,
            "instance_id": "t1",
            "output": str(episode),
            "continuation_id": "dedicated-store-episode-1",
        }],
    }), encoding="utf-8")
    output = tmp_path / "selective"
    output.mkdir()
    args = build_parser().parse_args([
        "--agent", "pi", "--task-registry", "tasks.json",
        "--task-count", "3", "--output", str(output),
        "--arm", "RECENT_FRONTIER_M2_P1", "--session-id", "session-1",
        "--upstream", "http://model/v1",
        "--ollama-tags-url", "http://model/api/tags",
        "--model", "qwen3-coder:30b-ctx131k", "--model-revision", "revision",
        "--tokenizer", "tokenizer", "--tokenizer-revision", "tokenizer-revision",
        "--model-config", "model.json", "--policy", "frontier_dag_retirement",
        "--frontier-allow-heuristic", "--carry-full-prefix-from", str(source),
    ])
    tasks = [{"instance_id": row} for row in ("t1", "t2", "t3")]

    episodes, continuation_id, requests, successes = _prepare_resume(
        args, output=output, tasks=tasks
    )

    assert episodes[0]["carried_full_prefix"] is True
    assert continuation_id == "dedicated-store-episode-1"
    assert requests == successes == 1
    assert json.loads((output / "proxy_trace.jsonl").read_text()) == full_row


def test_prepare_resume_rejects_nonexact_carried_full_prefix(tmp_path: Path) -> None:
    import pytest

    source = tmp_path / "full-prefix"
    source.mkdir()
    episode = source / "episode-01"
    episode.mkdir()
    (episode / "run_manifest.json").write_text(json.dumps({
        "model": "qwen3-coder:30b-ctx131k",
        "model_revision": "revision",
    }), encoding="utf-8")
    (episode / "preds.json").write_text("{}", encoding="utf-8")
    (source / "native_session").mkdir()
    (source / "native_session" / "state.json").write_text("{}", encoding="utf-8")
    (source / "proxy_trace.jsonl").write_text(json.dumps({
        "request_index": 1,
        "policy": "full",
        "exact_request_passthrough": False,
        "full_tokens": 100,
        "materialized_tokens": 90,
        "upstream_status": 200,
    }) + "\n", encoding="utf-8")
    (source / "campaign_state.json").write_text(json.dumps({
        "agent": "pi", "arm": "FULL", "session_id": "session-1",
        "task_count_declared": 3, "task_count_completed": 1,
        "task_order": ["t1", "t2", "t3"], "policy": "full",
        "episodes": [{
            "ordinal": 1, "instance_id": "t1", "output": str(episode),
            "continuation_id": "dedicated-store-episode-1",
        }],
    }), encoding="utf-8")
    output = tmp_path / "selective"
    output.mkdir()
    args = build_parser().parse_args([
        "--agent", "pi", "--task-registry", "tasks.json",
        "--task-count", "3", "--output", str(output),
        "--arm", "RECENT_FRONTIER_M2_P1", "--session-id", "session-1",
        "--upstream", "http://model/v1",
        "--ollama-tags-url", "http://model/api/tags",
        "--model", "qwen3-coder:30b-ctx131k", "--model-revision", "revision",
        "--tokenizer", "tokenizer", "--tokenizer-revision", "tokenizer-revision",
        "--model-config", "model.json", "--policy", "frontier_dag_retirement",
        "--frontier-allow-heuristic", "--carry-full-prefix-from", str(source),
    ])
    tasks = [{"instance_id": row} for row in ("t1", "t2", "t3")]

    with pytest.raises(ValueError, match="selected or failed requests"):
        _prepare_resume(args, output=output, tasks=tasks)
