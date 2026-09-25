from pathlib import Path

from experiments.paper8_5_agent_memory.run_cross_agent_session_campaign import (
    _agent_arguments,
    _policy,
    _tasks,
    build_parser,
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
