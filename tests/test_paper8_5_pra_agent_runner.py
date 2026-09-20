from __future__ import annotations

import pytest

from experiments.paper8_5_agent_memory.run_pra_agent_swebench import (
    DockerWorkspaceTools,
)


def test_docker_workspace_rejects_absolute_and_parent_paths() -> None:
    assert DockerWorkspaceTools._relative_path("django/db/models.py") == (
        "django/db/models.py"
    )
    with pytest.raises(PermissionError):
        DockerWorkspaceTools._relative_path("/etc/passwd")
    with pytest.raises(PermissionError):
        DockerWorkspaceTools._relative_path("../outside")


def test_docker_workspace_exposes_typed_portable_tools() -> None:
    toolset = DockerWorkspaceTools("docker", "task-container").toolset("paper8-5")

    assert {resource.name for resource in toolset.resources} == {
        "list_files",
        "read_file",
        "search_text",
        "git_status",
        "write_file",
        "replace_text",
        "run_command",
    }
    by_name = {resource.name: resource for resource in toolset.resources}
    assert by_name["read_file"].side_effect_class.value == "read"
    assert by_name["replace_text"].side_effect_class.value == "write"
