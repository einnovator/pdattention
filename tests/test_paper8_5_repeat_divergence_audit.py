import json

from experiments.paper8_5_agent_memory.audit_repeat_divergence import build_audit


def _trajectory(command: str, submission: str) -> dict:
    return {
        "instance_id": "org__repo-1",
        "info": {"exit_status": "Submitted", "submission": submission},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": f"```mswea_bash_command\n{command}\n```",
            },
        ],
    }


def test_repeat_audit_localizes_first_backend_action_fork(tmp_path) -> None:
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    paths = (tmp_path / "a.json", tmp_path / "b.json")
    paths[0].write_text(json.dumps(_trajectory("grep value a.py", patch)))
    paths[1].write_text(json.dumps(_trajectory("grep -n value a.py", "source")))

    audit = build_audit(*paths)

    assert audit["common_visible_message_prefix"] == 2
    assert audit["first_divergent_message_role"] == "assistant"
    assert audit["first_divergent_action_index_one_based"] == 1
    assert audit["runs"][0]["submission_valid_git_diff"] is True
    assert audit["runs"][1]["submission_valid_git_diff"] is False
    assert "backend/model trajectory variability" in audit["interpretation"]
