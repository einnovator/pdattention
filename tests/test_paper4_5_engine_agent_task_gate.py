from __future__ import annotations

import json
from pathlib import Path

from experiments.paper4_5_agent.summarize_engine_agent_task_gate import summarize


INSTANCE = "owner__repo-1"


def _run(
    path: Path, *, mode: str, resolved: bool, commands: list[str],
    retention: float, chat_template_digest: str | None = None,
) -> None:
    path.mkdir(parents=True)
    row = {
        "instance_id": INSTANCE,
        "run_id": path.name,
        "mode": mode,
        "resolved": resolved,
        "grader_outcome": "resolved" if resolved else "unresolved",
        "model_call_count": len(commands),
        "tool_call_count": len(commands),
        "context_budget_fraction": retention,
        "realized_retention_fraction": retention,
        "selected_history_reencoded_tokens": 0,
        "physical_kv_copy_bytes": 0,
        "consumer_temporary_bytes": 128,
        "consumer_temporary_peak_bytes": 64,
    }
    (path / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    trajectory = path / "chunk_00" / INSTANCE / f"{INSTANCE}.traj.json"
    trajectory.parent.mkdir(parents=True)
    messages = [
        {
            "role": "assistant",
            "content": f"run {command}",
            "extra": {"response": {"id": str(index)}, "actions": [{"command": command}]},
        }
        for index, command in enumerate(commands)
    ]
    trajectory.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    if chat_template_digest is not None:
        (path / "run_manifest.json").write_text(json.dumps({
            "gateway_preflight": {
                "chat_template_profile": "stable-test",
                "chat_template_digest": chat_template_digest,
            }
        }), encoding="utf-8")


def test_100_percent_exactness_arms_90_percent(tmp_path: Path) -> None:
    plain, pra100 = tmp_path / "plain", tmp_path / "pra100"
    _run(plain, mode="no-pra", resolved=True, commands=["ls", "edit"], retention=1.0)
    _run(
        pra100, mode="direct-native-pra", resolved=True,
        commands=["ls", "edit"], retention=1.0,
    )

    result = summarize(
        engine="test", instance_id=INSTANCE,
        plain_dir=plain, pra100_dir=pra100, pra90_dir=None,
    )

    assert result["classification"] == "qualified_for_90_percent_execution"
    assert result["gates"]["pra_100_exact_action_trajectory"] is True
    assert result["gates"]["pra_90_execution_allowed"] is True
    assert result["gates"]["engine_task_gate_complete"] is False
    assert result["gates"]["easy14_expansion_allowed"] is False


def test_100_percent_divergence_is_an_implementation_bug(tmp_path: Path) -> None:
    plain, pra100 = tmp_path / "plain", tmp_path / "pra100"
    _run(plain, mode="no-pra", resolved=True, commands=["ls", "edit"], retention=1.0)
    _run(
        pra100, mode="direct-native-pra", resolved=True,
        commands=["ls", "different-edit"], retention=1.0,
    )

    result = summarize(
        engine="test", instance_id=INSTANCE,
        plain_dir=plain, pra100_dir=pra100, pra90_dir=None,
    )

    assert result["classification"] == "implementation_bug_at_100_percent"
    assert result["gates"]["pra_100_behavioral_parity"] is False
    assert result["gates"]["pra_90_execution_allowed"] is False


def test_100_percent_rejects_chat_template_digest_mismatch(tmp_path: Path) -> None:
    plain, pra100 = tmp_path / "plain", tmp_path / "pra100"
    _run(
        plain, mode="no-pra", resolved=True, commands=["ls"], retention=1.0,
        chat_template_digest="digest-a",
    )
    _run(
        pra100, mode="direct-native-pra", resolved=True, commands=["ls"],
        retention=1.0, chat_template_digest="digest-b",
    )
    result = summarize(
        engine="test", instance_id=INSTANCE,
        plain_dir=plain, pra100_dir=pra100, pra90_dir=None,
    )
    assert result["gates"]["pra_100_exact_action_trajectory"] is True
    assert result["gates"]["pra_100_chat_template_digest_match"] is False
    assert result["gates"]["pra_100_behavioral_parity"] is False


def test_published_llamacpp_task01_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "docs/papers/shared/results/paper4_5_runtime_productization/coding_agents"
        / "engine_agent_gates/llamacpp_task01/gate.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["plain"]["task_success"] is True
    assert result["pra_100"]["task_success"] is True
    assert result["gates"]["pra_100_exact_action_trajectory"] is True
    assert result["plain"]["calls_to_solution"] == 30
    assert result["pra_100"]["calls_to_solution"] == 30
    assert result["pra_90"]["task_success"] is True
    assert result["pra_90"]["calls_to_solution"] == 25
    assert result["gates"]["pra_90_first_action_divergence"] == 7
    assert result["pra_90"]["realized_retention_fraction"] == 0.9332197124270866
    assert result["pra_90"]["selected_history_reencoded_tokens"] == 0
    assert result["pra_90"]["physical_kv_copy_bytes"] == 0
    assert result["pra_90"]["host_to_device_bytes"] == 0
    assert result["pra_90"]["consumer_temporary_bytes"] is None
    assert result["gates"]["easy14_expansion_allowed"] is False
