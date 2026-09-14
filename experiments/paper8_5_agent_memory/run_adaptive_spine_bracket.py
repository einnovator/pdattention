"""Execute the registered N-issue completed-spine bracket to a terminal gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .adaptive_spine_bracket import recommend


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _campaign_command(args: argparse.Namespace, config_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m", "experiments.paper8_5_agent_memory.run_autonomous_multi_issue_campaign",
        "--spec", str(args.spec),
        "--output", str(args.output),
        "--upstream-base-url", args.upstream_base_url,
        "--tokenizer", args.tokenizer,
        "--sequence-id", args.sequence_id,
        "--strategy-id", "S03_completed_episode_spine",
        "--strategy-config-id", config_id,
        "--health-probe-count", str(args.health_probe_count),
        "--health-latency-ceiling-seconds", str(args.health_latency_ceiling_seconds),
        "--health-timeout-seconds", str(args.health_timeout_seconds),
        "--upstream-request-timeout-seconds", str(args.upstream_request_timeout_seconds),
    ]
    if args.docker_executable:
        command.extend(("--docker-executable", args.docker_executable))
    if args.docker_platform:
        command.extend(("--docker-platform", args.docker_platform))
    if args.grade_auxiliary_workspace_state:
        command.append("--grade-auxiliary-workspace-state")
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.output / "adaptive_spine_bracket.json"
    audit: dict[str, Any] = {
        "schema_version": 1,
        "sequence_id": args.sequence_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "decisions": [],
    }
    for _ in range(5):
        state_path = args.output / "campaign_state.json"
        if not state_path.is_file():
            decision = {"decision": "wait", "reason": "campaign state is absent"}
        else:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            decision = recommend(state, args.sequence_id)
        decision["observed_at"] = datetime.now(timezone.utc).isoformat()
        audit["decisions"].append(decision)
        _write(audit_path, audit)
        if decision["decision"] != "run":
            audit["terminal_decision"] = decision
            audit["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write(audit_path, audit)
            return audit
        process = subprocess.run(
            _campaign_command(args, str(decision["strategy_config_id"])),
            check=False,
        )
        audit["decisions"][-1]["campaign_returncode"] = process.returncode
        _write(audit_path, audit)
        if process.returncode:
            audit["terminal_decision"] = {
                "decision": "infrastructure_error",
                "strategy_config_id": decision["strategy_config_id"],
                "returncode": process.returncode,
            }
            audit["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write(audit_path, audit)
            return audit
    raise RuntimeError("adaptive bracket exceeded the five registered configurations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--docker-executable")
    parser.add_argument("--docker-platform")
    parser.add_argument("--grade-auxiliary-workspace-state", action="store_true")
    parser.add_argument("--health-probe-count", type=int, default=3)
    parser.add_argument("--health-latency-ceiling-seconds", type=float, default=60.0)
    parser.add_argument("--health-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--upstream-request-timeout-seconds", type=int, default=180)
    return parser


def main() -> None:
    audit = run(build_parser().parse_args())
    print(json.dumps(audit["terminal_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
