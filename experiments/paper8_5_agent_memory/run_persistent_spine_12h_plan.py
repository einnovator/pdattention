"""Execute the gated 12-hour persistent-spine qualification sequence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from .adaptive_spine_bracket import paired_point
from .export_multi_issue_evidence import build as build_evidence, write_bundle
from .persistent_spine_gate import evaluate
from .run_autonomous_multi_issue_campaign import campaign_cells
from .multi_issue_curves import aggregate_frontier, render_frontier_plots
from .multi_issue_frontier import load_strategy_registry, reduce_multi_issue_runs


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selected_cells(
    spec: Mapping[str, Any], *, sequence_id: str, strategy_id: str,
    strategy_config_id: str | None,
) -> list[dict[str, Any]]:
    return [
        row for row in campaign_cells(spec)
        if row["sequence_id"] == sequence_id
        and row["strategy_id"] == strategy_id
        and (
            strategy_config_id is None
            or row["strategy_config_id"] == strategy_config_id
        )
    ]


def _complete(state: Mapping[str, Any], cells: list[Mapping[str, Any]]) -> bool:
    observed = state.get("cells") or {}
    return bool(cells) and all(
        (observed.get(str(cell["cell_id"])) or {}).get("status") == "complete"
        for cell in cells
    )


def _command(
    args: argparse.Namespace, *, sequence_id: str, strategy_id: str,
    strategy_config_id: str | None,
) -> list[str]:
    command = [
        sys.executable, "-m",
        "experiments.paper8_5_agent_memory.run_autonomous_multi_issue_campaign",
        "--spec", str(args.spec), "--output", str(args.output),
        "--upstream-base-url", args.upstream_base_url,
        "--tokenizer", args.tokenizer,
        "--sequence-id", sequence_id, "--strategy-id", strategy_id,
        "--health-probe-count", str(args.health_probe_count),
        "--health-latency-ceiling-seconds", str(args.health_latency_ceiling_seconds),
        "--health-timeout-seconds", str(args.health_timeout_seconds),
        "--upstream-request-timeout-seconds", str(args.upstream_request_timeout_seconds),
    ]
    upstream_qualification_path = getattr(args, "upstream_qualification_path", None)
    if upstream_qualification_path:
        command.extend((
            "--upstream-qualification-path", upstream_qualification_path,
            "--upstream-connect-attempts",
            str(getattr(args, "upstream_connect_attempts", 1)),
            "--upstream-connect-retry-seconds",
            str(getattr(args, "upstream_connect_retry_seconds", 1.0)),
        ))
    if strategy_config_id:
        command.extend(("--strategy-config-id", strategy_config_id))
    if args.docker_executable:
        command.extend(("--docker-executable", args.docker_executable))
    if args.docker_platform:
        command.extend(("--docker-platform", args.docker_platform))
    if args.grade_auxiliary_workspace_state:
        command.append("--grade-auxiliary-workspace-state")
    return command


def _ensure(
    args: argparse.Namespace, spec: Mapping[str, Any], audit: dict[str, Any], *,
    sequence_id: str, strategy_id: str, strategy_config_id: str | None = None,
) -> bool:
    expected = _selected_cells(
        spec, sequence_id=sequence_id, strategy_id=strategy_id,
        strategy_config_id=strategy_config_id,
    )
    state_path = args.output / "campaign_state.json"
    for attempt in range(1, args.max_infrastructure_attempts + 1):
        state = _read(state_path) if state_path.is_file() else {"cells": {}}
        if _complete(state, expected):
            return True
        process = subprocess.run(
            _command(
                args, sequence_id=sequence_id, strategy_id=strategy_id,
                strategy_config_id=strategy_config_id,
            ),
            check=False,
        )
        event = {
            "sequence_id": sequence_id, "strategy_id": strategy_id,
            "strategy_config_id": strategy_config_id,
            "attempt": attempt, "returncode": process.returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        audit["events"].append(event)
        _write(args.audit, audit)
        state = _read(state_path) if state_path.is_file() else {"cells": {}}
        if _complete(state, expected):
            return True
        if attempt < args.max_infrastructure_attempts:
            time.sleep(args.retry_wait_seconds)
    return False


def _single_repeat_point(
    state: Mapping[str, Any], *, sequence_id: str, config_id: str,
) -> dict[str, Any]:
    cells = state["cells"]
    control = cells[f"{sequence_id}__S01_persistent_full__r01"]
    candidate = cells[
        f"{sequence_id}__S03_completed_episode_spine-{config_id}__r01"
    ]
    return {
        **paired_point(candidate, control),
        "calls_delta_vs_persistent_full": (
            int(candidate["calls"]) - int(control["calls"])
        ),
    }


def _full_reliability_gate(
    state: Mapping[str, Any], *, sequence_id: str, minimum_all_solved: int,
) -> dict[str, Any]:
    controls = [
        row for row in state.get("cells", {}).values()
        if row.get("sequence_id") == sequence_id
        and row.get("strategy_id") == "S01_persistent_full"
        and row.get("status") == "complete"
    ]
    all_solved = sum(bool(row.get("all_issues_resolved")) for row in controls)
    return {
        "decision": (
            "advance" if all_solved >= minimum_all_solved
            else "stop_full_reliability_below_gate"
        ),
        "sequence_id": sequence_id,
        "complete_full_repeats": len(controls),
        "all_solved_full_repeats": all_solved,
        "minimum_all_solved_full_repeats": minimum_all_solved,
    }


def _write_products(output: Path) -> None:
    write_bundle(build_evidence(output), output / "evidence_bundle")
    frontier_path = output / "frontier_runs.jsonl"
    if not frontier_path.is_file():
        return
    runs = [
        json.loads(line) for line in frontier_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not runs:
        return
    registry_path = (
        Path(__file__).resolve().parent / "configs" /
        "multi_issue_strategy_registry_v1.json"
    )
    reduction = reduce_multi_issue_runs(runs, load_strategy_registry(registry_path))
    reduction_path = output / "persistent_frontier_reduction.json"
    reduction_path.write_text(json.dumps(reduction, indent=2) + "\n", encoding="utf-8")
    summary = aggregate_frontier(reduction)
    curves = output / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    (curves / "curve_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    try:
        render_frontier_plots(summary, curves)
    except ModuleNotFoundError as error:
        # The execution host intentionally carries only the agent environment.
        # Preserve the reduction and defer rendering to the analysis host.
        (curves / "plot_deferred.json").write_text(json.dumps({
            "status": "deferred_missing_dependency",
            "dependency": error.name,
            "reason": str(error),
        }, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = _read(args.spec)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper8_5_persistent_spine_12h_plan",
        "campaign_id": spec["campaign_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "events": [], "gates": [],
    }
    _write(args.audit, audit)
    for sequence_id in args.sequence_id:
        if not _ensure(
            args, spec, audit, sequence_id=sequence_id,
            strategy_id="S01_persistent_full",
        ):
            audit["terminal"] = {
                "decision": "infrastructure_pause", "sequence_id": sequence_id,
                "stage": "persistent_full",
            }
            break
        minimum_full_successes = int(getattr(
            args, "minimum_all_solved_full_repeats", 0
        ))
        if minimum_full_successes:
            full_gate = _full_reliability_gate(
                _read(args.output / "campaign_state.json"),
                sequence_id=sequence_id,
                minimum_all_solved=minimum_full_successes,
            )
            audit["gates"].append(full_gate)
            _write(args.audit, audit)
            if full_gate["decision"] != "advance":
                audit["terminal"] = full_gate
                break
        if not _ensure(
            args, spec, audit, sequence_id=sequence_id,
            strategy_id="S03_completed_episode_spine",
            strategy_config_id="r4m2v2_tasks1",
        ):
            audit["terminal"] = {
                "decision": "infrastructure_pause", "sequence_id": sequence_id,
                "stage": "r4m2v2",
            }
            break
        gate = evaluate(
            _read(args.output / "campaign_state.json"),
            sequence_id=sequence_id, strategy_config_id="r4m2v2_tasks1",
            required_repeats=2,
        )
        audit["gates"].append(gate)
        _write(args.audit, audit)
        if gate["decision"] != "advance":
            aggregate = float(
                gate.get("aggregate_failure_aware_saving_vs_persistent_full") or 0
            )
            lost = int(gate.get("lost_persistent_full_successes") or 0)
            bracket = (
                ("r6m2v2_tasks1", "r8m2v2_tasks1")
                if lost or aggregate > 0.50 else ("r2m2v2_tasks1",)
            )
            for config_id in bracket:
                if not _ensure(
                    args, spec, audit, sequence_id=sequence_id,
                    strategy_id="S03_completed_episode_spine",
                    strategy_config_id=config_id,
                ):
                    break
                point = _single_repeat_point(
                    _read(args.output / "campaign_state.json"),
                    sequence_id=sequence_id, config_id=config_id,
                )
                audit["gates"].append({
                    "decision": "exploratory_rebracket",
                    "sequence_id": sequence_id,
                    "strategy_config_id": config_id, **point,
                })
                if (
                    point["lost_persistent_full_successes"] == 0
                    and 0.30 <= point["failure_aware_saving_vs_persistent_full"] <= 0.50
                ):
                    break
            audit["terminal"] = gate
            break
    else:
        # Only after all generalization gates pass, establish the local N=2
        # policy neighborhood and the fresh-session normalization control.
        frontier_sequence = args.sequence_id[0]
        if not _ensure(
            args, spec, audit, sequence_id=frontier_sequence,
            strategy_id="S00_fresh_full",
        ):
            audit["terminal"] = {
                "decision": "infrastructure_pause", "sequence_id": frontier_sequence,
                "stage": "fresh_full",
            }
        else:
            neighborhood_complete = True
            for config_id in ("r2m2v2_tasks1", "r6m2v2_tasks1"):
                complete = _ensure(
                    args, spec, audit, sequence_id=frontier_sequence,
                    strategy_id="S03_completed_episode_spine",
                    strategy_config_id=config_id,
                )
                if not complete:
                    neighborhood_complete = False
                    audit["events"].append({
                        "decision": "infrastructure_pause",
                        "sequence_id": frontier_sequence,
                        "strategy_config_id": config_id,
                    })
                    break
                point = _single_repeat_point(
                    _read(args.output / "campaign_state.json"),
                    sequence_id=frontier_sequence, config_id=config_id,
                )
                audit["gates"].append({
                    "decision": (
                        "retain_frontier_point"
                        if point["lost_persistent_full_successes"] == 0
                        else "reject_lost_full_success"
                    ),
                    "sequence_id": frontier_sequence,
                    "strategy_config_id": config_id,
                    **point,
                })
                _write(args.audit, audit)
            if neighborhood_complete:
                audit["terminal"] = {"decision": "planned_execution_complete"}
            else:
                audit["terminal"] = {
                    "decision": "infrastructure_pause",
                    "sequence_id": frontier_sequence,
                    "stage": "local_frontier",
                }
    audit["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write(args.audit, audit)
    if (args.output / "campaign_state.json").is_file():
        _write_products(args.output)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sequence-id", action="append", required=True)
    parser.add_argument("--docker-executable")
    parser.add_argument("--docker-platform")
    parser.add_argument("--grade-auxiliary-workspace-state", action="store_true")
    parser.add_argument("--health-probe-count", type=int, default=3)
    parser.add_argument("--health-latency-ceiling-seconds", type=float, default=60)
    parser.add_argument("--health-timeout-seconds", type=float, default=90)
    parser.add_argument("--upstream-request-timeout-seconds", type=int, default=180)
    parser.add_argument("--upstream-qualification-path")
    parser.add_argument("--upstream-connect-attempts", type=int, default=1)
    parser.add_argument("--upstream-connect-retry-seconds", type=float, default=1.0)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    parser.add_argument("--retry-wait-seconds", type=float, default=60)
    parser.add_argument(
        "--minimum-all-solved-full-repeats",
        type=int,
        default=0,
        help="Stop before policy execution unless this many FULL repeats solve all issues.",
    )
    return parser


def main() -> None:
    audit = run(build_parser().parse_args())
    print(json.dumps(audit["terminal"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
