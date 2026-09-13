"""Run the locked Paper 8.5 autonomous quality--saving development campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .run_autonomous_swebench import load_locked_task


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve(spec_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (spec_path.parent / path).resolve()


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _slug(instance_id: str) -> str:
    return instance_id.replace("__", "-").replace("/", "-")


def campaign_cells(spec: Mapping[str, Any], task_ids: Sequence[str]) -> list[dict[str, Any]]:
    repeats = int((spec.get("controls") or {}).get("full_repeats", 2))
    if repeats < 2:
        raise ValueError("at least two FULL controls are required")
    cells: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(task_ids, 1):
        pair_id = f"{spec['campaign_id']}:{task_id}:seed{spec['generation']['seed']}"
        for repeat in range(repeats):
            cells.append({
                "cell_id": f"task{task_index:02d}-full-{chr(97 + repeat)}",
                "task_index": task_index,
                "instance_id": task_id,
                "pair_id": pair_id,
                "arm_id": f"full_{chr(97 + repeat)}",
                "policy": "full",
                "negative_realization": "drop",
                "negative_fallback": "none",
                "budget_fraction": 1.0,
                "is_control": True,
            })
    # Arm-major ordering enables an early stop before widening a weak arm.
    for arm in spec.get("arms") or ():
        for task_index, task_id in enumerate(task_ids, 1):
            cells.append({
                "cell_id": f"task{task_index:02d}-{arm['arm_id']}",
                "task_index": task_index,
                "instance_id": task_id,
                "pair_id": f"{spec['campaign_id']}:{task_id}:seed{spec['generation']['seed']}",
                "is_control": False,
                **dict(arm),
            })
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("campaign cell IDs must be unique")
    return cells


def adaptive_gate(
    arm_results: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> tuple[str, str]:
    completed = [row for row in arm_results if row.get("status") == "complete"]
    failures = sum(row.get("official_resolved") is False for row in completed)
    if failures >= int(thresholds.get("stop_after_failures", 2)):
        return "stop", f"{failures} official failures reached the stop boundary"
    yield_n = int(thresholds.get("minimum_tasks_before_yield_gate", 2))
    if len(completed) >= yield_n:
        mean_saving = sum(float(row.get("gross_saving_fraction") or 0) for row in completed) / len(completed)
        if mean_saving < float(thresholds.get("minimum_gross_saving_fraction", .02)):
            return "stop", f"mean gross saving {mean_saving:.4f} is below the yield gate"
    accuracy_n = int(thresholds.get("minimum_tasks_before_accuracy_gate", 3))
    if len(completed) >= accuracy_n:
        accuracy = sum(bool(row.get("official_resolved")) for row in completed) / len(completed)
        if accuracy < float(thresholds.get("minimum_task_resolution", .8)):
            return "stop", f"official resolution {accuracy:.4f} is below the accuracy gate"
    return "continue", "arm remains within predeclared gates"


def _completed_result(path: Path) -> dict[str, Any] | None:
    metrics_path = path / "autonomous_metrics.json"
    manifest_path = path / "run_manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        return None
    metrics = _read(metrics_path)
    official = metrics.get("official_result") or {}
    if not isinstance(official.get("resolved"), bool):
        return None
    full = int(metrics.get("cumulative_full_tokens") or 0)
    materialized = int(metrics.get("cumulative_materialized_tokens") or 0)
    if full <= 0:
        return None
    return {
        "status": "complete",
        "official_resolved": official["resolved"],
        "calls": int(metrics.get("calls") or 0),
        "cumulative_full_tokens": full,
        "cumulative_materialized_tokens": materialized,
        "gross_saving_fraction": 1 - materialized / full,
    }


def _retry_path(base: Path) -> Path:
    if not base.exists() or (base.is_dir() and not any(base.iterdir())):
        return base
    for attempt in range(1, 1000):
        candidate = base.with_name(f"{base.name}__retry{attempt:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many preserved retry attempts for {base}")


def write_curve_spec(state: Mapping[str, Any], output: Path) -> Path:
    complete = {
        str(cell_id): row for cell_id, row in state["cells"].items()
        if row.get("status") == "complete"
    }
    runs: list[dict[str, str]] = []
    pairs: list[dict[str, str]] = []
    by_task: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for cell_id, row in complete.items():
        by_task.setdefault(str(row["instance_id"]), []).append((cell_id, row))
    for task_rows in by_task.values():
        controls = sorted(
            (item for item in task_rows if item[1].get("is_control")),
            key=lambda item: str(item[1]["arm_id"]),
        )
        eligible = set(cell_id for cell_id, _ in controls)
        if len(controls) >= 2:
            eligible.update(
                cell_id for cell_id, row in task_rows if not row.get("is_control")
            )
        for cell_id, row in task_rows:
            if cell_id in eligible:
                runs.append({
                    "name": cell_id,
                    "path": str(Path(str(row["output"])).resolve()),
                })
        if len(controls) < 2:
            continue
        baseline = controls[0][0]
        pairs.append({
            "candidate": controls[1][0], "baseline": baseline,
            "pairing_reason": "repeat-control stochastic trajectory envelope",
        })
        for cell_id, row in task_rows:
            if not row.get("is_control"):
                pairs.append({"candidate": cell_id, "baseline": baseline})
    spec = {
        "schema_version": 1,
        "study": "paper8_5_agent_memory_tradeoff_curve_spec",
        "frozen_comparisons": [],
        "autonomous_runs": sorted(runs, key=lambda row: row["name"]),
        "autonomous_pairs": sorted(pairs, key=lambda row: row["candidate"]),
        "adaptive_pruning": {},
    }
    path = output / "tradeoff_spec.json"
    _write(path, spec)
    return path


def _command(
    *, spec: Mapping[str, Any], benchmark: Path, output: Path,
    cell: Mapping[str, Any], args: argparse.Namespace,
) -> list[str]:
    generation = spec["generation"]
    history = spec["history"]
    command = [
        sys.executable, "-m", "experiments.paper8_5_agent_memory.run_autonomous_swebench",
        "--benchmark-card", str(benchmark),
        "--instance-id", str(cell["instance_id"]),
        "--output", str(output),
        "--run-id", str(cell["cell_id"]),
        "--pair-id", str(cell["pair_id"]),
        "--upstream-base-url", args.upstream_base_url,
        "--model", str(spec["model"]),
        "--served-model", str(spec["served_model"]),
        "--model-revision", str(spec["model_revision"]),
        "--tokenizer", args.tokenizer,
        "--tokenizer-revision", str(spec["tokenizer_revision"]),
        "--policy", str(cell["policy"]),
        "--negative-realization", str(cell["negative_realization"]),
        "--negative-fallback", str(cell["negative_fallback"]),
        "--budget-fraction", str(cell["budget_fraction"]),
        "--head", str(history["head_turns"]),
        "--tail", str(history["tail_turns"]),
        "--search-delay-turns", str(history["search_delay_turns"]),
        "--write-delay-turns", str(history["write_delay_turns"]),
        "--same-span-reads-to-keep", str(history["same_span_reads_to_keep"]),
        "--working-set-resources", str(history["working_set_resources"]),
        "--temperature", str(generation["temperature"]),
        "--top-p", str(generation["top_p"]),
        "--seed", str(generation["seed"]),
        "--max-calls", str(generation["max_calls"]),
    ]
    if generation.get("max_completion_tokens") is not None:
        command.extend((
            "--max-completion-tokens", str(generation["max_completion_tokens"]),
        ))
    if args.docker_executable:
        command.extend(("--docker-executable", args.docker_executable))
    if args.docker_platform:
        command.extend(("--docker-platform", args.docker_platform))
    for value in args.pythonpath:
        command.extend(("--pythonpath", value))
    if args.skip_grading:
        command.append("--skip-grading")
    if args.preflight_only:
        command.append("--preflight-only")
    return command


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = args.spec.resolve()
    spec = _read(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported campaign schema")
    benchmark = _resolve(spec_path, str(spec["benchmark_card"]))
    card = _read(benchmark)
    task_ids = list(card["instance_ids"])
    for index, task_id in enumerate(task_ids, 1):
        load_locked_task(benchmark, task_index=index, instance_id=None)
        if task_id != card["instance_ids"][index - 1]:
            raise AssertionError("locked task order changed")
    selected_tasks = set(args.task_index or range(1, len(task_ids) + 1))
    selected_arms = set(args.arm_id or ())
    cells = [
        row for row in campaign_cells(spec, task_ids)
        if row["task_index"] in selected_tasks
        and (not selected_arms or row["arm_id"] in selected_arms)
    ]
    output = args.output.resolve()
    state_path = output / "campaign_state.json"
    state = _read(state_path) if state_path.is_file() else {
        "schema_version": 1,
        "study": "paper8_5_autonomous_quality_saving_campaign",
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": _digest(spec),
        "benchmark_card": str(benchmark),
        "cells": {},
        "arm_gates": {},
    }
    if state["campaign_spec_sha256"] != _digest(spec):
        raise ValueError("campaign state belongs to a different specification")
    executed = 0
    for cell in cells:
        cell_id = str(cell["cell_id"])
        base_output = output / f"task_{cell['task_index']:02d}_{_slug(str(cell['instance_id']))}" / str(cell["arm_id"])
        recorded_output = Path(str(
            state["cells"].get(cell_id, {}).get("output", base_output)
        ))
        prior = _completed_result(recorded_output) or _completed_result(base_output)
        if prior is not None:
            completed_output = (
                recorded_output if _completed_result(recorded_output) is not None
                else base_output
            )
            state["cells"][cell_id] = {
                **cell, **prior, "output": str(completed_output)
            }
            continue
        cell_output = _retry_path(base_output)
        if not cell["is_control"]:
            prior_arm = [
                row for row in state["cells"].values()
                if row.get("arm_id") == cell["arm_id"]
            ]
            gate, reason = adaptive_gate(prior_arm, spec["adaptive_stopping"])
            state["arm_gates"][cell["arm_id"]] = {"decision": gate, "reason": reason}
            if gate == "stop":
                state["cells"][cell_id] = {
                    **cell, "status": "stopped_by_gate", "reason": reason,
                    "output": str(cell_output),
                }
                _write(state_path, state)
                continue
        if args.max_runs is not None and executed >= args.max_runs:
            break
        command = _command(
            spec=spec, benchmark=benchmark, output=cell_output, cell=cell, args=args,
        )
        state["cells"][cell_id] = {
            **cell, "status": "running", "command": command,
            "output": str(cell_output), "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _write(state_path, state)
        if args.dry_run:
            state["cells"][cell_id]["status"] = "planned"
            _write(state_path, state)
            continue
        completed = subprocess.run(command, check=False)
        result = _completed_result(cell_output)
        state["cells"][cell_id].update(result or {
            "status": "infrastructure_error",
            "returncode": completed.returncode,
        })
        state["cells"][cell_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write(state_path, state)
        executed += 1
        if result is None and not args.continue_on_error:
            break
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write(state_path, state)
    curve_spec = write_curve_spec(state, output)
    if args.reduce and any(
        not row.get("is_control") and row.get("status") == "complete"
        for row in state["cells"].values()
    ):
        from .tradeoff_curves import build
        build(curve_spec, output / "curves")
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--docker-executable")
    parser.add_argument("--docker-platform")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--task-index", type=int, action="append")
    parser.add_argument("--arm-id", action="append")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-grading", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reduce", action="store_true")
    return parser


def main() -> None:
    state = run_campaign(build_parser().parse_args())
    counts: dict[str, int] = {}
    for row in state["cells"].values():
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"campaign_id": state["campaign_id"], "status_counts": counts}))


if __name__ == "__main__":
    main()
