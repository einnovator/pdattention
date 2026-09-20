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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(instance_id: str) -> str:
    return instance_id.replace("__", "-").replace("/", "-")


def validate_campaign_spec(spec: Mapping[str, Any]) -> None:
    """Reject unbounded generations before launching an autonomous cohort."""

    generation = spec.get("generation") or {}
    max_completion_tokens = generation.get("max_completion_tokens")
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens <= 0
    ):
        raise ValueError(
            "autonomous quality campaigns require a positive, frozen "
            "max_completion_tokens value"
        )


def campaign_cells(spec: Mapping[str, Any], task_ids: Sequence[str]) -> list[dict[str, Any]]:
    repeats = int((spec.get("controls") or {}).get("full_repeats", 2))
    if repeats < 2:
        raise ValueError("at least two FULL controls are required")
    cells: list[dict[str, Any]] = []
    pair_campaign_id = str(spec.get("baseline_pair_campaign_id") or spec["campaign_id"])
    for task_index, task_id in enumerate(task_ids, 1):
        pair_id = f"{pair_campaign_id}:{task_id}:seed{spec['generation']['seed']}"
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
                "pair_id": f"{pair_campaign_id}:{task_id}:seed{spec['generation']['seed']}",
                "is_control": False,
                **dict(arm),
            })
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("campaign cell IDs must be unique")
    return cells


def import_completed_controls(
    state: dict[str, Any],
    cells: Sequence[Mapping[str, Any]],
    source_state_path: Path,
    *,
    expected_campaign_id: str,
) -> int:
    """Import exact completed FULL cells without pretending revisions match.

    The reducer permits only a repository-revision mismatch for these pairs;
    every model, harness, dataset, environment, and agent-behavior identity is
    still checked from the immutable run manifests.
    """

    source_path = source_state_path.resolve()
    source = _read(source_path)
    if source.get("campaign_id") != expected_campaign_id:
        raise ValueError(
            "imported control state belongs to a different baseline campaign"
        )
    expected = {
        str(row["cell_id"]): dict(row) for row in cells if row.get("is_control")
    }
    imported = 0
    for cell_id, source_row in (source.get("cells") or {}).items():
        if cell_id not in expected or source_row.get("status") != "complete":
            continue
        target = expected[cell_id]
        for key in (
            "instance_id", "arm_id", "policy", "negative_realization",
            "negative_fallback", "budget_fraction", "pair_id", "is_control",
        ):
            if source_row.get(key) != target.get(key):
                raise ValueError(
                    f"imported control {cell_id} differs in frozen field {key}"
                )
        output = Path(str(source_row.get("output", ""))).resolve()
        result = _completed_result(output)
        if result is None:
            raise ValueError(f"imported control {cell_id} is not a completed run")
        existing = state["cells"].get(cell_id)
        if existing and existing.get("status") == "complete":
            continue
        state["cells"][cell_id] = {
            **target,
            **result,
            "output": str(output),
            "imported_control": True,
            "source_campaign_state": str(source_path),
            "source_campaign_state_sha256": _file_digest(source_path),
        }
        imported += 1
    state["imported_control_states"] = [{
        "path": str(source_path),
        "sha256": _file_digest(source_path),
        "campaign_id": source.get("campaign_id"),
    }]
    return imported


def adaptive_gate(
    arm_results: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> tuple[str, str]:
    completed = [row for row in arm_results if row.get("status") == "complete"]
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(completed):
        task_id = row.get("instance_id") or row.get("task_id")
        if not task_id:
            raise ValueError(
                f"completed adaptive-gate row {index} lacks an independent task identity"
            )
        by_task.setdefault(str(task_id), []).append(row)
    task_rows: list[dict[str, Any]] = []
    for task_id, rows in sorted(by_task.items()):
        graded = [row for row in rows if isinstance(row.get("official_resolved"), bool)]
        full = sum(int(row.get("cumulative_full_tokens") or 0) for row in rows)
        materialized = sum(
            int(row.get("cumulative_materialized_tokens") or 0) for row in rows
        )
        saving = (
            1 - materialized / full if full > 0
            else sum(float(row.get("gross_saving_fraction") or 0) for row in rows)
            / len(rows)
        )
        task_rows.append({
            "task_id": task_id,
            "saving": saving,
            "resolution": (
                sum(bool(row["official_resolved"]) for row in graded) / len(graded)
                if graded else None
            ),
            "genuine_failure": bool(graded) and not any(
                bool(row["official_resolved"]) for row in graded
            ),
        })
    grade_valid = [row for row in task_rows if row["resolution"] is not None]
    failures = sum(bool(row["genuine_failure"]) for row in grade_valid)
    if failures >= int(thresholds.get("stop_after_failures", 2)):
        return "stop", f"{failures} official failures reached the stop boundary"
    yield_n = int(thresholds.get("minimum_tasks_before_yield_gate", 2))
    if len(task_rows) >= yield_n:
        mean_saving = sum(float(row["saving"]) for row in task_rows) / len(task_rows)
        if mean_saving < float(thresholds.get("minimum_gross_saving_fraction", .02)):
            return "stop", f"mean gross saving {mean_saving:.4f} is below the yield gate"
    accuracy_n = int(thresholds.get("minimum_tasks_before_accuracy_gate", 3))
    if len(grade_valid) >= accuracy_n:
        accuracy = sum(float(row["resolution"]) for row in grade_valid) / len(grade_valid)
        if accuracy < float(thresholds.get("minimum_task_resolution", .8)):
            return "stop", f"official resolution {accuracy:.4f} is below the accuracy gate"
    return "continue", "arm remains within predeclared gates"


def candidate_control_gate(
    state: Mapping[str, Any], cell: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[bool, str]:
    """Admit a candidate only after its declared successful FULL controls."""

    controls_spec = spec.get("controls") or {}
    if not bool(controls_spec.get("require_all_resolved_before_candidate", False)):
        return True, "successful FULL-control admission is not required"
    controls = [
        row for row in (state.get("cells") or {}).values()
        if row.get("instance_id") == cell.get("instance_id")
        and row.get("is_control")
    ]
    expected = int(controls_spec.get("full_repeats", 2))
    admitted = (
        len(controls) == expected
        and all(row.get("status") == "complete" for row in controls)
        and all(row.get("official_resolved") is True for row in controls)
    )
    return admitted, (
        "every same-task FULL control completed and officially resolved"
        if admitted else
        "candidate requires every same-task FULL control to complete and officially resolve"
    )


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
    trace_path = path / "request_selection.jsonl"
    if not trace_path.is_file():
        return None
    trace = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexes = [int(row["request_index"]) for row in trace]
    sparse = bool(indexes) and sorted(indexes) != list(
        range(min(indexes), max(indexes) + 1)
    )
    upstream_errors = int(metrics.get("upstream_error_calls") or 0)
    if sparse or upstream_errors:
        return {
            "status": "infrastructure_contaminated",
            "official_resolved": official["resolved"],
            "official_error": bool(official.get("error", False)),
            "calls": int(metrics.get("calls") or 0),
            "cumulative_full_tokens": full,
            "cumulative_materialized_tokens": materialized,
            "gross_saving_fraction": 1 - materialized / full,
            "trace_sparse_request_indexes": sparse,
            "upstream_error_calls": upstream_errors,
            "evidence_admitted": False,
            "quarantine_reason": (
                "unlogged request attempts in legacy trace" if sparse
                else "upstream request failure occurred during trajectory"
            ),
        }
    return {
        "status": "complete",
        "official_resolved": official["resolved"],
        "official_error": bool(official.get("error", False)),
        "official_outcome": (
            "grader_error" if bool(official.get("error", False))
            else "resolved" if bool(official["resolved"])
            else "unresolved"
        ),
        "calls": int(metrics.get("calls") or 0),
        "cumulative_full_tokens": full,
        "cumulative_materialized_tokens": materialized,
        "gross_saving_fraction": 1 - materialized / full,
        "cumulative_full_tokens": full,
        "cumulative_materialized_tokens": materialized,
    }


def _retry_path(base: Path, *, reserved: Sequence[Path] = ()) -> Path:
    reserved_paths = {path.resolve() for path in reserved}
    if (
        base.resolve() not in reserved_paths
        and (not base.exists() or (base.is_dir() and not any(base.iterdir())))
    ):
        return base
    for attempt in range(1, 1000):
        candidate = base.with_name(f"{base.name}__retry{attempt:02d}")
        if candidate.resolve() not in reserved_paths and not candidate.exists():
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
    def bind_legacy_contract(
        pair: dict[str, Any], candidate_row: Mapping[str, Any],
        baseline_row: Mapping[str, Any],
    ) -> None:
        manifests = []
        for row in (candidate_row, baseline_row):
            manifest = Path(str(row["output"])) / "run_manifest.json"
            manifests.append(_read(manifest) if manifest.is_file() else {})
        if any(int(value.get("schema_version") or 1) < 2 for value in manifests):
            pair.update({
                "allow_legacy_pair": True,
                "pairing_reason": (
                    "historical schema-1 run; immutable fields available in both "
                    "manifests are reducer-checked, while missing identity fields "
                    "are disclosed rather than treated as matching evidence"
                ),
            })

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
        repeat_pair: dict[str, Any] = {
            "candidate": controls[1][0], "baseline": baseline,
            "pairing_reason": "repeat-control stochastic trajectory envelope",
        }
        bind_legacy_contract(repeat_pair, controls[1][1], controls[0][1])
        pairs.append(repeat_pair)
        for cell_id, row in task_rows:
            if not row.get("is_control"):
                pair = {"candidate": cell_id, "baseline": baseline}
                if controls[0][1].get("imported_control"):
                    pair.update({
                        "allow_legacy_pair": True,
                        "pairing_reason": (
                            "selector-only implementation revision changed; FULL "
                            "requests remain exact pass-through and every other frozen "
                            "execution identity must match"
                        ),
                    })
                bind_legacy_contract(pair, row, controls[0][1])
                pairs.append(pair)
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
    ollama_tags_url = getattr(args, "ollama_tags_url", None)
    if ollama_tags_url:
        command.extend(("--ollama-tags-url", ollama_tags_url))
    materialization = {
        "materialization_mode": "--materialization-mode",
        "materialization_threshold_tokens": "--materialization-threshold-tokens",
        "materialization_head_lines": "--materialization-head-lines",
        "materialization_tail_lines": "--materialization-tail-lines",
        "materialization_match_context_lines": "--materialization-match-context-lines",
        "materialization_max_matched_lines": "--materialization-max-matched-lines",
    }
    for key, option in materialization.items():
        if key in cell:
            command.extend((option, str(cell[key])))
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
    if args.grade_auxiliary_workspace_state:
        command.append("--grade-auxiliary-workspace-state")
    if args.preflight_only:
        command.append("--preflight-only")
    return command


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = args.spec.resolve()
    spec = _read(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported campaign schema")
    validate_campaign_spec(spec)
    benchmark = _resolve(spec_path, str(spec["benchmark_card"]))
    card = _read(benchmark)
    task_ids = list(card["instance_ids"])
    for index, task_id in enumerate(task_ids, 1):
        load_locked_task(benchmark, task_index=index, instance_id=None)
        if task_id != card["instance_ids"][index - 1]:
            raise AssertionError("locked task order changed")
    selected_tasks = set(args.task_index or range(1, len(task_ids) + 1))
    selected_arms = set(args.arm_id or ())
    all_cells = campaign_cells(spec, task_ids)
    cells = [
        row for row in all_cells
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
    if args.import_control_state:
        expected_campaign_id = str(
            spec.get("baseline_pair_campaign_id") or spec["campaign_id"]
        )
        import_completed_controls(
            state,
            all_cells,
            args.import_control_state,
            expected_campaign_id=expected_campaign_id,
        )
        _write(state_path, state)
    executed = 0
    for cell in cells:
        cell_id = str(cell["cell_id"])
        base_output = output / f"task_{cell['task_index']:02d}_{_slug(str(cell['instance_id']))}" / str(cell["arm_id"])
        recorded_output = Path(str(
            state["cells"].get(cell_id, {}).get("output", base_output)
        ))
        prior = _completed_result(recorded_output) or _completed_result(base_output)
        if prior is not None and prior.get("status") == "complete":
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
            admitted, reason = candidate_control_gate(state, cell, spec)
            if not admitted:
                state["cells"][cell_id] = {
                    **cell,
                    "status": "stopped_by_control_gate",
                    "reason": reason,
                    "output": str(cell_output),
                }
                _write(state_path, state)
                continue
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
    parser.add_argument(
        "--ollama-tags-url",
        help=(
            "Optional /api/tags URL forwarded to every cell. When supplied, "
            "each run must observe the frozen model digest before execution."
        ),
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--docker-executable")
    parser.add_argument("--docker-platform")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--task-index", type=int, action="append")
    parser.add_argument("--arm-id", action="append")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-grading", action="store_true")
    parser.add_argument("--grade-auxiliary-workspace-state", action="store_true")
    parser.add_argument(
        "--import-control-state",
        type=Path,
        help=(
            "Reuse completed FULL cells from a baseline campaign state. Only a "
            "repository-revision mismatch is eligible in the reducer."
        ),
    )
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
