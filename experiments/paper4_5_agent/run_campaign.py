"""Resumable scheduler that enforces no-PRA reproduction before PRA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reports import write_reports
from .reproduction import OfficialResult, review_result
from .schema import CampaignConfig, CampaignMode, ReproductionStatus


def run_campaign(
    config_path: Path,
    *,
    max_hours: float,
    resume: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute enabled cells in order and persist state after every transition."""

    config = CampaignConfig.load(config_path)
    repository = _repository_root(config_path.resolve())
    root = (repository / config.output_directory).resolve()
    state_path = root / "campaign_state.json"
    state: dict[str, Any] = {"campaign_id": config.campaign_id, "cells": {}}
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    cells = state.setdefault("cells", {})
    deadline = time.monotonic() + max_hours * 3600

    stage_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    scheduled_cells = sorted(
        enumerate(config.cells),
        key=lambda item: (stage_order[item[1].stage], item[1].priority, item[0]),
    )
    for _declared_index, cell in scheduled_cells:
        record = cells.setdefault(cell.cell_id, {"state": "PENDING"})
        if record.get("state") == "COMPLETED":
            continue
        if not cell.enabled:
            record.update(state="SKIPPED", reason="Cell disabled in campaign config.")
            _persist(config, state, root, state_path)
            continue
        gate = _treatment_gate(cell, cells)
        if gate:
            record.update(state="BLOCKED", reason=gate)
            _persist(config, state, root, state_path)
            continue
        if time.monotonic() >= deadline:
            record.update(state="PENDING", reason="Campaign wall-clock deadline reached before launch.")
            _persist(config, state, root, state_path)
            break
        if dry_run:
            record.update(
                state="PENDING", command=_expand_command(cell.command), dry_run=True,
                agent_id=cell.agent_id, connection=cell.connection,
                engine_pra_enabled=cell.engine_pra_enabled,
                gateway_pra_enabled=cell.gateway_pra_enabled,
                gateway_mode=cell.gateway_mode, comparison_group=cell.comparison_group,
                paired_cell=cell.paired_cell,
                evidence_role=cell.evidence_role,
                selection_contract=cell.selection_contract,
                priority=cell.priority,
            )
            record.pop("reason", None)
            record.pop("error", None)
            _persist(config, state, root, state_path)
            continue

        command = _expand_command(cell.command)
        record.update(
            state="RUNNING", started_at=_now(), command=command,
            agent_id=cell.agent_id, connection=cell.connection,
            engine_pra_enabled=cell.engine_pra_enabled,
            gateway_pra_enabled=cell.gateway_pra_enabled,
            gateway_mode=cell.gateway_mode, comparison_group=cell.comparison_group,
            paired_cell=cell.paired_cell,
            evidence_role=cell.evidence_role,
            selection_contract=cell.selection_contract,
            priority=cell.priority,
        )
        record.pop("reason", None)
        record.pop("error", None)
        _persist(config, state, root, state_path)
        workdir = (repository / cell.working_directory).resolve()
        environment = _campaign_environment(cell.environment)
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                command, cwd=workdir, env=environment,
                capture_output=True, text=True, timeout=cell.timeout_seconds, check=False,
            )
            (log_dir / f"{cell.cell_id}.stdout").write_text(completed.stdout, encoding="utf-8")
            (log_dir / f"{cell.cell_id}.stderr").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"command exited with code {completed.returncode}")
            result_path = (repository / cell.result.path).resolve()
            result = OfficialResult.load(result_path)
            review = review_result(
                config.baseline(cell.baseline_id), result,
                absolute_tolerance=cell.result.absolute_tolerance,
                require_exact_cohort=cell.result.require_exact_cohort,
            ) if cell.mode == CampaignMode.NATIVE else None
            record.update(
                state="COMPLETED", finished_at=_now(), result=result.model_dump(mode="json"),
                review=review.model_dump(mode="json") if review else None,
                reproduction_status=(review.status.value if review else None),
            )
            record.pop("reason", None)
            record.pop("error", None)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            record.update(state="FAILED", finished_at=_now(), error=str(exc))
        _persist(config, state, root, state_path)
    return state


def record_result(config_path: Path, *, cell_id: str, result_path: Path) -> dict[str, Any]:
    """Import a result produced on another host into the durable campaign state."""

    config = CampaignConfig.load(config_path)
    repository = _repository_root(config_path.resolve())
    root = (repository / config.output_directory).resolve()
    state_path = root / "campaign_state.json"
    state: dict[str, Any] = {"campaign_id": config.campaign_id, "cells": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    try:
        cell = next(row for row in config.cells if row.cell_id == cell_id)
    except StopIteration as exc:
        raise ValueError(f"unknown campaign cell {cell_id}") from exc
    gate = _treatment_gate(cell, state.setdefault("cells", {}))
    if gate:
        raise ValueError(gate)
    result = OfficialResult.load(result_path)
    review = review_result(
        config.baseline(cell.baseline_id), result,
        absolute_tolerance=cell.result.absolute_tolerance,
        require_exact_cohort=cell.result.require_exact_cohort,
    ) if cell.mode == CampaignMode.NATIVE else None
    state["cells"][cell.cell_id] = {
        "state": "COMPLETED", "finished_at": _now(),
        "result": result.model_dump(mode="json"),
        "review": review.model_dump(mode="json") if review else None,
        "reproduction_status": review.status.value if review else None,
        "imported_from": str(result_path),
        "agent_id": cell.agent_id, "connection": cell.connection,
        "engine_pra_enabled": cell.engine_pra_enabled,
        "gateway_pra_enabled": cell.gateway_pra_enabled,
        "gateway_mode": cell.gateway_mode,
        "comparison_group": cell.comparison_group,
        "paired_cell": cell.paired_cell,
        "evidence_role": cell.evidence_role,
        "selection_contract": cell.selection_contract,
        "priority": cell.priority,
    }
    _persist(config, state, root, state_path)
    return state


def _treatment_gate(cell: Any, cells: dict[str, Any]) -> str | None:
    if cell.mode == CampaignMode.NATIVE:
        return None
    baseline = cells.get(cell.baseline_cell or "", {})
    if baseline.get("reproduction_status") != ReproductionStatus.BASELINE_REPRODUCED.value:
        return (
            f"Treatment requires {cell.baseline_cell}=BASELINE_REPRODUCED; "
            f"observed {baseline.get('reproduction_status', baseline.get('state', 'PENDING'))}."
        )
    observed_score = (baseline.get("result") or {}).get("score")
    if observed_score is None or observed_score < cell.minimum_baseline_score:
        return (
            f"Treatment requires baseline score >= {cell.minimum_baseline_score:.3f}; "
            f"observed {observed_score!r}."
        )
    if cell.paired_cell:
        paired = cells.get(cell.paired_cell, {})
        if paired.get("state") != "COMPLETED":
            return (
                f"Treatment requires paired cell {cell.paired_cell}=COMPLETED; "
                f"observed {paired.get('state', 'PENDING')}."
            )
    return None


def _persist(config: CampaignConfig, state: dict[str, Any], root: Path, state_path: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(state_path)
    write_reports(config, state, root)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_environment(overrides: dict[str, str]) -> dict[str, str]:
    """Keep child cells on the interpreter selected for the scheduler."""

    environment = os.environ.copy()
    environment.update({key: os.path.expandvars(value) for key, value in overrides.items()})
    # Do not resolve the executable symlink: POSIX virtualenv ``python`` often
    # points into /usr/bin, while its containing directory is what activates
    # the environment for child commands.
    interpreter_directory = str(Path(sys.executable).absolute().parent)
    environment["PATH"] = os.pathsep.join(
        (interpreter_directory, environment.get("PATH", ""))
    ).rstrip(os.pathsep)
    return environment


def _expand_command(command: tuple[str, ...]) -> list[str]:
    """Expand endpoint variables without invoking a shell or exposing credentials."""

    return [os.path.expandvars(value) for value in command]


def _repository_root(config_path: Path) -> Path:
    """Find the checkout root without assuming a fixed config nesting depth."""

    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "experiments").is_dir():
            return parent
    return Path.cwd().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=16.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-cell")
    parser.add_argument("--record-result", type=Path)
    args = parser.parse_args()
    if bool(args.record_cell) != bool(args.record_result):
        parser.error("--record-cell and --record-result must be provided together")
    state = (
        record_result(args.config, cell_id=args.record_cell, result_path=args.record_result)
        if args.record_cell else
        run_campaign(args.config, max_hours=args.max_hours, resume=args.resume, dry_run=args.dry_run)
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
