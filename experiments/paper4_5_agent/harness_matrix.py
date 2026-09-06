"""Crash-resumable coding-agent matrix over official Harbor tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import Field, model_validator

from experiments.agents.runner import import_harbor_job, load_runs
from experiments.agents.schema import (
    BenchmarkManifest,
    PRAProfile,
    PRAMode,
    StrictModel,
)


class MatrixModel(StrictModel):
    """One served model and the direct/gateway routes that reach its engine."""

    model_id: str
    served_model: str
    model_revision: str | None = None
    engine: str
    engine_version: str
    quantization: str | None = None
    base_url_env: str | None = None
    api_key_env: str = "PRA_AGENT_API_KEY"
    routes: tuple["MatrixRoute", ...] = ()
    enabled: bool = True

    @model_validator(mode="after")
    def endpoint_contract_is_complete(self) -> "MatrixModel":
        if not self.routes and not self.base_url_env:
            raise ValueError("a legacy model requires base_url_env")
        route_ids = [route.route_id for route in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError(f"route_id values must be unique for {self.model_id}")
        known = set(route_ids)
        for route in self.routes:
            if route.requires_route and route.requires_route not in known:
                raise ValueError(
                    f"route {route.route_id} requires unknown route {route.requires_route}"
                )
        return self


class MatrixRoute(StrictModel):
    """One agent-to-engine path with explicit PRA placement semantics."""

    route_id: str
    connection: Literal["direct", "gateway"]
    base_url_env: str
    api_key_env: str | None = None
    pra_mode: PRAMode = PRAMode.NONE
    pra_profile: PRAProfile = PRAProfile.NONE
    engine_pra_enabled: bool = False
    gateway_pra_enabled: bool = False
    gateway_mode: Literal["G00", "G10", "G01", "G11"] | None = None
    engine_target_id: str
    comparison_group: str | None = None
    requires_route: str | None = None
    preflight: bool = True
    enabled: bool = True

    @model_validator(mode="after")
    def placement_is_coherent(self) -> "MatrixRoute":
        if self.connection == "direct" and (self.gateway_pra_enabled or self.gateway_mode):
            raise ValueError("a direct route cannot enable or name a gateway")
        if self.connection == "gateway" and self.gateway_mode is None:
            raise ValueError("a gateway route requires gateway_mode")
        if self.gateway_pra_enabled and self.gateway_mode not in {"G01", "G11"}:
            raise ValueError("gateway PRA requires G01 or G11 mediation")
        if self.pra_mode == PRAMode.NATIVE_MEMORY and not self.engine_pra_enabled:
            raise ValueError("native-memory requires engine_pra_enabled=true")
        if self.pra_mode == PRAMode.NONE and self.pra_profile != PRAProfile.NONE:
            raise ValueError("a No-PRA route must use profile none")
        if self.pra_mode != PRAMode.NONE and self.pra_profile == PRAProfile.NONE:
            raise ValueError("a PRA route requires an explicit profile")
        return self


class MatrixHarness(StrictModel):
    """A Harbor agent implementation and its pinned constructor arguments."""

    harness_id: str
    agent: str
    result_agent_name: str | None = None
    version: str
    model_name: str | None = None
    kwargs: Mapping[str, str | int | float | bool] = Field(default_factory=dict)
    agent_class: Literal["open_source", "commercial", "first_party"] = "open_source"
    connections: tuple[Literal["direct", "gateway"], ...] = ("direct", "gateway")
    notes: tuple[str, ...] = ()
    enabled: bool = True


class HarnessMatrixConfig(StrictModel):
    """Frozen baseline or PRA-transport matrix used to qualify agent harnesses."""

    schema_version: Literal[1, 2] = 1
    matrix_kind: Literal["baseline", "pra_transport"] = "baseline"
    evidence_role: Literal[
        "baseline_admission", "transport_qualification"
    ] = "baseline_admission"
    campaign_id: str
    manifest: str
    output_directory: str
    agent_host: str
    hardware: Mapping[str, Any] = Field(default_factory=dict)
    minimum_runs: int = Field(default=15, ge=1)
    minimum_runs_per_harness: int = Field(default=10, ge=1)
    minimum_success_rate: float = Field(default=0.10, ge=0, le=1)
    maximum_success_rate: float = Field(default=0.90, ge=0, le=1)
    baseline_admission_path: str | None = None
    models: tuple[MatrixModel, ...]
    harnesses: tuple[MatrixHarness, ...]

    @model_validator(mode="after")
    def unique_matrix_ids(self) -> "HarnessMatrixConfig":
        for name, values in (
            ("model_id", [row.model_id for row in self.models]),
            ("harness_id", [row.harness_id for row in self.harnesses]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} values must be unique")
        if not any(row.enabled for row in self.models):
            raise ValueError("at least one model must be enabled")
        if not any(row.enabled for row in self.harnesses):
            raise ValueError("at least one harness must be enabled")
        if self.matrix_kind == "pra_transport":
            if self.evidence_role != "transport_qualification":
                raise ValueError(
                    "pra_transport matrices are transport qualification, not efficacy"
                )
            if self.schema_version != 2:
                raise ValueError("pra_transport matrices require schema_version 2")
            if not self.baseline_admission_path:
                raise ValueError("pra_transport matrices require baseline_admission_path")
            for model in self.models:
                if not model.enabled:
                    continue
                groups: dict[str, list[MatrixRoute]] = {}
                for route in model.routes:
                    if route.enabled and route.comparison_group:
                        groups.setdefault(route.comparison_group, []).append(route)
                if not groups:
                    raise ValueError(f"{model.model_id} has no matched comparison group")
                for group, routes in groups.items():
                    connections = {route.connection for route in routes}
                    targets = {route.engine_target_id for route in routes}
                    if connections != {"direct", "gateway"} or len(targets) != 1:
                        raise ValueError(
                            f"comparison group {group} must pair direct and gateway routes "
                            "to one engine_target_id"
                        )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "HarnessMatrixConfig":
        return cls.model_validate(
            yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        )


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silently shadowed configuration fields."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping,
)


def matrix_cells(
    config: HarnessMatrixConfig, manifest: BenchmarkManifest,
) -> list[tuple[str, MatrixModel, MatrixHarness, str, int, MatrixRoute]]:
    """Enumerate the frozen matrix in task-major order for early cross-harness data."""

    cells = []
    for model in config.models:
        if not model.enabled:
            continue
        for repeat in range(manifest.repeats):
            for task_id in manifest.task_ids:
                for harness in config.harnesses:
                    if not harness.enabled:
                        continue
                    for route in _model_routes(model):
                        if not route.enabled or route.connection not in harness.connections:
                            continue
                        cell_id = _matrix_cell_id(
                            model=model, harness=harness, task_id=task_id,
                            repeat=repeat, route=route,
                        )
                        cells.append((cell_id, model, harness, task_id, repeat, route))
    return cells


def _model_routes(model: MatrixModel) -> tuple[MatrixRoute, ...]:
    """Map a version-1 model to its historical direct No-PRA route."""

    if model.routes:
        return model.routes
    return (MatrixRoute(
        route_id="direct-no-pra", connection="direct",
        base_url_env=str(model.base_url_env), api_key_env=model.api_key_env,
        pra_mode=PRAMode.NONE, pra_profile=PRAProfile.NONE,
        engine_target_id=model.model_id, preflight=False,
    ),)


def harbor_command(
    *, harbor: str, manifest: BenchmarkManifest, model: MatrixModel,
    harness: MatrixHarness, task_id: str, job_directory: Path,
    base_url: str, api_key: str,
) -> list[str]:
    """Build one isolated official-Harbor trial command."""

    qualified_task = task_id if task_id.startswith("terminal-bench/") else f"terminal-bench/{task_id}"
    command = [
        harbor, "run", "-d", manifest.dataset, "-a", harness.agent,
        "-m", harness.model_name or f"openai/{model.served_model}",
    ]
    kwargs = {"version": harness.version, **harness.kwargs}
    for name, value in kwargs.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        command.extend(("--ak", f"{name}={rendered}"))
    command.extend((
        "-i", qualified_task,
        "--agent-env", f"OPENAI_API_KEY={api_key}",
        "--agent-env", f"OPENAI_BASE_URL={base_url.rstrip('/')}",
        "--allow-agent-host", _endpoint_host(base_url),
        "--jobs-dir", str(job_directory), "-n", "1", "-y",
    ))
    return command


def run_matrix(
    config_path: Path, *, resume: bool, dry_run: bool = False,
    max_cells: int | None = None, harbor: str = "harbor",
) -> dict[str, Any]:
    """Run and checkpoint a frozen matrix, preserving failed cells for retry."""

    config = HarnessMatrixConfig.load(config_path)
    repository = _repository_root(config_path.resolve())
    manifest = BenchmarkManifest.load(repository / config.manifest)
    if manifest.benchmark != "terminal-bench":
        raise ValueError("the Harbor matrix requires a terminal-bench manifest")
    output = (repository / config.output_directory).resolve()
    state_path = output / "matrix_state.json"
    state: dict[str, Any] = {
        "schema_version": config.schema_version,
        "campaign_id": config.campaign_id,
        "matrix_kind": config.matrix_kind,
        "evidence_role": config.evidence_role,
        "pra_enabled": any(
            route.pra_mode != PRAMode.NONE
            for model in config.models if model.enabled
            for route in _model_routes(model) if route.enabled
        ),
        "manifest": config.manifest,
        "cells": {},
    }
    if resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    launched = 0
    admission = _load_baseline_admission(config, repository)
    preflighted: set[tuple[str, str, str]] = set()

    for cell_id, model, harness_spec, task_id, repeat, route in matrix_cells(config, manifest):
        record = state.setdefault("cells", {}).setdefault(cell_id, {"state": "PENDING"})
        if record.get("state") == "COMPLETED":
            continue
        if max_cells is not None and launched >= max_cells:
            break
        gate = _route_gate(
            config=config, model=model, harness=harness_spec, route=route,
            task_id=task_id, repeat=repeat, cells=state["cells"], admission=admission,
        )
        if gate:
            record.update(
                state="BLOCKED", reason=gate, model=model.model_id, task_id=task_id,
                harness=harness_spec.harness_id, repeat=repeat, route_id=route.route_id,
                connection=route.connection, pra_mode=route.pra_mode.value,
                engine_pra_enabled=route.engine_pra_enabled,
                gateway_pra_enabled=route.gateway_pra_enabled,
            )
            _persist(config, manifest, state, output, state_path, admission=admission)
            continue
        base_url = os.environ.get(route.base_url_env)
        api_key_env = route.api_key_env or model.api_key_env
        api_key = os.environ.get(api_key_env, "pra-local")
        if not base_url and not dry_run:
            record.update(state="BLOCKED", reason=f"missing environment variable {route.base_url_env}")
            _persist(config, manifest, state, output, state_path, admission=admission)
            continue
        if not dry_run and route.preflight:
            key = (base_url or "", route.route_id, model.served_model)
            if key not in preflighted:
                try:
                    _route_preflight(
                        base_url or "", api_key=api_key, model=model, route=route,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    record.update(state="BLOCKED", reason=str(exc))
                    _persist(config, manifest, state, output, state_path, admission=admission)
                    continue
                preflighted.add(key)
        cell_dir = output / "cells" / cell_id
        normalized_dir = cell_dir / "normalized"
        if resume and not dry_run:
            recovered = _completed_attempt(cell_dir)
            if recovered is not None:
                try:
                    row = _normalize_attempt(
                        attempt_dir=recovered, normalized_dir=normalized_dir,
                        manifest=manifest, model=model, config=config,
                        harness=harness_spec, route=route, task_id=task_id,
                    )
                    invalid_reason = _invalid_trial_reason(row)
                    if invalid_reason:
                        record.update(
                            state="INVALID", finished_at=_now(),
                            reason=invalid_reason, normalized_result=None,
                        )
                    else:
                        record.update(
                            state="COMPLETED", finished_at=_now(),
                            success=row.outcome.success,
                            official_score=row.outcome.official_score,
                            normalized_result=str(
                                (normalized_dir / "runs.jsonl").relative_to(output)
                            ),
                        )
                    record["active_attempt"] = recovered.name
                    record["attempts"] = max(
                        int(record.get("attempts", 0)), int(recovered.name[1:]),
                    )
                    _persist(config, manifest, state, output, state_path, admission=admission)
                    continue
                except (OSError, ValueError):
                    pass
        attempt = int(record.get("attempts", 0)) + 1
        jobs_dir = cell_dir / "attempts" / f"a{attempt:03d}"
        command = harbor_command(
            harbor=harbor, manifest=manifest, model=model, harness=harness_spec,
            task_id=task_id, job_directory=jobs_dir,
            base_url=base_url or "http://HOST_REQUIRED/v1", api_key=api_key,
        )
        record.update(
            state="PENDING" if dry_run else "RUNNING",
            model=model.model_id, task_id=task_id, harness=harness_spec.harness_id,
            repeat=repeat, route_id=route.route_id, connection=route.connection,
            pra_enabled=route.pra_mode != PRAMode.NONE,
            pra_mode=route.pra_mode.value, pra_profile=route.pra_profile.value,
            engine_pra_enabled=route.engine_pra_enabled,
            gateway_pra_enabled=route.gateway_pra_enabled,
            gateway_mode=route.gateway_mode, engine_target_id=route.engine_target_id,
            comparison_group=route.comparison_group, attempts=attempt,
            active_attempt=f"a{attempt:03d}", command=_redact(command),
        )
        _persist(config, manifest, state, output, state_path, admission=admission)
        if dry_run:
            launched += 1
            continue

        launched += 1
        record["started_at"] = _now()
        cell_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(repository), environment.get("PYTHONPATH")) if value
        )
        completed = subprocess.run(
            command, cwd=repository, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        (cell_dir / "launcher.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            record.update(
                state="FAILED", finished_at=_now(), returncode=completed.returncode,
                reason="Harbor trial failed; inspect launcher.log and resume to retry.",
            )
            _persist(config, manifest, state, output, state_path, admission=admission)
            continue
        try:
            row = _normalize_attempt(
                attempt_dir=jobs_dir, normalized_dir=normalized_dir,
                manifest=manifest, model=model, config=config,
                harness=harness_spec, route=route, task_id=task_id,
            )
            invalid_reason = _invalid_trial_reason(row)
            if invalid_reason:
                record.update(
                    state="INVALID", finished_at=_now(), reason=invalid_reason,
                    normalized_result=None,
                )
                _persist(config, manifest, state, output, state_path, admission=admission)
                continue
            record.update(
                state="COMPLETED", finished_at=_now(),
                success=row.outcome.success,
                official_score=row.outcome.official_score,
                normalized_result=str((normalized_dir / "runs.jsonl").relative_to(output)),
            )
        except (OSError, ValueError) as exc:
            record.update(state="FAILED", finished_at=_now(), reason=str(exc))
        _persist(config, manifest, state, output, state_path, admission=admission)
    _persist(config, manifest, state, output, state_path, admission=admission)
    return state


def _normalize_attempt(
    *, attempt_dir: Path, normalized_dir: Path, manifest: BenchmarkManifest,
    model: MatrixModel, config: HarnessMatrixConfig, harness: MatrixHarness,
    route: MatrixRoute, task_id: str,
) -> Any:
    rows = import_harbor_job(
        attempt_dir, manifest, output=normalized_dir,
        engine=model.engine, engine_version=model.engine_version,
        host=config.agent_host, hardware=config.hardware,
        model=model.served_model, model_revision=model.model_revision,
        quantization=model.quantization, pra_mode=route.pra_mode,
        pra_profile=route.pra_profile, connection=route.connection,
        engine_pra_enabled=route.engine_pra_enabled,
        gateway_pra_enabled=route.gateway_pra_enabled,
        gateway_mode=route.gateway_mode,
        protocol="openai-chat-completions",
        run_metadata={
            "matrix_harness_id": harness.harness_id,
            "route_id": route.route_id,
            "engine_target_id": route.engine_target_id,
            "comparison_group": route.comparison_group,
        },
    )
    if len(rows) != 1 or rows[0].identity.task_id != task_id:
        raise ValueError(f"expected one normalized row for {task_id}, found {len(rows)}")
    return rows[0]


def _completed_attempt(cell_dir: Path) -> Path | None:
    """Find the newest Harbor attempt whose job has reached a terminal state."""

    attempts = cell_dir / "attempts"
    for attempt in sorted(attempts.glob("a[0-9][0-9][0-9]"), reverse=True):
        for result_path in sorted(attempt.glob("*/result.json"), reverse=True):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            stats = payload.get("stats") or {}
            total = int(payload.get("n_total_trials") or 0)
            terminal = int(stats.get("n_completed_trials") or 0) + int(
                stats.get("n_cancelled_trials") or 0
            )
            if total > 0 and terminal == total and not stats.get("n_running_trials"):
                return attempt
    return None


def _invalid_trial_reason(row: Any) -> str | None:
    """Reject pre-inference adapter failures from model-quality statistics."""

    failure = row.outcome.failure_kind
    no_model_activity = (
        row.behavior.model_calls == 0
        and row.tokens.input_tokens == 0
        and row.tokens.output_tokens == 0
    )
    pre_inference_failures = {
        "NonZeroAgentExitCodeError",
        "AgentSetupTimeoutError",
        "EnvironmentBuildError",
        "EnvironmentStartupError",
    }
    if failure in pre_inference_failures and no_model_activity:
        return (
            f"Harness/infrastructure failure before model activity: {failure}. "
            "The cell remains retryable and is excluded from admission statistics."
        )
    return None


def _load_baseline_admission(
    config: HarnessMatrixConfig, repository: Path,
) -> dict[str, Any]:
    """Load the frozen No-PRA admission decision used by a PRA matrix."""

    if config.matrix_kind == "baseline":
        return {}
    path = (repository / str(config.baseline_admission_path)).resolve()
    if not path.is_file():
        return {
            "status": "BLOCKED", "eligible": False,
            "reason": f"Baseline admission artifact is missing: {path}",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED", "eligible": False,
            "reason": f"Baseline admission artifact is invalid: {exc}",
        }
    gate = dict(payload.get("admission_gate") or payload)
    baseline_summary = payload.get("summary") or {}
    by_harness = baseline_summary.get("by_harness") or {}
    harness_gates = {}
    for name, values in by_harness.items():
        runs = int(values.get("runs") or 0)
        successes = int(values.get("successes") or 0)
        rate = values.get("success_rate")
        eligible = (
            runs >= config.minimum_runs_per_harness
            and isinstance(rate, (int, float))
            and config.minimum_success_rate <= float(rate) <= config.maximum_success_rate
        )
        harness_gates[str(name)] = {
            "status": "ELIGIBLE" if eligible else "BLOCKED",
            "eligible": eligible, "runs": runs, "successes": successes,
            "official_success_rate": rate,
            "reason": (
                "Harness baseline is inside the preregistered comparison band."
                if eligible else
                "Harness baseline does not satisfy its size and success-rate gate."
            ),
        }
    if "eligible" not in gate and payload.get("official_grader") is True:
        score = payload.get("score")
        total = int(payload.get("total") or 0)
        eligible = (
            isinstance(score, (int, float))
            and total >= config.minimum_runs
            and config.minimum_success_rate <= float(score) <= config.maximum_success_rate
        )
        gate = {
            "status": "ELIGIBLE" if eligible else "BLOCKED",
            "eligible": eligible,
            "runs": total,
            "successes": int(payload.get("resolved") or 0),
            "official_success_rate": score,
            "target_range": [config.minimum_success_rate, config.maximum_success_rate],
            "reason": (
                "The official No-PRA cohort is inside the preregistered comparison band."
                if eligible else
                "The official No-PRA cohort does not satisfy the size and success-rate gate."
            ),
        }
    eligible = gate.get("eligible") is True and gate.get("status") == "ELIGIBLE"
    if harness_gates:
        enabled_names = {
            row.result_agent_name or row.agent for row in config.harnesses if row.enabled
        }
        eligible = all(harness_gates.get(name, {}).get("eligible") for name in enabled_names)
        gate["status"] = "ELIGIBLE" if eligible else "BLOCKED"
        gate["reason"] = (
            "Every enabled harness has an admitted No-PRA baseline."
            if eligible else
            "At least one enabled harness lacks an admitted No-PRA baseline."
        )
    return {
        **dict(gate), "eligible": eligible, "by_harness": harness_gates,
        "admission_scope": "per_harness",
        "eligible_harnesses": sorted(
            name for name, values in harness_gates.items() if values.get("eligible")
        ),
        "source": str(path.relative_to(repository)),
        "reason": str(gate.get("reason") or (
            "Baseline admitted." if eligible else "Baseline is not admitted."
        )),
    }


def _route_gate(
    *, config: HarnessMatrixConfig, model: MatrixModel, harness: MatrixHarness,
    route: MatrixRoute, task_id: str, repeat: int,
    cells: Mapping[str, Any], admission: Mapping[str, Any],
) -> str | None:
    """Enforce baseline admission and direct-before-gateway pair ordering."""

    if config.matrix_kind == "pra_transport":
        name = harness.result_agent_name or harness.agent
        harness_gate = (admission.get("by_harness") or {}).get(name)
        if harness_gate is not None and not harness_gate.get("eligible"):
            return (
                f"PRA transport matrix requires an admitted {name} baseline: "
                f"{harness_gate.get('reason')}"
            )
        if harness_gate is None and not admission.get("eligible"):
            return (
                "PRA transport matrix requires an admitted No-PRA baseline: "
                f"{admission.get('reason')}"
            )
    if route.requires_route:
        required = next(
            candidate for candidate in _model_routes(model)
            if candidate.route_id == route.requires_route
        )
        required_id = _matrix_cell_id(
            model=model, harness=harness, task_id=task_id,
            repeat=repeat, route=required,
        )
        state = cells.get(required_id, {}).get("state", "PENDING")
        if state != "COMPLETED":
            return f"Route {route.route_id} requires {required_id}=COMPLETED; observed {state}."
    return None


def _matrix_cell_id(
    *, model: MatrixModel, harness: MatrixHarness, task_id: str,
    repeat: int, route: MatrixRoute,
) -> str:
    route_part = f"__{route.route_id}" if model.routes else ""
    return f"{model.model_id}__{task_id}__{harness.harness_id}{route_part}__r{repeat}"


def _route_preflight(
    base_url: str, *, api_key: str, model: MatrixModel, route: MatrixRoute,
) -> None:
    """Verify that a declared PRA placement is exposed by the selected endpoint."""

    root = base_url.rstrip("/").removesuffix("/v1")

    def get(path: str) -> dict[str, Any]:
        try:
            request = urllib.request.Request(
                f"{root}{path}", headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"route preflight failed for {root}{path}: {exc}") from exc

    catalog = get("/v1/models")
    model_ids = {
        str(row.get("id")) for row in catalog.get("data", ())
        if isinstance(row, Mapping) and row.get("id")
    }
    if model.served_model not in model_ids:
        raise RuntimeError(
            f"route {route.route_id} does not advertise {model.served_model!r}"
        )
    if not (route.engine_pra_enabled or route.gateway_pra_enabled):
        return
    capabilities = get("/v1/pra/capabilities")
    effective = capabilities.get("effective_capabilities") or capabilities
    engine = capabilities.get("engine") or {}
    native_kv = bool(effective.get("native_kv") or engine.get("native_kv"))
    if route.engine_pra_enabled and not native_kv:
        raise RuntimeError(
            f"route {route.route_id} claims a PRA engine but native_kv is not effective"
        )
    observed_mode = capabilities.get("gateway_mode")
    if route.connection == "gateway" and observed_mode != route.gateway_mode:
        raise RuntimeError(
            f"route {route.route_id} expected gateway mode {route.gateway_mode}, "
            f"observed {observed_mode!r}"
        )


def _persist(
    config: HarnessMatrixConfig, manifest: BenchmarkManifest,
    state: dict[str, Any], output: Path, state_path: Path,
    *, admission: Mapping[str, Any] | None = None,
) -> None:
    """Atomically persist state and rebuild aggregate rows from completed cells."""

    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(state_path)
    rows = []
    for record in state.get("cells", {}).values():
        relative = record.get("normalized_result")
        if record.get("state") == "COMPLETED" and relative:
            rows.extend(load_runs(output / relative))
    aggregate = output / "runs.jsonl"
    aggregate.write_text("".join(row.json_line() + "\n" for row in rows), encoding="utf-8")
    summary = _summarize(rows)
    gate = (
        dict(admission or {})
        if config.matrix_kind == "pra_transport"
        else _admission_gate(config, rows)
    )
    comparisons = _paired_route_comparisons(rows)
    (output / "summary.json").write_text(
        json.dumps({
            "campaign_id": config.campaign_id,
            "matrix_kind": config.matrix_kind,
            "evidence_role": config.evidence_role,
            "pra_enabled": state.get("pra_enabled", False),
            "expected_runs": len(matrix_cells(config, manifest)),
            "completed_runs": len(rows),
            "summary": summary,
            "admission_gate": gate,
            "paired_route_comparisons": comparisons,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(
        output / "report.md", config=config, manifest=manifest,
        expected=len(matrix_cells(config, manifest)), summary=summary, gate=gate,
        comparisons=comparisons,
    )


def _write_markdown_report(
    path: Path, *, config: HarnessMatrixConfig, manifest: BenchmarkManifest,
    expected: int, summary: Mapping[str, Any], gate: Mapping[str, Any],
    comparisons: list[Mapping[str, Any]],
) -> None:
    """Render a reviewable checkpoint without weakening official-result semantics."""

    lines = [
        f"# {config.campaign_id}", "",
        (
            "This is an official-Harbor **No-PRA baseline**. It does not estimate a PRA effect."
            if config.matrix_kind == "baseline" else
            "This is a matched **PRA engine direct vs PRA gateway + PRA engine** matrix."
        ),
        "",
        f"- Frozen manifest: `{manifest.name}` ({len(manifest.task_ids)} tasks)",
        f"- Evidence role: `{config.evidence_role}`",
        f"- Completed: `{summary['runs']}/{expected}` trials",
        f"- Admission: `{gate['status']}` - {gate['reason']}",
        f"- Tasks solved by any harness: `{summary['tasks_solved_any']}/"
        f"{summary['unique_tasks']}`",
        "", "| Harness / route | Runs | Success | Reported input tokens | Token coverage | Model calls | Tool calls | Wall h |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for harness, values in sorted(summary["by_harness_route"].items()):
        lines.append(
            f"| `{harness}` | {values['runs']} | {values['successes']}/{values['runs']} "
            f"({values['success_rate']:.1%}) | {values['input_tokens']:,} | "
            f"{values['token_reported_runs']}/{values['runs']} | "
            f"{values['model_calls']:,} | "
            f"{values['tool_calls']:,} | {values['wall_ms'] / 3_600_000:.2f} |"
        )
    lines.extend((
        "", (
            "The admission decision requires the complete preregistered matrix. "
            if config.matrix_kind == "baseline" else
            "Treatment admission is applied per harness from the frozen No-PRA matrix. "
        ) + "Harness rows are not an agent ranking because prompts, tools, and loop policies differ.", "",
    ))
    if comparisons:
        lines.extend((
            "## Matched direct/gateway comparisons", "",
            "| Agent | Model | Group | Pairs | Outcome matches | Direct success | Gateway success | Regressions | Recoveries |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ))
        for row in comparisons:
            lines.append(
                f"| `{row['agent']}` | `{row['model']}` | `{row['comparison_group']}` | "
                f"{row['pairs']} | {row['outcome_matches']} | {row['direct_successes']} | "
                f"{row['gateway_successes']} | {row['regressions']} | {row['recoveries']} |"
            )
        lines.extend((
            "", "A gateway effect is interpretable only for complete task/repeat pairs that "
            "share the same engine target, model, PRA mode, and PRA profile.", "",
        ))
    path.write_text("\n".join(lines), encoding="utf-8")


def _summarize(rows: list[Any]) -> dict[str, Any]:
    """Summarize official rows without importing the PRA runtime package."""

    by_harness: dict[str, dict[str, int | float]] = {}
    by_harness_route: dict[str, dict[str, int | float]] = {}
    for row in rows:
        route_id = str(row.metadata.get("route_id") or row.identity.connection)
        values = (
            {"runs": 0, "successes": 0, "input_tokens": 0, "output_tokens": 0,
             "token_reported_runs": 0, "model_calls": 0, "tool_calls": 0,
             "wall_ms": 0.0}
        )
        for key in (row.identity.agent, f"{row.identity.agent} / {route_id}"):
            target = by_harness if key == row.identity.agent else by_harness_route
            bucket = target.setdefault(key, dict(values))
            bucket["runs"] += 1
            bucket["successes"] += int(row.outcome.success)
            bucket["input_tokens"] += row.tokens.input_tokens
            bucket["output_tokens"] += row.tokens.output_tokens
            bucket["token_reported_runs"] += int(
                bool(row.tokens.input_tokens or row.tokens.output_tokens)
            )
            bucket["model_calls"] += row.behavior.model_calls
            bucket["tool_calls"] += row.behavior.tool_calls
            bucket["wall_ms"] += row.timings.task_wall_ms
    for bucket in (*by_harness.values(), *by_harness_route.values()):
        runs = int(bucket["runs"])
        bucket["success_rate"] = bucket["successes"] / runs if runs else 0.0
    successes = sum(row.outcome.success for row in rows)
    task_ids = {row.identity.task_id for row in rows}
    solved_tasks = {row.identity.task_id for row in rows if row.outcome.success}
    return {
        "runs": len(rows), "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "unique_tasks": len(task_ids), "tasks_solved_any": len(solved_tasks),
        "token_reported_runs": sum(
            bool(row.tokens.input_tokens or row.tokens.output_tokens) for row in rows
        ),
        "input_tokens": sum(row.tokens.input_tokens for row in rows),
        "output_tokens": sum(row.tokens.output_tokens for row in rows),
        "by_harness": by_harness, "by_harness_route": by_harness_route,
    }


def _paired_route_comparisons(rows: list[Any]) -> list[dict[str, Any]]:
    """Compare only exact direct/gateway pairs from one declared route group."""

    grouped: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        group = row.metadata.get("comparison_group")
        if not group:
            continue
        key = (
            row.identity.agent, row.identity.model, str(group),
            row.identity.task_id, row.identity.repeat,
        )
        grouped.setdefault(key, {})[row.identity.connection] = row
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (agent, model, group, _task, _repeat), pair in grouped.items():
        if set(pair) != {"direct", "gateway"}:
            continue
        direct, gateway = pair["direct"], pair["gateway"]
        if direct.metadata.get("engine_target_id") != gateway.metadata.get("engine_target_id"):
            continue
        key = (agent, model, group)
        bucket = buckets.setdefault(key, {
            "agent": agent, "model": model, "comparison_group": group,
            "pairs": 0, "outcome_matches": 0, "direct_successes": 0,
            "gateway_successes": 0, "regressions": 0, "recoveries": 0,
            "direct_input_tokens": 0, "gateway_input_tokens": 0,
            "direct_wall_ms": 0.0, "gateway_wall_ms": 0.0,
        })
        direct_ok = bool(direct.outcome.success)
        gateway_ok = bool(gateway.outcome.success)
        bucket["pairs"] += 1
        bucket["outcome_matches"] += int(direct_ok == gateway_ok)
        bucket["direct_successes"] += int(direct_ok)
        bucket["gateway_successes"] += int(gateway_ok)
        bucket["regressions"] += int(direct_ok and not gateway_ok)
        bucket["recoveries"] += int(gateway_ok and not direct_ok)
        bucket["direct_input_tokens"] += direct.tokens.input_tokens
        bucket["gateway_input_tokens"] += gateway.tokens.input_tokens
        bucket["direct_wall_ms"] += direct.timings.task_wall_ms
        bucket["gateway_wall_ms"] += gateway.timings.task_wall_ms
    return [buckets[key] for key in sorted(buckets)]


def _admission_gate(config: HarnessMatrixConfig, rows: list[Any]) -> dict[str, Any]:
    """Apply the preregistered floor, ceiling, and full-matrix requirements."""

    successes = sum(row.outcome.success for row in rows)
    rate = successes / len(rows) if rows else None
    if len(rows) < config.minimum_runs:
        status = "BLOCKED"
        reason = f"Only {len(rows)} completed runs; all {config.minimum_runs} are required."
    elif successes == 0:
        status = "BLOCKED"
        reason = "No-PRA official success is zero; PRA efficacy comparisons are floor-confounded."
    elif rate is not None and rate < config.minimum_success_rate:
        status = "BLOCKED"
        reason = f"No-PRA success {rate:.1%} is below the promotion floor."
    elif rate is not None and rate > config.maximum_success_rate:
        status = "BLOCKED"
        reason = f"No-PRA success {rate:.1%} exceeds the promotion ceiling."
    else:
        status = "ELIGIBLE"
        reason = "The complete no-PRA matrix is inside the preregistered comparison band."
    return {
        "status": status, "eligible": status == "ELIGIBLE", "runs": len(rows),
        "successes": successes, "official_success_rate": rate,
        "target_range": [config.minimum_success_rate, config.maximum_success_rate],
        "reason": reason,
    }


def _endpoint_host(url: str) -> str:
    from urllib.parse import urlparse

    host = urlparse(url).hostname
    if not host:
        raise ValueError(f"base URL has no host: {url!r}")
    return host


def _redact(command: list[str]) -> list[str]:
    return [
        "OPENAI_API_KEY=***" if value.startswith("OPENAI_API_KEY=") else value
        for value in command
    ]


def _repository_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "experiments").is_dir():
            return parent
    return Path.cwd().resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--harbor", default="harbor")
    args = parser.parse_args()
    state = run_matrix(
        args.config, resume=args.resume, dry_run=args.dry_run,
        max_cells=args.max_cells, harbor=args.harbor,
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
