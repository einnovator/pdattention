"""Canonical product command line for PRA model, runtime, and agent workflows."""

from __future__ import annotations

import json
import sys
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import torch
import yaml

from .agent_profiles import AgentLauncher, AgentProfile, AgentProfileRegistry, load_mcp_config
from .agent_config import MCPServerConfig, PRAAgentSettings
from .agent_mcp import MCPClientManager
from .bundle import (
    BundleBuilder,
    BundleResolver,
    HubBundleCatalog,
    HubPublisher,
    PRAModelBundle,
    TrustedBundleRegistry,
)
from .evaluation import evaluate_router_features
from .execution_modes import ExecutionModeResolver
from .gateway_cli import gateway_cli
from .management_cli import engine_cli
from pra_registry.cli import registry_cli
from pra_control.cli import control_cli
from pra_router.cli import router_cli
from .model import PRAForCausalLM
from .onboarding import DoctorService, ModelInspector, ModelValidator, OnboardingPipeline, ProfileCalibrator, StructuralAdapterBuilder
from .observability import Observability, load_observability_config
from .product_config import dump_data
from .product_qualification import (
    EngineProductRegistry,
    QualificationService,
    assessment_init,
    environment_report,
    load_run,
    recommend_run,
    resolve_run_mode,
    render_report,
)
from .profile_benchmarks import ProfileBenchmarkRegistry
from .router import PRARouter
from .runtime import PRARuntimeConfig, VLLMThinBackend, runtime_capabilities
from .runtime_providers import RuntimeConfig, RuntimeManager, parse_engine_arguments
from .training import load_feature_rows, train_router
from .tui import AgentShell


# Keep the commercial Control Plane launcher beside the open engine and Registry
# commands while loading its web stack only when ``serve`` is invoked.


def _format(json_output: bool, yaml_output: bool) -> str:
    if json_output and yaml_output:
        raise click.UsageError("Choose only one of --json or --yaml.")
    return "json" if json_output else "yaml" if yaml_output else "human"


def _emit(value: Any, *, json_output: bool = False, yaml_output: bool = False) -> None:
    click.echo(dump_data(value, _format(json_output, yaml_output)))


def _output_options(function):
    function = click.option("--yaml", "yaml_output", is_flag=True, help="Emit YAML.")(function)
    function = click.option("--json", "json_output", is_flag=True, help="Emit JSON.")(function)
    return function


def _observability_options(function):
    decorators = (
        click.option("--observability", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option("--otel", is_flag=True, help="Enable OpenTelemetry tracing explicitly."),
        click.option("--otel-endpoint", metavar="URL"),
        click.option("--prometheus", is_flag=True, help="Enable Prometheus metrics explicitly."),
        click.option("--prometheus-port", type=click.IntRange(min=1, max=65535)),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


def _telemetry(observability, otel, otel_endpoint, prometheus, prometheus_port, *, service, start_server=True):
    overrides = {}
    if otel or otel_endpoint or prometheus or prometheus_port:
        overrides["enabled"] = True
    if otel or otel_endpoint:
        overrides["otel"] = {"enabled": True, **({"endpoint": otel_endpoint} if otel_endpoint else {})}
    if prometheus or prometheus_port:
        overrides["prometheus"] = {"enabled": True, **({"port": prometheus_port} if prometheus_port else {})}
    return Observability(
        load_observability_config(observability, overrides=overrides, service=service),
        start_server=start_server,
    )


def _metric(value: Any) -> str:
    if value is None:
        return "NOT_MEASURED"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _emit_doctor(value: Mapping[str, Any]) -> None:
    system = value["system"]
    click.echo("System")
    for name in ("os", "python", "executable", "cpu", "accelerator", "memory_bytes", "disk_free_bytes"):
        click.echo(f"  {name.replace('_', ' ').title()}: {_metric(system[name])}")
    click.echo("\nEngines")
    for row in value["engines"]:
        click.echo(f"  {row['name']} ({row['engine']})")
        click.echo(
            f"    Installed: {row['installed']}  Version: {_metric(row['version'])}  "
            f"Connection: {row['connection']}  Reachable: {_metric(row['reachable'])}"
        )
        click.echo(
            f"    Selected Context: {row['selected_context']}  Typed Transport: {row['typed_transport']}  "
            f"Native Memory: {row['native_memory']}  Native Serving: {row['native_serving']}"
        )
        click.echo(f"    Evidence: {row['evidence']} ({row['provenance']})")
    click.echo("\nModels and adapters")
    click.echo(f"  Known local bundles: {len(value['models_and_adapters']['known_local_bundles'])}")
    click.echo(f"  Profile registry: {value['models_and_adapters']['profile_registry']}")
    click.echo("\nProblems / next action")
    if value["problems"]:
        for problem in value["problems"]:
            click.echo(f"  - {problem}")
    else:
        click.echo("  No environment-level problems detected.")
    click.echo(f"  Next: {value['next_action']}")


def _emit_product_inspect(value: Mapping[str, Any]) -> None:
    """Render model qualification and bundle discovery as an actionable summary."""

    model = value["model"]["model"]
    engine = value["engine"]
    requested_engine = value.get("requested_engine", engine["engine"])
    click.echo(f"Model: {model['id']}")
    click.echo(f"Revision: {model['revision']}")
    click.echo(f"Engine: {requested_engine} ({engine['name']})")
    click.echo(f"Recommendation: {value['current_recommendation']}")

    published = value.get("published_bundle")
    if published:
        heading = (
            "\nPublished PRA bundle found"
            if published["status"] == "FOUND"
            else "\nPublished PRA bundle"
        )
        click.echo(heading)
        click.echo(f"  Repository: {_metric(published['source'])}")
        click.echo(f"  Revision: {_metric(published['bundle_revision'])}")
        click.echo(f"  Base revision: {_metric(published['base_revision'])}")
        click.echo(f"  Compatibility: {published['compatibility']}")
        click.echo(f"  Trust: {published['trust']}")
        click.echo(f"  Qualification: {_metric(published['qualification'])}")
        if published["status"] == "FOUND":
            click.echo(f"  Resolve: pra inspect {model['id']} -e {requested_engine} -a auto")
        else:
            click.echo(f"  Status: {published['status']}")

    resolution = value.get("bundle_resolution")
    if resolution:
        click.echo("\nPRA bundle resolution")
        click.echo(f"  Status: {resolution['status']}")
        click.echo(f"  Repository: {_metric(resolution['source'])}")
        click.echo(f"  Revision: {_metric(resolution['resolved_revision'])}")
        click.echo(f"  Trust: {resolution['trust']}")
        click.echo(f"  Local path: {_metric(resolution['local_path'])}")
        bundle = resolution.get("bundle")
        if bundle:
            click.echo(f"  Base revision: {_metric(bundle['base_model'].get('revision'))}")
            click.echo("  Compatibility: exact")


def _emit_bundle_catalog(
    value: Mapping[str, Any], *, json_output: bool, yaml_output: bool
) -> None:
    """Render bundle discovery compactly while preserving structured output."""

    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
        return
    click.echo(f"PRA bundle catalog ({value['count']})")
    click.echo(f"Source: {value['source']}")
    for row in value["bundles"]:
        repo = row.get("bundle_repo", row.get("repo_id"))
        click.echo(f"\n{repo}")
        click.echo(f"  Base model: {_metric(row.get('base_model'))}")
        click.echo(f"  Qualification: {_metric(row.get('qualification'))}")
        click.echo(f"  Trust: {_metric(row.get('trust'))}")
        if "auto_resolvable" in row:
            click.echo(f"  Auto resolvable: {row['auto_resolvable']}")
        engines = row.get("engine_compatibility")
        if engines:
            rendered = ", ".join(f"{name}={status}" for name, status in engines.items())
            click.echo(f"  Engines: {rendered}")
        profiles = row.get("profiles") or ()
        if profiles:
            click.echo(f"  Profiles: {', '.join(profiles)}")


def _emit_qualification(value: Mapping[str, Any], run_directory: Path | None = None) -> None:
    click.echo("MODE                STATUS              F1            VISIBLE TOKENS  TTFT P95 MS   SUCCESS REQ/S")
    for row in value["modes"].values():
        click.echo(
            f"{row['label']:<19} {row['status']:<19} {_metric(row['quality']['f1']):<13} "
            f"{_metric(row['context']['visible_input_tokens']):<15} "
            f"{_metric(row['performance']['ttft_p95_ms']):<13} "
            f"{_metric(row['performance']['successful_requests_per_second'])}"
        )
    click.echo("\nAttribution")
    for name in ("context_gain", "native_gain", "serving_gain"):
        gain = value["attribution"][name]
        measured = ", ".join(
            f"{key}={_metric(item)}" for key, item in gain.items()
            if key != "comparison" and item is not None
        ) or "NOT_MEASURED"
        click.echo(f"  {gain['comparison']}: {measured}")
    recommendation = value["recommendation"]
    click.echo(f"\nRecommendation: {recommendation['recommended_mode'] or 'No production recommendation'}")
    click.echo(f"  {recommendation['reason']}")
    for limitation in recommendation["limitations"]:
        click.echo(f"  - {limitation}")
    click.echo(f"Missing measurements: {len(value['missing_measurements'])}")
    if run_directory is not None:
        click.echo(f"Run directory: {run_directory}")


def _engine_config(
    model, engine, revision, device, endpoint, host, port, pra_bundle, profile,
    storage, storage_config, engine_args=(), verbose=False, observability=None,
    otel=False, otel_endpoint=None, prometheus=False, prometheus_port=None,
    management_api=False, management_host="127.0.0.1", management_port=9101,
    management_auth_mode="none", management_token_env="PRA_MANAGEMENT_TOKEN",
    management_metrics_url=None, management_trace_url=None,
    management_grafana_url=None,
):
    try:
        options = parse_engine_arguments(engine_args)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    return RuntimeConfig(
        engine=engine, model=model, revision=revision, device=device, endpoint=endpoint,
        host=host, port=port, pra_bundle=pra_bundle, profile=profile,
        engine_options=options, verbose=verbose, storage_profile=storage,
        storage_config=str(storage_config) if storage_config is not None else None,
        observability_config=str(observability) if observability is not None else None,
        otel=otel,
        otel_endpoint=otel_endpoint,
        prometheus=prometheus,
        prometheus_port=prometheus_port or 9464,
        management_api=management_api,
        management_host=management_host,
        management_port=management_port,
        management_auth_mode=management_auth_mode,
        management_token_env=management_token_env,
        management_metrics_url=management_metrics_url,
        management_trace_url=management_trace_url,
        management_grafana_url=management_grafana_url,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="pra-hf")
def cli() -> None:
    """Inspect, adapt, calibrate, run, and publish Progressive Retrieval Attention."""


cli.add_command(gateway_cli)
cli.add_command(engine_cli)
cli.add_command(registry_cli)
cli.add_command(control_cli)
cli.add_command(router_cli)


@cli.command("doctor")
@click.option("-v", "--verbose", is_flag=True)
@_output_options
def doctor(verbose, json_output, yaml_output) -> None:
    """Inspect the system, engines, local artifacts, and next action."""
    value = environment_report()
    if verbose:
        value["legacy_dependency_checks"] = DoctorService().run(verbose=True)
    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
    else:
        _emit_doctor(value)


@cli.command("engines")
@click.option("--details", metavar="ENGINE")
@_output_options
def engines(details, json_output, yaml_output) -> None:
    """Show the registry-backed engine capability and recommendation matrix."""
    registry = EngineProductRegistry.default()
    value = registry.details(details) if details else registry.matrix()
    if json_output or yaml_output or details:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
        return
    rows = value["engines"]
    click.echo("ENGINE          SELECTED CONTEXT  NATIVE MEMORY   NATIVE SERVING  RECOMMENDED")
    for row in rows:
        recommendation = str(row["recommended"]).split(".")[0]
        click.echo(
            f"{row['engine']:<15} {row['selected_context']:<17} "
            f"{row['native_memory']:<15} {row['native_serving']:<15} {recommendation}"
        )
    click.echo(f"\nProvenance: {value['provenance']} ({value['registry_version']})")


@cli.command("inspect")
@click.argument("model")
@click.option("-e", "--engine", default="hf", show_default=True)
@click.option("-r", "--revision")
@click.option(
    "-a", "--pra-bundle", default=None, metavar="AUTO|NONE|PATH|REPO",
    help="Resolve and validate a bundle. Omit to discover published bundles without downloading them.",
)
@_output_options
def product_inspect(model, engine, revision, pra_bundle, json_output, yaml_output) -> None:
    """Inspect one MODEL and ENGINE as a deployable combination."""
    try:
        metadata = ModelInspector().inspect(model, revision=revision)
        value = QualificationService().inspect(model, engine, metadata)
        value["requested_engine"] = engine
        resolved_model_revision = metadata["model"]["revision"]
        resolver = BundleResolver()
        if pra_bundle is None:
            value["published_bundle"] = resolver.discover(
                model=model,
                model_revision=resolved_model_revision,
                engine=engine,
            ).to_dict()
        else:
            value["bundle_resolution"] = resolver.resolve(
                pra_bundle,
                model=model,
                model_revision=resolved_model_revision,
                engine=engine,
            ).to_dict()
    except (KeyError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
    else:
        _emit_product_inspect(value)


@cli.command("evaluate")
@click.argument("model")
@click.option("-e", "--engine", required=True)
@click.option("-D", "--dataset", required=True)
@click.option("--measurements", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Import measured mode results as JSON.")
@click.option("--include-native-memory", is_flag=True)
@click.option("--include-native-serving", is_flag=True)
@click.option("--quality-threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.95, show_default=True)
@click.option("-r", "--revision")
@click.option("-a", "--pra-bundle", default="auto", show_default=True)
@click.option("-p", "--profile", default="recommended", show_default=True)
@click.option("-o", "--output", type=click.Path(path_type=Path))
@_output_options
def evaluate(model, engine, dataset, measurements, include_native_memory, include_native_serving, quality_threshold, revision, pra_bundle, profile, output, json_output, yaml_output) -> None:
    """Compare execution modes using one frozen selection and explicit gates."""
    destination = output or (
        Path(".pra/runs") / f"{model.replace('/', '--')}--{engine}--{int(__import__('time').time())}"
    )
    try:
        value = QualificationService().evaluate(
            model,
            engine=engine,
            dataset=dataset,
            output=destination,
            measurements=measurements,
            include_native_memory=include_native_memory,
            include_native_serving=include_native_serving,
            quality_threshold=quality_threshold,
            revision=revision,
            profile=profile,
        )
        value["bundle_resolution"] = BundleResolver().resolve(
            pra_bundle,
            model=model,
            model_revision=revision,
            engine=engine,
        ).to_dict()
    except (KeyError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    value["run_directory"] = str(destination)
    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
    else:
        _emit_qualification(value, destination)


@cli.command("recommend")
@click.argument("run", type=click.Path(exists=True, path_type=Path))
@click.option("--allow-unqualified-native", is_flag=True, hidden=True)
@_output_options
def recommend(run, allow_unqualified_native, json_output, yaml_output) -> None:
    """Recommend a mode from a completed qualification run."""
    try:
        document = load_run(run)
        value = recommend_run(document, allow_unqualified_native=allow_unqualified_native)
        value["mode_resolution"] = resolve_run_mode(
            document, allow_unqualified_native=allow_unqualified_native
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if Path(run).is_dir():
        (Path(run) / "recommendation.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
    else:
        _emit(value)


@cli.command("report")
@click.argument("run", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "format_name", type=click.Choice(["md", "html", "json"]), default="md", show_default=True)
@click.option("-o", "--output", type=click.Path(path_type=Path))
def report(run, format_name, output) -> None:
    """Export a qualification run or canonical evidence record."""
    try:
        document = load_run(run)
        rendered = render_report(document, format_name)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if format_name == "json" and output is None:
        click.echo(rendered, nl=False)
        return
    destination = output or (Path(run) / f"report.{format_name}" if Path(run).is_dir() else Path(f"report.{format_name}"))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(rendered, encoding="utf-8")
    click.echo(str(destination))


@cli.group("qualify")
def qualify_cli() -> None:
    """Run optional Native Memory and Native Serving qualification gates."""


def _qualification_options(function):
    decorators = (
        click.option("-e", "--engine", required=True),
        click.option("-D", "--dataset", required=True),
        click.option("--measurements", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option("-o", "--output", type=click.Path(path_type=Path), required=True),
        click.option("--quality-threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.95, show_default=True),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


@qualify_cli.command("native-memory")
@click.argument("model")
@_qualification_options
@_output_options
def qualify_native_memory(model, engine, dataset, measurements, output, quality_threshold, json_output, yaml_output) -> None:
    """Compare Selected Context with frozen-selection HOT and WARM memory."""
    try:
        value = QualificationService().evaluate(
            model, engine=engine, dataset=dataset, output=output,
            measurements=measurements, include_native_memory=True,
            quality_threshold=quality_threshold,
        )
    except (KeyError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
    else:
        _emit_qualification(value, output)


@qualify_cli.command("native-serving")
@click.argument("model")
@_qualification_options
@_output_options
def qualify_native_serving(model, engine, dataset, measurements, output, quality_threshold, json_output, yaml_output) -> None:
    """Measure scheduler-owned Native Serving beyond Native Memory."""
    try:
        value = QualificationService().evaluate(
            model, engine=engine, dataset=dataset, output=output,
            measurements=measurements, include_native_memory=True,
            include_native_serving=True, quality_threshold=quality_threshold,
        )
    except (KeyError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
    else:
        _emit_qualification(value, output)


@cli.group("assess")
def assess_cli() -> None:
    """Create a reproducible enterprise Optimization Assessment."""


@assess_cli.command("init")
@click.argument("name")
@click.option("--root", type=click.Path(path_type=Path), default=Path(".pra/assessments"), show_default=True)
def assess_init(name, root) -> None:
    """Create an editable assessment directory."""
    click.echo(str(assessment_init(name, root=root)))


@assess_cli.command("run")
@click.argument("assessment", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--measurements", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_output_options
def assess_run(assessment, measurements, json_output, yaml_output) -> None:
    """Run the configured assessment and persist its evidence artifacts."""
    config = yaml.safe_load((assessment / "config.yaml").read_text(encoding="utf-8")) or {}
    try:
        value = QualificationService().evaluate(
            str(config["model"]), engine=str(config["engine"]), dataset=str(config["dataset"]),
            output=assessment, measurements=measurements,
            include_native_memory=bool(config.get("include_native_memory")),
            include_native_serving=bool(config.get("include_native_serving")),
            profile=str(config.get("profile", "recommended")),
        )
    except (KeyError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output or yaml_output:
        _emit(value, json_output=json_output, yaml_output=yaml_output)
    else:
        _emit_qualification(value, assessment)


@assess_cli.command("report")
@click.argument("assessment", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--format", "format_name", type=click.Choice(["md", "html", "json"]), default="md", show_default=True)
def assess_report(assessment, format_name) -> None:
    """Regenerate an assessment report from its stored metrics."""
    document = load_run(assessment)
    destination = assessment / f"report.{format_name}"
    destination.write_text(render_report(document, format_name), encoding="utf-8")
    click.echo(str(destination))


@cli.group("model")
def model_cli() -> None:
    """Inspect and onboard model architectures."""


@model_cli.command("inspect")
@click.argument("model")
@click.option("-r", "--revision")
@click.option("--validate", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@_output_options
def model_inspect(model, revision, validate, verbose, json_output, yaml_output) -> None:
    """Inspect MODEL without loading full weights by default."""
    value = ModelInspector().inspect(model, revision=revision)
    if validate:
        value["validation"] = ModelValidator().validate(model, revision=revision, load_weights=True)
    if verbose:
        value["resolution"] = ["explicit model", "HF config metadata", "builtin structural mapping"]
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@model_cli.command("adapt")
@click.argument("model")
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
@click.option("-r", "--revision")
@click.option("-d", "--device", default="auto", show_default=True)
@click.option("-f", "--force", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@_output_options
def model_adapt(model, output, revision, device, force, verbose, json_output, yaml_output) -> None:
    """Generate a declarative structural adapter and validation record."""
    try:
        spec, path = StructuralAdapterBuilder().build(model, revision=revision, output=output, force=force)
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    validation = ModelValidator().validate(model, adapter=output, revision=revision, output=output)
    _emit({"adapter": str(path), "device_requested": device, "validation": validation, "specification": spec.to_dict()}, json_output=json_output, yaml_output=yaml_output)


@model_cli.command("validate")
@click.argument("model")
@click.option("-a", "--adapter")
@click.option("-r", "--revision")
@click.option("-d", "--device", default="auto", show_default=True)
@click.option("-s", "--suite", default="smoke", show_default=True)
@click.option("-o", "--output", type=click.Path(path_type=Path))
@_output_options
def model_validate(model, adapter, revision, device, suite, output, json_output, yaml_output) -> None:
    """Re-run the structural-adapter validation ladder."""
    value = ModelValidator().validate(
        model,
        adapter=adapter,
        revision=revision,
        suite=suite,
        output=output,
        load_weights=True,
        device=device,
    )
    value["device_requested"] = device
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@model_cli.command("onboard")
@click.argument("model", required=False)
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-s", "--suite", default="standard", show_default=True)
@click.option("-o", "--output", type=click.Path(path_type=Path), default=Path(".pra/runs"), show_default=True)
@click.option("-d", "--device", default="auto", show_default=True)
@click.option("-e", "--engine", default="hf", show_default=True)
@click.option("-r", "--revision")
@click.option("-f", "--force", is_flag=True)
@click.option("-j", "--jobs", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("-v", "--verbose", is_flag=True)
@_output_options
def model_onboard(model, manifest, suite, output, device, engine, revision, force, jobs, verbose, json_output, yaml_output) -> None:
    """Run inspection, adaptation, validation, and runtime packaging."""
    if bool(model) == bool(manifest):
        raise click.UsageError("Pass one MODEL or --manifest.")
    rows = [{"model": model, "revision": revision}]
    if manifest:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        rows = list(document.get("models", ()))
    results = []
    for row in rows:
        model_id = str(row["model"])
        result = OnboardingPipeline().run(model_id, output=output / model_id.replace("/", "--"), revision=row.get("revision", revision), suite=suite, force=force)
        result["requested"] = {"device": device, "engine": engine, "jobs": jobs}
        results.append(result)
    _emit({"runs": results}, json_output=json_output, yaml_output=yaml_output)


@cli.group("adapter")
def adapter_cli() -> None:
    """Inspect, train, and evaluate learned PRA adapters."""


@adapter_cli.command("inspect")
@click.argument("source")
@_output_options
def adapter_inspect(source, json_output, yaml_output) -> None:
    _emit(PRARouter.from_pretrained(source).artifact_config(), json_output=json_output, yaml_output=yaml_output)


@adapter_cli.group("train")
def adapter_train_cli() -> None:
    """Train routing, memory-use, and late-band adapters."""


@adapter_train_cli.command("routing")
@click.argument("model")
@click.option("-D", "--dataset", "datasets", multiple=True)
@click.option("--validation", "validation_datasets", multiple=True)
@click.option("--train-features", multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--validation-features", multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option("--model-family", type=click.Choice(["qwen", "llama", "gemma3"]), required=True)
@click.option("--routing-dim", default=128, show_default=True)
@click.option("--steps", default=512, show_default=True)
@click.option("--seed", default=53, show_default=True)
@click.option("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
@_output_options
def adapter_train_routing(model, datasets, validation_datasets, train_features, validation_features, output, model_family, routing_dim, steps, seed, device, json_output, yaml_output) -> None:
    """Train a routing adapter under the dataset-level public namespace."""
    if not train_features or not validation_features:
        raise click.UsageError("This release requires cached --train-features and --validation-features; -D records dataset provenance.")
    adapter, metrics = train_router(
        load_feature_rows(train_features), load_feature_rows(validation_features),
        routing_width=routing_dim, steps=steps, seed=seed, device=device,
        metadata={"base_model": model, "model_family": model_family, "training_datasets": list(datasets), "validation_datasets": list(validation_datasets)},
    )
    adapter.save_pretrained(output)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _emit({"output": str(output), "metrics": metrics}, json_output=json_output, yaml_output=yaml_output)


@adapter_train_cli.command("memory")
def adapter_train_memory() -> None:
    raise click.ClickException("Memory-adapter training remains research-only; no certified dataset pipeline is packaged yet.")


@adapter_train_cli.command("late-band")
def adapter_train_late_band() -> None:
    raise click.ClickException("Late-band LoRA remains research-only and is not a certified product path.")


@adapter_cli.command("eval")
@click.argument("source")
@click.option("--features", required=True, multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--query-strategy", default="last", show_default=True)
@click.option("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
@_output_options
def adapter_eval(source, features, query_strategy, device, json_output, yaml_output) -> None:
    adapter = PRARouter.from_pretrained(source, device=device)
    _emit(evaluate_router_features(adapter, load_feature_rows(features), query_strategy=query_strategy, device=device), json_output=json_output, yaml_output=yaml_output)


@cli.group("profiles")
def profiles_cli() -> None:
    """Inspect and calibrate evidence-aware profiles."""


@profiles_cli.command("show")
@click.argument("model")
@click.option("-w", "--workload")
@click.option("--registry", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_output_options
def profiles_show(model, workload, registry, json_output, yaml_output) -> None:
    values = ProfileBenchmarkRegistry.from_path(registry) if registry else ProfileBenchmarkRegistry.default()
    _emit(
        values.inspect(model, workload=workload),
        json_output=json_output or not yaml_output,
        yaml_output=yaml_output,
    )


@profiles_cli.command("calibrate")
@click.argument("model")
@click.option("-s", "--suite", default="standard", show_default=True)
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
@click.option("-d", "--device", default="auto", show_default=True)
@click.option("-e", "--engine", default="hf", show_default=True)
@click.option("-w", "--workload")
@click.option("--registry", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_output_options
def profiles_calibrate(model, suite, output, device, engine, workload, registry, json_output, yaml_output) -> None:
    value = ProfileCalibrator().calibrate(model, output=output, workload=workload, suite=suite, engine=engine, registry=registry)
    value["device_requested"] = device
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@profiles_cli.command("compare")
@click.argument("model")
@click.option("-w", "--workload")
@_output_options
def profiles_compare(model, workload, json_output, yaml_output) -> None:
    value = ProfileBenchmarkRegistry.default().inspect(model, workload=workload)
    _emit({"model": model, "profiles": value["profiles"]}, json_output=json_output, yaml_output=yaml_output)


@profiles_cli.command("report")
@click.argument("model")
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
def profiles_report(model, output) -> None:
    value = ProfileBenchmarkRegistry.default().inspect(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# PRA profile report\n\n```yaml\n" + dump_data(value, "yaml") + "\n```\n", encoding="utf-8")
    click.echo(str(output))


@cli.group("bundle")
def bundle_cli() -> None:
    """Build and inspect portable PRA bundles."""


@bundle_cli.command("build")
@click.argument("run", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
@click.option("--force", is_flag=True, help="Replace a non-empty output directory.")
@_output_options
def bundle_build(run, output, force, json_output, yaml_output) -> None:
    bundle = BundleBuilder().build(run, output, force=force)
    _emit({"output": str(output), "bundle": bundle.to_dict()}, json_output=json_output, yaml_output=yaml_output)


@bundle_cli.command("inspect")
@click.argument("source")
@click.option("-r", "--revision")
@_output_options
def bundle_inspect(source, revision, json_output, yaml_output) -> None:
    _emit(PRAModelBundle.from_pretrained(source, revision=revision).to_dict(), json_output=json_output, yaml_output=yaml_output)


@bundle_cli.command("validate")
@click.argument("source")
@_output_options
def bundle_validate(source, json_output, yaml_output) -> None:
    bundle = PRAModelBundle.from_pretrained(source)
    _emit(bundle.validate(), json_output=json_output, yaml_output=yaml_output)


@bundle_cli.command("card")
@click.argument("source")
@click.option("--update", is_flag=True, help="Write the generated card to README.md.")
@_output_options
def bundle_card(source, update, json_output, yaml_output) -> None:
    """Generate or update a rich Hugging Face model card."""

    bundle = PRAModelBundle.from_pretrained(source, validate=False)
    card = BundleBuilder.model_card(bundle)
    output = None
    if update:
        if bundle.local_path is None:
            raise click.ClickException("A resolved local bundle path is required for --update.")
        output = bundle.local_path / "README.md"
        output.write_text(card, encoding="utf-8")
        PRAModelBundle.from_pretrained(bundle.local_path).validate()
    if json_output or yaml_output:
        _emit({"source": source, "output": str(output) if output else None, "card": card}, json_output=json_output, yaml_output=yaml_output)
    elif update:
        click.echo(str(output))
    else:
        click.echo(card)


@bundle_cli.command("list")
@click.option("--model")
@click.option("--family")
@_output_options
def bundle_list(model, family, json_output, yaml_output) -> None:
    """List immutable bundles in the trusted auto-resolution registry."""

    rows = TrustedBundleRegistry.default().list(model=model, family=family)
    _emit({"bundles": rows, "count": len(rows)}, json_output=json_output, yaml_output=yaml_output)


@bundle_cli.command("resolve")
@click.argument("model")
@click.option("-e", "--engine", default="hf", show_default=True)
@click.option("-r", "--revision")
@click.option("-a", "--pra-bundle", default="auto", show_default=True)
@_output_options
def bundle_resolve(model, engine, revision, pra_bundle, json_output, yaml_output) -> None:
    """Explain bundle selection and pin the resolved Hub revision."""

    value = BundleResolver().resolve(
        pra_bundle, model=model, model_revision=revision, engine=engine
    )
    _emit(value.to_dict(), json_output=json_output, yaml_output=yaml_output)


@cli.group("hf")
def hf_cli() -> None:
    """Discover, authenticate, pull, and publish Hugging Face artifacts."""


@hf_cli.command("login")
@click.option("--check", is_flag=True, help="Check existing Hub authentication without prompting.")
@_output_options
def hf_login(check, json_output, yaml_output) -> None:
    try:
        from huggingface_hub import HfApi, login
    except ImportError as error:
        raise click.ClickException("Install the hf-hub optional dependency.") from error
    if check:
        try:
            identity = HfApi().whoami()
        except Exception as error:
            raise click.ClickException(
                "No usable Hugging Face authentication was found. Run `pra hf login`."
            ) from error
        _emit(
            {
                "status": "AUTHENTICATED",
                "name": identity.get("name"),
                "organizations": [
                    item.get("name") for item in identity.get("orgs", ())
                    if item.get("name")
                ],
            },
            json_output=json_output,
            yaml_output=yaml_output,
        )
        return
    login()


@hf_cli.command("list")
@click.option("--query", help="Filter trusted metadata by a case-insensitive substring.")
@click.option("--model", help="Require an exact base-model identifier.")
@click.option("--family", help="Filter by model family or architecture.")
@click.option("-e", "--engine", help="Require compatibility with this engine.")
@click.option("--qualification", help="Require an exact qualification tier.")
@_output_options
def hf_list(query, model, family, engine, qualification, json_output, yaml_output) -> None:
    """List pinned PRA bundles trusted for automatic resolution."""

    rows = TrustedBundleRegistry.default().list(
        query=query,
        model=model,
        family=family,
        engine=engine,
        qualification=qualification,
    )
    _emit_bundle_catalog(
        {"source": "trusted-registry", "bundles": rows, "count": len(rows)},
        json_output=json_output,
        yaml_output=yaml_output,
    )


@hf_cli.command("search")
@click.argument("query", required=False, default="pra")
@click.option("--author", default="EInnovator", show_default=True, help="Limit results to one Hub namespace.")
@click.option("--all-authors", is_flag=True, help="Search all Hub namespaces; results remain untrusted unless registered.")
@click.option("--limit", type=click.IntRange(min=1, max=100), default=20, show_default=True)
@_output_options
def hf_search(query, author, all_authors, limit, json_output, yaml_output) -> None:
    """Search live Hugging Face metadata for PRA model bundles."""

    try:
        rows = HubBundleCatalog().search(
            query,
            author=None if all_authors else author,
            limit=limit,
        )
    except ImportError as error:
        raise click.ClickException(str(error)) from error
    except Exception as error:
        raise click.ClickException(f"Hugging Face bundle search failed: {error}") from error
    _emit_bundle_catalog(
        {
            "source": "hugging-face-hub",
            "query": query,
            "author": None if all_authors else author,
            "bundles": rows,
            "count": len(rows),
        },
        json_output=json_output,
        yaml_output=yaml_output,
    )


@hf_cli.command("pull")
@click.argument("repo_id")
@click.option("-o", "--output", type=click.Path(path_type=Path))
@click.option("-r", "--revision")
@_output_options
def hf_pull(repo_id, output, revision, json_output, yaml_output) -> None:
    """Pull and validate a bundle, using the normal HF cache by default."""

    _emit(HubPublisher().pull(repo_id, output, revision=revision), json_output=json_output, yaml_output=yaml_output)


@hf_cli.command("push")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("repo_id")
@click.option("-r", "--revision")
@click.option("-y", "--yes", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--private/--public", default=False, help="Set repository visibility when created.")
@click.option("--collection", help="Collection slug, or namespace/name to create.")
@click.option("--license", "license_name")
@click.option("--commit-message", default="Publish PRA model bundle", show_default=True)
@click.option("--tag", help="Create an immutable release tag after upload.")
@_output_options
def hf_push(bundle, repo_id, revision, yes, dry_run, private, collection, license_name, commit_message, tag, json_output, yaml_output) -> None:
    if not dry_run and not yes and not click.confirm(f"Publish PRA bundle to {repo_id}?", default=False):
        raise click.Abort()
    value = HubPublisher().push(
        bundle,
        repo_id,
        revision=revision,
        dry_run=dry_run,
        private=private,
        collection=collection,
        license_name=license_name,
        commit_message=commit_message,
        tag=tag,
    )
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@hf_cli.command("publish-manifest")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True)
@click.option("-y", "--yes", is_flag=True)
@_output_options
def hf_publish_manifest(manifest, dry_run, yes, json_output, yaml_output) -> None:
    """Validate or publish a resumable declarative bundle release list."""

    if not dry_run and not yes and not click.confirm("Publish every valid bundle in this manifest?", default=False):
        raise click.Abort()
    value = HubPublisher().publish_manifest(manifest, dry_run=dry_run)
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@hf_cli.command("inspect")
@click.argument("source")
@_output_options
def hf_inspect(source, json_output, yaml_output) -> None:
    _emit(PRAModelBundle.from_pretrained(source).to_dict(), json_output=json_output, yaml_output=yaml_output)


@cli.group("runtime")
def runtime_cli() -> None:
    """Launch engines through the RuntimeProvider contract."""


@runtime_cli.command("init")
@click.argument("directory", type=click.Path(path_type=Path))
@click.option("--max-native-index-tokens", type=click.IntRange(min=1))
@click.option("--max-native-index-bytes", type=click.IntRange(min=1))
@click.option("--defer-native-index", is_flag=True)
@click.option("--storage", type=click.Choice(["memory", "balanced", "persistent", "minimal"]), default="balanced", show_default=True)
def runtime_init(directory, max_native_index_tokens, max_native_index_bytes, defer_native_index, storage) -> None:
    """Create a portable PRA runtime configuration directory."""

    from .adaptive_context_runtime import ContextPolicy
    from .storage_lifecycle import PRAStoragePolicy

    config = PRARuntimeConfig(
        context_policy=ContextPolicy(
            max_native_index_tokens=max_native_index_tokens,
            max_native_index_bytes=max_native_index_bytes,
            defer_native_index=defer_native_index,
        ),
        storage=PRAStoragePolicy.named(storage),
    )
    click.echo(str(config.save_pretrained(directory)))


def _runtime_options(function):
    decorators = (
        click.option("-e", "--engine", default="hf", show_default=True), click.option("-r", "--revision"),
        click.option("-d", "--device", default="auto", show_default=True), click.option("-u", "--endpoint"),
        click.option("--host", default="127.0.0.1", show_default=True), click.option("--port", type=int, default=8000, show_default=True),
        click.option("-a", "--pra-bundle"), click.option("-p", "--profile"),
        click.option("--storage", type=click.Choice(["memory", "balanced", "persistent", "minimal"]), default="balanced", show_default=True),
        click.option("--storage-config", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option("--engine-arg", "engine_args", multiple=True), click.option("-v", "--verbose", is_flag=True),
        click.option("--observability", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option("--otel", is_flag=True), click.option("--otel-endpoint"),
        click.option("--prometheus", is_flag=True), click.option("--prometheus-port", type=click.IntRange(min=1, max=65535)),
        click.option("--management-api", is_flag=True, help="Enable the separate PRA management listener."),
        click.option("--management-host", default="127.0.0.1", show_default=True),
        click.option("--management-port", type=click.IntRange(min=1, max=65535), default=9101, show_default=True),
        click.option("--management-auth-mode", type=click.Choice(["none", "static_bearer", "jwt_oidc", "mtls"]), default="none", show_default=True),
        click.option("--management-token-env", default="PRA_MANAGEMENT_TOKEN", show_default=True),
        click.option("--management-metrics-url"), click.option("--management-trace-url"),
        click.option("--management-grafana-url"),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


@runtime_cli.command("serve")
@click.argument("model")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["auto", "selected-context", "native-memory", "native-serving"]),
    default="auto",
    show_default=True,
    help="Choose the qualified product execution mode.",
)
@click.option("--explain", is_flag=True, help="Explain mode evidence and resolution.")
@click.option("--allow-unqualified-native", is_flag=True, hidden=True)
@_runtime_options
@_output_options
def runtime_serve(model, mode, explain, allow_unqualified_native, engine, revision, device, endpoint, host, port, pra_bundle, profile, storage, storage_config, engine_args, verbose, observability, otel, otel_endpoint, prometheus, prometheus_port, management_api, management_host, management_port, management_auth_mode, management_token_env, management_metrics_url, management_trace_url, management_grafana_url, json_output, yaml_output) -> None:
    """Serve MODEL with an explicit or policy-selected execution mode."""
    try:
        engine_row = EngineProductRegistry.default().resolve(engine)
    except KeyError as error:
        raise click.ClickException(str(error)) from error
    try:
        resolution = ExecutionModeResolver().resolve(
            mode,
            engine_row,
            allow_unqualified_native=allow_unqualified_native,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    selected_mode = resolution.resolved_mode.value
    resolved_profile = "BALANCED" if profile == "recommended" else profile
    mode_args = (*engine_args, f"execution-mode={selected_mode}")
    config = _engine_config(model, engine, revision, device, endpoint, host, port, pra_bundle, resolved_profile, storage, storage_config, mode_args, verbose, observability, otel, otel_endpoint, prometheus, prometheus_port, management_api, management_host, management_port, management_auth_mode, management_token_env, management_metrics_url, management_trace_url, management_grafana_url)
    try:
        manager = RuntimeManager()
        handle = manager.serve(config)
        value = handle.to_dict()
        value["status"] = manager.health(handle).status
    except (KeyError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    value.update(
        {
            "requested_mode": mode,
            "resolved_mode": selected_mode,
            "resolution_reason": resolution.reason,
            "mode_resolution": resolution.to_dict(),
        }
    )
    if explain and not json_output and not yaml_output:
        click.echo(resolution.explain())
        click.echo("")
    _emit(value, json_output=json_output, yaml_output=yaml_output)


# The product journey uses ``pra serve``; the grouped spelling remains stable.
cli.add_command(runtime_serve, "serve")


@runtime_cli.command("inspect")
@click.argument("model", required=False)
@_runtime_options
@_output_options
def runtime_inspect(model, engine, revision, device, endpoint, host, port, pra_bundle, profile, storage, storage_config, engine_args, verbose, observability, otel, otel_endpoint, prometheus, prometheus_port, management_api, management_host, management_port, management_auth_mode, management_token_env, management_metrics_url, management_trace_url, management_grafana_url, json_output, yaml_output) -> None:
    config = _engine_config(model, engine, revision, device, endpoint, host, port, pra_bundle, profile, storage, storage_config, engine_args, verbose, observability, otel, otel_endpoint, prometheus, prometheus_port, management_api, management_host, management_port, management_auth_mode, management_token_env, management_metrics_url, management_trace_url, management_grafana_url)
    try:
        value = RuntimeManager().inspect(config)
    except KeyError as error:
        raise click.ClickException(str(error)) from error
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@runtime_cli.command("doctor")
@click.option("-e", "--engine", default="hf", show_default=True)
@click.option("-u", "--endpoint")
@_output_options
def runtime_doctor(engine, endpoint, json_output, yaml_output) -> None:
    try:
        value = RuntimeManager().doctor(RuntimeConfig(engine=engine, endpoint=endpoint))
    except KeyError as error:
        raise click.ClickException(str(error)) from error
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@runtime_cli.command("benchmark")
@click.argument("model")
@click.option("-e", "--engine", default="hf", show_default=True)
@click.option("-p", "--profile")
@click.option("-d", "--device", default="auto", show_default=True)
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
@click.option("--storage", type=click.Choice(["memory", "balanced", "persistent", "minimal"]), default="balanced", show_default=True)
@click.option("--storage-config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_output_options
def runtime_benchmark(model, engine, profile, device, output, storage, storage_config, json_output, yaml_output) -> None:
    value = RuntimeManager().benchmark(RuntimeConfig(engine=engine, model=model, profile=profile, device=device, storage_profile=storage, storage_config=str(storage_config) if storage_config else None), output)
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@runtime_cli.command("capabilities")
@_output_options
def runtime_capability_report(json_output, yaml_output) -> None:
    _emit(runtime_capabilities(), json_output=json_output, yaml_output=yaml_output)


@runtime_cli.command("prepare-vllm", hidden=True)
@click.argument("prompt")
@click.option("--selected-uri", "selected_uris", multiple=True)
@click.option("--materialized-tokens", type=int, default=0, show_default=True)
@_output_options
def runtime_prepare_vllm(prompt, selected_uris, materialized_tokens, json_output, yaml_output) -> None:
    value = VLLMThinBackend().prepare(prompt, selected_uris=selected_uris, materialized_tokens=materialized_tokens)
    _emit(value.__dict__, json_output=json_output, yaml_output=yaml_output)


@cli.group("agent")
def agent_cli() -> None:
    """Chat, run tasks, and launch the optional web UI."""


@agent_cli.group("mcp")
def agent_mcp_cli() -> None:
    """Configure and inspect PRA Agent MCP clients."""


def _agent_settings(path: Path | None) -> PRAAgentSettings:
    return PRAAgentSettings.from_file(path) if path and path.exists() else PRAAgentSettings()


@agent_mcp_cli.command("list")
@click.option("--config", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--connect", is_flag=True, help="Connect and include live status.")
@_output_options
def agent_mcp_list(config, connect, json_output, yaml_output) -> None:
    """List configured MCP servers without exposing credentials."""
    settings = _agent_settings(config)
    if connect:
        async def inspect():
            manager = MCPClientManager(settings.mcp)
            try:
                await manager.connect_all()
                return [vars(row) | {"state": row.state.value} for row in await manager.list_servers()]
            finally:
                await manager.disconnect_all()
        rows = asyncio.run(inspect())
    else:
        rows = [{"name": name, **server.model_dump(mode="json", exclude={"auth"}),
                 "auth": server.auth.model_dump(mode="json", exclude_none=True)}
                for name, server in settings.mcp.servers.items()]
    _emit({"servers": rows}, json_output=json_output, yaml_output=yaml_output)


@agent_mcp_cli.command("add")
@click.argument("name")
@click.argument("url")
@click.option("--config", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--save", is_flag=True, help="Persist explicitly; otherwise only validate.")
@click.option("--required", is_flag=True)
def agent_mcp_add(name, url, config, save, required) -> None:
    """Add an HTTP MCP server to Agent configuration."""
    settings = _agent_settings(config)
    settings.mcp.servers[name] = MCPServerConfig(url=url, required=required)
    if save:
        settings.save(config)
        click.echo(f"Saved MCP server {name} to {config}.")
    else:
        click.echo(f"Validated MCP server {name}; pass --save to persist.")


@agent_mcp_cli.command("remove")
@click.argument("name")
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--save", is_flag=True, help="Persist explicitly; otherwise only validate.")
def agent_mcp_remove(name, config, save) -> None:
    """Remove a configured MCP server."""
    settings = _agent_settings(config)
    if name not in settings.mcp.servers:
        raise click.ClickException(f"Unknown MCP server: {name}")
    del settings.mcp.servers[name]
    if save:
        settings.save(config)
        click.echo(f"Removed MCP server {name} from {config}.")
    else:
        click.echo(f"Validated removal of {name}; pass --save to persist.")


def _resolve_agent_profile(
    profile, config, model, pra, engine, endpoint, workspace, skills,
    context_transport=None, allow_text_fallback=None,
):
    if config:
        try:
            application = PRAAgentSettings.from_file(config)
        except Exception:
            application = None
        if application is not None:
            selected_provider = application.agent.provider or next(iter(application.providers), None)
            provider = application.providers.get(selected_provider) if selected_provider else None
            app_overrides = {"agent": {}}
            if model:
                app_overrides["agent"]["model"] = model
            if endpoint and selected_provider:
                app_overrides["providers"] = {selected_provider: {"base_url": endpoint}}
            source_file = application.source_file
            application = PRAAgentSettings.merge(application, app_overrides)
            application.source_file = source_file
            provider = application.providers.get(selected_provider) if selected_provider else provider
            values = {
                "model": {"id": application.agent.model or (provider.model if provider else None)},
                "runtime": {
                    "mode": "gateway" if provider and provider.base_url else "embedded",
                    "engine": engine or (provider.type if provider else "hf"),
                    "endpoint": endpoint or (provider.base_url if provider else None),
                },
                "workspace": str(workspace or "."),
                "sessions": {"path": application.session.path, "resume_last": application.session.resume_last},
                "tools": {"allow_writes": application.agent.allow_writes,
                          "allow_destructive": application.agent.allow_destructive,
                          "candidates": application.agent.tool_candidates,
                          "max_rounds": application.agent.max_tool_rounds},
                "skills": {"directories": [str(path) for path in skills]},
                "context": {"records": application.agent.context_records,
                            "transport": context_transport or "auto",
                            "allow_text_fallback": True if allow_text_fallback is None else allow_text_fallback},
                "generation": {"max_new_tokens": application.agent.max_new_tokens},
                "pra": {"profile": pra} if pra else {},
            }
            selected = replace(AgentProfile.from_dict("config", values), application_settings=application)
            return selected, (f"application:{config}",)
    overrides = {}
    if model:
        overrides["model"] = model
    if pra:
        overrides["pra"] = {"profile": pra}
    runtime = {}
    if engine:
        runtime["engine"] = engine
    if endpoint:
        runtime.update({"endpoint": endpoint, "mode": "gateway"})
    if runtime:
        overrides["runtime"] = runtime
    if workspace:
        overrides["workspace"] = str(workspace)
    if skills:
        overrides["skills"] = {"directories": [str(path) for path in skills]}
    if context_transport is not None or allow_text_fallback is not None:
        context = {}
        if context_transport is not None:
            context["transport"] = context_transport
        if allow_text_fallback is not None:
            context["allow_text_fallback"] = allow_text_fallback
        overrides["context"] = context
    return AgentProfileRegistry().resolve(profile_name=profile, config_path=config, overrides=overrides)


def _agent_options(function):
    decorators = (
        click.option("-p", "--profile", help="Named agent profile."), click.option("-P", "--pra", help="PRA profile/bundle/config override."),
        click.option("-m", "--model"), click.option("-e", "--engine"), click.option("-u", "--endpoint"),
        click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option("-w", "--workspace", type=click.Path(path_type=Path)),
        click.option("-s", "--skills", multiple=True, type=click.Path(exists=True, file_okay=False, path_type=Path)),
        click.option("--context-transport", type=click.Choice(["auto", "pra", "text"])),
        click.option("--allow-text-fallback/--no-text-fallback", default=None),
        click.option("--session"), click.option("-r", "--resume", is_flag=True), click.option("-t", "--task"),
        click.option("-v", "--verbose", is_flag=True),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


@agent_cli.command("chat")
@click.argument("legacy_model", required=False)
@_agent_options
@_observability_options
def agent_chat(legacy_model, profile, pra, model, engine, endpoint, config, workspace, skills, context_transport, allow_text_fallback, session, resume, task, verbose, observability, otel, otel_endpoint, prometheus, prometheus_port) -> None:
    """Open the persistent TUI; no flags uses the default profile."""
    selected, trace = _resolve_agent_profile(profile, config, model or legacy_model, pra, engine, endpoint, workspace, skills, context_transport, allow_text_fallback)
    telemetry = _telemetry(observability, otel, otel_endpoint, prometheus, prometheus_port, service="agent")
    launch = AgentLauncher().launch(selected, observability=telemetry)
    if verbose:
        _emit({"resolution": trace, "profile": selected.redacted_dict(), "mcp": load_mcp_config(selected)})
    _emit(launch.summary)
    try:
        launch.agent.start_session(session, resume=resume or selected.resume_last, task_description=task)
        asyncio.run(launch.agent.start())
        AgentShell(launch.agent).run()
    finally:
        asyncio.run(launch.agent.aclose())
        telemetry.close()


@agent_cli.command("run")
@click.argument("prompt", required=False)
@_agent_options
@click.option("--json", "json_output", is_flag=True)
@_observability_options
def agent_run(prompt, profile, pra, model, engine, endpoint, config, workspace, skills, context_transport, allow_text_fallback, session, resume, task, verbose, json_output, observability, otel, otel_endpoint, prometheus, prometheus_port) -> None:
    """Run one noninteractive turn from an argument or stdin."""
    query = prompt if prompt is not None else sys.stdin.read()
    if not query.strip():
        raise click.UsageError("Provide PROMPT or pipe input on stdin.")
    selected, trace = _resolve_agent_profile(profile, config, model, pra, engine, endpoint, workspace, skills, context_transport, allow_text_fallback)
    telemetry = _telemetry(observability, otel, otel_endpoint, prometheus, prometheus_port, service="agent")
    launch = AgentLauncher().launch(selected, observability=telemetry)
    try:
        launch.agent.start_session(session, resume=resume or selected.resume_last, task_description=task)
        asyncio.run(launch.agent.start())
        turn = launch.agent.run_turn(query)
        if json_output:
            _emit({"response": turn.text, "session_id": turn.session.session_id, "tool_calls": len(turn.tool_executions), "selected_records": list(turn.selected_record_ids), "task_state": turn.session.tasks.to_dict(), "metrics": launch.agent.runtime.inspect(), "resolution": trace if verbose else None}, json_output=True)
        else:
            click.echo(turn.text)
    finally:
        asyncio.run(launch.agent.aclose())
        telemetry.close()


@agent_cli.command("inspect")
@click.option("-p", "--profile")
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_output_options
def agent_inspect(profile, config, json_output, yaml_output) -> None:
    selected, sources = _resolve_agent_profile(
        profile, config, None, None, None, None, None, (), None, None
    )
    mcp = (
        selected.application_settings.redacted().get("mcp", {})
        if selected.application_settings else load_mcp_config(selected)
    )
    _emit({"sources": sources, "profile": selected.redacted_dict(), "mcp": mcp}, json_output=json_output, yaml_output=yaml_output)


@agent_cli.command("start")
@click.option("-p", "--profile")
@click.option("-P", "--pra")
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-h", "--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("-d", "--detach", is_flag=True)
@click.option("-o", "--open", "open_browser", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@_output_options
def agent_start(profile, pra, config, host, port, detach, open_browser, verbose, json_output, yaml_output) -> None:
    """Start the experimental optional FastAPI agent UI."""
    try:
        from .agent_web import AgentWebLifecycle
    except ImportError as error:
        raise click.ClickException("Install the 'web' optional dependency.") from error
    state = AgentWebLifecycle().start(
        host=host,
        port=port,
        profile=profile,
        pra_override=pra,
        config_path=str(config) if config else None,
        detach=detach,
        open_browser=open_browser,
    )
    if state:
        _emit(state.to_dict(), json_output=json_output, yaml_output=yaml_output)


@agent_cli.command("stop")
def agent_stop() -> None:
    """Safely stop a detached PRA Agent Web UI."""
    try:
        from .agent_web import AgentWebLifecycle
    except ImportError as error:
        raise click.ClickException("Install the 'web' optional dependency.") from error
    click.echo(AgentWebLifecycle().stop())


@cli.command("ask", hidden=True)
@click.argument("model")
@click.argument("question")
@click.option("--routing-adapter")
@click.option("--reference", "references", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-new-tokens", default=64, show_default=True)
def legacy_ask(model, question, routing_adapter, references, max_new_tokens) -> None:
    pra = PRAForCausalLM.from_pretrained(model, routing_adapter=routing_adapter)
    for path in references:
        pra.add_reference_file(path)
    click.echo(pra.generate(question, max_new_tokens=max_new_tokens, return_details=True).text)


@click.group(name="pra-hf", context_settings={"help_option_names": ["-h", "--help"]})
def deprecated_cli() -> None:
    """Deprecated compatibility command; use ``pra``."""
    click.echo("`pra-hf` is deprecated; use `pra`.", err=True)


for _name, _command in cli.commands.items():
    deprecated_cli.add_command(_command, _name)


if __name__ == "__main__":
    cli()
