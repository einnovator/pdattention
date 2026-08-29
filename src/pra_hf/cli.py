"""Canonical product command line for PRA model, runtime, and agent workflows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import click
import torch
import yaml

from .agent_profiles import AgentLauncher, AgentProfileRegistry, load_mcp_config
from .bundle import BundleBuilder, HubPublisher, PRAModelBundle
from .evaluation import evaluate_router_features
from .gateway_cli import gateway_cli
from .model import PRAForCausalLM
from .onboarding import DoctorService, ModelInspector, ModelValidator, OnboardingPipeline, ProfileCalibrator, StructuralAdapterBuilder
from .product_config import dump_data
from .profile_benchmarks import ProfileBenchmarkRegistry
from .router import PRARouter
from .runtime import VLLMThinBackend, runtime_capabilities
from .runtime_providers import RuntimeConfig, RuntimeManager, parse_engine_arguments
from .training import load_feature_rows, train_router
from .tui import AgentShell


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


def _engine_config(model, engine, revision, device, endpoint, host, port, pra_bundle, profile, engine_args=(), verbose=False):
    try:
        options = parse_engine_arguments(engine_args)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    return RuntimeConfig(
        engine=engine, model=model, revision=revision, device=device, endpoint=endpoint,
        host=host, port=port, pra_bundle=pra_bundle, profile=profile,
        engine_options=options, verbose=verbose,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="pra-hf")
def cli() -> None:
    """Inspect, adapt, calibrate, run, and publish Progressive Retrieval Attention."""


cli.add_command(gateway_cli)


@cli.command("doctor")
@click.option("-v", "--verbose", is_flag=True)
@_output_options
def doctor(verbose, json_output, yaml_output) -> None:
    """Report core, accelerator, Hub, and optional runtime availability."""
    _emit(DoctorService().run(verbose=verbose), json_output=json_output, yaml_output=yaml_output)


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
    _emit(values.inspect(model, workload=workload), json_output=json_output, yaml_output=yaml_output)


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
@_output_options
def bundle_build(run, output, json_output, yaml_output) -> None:
    bundle = BundleBuilder().build(run, output)
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
    _emit({"status": "VALID", "model": bundle.base_model, "schema_version": bundle.schema_version}, json_output=json_output, yaml_output=yaml_output)


@cli.group("hf")
def hf_cli() -> None:
    """Authenticate, pull, and publish artifacts on Hugging Face Hub."""


@hf_cli.command("login")
def hf_login() -> None:
    try:
        from huggingface_hub import login
    except ImportError as error:
        raise click.ClickException("Install the hf-hub optional dependency.") from error
    login()


@hf_cli.command("pull")
@click.argument("repo_id")
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
@click.option("-r", "--revision")
def hf_pull(repo_id, output, revision) -> None:
    click.echo(str(HubPublisher().pull(repo_id, output, revision=revision)))


@hf_cli.command("push")
@click.argument("bundle", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("repo_id")
@click.option("-r", "--revision")
@click.option("-y", "--yes", is_flag=True)
@click.option("--dry-run", is_flag=True)
@_output_options
def hf_push(bundle, repo_id, revision, yes, dry_run, json_output, yaml_output) -> None:
    if not dry_run and not yes and not click.confirm(f"Publish PRA bundle to {repo_id}?", default=False):
        raise click.Abort()
    _emit(HubPublisher().push(bundle, repo_id, revision=revision, dry_run=dry_run), json_output=json_output, yaml_output=yaml_output)


@hf_cli.command("inspect")
@click.argument("source")
@_output_options
def hf_inspect(source, json_output, yaml_output) -> None:
    _emit(PRAModelBundle.from_pretrained(source).to_dict(), json_output=json_output, yaml_output=yaml_output)


@cli.group("runtime")
def runtime_cli() -> None:
    """Launch engines through the RuntimeProvider contract."""


def _runtime_options(function):
    decorators = (
        click.option("-e", "--engine", default="hf", show_default=True), click.option("-r", "--revision"),
        click.option("-d", "--device", default="auto", show_default=True), click.option("-u", "--endpoint"),
        click.option("--host", default="127.0.0.1", show_default=True), click.option("--port", type=int, default=8000, show_default=True),
        click.option("-a", "--pra-bundle"), click.option("-p", "--profile"),
        click.option("--engine-arg", "engine_args", multiple=True), click.option("-v", "--verbose", is_flag=True),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


@runtime_cli.command("serve")
@click.argument("model")
@_runtime_options
@_output_options
def runtime_serve(model, engine, revision, device, endpoint, host, port, pra_bundle, profile, engine_args, verbose, json_output, yaml_output) -> None:
    config = _engine_config(model, engine, revision, device, endpoint, host, port, pra_bundle, profile, engine_args, verbose)
    try:
        manager = RuntimeManager()
        handle = manager.serve(config)
        value = handle.to_dict()
        value["status"] = manager.health(handle).status
    except (KeyError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    _emit(value, json_output=json_output, yaml_output=yaml_output)


@runtime_cli.command("inspect")
@click.argument("model", required=False)
@_runtime_options
@_output_options
def runtime_inspect(model, engine, revision, device, endpoint, host, port, pra_bundle, profile, engine_args, verbose, json_output, yaml_output) -> None:
    config = _engine_config(model, engine, revision, device, endpoint, host, port, pra_bundle, profile, engine_args, verbose)
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
@_output_options
def runtime_benchmark(model, engine, profile, device, output, json_output, yaml_output) -> None:
    value = RuntimeManager().benchmark(RuntimeConfig(engine=engine, model=model, profile=profile, device=device), output)
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


def _resolve_agent_profile(profile, config, model, pra, engine, endpoint, workspace, skills):
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
    return AgentProfileRegistry().resolve(profile_name=profile, config_path=config, overrides=overrides)


def _agent_options(function):
    decorators = (
        click.option("-p", "--profile", help="Named agent profile."), click.option("-P", "--pra", help="PRA profile/bundle/config override."),
        click.option("-m", "--model"), click.option("-e", "--engine"), click.option("-u", "--endpoint"),
        click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option("-w", "--workspace", type=click.Path(path_type=Path)),
        click.option("-s", "--skills", multiple=True, type=click.Path(exists=True, file_okay=False, path_type=Path)),
        click.option("--session"), click.option("-r", "--resume", is_flag=True), click.option("-t", "--task"),
        click.option("-v", "--verbose", is_flag=True),
    )
    for decorator in reversed(decorators):
        function = decorator(function)
    return function


@agent_cli.command("chat")
@click.argument("legacy_model", required=False)
@_agent_options
def agent_chat(legacy_model, profile, pra, model, engine, endpoint, config, workspace, skills, session, resume, task, verbose) -> None:
    """Open the persistent TUI; no flags uses the default profile."""
    selected, trace = _resolve_agent_profile(profile, config, model or legacy_model, pra, engine, endpoint, workspace, skills)
    launch = AgentLauncher().launch(selected)
    if verbose:
        _emit({"resolution": trace, "profile": selected.redacted_dict(), "mcp": load_mcp_config(selected)})
    _emit(launch.summary)
    try:
        launch.agent.start_session(session, resume=resume or selected.resume_last, task_description=task)
        AgentShell(launch.agent).run()
    finally:
        launch.agent.close()


@agent_cli.command("run")
@click.argument("prompt", required=False)
@_agent_options
@click.option("--json", "json_output", is_flag=True)
def agent_run(prompt, profile, pra, model, engine, endpoint, config, workspace, skills, session, resume, task, verbose, json_output) -> None:
    """Run one noninteractive turn from an argument or stdin."""
    query = prompt if prompt is not None else sys.stdin.read()
    if not query.strip():
        raise click.UsageError("Provide PROMPT or pipe input on stdin.")
    selected, trace = _resolve_agent_profile(profile, config, model, pra, engine, endpoint, workspace, skills)
    launch = AgentLauncher().launch(selected)
    try:
        launch.agent.start_session(session, resume=resume or selected.resume_last, task_description=task)
        turn = launch.agent.run_turn(query)
        if json_output:
            _emit({"response": turn.text, "session_id": turn.session.session_id, "tool_calls": len(turn.tool_executions), "selected_records": list(turn.selected_record_ids), "task_state": turn.session.tasks.to_dict(), "metrics": launch.agent.runtime.inspect(), "resolution": trace if verbose else None}, json_output=True)
        else:
            click.echo(turn.text)
    finally:
        launch.agent.close()


@agent_cli.command("inspect")
@click.option("-p", "--profile")
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_output_options
def agent_inspect(profile, config, json_output, yaml_output) -> None:
    selected, sources = AgentProfileRegistry().resolve(profile_name=profile, config_path=config)
    _emit({"sources": sources, "profile": selected.redacted_dict(), "mcp": load_mcp_config(selected)}, json_output=json_output, yaml_output=yaml_output)


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


# Compatibility aliases retained for one release cycle.
@cli.group("router", hidden=True)
def router_cli() -> None:
    """Deprecated alias for ``pra adapter``."""


router_cli.add_command(adapter_inspect, "inspect")
router_cli.add_command(adapter_train_routing, "train")
router_cli.add_command(adapter_eval, "eval")


@cli.command("inspect", hidden=True)
@click.argument("model")
def legacy_inspect(model) -> None:
    _emit(ModelInspector().inspect(model))


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
