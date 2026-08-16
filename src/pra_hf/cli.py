"""Small command-line interface for the product-facing PRA-HF workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import click
import torch

from .evaluation import evaluate_router_features
from .adaptive_runtime import (
    AdaptiveRetryAgent,
    ControllerFeatures,
    HandRuleController,
    LinearEffortController,
    StopPolicy,
    default_effort_profiles,
    save_effort_profiles,
    validate_effort_ladder,
)
from .model import PRAForCausalLM
from .router import PRARouter
from .training import load_feature_rows, train_router


def _echo_json(value) -> None:
    click.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@click.group()
def cli() -> None:
    """Attach sparse, URI-addressed native-K/V memory to supported HF models."""


@cli.command()
@click.argument("model")
def inspect(model: str) -> None:
    """Inspect MODEL compatibility without loading its weights."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    family = str(getattr(config, "model_type", "unknown"))
    _echo_json(
        {
            "model": model,
            "family": family,
            "supported": family in {"qwen2", "qwen3", "llama", "gemma3_text"},
            "layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "attention_heads": getattr(config, "num_attention_heads", None),
            "kv_heads": getattr(config, "num_key_value_heads", None),
            "max_positions": getattr(config, "max_position_embeddings", None),
            "attention_pattern": getattr(config, "layer_types", None),
        }
    )


@cli.group()
def router() -> None:
    """Train, evaluate, and inspect compact routing adapters."""


@router.command("inspect")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def router_inspect(directory: Path) -> None:
    """Display router architecture, provenance, and reported metrics."""
    _echo_json(PRARouter.from_pretrained(directory).artifact_config())


@router.command("train")
@click.option("--train-features", required=True, multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--validation-features", required=True, multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--base-model", required=True)
@click.option("--model-family", required=True, type=click.Choice(["qwen", "llama", "gemma3"]))
@click.option("--dataset", "datasets", multiple=True, default=("QASPER",), show_default=True)
@click.option("--routing-dim", default=128, show_default=True)
@click.option("--steps", default=512, show_default=True)
@click.option("--seed", default=53, show_default=True)
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
def router_train(train_features, validation_features, output, base_model, model_family, datasets, routing_dim, steps, seed, device) -> None:
    """Train an asymmetric linear router from frozen feature files."""
    train_rows = load_feature_rows(train_features)
    validation_rows = load_feature_rows(validation_features)
    adapter, metrics = train_router(
        train_rows,
        validation_rows,
        routing_width=routing_dim,
        steps=steps,
        seed=seed,
        device=device,
        metadata={
            "base_model": base_model,
            "model_family": model_family,
            "routing_representation": "attention_input_hidden_state",
            "training_datasets": list(datasets),
        },
    )
    adapter.save_pretrained(output)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _echo_json({"output": str(output), **metrics})


@router.command("eval")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--features", required=True, multiple=True, type=click.Path(exists=True, path_type=Path))
@click.option("--query-strategy", default="last", show_default=True)
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
def router_eval(directory, features, query_strategy, device) -> None:
    """Emit the standard recall-sparsity report for FEATURE files."""
    adapter = PRARouter.from_pretrained(directory, device=device)
    _echo_json(
        evaluate_router_features(
            adapter,
            load_feature_rows(features),
            query_strategy=query_strategy,
            device=device,
        )
    )


@cli.group()
def adaptive() -> None:
    """Inspect and plan manual or automatic adaptive-PRA effort."""


@adaptive.command("profiles")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
def adaptive_profiles(output: Path | None) -> None:
    """Print the validated default E0/E1/E2 control vectors."""

    profiles = default_effort_profiles()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        save_effort_profiles(output, profiles)
    _echo_json({"profiles": [profile.to_dict() for profile in profiles], "output": output})


@adaptive.command("plan")
@click.option("--mode", type=click.Choice(["manual", "auto"]), default="auto", show_default=True)
@click.option("--effort", type=click.Choice(["low", "medium", "high"]), default="low", show_default=True)
@click.option("--features", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--controller", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-retries", type=click.IntRange(0), default=2, show_default=True)
@click.option("--max-search-budget", type=click.IntRange(1))
@click.option("--max-active-kv", type=click.IntRange(1))
@click.option("--latency-budget", type=click.FloatRange(min=0.0, min_open=True))
@click.option("--confidence-threshold", type=click.FloatRange(0.0, 1.0), default=0.35, show_default=True)
@click.option("--facets", type=click.IntRange(1))
@click.option("--roots", type=click.IntRange(1))
@click.option("--neighbors", type=click.IntRange(1))
@click.option("--hops", type=click.IntRange(0))
@click.option("--conceptual-budget", type=click.IntRange(1))
@click.option("--routing-threshold", type=click.FloatRange(0.0, 1.0))
@click.option("--search-layer", "search_layers", multiple=True, type=int)
@click.option("--consumer-layer", "consumer_layers", multiple=True, type=int)
@click.option("--granularity", type=click.IntRange(1))
@click.option("--materialization-policy")
@click.option("--trace-output", type=click.Path(dir_okay=False, path_type=Path))
def adaptive_plan(
    mode,
    effort,
    features,
    controller,
    max_retries,
    max_search_budget,
    max_active_kv,
    latency_budget,
    confidence_threshold,
    facets,
    roots,
    neighbors,
    hops,
    conceptual_budget,
    routing_threshold,
    search_layers,
    consumer_layers,
    granularity,
    materialization_policy,
    trace_output,
) -> None:
    """Resolve budgets and observable features into an executable effort plan."""

    profiles = list(default_effort_profiles())
    aliases = {"low": 0, "medium": 1, "high": 2}
    runtime_features = ControllerFeatures.from_runtime_mapping(
        json.loads(features.read_text(encoding="utf-8")) if features else {}
    )
    learned = None
    if controller:
        payload = json.loads(controller.read_text(encoding="utf-8"))
        learned = LinearEffortController.from_dict(payload.get("controller", payload))
    hand = HandRuleController(0.45, 0.72, 0.12, 0.04)
    selected = (
        aliases[effort]
        if mode == "manual"
        else profiles.index(next(
            profile
            for profile in profiles
            if profile.name == (learned.choose(runtime_features) if learned else hand.choose(
                runtime_features, [value.name for value in profiles]
            ))
        ))
    )
    overrides = {
        "facet_count": facets,
        "retained_roots": roots,
        "neighbors_per_expansion": neighbors,
        "hop_depth": hops,
        "conceptual_budget": conceptual_budget,
        "native_kv_budget": max_active_kv,
        "routing_threshold": routing_threshold,
        "search_layers": tuple(search_layers) or None,
        "consumer_layers": tuple(consumer_layers) or None,
        "granularity_tokens": granularity,
        "materialization_policy": materialization_policy,
    }
    if any(value is not None for value in overrides.values()):
        profile = profiles[selected]
        profiles[selected] = replace(
            profile,
            **{name: value for name, value in overrides.items() if value is not None},
        )
        # Explicit per-profile overrides are a manual plan.  They do not mutate
        # or redefine the globally validated monotonic ladder.
    agent = AdaptiveRetryAgent(
        validate_effort_ladder(default_effort_profiles()),
        StopPolicy(max_incorrect_probability=confidence_threshold),
        max_retries=max_retries,
        max_search_budget=max_search_budget,
        max_active_kv=max_active_kv,
        latency_budget_seconds=latency_budget,
    )
    admissible = agent.admissible_profiles()
    if profiles[selected].name not in {profile.name for profile in admissible}:
        selected = max(index for index, profile in enumerate(profiles) if profile.name in {item.name for item in admissible})
    plan = {
        "mode": mode,
        "selected_profile": profiles[selected].to_dict(),
        "max_retries": max_retries,
        "max_search_budget": max_search_budget,
        "max_active_kv": max_active_kv,
        "latency_budget_seconds": latency_budget,
        "confidence_threshold": confidence_threshold,
        "trace_schema": [
            "attempt", "effort", "control_vector", "active_native_kv",
            "incorrect_probability", "latency", "escalation_reason", "stop_reason",
        ],
    }
    if trace_output:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    _echo_json(plan)


@cli.command()
@click.argument("model")
@click.argument("question")
@click.option("--routing-adapter", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--reference", "references", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-new-tokens", default=64, show_default=True)
def ask(model, question, routing_adapter, references, max_new_tokens) -> None:
    """Ask one question using optional text-file references."""
    pra = PRAForCausalLM.from_pretrained(model, routing_adapter=routing_adapter)
    for path in references:
        pra.add_reference_file(path)
    result = pra.generate(question, max_new_tokens=max_new_tokens, return_details=True)
    click.echo(result.text)
    _echo_json(result.stats)


@cli.command()
@click.argument("model")
@click.option("--routing-adapter", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--reference", "references", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def chat(model, routing_adapter, references) -> None:
    """Start a minimal terminal chat with persistent references."""
    pra = PRAForCausalLM.from_pretrained(model, routing_adapter=routing_adapter)
    for path in references:
        pra.add_reference_file(path)
    messages = []
    while True:
        question = click.prompt("you", prompt_suffix="> ")
        if question.strip().lower() in {"quit", "exit"}:
            break
        messages.append({"role": "user", "content": question})
        answer = pra.chat(messages)
        click.echo(f"assistant> {answer}")
        messages.append({"role": "assistant", "content": answer})
