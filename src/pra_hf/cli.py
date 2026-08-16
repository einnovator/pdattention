"""Small command-line interface for the product-facing PRA-HF workflow."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from .evaluation import evaluate_router_features
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
