from __future__ import annotations

import copy
import random
from pathlib import Path

import click
import torch
import yaml

from common.config import deep_update, read_yaml as read_yaml_file
from common.train import resolve_device
from config.model_size import estimate_model_size
from data.datamodules import PRADataModule
from data.datasets import read_jsonl
from .config import CacheServiceConfig, PRAConfig, ResolverServiceConfig, TrainConfig
from .eval import run_evaluation
from .pra_train import train_pra_model


DEFAULT_CONFIG_PATH = Path("config") / "config.yml"

BUILTIN_CONFIG = {
    "model": {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "n_vanilla_layers": 0,
        "n_mixed_layers": 0,
        "d_ff": None,
        "max_seq_len": 96,
        "dropout": 0.0,
    },
    "pra": {
        "max_prompt_direct_tokens": None,
        "prompt_overflow_mode": "truncate",
        "max_prompt_gists": None,
        "pra_layer_ids": [2, 3],
        "top_k_references": 2,
        "top_k_chunks_per_reference": 1,
        "trigger_threshold": 0.2,
        "memory_transport": "native_kv",
        "memory_alpha": 0.5,
        "search_strategy": "hierarchical",
        "routing_backend": "tensorized",
        "reference_score_aggregation": "max",
        "reference_level_gist_mode": None,
        "reference_gists_per_reference": 1,
        "reference_gist_score_aggregation": "max",
        "gist_mode": "mean",
        "gists_per_chunk": 1,
        "gist_score_aggregation": "max",
        "max_gists_per_reference": 4,
        "gist_overflow_policy": "truncate",
        "chunking_mode": "none",
        "fixed_chunk_tokens": 64,
        "fixed_chunk_overlap_tokens": 0,
        "reference_overflow_policy": "truncate",
        "detail_materialization": "selected_chunks",
        "kv_cache_residency": "gpu",
        "kv_cache_pin_memory": False,
        "kv_cache_non_blocking": False,
        "memory_bucket_count": 1,
        "memory_bucket_strategy": "optimal_contiguous",
        "cache_build_mode": "detached",
        "use_summary": False,
        "summary_mode": "replace",
        "recursive_refs_enabled": False,
        "recursive_max_depth": 2,
        "recursive_max_total_references": 16,
        "recursive_max_total_tokens": 2048,
        "recursive_max_children_per_reference": 8,
        "recursive_cycle_policy": "skip",
        "recursive_missing_ref_policy": "warn",
    },
    "train": {
        "steps": 300,
        "batch_size": 8,
        "lr": 3e-4,
        "device": "auto",
        "out": "pra_tiny.pt",
        "dataset_stage": "stage0_synthetic_memory",
        "data_dir": "data",
        "max_examples": None,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    },
    "resolver": {
        "type": "in_memory",
        "options": {},
    },
    "cache": {
        "type": "simple",
        "options": {},
    },
    "eval": {
        "ckpt": "pra_tiny.pt",
        "device": "auto",
        "examples": 20,
        "dataset_stage": "stage0_synthetic_memory",
        "data_dir": "data",
        "max_examples": None,
        "max_new_tokens": 24,
        "seed": 0,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    },
}


def read_yaml(path: Path) -> dict:
    if not path.exists():
        click.echo(f"Warning: config file not found: {path}", err=True)
        return {}
    return read_yaml_file(path)


def apply_named_model(cfg: dict, model_name: str = "default") -> dict:
    """Apply a named model profile over the global config sections."""
    models = cfg.get("models") or {}
    profile = models.get(model_name)
    if profile is None:
        if model_name == "default":
            return cfg
        raise KeyError(f"Unknown model profile: {model_name}")

    profile = copy.deepcopy(profile)
    model_keys = set(cfg["model"])
    direct_model_overrides = {key: profile.pop(key) for key in list(profile) if key in model_keys}
    if "model" in profile:
        deep_update(cfg["model"], profile["model"] or {})
    deep_update(cfg["model"], direct_model_overrides)
    for section in ("train", "pra", "resolver", "cache", "standalone"):
        if section in profile:
            deep_update(cfg.setdefault(section, {}), profile[section] or {})
    cfg["selected_model"] = model_name
    return cfg


def load_config(config_path: str | None = None, model_name: str = "default") -> dict:
    cfg = copy.deepcopy(BUILTIN_CONFIG)
    deep_update(cfg, read_yaml(DEFAULT_CONFIG_PATH))
    if config_path:
        deep_update(cfg, read_yaml(Path(config_path)))
    return apply_named_model(cfg, model_name)


def parse_layer_ids(value) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(v) for v in value)


def command_overrides(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}


def model_kwargs(cfg: dict) -> dict:
    model = cfg["model"]
    pra = cfg["pra"]
    return {
        "d_model": int(model["d_model"]),
        "n_heads": int(model["n_heads"]),
        "n_layers": int(model["n_layers"]),
        "n_vanilla_layers": int(model.get("n_vanilla_layers", 0)),
        "n_mixed_layers": int(model.get("n_mixed_layers", 0)),
        "d_ff": int(model["d_ff"]) if model.get("d_ff") is not None else None,
        "max_seq_len": int(model["max_seq_len"]),
        "dropout": float(model["dropout"]),
        "model_variant": str(model.get("model_variant", "custom")),
        "pra_layer_ids": parse_layer_ids(pra["pra_layer_ids"]),
        "top_k_references": int(pra["top_k_references"]),
        "top_k_chunks_per_reference": int(pra["top_k_chunks_per_reference"]),
        "trigger_threshold": float(pra["trigger_threshold"]),
        "memory_transport": str(pra.get("memory_transport", "native_kv")),
        "memory_alpha": float(pra["memory_alpha"]),
        **{
            key: pra[key]
            for key in ("use_cross_attention_memory", "use_concat_memory")
            if pra.get(key) is not None
        },
        **{
            key: pra[key]
            for key in (
                "max_prompt_direct_tokens",
                "prompt_overflow_mode",
                "max_prompt_gists",
                "search_strategy",
                "routing_backend",
                "reference_score_aggregation",
                "reference_level_gist_mode",
                "reference_gists_per_reference",
                "reference_gist_score_aggregation",
                "gist_mode",
                "gists_per_chunk",
                "gist_score_aggregation",
                "max_gists_per_reference",
                "gist_overflow_policy",
                "gist_kmeans_max_iters",
                "gist_kmeans_init",
                "gist_kmeans_tol",
                "gist_kmeans_normalize",
                "gist_kmeans_seed",
                "gist_kmeans_empty_cluster_policy",
                "gist_som_steps",
                "gist_som_learning_rate",
                "gist_som_final_learning_rate",
                "gist_som_neighborhood_radius",
                "gist_som_final_neighborhood_radius",
                "gist_som_distance",
                "gist_som_normalize",
                "gist_som_init",
                "gist_som_seed",
                "gist_som_topology",
                "gist_prototype_method",
                "gist_prototype_init",
                "gist_prototype_refine",
                "gist_prototype_normalize",
                "gist_prototype_distance",
                "gist_prototype_seed",
                "gist_hybrid_global_mode",
                "gist_hybrid_local_mode",
                "gist_hybrid_global_count",
                "gist_hybrid_deduplicate",
                "gist_hybrid_min_cosine_separation",
                "chunking_mode",
                "fixed_chunk_tokens",
                "fixed_chunk_overlap_tokens",
                "reference_overflow_policy",
                "detail_materialization",
                "kv_cache_residency",
                "kv_cache_pin_memory",
                "kv_cache_non_blocking",
                "memory_bucket_count",
                "memory_bucket_strategy",
                "cache_build_mode",
                "use_summary",
                "summary_mode",
                "recursive_refs_enabled",
                "recursive_max_depth",
                "recursive_max_total_references",
                "recursive_max_total_tokens",
                "recursive_max_children_per_reference",
                "recursive_cycle_policy",
                "recursive_missing_ref_policy",
            )
            if key in pra
        },
    }


def build_pra_config(cfg: dict, *, vocab_size: int, batch_size: int, lr: float, steps: int, device: str) -> PRAConfig:
    return PRAConfig(
        vocab_size=vocab_size,
        batch_size=batch_size,
        lr=lr,
        steps=steps,
        device=device,
        **model_kwargs(cfg),
    )


def apply_service_overrides(cfg: dict, options: dict) -> None:
    overrides = command_overrides(options)
    if "resolver_type" in overrides:
        cfg["resolver"]["type"] = overrides["resolver_type"]
    if "cache_type" in overrides:
        cfg["cache"]["type"] = overrides["cache_type"]


def apply_common_overrides(cfg: dict, options: dict) -> None:
    model_keys = {"d_model", "n_heads", "n_layers", "n_vanilla_layers", "n_mixed_layers", "d_ff", "max_seq_len", "dropout", "model_variant"}
    pra_keys = {
        "pra_layer_ids",
        "top_k_references",
        "top_k_chunks_per_reference",
        "trigger_threshold",
        "memory_transport",
        "use_cross_attention_memory",
        "use_concat_memory",
        "memory_alpha",
    }
    for key, value in command_overrides(options).items():
        if key in model_keys:
            cfg["model"][key] = value
        elif key in pra_keys:
            cfg["pra"][key] = parse_layer_ids(value) if key == "pra_layer_ids" else value


def common_model_options(func):
    options = [
        click.option("-M", "--d-model", type=int),
        click.option("-H", "--n-heads", type=int),
        click.option("-L", "--n-layers", type=int),
        click.option("-V", "--n-vanilla-layers", type=int),
        click.option("-Y", "--n-mixed-layers", type=int),
        click.option("-l", "--max-seq-len", type=int),
        click.option("-p", "--dropout", type=float),
        click.option("-P", "--pra-layer-ids", type=str, help="Comma-separated layer ids, e.g. 2,3."),
        click.option("-k", "--top-k-references", "--top-k-refs", type=int),
        click.option("--top-k-chunks-per-reference", type=int),
        click.option("-t", "--trigger-threshold", type=float),
        click.option(
            "--memory-transport",
            type=click.Choice(["native_kv", "cross_attention"]),
        ),
        click.option("-x/-X", "--use-cross-attention-memory/--no-use-cross-attention-memory", default=None),
        click.option("-u/-U", "--use-concat-memory/--no-use-concat-memory", default=None),
        click.option("-a", "--memory-alpha", type=float),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def common_service_options(func):
    options = [
        click.option("-R", "--resolver-type", type=str),
        click.option("-A", "--cache-type", type=str),
    ]
    for option in reversed(options):
        func = option(func)
    return func


@click.group()
def cli():
    """PRA standalone CLI."""


@cli.command()
@click.option("-c", "--config", "config_path", type=click.Path(dir_okay=False))
@click.option("--model", "model_name", default="default", show_default=True, help="Named model profile from the models section.")
@click.option("-s", "--steps", type=int)
@click.option("-b", "--batch-size", type=int)
@click.option("-r", "--lr", type=float)
@click.option("-d", "--device", type=str)
@click.option("-o", "--out", type=str)
@click.option("-g", "--dataset-stage", type=str)
@click.option("-D", "--data-dir", type=str)
@click.option("-m", "--max-examples", type=int)
@click.option("-w", "--num-workers", type=int)
@click.option("-i/-I", "--pin-memory/--no-pin-memory", default=None)
@click.option("-W/-N", "--persistent-workers/--no-persistent-workers", default=None)
@click.option("-q/-Q", "--shuffle/--no-shuffle", default=None)
@common_service_options
@common_model_options
def train(config_path, model_name, **options):
    """Train TinyPRA."""
    try:
        cfg = load_config(config_path, model_name=model_name)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    apply_common_overrides(cfg, options)
    apply_service_overrides(cfg, options)
    train_cfg = cfg["train"]
    for key, value in command_overrides(options).items():
        if key in train_cfg:
            train_cfg[key] = value

    device = resolve_device(train_cfg["device"])
    datamodule = PRADataModule(
        dataset_stage=train_cfg["dataset_stage"],
        data_dir=train_cfg["data_dir"],
        max_examples=train_cfg["max_examples"],
        batch_size=int(train_cfg["batch_size"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        shuffle=bool(train_cfg["shuffle"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]),
    ).load()
    pra_cfg = build_pra_config(
        cfg,
        vocab_size=datamodule.tokenizer.vocab_size,
        batch_size=int(train_cfg["batch_size"]),
        lr=float(train_cfg["lr"]),
        steps=int(train_cfg["steps"]),
        device=device,
    )
    out_path = Path(train_cfg["out"])
    train_runtime_cfg = TrainConfig(
        experiment_name=out_path.stem,
        output_dir=str(out_path.parent if out_path.parent != Path(".") else "out"),
        device=device,
        epochs=1,
        max_steps=int(train_cfg["steps"]) if int(train_cfg["steps"]) > 0 else None,
        batch_size=int(train_cfg["batch_size"]),
        grad_accum_steps=1,
        learning_rate=float(train_cfg["lr"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]),
        dataset_stage=train_cfg["dataset_stage"],
        data_dir=train_cfg["data_dir"],
        max_examples=train_cfg["max_examples"],
        shuffle=bool(train_cfg["shuffle"]),
        eval_every_steps=max(int(train_cfg["steps"]) + 1, 1),
        save_every_steps=max(int(train_cfg["steps"]) + 1, 1),
        log_every_steps=1,
        use_tensorboard=False,
        resolver_config=ResolverServiceConfig.from_value(cfg["resolver"]),
        cache_config=CacheServiceConfig.from_value(cfg["cache"]),
    )

    result = train_pra_model(
        cfg=pra_cfg,
        train_config=train_runtime_cfg,
        datamodule=datamodule,
        resolver_config=train_runtime_cfg.resolver_config,
        cache_config=train_runtime_cfg.cache_config,
    )
    torch.save(
        {
            "model": result["model"].state_dict(),
            "cfg": pra_cfg.__dict__,
            "stoi": datamodule.tokenizer.stoi,
            "itos": datamodule.tokenizer.itos,
            "dataset_stage": train_cfg["dataset_stage"],
            "resolver_config": train_runtime_cfg.resolver_config.__dict__,
            "cache_config": train_runtime_cfg.cache_config.__dict__,
        },
        train_cfg["out"],
    )
    click.echo(f"saved {train_cfg['out']}")


@cli.command(name="eval")
@click.option("-c", "--config", "config_path", type=click.Path(dir_okay=False))
@click.option("-K", "--ckpt", type=str)
@click.option("-d", "--device", type=str)
@click.option("-e", "--examples", type=int)
@click.option("-g", "--dataset-stage", type=str)
@click.option("-D", "--data-dir", type=str)
@click.option("-m", "--max-examples", type=int)
@click.option("-n", "--max-new-tokens", type=int)
@click.option("-S", "--seed", type=int)
@click.option("-b", "--batch-size", type=int)
@click.option("-w", "--num-workers", type=int)
@click.option("-i/-I", "--pin-memory/--no-pin-memory", default=None)
@click.option("-W/-N", "--persistent-workers/--no-persistent-workers", default=None)
@common_service_options
@common_model_options
def eval(config_path, **options):
    """Evaluate TinyPRA baselines."""
    cfg = load_config(config_path)
    apply_common_overrides(cfg, options)
    apply_service_overrides(cfg, options)
    eval_cfg = cfg["eval"]
    for key, value in command_overrides(options).items():
        if key in eval_cfg:
            eval_cfg[key] = value

    random.seed(int(eval_cfg["seed"]))
    torch.manual_seed(int(eval_cfg["seed"]))

    max_examples = eval_cfg["max_examples"] if eval_cfg["max_examples"] is not None else eval_cfg["examples"]
    datamodule = PRADataModule(
        dataset_stage=eval_cfg["dataset_stage"],
        data_dir=eval_cfg["data_dir"],
        max_examples=max_examples,
        batch_size=int(eval_cfg["batch_size"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
        pin_memory=bool(eval_cfg["pin_memory"]),
        persistent_workers=bool(eval_cfg["persistent_workers"]),
    ).load()
    run_evaluation(
        ckpt=eval_cfg["ckpt"],
        device=resolve_device(eval_cfg["device"]),
        datamodule=datamodule,
        max_new_tokens=int(eval_cfg["max_new_tokens"]),
        resolver_config=ResolverServiceConfig.from_value(cfg["resolver"]),
        cache_config=CacheServiceConfig.from_value(cfg["cache"]),
        **model_kwargs(cfg),
    )


@cli.group()
def config():
    """Inspect CLI configuration."""


@config.command()
@click.option("-c", "--config", "config_path", type=click.Path(dir_okay=False))
def show(config_path):
    """Show merged default configuration."""
    click.echo(yaml.safe_dump(load_config(config_path), sort_keys=False))


@cli.command()
@click.option("-c", "--config", "config_path", type=click.Path(dir_okay=False))
@click.option("--model", "model_name", default="default", show_default=True, help="Named model profile from the models section.")
@click.option("-v", "--vocab-size", type=int, default=128, show_default=True)
@click.option("-b", "--batch-size", type=int)
@click.option("-d", "--dtype", type=click.Choice(["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"]), default="float32", show_default=True)
@click.option("-o", "--optimizer", type=click.Choice(["adamw", "adam", "sgd", "none"]), default="adamw", show_default=True)
def size(config_path, model_name, vocab_size, batch_size, dtype, optimizer):
    """Estimate memory requirements for a named model profile."""
    try:
        cfg = load_config(config_path, model_name=model_name)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    estimate = estimate_model_size(
        cfg,
        vocab_size=vocab_size,
        batch_size=batch_size,
        dtype=dtype,
        optimizer=optimizer,
    )
    click.echo(yaml.safe_dump(estimate, sort_keys=False))


@cli.group()
def dataset():
    """Inspect datasets."""


@dataset.command()
@click.option("-c", "--config", "config_path", type=click.Path(dir_okay=False))
@click.option("-g", "--dataset-stage", type=str)
@click.option("-D", "--data-dir", type=str)
@click.option("-m", "--max-examples", type=int)
def show(config_path, dataset_stage, data_dir, max_examples):
    """Show stats for a named dataset stage."""
    cfg = load_config(config_path)
    stage = dataset_stage or cfg["train"]["dataset_stage"]
    root = data_dir or cfg["train"]["data_dir"]
    stage_path = Path(root) / stage
    datamodule = PRADataModule(stage, root, max_examples=max_examples, shuffle=False).load()
    docs = read_jsonl(stage_path / "documents.jsonl")
    refs = read_jsonl(stage_path / "references.jsonl")
    questions = read_jsonl(stage_path / "questions.jsonl")
    expected_refs = sum(len(sample.target_reference_ids) for sample in datamodule.dataset)
    expected_anchors = sum(len(sample.metadata.get("expected_anchors", [])) for sample in datamodule.dataset)
    stats = {
        "stage": stage,
        "data_dir": str(root),
        "documents": len(docs),
        "references": len(refs),
        "questions": len(questions),
        "loaded_examples": len(datamodule.dataset),
        "expected_refs": expected_refs,
        "expected_anchors": expected_anchors,
    }
    click.echo(yaml.safe_dump(stats, sort_keys=False))


if __name__ == "__main__":
    cli()
