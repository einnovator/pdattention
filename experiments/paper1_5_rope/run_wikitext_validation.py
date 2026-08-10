"""Train matched WikiText models and evaluate bounded positional continuity."""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from data.language_modeling import WikiTextDataModule  # noqa: E402
from data.tokenizer import BPETokenizer  # noqa: E402
from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    TIERS,
    environment_metadata,
    set_seed,
    write_csv,
    write_json,
)
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402


VALIDATION = RESULTS / "validation"
POSITION_CAPACITY = 256
MODEL_OPERATION_TOKENS = 32
OVERLAP_CORE_TOKENS = 16
OVERLAP_TOKENS = 16
RATIOS = (1, 2, 4, 8)
MODES = ("absolute", "sinusoidal", "rope")
COLORS = {"absolute": "#245A8D", "sinusoidal": "#327A5A", "rope": "#A34832"}


def _settings(tier: str, mode: str, smoke: bool) -> dict:
    return {
        **TIERS[tier],
        "position_encoding": mode,
        "max_seq_len": POSITION_CAPACITY,
        "training_seq_len": 128,
        "vocab_size": 2_000,
        "batch_size": 2 if smoke else (16 if tier == "tiny" else 8),
        "steps": 2 if smoke else TIERS[tier]["steps"],
        "learning_rate": 7e-4,
        "max_train_documents": 32 if smoke else 512,
        "max_eval_documents": 8 if smoke else 128,
        "max_train_blocks": 16 if smoke else 2_048,
    }


def _prepare_data(smoke: bool) -> WikiTextDataModule:
    root = REPO / "out" / "paper1_5_rope" / "validation" / "wikitext_data"
    tokenizer_path = root / ("tokenizer_smoke.json" if smoke else "tokenizer.json")
    tokenizer = (
        BPETokenizer.from_json(tokenizer_path.read_text(encoding="utf-8"))
        if tokenizer_path.exists()
        else None
    )
    module = WikiTextDataModule(
        data_dir=root,
        dataset_name="wikitext-2-raw-v1",
        vocab_size=2_000,
        seq_len=128,
        batch_size=16,
        max_train_documents=32 if smoke else 512,
        max_eval_documents=8 if smoke else 128,
        max_train_blocks=16 if smoke else 2_048,
        pin_memory=torch.cuda.is_available(),
        tokenizer=tokenizer,
    ).load()
    if not tokenizer_path.exists():
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text(module.tokenizer.to_json(), encoding="utf-8")
    return module


@torch.no_grad()
def _evaluate_loader(model, loader, device: str, max_batches: int = 16) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = total = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inputs, use_pra_memory=False)
        losses.append(float(F.cross_entropy(logits.flatten(0, 1), labels.flatten()).cpu()))
        correct += int(logits.argmax(dim=-1).eq(labels).sum())
        total += labels.numel()
    return statistics.fmean(losses), correct / max(total, 1)


def _train_or_load(
    *, tier: str, mode: str, seed: int, module: WikiTextDataModule, device: str, smoke: bool, force: bool
) -> tuple[TinyPRAModel, dict]:
    settings = _settings(tier, mode, smoke)
    run_dir = REPO / "out" / "paper1_5_rope" / "validation" / "wikitext" / tier / mode / f"seed-{seed}"
    checkpoint_path = run_dir / "checkpoint.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = PRAConfig(
        vocab_size=module.tokenizer.vocab_size,
        d_model=settings["d_model"],
        n_heads=settings["n_heads"],
        n_layers=settings["n_layers"],
        d_ff=settings["d_ff"],
        max_seq_len=settings["max_seq_len"],
        model_variant="td_sa",
        position_encoding=mode,
        dropout=0.0,
        device=device,
    )
    model = TinyPRAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0.0)
    start_step = 0
    history = []
    if checkpoint_path.exists() and not force:
        payload = torch.load(checkpoint_path, map_location=device)
        if payload.get("settings") == settings:
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["step"])
            history = list(payload.get("history", []))
    set_seed(seed)
    module.batch_size = int(settings["batch_size"])
    loader = module._loader(module.train_dataset, True)
    iterator = itertools.cycle(loader)
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, settings["steps"] + 1):
        batch = next(iterator)
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, use_pra_memory=False)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == settings["steps"]:
            val_loss, val_accuracy = _evaluate_loader(model, module.val_loader(), device)
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                }
            )
            print(
                f"wikitext {tier}/{mode}/seed-{seed} {step}/{settings['steps']} "
                f"train={history[-1]['train_loss']:.4f} val={val_loss:.4f}",
                flush=True,
            )
            model.train()
    if start_step < settings["steps"] or force:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "settings": settings,
                "step": settings["steps"],
                "history": history,
                "tokenizer": module.tokenizer.to_json(),
            },
            checkpoint_path,
        )
    return model.eval(), {
        "checkpoint": str(checkpoint_path),
        "checkpoint_reused": start_step >= settings["steps"] and not force,
        "steps": settings["steps"],
        "start_step": start_step,
        "training_seconds_this_run": time.perf_counter() - started,
        "history": history,
        "settings": settings,
    }


def _converted(source: TinyPRAModel, device: str) -> TinyPRAModel:
    target_cfg = PRAConfig(
        **{
            **source.cfg.__dict__,
            "model_variant": "td_pra",
            "memory_transport": "native_kv",
            "dropout": 0.0,
            "device": device,
        }
    )
    return convert_sa_model_to_pra(source, target_cfg).to(device).eval()


@torch.no_grad()
def _segmented(
    source: TinyPRAModel,
    converted: TinyPRAModel,
    inputs: list[int],
    *,
    stage: str,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], int, int]:
    use_overlap = stage == "offset_overlap"
    use_offsets = stage != "reset"
    core_size = OVERLAP_CORE_TOKENS if use_overlap else MODEL_OPERATION_TOKENS
    logits = []
    keys = {layer: [] for layer in range(source.cfg.n_layers)}
    processed = 0
    maximum_operation = 0
    for core_start in range(0, len(inputs), core_size):
        core_end = min(core_start + core_size, len(inputs))
        left = min(core_start, OVERLAP_TOKENS) if use_overlap else 0
        encode_start = core_start - left
        ids = inputs[encode_start:core_end]
        offset = encode_start if use_offsets else 0
        tensor = torch.tensor([ids], dtype=torch.long, device=next(source.parameters()).device)
        chunk_logits = source(tensor, use_pra_memory=False, position_offset=offset)
        captured = converted._encode_reference_tokens(
            ids,
            tensor.device,
            detach=True,
            use_pra_memory=False,
            position_offset=offset,
        )
        core_count = core_end - core_start
        logits.append(chunk_logits[:, left : left + core_count])
        for layer, kv in captured.items():
            keys[layer].append(kv.k[:, :, left : left + core_count])
        processed += len(ids)
        maximum_operation = max(maximum_operation, len(ids))
    return (
        torch.cat(logits, dim=1),
        {layer: torch.cat(values, dim=2) for layer, values in keys.items()},
        processed,
        maximum_operation,
    )


@torch.no_grad()
def _evaluate_model(
    source: TinyPRAModel,
    converted: TinyPRAModel,
    module: WikiTextDataModule,
    *,
    tier: str,
    mode: str,
    seed: int,
    git_sha: str,
    max_examples: int,
) -> list[dict]:
    tokens = module.test_dataset.tokens.tolist()
    rows = []
    for ratio in RATIOS:
        logical = MODEL_OPERATION_TOKENS * ratio
        stride = logical + 17
        for example_index in range(max_examples):
            start = example_index * stride
            window = tokens[start : start + logical + 1]
            if len(window) < logical + 1:
                break
            inputs, labels = window[:-1], torch.tensor(window[1:], device=next(source.parameters()).device)
            dense_tensor = torch.tensor([inputs], dtype=torch.long, device=labels.device)
            dense_logits = source(dense_tensor, use_pra_memory=False)
            dense_kv = converted._encode_reference_tokens(
                inputs, labels.device, detach=True, use_pra_memory=False, position_offset=0
            )
            dense_loss = float(F.cross_entropy(dense_logits[0], labels))
            for stage in ("reset", "offset", "offset_overlap"):
                logits, keys, processed, maximum = _segmented(source, converted, inputs, stage=stage)
                loss = float(F.cross_entropy(logits[0], labels))
                rows.append(
                    {
                        "git_sha": git_sha,
                        "seed": seed,
                        "model_tier": tier,
                        "position_mode": mode,
                        "dataset": "wikitext-2-raw-v1",
                        "stage": stage,
                        "example_id": example_index,
                        "logical_context": logical,
                        "model_operation_limit": MODEL_OPERATION_TOKENS,
                        "position_capacity": POSITION_CAPACITY if mode == "absolute" else None,
                        "logical_native_ratio": ratio,
                        "overlap_fraction": 0.5 if stage == "offset_overlap" else 0.0,
                        "maximum_native_operation": maximum,
                        "native_limit_violations": int(maximum > MODEL_OPERATION_TOKENS),
                        "processed_tokens": processed,
                        "overlap_cost_ratio": processed / logical,
                        "dense_loss": dense_loss,
                        "loss": loss,
                        "perplexity": math.exp(min(loss, 20.0)),
                        "token_accuracy": float(logits[0].argmax(dim=-1).eq(labels).float().mean()),
                        "layer0_k_rmse": float((keys[0] - dense_kv[0].k).square().mean().sqrt()),
                        "final_layer_k_rmse": float(
                            (keys[source.cfg.n_layers - 1] - dense_kv[source.cfg.n_layers - 1].k)
                            .square()
                            .mean()
                            .sqrt()
                        ),
                    }
                )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_tier"], row["position_mode"], row["stage"], row["logical_native_ratio"])].append(row)
    output = []
    for identity, values in sorted(groups.items()):
        result = dict(zip(("model_tier", "position_mode", "stage", "logical_native_ratio"), identity))
        result["seed_count"] = len({row["seed"] for row in values})
        result["example_count"] = len(values)
        for metric in (
            "loss",
            "perplexity",
            "token_accuracy",
            "layer0_k_rmse",
            "final_layer_k_rmse",
            "overlap_cost_ratio",
            "maximum_native_operation",
        ):
            observed = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = statistics.fmean(observed)
            result[f"{metric}_median"] = statistics.median(observed)
            result[f"{metric}_std"] = statistics.pstdev(observed)
        output.append(result)
    return output


def _plot(rows: list[dict], path: Path) -> None:
    present_modes = [mode for mode in MODES if any(row["position_mode"] == mode for row in rows)]
    present_tiers = [tier for tier in ("tiny", "small") if any(row["model_tier"] == tier for row in rows)]
    figure, axes = plt.subplots(
        len(present_tiers),
        len(present_modes),
        figsize=(3.7 * len(present_modes), 3.2 * len(present_tiers)),
        sharex=True,
        squeeze=False,
    )
    stages = (("reset", "--"), ("offset", "-"), ("offset_overlap", ":"))
    for column, mode in enumerate(present_modes):
        for row_index, tier in enumerate(present_tiers):
            axis = axes[row_index, column]
            for stage, style in stages:
                values = []
                for ratio in RATIOS:
                    observed = [
                        row["loss"]
                        for row in rows
                        if row["model_tier"] == tier
                        and row["position_mode"] == mode
                        and row["stage"] == stage
                        and row["logical_native_ratio"] == ratio
                    ]
                    values.append(statistics.fmean(observed))
                axis.plot(RATIOS, values, style, marker="o", color=COLORS[mode], label=stage)
            axis.set_title(f"{tier} / {mode}")
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("Next-token loss")
            if row_index == len(present_tiers) - 1:
                axis.set_xlabel("Logical / native-operation ratio")
            axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args) -> Path:
    metadata = environment_metadata()
    module = _prepare_data(args.smoke)
    training_rows = []
    rows = []
    for tier in args.tiers:
        for mode in args.position_modes:
            for seed in args.seeds:
                set_seed(seed)
                source, training = _train_or_load(
                    tier=tier,
                    mode=mode,
                    seed=seed,
                    module=module,
                    device=args.device,
                    smoke=args.smoke,
                    force=args.force,
                )
                converted = _converted(source, args.device)
                training_rows.append(
                    {
                        "git_sha": metadata["git_sha"],
                        "seed": seed,
                        "model_tier": tier,
                        "position_mode": mode,
                        "parameter_count": sum(p.numel() for p in source.parameters()),
                        "training_dataset": "wikitext-2-raw-v1",
                        "training_tokens": training["steps"]
                        * training["settings"]["batch_size"]
                        * training["settings"]["training_seq_len"],
                        "native_training_context": training["settings"]["training_seq_len"],
                        "optimizer": "AdamW",
                        "scheduler": "constant",
                        "batch_size": training["settings"]["batch_size"],
                        "steps": training["steps"],
                        "final_train_loss": training["history"][-1]["train_loss"],
                        "validation_loss": training["history"][-1]["validation_loss"],
                        "checkpoint": training["checkpoint"],
                        "checkpoint_reused": training["checkpoint_reused"],
                    }
                )
                rows.extend(
                    _evaluate_model(
                        source,
                        converted,
                        module,
                        tier=tier,
                        mode=mode,
                        seed=seed,
                        git_sha=metadata["git_sha"],
                        max_examples=1 if args.smoke else args.max_examples,
                    )
                )
                del source, converted
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    aggregate = _aggregate(rows)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    path = VALIDATION / "wikitext_position_validation.json"
    write_json(
        path,
        {
            "metadata": metadata,
            "smoke": args.smoke,
            "expectations_recorded_before_analysis": [
                {
                    "expected": "offsets reduce representation error for all three mechanisms",
                    "reason": "source coordinates should not depend on encoding segmentation",
                },
                {
                    "expected": "offsets often improve natural-text loss, but need not do so monotonically",
                    "reason": "contextualization may dominate position repair",
                },
                {
                    "expected": "overlap may reduce deeper error after offsets",
                    "reason": "left context is reintroduced at additional encoding cost",
                },
            ],
            "training": training_rows,
            "rows": rows,
            "aggregate": aggregate,
        },
    )
    write_csv(VALIDATION / "wikitext_training.csv", training_rows)
    write_csv(VALIDATION / "wikitext_position_validation.csv", rows)
    write_csv(VALIDATION / "wikitext_position_validation_aggregate.csv", aggregate)
    _plot(rows, VALIDATION / "wikitext_position_validation.png")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument("--position-modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
