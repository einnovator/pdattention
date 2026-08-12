"""Run the bounded Paper 2 last-14 conditional-LoRA convergence sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_last14_combo import (
    SEEDS,
    Variant,
    _aggregates,
    _compact_gold_scores,
    _configure_variant,
    _context_controls,
    _evaluate_memory,
    _freeze_backbone,
    _generate,
    _load_checkpoint,
    _prepare_records,
    _seed_aggregates,
    _train_variant,
    _verify_off_exact,
    _write_csv,
    last_band_layers,
)
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_hf import PRAMemoryAdapter, PRARouter, __version__ as pra_hf_version
from pra_torch.hf import PRAHFConfig, inject_pra, load_hf_routing_projection


RANKS = (4, 8, 16, 32)
BASE_STEPS = 32
BASE_LEARNING_RATE = 1e-3
SCREEN_SEED = 11


@dataclass(frozen=True)
class SweepConfig:
    """One fixed-rank optimization setting in the predeclared search space."""

    rank: int
    steps: int
    learning_rate: float
    stage: str

    @property
    def step_multiplier(self) -> float:
        return self.steps / BASE_STEPS

    @property
    def learning_rate_multiplier(self) -> float:
        return self.learning_rate / BASE_LEARNING_RATE

    @property
    def config_id(self) -> str:
        lr = f"{self.learning_rate_multiplier:g}".replace(".", "p")
        return f"lora_o_r{self.rank}_s{self.steps}_lr{lr}"

    @property
    def variant(self) -> Variant:
        return Variant(self.config_id, lora_rank=self.rank)


def stage_a_configs() -> list[SweepConfig]:
    """Return rank 4/8/16/32 at the current and doubled budgets."""

    return [
        SweepConfig(rank, steps, BASE_LEARNING_RATE, "A")
        for rank in RANKS
        for steps in (BASE_STEPS, 2 * BASE_STEPS)
    ]


def stage_b_configs(top_ranks: Iterable[int]) -> list[SweepConfig]:
    """Expand only the three best Stage-A ranks across bounded LR/budget values."""

    return [
        SweepConfig(rank, steps, BASE_LEARNING_RATE * lr_scale, "B")
        for rank in sorted(set(int(value) for value in top_ranks))
        for steps in (2 * BASE_STEPS, 4 * BASE_STEPS)
        for lr_scale in (0.5, 1.0, 2.0)
    ]


def _rows_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["dataset"], row["condition"]): row for row in rows}


def validation_metrics(aggregate_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute the equal-dataset validation criterion and routed diagnostic."""

    lookup = _rows_by_key(aggregate_rows)

    def value(dataset: str, condition: str, metric: str):
        row = lookup.get((dataset, condition), {})
        return row.get(metric)

    oracle = [
        value(dataset, "oracle", "gold_sequence_logprob_delta_vs_none_mean")
        for dataset in ("hotpotqa", "qasper")
    ]
    routed = [
        value(dataset, "routed", "gold_sequence_logprob_delta_vs_none_mean")
        for dataset in ("hotpotqa", "qasper")
    ]
    if any(item is None for item in oracle):
        raise ValueError("Both validation datasets require oracle sequence-logP metrics.")
    return {
        "combined_oracle_delta_logp": statistics.fmean(float(item) for item in oracle),
        "combined_routed_delta_logp": (
            statistics.fmean(float(item) for item in routed)
            if all(item is not None for item in routed)
            else None
        ),
        "hotpotqa_oracle_delta_logp": oracle[0],
        "qasper_oracle_delta_logp": oracle[1],
        "hotpotqa_routed_delta_logp": routed[0],
        "qasper_routed_delta_logp": routed[1],
    }


def rank_screen_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidate summaries without consulting test metrics."""

    return sorted(
        records,
        key=lambda row: (
            -float(row["combined_oracle_delta_logp"]),
            int(row["rank"]),
            int(row["steps"]),
            float(row["learning_rate"]),
        ),
    )


def top_stage_a_ranks(records: list[dict[str, Any]], count: int = 3) -> list[int]:
    """Choose distinct ranks by their best Stage-A held-out score."""

    best: dict[int, float] = {}
    for row in records:
        rank = int(row["rank"])
        best[rank] = max(best.get(rank, -math.inf), float(row["combined_oracle_delta_logp"]))
    return [rank for rank, _ in sorted(best.items(), key=lambda item: (-item[1], item[0]))[:count]]


def select_stage_c_configs(
    records: list[dict[str, Any]], count: int = 3
) -> list[str]:
    """Keep the best setting plus parameter-efficient, rank-diverse finalists."""

    ranked = rank_screen_records(records)
    selected: list[dict[str, Any]] = []
    for row in ranked:
        if not selected:
            selected.append(row)
            continue
        if int(row["rank"]) not in {int(value["rank"]) for value in selected}:
            selected.append(row)
        if len(selected) == count:
            break
    for row in ranked:
        if len(selected) == count:
            break
        if row not in selected:
            selected.append(row)
    return [str(row["config_id"]) for row in selected]


def choose_pareto_winner(
    records: list[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    """Choose the smallest rank within an absolute validation tolerance of best."""

    best_score = max(float(row["combined_oracle_delta_logp"]) for row in records)
    eligible = [
        row
        for row in records
        if float(row["combined_oracle_delta_logp"]) >= best_score - tolerance
    ]
    return min(
        eligible,
        key=lambda row: (
            int(row["memory_use_parameters"]),
            -float(row["combined_oracle_delta_logp"]),
            int(row["steps"]),
            abs(float(row["learning_rate_multiplier"]) - 1.0),
        ),
    )


def _config_manifest(args) -> dict[str, Any]:
    return {
        "protocol": "bounded last-14 PRA-conditional output-LoRA convergence sweep",
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "fixed_architecture": {
            "consumption_layers": "last-14",
            "target": "PRA-active native attention output projection",
            "alpha_rule": "alpha equals rank",
            "dropout": 0.0,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "objective": "oracle-memory gold answer mean negative log-likelihood",
            "routing_checkpoint": str(args.checkpoint),
            "routing_chunk_tokens": 32,
            "top_k_chunks_per_reference": 3,
            "max_materialized_memory_tokens": args.memory_tokens,
        },
        "stage_a": [asdict(config) | {"config_id": config.config_id} for config in stage_a_configs()],
        "stage_b_rule": {
            "rank_selection": "best three distinct Stage-A ranks on validation",
            "steps": [2 * BASE_STEPS, 4 * BASE_STEPS],
            "learning_rate_multipliers": [0.5, 1.0, 2.0],
        },
        "stage_c_rule": "best three rank-diverse configs; five validation and test seeds",
        "seeds": list(args.seeds),
        "selection_metric": "equal-weight HotpotQA/QASPER validation oracle sequence delta-logP",
        "pareto_tolerance_nats": args.pareto_tolerance,
        "test_access_rule": "test references are prepared only after Stage-C validation selection",
        "combination": "one residual-32 plus selected LoRA configuration",
        "identity_split": {
            "data_seed": args.data_seed,
            "train_offset": 0,
            "validation_offset": args.validation_offset,
            "test_offset": args.test_offset,
        },
    }


def _checkpoint_path(output_dir: Path, config: SweepConfig, seed: int) -> Path:
    return output_dir / "checkpoints" / f"{config.config_id}_seed{seed}.pt"


def _run_candidate(
    *,
    handle,
    tokenizer,
    train_records,
    validation_records,
    validation_controls,
    layers,
    route_layer,
    config: SweepConfig,
    seed: int,
    args,
    device,
    base_parameters: int,
    router_parameters: int,
    reference_record,
    off_reference,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report, checkpoint_bytes = _train_variant(
        handle,
        train_records,
        layers,
        config.variant,
        seed,
        config.steps,
        config.learning_rate,
        device,
        _checkpoint_path(args.output_dir, config, seed),
    )
    report.update(
        {
            "config_id": config.config_id,
            "rank": config.rank,
            "stage": config.stage,
            "step_multiplier": config.step_multiplier,
            "learning_rate_multiplier": config.learning_rate_multiplier,
            "epochs_equivalent": config.steps / len(train_records),
            "examples_seen": config.steps,
            "early_stopping": False,
            "checkpoint_bytes": checkpoint_bytes,
        }
    )
    report["pra_off_exact"] = _verify_off_exact(
        handle,
        tokenizer,
        reference_record,
        layers,
        off_reference,
        device,
        args.new_tokens,
    )
    report["adapter_bypass_exact"] = report["pra_off_exact"]
    report["cache_contract"] = "covered by exact disabled-path HF tests"
    if not report["pra_off_exact"]:
        raise AssertionError(f"PRA-off exactness failed for {config.config_id}, seed {seed}.")
    rows = _evaluate_memory(
        handle,
        tokenizer,
        validation_records,
        validation_controls,
        layers,
        route_layer,
        config.variant,
        seed,
        args.new_tokens,
        device,
        base_parameters,
        router_parameters,
        checkpoint_bytes,
        args.recovery_epsilon,
    )
    return report, rows


def _summary_record(
    config: SweepConfig,
    rows: list[dict[str, Any]],
    training_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    seed_rows = _seed_aggregates(rows)
    aggregate_rows = _aggregates(seed_rows)
    metrics = validation_metrics(aggregate_rows)
    first = rows[0]
    matching_reports = [row for row in training_reports if row["config_id"] == config.config_id]
    return {
        "config_id": config.config_id,
        "stage": config.stage,
        "rank": config.rank,
        "steps": config.steps,
        "step_multiplier": config.step_multiplier,
        "learning_rate": config.learning_rate,
        "learning_rate_multiplier": config.learning_rate_multiplier,
        "seeds": len({row["seed"] for row in rows}),
        "memory_use_parameters": int(first["memory_use_parameters"]),
        "memory_use_parameter_percent": float(first["memory_use_parameter_percent"]),
        "checkpoint_bytes": int(first["checkpoint_bytes"]),
        "training_seconds_mean": statistics.fmean(
            float(row.get("training_seconds", 0.0)) for row in matching_reports
        ),
        "examples_seen_per_seed": config.steps,
        "epochs_equivalent": config.steps / 12,
        "early_stopping": False,
        "pra_off_all_exact": all(row["pra_off_exact"] for row in matching_reports),
        **metrics,
    }


def _evaluate_checkpoint(
    *,
    handle,
    tokenizer,
    records,
    controls,
    layers,
    route_layer,
    config: SweepConfig,
    seed: int,
    args,
    device,
    base_parameters: int,
    router_parameters: int,
) -> list[dict[str, Any]]:
    _freeze_backbone(handle)
    _configure_variant(handle, config.variant, reset=True)
    checkpoint = _checkpoint_path(args.output_dir, config, seed)
    _, checkpoint_bytes = _load_checkpoint(checkpoint, handle, config.variant)
    return _evaluate_memory(
        handle,
        tokenizer,
        records,
        controls,
        layers,
        route_layer,
        config.variant,
        seed,
        args.new_tokens,
        device,
        base_parameters,
        router_parameters,
        checkpoint_bytes,
        args.recovery_epsilon,
    )


def _per_seed_validation_score(rows: list[dict[str, Any]], config_id: str) -> dict[int, float]:
    selected = [row for row in rows if row["variant"] == config_id]
    output = {}
    for seed in sorted({int(row["seed"]) for row in selected}):
        seed_rows = [row for row in selected if int(row["seed"]) == seed]
        output[seed] = float(validation_metrics(_aggregates(_seed_aggregates(seed_rows)))["combined_oracle_delta_logp"])
    return output


def _paired_summary(
    rows: list[dict[str, Any]], combo_id: str, winner_id: str
) -> list[dict[str, Any]]:
    aggregate = {
        (row["seed"], row["variant"], row["dataset"], row["condition"]): row
        for row in _seed_aggregates(rows)
    }
    output = []
    seeds = sorted({int(row["seed"]) for row in rows if row["variant"] == combo_id})
    for dataset in ("hotpotqa", "qasper"):
        for condition in ("oracle", "routed"):
            differences = [
                aggregate[(seed, combo_id, dataset, condition)]["gold_sequence_logprob_delta_vs_none"]
                - aggregate[(seed, winner_id, dataset, condition)]["gold_sequence_logprob_delta_vs_none"]
                for seed in seeds
            ]
            std = statistics.stdev(differences) if len(differences) > 1 else 0.0
            output.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "combo": combo_id,
                    "comparator": winner_id,
                    "paired_differences": differences,
                    "mean": statistics.fmean(differences),
                    "std": std,
                    "ci95": 2.776 * std / math.sqrt(5) if len(differences) == 5 else None,
                    "same_direction": all(value > 0 for value in differences)
                    or all(value < 0 for value in differences),
                }
            )
    return output


def _write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row if not isinstance(row[key], (dict, list))))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _plot_pareto(
    screen: list[dict[str, Any]],
    finalist: list[dict[str, Any]],
    winner_id: str,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.scatter(
        [row["memory_use_parameter_percent"] for row in screen],
        [row["combined_oracle_delta_logp"] for row in screen],
        color="#8A969F",
        marker="o",
        s=28,
        alpha=0.75,
        label="single-seed screen",
    )
    for row in finalist:
        selected = row["config_id"] == winner_id
        axis.scatter(
            row["memory_use_parameter_percent"],
            row["combined_oracle_delta_logp"],
            color="#A34832" if selected else "#245A8D",
            marker="*" if selected else "s",
            s=120 if selected else 58,
            label="Pareto selection" if selected else None,
            zorder=3,
        )
        axis.annotate(
            f"r{row['rank']} {row['step_multiplier']:g}x {row['learning_rate_multiplier']:g}xLR",
            (row["memory_use_parameter_percent"], row["combined_oracle_delta_logp"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Conditional LoRA parameters (% of base model)")
    axis.set_ylabel("Validation oracle delta-logP (equal dataset mean)")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"lora_parameter_pareto.{suffix}", dpi=190)
    plt.close(figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _package_sdk_artifacts(
    *,
    args,
    winner: SweepConfig,
    selected_seed: int,
    layers: tuple[int, ...],
    validation_record: dict[str, Any],
    test_record: dict[str, Any],
) -> dict[str, Any]:
    router_metadata = {
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "pra_version": pra_hf_version,
        "model_family": "qwen",
        "routing_representation": "attention_input_hidden_state",
        "routing_layer": layers[-1],
        "query_strategy": "last",
        "chunk_tokens": 32,
        "training_datasets": ["HotpotQA", "QASPER"],
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "git_sha": _git_sha(),
        "scope": "router used by the Paper 2 last-14 memory-use sweep",
    }
    router = PRARouter.from_experiment_checkpoint(
        args.checkpoint, metadata=router_metadata, device="cpu"
    )
    router.save_pretrained(args.output_router)
    checkpoint = _checkpoint_path(args.output_dir, winner, selected_seed)
    memory_metadata = {
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "pra_version": pra_hf_version,
        "model_family": "qwen",
        "target_modules": ["self_attn.o_proj"],
        "conditional_on": "PRA-active selected native-K/V attention",
        "selected_pra_depth": "last-14",
        "training_dataset": "HotpotQA",
        "dataset_split_seed": args.data_seed,
        "training_objective": "oracle-memory gold answer mean negative log-likelihood",
        "selection_rule": "validation-only Pareto rank within tolerance of best",
        "validation_metrics": validation_record,
        "test_metrics": test_record,
        "compatible_router": str(args.output_router.relative_to(ROOT)),
        "compatible_router_config_sha256": _sha256(args.output_router / "config.json"),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "git_sha": _git_sha(),
    }
    adapter = PRAMemoryAdapter.from_experiment_checkpoint(
        checkpoint,
        layer_ids=layers,
        alpha=float(winner.rank),
        dropout=0.0,
        metadata=memory_metadata,
    )
    adapter.save_pretrained(args.output_memory_adapter)
    return {
        "router": str(args.output_router.relative_to(ROOT)),
        "router_parameters": router.parameter_count,
        "memory_adapter": str(args.output_memory_adapter.relative_to(ROOT)),
        "memory_adapter_parameters": adapter.parameter_count,
        "selected_seed": selected_seed,
    }


def run(args) -> dict[str, Any]:
    if not args.allow_cpu and (args.device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("The overnight sweep requires CUDA; pass --allow-cpu only for smoke tests.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _config_manifest(args)
    (args.output_dir / "overnight_lora_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = load_hf_routing_projection(args.checkpoint, device=device)
    layers = last_band_layers(int(model.config.num_hidden_layers))
    route_layer = layers[-1]
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=layers,
            model_max_context_tokens=args.native_tokens,
            max_prompt_direct_tokens=args.prompt_tokens,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=3,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    base_parameters = sum(
        parameter.numel()
        for name, parameter in handle.model.named_parameters()
        if not name.startswith("pra_")
    )
    router_parameters = int(projection.parameter_count)
    train_examples = [
        example
        for example in load_split_examples(args.cache_dir, args.train_examples, 0, args.data_seed)
        if example["dataset"] == "hotpotqa"
    ]
    validation_examples = load_split_examples(
        args.cache_dir,
        args.validation_examples,
        args.validation_offset,
        args.data_seed,
    )
    identities = {
        "train": [row["id"] for row in train_examples],
        "validation": [row["id"] for row in validation_examples],
    }
    if set(identities["train"]) & set(identities["validation"]):
        raise AssertionError("Train and validation identities must be disjoint.")

    print("preparing train and validation references", flush=True)
    train_records = _prepare_records(handle, tokenizer, train_examples, layers, args, controls=False)
    validation_records = _prepare_records(
        handle, tokenizer, validation_examples, layers, args, controls=True
    )
    _configure_variant(handle, Variant("fixed"), reset=True)
    validation_controls, validation_control_rows = _context_controls(
        handle, tokenizer, validation_records, layers, args.new_tokens, device
    )
    reference_record = validation_records[0]
    off_reference = {
        "metrics": validation_controls[reference_record["example"]["id"]]["none"],
        "generation": validation_controls[reference_record["example"]["id"]]["none"]["generated_answer"],
    }

    training_reports: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    screen_records: list[dict[str, Any]] = []
    all_configs: dict[str, SweepConfig] = {}
    for config in stage_a_configs():
        all_configs[config.config_id] = config
        report, rows = _run_candidate(
            handle=handle,
            tokenizer=tokenizer,
            train_records=train_records,
            validation_records=validation_records,
            validation_controls=validation_controls,
            layers=layers,
            route_layer=route_layer,
            config=config,
            seed=args.screen_seed,
            args=args,
            device=device,
            base_parameters=base_parameters,
            router_parameters=router_parameters,
            reference_record=reference_record,
            off_reference=off_reference,
        )
        training_reports.append(report)
        screen_rows.extend(rows)
        record = _summary_record(config, rows, [report])
        screen_records.append(record)
        print(f"stage A {config.config_id} score={record['combined_oracle_delta_logp']:.4f}", flush=True)

    selected_ranks = top_stage_a_ranks(screen_records, args.stage_b_ranks)
    stage_b = stage_b_configs(selected_ranks)
    for config in stage_b:
        if config.config_id in all_configs:
            continue
        all_configs[config.config_id] = config
        report, rows = _run_candidate(
            handle=handle,
            tokenizer=tokenizer,
            train_records=train_records,
            validation_records=validation_records,
            validation_controls=validation_controls,
            layers=layers,
            route_layer=route_layer,
            config=config,
            seed=args.screen_seed,
            args=args,
            device=device,
            base_parameters=base_parameters,
            router_parameters=router_parameters,
            reference_record=reference_record,
            off_reference=off_reference,
        )
        training_reports.append(report)
        screen_rows.extend(rows)
        record = _summary_record(config, rows, [report])
        screen_records.append(record)
        print(f"stage B {config.config_id} score={record['combined_oracle_delta_logp']:.4f}", flush=True)

    finalist_ids = select_stage_c_configs(screen_records, args.finalists)
    finalist_configs = [all_configs[config_id] for config_id in finalist_ids]
    validation_finalist_rows: list[dict[str, Any]] = []
    for config in finalist_configs:
        for seed in args.seeds:
            if seed == args.screen_seed:
                matching = [
                    row
                    for row in screen_rows
                    if row["variant"] == config.config_id and row["seed"] == seed
                ]
                if matching:
                    validation_finalist_rows.extend(matching)
                    continue
            report, rows = _run_candidate(
                handle=handle,
                tokenizer=tokenizer,
                train_records=train_records,
                validation_records=validation_records,
                validation_controls=validation_controls,
                layers=layers,
                route_layer=route_layer,
                config=config,
                seed=seed,
                args=args,
                device=device,
                base_parameters=base_parameters,
                router_parameters=router_parameters,
                reference_record=reference_record,
                off_reference=off_reference,
            )
            training_reports.append(report)
            validation_finalist_rows.extend(rows)
            print(f"stage C validation {config.config_id} seed={seed}", flush=True)

    finalist_records = [
        _summary_record(
            config,
            [row for row in validation_finalist_rows if row["variant"] == config.config_id],
            training_reports,
        )
        for config in finalist_configs
    ]
    winner_record = choose_pareto_winner(finalist_records, args.pareto_tolerance)
    winner = all_configs[str(winner_record["config_id"])]
    validation_seed_scores = _per_seed_validation_score(
        validation_finalist_rows, winner.config_id
    )
    selected_seed = max(validation_seed_scores, key=lambda seed: (validation_seed_scores[seed], -seed))
    print(
        f"validation selected {winner.config_id} seed={selected_seed} "
        f"score={winner_record['combined_oracle_delta_logp']:.4f}",
        flush=True,
    )

    combo = SweepConfig(winner.rank, winner.steps, winner.learning_rate, "combo")
    combo_variant = Variant(
        f"combo_residual_32_{winner.config_id}", residual_width=32, lora_rank=winner.rank
    )
    combo_config = SweepConfig(winner.rank, winner.steps, winner.learning_rate, "combo")
    combo_checkpoint_paths: dict[int, Path] = {}
    combo_validation_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        _freeze_backbone(handle)
        report, checkpoint_bytes = _train_variant(
            handle,
            train_records,
            layers,
            combo_variant,
            seed,
            combo.steps,
            combo.learning_rate,
            device,
            args.output_dir / "checkpoints" / f"{combo_variant.name}_seed{seed}.pt",
        )
        report.update(
            {
                "config_id": combo_variant.name,
                "rank": combo.rank,
                "stage": "combo",
                "step_multiplier": combo.step_multiplier,
                "learning_rate_multiplier": combo.learning_rate_multiplier,
                "epochs_equivalent": combo.steps / len(train_records),
                "examples_seen": combo.steps,
                "early_stopping": False,
                "checkpoint_bytes": checkpoint_bytes,
            }
        )
        report["pra_off_exact"] = _verify_off_exact(
            handle, tokenizer, reference_record, layers, off_reference, device, args.new_tokens
        )
        if not report["pra_off_exact"]:
            raise AssertionError(f"PRA-off exactness failed for {combo_variant.name}, seed {seed}.")
        training_reports.append(report)
        combo_checkpoint_paths[seed] = args.output_dir / "checkpoints" / f"{combo_variant.name}_seed{seed}.pt"
        combo_validation_rows.extend(
            _evaluate_memory(
                handle,
                tokenizer,
                validation_records,
                validation_controls,
                layers,
                route_layer,
                combo_variant,
                seed,
                args.new_tokens,
                device,
                base_parameters,
                router_parameters,
                checkpoint_bytes,
                args.recovery_epsilon,
            )
        )
        print(f"combo validation seed={seed}", flush=True)

    del validation_records
    handle.cache.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("selection frozen; preparing untouched test references", flush=True)
    test_examples = load_split_examples(
        args.cache_dir, args.test_examples, args.test_offset, args.data_seed
    )
    identities["test"] = [row["id"] for row in test_examples]
    if (
        set(identities["train"]) & set(identities["test"])
        or set(identities["validation"]) & set(identities["test"])
    ):
        raise AssertionError("Test identities must be disjoint from train and validation.")
    test_records = _prepare_records(handle, tokenizer, test_examples, layers, args, controls=True)
    _configure_variant(handle, Variant("fixed"), reset=True)
    test_controls, test_control_rows = _context_controls(
        handle, tokenizer, test_records, layers, args.new_tokens, device
    )
    _configure_variant(handle, Variant("fixed"), reset=True)
    fixed_test_rows = _evaluate_memory(
        handle,
        tokenizer,
        test_records,
        test_controls,
        layers,
        route_layer,
        Variant("fixed"),
        0,
        args.new_tokens,
        device,
        base_parameters,
        router_parameters,
        0,
        args.recovery_epsilon,
    )
    test_rows = list(fixed_test_rows)
    for config in finalist_configs:
        for seed in args.seeds:
            test_rows.extend(
                _evaluate_checkpoint(
                    handle=handle,
                    tokenizer=tokenizer,
                    records=test_records,
                    controls=test_controls,
                    layers=layers,
                    route_layer=route_layer,
                    config=config,
                    seed=seed,
                    args=args,
                    device=device,
                    base_parameters=base_parameters,
                    router_parameters=router_parameters,
                )
            )
            print(f"test {config.config_id} seed={seed}", flush=True)
    for seed in args.seeds:
        _freeze_backbone(handle)
        _configure_variant(handle, combo_variant, reset=True)
        _, checkpoint_bytes = _load_checkpoint(
            combo_checkpoint_paths[seed], handle, combo_variant
        )
        test_rows.extend(
            _evaluate_memory(
                handle,
                tokenizer,
                test_records,
                test_controls,
                layers,
                route_layer,
                combo_variant,
                seed,
                args.new_tokens,
                device,
                base_parameters,
                router_parameters,
                checkpoint_bytes,
                args.recovery_epsilon,
            )
        )
        print(f"test combo seed={seed}", flush=True)

    test_seed_rows = _seed_aggregates(test_rows)
    test_aggregates = _aggregates(test_seed_rows)
    winner_test_aggregate = _aggregates(
        _seed_aggregates([row for row in test_rows if row["variant"] == winner.config_id])
    )
    combo_id = combo_variant.name
    paired = _paired_summary(test_rows, combo_id, winner.config_id)
    exactness = {
        "all_candidates_exact": all(row.get("pra_off_exact", False) for row in training_reports),
        "candidate_checks": len(training_reports),
        "logit_and_generation_check": "exact compact answer logits and greedy output per candidate",
        "adapter_bypass": "conditional branch is unreachable when PRA is disabled",
        "cache_check": "tests/test_hf_integration.py exact disabled cache parity",
        "native_limit_violations": handle.native_limit_violations,
    }
    sdk = _package_sdk_artifacts(
        args=args,
        winner=winner,
        selected_seed=selected_seed,
        layers=layers,
        validation_record=winner_record,
        test_record={"aggregates": winner_test_aggregate},
    )
    artifact = {
        "runtime": runtime_metadata(),
        "manifest": manifest,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model_parameters": base_parameters,
        "router_parameters": router_parameters,
        "layers": list(layers),
        "identities": identities,
        "identity_disjoint": True,
        "stage_a_selected_ranks": selected_ranks,
        "stage_c_finalists": finalist_ids,
        "pareto_winner": winner_record,
        "selected_artifact_seed": selected_seed,
        "screen_ranking": rank_screen_records(screen_records),
        "finalist_validation_ranking": rank_screen_records(finalist_records),
        "training_reports": training_reports,
        "validation_control_rows": validation_control_rows,
        "screen_validation_rows": screen_rows,
        "finalist_validation_rows": validation_finalist_rows,
        "combo_validation_rows": combo_validation_rows,
        "test_control_rows": test_control_rows,
        "test_rows": test_rows,
        "test_seed_aggregates": test_seed_rows,
        "test_aggregates": test_aggregates,
        "combo_paired_effects": paired,
        "exactness": exactness,
        "sdk_artifacts": sdk,
        "scope_note": "One-shot HotpotQA is an adaptation stress test; iterative retrieval remains Paper 2.5 scope.",
    }
    (args.output_dir / "overnight_lora_sweep.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_flat_csv(
        args.output_dir / "validation_ranking.csv", rank_screen_records(screen_records)
    )
    _write_flat_csv(
        args.output_dir / "finalist_validation.csv", rank_screen_records(finalist_records)
    )
    _write_csv(args.output_dir / "test_finalists.csv", test_aggregates)
    _write_csv(args.output_dir / "test_five_seed.csv", test_seed_rows)
    _write_csv(args.output_dir / "recovery_ratios.csv", test_aggregates)
    _write_flat_csv(args.output_dir / "pra_off_exactness.csv", [exactness])
    _write_flat_csv(args.output_dir / "combo_paired.csv", paired)
    _plot_pareto(screen_records, finalist_records, winner.config_id, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    results = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--screen-seed", type=int, default=SCREEN_SEED)
    parser.add_argument("--stage-b-ranks", type=int, default=3)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--pareto-tolerance", type=float, default=0.20)
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument("--train-examples", type=int, default=12)
    parser.add_argument("--validation-examples", type=int, default=4)
    parser.add_argument("--validation-offset", type=int, default=12)
    parser.add_argument("--test-examples", type=int, default=8)
    parser.add_argument("--test-offset", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--full-context-tokens", type=int, default=2048)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--recovery-epsilon", type=float, default=0.05)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=results / "routing" / "learned_adapter" / "checkpoints" / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=results / "overnight_lora_sweep")
    parser.add_argument(
        "--output-router",
        type=Path,
        default=ROOT / "artifacts" / "pra_hf" / "routers" / "qwen3-0.6b-joint-d128",
    )
    parser.add_argument(
        "--output-memory-adapter",
        type=Path,
        default=ROOT / "artifacts" / "pra_hf" / "memory_adapters" / "qwen3-0.6b-last14-lora",
    )
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    if args.screen_seed not in args.seeds:
        parser.error("--screen-seed must be included in --seeds")
    if not 1 <= args.stage_b_ranks <= len(RANKS):
        parser.error("--stage-b-ranks must be between 1 and 4")
    if not 2 <= args.finalists <= 3:
        parser.error("--finalists must be 2 or 3")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "pareto_winner": result["pareto_winner"],
                "selected_artifact_seed": result["selected_artifact_seed"],
                "sdk_artifacts": result["sdk_artifacts"],
                "exactness": result["exactness"],
            },
            indent=2,
            sort_keys=True,
        )
    )
