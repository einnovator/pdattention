"""Prepare the exact-slot Gemma Paper 4 training gate without launching training."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import torch
from transformers import Gemma3TextConfig

from pra_hf.pra_aware_training import gemma_layer_topology


DEFAULT_MODEL_ID = "google/gemma-3-1b-it"
DEFAULT_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lora_parameter_estimates(config, global_layers, rank):
    hidden = int(config.hidden_size)
    q_width = int(config.num_attention_heads * config.head_dim)
    kv_width = int(config.num_key_value_heads * config.head_dim)
    intermediate = int(config.intermediate_size)
    q = rank * (hidden + q_width)
    k_or_v = rank * (hidden + kv_width)
    o = rank * (q_width + hidden)
    mlp = rank * ((hidden + intermediate) * 2 + (intermediate + hidden))
    consumer = len(global_layers) * (q + o + mlp)
    interface = len(global_layers) * (q + k_or_v * 2 + o + mlp)
    broad = int(config.num_hidden_layers) * (q + k_or_v * 2 + o + mlp)
    return {
        "consumer_lora": consumer,
        "interface_lora": interface,
        "broad_lora": broad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/papers/shared/results/paper4_training/gemma_gate"),
    )
    parser.add_argument("--base-parameters", type=int, default=999_885_955)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = Gemma3TextConfig.from_pretrained(args.config, local_files_only=True)
    topology = gemma_layer_topology(config, "all_global")
    global_layers = tuple(row.layer_index for row in topology if row.pra_enabled)
    topology_rows = [row.__dict__ for row in topology]
    (args.output_dir / "gemma_layer_topology.json").write_text(
        json.dumps(
            {
                "model_id": DEFAULT_MODEL_ID,
                "revision": DEFAULT_REVISION,
                "placement": "all_global",
                "layers": topology_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    parameter_rows = []
    for rank in (8, 32):
        for regime, count in lora_parameter_estimates(config, global_layers, rank).items():
            parameter_rows.append(
                {
                    "regime": regime,
                    "rank": rank,
                    "estimated_trainable_parameters": count,
                    "estimated_trainable_fraction": count / args.base_parameters,
                    "target_layers": ",".join(map(str, global_layers)) if regime != "broad_lora" else "all",
                    "status": "estimate_verified_by_unit_scope_tests_not_loaded_model_count",
                }
            )
    write_csv(args.output_dir / "gemma_lora_parameter_estimates.csv", parameter_rows)

    matrix = f"""# Gemma checkpoint matrix

Pinned checkpoint: `{DEFAULT_MODEL_ID}` at `{DEFAULT_REVISION}`.

| Stage | Architecture | Trainable scope | Status |
|---|---|---|---|
| G0 | native local/global Gemma | none (evaluation) | pending measured baseline |
| G1 | global slots {global_layers} replaced by PRA | none | pending measured frozen baseline |
| G2 | exact-slot Gemma-PRA | Q/O + following MLP LoRA | implementation ready; training not launched |
| G3 | exact-slot Gemma-PRA | Q/K/V/O + following MLP LoRA | implementation ready; training not launched |
| G4 | exact-slot Gemma-PRA | broad decoder LoRA | implementation ready; training not launched |
| G5 | exact-slot Gemma-PRA | full weight | blocked on benchmark/distributed budget |
| G6 | Gemma-like PRA native | full scratch | gated on G5; not launched |
"""
    (args.output_dir / "gemma_checkpoint_matrix.md").write_text(matrix, encoding="utf-8")

    curriculum = {
        "ordinary_language_modeling": 0.40,
        "local_only": 0.15,
        "remote_memory": 0.20,
        "multi_hop_remote_memory": 0.15,
        "distractor_rich_memory": 0.10,
        "selection_policy": "fixed-oracle for causal consumer controls plus inherited router for non-oracle rows",
        "required_controls": [
            "no_memory",
            "matched_distractor",
            "evidence_only",
            "whole_parent",
            "shuffled_memory",
        ],
    }
    (args.output_dir / "gemma_training_curriculum.json").write_text(
        json.dumps(curriculum, indent=2), encoding="utf-8"
    )

    pending_files = (
        "gemma_native_baseline.csv",
        "gemma_frozen_pra.csv",
        "gemma_lora_consumer.csv",
        "gemma_lora_interface.csv",
        "gemma_lora_broad.csv",
        "gemma_full_weight_training.csv",
        "gemma_memory_modularity.csv",
        "gemma_global_kv_accounting.csv",
        "gemma_logical_context_scaling.csv",
        "gemma_quality_retention.csv",
    )
    for name in pending_files:
        write_csv(
            args.output_dir / name,
            [
                {
                    "status": "pending_100_to_500_step_benchmark_and_distributed_configuration",
                    "model_id": DEFAULT_MODEL_ID,
                    "revision": DEFAULT_REVISION,
                    "global_layers": ",".join(map(str, global_layers)),
                }
            ],
        )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    # Adam full tuning normally needs weights, gradients, FP32 moments, and
    # often master weights before activation/checkpoint overhead.
    lower_bound_bytes = args.base_parameters * (2 + 2 + 8)
    compute_gate = f"""# Gemma 3 1B compute gate

## Architecture audit

- Checkpoint: `{DEFAULT_MODEL_ID}` (`{DEFAULT_REVISION}`)
- Decoder layers: {config.num_hidden_layers}
- Native local window: {config.sliding_window}
- Exact native-global/PRA slots: {global_layers}
- Local layers remain unchanged.

## Local device

- Device: {device}
- Python: {platform.python_version()}
- PyTorch: {torch.__version__}
- CUDA available: {torch.cuda.is_available()}

The 1B checkpoint contains approximately {args.base_parameters:,} parameters.
Even a lower-bound mixed-precision Adam accounting (FP16 weights and gradients,
FP32 moments) is {lower_bound_bytes / 2**30:.1f} GiB before activations, temporary
buffers, reference K/V, and checkpoint staging. The local 4 GiB GPU therefore
cannot provide the required 100--500-step full-weight benchmark safely.

## Decision

Do not launch G2--G5 training in this checkpoint. Configure distributed or
larger-memory training first, then run 100--500 measured steps and record
tokens/s, peak device memory, optimizer state, forward/backward time, dataloader
fraction, and checkpoint size. Extrapolate each full schedule before approval.
"""
    (args.output_dir / "gemma_compute_gate.md").write_text(compute_gate, encoding="utf-8")
    findings = {
        "status": "architecture_and_adaptation_scopes_ready_training_not_started",
        "model_id": DEFAULT_MODEL_ID,
        "revision": DEFAULT_REVISION,
        "native_global_layers": list(global_layers),
        "local_layers_preserved": True,
        "long_run_gate": "requires_distributed_or_larger_memory_configuration",
        "reason": "local 4 GiB GPU is below full-weight optimizer plus activation requirements",
    }
    (args.output_dir / "gemma_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
