"""Evaluate retention and long-range strata from completed Tier 0 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pra_torch.controlled_local_sa import ControlledTokenizer, controlled_examples
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra
from pra_torch.pra_aware_training import install_adaptation_regime

from .run_tier0 import (
    PRA_REGIMES,
    evaluate_pra_condition,
    evaluate_sa,
    model_config,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/papers/shared/results/paper4_training/tier0"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-examples", type=int, default=512)
    args = parser.parse_args()
    config = json.loads((args.output_dir / "paper4_architecture_configs.json").read_text(encoding="utf-8"))
    tokenizer = ControlledTokenizer()
    test_data = controlled_examples(tokenizer, count=args.test_examples, seed=900_001)
    checkpoints = args.output_dir / "checkpoints"
    pra_layers = tuple(config["pra_layers"])
    rows = []
    for seed in config["seeds"]:
        local_cfg = model_config(
            tokenizer,
            window=config["local_window"],
            variant="td_sa",
            d_model=config["d_model"],
            layers=config["layers"],
            pra_layers=pra_layers,
            device=args.device,
        )
        target_cfg = model_config(
            tokenizer,
            window=config["local_window"],
            variant="td_layered_pra",
            d_model=config["d_model"],
            layers=config["layers"],
            pra_layers=pra_layers,
            device=args.device,
        )
        local = TinyPRAModel(local_cfg).to(args.device)
        local.load_state_dict(
            torch.load(checkpoints / f"local_sa_seed{seed}.pt", map_location=args.device, weights_only=True)["model"]
        )
        for regime in PRA_REGIMES:
            model = (
                TinyPRAModel(target_cfg).to(args.device)
                if regime == "native_scratch"
                else convert_sa_model_to_pra(local, target_cfg).to(args.device)
            )
            install_adaptation_regime(
                model,
                regime,
                lora_rank=config["lora_rank"],
                lora_alpha=2 * config["lora_rank"],
            )
            model.load_state_dict(
                torch.load(checkpoints / f"{regime}_seed{seed}.pt", map_location=args.device, weights_only=True)["model"]
            )
            full = evaluate_sa(
                model,
                test_data,
                tokenizer,
                batch_size=args.batch_size,
                device=args.device,
            )
            rows.append(
                {
                    "model": regime,
                    "seed": seed,
                    "depth": "all",
                    "condition": "full_context_no_memory",
                    **{key: value for key, value in full.items() if key != "condition"},
                }
            )
            for depth in (1, 2, 3, 4):
                subset = [example for example in test_data if example.depth == depth]
                for condition in ("matched_distractor", "evidence_only"):
                    result = evaluate_pra_condition(
                        model,
                        subset,
                        tokenizer,
                        condition=condition,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                    rows.append(
                        {
                            "model": regime,
                            "seed": seed,
                            "depth": depth,
                            **{
                                key: value
                                for key, value in result.items()
                                if key not in {"hidden_states", "layer_profiles"}
                            },
                        }
                    )
    write_csv(args.output_dir / "retention_and_depth_results.csv", rows)

    figure, axis = plt.subplots(figsize=(7.5, 4.3))
    for regime in PRA_REGIMES:
        means = []
        for depth in (1, 2, 3, 4):
            samples = [
                float(row["accuracy"])
                for row in rows
                if row["model"] == regime
                and row["condition"] == "evidence_only"
                and row["depth"] == depth
            ]
            means.append(sum(samples) / len(samples))
        axis.plot((1, 2, 3, 4), means, marker="o", label=regime.replace("_", " "))
    axis.set(xlabel="Associative-chain depth", ylabel="Evidence-only answer accuracy", xticks=(1, 2, 3, 4), ylim=(0, 1))
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figures = args.output_dir / "figures"
    figure.savefig(figures / "accuracy_by_chain_depth.png", dpi=180)
    figure.savefig(figures / "accuracy_by_chain_depth.pdf")
    plt.close(figure)
    print(args.output_dir / "retention_and_depth_results.csv")


if __name__ == "__main__":
    main()
