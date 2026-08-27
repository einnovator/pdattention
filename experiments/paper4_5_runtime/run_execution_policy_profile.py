"""Profile the implemented Paper 4.5 execution-policy points on a tiny HF LM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from pra_hf import PRAConfig, PRAForCausalLM


SEEDS = (11, 23, 37, 53, 71)


class TinyTokenizer:
    """Stable character tokenizer for an offline mechanism profile."""

    all_special_ids = (0, 1)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        values = [2 + (ord(char) % 61) for char in text]
        if add_special_tokens:
            values.insert(0, 1)
        return SimpleNamespace(input_ids=torch.tensor([values], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(value)) for value in token_ids)


def tiny_model(seed: int):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=96,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=66,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def policy_conditions():
    return {
        "request_shared_last": {
            "routing_layer_policy": "last_pra_layer",
        },
        "request_shared_first": {
            "routing_layer_policy": "first_pra_layer",
        },
        "request_per_layer": {
            "selection_layer_scope": "per_layer",
        },
        "token_shared_last": {
            "selection_stage": "token",
            "selection_layer_scope": "shared",
            "materialization_scope": "token",
            "routing_layer_policy": "last_pra_layer",
        },
        "token_shared_first": {
            "selection_stage": "token",
            "selection_layer_scope": "shared",
            "materialization_scope": "token",
            "routing_layer_policy": "first_pra_layer",
        },
        "token_per_layer": {
            "selection_stage": "token",
            "selection_layer_scope": "per_layer",
            "materialization_scope": "token",
        },
    }


def run(output: Path) -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        base = tiny_model(seed)
        state = base.state_dict()
        baseline = None
        for name, policy in policy_conditions().items():
            model = tiny_model(seed)
            model.load_state_dict(state)
            pra = PRAForCausalLM.from_model(
                model,
                TinyTokenizer(),
                pra_config=PRAConfig(
                    routing_layer=2,
                    consumption_layers=(0, 1, 2),
                    chunk_tokens=8,
                    selected_fraction=None,
                    top_k=2,
                    max_direct_context=24,
                    native_operation_limit=96,
                    max_materialized_tokens=16,
                    context_safety_reserve_tokens=0,
                    encoding_block_tokens=24,
                ),
            )
            pra.add_reference(
                "memory://alpha",
                text="alpha contains key amber; beta points to alpha",
            )
            pra.add_reference(
                "memory://delta",
                text="delta contains key cobalt; epsilon points to delta",
            )
            result = pra.generate(
                "Which key is connected to alpha?",
                max_new_tokens=3,
                return_details=True,
                do_sample=False,
                pra_policy=policy,
            )
            execution = result.stats["pra_execution"]
            selected = tuple(
                (item["reference_uri"], item["chunk_id"])
                for item in result.stats["selected"]
            )
            if name == "request_shared_last":
                baseline = selected
            overlap = len(set(selected) & set(baseline or ())) / max(
                len(set(selected) | set(baseline or ())), 1
            )
            rows.append(
                {
                    "seed": seed,
                    "condition": name,
                    "generation_seconds": result.latency_seconds,
                    "routing_operations": execution["routing_operations"],
                    "selection_epochs": execution["selection_epochs"],
                    "materialization_epochs": execution["materialization_epochs"],
                    "temporal_jaccard": execution["temporal_selection_jaccard_mean"],
                    "layer_jaccard": execution["layer_selection_jaccard_mean"],
                    "baseline_selection_jaccard": overlap,
                    "selected_chunks": len(selected),
                    "output_digest": hashlib.sha256(result.text.encode()).hexdigest()[:12],
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution_policy_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for condition in policy_conditions():
        group = [row for row in rows if row["condition"] == condition]
        summary.append(
            {
                "condition": condition,
                "seeds": len(group),
                "generation_seconds_mean": statistics.fmean(
                    row["generation_seconds"] for row in group
                ),
                "routing_operations_mean": statistics.fmean(
                    row["routing_operations"] for row in group
                ),
                "selection_epochs_mean": statistics.fmean(
                    row["selection_epochs"] for row in group
                ),
                "baseline_selection_jaccard_mean": statistics.fmean(
                    row["baseline_selection_jaccard"] for row in group
                ),
                "temporal_jaccard_mean": statistics.fmean(
                    row["temporal_jaccard"] for row in group
                ),
                "layer_jaccard_mean": statistics.fmean(
                    row["layer_jaccard"] for row in group
                ),
            }
        )
    (output / "execution_policy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    labels = [row["condition"].replace("_", "\n") for row in summary]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].bar(labels, [row["routing_operations_mean"] for row in summary], color="#2b6f77")
    axes[0].set_ylabel("Routing operations / request")
    axes[0].set_title("Measured semantic routing work")
    axes[1].bar(labels, [1000 * row["generation_seconds_mean"] for row in summary], color="#b75d3e")
    axes[1].set_ylabel("Generation wall time (ms)")
    axes[1].set_title("Tiny HF mechanism profile")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "execution_policy_tradeoff.pdf", bbox_inches="tight")
    figure.savefig(output / "execution_policy_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/paper4_5_runtime"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
