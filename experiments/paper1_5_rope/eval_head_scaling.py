"""Evaluate bounded implicit-head PRA as logical distance exceeds native context."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from data.datasets import SyntheticNativeKVFixedTargetDataset  # noqa: E402
from data.tokenizer import PRATokenizer  # noqa: E402
from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    TIERS,
    environment_metadata,
    refresh_manifest,
    set_seed,
    write_csv,
    write_json,
)
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402
from pra_torch.prompt import prepare_prompt_batch_for_pra  # noqa: E402


def _load(tier: str, mode: str, seed: int, device: str):
    path = REPO / "out" / "paper1_5_rope" / tier / mode / f"seed-{seed}" / "checkpoint.pt"
    checkpoint = torch.load(path, map_location=device)
    source_cfg = PRAConfig(**{**checkpoint["cfg"], "device": device})
    source = TinyPRAModel(source_cfg).to(device)
    source.load_state_dict(checkpoint["model"])
    target_cfg = PRAConfig(
        **{
            **source_cfg.__dict__,
            "model_variant": "td_pra",
            "memory_transport": "native_kv",
            "dropout": 0.0,
            "max_prompt_direct_tokens": 32,
            "prompt_overflow_mode": "implicit_reference",
            "prompt_position_mode": "historical",
            "reference_position_mode": "global",
            "chunking_mode": "fixed",
            "fixed_chunk_tokens": 32,
            "max_prompt_gists": 128,
            "max_gists_per_reference": 128,
            "top_k_references": 1,
            "top_k_chunks_per_reference": 8,
            "trigger_threshold": float("-inf"),
            "max_materialized_memory_tokens": 160,
            "context_safety_reserve_tokens": 0,
        }
    )
    converted = convert_sa_model_to_pra(source.eval(), target_cfg).to(device).eval()
    return source.eval(), converted, PRATokenizer.from_vocab(checkpoint["stoi"])


def _examples(max_examples: int):
    root = REPO / "out" / "native_kv_data" / "synthetic" / "split-2"
    return list(SyntheticNativeKVFixedTargetDataset(root))[:max_examples]


def _prompt_ids(sample, tokenizer, total_tokens: int) -> list[int]:
    row = sample.metadata["row"]
    source = tokenizer.encode(str(row["source_text"]))
    query = tokenizer.encode(sample.question)
    filler_pattern = tokenizer.encode(" neutral filler ") or [1]
    filler_count = max(total_tokens - len(source) - len(query), 0)
    filler = (filler_pattern * math.ceil(filler_count / len(filler_pattern)))[:filler_count]
    ids = [*source, *filler, *query]
    if len(ids) > total_tokens:
        # Preserve the local query and the nearest available historical source suffix.
        ids = ids[-total_tokens:]
    return ids


def evaluate(tier: str, mode: str, seed: int, device: str, max_examples: int):
    set_seed(seed)
    source, model, tokenizer = _load(tier, mode, seed, device)
    rows = []
    native_length = model.cfg.effective_model_max_context_tokens
    for ratio in (1, 2, 4, 8):
        for sample in _examples(max_examples):
            ids = _prompt_ids(sample, tokenizer, ratio * native_length)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            answer_id = tokenizer.encode(sample.answer.strip())[0]
            prepared = prepare_prompt_batch_for_pra(model, tokenizer, input_ids)
            model.set_pra_cache(prepared.caches[0])
            with torch.no_grad():
                logits = model(
                    prepared.input_ids,
                    attention_mask=prepared.attention_mask,
                    position_offset=prepared.position_offsets,
                )
                diagnostics = model.pra_diagnostics_by_layer()
                disabled = model(
                    prepared.input_ids,
                    use_pra_memory=False,
                    attention_mask=prepared.attention_mask,
                    position_offset=prepared.position_offsets,
                )
            prediction = logits[0, -1]
            disabled_prediction = disabled[0, -1]
            materialized = max(
                (
                    value.get("memory_tokens_materialized", 0.0)
                    for value in diagnostics.values()
                ),
                default=0.0,
            )
            entry = prepared.caches[0].get("pra://implicit/prompt/head")
            max_native = max(
                int(entry.metadata.get("max_encoding_input_tokens", 0)) if entry else 0,
                int(prepared.input_ids.shape[1]),
            )
            rows.append(
                {
                    "seed": seed,
                    "model_tier": tier,
                    "position_mode": mode,
                    "example_id": sample.id,
                    "logical_native_ratio": ratio,
                    "logical_context": len(ids),
                    "native_context": native_length,
                    "maximum_native_operation": max_native,
                    "native_limit_violations": int(max_native > native_length),
                    "direct_tokens": int(prepared.input_ids.shape[1]),
                    "implicit_head_tokens": len(ids) - int(prepared.input_ids.shape[1]),
                    "materialized_memory_tokens": materialized,
                    "loss": float(F.cross_entropy(prediction[None], torch.tensor([answer_id], device=device))),
                    "accuracy": int(prediction.argmax().item() == answer_id),
                    "disabled_loss": float(
                        F.cross_entropy(
                            disabled_prediction[None],
                            torch.tensor([answer_id], device=device),
                        )
                    ),
                    "disabled_accuracy": int(disabled_prediction.argmax().item() == answer_id),
                }
            )
    return rows


def _plot(rows: list[dict], path: Path):
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    colors = {"absolute": "#245A8D", "rope": "#A34832"}
    markers = {"tiny": "o", "small": "s"}
    for tier in ("tiny", "small"):
        for mode in ("absolute", "rope"):
            subset = [
                row for row in rows
                if row["model_tier"] == tier and row["position_mode"] == mode
            ]
            grouped = {}
            for row in subset:
                grouped.setdefault(row["logical_native_ratio"], []).append(row)
            x = sorted(grouped)
            loss = [sum(r["loss"] for r in grouped[v]) / len(grouped[v]) for v in x]
            gain = [
                sum(r["disabled_loss"] - r["loss"] for r in grouped[v]) / len(grouped[v])
                for v in x
            ]
            label = f"{tier} {mode}"
            style = "-" if tier == "small" else "--"
            axes[0].plot(x, loss, color=colors[mode], marker=markers[tier], linestyle=style, label=label)
            axes[1].plot(x, gain, color=colors[mode], marker=markers[tier], linestyle=style, label=label)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Logical / native context")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Implicit-head answer loss")
    axes[1].set_ylabel("Loss benefit over disabled memory")
    axes[1].axhline(0, color="#555555", linewidth=0.8, linestyle=":")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args):
    rows = []
    for tier in args.tiers:
        for mode in ("absolute", "rope"):
            for seed in args.seeds:
                rows.extend(evaluate(tier, mode, seed, args.device, args.max_examples))
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    write_json(
        RESULTS / "rope_head_scaling.json",
        {"metadata": environment_metadata(), "rows": rows},
    )
    write_csv(RESULTS / "rope_head_scaling.csv", rows)
    _plot(rows, RESULTS / "rope_head_scaling.png")
    refresh_manifest()
    return RESULTS / "rope_head_scaling.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-examples", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
