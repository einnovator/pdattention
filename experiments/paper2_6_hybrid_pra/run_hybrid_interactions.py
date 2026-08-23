"""Cross Paper 2.6 discovery channels with chunk, layer, and gist budgets.

The cohort is identity-disjoint from the primary confirmation.  This is a
retrieval interaction experiment: it deliberately does not claim downstream
generation effects, which are measured by ``run_end_to_end_confirmation``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_6_hybrid_pra.run_end_to_end_confirmation import (  # noqa: E402
    CHANNELS,
    _bounded_source,
    _fresh_examples,
    _load_model,
    _prompt,
    _route_channels,
)
from experiments.paper2_hf.qa.run_smoke import evidence_token_spans  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_checkpoint(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["model"],
                row["dataset"],
                row["chunk_tokens"],
                row["routing_layer_offset"],
                row["chunk_budget"],
                row["channel"],
            )
        ].append(row)
    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                "model": key[0],
                "dataset": key[1],
                "chunk_tokens": key[2],
                "routing_layer_offset": key[3],
                "chunk_budget": key[4],
                "channel": key[5],
                "examples": len(values),
                "evidence_recall": statistics.fmean(
                    float(row["evidence_recall"]) for row in values
                ),
                "precision": statistics.fmean(float(row["precision"]) for row in values),
                "mrr": statistics.fmean(float(row["mrr"]) for row in values),
                "complete_recovery": statistics.fmean(
                    float(row["complete_recovery"]) for row in values
                ),
                "scored_candidate_fraction": statistics.fmean(
                    float(row["scored_candidates"])
                    / max(
                        float(row["candidate_chunks"])
                        * float(row["token_index_queries"]),
                        1.0,
                    )
                    for row in values
                ),
            }
        )
    return output


def _plot(summary: list[dict], output: Path) -> None:
    models = sorted({row["model"] for row in summary})
    datasets = sorted({row["dataset"] for row in summary})
    figure, axes = plt.subplots(
        len(models), len(datasets), figsize=(6.0 * len(datasets), 4.2 * len(models)), squeeze=False
    )
    for model_index, model in enumerate(models):
        for dataset_index, dataset in enumerate(datasets):
            axis = axes[model_index][dataset_index]
            values = [
                row
                for row in summary
                if row["model"] == model
                and row["dataset"] == dataset
                and int(row["routing_layer_offset"]) == -1
                and int(row["chunk_tokens"]) * int(row["chunk_budget"]) == 128
            ]
            for channel in CHANNELS:
                channel_rows = [row for row in values if row["channel"] == channel]
                if not channel_rows:
                    continue
                channel_rows.sort(key=lambda row: int(row["chunk_tokens"]))
                axis.plot(
                    [int(row["chunk_tokens"]) for row in channel_rows],
                    [float(row["evidence_recall"]) for row in channel_rows],
                    marker="o",
                    label=channel,
                )
            axis.set(
                title=f"{model} / {dataset} (128 routed tokens)",
                xlabel="Routing chunk tokens",
                ylabel="Evidence recall",
                ylim=(-0.03, 1.03),
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"hybrid_interactions.{suffix}", dpi=190)
    plt.close(figure)


def _plot_budget(summary: list[dict], output: Path) -> None:
    models = sorted({row["model"] for row in summary})
    datasets = sorted({row["dataset"] for row in summary})
    figure, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(6.0 * len(datasets), 4.2 * len(models)),
        squeeze=False,
    )
    for model_index, model in enumerate(models):
        for dataset_index, dataset in enumerate(datasets):
            axis = axes[model_index][dataset_index]
            values = [
                row for row in summary
                if row["model"] == model
                and row["dataset"] == dataset
                and int(row["routing_layer_offset"]) == -1
                and int(row["chunk_tokens"]) == 32
            ]
            for channel in CHANNELS:
                channel_rows = [row for row in values if row["channel"] == channel]
                channel_rows.sort(key=lambda row: int(row["chunk_budget"]))
                axis.plot(
                    [int(row["chunk_budget"]) for row in channel_rows],
                    [float(row["evidence_recall"]) for row in channel_rows],
                    marker="o", label=channel,
                )
            axis.set(
                title=f"{model} / {dataset} (32-token chunks)",
                xlabel="Selected chunk budget",
                ylabel="Evidence recall",
                ylim=(-0.03, 1.03),
                xticks=(2, 4, 8),
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"hybrid_budget_interactions.{suffix}", dpi=190)
    plt.close(figure)


def run(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "hybrid_interaction_checkpoint_budget_reachable.jsonl"
    examples = _fresh_examples(
        args.cache_dir, args.examples_per_dataset, args.offset, args.seed
    )
    rows = _read_checkpoint(checkpoint)
    completed = {
        (
            row["model"],
            row["dataset"],
            row["example_id"],
            int(row["chunk_tokens"]),
            int(row["routing_layer_offset"]),
            int(row["chunk_budget"]),
            row["channel"],
        )
        for row in rows
    }
    model_manifest = {}
    for model_name in args.models:
        for chunk_tokens in args.chunk_tokens:
            args.chunk_tokens_current = chunk_tokens
            # _load_model reads chunk_tokens; keep the public argument immutable in outputs.
            original = args.chunk_tokens
            args.chunk_tokens = chunk_tokens
            model_id, revision, tokenizer, pra, layers = _load_model(model_name, args)
            args.chunk_tokens = original
            model_manifest[model_name] = {"model_id": model_id, "revision": revision}
            layer_choices = tuple(dict.fromkeys((layers[-1], layers[0])))
            for example_index, raw in enumerate(examples, 1):
                example = _bounded_source(tokenizer, raw, args.source_tokens)
                pra.clear_references()
                pra.add_reference(
                    example["source"], uri=f"interaction://{raw['dataset']}/{raw['id']}"
                )
                spans = evidence_token_spans(
                    tokenizer, example["source"], example["evidence"]
                )
                prompt_ids, prompt_mask, _ = _prompt(
                    tokenizer, example["question"], max_tokens=args.prompt_tokens
                )
                for layer in layer_choices:
                    offset = layer - int(pra.model.config.num_hidden_layers)
                    for budget in args.chunk_budgets:
                        pending = [
                            channel
                            for channel in CHANNELS
                            if (
                                model_name,
                                raw["dataset"],
                                raw["id"],
                                chunk_tokens,
                                offset,
                                budget,
                                channel,
                            )
                            not in completed
                        ]
                        if not pending:
                            continue
                        _, local = _route_channels(
                            pra,
                            tokenizer,
                            example,
                            prompt_ids,
                            prompt_mask,
                            spans,
                            routing_layer=layer,
                            chunk_budget=budget,
                        )
                        for row in local:
                            if row["channel"] not in pending:
                                continue
                            row.update(
                                model=model_name,
                                dataset=raw["dataset"],
                                example_id=raw["id"],
                                chunk_tokens=chunk_tokens,
                                routing_layer=layer,
                                routing_layer_offset=offset,
                                chunk_budget=budget,
                                branch_top_k=max(1, (budget + 1) // 2),
                                beam_size=max(1, (budget + 1) // 2),
                            )
                            rows.append(row)
                            _append_checkpoint(checkpoint, row)
                            completed.add(
                                (
                                    model_name,
                                    raw["dataset"],
                                    raw["id"],
                                    chunk_tokens,
                                    offset,
                                    budget,
                                    row["channel"],
                                )
                            )
                print(
                    f"[{model_name} chunk={chunk_tokens} {example_index}/{len(examples)}] "
                    f"{raw['dataset']} {raw['id']}",
                    flush=True,
                )
            del pra
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    unique = {}
    for row in rows:
        key = (
            row["model"], row["dataset"], row["example_id"],
            int(row["chunk_tokens"]), int(row["routing_layer_offset"]),
            int(row["chunk_budget"]), row["channel"],
        )
        unique[key] = row
    rows = list(unique.values())
    summary = _aggregate(rows)
    _write_csv(args.output / "hybrid_interaction_rows.csv", rows)
    _write_csv(args.output / "hybrid_interaction_summary.csv", summary)
    _plot(summary, args.output)
    _plot_budget(summary, args.output)
    findings = {
        "schema_version": "1.0",
        "models": model_manifest,
        "datasets": sorted({row["dataset"] for row in rows}),
        "fresh_examples_per_dataset": args.examples_per_dataset,
        "offset": args.offset,
        "chunk_tokens": list(args.chunk_tokens),
        "chunk_budgets": list(args.chunk_budgets),
        "budget_to_branch_top_k": {
            str(budget): max(1, (budget + 1) // 2)
            for budget in args.chunk_budgets
        },
        "selected_chunk_counts_by_budget": {
            str(budget): sorted(
                {
                    int(row["selected_chunks"])
                    for row in rows
                    if int(row["chunk_budget"]) == budget
                }
            )
            for budget in args.chunk_budgets
        },
        "matched_physical_token_budget": 128,
        "candidate_pool_saturated": all(
            abs(float(row["scored_candidate_fraction"]) - 1.0) < 1e-9
            for row in summary
        ),
        "scope_note": (
            "This small interaction cohort measures channel, granularity, budget, "
            "and layer geometry. Its bounded candidate pools saturate the token index, "
            "so indexed scaling and latency are reported by the separate systems run."
        ),
        "routing_layer_offsets": sorted({row["routing_layer_offset"] for row in rows}),
        "channels": list(CHANNELS),
        "rows": len(rows),
        "summary": summary,
    }
    (args.output / "hybrid_interaction_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    return findings


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("qwen3_0_6b", "smollm2_135m"), default=("qwen3_0_6b", "smollm2_135m"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--examples-per-dataset", type=int, default=8)
    parser.add_argument("--offset", type=int, default=70)
    parser.add_argument("--chunk-tokens", nargs="+", type=int, default=(16, 32, 64))
    parser.add_argument("--chunk-budgets", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument("--source-tokens", type=int, default=768)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--full-context-tokens", type=int, default=896)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--encoding-tokens", type=int, default=128)
    parser.add_argument("--native-limit", type=int, default=2048)
    parser.add_argument("--materialized-tokens", type=int, default=256)
    parser.add_argument("--consumption-layers", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / "pdattention/data/.hf_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/papers/shared/results/paper2_6_hybrid_pra/interaction_confirmation")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
