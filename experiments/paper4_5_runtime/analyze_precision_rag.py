"""Normalize the matched Qwen3-4B MLX precision RAG comparison."""

from __future__ import annotations

import csv
import gzip
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RAG = ROOT / "docs/papers/shared/results/paper4_5_runtime_productization/rag"
DEFAULT_OUTPUT = (
    ROOT
    / "docs/papers/shared/results/paper4_5_runtime_productization/precision_rag"
)
INPUTS = {
    "MLX-bfloat16": RAG / "multihoprag_l1_qwen3_4b_mlx_bf16_10.json.gz",
    "MLX-8bit": RAG / "multihoprag_l1_qwen3_4b_mlx_8bit_10.json.gz",
    "MLX-4bit": RAG / "multihoprag_l1_qwen3_4b_mlx_10.json.gz",
}
PRECISION_FAMILIES = {
    "MLX-bfloat16": "BF16",
    "MLX-8bit": "INT8",
    "MLX-4bit": "INT4",
}


def _load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def build_comparison() -> dict[str, Any]:
    documents = {encoding: _load(path) for encoding, path in INPUTS.items()}
    first = next(iter(documents.values()))
    invariant_fields = (
        "dataset", "stage", "candidate_counts", "token_budgets", "seed"
    )
    for encoding, document in documents.items():
        mismatches = [
            field for field in invariant_fields if document[field] != first[field]
        ]
        if mismatches:
            raise ValueError(
                f"{encoding} precision cohort mismatch: {', '.join(mismatches)}"
            )
        if document["hardware"] != first["hardware"]:
            raise ValueError(f"{encoding} was not measured on the matched hardware")

    rows: list[dict[str, Any]] = []
    for encoding, document in documents.items():
        family = PRECISION_FAMILIES[encoding]
        for candidate_count in document["candidate_counts"]:
            summary = {
                (row["condition"], row["candidate_count"]): row
                for row in document["summary"]
            }
            baseline = summary[("no_pra", candidate_count)]
            pra = summary[("pra_no_adaptor", candidate_count)]
            condition_rows = {
                condition: [
                    row for row in document["rows"]
                    if row["condition"] == condition
                    and row["candidate_count"] == candidate_count
                    and row["token_budget"] == document["token_budgets"][0]
                ]
                for condition in ("no_pra", "pra_no_adaptor")
            }
            serving = {
                condition: [row["serving_metrics"] for row in values]
                for condition, values in condition_rows.items()
            }
            row = {
                "model": document["model"],
                "model_revision": document["model_revision"],
                "precision_family": family,
                "precision_encoding": encoding,
                "hardware": document["hardware"],
                "engine": "mlx-lm",
                "dataset": document["dataset"],
                "evidence_tier": document["evidence_tier"],
                "seed": document["seed"],
                "examples": baseline["examples"],
                "candidate_count": candidate_count,
                "token_budget": document["token_budgets"][0],
                "no_pra_token_f1": baseline["token_f1"],
                "pra_no_adaptor_token_f1": pra["token_f1"],
                "pra_token_f1_delta": pra["token_f1"] - baseline["token_f1"],
                "no_pra_task_score": baseline["dataset_task_score"],
                "pra_no_adaptor_task_score": pra["dataset_task_score"],
                "pra_task_score_delta": (
                    pra["dataset_task_score"] - baseline["dataset_task_score"]
                ),
                "no_pra_supporting_document_coverage": baseline[
                    "supporting_document_coverage"
                ],
                "pra_supporting_document_coverage": pra[
                    "supporting_document_coverage"
                ],
                "no_pra_visible_tokens": baseline["physical_context_tokens"],
                "pra_visible_tokens": pra["physical_context_tokens"],
                "no_pra_total_latency_ms": baseline["total_latency_ms"],
                "pra_total_latency_ms": pra["total_latency_ms"],
                "pra_total_latency_ratio": (
                    pra["total_latency_ms"] / baseline["total_latency_ms"]
                ),
                "no_pra_ttft_p50_ms": statistics.median(
                    value["ttft_ms"] for value in serving["no_pra"]
                ),
                "pra_ttft_p50_ms": statistics.median(
                    value["ttft_ms"] for value in serving["pra_no_adaptor"]
                ),
                "no_pra_ttft_p95_ms": _percentile(
                    [value["ttft_ms"] for value in serving["no_pra"]], 0.95
                ),
                "pra_ttft_p95_ms": _percentile(
                    [value["ttft_ms"] for value in serving["pra_no_adaptor"]], 0.95
                ),
                "no_pra_output_tokens_per_second": statistics.fmean(
                    value["tokens_per_second"] for value in serving["no_pra"]
                ),
                "pra_output_tokens_per_second": statistics.fmean(
                    value["tokens_per_second"]
                    for value in serving["pra_no_adaptor"]
                ),
                "adaptor_condition": "NO_QUALIFIED_ADAPTER",
                "peak_memory_bytes": None,
            }
            rows.append(row)
    return {
        "schema_version": 1,
        "comparison": "Qwen3-4B matched BF16/INT8/INT4 MultiHop-RAG",
        "comparison_scope": (
            "Matched deployed pipelines: standard selected-text retrieval versus "
            "PRA hybrid retrieval with detached native K/V. Selector and transport "
            "are not held independently constant."
        ),
        "invariants": {
            field: first[field] for field in invariant_fields
        },
        "hardware": first["hardware"],
        "precision_provenance": {
            "MLX-bfloat16": {
                "immutable_revision": (
                    "1cfa9a7208912126459214e8b04321603b3df60c"
                ),
                "loaded_parameter_arrays": 398,
                "loaded_parameter_dtype": "mlx.core.bfloat16",
                "quantized_layers": 0,
            }
        },
        "inputs": {key: str(value.relative_to(ROOT)).replace("\\", "/") for key, value in INPUTS.items()},
        "rows": rows,
        "limitations": [
            "Ten questions and one dataset seed provide controlled, not production, evidence.",
            "The baseline and PRA arms use different selectors as well as different context transport, so this is not a selector-frozen representation ablation.",
            "No document-RAG adaptor is qualified, so the third canonical arm is absent.",
            "Peak memory was not captured by this harness.",
            "FP32 matched rows remain unmeasured; the source checkpoint itself is BF16.",
        ],
    }


def write_comparison(payload: Mapping[str, Any], output: Path = DEFAULT_OUTPUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = list(payload["rows"])
    (output / "precision_rag.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "precision_rag.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Encoding & Candidates & Base F1 & PRA F1 & $\Delta$F1 & Task $\Delta$ & Latency ratio \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['precision_encoding']} & {row['candidate_count']} & "
            f"{row['no_pra_token_f1']:.4f} & {row['pra_no_adaptor_token_f1']:.4f} & "
            f"{row['pra_token_f1_delta']:+.4f} & {row['pra_task_score_delta']:+.2f} & "
            f"{row['pra_total_latency_ratio']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (output / "generated_precision_rag.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    labels = [f"{row['precision_family']}\nK={row['candidate_count']}" for row in rows]
    x = list(range(len(rows)))
    axes[0].bar(x, [row["pra_token_f1_delta"] for row in rows], color="#26766f")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("PRA - baseline token F1")
    axes[1].bar(x, [row["pra_total_latency_ratio"] for row in rows], color="#c54f31")
    axes[1].axhline(1, color="black", linewidth=0.8)
    axes[1].set_ylabel("PRA / baseline total latency")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "precision_rag_comparison.png", dpi=180)
    figure.savefig(output / "precision_rag_comparison.pdf")
    plt.close(figure)


def main() -> None:
    payload = build_comparison()
    write_comparison(payload)
    print(json.dumps({"rows": len(payload["rows"]), "hardware": payload["hardware"]}, indent=2))


if __name__ == "__main__":
    main()
