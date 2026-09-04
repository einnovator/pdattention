"""Build compact Paper 3.2 tables and figures from measured run summaries."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean


def _load(path: Path | None) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else None


def _summarize_scale_run(run_dir: Path) -> dict[str, object]:
    manifest = _load(run_dir / "cohort_manifest.json")
    assert manifest is not None
    with gzip.open(run_dir / "condition_results.jsonl.gz", "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    rows = [row for row in rows if row.get("status") == "MEASURED"]

    def condition(name: str, regime: str) -> list[dict[str, object]]:
        return [
            row for row in rows if row["condition"] == name and row["regime"] == regime
        ]

    def averages(selected: list[dict[str, object]]) -> dict[str, float]:
        serving = [row["serving_metrics"] for row in selected]
        return {
            "examples": len(selected),
            "token_f1": mean(float(row["token_f1"]) for row in selected),
            "gold_answer_mean_nll": mean(float(row["gold_answer_mean_nll"]) for row in serving),
            "ttft_ms": mean(float(row["ttft_ms"]) for row in serving),
            "total_latency_ms": mean(float(row["total_latency_ms"]) for row in serving),
            "visible_prompt_tokens": mean(float(row["visible_prompt_tokens"]) for row in serving),
            "selected_native_kv_tokens": mean(
                float(row["selected_native_kv_tokens"]) for row in serving
            ),
        }

    text_cold = condition("PRA_SELECTED_CONTEXT_NO_ADAPTOR", "COLD")
    text_warm = condition("PRA_SELECTED_CONTEXT_NO_ADAPTOR", "WARM")
    prefix_warm = condition("PRA_SELECTED_CONTEXT_NO_ADAPTOR", "PREFIX_WARM")
    native_cold = condition("PRA_NATIVE_MEMORY_NO_ADAPTOR", "COLD")
    native_warm = condition("PRA_NATIVE_MEMORY_NO_ADAPTOR", "WARM")
    native_by_pair = {
        (row["example_id"], row["regime"], row["selection_receipt_id"]): row
        for row in native_cold + native_warm
    }
    paired = []
    for text_row in text_cold + text_warm:
        key = (
            text_row["example_id"],
            text_row["regime"],
            text_row["selection_receipt_id"],
        )
        paired.append((text_row, native_by_pair[key]))
    parity = {
        "pairs": len(paired),
        "output_matches": sum(left["prediction"] == right["prediction"] for left, right in paired),
        "logit_hash_matches": sum(
            left["serving_metrics"]["first_step_logits_sha256"]
            == right["serving_metrics"]["first_step_logits_sha256"]
            for left, right in paired
        ),
        "gold_nll_max_abs_delta": max(
            abs(
                float(left["serving_metrics"]["gold_answer_mean_nll"])
                - float(right["serving_metrics"]["gold_answer_mean_nll"])
            )
            for left, right in paired
        ),
    }
    return {
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "seed": manifest["seed"],
        "hardware": manifest["hardware"],
        "text_cold": averages(text_cold),
        "text_warm_reprefill": averages(text_warm),
        "text_exact_prefix": averages(prefix_warm),
        "native_cold": averages(native_cold),
        "native_warm": averages(native_warm),
        "parity": parity,
    }


def _composition_plot(summary: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    comparisons = summary["summary"]["fresh_packed_comparisons"]
    preferred = [
        "NATIVE_SOURCE_LOCAL",
        "NATIVE_GLOBAL_REBOUND",
        "REPAIR_BOUNDARY_0.25",
        "REPAIR_BOUNDARY_0.5",
        "REPAIR_LATER_PREFIX_0.25",
        "REPAIR_LATER_PREFIX_0.5",
        "REPAIR_EVEN_1",
    ]
    labels = [name for name in preferred if name in comparisons]
    values = [float(comparisons[name]["gold_nll_mean_abs_delta"]) for name in labels]
    display = [
        name.replace("NATIVE_", "").replace("GLOBAL_", "").replace("REPAIR_", "repair ")
        for name in labels
    ]
    fig, axis = plt.subplots(figsize=(7.4, 3.4))
    axis.plot(display, values, marker="o", color="#b42318", linewidth=2)
    axis.set_ylabel("Mean absolute NLL delta")
    axis.set_xlabel("Independent-memory realization")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _partial_materialization_plot(summary: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in summary["summary"]["conditions"]
        if row["resource_order_name"] == "canonical"
        and (row["condition"] == "FRESH_PACKED" or row["condition"].startswith("PARTIAL_"))
    ]
    policies = {
        "FRESH_PACKED": ("full", "#111827"),
        "PARTIAL_SCORE": ("score", "#2563eb"),
        "PARTIAL_ORACLE": ("evidence oracle", "#15803d"),
        "PARTIAL_WRONG": ("wrong memory", "#b42318"),
    }
    fig, (quality, nll) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for prefix, (label, color) in policies.items():
        selected = [row for row in rows if row["condition"].startswith(prefix)]
        selected.sort(key=lambda row: float(row["active_native_tokens"]))
        if not selected:
            continue
        x = [float(row["active_native_tokens"]) for row in selected]
        quality.plot(x, [float(row["token_f1"]) for row in selected], marker="o", label=label, color=color)
        nll.plot(x, [float(row["gold_answer_mean_nll"]) for row in selected], marker="o", label=label, color=color)
    quality.set_ylabel("Token F1")
    nll.set_ylabel("Gold-answer mean NLL")
    for axis in (quality, nll):
        axis.set_xlabel("Active native K/V entries")
        axis.grid(alpha=0.25)
    quality.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _nonprefix_reuse_plot(summary: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary["summary"]["conditions"]
    preferred = [
        "FRESH_PACKED",
        "ORDINARY_PREFIX_CACHE",
        "PRA_GLOBAL_REBOUND",
        "PRA_PARTIAL_0.5",
    ]
    selected = [next(row for row in rows if row["condition"] == name) for name in preferred]
    labels = [
        "fresh",
        "prefix cache",
        "PRA rebound",
        "PRA partial 50%",
    ]
    new = [float(row["newly_encoded_tokens"]) for row in selected]
    reused = [float(row["reused_tokens"]) for row in selected]
    total = [float(row["total_with_materialization_ms"]) for row in selected]
    x = list(range(len(labels)))
    width = 0.36
    fig, (tokens, latency) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    tokens.bar(
        [value - width / 2 for value in x],
        new,
        width,
        label="encode/recompute work",
        color="#2563eb",
    )
    tokens.bar(
        [value + width / 2 for value in x],
        reused,
        width,
        label="cached resource hits",
        color="#93c5fd",
    )
    tokens.set_ylabel("Mean K/V token count per turn")
    tokens.legend(fontsize=7)
    latency.bar(x, total, color="#0f766e")
    latency.set_ylabel("Mean total latency (ms)")
    for axis in (tokens, latency):
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scale_plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["model"]).split("/")[-1].replace("-4bit", "") for row in rows]
    ttft_reprefill = [
        float(row["native_warm"]["ttft_ms"]) / float(row["text_warm_reprefill"]["ttft_ms"])
        for row in rows
    ]
    ttft_prefix = [
        float(row["native_warm"]["ttft_ms"]) / float(row["text_exact_prefix"]["ttft_ms"])
        for row in rows
    ]
    x = list(range(len(rows)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.8, 3.5))
    axis.bar([value - width / 2 for value in x], ttft_reprefill, width, label="vs text re-prefill")
    axis.bar([value + width / 2 for value in x], ttft_prefix, width, label="vs exact prefix cache")
    axis.axhline(1.0, color="#111827", linewidth=1)
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("Native warm TTFT ratio")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _position_plot(summary: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    comparisons = summary["summary"]["fresh_packed_comparisons"]
    preferred = [
        "NATIVE_SOURCE_LOCAL",
        "NATIVE_GLOBAL_REBOUND",
        "POSITION_RESOURCE_ADJACENT",
        "POSITION_RANK_DISTANCE",
        "POSITION_SCORE_DISTANCE",
        "POSITION_NON_OVERLAPPING_NEAR_BANDS",
        "POSITION_RANDOM_DISTANCE",
    ]
    rows = [(name, comparisons[name]) for name in preferred if name in comparisons]
    labels = [
        name.replace("NATIVE_", "").replace("POSITION_", "").replace("NON_OVERLAPPING_", "")
        for name, _ in rows
    ]
    x = list(range(len(rows)))
    fig, axis = plt.subplots(figsize=(8.6, 3.5))
    axis.bar(x, [float(row["first_step_js_divergence_mean"]) for _, row in rows], color="#7c3aed")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Mean first-step JS vs fresh packed")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summarize_scale_composition(path: Path) -> dict[str, object]:
    manifest = _load(path / "manifest.json")
    assert manifest is not None
    comparisons = manifest["summary"]["fresh_packed_comparisons"]
    names = (
        "NATIVE_CONTIGUOUS",
        "NATIVE_SOURCE_LOCAL",
        "NATIVE_GLOBAL_REBOUND",
        "REPAIR_LATER_PREFIX_0.5",
        "REPAIR_EVEN_1",
        "PARTIAL_SCORE_0.75",
    )
    return {
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "examples": len(manifest["question_ids"]),
        "conditions": {name: comparisons[name] for name in names},
    }


def _scale_composition_plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    conditions = (
        ("NATIVE_SOURCE_LOCAL", "source-local"),
        ("NATIVE_GLOBAL_REBOUND", "packed rebound"),
        ("REPAIR_LATER_PREFIX_0.5", "50% later-prefix repair"),
    )
    labels = [str(row["model"]).split("/")[-1].replace("-4bit", "") for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.2, 3.5))
    for index, (condition, label) in enumerate(conditions):
        values = []
        for row in rows:
            result = row["conditions"][condition]
            values.append(float(result["output_matches"]) / float(result["pairs"]))
        positions = [value + (index - 1) * width for value in x]
        axis.bar(positions, values, width, label=label)
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Generated-output parity vs fresh packed")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _retrieval_plot(summary: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary["conditions"]
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    top_ks = sorted({int(row["top_k"]) for row in rows})
    width = 0.8 / max(len(methods), 1)
    fig, axis = plt.subplots(figsize=(8.4, 3.8))
    for index, method in enumerate(methods):
        values = {
            int(row["top_k"]): float(row["supporting_document_recall"])
            for row in rows
            if row["method"] == method
        }
        x = [position + (index - (len(methods) - 1) / 2) * width for position in range(len(top_ks))]
        axis.bar(x, [values.get(k, 0.0) for k in top_ks], width=width, label=method)
    axis.set_xticks(range(len(top_ks)), [f"R@{value}" for value in top_ks])
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Supporting-document recall")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition-manifest", type=Path)
    parser.add_argument("--retrieval-summary", type=Path)
    parser.add_argument("--service-summary", type=Path)
    parser.add_argument("--transport-summary", type=Path)
    parser.add_argument("--nonprefix-manifest", type=Path)
    parser.add_argument("--scale-run", type=Path, action="append", default=[])
    parser.add_argument("--position-manifest", type=Path)
    parser.add_argument("--scale-composition", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"schema_version": "paper3.2-publication-summary-v1"}
    composition = _load(args.composition_manifest)
    if composition is not None:
        result["composition"] = composition["summary"]
        _composition_plot(composition, args.output_dir / "composition_nll_repair_curve")
        _partial_materialization_plot(composition, args.output_dir / "partial_materialization_frontier")
    retrieval = _load(args.retrieval_summary)
    if retrieval is not None:
        result["local_retrieval"] = retrieval
        _retrieval_plot(retrieval, args.output_dir / "local_retrieval_recall")
    service = _load(args.service_summary)
    if service is not None:
        result["service_retrieval"] = service
        _retrieval_plot(service, args.output_dir / "service_retrieval_recall")
    transport = _load(args.transport_summary)
    if transport is not None:
        result["transport"] = transport
    nonprefix = _load(args.nonprefix_manifest)
    if nonprefix is not None:
        result["nonprefix_reuse"] = nonprefix["summary"]
        _nonprefix_reuse_plot(nonprefix, args.output_dir / "nonprefix_reuse")
    if args.scale_run:
        result["scale"] = [_summarize_scale_run(path) for path in args.scale_run]
        _scale_plot(result["scale"], args.output_dir / "scale_transport")
    position = _load(args.position_manifest)
    if position is not None:
        result["position"] = position["summary"]
        _position_plot(position, args.output_dir / "position_policy_js")
    if args.scale_composition:
        result["scale_composition"] = [
            _summarize_scale_composition(path) for path in args.scale_composition
        ]
        _scale_composition_plot(
            result["scale_composition"], args.output_dir / "scale_composition_parity"
        )
    (args.output_dir / "publication_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
