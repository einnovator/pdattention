"""Build compact Paper 3.2 tables and figures from measured run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path | None) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else None


def _composition_plot(summary: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    comparisons = summary["summary"]["fresh_packed_comparisons"]
    preferred = [
        "NATIVE_SOURCE_LOCAL",
        "NATIVE_GLOBAL_REBOUND",
        "REPAIR_0.25",
        "REPAIR_0.5",
        "REPAIR_0.75",
        "REPAIR_1",
    ]
    labels = [name for name in preferred if name in comparisons]
    values = [float(comparisons[name]["gold_nll_mean_abs_delta"]) for name in labels]
    display = [
        name.replace("NATIVE_", "").replace("GLOBAL_", "").replace("REPAIR_", "repair ")
        for name in labels
    ]
    fig, axis = plt.subplots(figsize=(7.4, 3.4))
    axis.plot(display, values, marker="o", color="#b42318", linewidth=2)
    axis.set_ylabel("Mean absolute gold-answer NLL delta")
    axis.set_xlabel("Independent-memory realization")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=20)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"schema_version": "paper3.2-publication-summary-v1"}
    composition = _load(args.composition_manifest)
    if composition is not None:
        result["composition"] = composition["summary"]
        _composition_plot(composition, args.output_dir / "composition_nll_repair_curve")
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
    (args.output_dir / "publication_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
