"""Summarize the matched Paper 3.2/Paper 3.3 protocol reconciliation."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "paper3.2-paper3.3-protocol-reconciliation-summary-v1"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 32033, samples: int = 10_000
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _paired(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["cell"]), str(row["example_id"])), {})[
            str(row["condition"])
        ] = row
    result = []
    for cell in sorted({key[0] for key in grouped}):
        pairs = [
            pair
            for (candidate_cell, _), pair in grouped.items()
            if candidate_cell == cell
            and {"PACKED_RAG", "INDEPENDENT_PRA"}.issubset(pair)
        ]
        if not pairs:
            continue
        entry: dict[str, object] = {
            "cell": cell,
            "cohort": pairs[0]["PACKED_RAG"]["cohort"],
            "source_token_budget": pairs[0]["PACKED_RAG"]["source_token_budget"],
            "geometry": pairs[0]["PACKED_RAG"]["geometry"],
            "pairs": len(pairs),
            "selection_receipt_matches": sum(
                pair["PACKED_RAG"]["selection_receipt_id"]
                == pair["INDEPENDENT_PRA"]["selection_receipt_id"]
                for pair in pairs
            ),
            "output_matches": sum(
                pair["PACKED_RAG"]["prediction"]
                == pair["INDEPENDENT_PRA"]["prediction"]
                for pair in pairs
            ),
        }
        for metric, source in (
            ("official_score", "official_multihop_rag_score"),
            ("token_f1", "token_f1"),
            ("gold_answer_mean_nll", "gold_answer_mean_nll"),
        ):
            deltas = [
                float(pair["INDEPENDENT_PRA"][source])
                - float(pair["PACKED_RAG"][source])
                for pair in pairs
            ]
            low, high = _bootstrap_mean_ci(deltas)
            entry[f"{metric}_delta_independent_minus_packed"] = statistics.fmean(deltas)
            entry[f"{metric}_delta_ci95"] = [low, high]
        result.append(entry)
    return result


def _conditions(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["cell"]), str(row["condition"])), []).append(row)
    result = []
    for (cell, condition), values in sorted(grouped.items()):
        result.append(
            {
                "cell": cell,
                "cohort": values[0]["cohort"],
                "source_token_budget": values[0]["source_token_budget"],
                "geometry": values[0]["geometry"],
                "budget_token_unit": (
                    "model-native"
                    if values[0]["geometry"]
                    == "paper32_256_chunk_level_bm25v2"
                    else "source"
                ),
                "condition": condition,
                "examples": len(values),
                "official_score": _mean(values, "official_multihop_rag_score"),
                "token_f1": _mean(values, "token_f1"),
                "gold_answer_mean_nll": _mean(values, "gold_answer_mean_nll"),
                "selected_source_tokens": _mean(values, "selected_source_tokens"),
                "selected_native_tokens": _mean(values, "selected_native_tokens"),
                "supporting_document_coverage": _mean(
                    values, "supporting_document_coverage"
                ),
            }
        )
    return result


def _matched_contrasts(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare protocol cells on shared identities with paired intervals."""

    specifications = (
        (
            "paper33_budget_1024_minus_512",
            "paper33_test_1024",
            "paper33_test_512",
        ),
        (
            "paper32_budget_1024_minus_512",
            "paper32_eval_1024",
            "paper32_eval_512",
        ),
        (
            "paper32_legacy_geometry_minus_common_1024",
            "paper32_eval_legacy_geometry_bm25v2",
            "paper32_eval_1024",
        ),
    )
    indexed = {
        (str(row["cell"]), str(row["condition"]), str(row["example_id"])): row
        for row in rows
    }
    result = []
    for name, left_cell, right_cell in specifications:
        for condition in ("PACKED_RAG", "INDEPENDENT_PRA"):
            identities = sorted(
                example_id
                for cell, candidate_condition, example_id in indexed
                if cell == left_cell
                and candidate_condition == condition
                and (right_cell, condition, example_id) in indexed
            )
            entry: dict[str, object] = {
                "contrast": name,
                "left_cell": left_cell,
                "right_cell": right_cell,
                "condition": condition,
                "pairs": len(identities),
            }
            for metric, source in (
                ("official_score", "official_multihop_rag_score"),
                ("token_f1", "token_f1"),
                ("gold_answer_mean_nll", "gold_answer_mean_nll"),
            ):
                deltas = [
                    float(indexed[(left_cell, condition, identity)][source])
                    - float(indexed[(right_cell, condition, identity)][source])
                    for identity in identities
                ]
                low, high = _bootstrap_mean_ci(deltas)
                entry[f"{metric}_delta"] = statistics.fmean(deltas)
                entry[f"{metric}_delta_ci95"] = [low, high]
            result.append(entry)
    return result


def _bm25_revision_contrasts(
    rows: Sequence[Mapping[str, object]], historical_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Pair the original BM25-v1 run with the corrected v2 legacy replay."""

    current = {
        (str(row["example_id"]), str(row["condition"])): row
        for row in rows
        if row["cell"] == "paper32_eval_legacy_geometry_bm25v2"
    }
    old_names = {
        "PACKED_RAG": "A_FULL_CAUSAL_RAG",
        "INDEPENDENT_PRA": "C_INDEPENDENT_PRA",
    }
    result = []
    for condition, old_condition in old_names.items():
        old = {
            str(row["example_id"]): row
            for row in historical_rows
            if row["condition"] == old_condition
        }
        identities = sorted(
            identity for identity in old if (identity, condition) in current
        )
        selection_jaccards = []
        selection_matches = 0
        for identity in identities:
            before = set(str(value) for value in old[identity]["record_ids"])
            after = set(
                str(value) for value in current[(identity, condition)]["selected_chunk_ids"]
            )
            selection_matches += before == after
            selection_jaccards.append(len(before & after) / max(len(before | after), 1))
        entry: dict[str, object] = {
            "contrast": "bm25_v2_minus_v1_same_legacy_protocol",
            "condition": condition,
            "pairs": len(identities),
            "exact_selection_matches": selection_matches,
            "selected_chunk_jaccard_mean": statistics.fmean(selection_jaccards),
        }
        for metric, current_field, old_field in (
            ("official_score", "official_multihop_rag_score", "official_multihop_rag_score"),
            ("token_f1", "token_f1", "token_f1"),
            ("gold_answer_mean_nll", "gold_answer_mean_nll", "gold_answer_mean_nll"),
            (
                "supporting_document_coverage",
                "supporting_document_coverage",
                "supporting_document_coverage",
            ),
            ("selected_native_tokens", "selected_native_tokens", "physical_native_tokens"),
        ):
            deltas = [
                float(current[(identity, condition)][current_field])
                - float(old[identity][old_field])
                for identity in identities
            ]
            low, high = _bootstrap_mean_ci(deltas)
            entry[f"{metric}_delta"] = statistics.fmean(deltas)
            entry[f"{metric}_delta_ci95"] = [low, high]
        result.append(entry)
    return result


def _historical_anchors(paper32: Path, paper33: Path) -> list[dict[str, object]]:
    old = json.loads(paper32.read_text(encoding="utf-8"))
    old_by_condition = {
        str(row["condition"]): row for row in old["summary"]["conditions"]
    }
    new = json.loads(paper33.read_text(encoding="utf-8"))
    new_by_condition = {str(row["condition"]): row for row in new["summary"]}
    return [
        {
            "anchor": "paper32_reported_1024",
            "cohort": "paper32_eval",
            "condition": condition,
            "examples": old_by_condition[source]["examples"],
            "official_score": old_by_condition[source]["official_multihop_rag_score"],
            "token_f1": old_by_condition[source]["token_f1"],
            "gold_answer_mean_nll": old_by_condition[source]["gold_answer_mean_nll"],
            "selected_native_tokens": old_by_condition[source]["physical_native_tokens"],
        }
        for condition, source in (
            ("PACKED_RAG", "A_FULL_CAUSAL_RAG"),
            ("INDEPENDENT_PRA", "C_INDEPENDENT_PRA"),
        )
    ] + [
        {
            "anchor": "paper33_reported_512",
            "cohort": "paper33_test",
            "condition": condition,
            "examples": new_by_condition[condition]["examples"],
            "official_score": new_by_condition[condition][
                "official_multihop_rag_score_mean"
            ],
            "token_f1": new_by_condition[condition]["token_f1_mean"],
            "gold_answer_mean_nll": new_by_condition[condition][
                "gold_answer_mean_nll_mean"
            ],
            "selected_native_tokens": new_by_condition[condition][
                "initial_native_tokens_mean"
            ],
        }
        for condition in ("PACKED_RAG", "INDEPENDENT_PRA")
    ]


def _factor_effects(conditions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    lookup = {
        (str(row["cell"]), str(row["condition"])): row for row in conditions
    }

    def difference(
        left: str, right: str, condition: str, metric: str
    ) -> float:
        return float(lookup[(left, condition)][metric]) - float(
            lookup[(right, condition)][metric]
        )

    result: dict[str, object] = {}
    for condition in ("PACKED_RAG", "INDEPENDENT_PRA"):
        result[condition] = {
            "budget_1024_minus_512_paper33": {
                metric: difference(
                    "paper33_test_1024", "paper33_test_512", condition, metric
                )
                for metric in ("official_score", "token_f1", "gold_answer_mean_nll")
            },
            "budget_1024_minus_512_paper32": {
                metric: difference(
                    "paper32_eval_1024", "paper32_eval_512", condition, metric
                )
                for metric in ("official_score", "token_f1", "gold_answer_mean_nll")
            },
            "cohort_paper33_minus_paper32_at_512": {
                metric: difference(
                    "paper33_test_512", "paper32_eval_512", condition, metric
                )
                for metric in ("official_score", "token_f1", "gold_answer_mean_nll")
            },
            "cohort_paper33_minus_paper32_at_1024": {
                metric: difference(
                    "paper33_test_1024", "paper32_eval_1024", condition, metric
                )
                for metric in ("official_score", "token_f1", "gold_answer_mean_nll")
            },
            "legacy_geometry_minus_common_at_1024_paper32": {
                metric: difference(
                    "paper32_eval_legacy_geometry_bm25v2",
                    "paper32_eval_1024",
                    condition,
                    metric,
                )
                for metric in ("official_score", "token_f1", "gold_answer_mean_nll")
            },
        }
    return result


def _write_table(conditions: Sequence[Mapping[str, object]], output: Path) -> None:
    labels = {
        "paper33_test_512": "Paper 3.3 test, 512",
        "paper33_test_1024": "Paper 3.3 test, 1,024",
        "paper32_eval_512": "Paper 3.2 eval, 512",
        "paper32_eval_1024": "Paper 3.2 eval, 1,024",
        "paper32_eval_legacy_geometry_bm25v2": "Paper 3.2 eval, legacy geometry v2",
    }
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Cohort / budget & Realization & $n$ & Budget tok. & Native tok. & Official & F1 & Gold NLL \\",
        r"\midrule",
    ]
    order = {
        (cell, condition): index
        for index, (cell, condition) in enumerate(
            (cell, condition)
            for cell in (
                "paper33_test_512",
                "paper33_test_1024",
                "paper32_eval_512",
                "paper32_eval_1024",
                "paper32_eval_legacy_geometry_bm25v2",
            )
            for condition in ("PACKED_RAG", "INDEPENDENT_PRA")
        )
    }
    for row in sorted(
        conditions, key=lambda value: order[(str(value["cell"]), str(value["condition"]))]
    ):
        condition = (
            "Packed" if row["condition"] == "PACKED_RAG" else "Independent PRA"
        )
        lines.append(
            f"{labels[str(row['cell'])]} & {condition} & {row['examples']} & "
            f"{float(row['selected_source_tokens']):.1f} & "
            f"{float(row['selected_native_tokens']):.1f} & "
            f"{float(row['official_score']):.3f} & {float(row['token_f1']):.3f} & "
            f"{float(row['gold_answer_mean_nll']):.3f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gap_table(
    conditions: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    labels = {
        "paper33_test_512": "Paper 3.3 test, 512",
        "paper33_test_1024": "Paper 3.3 test, 1,024",
        "paper32_eval_512": "Paper 3.2 eval, 512",
        "paper32_eval_1024": "Paper 3.2 eval, 1,024",
        "paper32_eval_legacy_geometry_bm25v2": "Paper 3.2 legacy-v2",
    }
    cells = tuple(labels)
    lookup = {
        (str(row["cell"]), str(row["condition"])): row for row in conditions
    }
    paired = {str(row["cell"]): row for row in pairs}
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Cohort / budget & $n$ & Packed Off. & PRA Off. & $\Delta$ Off. [95\% CI] & Packed F1 & PRA F1 \\",
        r"\midrule",
    ]
    for cell in cells:
        packed = lookup[(cell, "PACKED_RAG")]
        independent = lookup[(cell, "INDEPENDENT_PRA")]
        effect = paired[cell]
        low, high = effect["official_score_delta_ci95"]
        lines.append(
            f"{labels[cell]} & {effect['pairs']} & "
            f"{float(packed['official_score']):.3f} & "
            f"{float(independent['official_score']):.3f} & "
            f"{float(effect['official_score_delta_independent_minus_packed']):+.3f} "
            f"[{float(low):+.3f},{float(high):+.3f}] & "
            f"{float(packed['token_f1']):.3f} & "
            f"{float(independent['token_f1']):.3f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(
    conditions: Sequence[Mapping[str, object]],
    historical: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    cells = [
        "paper33_test_512",
        "paper33_test_1024",
        "paper32_eval_512",
        "paper32_eval_1024",
        "paper32_reported_1024",
        "paper32_eval_legacy_geometry_bm25v2",
    ]
    labels = [
        "P3.3\n512",
        "P3.3\n1,024",
        "P3.2\n512",
        "P3.2\n1,024",
        "P3.2 legacy\nBM25-v1",
        "P3.2 legacy\nBM25-v2",
    ]
    lookup = {
        (str(row["cell"]), str(row["condition"])): row for row in conditions
    }
    lookup.update(
        {
            (str(row["anchor"]), str(row["condition"])): row
            for row in historical
            if row["anchor"] == "paper32_reported_1024"
        }
    )
    x = list(range(len(cells)))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for offset, condition, label, color in (
        (-width / 2, "PACKED_RAG", "Packed RAG", "#276FBF"),
        (width / 2, "INDEPENDENT_PRA", "Independent PRA", "#E4572E"),
    ):
        axes[0].bar(
            [value + offset for value in x],
            [float(lookup[(cell, condition)]["official_score"]) for cell in cells],
            width,
            label=label,
            color=color,
        )
        axes[1].bar(
            [value + offset for value in x],
            [float(lookup[(cell, condition)]["token_f1"]) for cell in cells],
            width,
            label=label,
            color=color,
        )
    for axis, title, ylabel in (
        (axes[0], "Answer containment", "Official score"),
        (axes[1], "Token overlap", "Token F1"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"protocol_reconciliation.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--paper32-manifest", type=Path, required=True)
    parser.add_argument("--paper33-summary", type=Path, required=True)
    parser.add_argument(
        "--paper32-results",
        type=Path,
        default=Path(
            "docs/papers/shared/results/paper3_2_rag/crossdoc_adapter/"
            "qwen3_1_7b_rank8_five_seed/condition_results.jsonl.gz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = _load_jsonl(args.run_dir / "rows.jsonl")
    conditions = _conditions(rows)
    pairs = _paired(rows)
    historical = _historical_anchors(args.paper32_manifest, args.paper33_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "conditions": conditions,
        "paired_effects": pairs,
        "matched_protocol_contrasts": _matched_contrasts(rows),
        "bm25_revision_contrasts": _bm25_revision_contrasts(
            rows, _load_jsonl(args.paper32_results)
        ),
        "factor_effects": _factor_effects(conditions),
        "historical_anchors": historical,
    }
    (args.output_dir / "publication_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_table(conditions, args.output_dir / "generated_protocol_reconciliation_table.tex")
    _write_gap_table(
        conditions, pairs, args.output_dir / "generated_protocol_gap_table.tex"
    )
    _plot(conditions, historical, args.output_dir)
    print(json.dumps({"pairs": sum(row["pairs"] for row in pairs)}, sort_keys=True))


if __name__ == "__main__":
    main()
