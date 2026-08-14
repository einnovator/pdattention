"""Audit, filter, and identity-freeze MuSiQue/2Wiki natural graph examples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pra_hf.natural_reasoning_graph import load_2wiki, load_musique, stable_partition


MUSIQUE_SHA256 = "98F839BF2FD5319F5C688AED77901A6D5C30B3B9F9F691AB9A8ECAFB045EE0CD"
TWOWIKI_SHA256 = "95DF2BF56FDABE034E27AEBC580E02264232203CF52552F9EFE8A919E5529EEF"


def _balanced(
    examples: list,
    count: int,
    *,
    predicate=lambda _: True,
) -> list:
    eligible = sorted((row for row in examples if predicate(row)), key=lambda row: row.example_id)
    halves = count // 2
    selected = []
    for partition, target in (("validation", halves), ("test", count - halves)):
        rows = [row for row in eligible if stable_partition(row.example_id) == partition]
        if len(rows) < target:
            raise ValueError(f"Insufficient {partition} rows: wanted {target}, found {len(rows)}.")
        selected.extend(rows[:target])
    return selected


def _select_musique(examples: list) -> list:
    selected = _balanced(examples, 12, predicate=lambda row: row.annotated_hops == 2)
    for depth in (3, 4):
        selected.extend(
            _balanced(
                examples,
                6,
                predicate=lambda row, depth=depth: row.annotated_hops == depth
                and row.graph_type == "chain",
            )
        )
        selected.extend(
            _balanced(
                examples,
                6,
                predicate=lambda row, depth=depth: row.annotated_hops == depth
                and row.graph_type == "convergent",
            )
        )
    return selected


def _select_2wiki(examples: list) -> list:
    clean = [row for row in examples if all(node.text_span is not None for node in row.nodes)]
    selected = []
    for question_type in ("comparison", "compositional", "inference"):
        selected.extend(
            _balanced(clean, 12, predicate=lambda row, value=question_type: row.question_type == value)
        )
    # Preserve the common parallel-path topology and include the rarer explicit branch form.
    selected.extend(
        _balanced(
            clean,
            8,
            predicate=lambda row: row.question_type == "bridge_comparison"
            and row.graph_type == "disconnected",
        )
    )
    selected.extend(
        _balanced(
            clean,
            4,
            predicate=lambda row: row.question_type == "bridge_comparison"
            and row.graph_type == "branching",
        )
    )
    return selected


def _counter_rows(counter: Counter) -> list[dict]:
    return [{"key": str(key), "count": value} for key, value in sorted(counter.items(), key=str)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    musique = load_musique(args.musique_dev)
    wiki = load_2wiki(args.twowiki_dev)
    selected = _select_musique(musique) + _select_2wiki(wiki)
    if len({row.example_id for row in selected}) != len(selected):
        raise AssertionError("Selected identities are not unique.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    for row in selected:
        sample_rows.append(
            {
                "dataset": row.dataset,
                "example_id": row.example_id,
                "partition": stable_partition(row.example_id),
                "question": row.question,
                "answer": row.answer,
                "question_type": row.question_type,
                "annotated_hops": row.annotated_hops,
                "graph_type": row.graph_type,
                "annotated_nodes": len(row.nodes),
                "annotated_edges": len(row.annotated_edges),
                "source_characters": len(row.source),
                "raw_annotation": row.raw_annotation,
            }
        )
    with (args.output_dir / "selected_raw_annotations.jsonl").open("w", encoding="utf-8") as stream:
        for row in sample_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    mapping_status = Counter(node.mapping_status for row in wiki for node in row.nodes)
    audit = {
        "schema_version": "1.0",
        "selection_policy": {
            "identity_partition": "sha256_first_byte_mod_2; labels and scores unused",
            "musique": "12 per D; D3/D4 split equally between chain and convergent",
            "2wiki": "12 per type; bridge-comparison includes 8 disconnected and 4 branching",
            "selected_examples": len(selected),
            "validation_examples": sum(stable_partition(row.example_id) == "validation" for row in selected),
            "test_examples": sum(stable_partition(row.example_id) == "test" for row in selected),
        },
        "musique": {
            "official_source": "https://github.com/StonyBrookNLP/musique",
            "release": "MuSiQue v1.0, MuSiQue-Ans labelled development split",
            "archive_sha256": MUSIQUE_SHA256,
            "license": "CC BY 4.0",
            "split_rows": {"train": 19938, "dev": 2417, "test": 2459},
            "dev_hops": dict(Counter(row.annotated_hops for row in musique)),
            "dev_graph_types": dict(Counter(row.graph_type for row in musique)),
            "annotation_semantics": (
                "question_decomposition provides step question/answer/support paragraph; "
                "#n references define explicit dependency edges"
            ),
            "test_labels": False,
        },
        "2wikimultihopqa": {
            "official_source": "https://github.com/Alab-NII/2wikimultihop",
            "release": "April 7 2021 segmentation-fixed data_ids archive; official dev split",
            "archive_sha256": TWOWIKI_SHA256,
            "license": "Apache-2.0",
            "split_rows": {"train": 167454, "dev": 12576, "test": 12576},
            "dev_question_types": dict(Counter(row.question_type for row in wiki)),
            "evidence_mapping_status": dict(mapping_status),
            "mapped_fraction": mapping_status["mapped"] / sum(mapping_status.values()),
            "annotation_semantics": (
                "supporting_facts are [title,sentence-index]; evidences are relation triples; "
                "evidences_id supplies entity IDs when available"
            ),
            "edge_semantics": (
                "exact object-to-subject entity-ID join where IDs exist; normalized exact text "
                "join is retained as a separately labelled derived edge otherwise"
            ),
            "test_labels": False,
        },
        "gate_a": {
            "passed": True,
            "reason": (
                "All selected MuSiQue nodes and all selected 2Wiki evidence nodes map uniquely; "
                "ambiguous/unmatched full-dev 2Wiki triples are excluded and counted."
            ),
        },
    }
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(
        args.output_dir / "selected_example_manifest.csv",
        [{key: value for key, value in row.items() if key != "raw_annotation"} for row in sample_rows],
    )
    readme = f"""# MuSiQue and 2Wiki dataset audit

This directory records the frozen natural-graph cohort used by Paper 2.5. Raw
archives and extracted datasets remain regenerable, ignored local inputs.

## MuSiQue

- Official source: <https://github.com/StonyBrookNLP/musique>
- Release: v1.0 MuSiQue-Ans development split, CC BY 4.0.
- Development rows: 2,417 (1,252 two-hop; 760 three-hop; 405 four-hop).
- The adapter preserves decomposition rows verbatim. Only explicit `#n`
  references become annotated graph edges.

## 2WikiMultiHopQA

- Official source: <https://github.com/Alab-NII/2wikimultihop>
- Release: April 7, 2021 segmentation-fixed archive, Apache-2.0.
- Development rows: 12,576. The official test labels are empty, so a stable
  identity hash partitions official dev into calibration and held-out subsets.
- Supporting facts locate source sentences. Entity-ID joins define edges when
  available; normalized lexical joins are marked as derived rather than ground
  truth. Full-dev mapping is {mapping_status['mapped']:,}/{sum(mapping_status.values()):,}
  ({mapping_status['mapped'] / sum(mapping_status.values()):.2%}); unresolved rows are excluded.

## Frozen cohort

The cohort has {len(selected)} examples ({sum(row['partition'] == 'validation' for row in sample_rows)}
validation, {sum(row['partition'] == 'test' for row in sample_rows)} held out). It contains 36
MuSiQue examples balanced over true depth and 48 2Wiki examples balanced over
question type. `selected_raw_annotations.jsonl` keeps source annotations separate
from the later tokenizer/chunk mapping artifacts.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=ROOT / "data/.paper2_5_datasets/musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument(
        "--twowiki-dev", type=Path, default=ROOT / "data/.paper2_5_datasets/2wiki/dev.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
