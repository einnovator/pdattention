"""Build the reusable Paper 2.7 natural query-facet benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pra_hf.natural_query_facets import (
    NaturalFacetAnnotation,
    annotation_from_2wiki,
    annotation_from_musique,
    annotation_summary,
    scorable_labels,
)

from experiments.paper2_7_query_graph.helpers import file_sha256, write_json


def _load_musique(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_2wiki(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_ids(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {(row["dataset"], row["example_id"]) for row in csv.DictReader(stream)}


def _stable_order(dataset: str, identity: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{identity}".encode("utf-8")).hexdigest()


def _valid(annotation: NaturalFacetAnnotation) -> bool:
    labels, _ = scorable_labels(annotation)
    return (
        labels.numel() >= 4
        and torch_unique_count(labels) == len(annotation.source_facets)
        and len(annotation.source_facets) >= 2
    )


def torch_unique_count(labels) -> int:
    return len(set(map(int, labels.tolist())))


def _select(
    annotations: list[NaturalFacetAnnotation],
    *,
    validation: int,
    test: int,
    seed: int,
) -> list[NaturalFacetAnnotation]:
    annotations.sort(key=lambda row: _stable_order(row.dataset, row.example_id, seed))
    chosen = annotations[: validation + test]
    if len(chosen) != validation + test:
        raise RuntimeError("The source dataset does not contain enough valid annotations.")
    return [
        NaturalFacetAnnotation(
            dataset=row.dataset,
            example_id=row.example_id,
            split="validation" if index < validation else "test",
            question=row.question,
            source_schema=row.source_schema,
            source_facets=row.source_facets,
            source_dependencies=row.source_dependencies,
            units=row.units,
        )
        for index, row in enumerate(chosen)
    ]


def run(args) -> dict:
    excluded = _legacy_ids(args.legacy_rows)
    musique = []
    for row in _load_musique(args.musique_dev):
        if ("musique", str(row["id"])) in excluded or not row.get("answerable", True):
            continue
        if len(row.get("question_decomposition") or ()) < 2:
            continue
        annotation = annotation_from_musique(row, split="candidate")
        if _valid(annotation):
            musique.append(annotation)
    wiki = []
    for row in _load_2wiki(args.twowiki_dev):
        if ("2wikimultihopqa", str(row["_id"])) in excluded:
            continue
        if len(row.get("evidences") or ()) < 2:
            continue
        try:
            annotation = annotation_from_2wiki(row, split="candidate")
        except ValueError:
            continue
        if _valid(annotation):
            wiki.append(annotation)

    annotations = _select(
        wiki, validation=args.validation_per_dataset, test=args.test_per_dataset, seed=args.seed
    ) + _select(
        musique, validation=args.validation_per_dataset, test=args.test_per_dataset, seed=args.seed
    )
    annotations.sort(key=lambda row: (row.split, row.dataset, row.example_id))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = args.output_dir / "annotations.jsonl"
    with annotation_path.open("w", encoding="utf-8") as stream:
        for row in annotations:
            stream.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")

    summary = annotation_summary(annotations)
    manifest_rows = [
        {"dataset": row.dataset, "example_id": row.example_id, "split": row.split}
        for row in annotations
    ]
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_rows, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "seed": args.seed,
        "identity_disjoint_from_legacy_74": True,
        "legacy_identities_excluded": len(excluded),
        "identity_sha256": manifest_hash,
        "annotations_sha256": file_sha256(annotation_path),
        "source_paths": {
            "2wikimultihopqa": str(args.twowiki_dev),
            "musique": str(args.musique_dev),
        },
        "summary": summary,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "annotation_guide.md").write_text(
        """# Natural query-facet annotation guide

## Scope

The benchmark maps dataset-authored reasoning metadata onto the original query
surface. MuSiQue contributes its `question_decomposition`; 2WikiMultiHopQA
contributes ordered `evidences` relation triples. These records are natural
questions, but the span mapping is deterministic dataset-derived supervision,
not a new human annotation campaign.

## Unit and facet rules

- Units are Unicode word/punctuation spans with exact character offsets.
- Each MuSiQue decomposition step is one facet.
- 2Wiki compositional questions use one facet per evidence relation; comparison
  questions group connected evidence relations into their parallel query branches.
- Exact multi-token source phrases anchor otherwise ambiguous surface words.
- Unique lexical matches anchor units to facets.
- Answer-type words map to the terminal facet.
- Unmatched content words map to the nearest source anchor.
- Function words, punctuation, and multiply matched terms remain shared/global.
- Shared/global units are excluded from primary ARI, NMI, and pairwise F1.
- Non-contiguous facets are permitted and measured.

## Reliability boundary

The mapper is deterministic and rerunnable, but no human inter-annotator
agreement is claimed. Source strings, token decisions, confidence, and mapping
reasons are persisted for audit. Generated LLM subquestions are evaluated only
as predictions and never used as gold labels.
""",
        encoding="utf-8",
    )
    return manifest


def parse_args():
    inherited = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--validation-per-dataset", type=int, default=20)
    parser.add_argument("--test-per-dataset", type=int, default=100)
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "2wiki/dev.json")
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=inherited / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument(
        "--legacy-rows",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_7_query_graph/natural/natural_retrieval_rows.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/paper2_7_query_facets"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
