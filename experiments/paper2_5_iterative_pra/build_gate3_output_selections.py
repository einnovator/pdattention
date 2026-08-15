"""Freeze executable Paper 2.5 discovery sets before answer generation.

The builder reuses the validated global semantic entry score and exact native
Q/K graph.  Annotation groups are attached only after each selected set is
fixed; only the named oracle control may use them during selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_natural_graph_depth import (
    _atomic_native,
    _feature_example,
    _node_depths,
    _node_recovery,
    _parent_hidden,
    _search,
    _selected_token_metrics,
    _transition_rows,
)
from pra_hf.natural_reasoning_graph import map_example_to_parents
from pra_hf.output_validation import condition_manifest
from pra_hf.query_facets import score_semantic_query_facets
from pra_hf.semantic_graph_search import build_native_parent_adjacency
from pra_torch.hf import load_hf_routing_projection


SEED = 11
ROOT_BREADTH = 4
HOPS = 4
POLICIES = {
    "musique": {
        "graph_sparse": {"chunk_size": 16, "K": 6, "B": 16},
        "graph_balanced": {"chunk_size": 128, "K": 6, "B": 16},
        "graph_high": {"chunk_size": 128, "K": 8, "B": 16},
    },
    "2wikimultihopqa": {
        "graph_sparse": {"chunk_size": 16, "K": 6, "B": 16},
        "graph_balanced": {"chunk_size": 128, "K": 4, "B": 6},
        "graph_high": {"chunk_size": 128, "K": 8, "B": 16},
    },
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _layerwise_index(path: Path) -> dict[tuple[str, str], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    return {
        (row["dataset"], row["example_id"]): root / row["path"]
        for row in manifest["entries"]
    }


def _semantic_parent_scores(feature, mapping, projection, device) -> torch.Tensor:
    """Score one global query state against parent means without task labels."""
    query = feature["query_hidden_states"][-1:].float().to(device)
    parent = _parent_hidden(feature["token_hidden"], mapping.parent_spans).to(device)
    scored = score_semantic_query_facets(
        projection.project_query(query), projection.project_memory(parent)
    )
    return scored.component_scores[:, 0, :].max(dim=0).values.detach().cpu()


def _adjacency(feature, mapping, chunk_size, device, *, layer_data=None):
    source = feature if layer_data is None else layer_data
    q, k, mask, parents = _atomic_native(source, chunk_size)
    return build_native_parent_adjacency(
        q.to(device),
        k.to(device),
        mask.to(device),
        parents.to(device),
        len(mapping.parent_spans),
        token_reduction="top_m_mean",
        head_reduction="top_m_mean",
        top_m=4,
    ).scores.detach().cpu()


def _minimum_depth(scores, roots, mapping, k, b) -> int | None:
    for depth in range(HOPS + 1):
        result = _search(scores, roots, k, depth, b)
        if _node_recovery(result.visited, mapping)[1]:
            return depth
    return None


def _selection_row(
    feature,
    example,
    mapping,
    *,
    selection: str,
    visited,
    roots,
    search_layer: int,
    k: int,
    b: int | None,
    scores: torch.Tensor | None,
    search_seconds: float = 0.0,
) -> dict:
    visited = tuple(sorted(set(map(int, visited))))
    recall, complete, node_hits = _node_recovery(visited, mapping)
    depths = _node_depths(example)
    later = [node.node_id for node in example.nodes if depths[node.node_id] > 1]
    later_recall = (
        sum(float(node_hits[node]) for node in later) / len(later) if later else 1.0
    )
    transition_rows = _transition_rows(feature, example, mapping, scores) if scores is not None else []
    edge_recall = (
        sum(float(row[f"recovered_at_{k}"]) for row in transition_rows) / len(transition_rows)
        if transition_rows
        else 1.0
    )
    spans = [mapping.parent_spans[parent] for parent in visited]
    metrics = _selected_token_metrics(visited, mapping, feature)
    return {
        "dataset": feature["dataset"],
        "example_id": feature["example_id"],
        "partition": feature["partition"],
        "question_type": feature["question_type"],
        "annotated_hops": int(feature["annotated_hops"]),
        "graph_type": feature["graph_type"],
        "selection": selection,
        "oracle_selection": selection == "oracle_evidence",
        "entry_policy": "oracle_annotation" if selection == "oracle_evidence" else "global_semantic_top4",
        "search_layer": int(search_layer),
        "chunk_size": int(mapping.parent_spans[0][1] - mapping.parent_spans[0][0]),
        "K": int(k),
        "B": b,
        "H": HOPS if scores is not None else 0,
        "root_parent_ids": list(map(int, roots)),
        "selected_parent_ids": list(visited),
        "selected_spans": [list(map(int, span)) for span in spans],
        "root_recall": float(bool(set(roots).intersection(mapping.root_parent_ids))),
        "oracle_evidence_recall": recall,
        "later_evidence_recall": later_recall,
        "complete_evidence_recovery": int(complete),
        "annotated_edge_recall": edge_recall,
        "complete_path_survival": int(complete and all(node_hits.values())),
        "native_recovery_depth": (
            _minimum_depth(scores, roots, mapping, k, b) if scores is not None else (0 if complete else None)
        ),
        "visited_parents": len(visited),
        "search_seconds": float(search_seconds),
        "source_tokens": int(feature["source_tokens"]),
        "evidence_token_spans": [list(map(int, span)) for span in feature["node_token_spans"].values()],
        **metrics,
    }


def _build_example(feature, projection, device, layerwise_path: Path) -> list[dict]:
    example = _feature_example(feature)
    dataset_policies = POLICIES[feature["dataset"]]
    by_chunk: dict[int, tuple] = {}
    rows = []
    for chunk_size in sorted({128, *(row["chunk_size"] for row in dataset_policies.values())}):
        mapping = map_example_to_parents(
            example,
            int(feature["source_tokens"]),
            feature["node_token_spans"],
            chunk_size=chunk_size,
        )
        semantic = _semantic_parent_scores(feature, mapping, projection, device)
        roots = tuple(
            int(value)
            for value in torch.argsort(semantic, descending=True, stable=True)[:ROOT_BREADTH]
        )
        by_chunk[chunk_size] = (mapping, roots)

    balanced = dataset_policies["graph_balanced"]
    mapping, roots = by_chunk[balanced["chunk_size"]]
    one_shot = tuple(
        int(value)
        for value in torch.argsort(
            _semantic_parent_scores(feature, mapping, projection, device),
            descending=True,
            stable=True,
        )[: balanced["B"]]
    )
    rows.append(
        _selection_row(
            feature,
            example,
            mapping,
            selection="one_shot",
            visited=one_shot,
            roots=one_shot,
            search_layer=27,
            k=0,
            b=balanced["B"],
            scores=None,
        )
    )

    for selection, policy in dataset_policies.items():
        mapping, roots = by_chunk[policy["chunk_size"]]
        scores = _adjacency(feature, mapping, policy["chunk_size"], device)
        result = _search(scores, roots, policy["K"], HOPS, policy["B"])
        rows.append(
            _selection_row(
                feature,
                example,
                mapping,
                selection=selection,
                visited=result.visited,
                roots=roots,
                search_layer=27,
                k=policy["K"],
                b=policy["B"],
                scores=scores,
                search_seconds=result.search_seconds,
            )
        )

    oracle_mapping, _ = by_chunk[128]
    rows.append(
        _selection_row(
            feature,
            example,
            oracle_mapping,
            selection="oracle_evidence",
            visited=oracle_mapping.oracle_parent_ids,
            roots=oracle_mapping.root_parent_ids,
            search_layer=27,
            k=0,
            b=len(oracle_mapping.oracle_parent_ids),
            scores=None,
        )
    )

    layerwise = torch.load(layerwise_path, map_location="cpu", weights_only=False, mmap=True)
    mapping, roots = by_chunk[balanced["chunk_size"]]
    scores = _adjacency(
        feature,
        mapping,
        balanced["chunk_size"],
        device,
        layer_data=layerwise["layers"][12],
    )
    result = _search(scores, roots, balanced["K"], HOPS, balanced["B"])
    rows.append(
        _selection_row(
            feature,
            example,
            mapping,
            selection="graph_balanced_l12",
            visited=result.visited,
            roots=roots,
            search_layer=12,
            k=balanced["K"],
            b=balanced["B"],
            scores=scores,
            search_seconds=result.search_seconds,
        )
    )
    return rows


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    features = torch.load(args.feature_file, map_location="cpu", weights_only=False, mmap=True)
    layerwise = _layerwise_index(args.layerwise_manifest)
    projection = load_hf_routing_projection(args.projection, device=device)
    rows = []
    for index, feature in enumerate(features, start=1):
        key = (feature["dataset"], feature["example_id"])
        rows.extend(_build_example(feature, projection, device, layerwise[key]))
        print(
            f"[output-selection {index}/{len(features)}] {feature['dataset']} "
            f"{feature['example_id']}",
            flush=True,
        )
    non_oracle = [row for row in rows if row["selection"] != "oracle_evidence"]
    if any(row["oracle_selection"] for row in non_oracle):
        raise AssertionError("oracle labels leaked into an executable selection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "frozen_before_generation": True,
        "projection_seed": SEED,
        "root_breadth": ROOT_BREADTH,
        "max_search_hops": HOPS,
        "policies": POLICIES,
        "protocol": condition_manifest(),
        "oracle_labels_available_to_executable_discovery": False,
        "rows": rows,
    }
    (args.output_dir / "gate3_discovery_selections.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(
        args.output_dir / "gate3_discovery_selections.csv",
        [
            {key: value for key, value in row.items() if key not in {"selected_parent_ids", "selected_spans", "root_parent_ids", "evidence_token_spans"}}
            for row in rows
        ],
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    natural = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth"
    layerwise = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/layerwise_graph"
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation"
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-file", type=Path, default=natural / "natural_graph_features.pt")
    parser.add_argument("--layerwise-manifest", type=Path, default=layerwise / "layerwise_feature_manifest.json")
    parser.add_argument(
        "--projection",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter/checkpoints/asymmetric_linear_d128_last_joint_seed11_margin_exhaustive.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    return parser.parse_args()


if __name__ == "__main__":
    artifact = run(parse_args())
    print(json.dumps({"rows": len(artifact["rows"]), "output": "gate3_discovery_selections.json"}, indent=2))
