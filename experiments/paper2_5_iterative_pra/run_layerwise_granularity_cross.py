"""Run the gated early/middle/late layer by parent-granularity cross."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.run_natural_graph_depth import (
    _feature_example,
    _node_recovery,
    _search,
    _strict_path_survival,
    _transition_rows,
)
from pra_hf.natural_reasoning_graph import map_example_to_parents
from pra_hf.semantic_graph_search import build_native_parent_adjacency


LAYERS = (0, 12, 27)
CHUNKS = (32, 128, 256)
K = 6
B = 16
H = 4


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field, "") not in ("", None)]
    return statistics.fmean(values) if values else float("nan")


def _native(layer_feature, chunk):
    spans = [tuple(map(int, span)) for span in layer_feature["local_spans"]]
    return (
        layer_feature["local_pre_query"],
        layer_feature["local_pre_key"],
        layer_feature["local_token_mask"],
        torch.tensor([start // chunk for start, _ in spans], dtype=torch.long),
    )


def _minimum_depth(scores, roots, mapping):
    minimum = None
    final_recall, final_complete = 0.0, False
    for hops in range(H + 1):
        result = _search(scores, roots, K, hops, B)
        recall, complete, _ = _node_recovery(result.visited, mapping)
        if complete and minimum is None:
            minimum = hops
        if hops == H:
            final_recall, final_complete = recall, complete
    return minimum, final_recall, final_complete


def _plots(output, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for layer in LAYERS:
        wiki = [
            row for row in summary
            if row["dataset"] == "2wikimultihopqa" and row["layer"] == layer
        ]
        musique = [
            row for row in summary
            if row["dataset"] == "musique_D4" and row["layer"] == layer
        ]
        axes[0].plot(
            [row["chunk_size"] for row in wiki],
            [row["edge_R6"] for row in wiki], marker="o", label=f"layer {layer}"
        )
        axes[1].plot(
            [row["chunk_size"] for row in musique],
            [row["complete_recovery"] for row in musique], marker="o", label=f"layer {layer}"
        )
    axes[0].set(xlabel="Search chunk (tokens)", ylabel="2Wiki preserved-edge R@6")
    axes[1].set(xlabel="Search chunk (tokens)", ylabel="MuSiQue D=4 complete recovery")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "layer_granularity_cross.png", dpi=180)
    plt.close(fig)


def run(args):
    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows, transitions = [], []
    for index, entry in enumerate(manifest["entries"], start=1):
        feature = torch.load(
            args.output_dir / entry["path"], map_location="cpu", weights_only=False
        )
        example = _feature_example({**feature, "question": feature.get("question", "")})
        for layer in LAYERS:
            for chunk in CHUNKS:
                mapping = map_example_to_parents(
                    example,
                    int(feature["source_tokens"]),
                    feature["node_token_spans"],
                    chunk_size=chunk,
                )
                q, k, mask, parents = _native(feature["layers"][layer], chunk)
                adjacency = build_native_parent_adjacency(
                    q.to(device),
                    k.to(device),
                    mask.to(device),
                    parents.to(device),
                    len(mapping.parent_spans),
                    token_reduction="top_m_mean",
                    head_reduction="top_m_mean",
                    top_m=4,
                )
                scores = adjacency.scores.detach().cpu()
                edge_rows = _transition_rows(feature, example, mapping, scores)
                for edge in edge_rows:
                    edge.update({"layer": layer, "chunk_size": chunk})
                transitions.extend(edge_rows)
                minimum, recall, complete = _minimum_depth(
                    scores, mapping.root_parent_ids, mapping
                )
                rows.append(
                    {
                        "dataset": feature["dataset"],
                        "example_id": feature["example_id"],
                        "partition": feature["partition"],
                        "annotated_hops": feature["annotated_hops"],
                        "layer": layer,
                        "chunk_size": chunk,
                        "minimum_native_depth": "" if minimum is None else minimum,
                        "complete_recovery": int(complete),
                        "node_recall": recall,
                        "parent_count": len(mapping.parent_spans),
                    }
                )
        print(
            f"[layer-chunk {index}/{len(manifest['entries'])}] "
            f"{feature['dataset']} {feature['example_id']}", flush=True
        )
    summary = []
    for layer in LAYERS:
        for chunk in CHUNKS:
            wiki = [
                row for row in transitions
                if row["dataset"] == "2wikimultihopqa"
                and row["partition"] == "test"
                and row["layer"] == layer
                and row["chunk_size"] == chunk
                and row["mapping_status"] == "preserved"
            ]
            wiki_all = [
                row for row in transitions
                if row["dataset"] == "2wikimultihopqa"
                and row["partition"] == "test"
                and row["layer"] == layer
                and row["chunk_size"] == chunk
            ]
            summary.append(
                {
                    "dataset": "2wikimultihopqa",
                    "layer": layer,
                    "chunk_size": chunk,
                    "edge_R6": _mean(wiki, "recovered_at_6"),
                    "edge_MRR": _mean(wiki, "reciprocal_rank"),
                    "strict_path_K6": _strict_path_survival(wiki_all, 6),
                    "preserved_transitions": len(wiki),
                }
            )
            for depth in (2, 3, 4):
                selected = [
                    row for row in rows
                    if row["dataset"] == "musique"
                    and row["partition"] == "test"
                    and row["layer"] == layer
                    and row["chunk_size"] == chunk
                    and int(row["annotated_hops"]) == depth
                ]
                summary.append(
                    {
                        "dataset": f"musique_D{depth}",
                        "layer": layer,
                        "chunk_size": chunk,
                        "complete_recovery": _mean(selected, "complete_recovery"),
                        "node_recall": _mean(selected, "node_recall"),
                        "minimum_native_depth_if_complete": _mean(
                            selected, "minimum_native_depth"
                        ),
                        "examples": len(selected),
                    }
                )
    canonical = next(
        row for row in summary
        if row["dataset"] == "2wikimultihopqa"
        and row["layer"] == 27
        and row["chunk_size"] == 128
    )
    if not math.isclose(canonical["edge_R6"], 0.88, abs_tol=1e-12):
        raise AssertionError(f"Layer-27/chunk-128 reproduction failed: {canonical}")
    _write_csv(args.output_dir / "layer_granularity_rows.csv", rows)
    _write_csv(args.output_dir / "layer_granularity_transition_rows.csv", transitions)
    _write_csv(args.output_dir / "layer_granularity_summary.csv", summary)
    _plots(args.output_dir, summary)
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layers": list(LAYERS),
        "chunk_sizes": list(CHUNKS),
        "K": K,
        "B": B,
        "H": H,
        "selection": "a-priori early/middle/late structural representatives",
        "canonical_layer27_chunk128_exact": True,
        "summary": summary,
    }
    (args.output_dir / "layer_granularity_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifact


def parse_args():
    parser = argparse.ArgumentParser()
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/layerwise_graph"
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--manifest", type=Path, default=output / "layerwise_feature_manifest.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
