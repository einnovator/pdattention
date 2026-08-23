"""Evaluate natural query-facet recovery from frozen decoder states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import write_csv, write_json
from pra_hf.natural_query_facets import (
    NaturalFacetAnnotation,
    align_subquestions_to_units,
    evaluate_natural_partition,
    interleaving_statistics,
    scorable_labels,
)
from pra_hf.query_graph import QueryUnitProvenance, build_query_graph, graph_memory_bytes
from pra_hf.query_graph_cluster import connected_components, deterministic_kmeans, weighted_label_propagation


METHODS = ("global", "fixed_window", "syntax", "embedding_kmeans", "graph_cc", "graph_lp", "llm")


def _annotations(path: Path) -> list[NaturalFacetAnnotation]:
    return [
        NaturalFacetAnnotation.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _llm_map(path: Path | None) -> dict[tuple[str, str], list[str]]:
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    return {
        (str(row["dataset"]), str(row["example_id"])): list(row.get("subquestions") or ())
        for row in rows
    }


def _layer_candidates(layer_count: int) -> tuple[int, ...]:
    return tuple(sorted({max(1, layer_count // 2), max(1, 3 * layer_count // 4), layer_count}))


def _encode(model, tokenizer, annotation: NaturalFacetAnnotation, layers, device):
    encoded = tokenizer(
        annotation.question,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    result = {}
    for layer in layers:
        token_hidden = output.hidden_states[layer][0].float().cpu()
        units = []
        for unit in annotation.units:
            indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > start and end > unit.char_start and start < unit.char_end
            ]
            if not indices:
                centers = [((start + end) / 2, index) for index, (start, end) in enumerate(offsets) if end > start]
                indices = [min(centers, key=lambda row: abs(row[0] - unit.char_start))[1]]
            units.append(token_hidden[indices].mean(0))
        result[layer] = torch.stack(units)
    return result


def _fixed_labels(count: int, width: int) -> torch.Tensor:
    return torch.arange(count, dtype=torch.long) // width


def _syntax_labels(annotation: NaturalFacetAnnotation) -> torch.Tensor:
    labels = torch.zeros(len(annotation.units), dtype=torch.long)
    current = 0
    boundary = False
    for index, unit in enumerate(annotation.units):
        word = unit.text.casefold()
        if boundary and word not in {",", ";", ":", "and", "or", "then", "while", "whereas"}:
            current += 1
            boundary = False
        labels[index] = current
        if word in {",", ";", ":", "and", "or", "then", "while", "whereas"}:
            boundary = True
    return labels


def _graph(hidden: torch.Tensor, annotation: NaturalFacetAnnotation, *, top_k: int, threshold: float):
    provenance = tuple(
        QueryUnitProvenance(
            unit_id=unit.unit_id,
            token_start=unit.unit_id,
            token_end=unit.unit_id + 1,
            text=unit.text,
        )
        for unit in annotation.units
    )
    return build_query_graph(
        hidden,
        provenance=provenance,
        contextual_weight=1.0,
        top_k=top_k,
        threshold=threshold,
        policy="union",
    )


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def _paired_ci(rows, left: str, right: str, *, seed: int = 20260823, draws: int = 5000):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["dataset"], row["example_id"])][row["method"]] = float(row["ari"])
    deltas = [values[left] - values[right] for values in pairs.values() if left in values and right in values]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(draws))
    return {
        "left": left,
        "right": right,
        "examples": len(deltas),
        "mean_ari_delta": sum(deltas) / len(deltas),
        "ci95": [boot[int(0.025 * draws)], boot[int(0.975 * draws)]],
        "better": sum(value > 0 for value in deltas),
        "worse": sum(value < 0 for value in deltas),
        "unchanged": sum(value == 0 for value in deltas),
    }


def _geometry(hidden: torch.Tensor, annotation: NaturalFacetAnnotation, predicted: torch.Tensor) -> dict[str, float]:
    target, unit_ids = scorable_labels(annotation)
    values = F.normalize(hidden[unit_ids].float(), dim=-1)
    similarity = values @ values.T
    same = target[:, None] == target[None, :]
    eye = torch.eye(len(target), dtype=torch.bool)
    within = similarity[same & ~eye]
    between = similarity[~same]
    neighbor = similarity.masked_fill(eye, -2).argmax(1)
    local_purity = (target[neighbor] == target).float().mean()
    predicted = predicted[unit_ids]
    component_purity = []
    for label in torch.unique(predicted):
        members = target[predicted == label]
        counts = torch.bincount(members)
        component_purity.append(float(counts.max() / counts.sum()))
    centroids = torch.stack([values[target == label].mean(0) for label in torch.unique(target)])
    centroid_separation = 1.0 if len(centroids) == 1 else float(1.0 - (F.normalize(centroids, dim=-1) @ F.normalize(centroids, dim=-1).T)[~torch.eye(len(centroids), dtype=torch.bool)].mean())
    return {
        "within_cosine": float(within.mean()) if within.numel() else 1.0,
        "between_cosine": float(between.mean()) if between.numel() else 0.0,
        "local_neighbor_purity": float(local_purity),
        "centroid_separation": centroid_separation,
        "graph_component_purity": sum(component_purity) / len(component_purity),
    }


def _select(validation, features, layer_candidates):
    window_scores = {}
    for width in (2, 3, 4, 5):
        window_scores[width] = _mean([
            evaluate_natural_partition(_fixed_labels(len(row.units), width), row)
            for row in validation
        ], "ari")
    fixed_width = max(window_scores, key=lambda value: (window_scores[value], -value))
    candidates = []
    for layer in layer_candidates:
        for top_k in (1, 2, 3):
            for threshold in (0.35, 0.45, 0.55, 0.65):
                metrics = []
                for annotation in validation:
                    hidden = features[(annotation.dataset, annotation.example_id)][layer]
                    labels = connected_components(_graph(hidden, annotation, top_k=top_k, threshold=threshold)).labels
                    metrics.append(evaluate_natural_partition(labels, annotation))
                candidates.append({"layer": layer, "top_k": top_k, "threshold": threshold, "mean_ari": _mean(metrics, "ari"), "mean_pairwise_f1": _mean(metrics, "pairwise_f1")})
    selected = max(candidates, key=lambda row: (row["mean_ari"], row["mean_pairwise_f1"], -row["top_k"], row["threshold"]))
    return fixed_width, window_scores, selected, candidates


def run(args):
    annotations = _annotations(args.annotations)
    llm = _llm_map(args.llm_predictions)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_path = args.output_dir / "natural_query_features.pt"
    encode_rows = []
    if cache_path.exists() and not args.refresh_features:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("model_id") != args.model_id:
            raise ValueError("The cached natural-query features belong to another model.")
        features = cached["features"]
        layers = tuple(sorted(next(iter(features.values())).keys()))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision or None)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision or None, torch_dtype=dtype).to(device).eval()
        layer_count = int(model.config.num_hidden_layers)
        layers = _layer_candidates(layer_count)
        features = {}
        for index, annotation in enumerate(annotations, 1):
            started = time.perf_counter()
            features[(annotation.dataset, annotation.example_id)] = _encode(model, tokenizer, annotation, layers, device)
            encode_rows.append({"dataset": annotation.dataset, "example_id": annotation.example_id, "encode_ms": (time.perf_counter() - started) * 1000.0})
            if index % 40 == 0:
                print(f"encoded {index}/{len(annotations)}", flush=True)
    validation = [row for row in annotations if row.split == "validation"]
    test = [row for row in annotations if row.split == "test"]
    fixed_width, window_scores, selected, selection_rows = _select(validation, features, layers)
    layer = int(selected["layer"])
    rows = []
    geometry_rows = []
    for annotation in test:
        hidden = features[(annotation.dataset, annotation.example_id)][layer]
        graph_started = time.perf_counter()
        graph = _graph(hidden, annotation, top_k=int(selected["top_k"]), threshold=float(selected["threshold"]))
        cc = connected_components(graph)
        graph_ms = (time.perf_counter() - graph_started) * 1000.0
        method_labels = {
            "global": torch.zeros(len(annotation.units), dtype=torch.long),
            "fixed_window": _fixed_labels(len(annotation.units), fixed_width),
            "syntax": _syntax_labels(annotation),
            "embedding_kmeans": deterministic_kmeans(hidden, max(1, min(6, round(math.sqrt(len(annotation.units) / 2.0))))).labels,
            "graph_cc": cc.labels,
            "graph_lp": weighted_label_propagation(graph).labels,
        }
        key = (annotation.dataset, annotation.example_id)
        if key in llm:
            method_labels["llm"] = align_subquestions_to_units(annotation, llm[key])
        strata = interleaving_statistics(annotation)
        for method, labels in method_labels.items():
            metric = evaluate_natural_partition(labels, annotation)
            rows.append({
                "model_id": args.model_id,
                "dataset": annotation.dataset,
                "example_id": annotation.example_id,
                "method": method,
                "units": len(annotation.units),
                "target_facets": len(annotation.source_facets),
                "predicted_facets": int(torch.unique(labels).numel()),
                **strata,
                **metric,
                "decomposition_ms": graph_ms if method.startswith("graph_") else 0.0,
                "graph_edges": graph.edge_count if method.startswith("graph_") else 0,
                "graph_bytes": graph_memory_bytes(graph) if method.startswith("graph_") else 0,
            })
        geometry_rows.append({"model_id": args.model_id, "dataset": annotation.dataset, "example_id": annotation.example_id, **_geometry(hidden, annotation, cc.labels)})
    summary = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for method in METHODS:
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if not group:
                continue
            summary.append({"model_id": args.model_id, "dataset": dataset, "method": method, "examples": len(group), **{f"mean_{key}": _mean(group, key) for key in ("ari", "nmi", "pairwise_f1", "facet_count_abs_error", "over_segmented", "under_segmented", "singleton_rate", "decomposition_ms")}})
    comparisons = [_paired_ci(rows, "graph_cc", baseline) for baseline in ("fixed_window", "syntax", "embedding_kmeans")]
    if any(row["method"] == "llm" for row in rows):
        comparisons.append(_paired_ci(rows, "graph_cc", "llm"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "natural_facet_rows.csv", rows)
    write_csv(args.output_dir / "natural_facet_summary.csv", summary)
    write_csv(args.output_dir / "natural_geometry_rows.csv", geometry_rows)
    write_csv(args.output_dir / "validation_graph_sweep.csv", selection_rows)
    write_csv(args.output_dir / "encoding_latency.csv", encode_rows)
    findings = {
        "schema_version": "1.0",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "device": str(device),
        "examples": len(annotations),
        "test_examples": len(test),
        "layer_candidates": list(layers),
        "fixed_window_validation_scores": window_scores,
        "selected_fixed_window": fixed_width,
        "selected_graph_policy": selected,
        "summary": summary,
        "paired_ari": comparisons,
        "mean_geometry": {key: _mean(geometry_rows, key) for key in ("within_cosine", "between_cosine", "local_neighbor_purity", "centroid_separation", "graph_component_purity")},
        "llm_predictions_available": len(llm),
    }
    write_json(args.output_dir / "natural_facet_findings.json", findings)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"global": "#6c757d", "fixed_window": "#d98c10", "syntax": "#b85c38", "embedding_kmeans": "#7a5195", "graph_cc": "#2878b5", "graph_lp": "#2a9d8f", "llm": "#3d405b"}
    for axis, dataset in zip(axes, sorted({row["dataset"] for row in summary})):
        group = [row for row in summary if row["dataset"] == dataset]
        axis.bar(range(len(group)), [row["mean_ari"] for row in group], color=[colors[row["method"]] for row in group])
        axis.set_xticks(range(len(group)), [row["method"].replace("_", "\n") for row in group])
        axis.set_title(dataset)
        axis.set_ylabel("Adjusted Rand index")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "natural_facet_quality.pdf")
    fig.savefig(args.output_dir / "natural_facet_quality.png", dpi=180)
    plt.close(fig)
    if not cache_path.exists() or args.refresh_features:
        torch.save({"model_id": args.model_id, "selected_layer": layer, "annotations": [row.to_dict() for row in annotations], "features": features}, cache_path)
    return findings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl")
    parser.add_argument("--llm-predictions", type=Path)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
