"""Correctness and latency smoke study for Paper 2.7 tensor primitives."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import git_metadata, write_csv, write_json  # noqa: E402
from pra_hf.query_graph import build_query_graph, graph_memory_bytes  # noqa: E402
from pra_hf.query_graph_cluster import (  # noqa: E402
    connected_components,
    threshold_filtration,
    weighted_label_propagation,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    parity = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for count in (8, 16, 32, 64, 128):
        source = torch.randn((count, args.width), generator=generator)
        cpu_labels = {}
        for top_k in (2, 4, 8, 16, 32):
            for device in devices:
                hidden = source.to(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                _sync(device)
                started = time.perf_counter()
                graph = build_query_graph(
                    hidden,
                    top_k=top_k,
                    threshold=args.threshold,
                    policy="union",
                )
                _sync(device)
                graph_ms = (time.perf_counter() - started) * 1000.0
                for method, cluster in (
                    ("cc", connected_components),
                    ("label_propagation", weighted_label_propagation),
                ):
                    _sync(device)
                    started = time.perf_counter()
                    result = cluster(graph)
                    _sync(device)
                    cluster_ms = (time.perf_counter() - started) * 1000.0
                    key = (count, top_k, method)
                    if device.type == "cpu":
                        cpu_labels[key] = result.labels
                    else:
                        parity.append(
                            {
                                "nodes": count,
                                "top_k": min(top_k, count - 1),
                                "method": method,
                                "exact_label_match": int(
                                    torch.equal(cpu_labels[key], result.labels.cpu())
                                ),
                            }
                        )
                    rows.append(
                        {
                            "device": str(device),
                            "nodes": count,
                            "top_k": min(top_k, count - 1),
                            "edges": graph.edge_count,
                            "method": method,
                            "clusters": result.cluster_count,
                            "iterations": result.iterations,
                            "converged": int(result.converged),
                            "graph_ms": graph_ms,
                            "cluster_ms": cluster_ms,
                            "graph_bytes": graph_memory_bytes(graph),
                            "peak_gpu_bytes": (
                                int(torch.cuda.max_memory_allocated(device))
                                if device.type == "cuda"
                                else 0
                            ),
                        }
                    )
        filtration = threshold_filtration(
            build_query_graph(source, top_k=min(8, count - 1), policy="union"),
            (0.0, 0.25, 0.5, 0.75),
        )
        if any(
            right.result.cluster_count < left.result.cluster_count
            for left, right in zip(filtration, filtration[1:])
        ):
            raise AssertionError("Threshold filtration merged components.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "primitive_benchmark.csv", rows)
    write_csv(args.output_dir / "cpu_cuda_parity.csv", parity)
    findings = {
        "schema_version": "1.0",
        "git": git_metadata(),
        "seed": args.seed,
        "width": args.width,
        "threshold": args.threshold,
        "devices": [str(value) for value in devices],
        "rows": len(rows),
        "cpu_cuda_exact_parity": all(row["exact_label_match"] for row in parity),
        "filtration_monotonic": True,
        "claim": "minimal tensor-native baseline, not a fastest-kernel claim",
    }
    write_json(args.output_dir / "primitive_findings.json", findings)
    return findings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_7_query_graph/primitives",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
