"""Generate Paper 4.5 tables, figures, macros, and findings."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results"
OUTPUT = RESULTS / "paper4_5_runtime"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(rows, study, mode, device, batch):
    return next(
        row
        for row in rows
        if row["study"] == study
        and row["mode"] == mode
        and row["device"] == device
        and int(row["batch"]) == batch
    )


def _save_figure(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cpu = _json(OUTPUT / "cpu" / "runtime_benchmark.json")
    cuda = _json(OUTPUT / "cuda" / "runtime_benchmark.json")
    combined = [*cpu["summary"], *cuda["summary"]]
    execution_summary = json.loads(
        (OUTPUT / "execution_policy_summary.json").read_text(encoding="utf-8")
    )
    execution = {row["condition"]: row for row in execution_summary}
    _write_csv(OUTPUT / "runtime_combined_summary.csv", combined)

    qwen = _json(RESULTS / "paper2_hf" / "productization" / "qwen_product_demo.json")
    lifecycle = _json(
        RESULTS / "paper4_training" / "external_memory" / "external_memory_findings.json"
    )
    qwen_combined = qwen["aggregates"]["combined"]
    inherited = [
        {"source": "paper2_qwen_product_demo", "metric": key, "value": value}
        for key, value in qwen_combined.items()
    ] + [
        {"source": "paper4_external_lifecycle", "metric": f"ttft_proxy_ms_{phase}", "value": value}
        for phase, value in lifecycle["mean_ttft_proxy_ms"].items()
    ]
    _write_csv(OUTPUT / "inherited_metrics.csv", inherited)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = []
    values = []
    colors = []
    for device, color in (("cpu", "#287271"), ("cuda", "#d97732")):
        for study, short in (("indexed_gather", "index"), ("interval_pack", "pack")):
            for batch in (1, 4):
                row = _row(combined, study, "eager_warm", device, batch)
                labels.append(f"{device}\n{short} B{batch}")
                values.append(1000 * row["median_seconds"])
                colors.append(color)
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Portable selected-K/V primitives")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    _save_figure(fig, "portable_primitive_latency")

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    hierarchy_modes = ("hbm_resident", "pinned_cpu_to_cuda", "ordinary_cpu_to_cuda")
    hierarchy_labels = ("HBM resident", "Pinned CPU", "Ordinary CPU")
    palette = ("#2a9d8f", "#e9c46a", "#c14953")
    x = range(len(hierarchy_modes))
    width = 0.36
    for offset, batch in ((-width / 2, 1), (width / 2, 4)):
        values = [
            1000 * _row(combined, "memory_hierarchy", mode, "cuda", batch)["median_seconds"]
            for mode in hierarchy_modes
        ]
        bars = ax.bar([value + offset for value in x], values, width, label=f"batch {batch}", color=palette, alpha=1.0 if batch == 1 else 0.62)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x), hierarchy_labels)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Selected-K/V residency and transfer")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, "memory_hierarchy_latency")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    layout_modes = ("layer_major", "reference_major", "chunk_major", "block_major")
    layout_labels = ("Layer", "Reference", "Chunk", "Block")
    layout_colors = ("#5b5f97", "#2a9d8f", "#e9c46a", "#c14953")
    for ax, device in zip(axes, ("cpu", "cuda")):
        x = range(len(layout_modes))
        width = 0.36
        for offset, batch in ((-width / 2, 1), (width / 2, 4)):
            values = [
                1000 * _row(combined, "layout_gather", mode, device, batch)["median_seconds"]
                for mode in layout_modes
            ]
            ax.bar(
                [value + offset for value in x],
                values,
                width,
                label=f"batch {batch}",
                color=layout_colors,
                alpha=1.0 if batch == 1 else 0.62,
            )
        ax.set_xticks(list(x), layout_labels, rotation=18)
        ax.set_title(device.upper())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Median warm gather (ms)")
    axes[1].legend(frameon=False)
    fig.suptitle("Contiguous physical K/V layouts with logical remapping")
    _save_figure(fig, "layout_gather_latency")

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    stages = ("ingest_seconds", "query_encoding_seconds", "routing_seconds", "routed_wall_seconds")
    stage_labels = ("Reference ingest", "Query encode", "Route", "Generate wall")
    stage_values = [qwen_combined[name] for name in stages]
    bars = ax.bar(stage_labels, stage_values, color=("#5b5f97", "#ffc145", "#2a9d8f", "#c14953"))
    ax.set_yscale("log")
    ax.set_ylabel("Mean seconds, log scale")
    ax.set_title("Inherited Qwen3-0.6B public-API profile (4 examples)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, stage_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    _save_figure(fig, "inherited_end_to_end_profile")

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    phases = ("cold", "warm", "hot")
    phase_values = [lifecycle["mean_ttft_proxy_ms"][phase] for phase in phases]
    bars = ax.bar(("Cold", "Warm", "Hot"), phase_values, color=("#c14953", "#e9c46a", "#2a9d8f"))
    ax.set_ylabel("Mean TTFT proxy (ms)")
    ax.set_title("Inherited authenticated memory lifecycle (8 examples)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, phase_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    _save_figure(fig, "cold_warm_hot_profile")

    cuda_index_b1 = _row(combined, "indexed_gather", "eager_warm", "cuda", 1)
    cuda_index_b4 = _row(combined, "indexed_gather", "eager_warm", "cuda", 4)
    cuda_pack_b1 = _row(combined, "interval_pack", "eager_warm", "cuda", 1)
    cuda_pack_b4 = _row(combined, "interval_pack", "eager_warm", "cuda", 4)
    cpu_index_b1 = _row(combined, "indexed_gather", "eager_warm", "cpu", 1)
    cpu_index_b4 = _row(combined, "indexed_gather", "eager_warm", "cpu", 4)
    cpu_pack_b1 = _row(combined, "interval_pack", "eager_warm", "cpu", 1)
    cpu_pack_b4 = _row(combined, "interval_pack", "eager_warm", "cpu", 4)
    hbm_b1 = _row(combined, "memory_hierarchy", "hbm_resident", "cuda", 1)
    hbm_b4 = _row(combined, "memory_hierarchy", "hbm_resident", "cuda", 4)
    pinned_b1 = _row(combined, "memory_hierarchy", "pinned_cpu_to_cuda", "cuda", 1)
    pinned_b4 = _row(combined, "memory_hierarchy", "pinned_cpu_to_cuda", "cuda", 4)
    ordinary_b1 = _row(combined, "memory_hierarchy", "ordinary_cpu_to_cuda", "cuda", 1)
    ordinary_b4 = _row(combined, "memory_hierarchy", "ordinary_cpu_to_cuda", "cuda", 4)
    cuda_layout_b1 = {
        mode: _row(combined, "layout_gather", mode, "cuda", 1)
        for mode in ("layer_major", "reference_major", "chunk_major", "block_major")
    }
    cuda_layout_b4 = {
        mode: _row(combined, "layout_gather", mode, "cuda", 4)
        for mode in ("layer_major", "reference_major", "chunk_major", "block_major")
    }
    agent_plugins = json.loads(
        (OUTPUT / "agent_plugin_contract_summary.json").read_text(encoding="utf-8")
    )
    findings = {
        "status": "portable_eager_measured_compile_and_engine_gates_negative",
        "cuda_indexed_gather_ms_batch1": 1000 * cuda_index_b1["median_seconds"],
        "cuda_interval_pack_ms_batch1": 1000 * cuda_pack_b1["median_seconds"],
        "cuda_hbm_pack_ms_batch1": 1000 * hbm_b1["median_seconds"],
        "cuda_pinned_transfer_ms_batch1": 1000 * pinned_b1["median_seconds"],
        "cuda_ordinary_transfer_ms_batch1": 1000 * ordinary_b1["median_seconds"],
        "pinned_over_hbm_latency_ratio": pinned_b1["median_seconds"] / hbm_b1["median_seconds"],
        "ordinary_over_hbm_latency_ratio": ordinary_b1["median_seconds"] / hbm_b1["median_seconds"],
        "cuda_layout_fastest_batch1": min(
            cuda_layout_b1, key=lambda mode: cuda_layout_b1[mode]["median_seconds"]
        ),
        "cuda_layout_slowest_over_fastest_batch1": max(
            row["median_seconds"] for row in cuda_layout_b1.values()
        ) / min(row["median_seconds"] for row in cuda_layout_b1.values()),
        "cuda_layout_fastest_batch4": min(
            cuda_layout_b4, key=lambda mode: cuda_layout_b4[mode]["median_seconds"]
        ),
        "cuda_layout_slowest_over_fastest_batch4": max(
            row["median_seconds"] for row in cuda_layout_b4.values()
        ) / min(row["median_seconds"] for row in cuda_layout_b4.values()),
        "torch_compile_cpu": "unsupported_missing_msvc",
        "torch_compile_cuda": "unsupported_compute_capability_5_0",
        "optional_engines_measured": [],
        "paper2_inherited_examples": qwen_combined["examples"],
        "paper2_inherited_routing_seconds": qwen_combined["routing_seconds"],
        "paper4_lifecycle_examples": lifecycle["examples"],
        "execution_policy_seeds": execution["request_shared_last"]["seeds"],
        "token_shared_routing_operations": execution["token_shared_first"][
            "routing_operations_mean"
        ],
        "token_per_layer_routing_operations": execution["token_per_layer"][
            "routing_operations_mean"
        ],
        "agent_plugin_cases": sum(row["cases"] for row in agent_plugins.values()),
        "deepseek_agent_contract_pass_rate": agent_plugins["deepseek_harness"][
            "contract_pass_rate"
        ],
        "pi_agent_contract_pass_rate": agent_plugins["pi_coding_agent"][
            "contract_pass_rate"
        ],
    }
    (OUTPUT / "findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    capabilities = cuda["capabilities"]
    compatibility = [
        {"runtime": "HF/PyTorch eager", "status": "measured", "boundary": "native PRA model and portable K/V primitives"},
        {"runtime": "Standalone gateway", "status": "contract tested", "boundary": "G00/G10/G01/G11 JSON mediation; non-streaming"},
        {"runtime": "OpenAI-compatible HTTP", "status": "E0 implemented", "boundary": "pass-through and explicit text fallback"},
        {"runtime": "DeepSeek/Pi agent bridges", "status": "contract tested", "boundary": "typed event/RPC capture and explicit G10 fallback"},
        {"runtime": "torch.compile", "status": "blocked on host", "boundary": "API implemented; CPU compiler absent and GPU too old for Triton"},
        {"runtime": "Triton/custom CUDA", "status": "not run", "boundary": "requires supported GPU/toolchain after profiling"},
        {"runtime": "vLLM thin", "status": "contract only", "boundary": "scheduler-unaware identity/KV handoff"},
        {"runtime": "SGLang/FreeToken", "status": "E0 feasible, not run", "boundary": "compatible HTTP; no native PRA backend"},
        {"runtime": "TensorRT-LLM/MLX", "status": "architectural only", "boundary": "not installed on measured host"},
    ]
    _write_csv(OUTPUT / "compatibility_matrix.csv", compatibility)
    manifest = {
        "protocol": cpu["protocol"],
        "cpu_capabilities": cpu["capabilities"],
        "cuda_capabilities": capabilities,
        "inherited_paper2_artifact": "paper2_hf/productization/qwen_product_demo.json",
        "inherited_paper4_artifact": "paper4_training/external_memory/external_memory_findings.json",
        "quality_metrics_recomputed": False,
        "engine_speed_claims": False,
        "execution_policy_profile": "five-seed tiny random-weight HF mechanism check",
        "agent_plugin_profile": "two public event vocabularies, five deterministic seeds each",
        "paper8_native_geometry_integrated": True,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    tex = rf"""% Generated by experiments.paper4_5_runtime.summarize_runtime
\newcommand{{\RuntimeCudaIndexBatchOneMs}}{{{1000 * cuda_index_b1['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaIndexBatchFourMs}}{{{1000 * cuda_index_b4['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaPackBatchOneMs}}{{{1000 * cuda_pack_b1['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaPackBatchFourMs}}{{{1000 * cuda_pack_b4['median_seconds']:.3f}}}
\newcommand{{\RuntimeCpuIndexBatchOneMs}}{{{1000 * cpu_index_b1['median_seconds']:.3f}}}
\newcommand{{\RuntimeCpuIndexBatchFourMs}}{{{1000 * cpu_index_b4['median_seconds']:.3f}}}
\newcommand{{\RuntimeCpuPackBatchOneMs}}{{{1000 * cpu_pack_b1['median_seconds']:.3f}}}
\newcommand{{\RuntimeCpuPackBatchFourMs}}{{{1000 * cpu_pack_b4['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaHBMMs}}{{{1000 * hbm_b1['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaHBMBatchFourMs}}{{{1000 * hbm_b4['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaPinnedMs}}{{{1000 * pinned_b1['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaPinnedBatchFourMs}}{{{1000 * pinned_b4['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaOrdinaryMs}}{{{1000 * ordinary_b1['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaOrdinaryBatchFourMs}}{{{1000 * ordinary_b4['median_seconds']:.3f}}}
\newcommand{{\RuntimeCudaIndexBatchFourThroughput}}{{{cuda_index_b4['selected_tokens_per_second'] / 1e6:.2f}}}
\newcommand{{\RuntimeCpuIndexBatchFourThroughput}}{{{cpu_index_b4['selected_tokens_per_second'] / 1e6:.2f}}}
\newcommand{{\RuntimePinnedRatio}}{{{findings['pinned_over_hbm_latency_ratio']:.2f}}}
\newcommand{{\RuntimeOrdinaryRatio}}{{{findings['ordinary_over_hbm_latency_ratio']:.2f}}}
\newcommand{{\RuntimeLayoutSpreadBatchOne}}{{{(findings['cuda_layout_slowest_over_fastest_batch1'] - 1) * 100:.1f}\%}}
\newcommand{{\RuntimeLayoutSpreadBatchFour}}{{{(findings['cuda_layout_slowest_over_fastest_batch4'] - 1) * 100:.1f}\%}}
\newcommand{{\RuntimeLayoutFastestBatchOne}}{{{findings['cuda_layout_fastest_batch1'].replace('_', '-')}}}
\newcommand{{\RuntimeLayoutFastestBatchFour}}{{{findings['cuda_layout_fastest_batch4'].replace('_', '-')}}}
\newcommand{{\InheritedRoutingMs}}{{{1000 * qwen_combined['routing_seconds']:.2f}}}
\newcommand{{\InheritedIngestSeconds}}{{{qwen_combined['ingest_seconds']:.2f}}}
\newcommand{{\InheritedGenerateSeconds}}{{{qwen_combined['routed_wall_seconds']:.2f}}}
\newcommand{{\LifecycleColdMs}}{{{lifecycle['mean_ttft_proxy_ms']['cold']:.2f}}}
\newcommand{{\LifecycleWarmMs}}{{{lifecycle['mean_ttft_proxy_ms']['warm']:.2f}}}
\newcommand{{\LifecycleHotMs}}{{{lifecycle['mean_ttft_proxy_ms']['hot']:.2f}}}
\newcommand{{\ExecutionPolicySeeds}}{{{execution['request_shared_last']['seeds']}}}
\newcommand{{\ExecutionRequestSharedMs}}{{{1000 * execution['request_shared_last']['generation_seconds_mean']:.1f}}}
\newcommand{{\ExecutionRequestPerLayerMs}}{{{1000 * execution['request_per_layer']['generation_seconds_mean']:.1f}}}
\newcommand{{\ExecutionTokenSharedMs}}{{{1000 * execution['token_shared_first']['generation_seconds_mean']:.1f}}}
\newcommand{{\ExecutionTokenPerLayerMs}}{{{1000 * execution['token_per_layer']['generation_seconds_mean']:.1f}}}
\newcommand{{\ExecutionTokenSharedRoutes}}{{{execution['token_shared_first']['routing_operations_mean']:.1f}}}
\newcommand{{\ExecutionTokenPerLayerRoutes}}{{{execution['token_per_layer']['routing_operations_mean']:.1f}}}
\newcommand{{\ExecutionRoutingReduction}}{{{execution['token_per_layer']['routing_operations_mean'] / execution['token_shared_first']['routing_operations_mean']:.1f}}}
"""
    (OUTPUT / "generated_runtime_results.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
