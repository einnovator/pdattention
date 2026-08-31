"""Build the shared selector-frozen manifest and engine maturity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pra_hf.engine_qualification import FrozenSelection, QualificationManifest


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/engine_qualification"


ENGINE_MATURITY = (
    {
        "engine": "vLLM",
        "highest_validated_level": "E2",
        "highest_observed_level": "E2",
        "level_status": "NATURAL_WORKLOAD",
        "quality_frontier": "NATURAL_WORKLOAD",
        "memory_telemetry": "CONTROLLED",
        "tail_telemetry": "MEASURED",
        "distributed_storage": "NOT_MEASURED",
        "main_gate": "CUDA-native page attachment and LMCache under continuous batching",
        "provenance": "docs/papers/paper6_vllm/paper.pdf",
    },
    {
        "engine": "SGLang",
        "highest_validated_level": "E2",
        "highest_observed_level": "E3",
        "level_status": "NATURAL_WORKLOAD",
        "quality_frontier": "NATURAL_WORKLOAD",
        "memory_telemetry": "CONTROLLED",
        "tail_telemetry": "MODEL_BACKED",
        "distributed_storage": "CANDIDATE",
        "main_gate": "Scheduler-owned distributed HiCache and off-node tier curves",
        "provenance": "docs/papers/paper6_1_sglang/paper.pdf",
    },
    {
        "engine": "MLX",
        "highest_validated_level": "E2",
        "highest_observed_level": "E2",
        "level_status": "NATURAL_WORKLOAD",
        "quality_frontier": "NATURAL_WORKLOAD",
        "memory_telemetry": "MODEL_BACKED",
        "tail_telemetry": "MODEL_BACKED",
        "distributed_storage": "NOT_APPLICABLE",
        "main_gate": "Live fused one-softmax attention and memory-pressure frontier",
        "provenance": "docs/papers/paper6_2_mlx/paper.pdf",
    },
    {
        "engine": "OpenVINO",
        "highest_validated_level": "E0",
        "highest_observed_level": "E0",
        "level_status": "NATURAL_WORKLOAD",
        "quality_frontier": "NATURAL_WORKLOAD",
        "memory_telemetry": "NOT_MEASURED",
        "tail_telemetry": "MEASURED",
        "distributed_storage": "NOT_APPLICABLE",
        "main_gate": "Larger distractor cohort, then a feasible native-attention seam",
        "provenance": "research/paper6-3-openvino",
    },
    {
        "engine": "TensorRT-LLM",
        "highest_validated_level": "E0",
        "highest_observed_level": "E1",
        "level_status": "MEASURED",
        "quality_frontier": "CONTROLLED",
        "memory_telemetry": "MEASURED",
        "tail_telemetry": "MEASURED",
        "distributed_storage": "CANDIDATE",
        "main_gate": "Official KV connector plus native selected K/V",
        "provenance": "research/paper6-4-tensorrt-llm",
    },
    {
        "engine": "AirLLM",
        "highest_validated_level": "E0",
        "highest_observed_level": "E2",
        "level_status": "CONTROLLED",
        "quality_frontier": "CONTROLLED",
        "memory_telemetry": "CONTROLLED",
        "tail_telemetry": "NOT_MEASURED",
        "distributed_storage": "NOT_APPLICABLE",
        "main_gate": "Natural QA on a model materially larger than accelerator VRAM",
        "provenance": "research/paper6-6-airllm@67549ed",
    },
)


def build_manifest() -> QualificationManifest:
    selection = FrozenSelection.create(
        example_id="synthetic-codeword-0001",
        query="What is the requested verification code?",
        candidate_ids=("resource:target", "resource:distractor-1"),
        selected_ids=("resource:target",),
        selected_intervals=(("resource:target", 0, 18),),
    )
    return QualificationManifest(
        manifest_id="pra-engine-qualification-2026-08-v1",
        selections=(selection,),
    )


def write_maturity_json(path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "evidence_vocabulary": [
            "MEASURED", "CONTROLLED", "MODEL_BACKED", "NATURAL_WORKLOAD",
            "CANDIDATE", "BLOCKED", "NOT_MEASURED", "NOT_APPLICABLE",
        ],
        "rows": list(ENGINE_MATURITY),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_maturity_table(path: Path) -> None:
    compact = {
        "NATURAL_WORKLOAD": "natural",
        "MODEL_BACKED": "model",
        "MEASURED": "measured",
        "CONTROLLED": "controlled",
        "CANDIDATE": "candidate",
        "BLOCKED": "blocked",
        "NOT_MEASURED": "not measured",
        "NOT_APPLICABLE": "n/a",
    }
    lines = [
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Engine & Validated & Observed & Quality & Memory & Tail & Distributed \\\\",
        "\\midrule",
    ]
    for row in ENGINE_MATURITY:
        lines.append(
            f"{row['engine']} & {row['highest_validated_level']} & "
            f"{row['highest_observed_level']} & {compact[str(row['quality_frontier'])]} & "
            f"{compact[str(row['memory_telemetry'])]} & "
            f"{compact[str(row['tail_telemetry'])]} & "
            f"{compact[str(row['distributed_storage'])]} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_maturity_figure(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    columns = ("E0", "E1", "E2", "E3", "Quality", "Memory", "Tail", "Distributed")
    status_value = {
        "NOT_APPLICABLE": 0,
        "NOT_MEASURED": 0,
        "BLOCKED": 1,
        "CANDIDATE": 1,
        "CONTROLLED": 2,
        "MODEL_BACKED": 3,
        "MEASURED": 3,
        "NATURAL_WORKLOAD": 4,
    }
    level_index = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}
    values = []
    for row in ENGINE_MATURITY:
        validated = level_index[str(row["highest_validated_level"])]
        observed = level_index[str(row["highest_observed_level"])]
        integration = [4 if index <= validated else (2 if index <= observed else 0) for index in range(4)]
        values.append(integration + [
            status_value[str(row["quality_frontier"])],
            status_value[str(row["memory_telemetry"])],
            status_value[str(row["tail_telemetry"])],
            status_value[str(row["distributed_storage"])],
        ])
    figure, axis = plt.subplots(figsize=(9.2, 3.7))
    cmap = ListedColormap(("#eeeeee", "#d9822b", "#e6bd45", "#4c8fc8", "#317a4b"))
    axis.imshow(values, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    axis.set_xticks(range(len(columns)), labels=columns)
    axis.set_yticks(range(len(ENGINE_MATURITY)), labels=[row["engine"] for row in ENGINE_MATURITY])
    axis.set_xlabel("Evidence maturity, not raw throughput")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_manifest().write(args.output_dir / "qualification_manifest.json")
    write_maturity_json(args.output_dir / "engine_maturity.json")
    write_maturity_table(args.output_dir / "generated_engine_maturity.tex")
    write_maturity_figure(args.output_dir / "engine_maturity.png")
    print(json.dumps({
        "manifest": str(args.output_dir / "qualification_manifest.json"),
        "engines": len(ENGINE_MATURITY),
    }, indent=2))


if __name__ == "__main__":
    main()
