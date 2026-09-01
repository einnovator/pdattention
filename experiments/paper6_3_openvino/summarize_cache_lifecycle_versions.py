"""Summarize the process-isolated OpenVINO cache lifecycle matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(input_dir: Path, output_dir: Path) -> Mapping[str, object]:
    artifacts = sorted(input_dir.glob("openvino_*_*.json"))
    if not artifacts:
        raise RuntimeError(f"No lifecycle artifacts found under {input_dir}")

    rows = []
    for path in artifacts:
        payload = _load(path)
        rows.append(
            {
                "version": payload["packages"]["openvino-genai"],
                "cache_mode": payload["cache_mode"],
                "scenario": payload["scenario"],
                "status": payload["measurement_status"],
                "completed_steps": payload["completed_steps"],
                "requested_steps": payload["requested_steps"],
                "all_expected_answers": payload["all_expected_answers"],
                "artifact": path.name,
            }
        )
    rows.sort(key=lambda row: (row["version"], row["cache_mode"], row["scenario"]))

    labels = {
        "short": "short repeated",
        "long": "long once",
        "long_short": r"long $\rightarrow$ short",
        "long_long": r"long $\rightarrow$ long",
    }
    tex = [
        r"\begin{tabular}{llllrr}\toprule",
        r"GenAI & Cache & Workload & Status & Completed & Quality\\\midrule",
    ]
    for row in rows:
        status = str(row["status"]).replace("NOT_RUN_CACHE_", "").lower()
        tex.append(
            "{} & {} & {} & {} & {}/{} & {} \\\\".format(
                row["version"],
                row["cache_mode"],
                labels[row["scenario"]],
                status,
                row["completed_steps"],
                row["requested_steps"],
                "yes" if row["all_expected_answers"] else "--",
            )
        )
    tex.append(r"\bottomrule\end{tabular}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generated_cache_lifecycle_version_table.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "1.0",
        "experiment": "paper6_3_openvino_cache_lifecycle_versions_v1",
        "evidence_tier": "LIVE_ENGINE_LIFECYCLE_REGRESSION",
        "rows": rows,
    }
    (output_dir / "cache_lifecycle_version_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.input_dir, args.output_dir)
    print(json.dumps({"rows": len(payload["rows"]), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
