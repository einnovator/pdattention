"""Resume the five-seed Paper 6.5 model-choice curve through K=32."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper6_5_tools.run_missing_experiments import (
    OllamaLabelModel,
    _choice_prompt,
)


MISSING = ROOT / "docs/papers/shared/results/paper6_5_tools/missing_experiments"
OUTPUT = ROOT / "docs/papers/shared/results/paper6_5_tools/final_curves"
POLICIES = ("A1_fused", "A2_raw_union", "A3_diversity_union")
K_VALUES = (12, 16, 24, 32)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [
        row for row in _read_jsonl(MISSING / "large_catalog_palettes.jsonl")
        if int(row["catalog_size"]) == 8192
        and row["policy"] in args.policies
        and int(row["max_candidates"]) in K_VALUES
    ]
    needed = {
        (int(row["catalog_size"]), int(row["seed"]), name)
        for row in selected for name in row["candidate_names"]
    }
    views = {
        (int(row["catalog_size"]), int(row["seed"]), row["name"]): row
        for row in _read_jsonl(MISSING / "progressive_tool_views.jsonl")
        if (int(row["catalog_size"]), int(row["seed"]), row["name"]) in needed
    }
    checkpoint_path = args.output / "tool_choice_k_curve_checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists() and not args.fresh else {"rows": []}
    )
    rows = [
        row for row in checkpoint.get("rows", ()) if row["policy"] in args.policies
    ]
    completed = {
        (int(row["seed"]), row["query_id"], row["policy"], int(row["max_candidates"]))
        for row in rows
    }
    model = OllamaLabelModel(args.ollama_model)
    for index, palette in enumerate(selected, start=1):
        key = (
            int(palette["seed"]), palette["query_id"], palette["policy"],
            int(palette["max_candidates"]),
        )
        if key in completed:
            continue
        view_rows = [
            views[(8192, int(palette["seed"]), name)]
            for name in palette["candidate_names"]
        ]
        payload = "\n".join(str(row["selection_payload"]) for row in view_rows)
        if palette["target_in_palette"]:
            for attempt in range(3):
                try:
                    chosen, costs = model.choose(
                        _choice_prompt(str(palette["query"]), payload, "tool"),
                        palette["candidate_names"],
                    )
                    break
                except urllib.error.HTTPError as error:
                    detail = error.read().decode("utf-8", errors="replace")
                    if attempt == 2:
                        raise RuntimeError(f"Ollama failed after three attempts: {detail}") from error
                    print(f"Ollama retry {attempt + 1}/2: {detail}", flush=True)
                    time.sleep(5 * (attempt + 1))
        else:
            chosen = ""
            costs = {
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "batch_wall_seconds": 0.0,
                "amortized_wall_seconds": 0.0,
                "ttft_seconds_upper_bound": 0.0,
                "batch_size": 0,
                "ttft_method": "not_called_target_absent",
                "selected_mean_log_probability": "",
                "choice_margin": "",
                "raw_label_response": "",
            }
        rows.append({
            "catalog_size": 8192,
            "seed": palette["seed"],
            "scoring_device": str(model.device),
            "query_id": palette["query_id"],
            "policy": palette["policy"],
            "max_candidates": palette["max_candidates"],
            "candidate_view": "compact",
            "target_name": palette["target_name"],
            "target_in_palette": palette["target_in_palette"],
            "chosen_tool": chosen,
            "choice_correct": int(chosen == palette["target_name"]),
            "conditional_choice_denominator": palette["target_in_palette"],
            "candidate_count": palette["candidate_count"],
            "candidate_names": "|".join(palette["candidate_names"]),
            "unsafe_candidates": "|".join(palette["unsafe_candidates"]),
            "unsafe_exposure": int(bool(palette["unsafe_candidates"])),
            "unsafe_choice": int(chosen in set(palette["unsafe_candidates"])),
            "useful_candidates": "|".join(palette["useful_candidates"]),
            "selection_view_tokens": sum(int(row["selection_tokens"]) for row in view_rows),
            "all_candidate_full_tokens": sum(int(row["full_tokens"]) for row in view_rows),
            **costs,
            "generated_text": json.dumps({"capability": chosen}, separators=(",", ":")),
        })
        _write_checkpoint(checkpoint_path, {
            "protocol": {
                "catalog_size": 8192,
                "seeds": [11, 23, 37, 53, 71],
                "queries": 8,
                "policies": list(args.policies),
                "k_values": list(K_VALUES),
                "backend": args.ollama_model,
                "temperature": 0,
            },
            "rows": rows,
        })
        print(f"choice {len(rows)}/{len(selected)} {key} correct={chosen == palette['target_name']}", flush=True)

    existing_path = MISSING / "large_catalog_palette_choice.csv"
    existing = []
    if existing_path.exists():
        with existing_path.open(newline="", encoding="utf-8") as stream:
            existing = [
                dict(row) for row in csv.DictReader(stream)
                if int(row["catalog_size"]) == 8192
                and row["policy"] in POLICIES
                and int(row["max_candidates"]) <= 10
                and row["candidate_view"] == "compact"
            ]
    _write_csv(args.output / "tool_choice_k_curve_raw.csv", [*existing, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ollama-model", default="qwen3:0.6b")
    parser.add_argument(
        "--policies", nargs="+", choices=POLICIES, default=("A3_diversity_union",)
    )
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
