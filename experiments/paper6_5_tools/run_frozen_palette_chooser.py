"""Evaluate a stronger chooser without changing Paper 6.5 retrieval results.

The runner consumes the already-published 8,192-tool candidate palettes.  It
therefore changes only the language model that discriminates among candidates.
An independent oracle-full-schema pass measures whether the same model can
construct the correct tool call once discovery and choice are removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools"
MISSING = RESULTS / "missing_experiments"
PROGRESSIVE = RESULTS / "progressive_disclosure"
OUTPUT = RESULTS / "chooser_decomposition"
LABEL_PATTERN = re.compile(r"(?<!\d)(\d+)(?!\d)")
JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choice_prompt(query: str, payload: str, names: Sequence[str]) -> str:
    """Build the exact raw-label protocol used by the 0.6B baseline."""

    return (
        "Choose the single tool that best fits the request. "
        "The bounded records below are the only valid candidates.\n\n"
        f"REQUEST:\n{query}\n\nCANDIDATE RECORDS:\n{payload}"
        "\n\nVALID LABELS:\n"
        + "\n".join(f"{index} = {name}" for index, name in enumerate(names))
        + "\nReturn only the integer label of the best candidate.\nLABEL:"
    )


def execution_prompt(query: str, full_payload: str) -> str:
    """Remove retrieval and choice, leaving only argument construction."""

    return (
        "Use the one available tool to satisfy the request. Supply every required "
        "argument. Return only JSON with keys name and arguments.\n\n"
        f"REQUEST:\n{query}\n\nAVAILABLE TOOL RECORD:\n{full_payload}\n\nJSON:"
    )


def parse_label(text: str, names: Sequence[str]) -> str:
    match = LABEL_PATTERN.search(text)
    if match is None:
        return ""
    label = int(match.group(1))
    return names[label] if 0 <= label < len(names) else ""


def parse_call(text: str) -> tuple[str, dict[str, Any]]:
    """Parse the first complete-looking JSON tool call from model output."""

    match = JSON_PATTERN.search(text)
    if match is None:
        return "", {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", {}
    name = payload.get("name", "")
    arguments = payload.get("arguments", {})
    return (
        str(name) if isinstance(name, str) else "",
        dict(arguments) if isinstance(arguments, dict) else {},
    )


def arguments_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    def normalize(value: Any) -> Any:
        return value.strip().casefold() if isinstance(value, str) else value

    return all(key in actual and normalize(actual[key]) == normalize(value) for key, value in expected.items())


class MLXGenerator:
    """Small adapter around ``mlx-lm`` generation with deterministic decoding."""

    def __init__(self, model_id: str) -> None:
        from mlx_lm import load

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)

    def generate(self, prompt: str, *, max_tokens: int) -> tuple[str, dict[str, Any]]:
        from mlx_lm import generate

        prompt_tokens = len(self.tokenizer.encode(prompt))
        started = time.perf_counter()
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        return text, {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(self.tokenizer.encode(text)),
            "wall_seconds": elapsed,
        }


def selected_palettes(k_values: Sequence[int]) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(MISSING / "large_catalog_palettes.jsonl")
        if int(row["catalog_size"]) == 8192
        and row["policy"] == "A3_diversity_union"
        and int(row["max_candidates"]) in k_values
    ]


def load_views(palettes: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    needed = {
        (int(row["seed"]), name)
        for row in palettes
        for name in row["candidate_names"]
    }
    return {
        (int(row["seed"]), str(row["name"])): row
        for row in read_jsonl(MISSING / "progressive_tool_views.jsonl")
        if int(row["catalog_size"]) == 8192
        and (int(row["seed"]), str(row["name"])) in needed
    }


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    palettes = selected_palettes(args.k_values)
    views = load_views(palettes)
    checkpoint_path = args.output / f"{args.run_name}_checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists() and not args.fresh
        else {"choice_rows": [], "execution_rows": []}
    )
    choice_rows = list(checkpoint.get("choice_rows", ()))
    execution_rows = list(checkpoint.get("execution_rows", ()))
    completed_choices = {
        (int(row["seed"]), str(row["query_id"]), int(row["max_candidates"]))
        for row in choice_rows
    }
    completed_execution = {str(row["query_id"]) for row in execution_rows}
    generator = MLXGenerator(args.model)
    model_label = args.model_label or args.model

    for palette in palettes:
        key = (int(palette["seed"]), str(palette["query_id"]), int(palette["max_candidates"]))
        if key in completed_choices:
            continue
        names = tuple(str(name) for name in palette["candidate_names"])
        if palette["target_in_palette"]:
            payload = "\n".join(views[(key[0], name)]["selection_payload"] for name in names)
            raw, costs = generator.generate(choice_prompt(str(palette["query"]), payload, names), max_tokens=8)
            chosen = parse_label(raw, names)
        else:
            raw, chosen = "", ""
            costs = {"prompt_tokens": 0, "generated_tokens": 0, "wall_seconds": 0.0}
        choice_rows.append({
            "model": model_label,
            "catalog_size": 8192,
            "seed": key[0],
            "query_id": key[1],
            "max_candidates": key[2],
            "target_name": palette["target_name"],
            "target_in_palette": int(palette["target_in_palette"]),
            "chosen_tool": chosen,
            "choice_correct": int(chosen == palette["target_name"]),
            "candidate_names": "|".join(names),
            "raw_response": raw.replace("\r", "\\r").replace("\n", "\\n"),
            **costs,
        })
        atomic_json(checkpoint_path, {
            "protocol": {"model": model_label, "k_values": args.k_values, "frozen_palettes": True},
            "choice_rows": choice_rows,
            "execution_rows": execution_rows,
        })
        print(f"choice {len(choice_rows)}/{len(palettes)} {key}", flush=True)

    cases = json.loads((PROGRESSIVE / "tool_progressive_cases.json").read_text(encoding="utf-8"))["rows"]
    seed = min(int(row["seed"]) for row in palettes)
    all_seed_views = {
        str(row["name"]): row
        for row in read_jsonl(MISSING / "progressive_tool_views.jsonl")
        if int(row["catalog_size"]) == 8192 and int(row["seed"]) == seed
    }
    for case in cases:
        if case["query_id"] in completed_execution:
            continue
        target = str(case["target_name"])
        raw, costs = generator.generate(
            execution_prompt(str(case["query"]), str(all_seed_views[target]["full_payload"])),
            max_tokens=128,
        )
        chosen, arguments = parse_call(raw)
        semantic = arguments_match(arguments, case["expected_arguments"])
        execution_rows.append({
            "model": model_label,
            "query_id": case["query_id"],
            "target_name": target,
            "chosen_tool": chosen,
            "call_parse_valid": int(bool(chosen)),
            "tool_correct": int(chosen == target),
            "arguments_correct": int(semantic),
            "execution_correct": int(chosen == target and semantic),
            "expected_arguments": json.dumps(case["expected_arguments"], sort_keys=True),
            "actual_arguments": json.dumps(arguments, sort_keys=True),
            "raw_response": raw.replace("\r", "\\r").replace("\n", "\\n"),
            **costs,
        })
        atomic_json(checkpoint_path, {
            "protocol": {"model": model_label, "k_values": args.k_values, "frozen_palettes": True},
            "choice_rows": choice_rows,
            "execution_rows": execution_rows,
        })
        print(f"execution {len(execution_rows)}/{len(cases)} {case['query_id']}", flush=True)

    write_csv(args.output / f"{args.run_name}_choice.csv", choice_rows)
    write_csv(args.output / f"{args.run_name}_execution.csv", execution_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--run-name", default="qwen3_14b_mlx")
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 16, 32])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
