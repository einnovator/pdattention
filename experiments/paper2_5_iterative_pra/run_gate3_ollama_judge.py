"""Run a supplementary local Ollama judge over the canonical blind package."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from experiments.paper2_hf.score_behavioral_judge_results import score_response


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[row["item_id"]] = row
    return rows


def _judge_item(args, prompt: str, item: dict, schema: dict) -> dict:
    request = {
        "model": args.model,
        "stream": False,
        # Older local Ollama runners can exhaust memory while compiling the full
        # array schema. Canonical validation still runs on every completed file.
        "format": "json",
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Evaluate exactly this one blinded item and return one response item:\n"
                + json.dumps(item, ensure_ascii=False),
            },
        ],
        "options": {
            "temperature": 0,
            "seed": args.seed,
            "num_ctx": args.num_ctx,
        },
    }
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            response = _post(args.url, request, args.timeout)
            parsed = json.loads(response["message"]["content"])
            rows = parsed.get("items", [])
            if len(rows) != 1 or rows[0].get("item_id") != item["item_id"]:
                raise ValueError("judge returned the wrong item identity")
            parsed["schema_version"] = "1.0"
            parsed["judge_name"] = args.model
            return rows[0]
        except (KeyError, ValueError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(attempt)
    raise RuntimeError(f"judge failed {item['item_id']} after {args.retries} attempts") from last_error


def run(args: argparse.Namespace) -> dict:
    package = json.loads(args.items.read_text(encoding="utf-8"))
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    prompt = args.prompt.read_text(encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_checkpoint(args.checkpoint)
    for index, item in enumerate(package["items"], start=1):
        if item["item_id"] in completed:
            continue
        row = _judge_item(args, prompt, item, schema)
        with args.checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        completed[row["item_id"]] = row
        print(f"[local-judge {index}/{len(package['items'])}] {row['item_id']}", flush=True)
    response = {
        "schema_version": package["schema_version"],
        "judge_name": args.model,
        "items": [completed[item["item_id"]] for item in package["items"]],
    }
    scored = score_response(response, truth)
    control_rows = [
        row
        for row in scored["aggregates"]
        if row["comparison_group"].startswith("calibration_")
    ]
    identical = [row for row in control_rows if row["comparison_group"] == "calibration_identical"]
    corrupted = [row for row in control_rows if row["comparison_group"] == "calibration_corrupted"]
    validation = {
        "identity_semantic_equivalence_ge_90": bool(identical) and all(
            row["semantic_equivalence_mean"] >= 90 for row in identical
        ),
        "identity_relative_quality_abs_le_10": bool(identical) and all(
            abs(row["relative_quality_target_mean"]) <= 10 for row in identical
        ),
        "corrupted_answer_disfavored": bool(corrupted) and all(
            row["relative_quality_target_mean"] <= -20 for row in corrupted
        ),
        "order_direction_agreement_ge_80pct": (
            scored["order_reversal"]["relative_quality_direction_agreement"] >= 0.80
        ),
        "approved_external_sota_judge": False,
    }
    validation["protocol_controls_passed"] = all(
        value for key, value in validation.items() if key != "approved_external_sota_judge"
    )
    validation["headline_eligible"] = bool(
        validation["protocol_controls_passed"]
        and validation["approved_external_sota_judge"]
    )
    scored["validation"] = validation
    args.output.write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.scored_output.write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    return scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path("docs/papers/shared/results/paper2_5_iterative_pra/output_validation/behavioral_judge")
    parser.add_argument("--items", type=Path, default=default / "behavioral_judge_items.json")
    parser.add_argument("--truth", type=Path, default=default / "behavioral_judge_truth.json")
    parser.add_argument("--schema", type=Path, default=default / "behavioral_judge_response.schema.json")
    parser.add_argument("--prompt", type=Path, default=default / "behavioral_judge_prompt.txt")
    parser.add_argument("--checkpoint", type=Path, default=default / "ollama_judge_checkpoint.jsonl")
    parser.add_argument("--output", type=Path, default=default / "ollama_judge_response.json")
    parser.add_argument("--scored-output", type=Path, default=default / "ollama_judge_scored.json")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--seed", type=int, default=2505)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"judge": result["judge_name"], "pairs": result["underlying_pair_count"]}, indent=2))
