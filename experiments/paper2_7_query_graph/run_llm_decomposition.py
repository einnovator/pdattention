"""Generate the Paper 2.7 linguistic subquestion baseline with Ollama."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import write_json
from pra_hf.natural_query_facets import NaturalFacetAnnotation


SYSTEM = """You decompose multi-hop questions for retrieval. Return only JSON that matches the schema. Do not answer the question."""
TEMPLATE = """Decompose the question into the smallest set of retrieval-relevant subquestions needed to answer it. Preserve every entity, relation, comparison, and constraint. Each subquestion must be self-contained where the original wording permits. Do not answer the question.

Question: {question}"""
SCHEMA = {
    "type": "object",
    "properties": {
        "subquestions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        }
    },
    "required": ["subquestions"],
}


def _annotations(path: Path):
    return [
        NaturalFacetAnnotation.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _request(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_subquestions(raw: str) -> tuple[list[str], str | None]:
    try:
        parsed = json.loads(raw or "{}")
        return [str(value).strip() for value in parsed.get("subquestions", []) if str(value).strip()], None
    except json.JSONDecodeError as exc:
        tail = (raw or "").split('"subquestions"', 1)[-1]
        values = []
        for encoded in re.findall(r'"((?:\\.|[^"\\])*)"', tail):
            try:
                value = json.loads(f'"{encoded}"').strip()
            except json.JSONDecodeError:
                continue
            if value:
                values.append(value)
        return values, f"partial_json_recovery: {exc}" if values else f"JSONDecodeError: {exc}"


def run(args):
    annotations = _annotations(args.annotations)
    prior = {}
    if args.output.exists():
        value = json.loads(args.output.read_text(encoding="utf-8"))
        prior = {(row["dataset"], row["example_id"]): row for row in value.get("rows", [])}
    rows = []
    for index, annotation in enumerate(annotations, 1):
        key = (annotation.dataset, annotation.example_id)
        if key in prior and prior[key].get("subquestions"):
            rows.append(prior[key])
            continue
        prompt = TEMPLATE.format(question=annotation.question)
        payload = {
            "model": args.model,
            "system": SYSTEM,
            "prompt": prompt,
            "format": SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "seed": args.seed, "num_predict": args.num_predict},
        }
        started = time.perf_counter()
        error = None
        response = {}
        for attempt in range(2):
            try:
                response = _request(args.url, payload, args.timeout)
                break
            except Exception as exc:  # Retry transport/model-server failures only.
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(attempt + 1)
        subquestions, parse_note = _parse_subquestions(response.get("response", ""))
        if parse_note:
            error = parse_note
        rows.append({
            "dataset": annotation.dataset,
            "example_id": annotation.example_id,
            "split": annotation.split,
            "question": annotation.question,
            "model": args.model,
            "temperature": 0,
            "prompt": prompt,
            "raw_response": response.get("response"),
            "subquestions": subquestions,
            "subquestion_count": len(subquestions),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "prompt_eval_count": int(response.get("prompt_eval_count", 0)),
            "eval_count": int(response.get("eval_count", 0)),
            "error": error,
        })
        write_json(args.output, {"schema_version": "1.0", "system": SYSTEM, "template": TEMPLATE, "rows": rows})
        if index % 20 == 0:
            print(f"generated {index}/{len(annotations)}", flush=True)
    artifact = {
        "schema_version": "1.0",
        "system": SYSTEM,
        "template": TEMPLATE,
        "schema": SCHEMA,
        "model": args.model,
        "temperature": 0,
        "max_generated_tokens": args.num_predict,
        "seed": args.seed,
        "examples": len(rows),
        "failures": sum(not row["subquestions"] for row in rows),
        "strict_json_examples": sum(not row.get("error") for row in rows),
        "partial_json_recoveries": sum(str(row.get("error", "")).startswith("partial_json_recovery") for row in rows),
        "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
        "total_prompt_tokens": sum(row["prompt_eval_count"] for row in rows),
        "total_generated_tokens": sum(row["eval_count"] for row in rows),
        "rows": rows,
    }
    write_json(args.output, artifact)
    return {key: value for key, value in artifact.items() if key != "rows"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b-it-qat")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--num-predict", type=int, default=96)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
