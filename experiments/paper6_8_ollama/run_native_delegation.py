"""Compare stock Ollama E0 with negotiated llama.cpp native E2 delegation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

from pra_hf.deployment import PRAWireRequest, PRAWireResource
from pra_ollama import OllamaEngineAdapter, OllamaLlamaCppBackendExecutor


CASES = (
    (
        "code",
        "The launch code is CERULEAN-7.\n",
        "What is the launch code?",
        "The launch code is",
        "cerulean-7",
    ),
    (
        "capital",
        "The capital of North Veridia is Lumenport.\n",
        "What is the capital of North Veridia?",
        "The capital of North Veridia is",
        "lumenport",
    ),
    (
        "owner",
        "The Atlas service is maintained by Priya Nair.\n",
        "Who maintains the Atlas service?",
        "The Atlas service is maintained by",
        "priya nair",
    ),
    (
        "date",
        "Project Glasswing launches on 17 October 2031.\n",
        "When does Project Glasswing launch?",
        "Project Glasswing launches on",
        "17 october 2031",
    ),
    (
        "numeric",
        "The approved pressure limit is 47 kilopascals.\n",
        "What is the approved pressure limit?",
        "The approved pressure limit is",
        "47 kilopascals",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_get(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _request(
    model: str,
    case_id: str,
    resource: str,
    question: str,
    completion_prompt: str,
    repeat: int,
) -> PRAWireRequest:
    return PRAWireRequest(
        request_id=f"{case_id}-{repeat}",
        tenant_id="paper6.8",
        session_id="delegation",
        model=model,
        messages=(
            {
                "role": "user",
                "content": question + " Answer with only the requested value.",
            },
        ),
        resources=(
            PRAWireResource(
                resource_id=case_id,
                uri=f"memory://paper6.8/{case_id}",
                text=resource,
            ),
        ),
        max_new_tokens=12,
        engine_hints={"prompt": completion_prompt, "temperature": 0},
    )


def _timed(adapter: OllamaEngineAdapter, request: PRAWireRequest):
    started = time.perf_counter()
    result = adapter.generate(request)
    return result, (time.perf_counter() - started) * 1000.0


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def run(args: argparse.Namespace) -> dict[str, object]:
    artifact_digest = _sha256(args.model_blob)
    if artifact_digest != args.model_blob_sha256:
        raise RuntimeError(
            f"Native model blob digest mismatch: {artifact_digest} != "
            f"{args.model_blob_sha256}"
        )
    manifest = json.loads(args.ollama_manifest.read_text(encoding="utf-8"))
    model_layers = [
        row for row in manifest.get("layers", ())
        if row.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    if len(model_layers) != 1 or model_layers[0].get("digest") != f"sha256:{artifact_digest}":
        raise RuntimeError("Ollama manifest does not reference the native model blob.")
    backend = OllamaLlamaCppBackendExecutor(
        args.native_url,
        backend_revision=args.backend_revision,
        model_artifact_digest=artifact_digest,
        resource_slot=args.resource_slot,
        request_slot=args.request_slot,
    )
    stock = OllamaEngineAdapter(args.ollama_url, keep_alive="10m")
    native = OllamaEngineAdapter(
        args.ollama_url, backend_executor=backend, keep_alive="10m"
    )
    tags = _json_get(f"{args.ollama_url.rstrip('/')}/api/tags")
    model_row = next(row for row in tags.get("models", []) if row.get("name") == args.model)

    rows: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        for case_id, resource, question, completion_prompt, answer in CASES:
            request = _request(
                args.model, case_id, resource, question, completion_prompt, repeat
            )
            e0, e0_ms = _timed(stock, request)
            e2, e2_ms = _timed(native, request)
            e2_warm, e2_warm_ms = _timed(native, request)
            rows.append(
                {
                    "repeat": repeat,
                    "case_id": case_id,
                    "answer": answer,
                    "resource_sha256": hashlib.sha256(resource.encode()).hexdigest(),
                    "e0_text": e0.text,
                    "e2_text": e2.text,
                    "e2_warm_text": e2_warm.text,
                    "e0_answer_correct": answer in e0.text.lower(),
                    "e2_answer_correct": answer in e2.text.lower(),
                    "e2_warm_answer_correct": answer in e2_warm.text.lower(),
                    "e0_ms": e0_ms,
                    "e2_ms": e2_ms,
                    "e2_warm_ms": e2_warm_ms,
                    "e0_prompt_tokens": e0.raw.get("prompt_eval_count"),
                    "e2_wire_tokens": e2.raw.get("pra", {}).get("wire_tokens"),
                    "e2_native_tokens": e2.raw.get("pra", {}).get("native_tokens"),
                    "e2_tokens": e2.raw.get("tokens", []),
                    "e2_warm_tokens": e2_warm.raw.get("tokens", []),
                    "resolved_level": native.capabilities().integration_level.value,
                }
            )
    native.close_session("delegation")

    e0_ms = [float(row["e0_ms"]) for row in rows]
    e2_ms = [float(row["e2_ms"]) for row in rows]
    warm_ms = [float(row["e2_warm_ms"]) for row in rows]
    summary = {
        "runs": len(rows),
        "negotiated_e2": sum(row["resolved_level"] == "E2" for row in rows),
        "e0_answer_correct": sum(bool(row["e0_answer_correct"]) for row in rows),
        "e2_answer_correct": sum(bool(row["e2_answer_correct"]) for row in rows),
        "e2_warm_answer_correct": sum(
            bool(row["e2_warm_answer_correct"]) for row in rows
        ),
        "e2_warm_exact": sum(row["e2_tokens"] == row["e2_warm_tokens"] for row in rows),
        "mean_e0_ms": statistics.mean(e0_ms),
        "mean_e2_ms": statistics.mean(e2_ms),
        "mean_e2_warm_ms": statistics.mean(warm_ms),
        "e2_over_e0": statistics.mean(e2_ms) / statistics.mean(e0_ms),
        "e2_warm_over_e0": statistics.mean(warm_ms) / statistics.mean(e0_ms),
        "e0_p95_ms": _percentile(e0_ms, 0.95),
        "e2_p95_ms": _percentile(e2_ms, 0.95),
        "e2_warm_p95_ms": _percentile(warm_ms, 0.95),
    }
    payload = {
        "schema_version": "paper6.8-ollama-native-delegation-v1",
        "experiment": "stock_ollama_e0_vs_negotiated_llamacpp_e2",
        "evidence_tier": "LIVE_PRODUCT_DELEGATION_COHORT",
        "configuration": {
            "ollama_url": args.ollama_url,
            "native_url": args.native_url,
            "model": args.model,
            "ollama_model_digest": model_row.get("digest"),
            "model_blob": str(args.model_blob),
            "ollama_manifest": str(args.ollama_manifest),
            "model_blob_sha256": artifact_digest,
            "same_model_artifact_verified": True,
            "backend_revision": args.backend_revision,
            "repeats": args.repeats,
        },
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--native-url", default="http://127.0.0.1:18087")
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--backend-revision", default="458681e1d5d")
    parser.add_argument("--model-blob", type=Path, required=True)
    parser.add_argument("--model-blob-sha256", required=True)
    parser.add_argument("--ollama-manifest", type=Path, required=True)
    parser.add_argument("--resource-slot", type=int, default=0)
    parser.add_argument("--request-slot", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args())["summary"], indent=2))
