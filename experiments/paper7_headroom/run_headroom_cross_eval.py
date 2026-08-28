"""Run released Headroom on the frozen Paper 7 workload and merge frozen controls."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/headroom_cross_eval"
OFFICIAL_PYTHON = Path(r"D:\git\rd\.venv-headroom-037\Scripts\python.exe")
HEADROOM_COMMIT = "32d7ca4577d599b8a5f811ada74cf31504302c9d"
HEADROOM_VERSION = "0.37.0"
SEEDS = (11, 23, 37, 53, 71)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in values for key in row}) if values else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _run_official_worker(cases: Path, output: Path, *args: str) -> None:
    command = [
        str(OFFICIAL_PYTHON),
        str(Path(__file__).with_name("headroom_official_worker.py")),
        "--input",
        str(cases),
        "--output",
        str(output),
        *args,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


class OllamaController:
    """Small deterministic JSON controller using the frozen local model."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path.is_file()
            else {}
        )

    def decide(self, prompt: str, seed: int) -> tuple[str, float, int]:
        request = {
            "model": "qwen3:0.6b",
            "stream": False,
            "messages": [
                {"role": "system", "content": "Return one valid JSON object only. /no_think"},
                {"role": "user", "content": prompt + "\n/no_think"},
            ],
            "options": {"temperature": 0, "seed": seed, "num_predict": 48},
            "keep_alive": "30m",
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is None:
            started = time.perf_counter()
            wire = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps(request).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(wire, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - started
            raw = str(body.get("message", {}).get("content", "{}"))
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                start, stop = raw.find("{"), raw.rfind("}")
                try:
                    value = json.loads(raw[start : stop + 1]) if start >= 0 < stop else {}
                except json.JSONDecodeError:
                    value = {}
            cached = {"value": value, "raw": raw, "latency_seconds": elapsed}
            self.cache[key] = cached
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8"
            )
        action = str(cached["value"].get("decision", "CONTINUE")).upper()
        if action not in {"CONTINUE", "RETRIEVE", "CALL_TOOL"}:
            action = "CONTINUE"
        prompt_tokens = math.ceil(len(prompt.encode("utf-8")) / 4)
        return action, float(cached["latency_seconds"]), prompt_tokens


def _expected_decision(row: Mapping[str, Any]) -> str:
    if int(row["evidence_visible_initially"]):
        return "CONTINUE"
    if not int(row["backing_contains_evidence"]):
        return "CALL_TOOL"
    return "RETRIEVE"


def _decision_prompt(row: Mapping[str, Any]) -> str:
    answer = str(row["expected_answer"])
    visible_codes = [answer] if answer.casefold() in str(row["compressed"]).casefold() else []
    external = not bool(int(row["backing_contains_evidence"]))
    return (
        "You control reversible context for one typed tool result. Apply this exact policy: "
        "if Schema-parsed ANSWER_CODE values is non-empty choose CONTINUE; otherwise if the "
        "record says external lookup is required choose CALL_TOOL; otherwise choose RETRIEVE. "
        "RETRIEVE resolves only the retained original and cannot acquire absent information.\n"
        f"Task: {row['query']}\n"
        f"Schema-parsed ANSWER_CODE values: {json.dumps(visible_codes)}\n"
        f"External lookup required: {json.dumps(external)}\n"
        f"CCR marker hashes: {json.dumps(row['hashes'])}\n"
        'Output exactly {"decision":"CONTINUE|RETRIEVE|CALL_TOOL"}.'
    )


def _official_rows(
    profile_rows: list[dict[str, Any]],
    condition: str,
    controller: OllamaController,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for row in profile_rows:
        if row["partition"] != "test":
            continue
        expected = _expected_decision(row)
        for seed in SEEDS:
            predicted, latency, prompt_tokens = controller.decide(_decision_prompt(row), seed)
            tool_payload = "" if expected != "CALL_TOOL" else str(row["expected_answer"])
            final_text = str(row["compressed"])
            retrieved_tokens = 0
            tool_calls = 0
            if predicted == "RETRIEVE":
                final_text += "\n" + "\n".join(row["retrieved_originals"])
                retrieved_tokens = int(row["retrieved_tokens"])
                tool_calls = 1
            elif predicted == "CALL_TOOL":
                final_text += "\n" + tool_payload
                retrieved_tokens = math.ceil(len(tool_payload.encode("utf-8")) / 4)
                tool_calls = 1
            success = int(str(row["expected_answer"]).casefold() in final_text.casefold())
            results.append({
                "condition": condition,
                "source": "HEADROOM_OFFICIAL",
                "execution_scope": row["execution_scope"],
                "case_id": row["case_id"],
                "case_class": row["case_class"],
                "partition": row["partition"],
                "seed": seed,
                "expected_action": expected,
                "predicted_action": predicted,
                "trigger_needed": int(expected != "CONTINUE"),
                "trigger_called": int(predicted != "CONTINUE"),
                "trigger_correct": int(predicted == expected),
                "evidence_visible_initially": int(row["evidence_visible_initially"]),
                "evidence_visible_final": success,
                "task_success": success,
                "initial_visible_tokens": int(row["compressed_tokens"]),
                "active_tokens": int(row["compressed_tokens"]) + retrieved_tokens,
                "retrieved_tokens": retrieved_tokens,
                "controller_tokens": prompt_tokens,
                "original_tokens": int(row["original_tokens"]),
                "backing_bytes": int(row["original_bytes"]),
                "index_bytes": int(row["compressed_bytes"]),
                "ingestion_seconds": float(row["compression_seconds"]),
                "retrieval_seconds": float(row["retrieval_seconds"]) if predicted == "RETRIEVE" else 0.0,
                "controller_seconds": latency,
                "total_latency_seconds": float(row["compression_seconds"]) + latency,
                "model_calls": 1,
                "tool_calls": tool_calls,
                "status": row["status"],
                "notes": f"profile={row['profile']}; markers={row['marker_count']}",
            })
            oracle_success = int(
                bool(row["evidence_visible_initially"])
                or bool(row["evidence_visible_after_retrieve"])
                or expected == "CALL_TOOL"
            )
            oracle.append({
                "condition": f"{condition}_ORACLE",
                "case_id": row["case_id"],
                "case_class": row["case_class"],
                "seed": seed,
                "oracle_action": expected,
                "task_success": oracle_success,
                "retrieval_possible": int(row["evidence_visible_after_retrieve"]),
                "external_acquisition_required": int(expected == "CALL_TOOL"),
            })
    return results, oracle


def _frozen_rows() -> list[dict[str, Any]]:
    path = (
        ROOT
        / "docs/papers/shared/results/paper7_records/full_pra_calibrated/adaptive_oracle_results.csv"
    )
    mapping = {
        "PRA_NATIVE": "PRA_FROZEN",
        "FULL": "FULL_BACKING",
        "COMPACT_ONLY": "COMPACT_ONLY",
        "CCR_TOOL": "CCR_STYLE",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            if source["partition"] != "test" or source["policy"] not in mapping:
                continue
            policy = source["policy"]
            predicted = source.get("predicted_action") or "CONTINUE"
            rows.append({
                "condition": mapping[policy],
                "source": "PAPER7_FROZEN" if policy != "CCR_TOOL" else "CCR_STYLE",
                "execution_scope": "frozen_paper7_runtime",
                "case_id": source["case_id"],
                "case_class": source["case_class"],
                "partition": source["partition"],
                "seed": int(source["seed"]),
                "expected_action": source["expected_action"],
                "predicted_action": predicted,
                "trigger_needed": int(source["expected_action"] != "CONTINUE"),
                "trigger_called": int(predicted != "CONTINUE"),
                "trigger_correct": int(source["operation_correct"]),
                "evidence_visible_initially": int(source["final_use_given_visible"]),
                "evidence_visible_final": int(source["evidence_visible"]),
                "task_success": int(source["task_success"]),
                "initial_visible_tokens": int(source["compact_tokens_estimate"]),
                "active_tokens": int(source["active_kv_tokens"]),
                "retrieved_tokens": int(source["materialized_tokens"]),
                "controller_tokens": int(source["controller_prompt_tokens"]),
                "original_tokens": "",
                "backing_bytes": "",
                "index_bytes": "",
                "ingestion_seconds": float(source["ingestion_seconds"]),
                "retrieval_seconds": float(source["runtime_seconds"]),
                "controller_seconds": float(source["controller_seconds"]),
                "total_latency_seconds": (
                    float(source["routing_seconds"])
                    + float(source["runtime_seconds"])
                    + float(source["controller_seconds"])
                ),
                "model_calls": int(source["model_passes"]),
                "tool_calls": int(source["tool_roundtrips"]),
                "status": "frozen",
                "notes": f"historical_policy={policy}",
            })
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases_path = OUTPUT / "paper7_cases.json"
    subprocess.run(
        ["python", str(Path(__file__).with_name("export_paper7_cases.py")), str(cases_path)],
        cwd=ROOT,
        check=True,
    )
    profiles = {
        "default": (OUTPUT / "headroom_default_raw.json", ()),
        "max4_ccr": (OUTPUT / "headroom_max4_ccr_raw.json", ("--max-items", "4", "--without-compaction")),
        "max8_ccr": (OUTPUT / "headroom_max8_ccr_raw.json", ("--max-items", "8", "--without-compaction")),
    }
    for path, args in profiles.values():
        if not path.is_file():
            _run_official_worker(cases_path, path, *args)
    raw = {name: json.loads(path.read_text(encoding="utf-8"))["rows"] for name, (path, _) in profiles.items()}

    validation_scores: dict[str, dict[str, float]] = {}
    for name in ("max4_ccr", "max8_ccr"):
        values = [row for row in raw[name] if row["partition"] == "validation"]
        validation_scores[name] = {
            "initial_evidence": sum(row["evidence_visible_initially"] for row in values) / len(values),
            "oracle_evidence": sum(row["evidence_visible_after_retrieve"] for row in values) / len(values),
            "mean_tokens": sum(row["compressed_tokens"] for row in values) / len(values),
        }
    tuned_name = max(
        validation_scores,
        key=lambda name: (
            validation_scores[name]["oracle_evidence"],
            validation_scores[name]["initial_evidence"],
            -validation_scores[name]["mean_tokens"],
        ),
    )

    controller = OllamaController(OUTPUT / "headroom_controller_cache.json")
    official_default, oracle_default = _official_rows(
        raw["default"], "HEADROOM_OFFICIAL_DEFAULT", controller
    )
    official_tuned, oracle_tuned = _official_rows(
        raw[tuned_name], "HEADROOM_OFFICIAL_TUNED", controller
    )
    all_rows = _frozen_rows() + official_default + official_tuned
    _write_csv(OUTPUT / "headroom_on_paper7_results.csv", all_rows)
    _write_csv(OUTPUT / "headroom_oracle_results.csv", oracle_default + oracle_tuned)

    manifest = {
        "schema_version": 1,
        "headroom": {
            "repository": "https://github.com/headroomlabs-ai/headroom",
            "commit": HEADROOM_COMMIT,
            "package": f"headroom-ai=={HEADROOM_VERSION}",
            "python": "3.10.11",
            "extras": ["proxy", "evals"],
            "official_venv": str(OFFICIAL_PYTHON.parent.parent),
        },
        "matched_controller": {
            "provider": "Ollama local OpenAI-compatible server",
            "model": "qwen3:0.6b",
            "seeds": list(SEEDS),
        },
        "profiles": {
            "default": {
                "configuration": "released ContentRouterConfig defaults",
                "max_items_after_crush": 15,
                "lossless_first_compaction": True,
            },
            "candidates": validation_scores,
            "selected_tuned": tuned_name,
            "selected_tuned_configuration": {
                "max_items_after_crush": 4,
                "lossless_first_compaction": False,
            },
        },
        "compatibility": [
            {
                "path": "official component stack",
                "status": "supported",
                "details": "ContentRouter + SmartCrusher + CCR store/markers/tool schema",
            },
            {
                "path": "OpenAI-compatible local proxy",
                "status": "supported_smoke",
                "details": "qwen3:0.6b through Ollama; automatic CCR continuation observed",
            },
            {
                "path": "local Qwen HF as upstream provider",
                "status": "unsupported",
                "details": "HF model is not an HTTP provider; evaluated through matched Ollama instead",
            },
            {
                "path": "external supported API",
                "status": "not_run",
                "details": "no external provider credential used",
            },
            {
                "path": "Kompress ML",
                "status": "unavailable_in_isolated_python310",
                "details": "no PyTorch in official venv; structural routes remained operational",
            },
            {
                "path": "TOIN cold/warm",
                "status": "not_run",
                "details": "stateless local proxy smoke; no persistent learning state",
            },
        ],
        "feature_audit": {
            "ContentRouter": "exercised",
            "SmartCrusher": "exercised",
            "CCR_store_marker": "exercised",
            "retrieval_tool_schema": "exercised",
            "response_handler": "proxy_smoke_only",
            "TOIN": "available_not_measured",
            "live_zone": "proxy_smoke_only",
            "state_reset": "reset_compression_store before every case",
            "security_scope": "process-local hash store only; no tenant authorization claim",
        },
        "claim_boundary": (
            "No matched official full-CCR run was used for the primary table. Official structural "
            "components were run directly; the full proxy was checked separately on the matched local model."
        ),
    }
    (OUTPUT / "headroom_official_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"selected tuned profile: {tuned_name}; wrote {len(all_rows)} rows")


if __name__ == "__main__":
    main()
