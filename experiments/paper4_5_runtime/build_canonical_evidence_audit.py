"""Audit canonical condition coverage across the public PRA bundle catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pra_hf.bundle_catalog import load_bundle_catalog


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/papers/shared/results/paper4_5_runtime_productization/canonical_evidence_audit.json"
CONDITIONS = ("no_pra", "pra_no_adaptor", "pra_adaptor_bundle")
FLAGSHIP_METRICS = (
    "token_f1", "exact_match", "visible_tokens", "ttft_p50_ms", "ttft_p95_ms",
    "ttft_p99_ms", "itl_p50_ms", "itl_p95_ms", "itl_p99_ms",
    "output_tokens_per_second", "requests_per_second",
    "completion_latency_mean_ms", "peak_memory_bytes", "cost_per_successful_task",
)


def build_audit(catalog: dict) -> dict:
    rows = []
    for bundle in catalog["bundles"]:
        paired_transport = "mac_scaling/" in str(bundle.get("artifact", ""))
        metrics = {}
        for metric in FLAGSHIP_METRICS:
            available = paired_transport and metric in {
                "token_f1", "exact_match", "visible_tokens", "ttft_p50_ms",
                "ttft_p95_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p95_ms",
                "itl_p99_ms", "output_tokens_per_second",
                "completion_latency_mean_ms", "peak_memory_bytes",
            }
            metrics[metric] = {
                "no_pra": "AVAILABLE_EXISTING" if available else "NEEDS_RUN",
                "pra_no_adaptor": "AVAILABLE_EXISTING" if available else "NEEDS_RUN",
                "pra_adaptor_bundle": "NEEDS_RUN",
            }
        metrics["evidence_recall"] = {
            "no_pra": "NOT_APPLICABLE",
            "pra_no_adaptor": "AVAILABLE_EXISTING" if "hf_catalog_adapters/" in str(bundle.get("artifact", "")) else "NEEDS_RUN",
            "pra_adaptor_bundle": "AVAILABLE_EXISTING" if "hf_catalog_adapters/" in str(bundle.get("artifact", "")) else "NEEDS_RUN",
        }
        rows.append({
            "bundle": bundle["repo"],
            "model": bundle["model"],
            "engine": bundle["engine"],
            "profile": bundle["profile"],
            "mode": bundle["recommendation"].split(" with ")[0],
            "artifact": bundle.get("artifact"),
            "metrics": metrics,
        })
    counts = {state: 0 for state in ("AVAILABLE_EXISTING", "NEEDS_RUN", "NOT_APPLICABLE", "BLOCKED")}
    for row in rows:
        for states in row["metrics"].values():
            for condition in CONDITIONS:
                counts[states[condition]] += 1
    return {
        "schema_version": 1,
        "catalog": catalog.get("collection"),
        "conditions": list(CONDITIONS),
        "policy": "Exact identity only; missing values are never encoded as zero.",
        "summary": counts,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_audit(load_bundle_catalog())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
