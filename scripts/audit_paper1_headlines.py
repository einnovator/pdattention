#!/usr/bin/env python3
"""Audit Paper 1 headline numbers against its frozen JSON artifacts.

This script performs no training or inference.  It verifies the values quoted in the
abstract, headline table, and principal results sections, records artifact hashes, and
writes a machine-readable receipt beside the paper.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results"
OUTPUT = ROOT / "docs" / "papers" / "paper1_standalone_pra" / "NUMERICAL_AUDIT.json"

ARTIFACTS = {
    "fragmentation": RESULTS / "pra_parameter_sensitivity_fragmentation.json",
    "scale": RESULTS / "pra_scale_sensitivity.json",
    "bounded": RESULTS / "pra_head_beyond_native_context.json",
    "residency": RESULTS / "pra_kv_residency.json",
    "routing": RESULTS / "pra_routing_index_reuse.json",
    "pretrained": RESULTS
    / "paper1_standalone_pra"
    / "mac_context_dilution"
    / "mlx_long_context_summary.json",
}


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def row(rows: list[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    matches = [entry for entry in rows if all(entry.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {keys}, found {len(matches)}")
    return matches[0]


checks: list[dict[str, Any]] = []


def check(name: str, observed: Any, expected: Any, tolerance: float = 0.0) -> None:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        passed = abs(float(observed) - float(expected)) <= tolerance
    else:
        passed = observed == expected
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "passed": passed,
        }
    )


fragmentation = load(ARTIFACTS["fragmentation"])["aggregate"]
independent_id = "fragmentation-k2-mean-g1-max-independent-b1-o0p0-local"
block_id = "fragmentation-k2-mean-g1-max-block_slice-b8-o0p0-local"
native_id = "fragmentation-k2-mean-g1-max-native_slice-b256-o0p0-local"
for dataset, before, repaired, routed in (
    ("hotpotqa", 0.546, 0.996, 0.876),
    ("qasper", 0.748, 0.999, 0.994),
):
    check(
        f"fragmentation.{dataset}.independent_oracle_rcb_3dp",
        round(row(fragmentation, dataset=dataset, config_id=independent_id)["native_oracle_rcb_mean"], 3),
        before,
    )
    check(
        f"fragmentation.{dataset}.block8_oracle_rcb_3dp",
        round(row(fragmentation, dataset=dataset, config_id=block_id)["native_oracle_rcb_mean"], 3),
        repaired,
    )
    check(
        f"fragmentation.{dataset}.native_slice_routed_rcb_3dp",
        round(row(fragmentation, dataset=dataset, config_id=native_id)["routed_rcb_mean"], 3),
        routed,
    )

scale = load(ARTIFACTS["scale"])["aggregate"]
small = [entry for entry in scale if entry["model_tier"] == "small" and entry["top_k_references"] == 8]
for dataset, active64, active256, fixed_fraction in (
    ("hotpotqa", 0.179, 0.044, (0.756, 0.772, 0.762)),
    ("qasper", 0.187, 0.049, (0.794, 0.803, 0.800)),
):
    r64 = row(small, dataset=dataset, split_count=64)
    r128 = row(small, dataset=dataset, split_count=128)
    r256 = row(small, dataset=dataset, split_count=256)
    check(f"scale.{dataset}.active_fraction_split64_3dp", round(r64["active_fraction_mean"], 3), active64)
    check(f"scale.{dataset}.active_fraction_split256_3dp", round(r256["active_fraction_mean"], 3), active256)
    for split_row, cutoff, expected in zip((r64, r128, r256), (8, 16, 32), fixed_fraction):
        check(
            f"scale.{dataset}.fixed_fraction_coverage_split{split_row['split_count']}_3dp",
            round(split_row[f"fraction_targets_covered_at_{cutoff}_mean"], 3),
            expected,
        )
    physical = [r64["retrieved_physical_kv_tokens_mean"], r128["retrieved_physical_kv_tokens_mean"], r256["retrieved_physical_kv_tokens_mean"]]
    check(f"scale.{dataset}.physical_kv_min_ge_10", min(physical) >= 10.0, True)
    check(f"scale.{dataset}.physical_kv_max_le_13", max(physical) <= 13.0, True)

bounded_doc = load(ARTIFACTS["bounded"])
bounded = bounded_doc["aggregate"]
routed192 = row(bounded, condition="head_routed", total_prompt_tokens=192)
trunc192 = row(bounded, condition="direct_truncation", total_prompt_tokens=192)
check("bounded.routed_loss_4dp", round(routed192["loss_mean"], 4), 1.0567)
check("bounded.truncation_loss_4dp", round(trunc192["loss_mean"], 4), 1.2287)
check("bounded.logical_tokens", routed192["total_prompt_tokens"], 192)
check("bounded.direct_tokens", bounded_doc["manifest"]["direct_context_tokens"], 8)
check("bounded.displaced_head_tokens", routed192["total_prompt_tokens"] - bounded_doc["manifest"]["direct_context_tokens"], 184)
check("bounded.max_source_input", max(entry["max_encoding_input_tokens"] for entry in bounded_doc["raw"]), 20)
check("bounded.max_retained_memory", max(entry["memory_tokens_materialized"] for entry in bounded_doc["raw"]), 20)
check(
    "bounded.max_combined_attention_kv",
    max(entry["direct_tail_tokens"] + entry["memory_tokens_materialized"] for entry in bounded_doc["raw"]),
    28,
)

routing = load(ARTIFACTS["routing"])["aggregate"]
routing256 = [entry for entry in routing if entry["split_count"] == 256]
warm_ms = [entry["tensorized_warm_index_ms_mean"] for entry in routing256]
speedup = [entry["warm_speedup_mean"] for entry in routing256]
check("routing.split256.warm_min_1dp", round(min(warm_ms), 1), 5.5)
check("routing.split256.warm_max_1dp", round(max(warm_ms), 1), 11.6)
check("routing.split256.speedup_min_0dp", round(min(speedup)), 94)
check("routing.split256.speedup_max_0dp", round(max(speedup)), 103)

residency = load(ARTIFACTS["residency"])["aggregate"]
cpu256 = [entry for entry in residency if entry["split_count"] == 256 and entry["kv_cache_residency"] == "cpu"]
gpu256 = [entry for entry in residency if entry["split_count"] == 256 and entry["kv_cache_residency"] == "gpu"]
cache_mib = [entry["cached_kv_bytes_mean"] / (1024**2) for entry in cpu256]
transfer_kib = [entry["selected_kv_transfer_bytes_mean"] / 1024 for entry in cpu256]
transfer_ms = [entry["selected_kv_transfer_ms_mean"] for entry in cpu256]
gpu_by_key = {(entry["dataset"], entry["model_tier"]): entry for entry in gpu256}
warm_overhead = [
    entry["warm_request_ms_mean"] - gpu_by_key[(entry["dataset"], entry["model_tier"])]["warm_request_ms_mean"]
    for entry in cpu256
]
check("residency.cache_mib_min_3dp", round(min(cache_mib), 3), 0.734)
check("residency.cache_mib_max_3dp", round(max(cache_mib), 3), 3.420)
check("residency.transfer_kib_min_1dp", round(min(transfer_kib), 1), 24.6)
check("residency.transfer_kib_max_1dp", round(max(transfer_kib), 1), 87.9)
check("residency.transfer_ms_min_1dp", round(min(transfer_ms), 1), 4.0)
check("residency.transfer_ms_max_1dp", round(max(transfer_ms), 1), 8.9)
check("residency.warm_overhead_ms_min_1dp", round(min(warm_overhead), 1), 4.0)
check("residency.warm_overhead_ms_max_1dp", round(max(warm_overhead), 1), 13.5)

pretrained = load(ARTIFACTS["pretrained"])["comparisons"]
q8_8k = row(pretrained, model_id="mlx-community/Qwen3-8B-4bit", context_target_tokens=8192)
q8_32k = row(pretrained, model_id="mlx-community/Qwen3-8B-4bit", context_target_tokens=32768)
q14_8k = row(pretrained, model_id="mlx-community/Qwen3-14B-4bit", context_target_tokens=8192)
q32_8k = row(pretrained, model_id="mlx-community/Qwen3-32B-4bit", context_target_tokens=8192)
check("pretrained.8b_8k.sequence_agreement", q8_8k["selected_native_sequence_agreement"], 1.0)
check("pretrained.14b_8k.sequence_agreement", q14_8k["selected_native_sequence_agreement"], 1.0)
check("pretrained.32b_8k.sequence_agreement", q32_8k["selected_native_sequence_agreement"], 13 / 15, 1e-12)
check("pretrained.8b_selected_mib_1dp", round(q8_8k["selected_native_active_detail_bytes"] / (1024**2), 1), 39.6)
check("pretrained.8b_8k_full_mib", q8_8k["full_native_active_detail_bytes"] / (1024**2), 1152.0)
check("pretrained.8b_32k_full_mib", q8_32k["full_native_active_detail_bytes"] / (1024**2), 4608.0)
check("pretrained.8b_8k_logprob_delta_3dp", round(q8_8k["full_minus_selected_gold_logprob"], 3), 1.192)
check("pretrained.14b_8k_logprob_delta_3dp", round(q14_8k["full_minus_selected_gold_logprob"], 3), -1.485)
check("pretrained.32b_8k_logprob_delta_3dp", round(q32_8k["full_minus_selected_gold_logprob"], 3), -0.999)

revision = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
).stdout.strip()
artifact_receipts = {
    name: {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for name, path in ARTIFACTS.items()
}
failed = [entry for entry in checks if not entry["passed"]]
receipt = {
    "schema_version": "paper1_headline_audit_v1",
    "audit_date": "2026-09-13",
    "repository_revision_before_editorial_changes": revision,
    "scope": "Frozen-artifact verification only; no training or inference rerun.",
    "status": "PASS" if not failed else "FAIL",
    "artifacts": artifact_receipts,
    "checks": checks,
}
OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(f"{receipt['status']}: {len(checks)} checks; receipt={OUTPUT.relative_to(ROOT)}")
if failed:
    for entry in failed:
        print(f"FAILED {entry['name']}: observed={entry['observed']} expected={entry['expected']}")
    raise SystemExit(1)
