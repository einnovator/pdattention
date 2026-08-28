"""Calibrate fixed PRA consumer profiles with corrected native-K/V replay.

The experiment injects every eligible attention layer once, publishes one
layer-native reference cache, and replays the same selected identity through
contiguous and sparse consumer profiles.  It never re-encodes selected text.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pra_hf.layer_profiles import (  # noqa: E402
    LayerSelection,
    common_calibration_candidates,
    eligible_layers,
)
from pra_torch.hf import PRAHFConfig, inject_pra  # noqa: E402
from pra_torch.memory import SelectedChunk  # noqa: E402

from run_cross_model_validation import MODEL_SPECS, SEMANTIC_CASES  # noqa: E402


RESULTS = ROOT / "docs/papers/shared/results/paper4_5_runtime/layer_profiles"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rank(logits: torch.Tensor, target_id: int) -> int:
    target = logits[..., target_id]
    return int((logits > target.unsqueeze(-1)).sum().item()) + 1


def _log_probability(logits: torch.Tensor, target_id: int) -> float:
    return float(torch.log_softmax(logits.float(), dim=-1)[0, target_id].item())


def _selected(entry, layer_ids: tuple[int, ...]) -> dict[int, list[list[SelectedChunk]]]:
    return {
        layer_id: [[
            SelectedChunk(
                entry=entry,
                chunk=chunk,
                reference_score=1.0,
                chunk_score=1.0,
                layer_id=layer_id,
                reference_rank=1,
                rank_within_reference=index,
                metadata={"selection_source": "fixed_full_reference_calibration"},
            )
            for index, chunk in enumerate(entry.layer_memory[layer_id].chunks, start=1)
        ]]
        for layer_id in layer_ids
    }


def _profile_candidates(layer_count: int) -> dict[str, LayerSelection]:
    candidates = common_calibration_candidates(layer_count)
    for count in (4, 8):
        if count <= layer_count:
            candidates[f"early_{count}"] = LayerSelection(
                mode="explicit", layers=tuple(range(count))
            )
            middle = max(0, (layer_count - count) // 2)
            candidates[f"middle_{count}"] = LayerSelection(
                mode="explicit", layers=tuple(range(middle, middle + count))
            )
    return candidates


@torch.no_grad()
def run_model(model_key: str, device_name: str) -> list[dict]:
    spec = MODEL_SPECS[model_key]
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    layer_count, allowed_layers, family = eligible_layers(model.config)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=allowed_layers,
            address_layer_ids=(allowed_layers[-1],),
            detail_kv_layer_ids=allowed_layers,
            routing_layer_ids=(allowed_layers[-1],),
            consumption_layer_ids=allowed_layers,
            model_max_context_tokens=512,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=256,
            context_safety_reserve_tokens=4,
            top_k_references=1,
            top_k_chunks_per_reference=32,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
        ),
    )
    candidates = _profile_candidates(len(allowed_layers))
    rows: list[dict] = []
    for case in SEMANTIC_CASES:
        reference_ids = tokenizer(
            case["reference"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        query_ids = tokenizer(
            case["query"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        target_ids = tokenizer(case["answer"], add_special_tokens=False).input_ids
        if not target_ids:
            continue
        target_id = int(target_ids[0])
        entry = handle.add_reference(
            f"calibration://{model_key}/{case['case_id']}",
            reference_ids,
            text=case["reference"],
        )
        handle.configure_memory_layers(set())
        full_ids = torch.cat((reference_ids, query_ids), dim=1).to(device)
        full_logits = model(full_ids, use_cache=False).logits[:, -1]
        no_memory_logits = model(query_ids.to(device), use_cache=False).logits[:, -1]
        full_lp = _log_probability(full_logits, target_id)
        no_memory_lp = _log_probability(no_memory_logits, target_id)
        route_chunks = entry.layer_memory[allowed_layers[-1]].chunks
        address_bytes = sum(
            int(chunk.metadata.get("routing_gist_bytes", 0)) for chunk in route_chunks
        )
        for name, selection in candidates.items():
            # Candidate indices are normalized over eligible topology.  Gemma's
            # allowed set is sparse in physical layer IDs, so select by ordinal.
            ordinal = selection.resolve(len(allowed_layers))
            layers = tuple(allowed_layers[index] for index in ordinal)
            fixed = _selected(entry, layers)
            handle.configure_memory_layers(set(layers), fixed_selections=fixed)
            positions = torch.arange(
                int(reference_ids.shape[1]),
                int(reference_ids.shape[1] + query_ids.shape[1]),
                device=device,
            ).unsqueeze(0)
            _sync(device)
            started = time.perf_counter()
            logits = model(
                query_ids.to(device), position_ids=positions, use_cache=False
            ).logits[:, -1]
            _sync(device)
            elapsed = time.perf_counter() - started
            detail_bytes = sum(
                int(chunk.metadata.get("detail_kv_bytes", 0))
                for layer in layers
                for chunk in entry.layer_memory[layer].chunks
            )
            lp = _log_probability(logits, target_id)
            rows.append(
                {
                    "model_key": model_key,
                    "model_id": spec.model_id,
                    "revision": spec.revision,
                    "family": family,
                    "case_id": case["case_id"],
                    "candidate": name,
                    "physical_layers": json.dumps(layers),
                    "consumer_layer_count": len(layers),
                    "consumer_layer_fraction": len(layers) / len(allowed_layers),
                    "address_layer_count": 1,
                    "address_index_bytes": address_bytes,
                    "detail_kv_bytes": detail_bytes,
                    "active_layer_tokens": int(reference_ids.shape[1]) * len(layers),
                    "latency_seconds": elapsed,
                    "target_log_probability": lp,
                    "full_context_log_probability": full_lp,
                    "no_memory_log_probability": no_memory_lp,
                    "delta_vs_full_context": lp - full_lp,
                    "delta_vs_no_memory": lp - no_memory_lp,
                    "target_rank": _rank(logits, target_id),
                    "full_context_target_rank": _rank(full_logits, target_id),
                    "max_abs_logit_error_vs_full": float(
                        (logits.float() - full_logits.float()).abs().max().item()
                    ),
                    "materialization": "full_selected_record",
                    "workload": "semantic_smoke",
                    "corrected_transport": True,
                }
            )
        handle.configure_memory_layers(set())
    del handle, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model_key"], row["candidate"]), []).append(row)
    result = []
    for (model_key, candidate), group in grouped.items():
        result.append(
            {
                "model_key": model_key,
                "family": group[0]["family"],
                "model_id": group[0]["model_id"],
                "revision": group[0]["revision"],
                "candidate": candidate,
                "physical_layers": group[0]["physical_layers"],
                "consumer_layer_count": group[0]["consumer_layer_count"],
                "consumer_layer_fraction": group[0]["consumer_layer_fraction"],
                "address_index_bytes_mean": sum(r["address_index_bytes"] for r in group) / len(group),
                "detail_kv_bytes_mean": sum(r["detail_kv_bytes"] for r in group) / len(group),
                "active_layer_tokens_mean": sum(r["active_layer_tokens"] for r in group) / len(group),
                "latency_seconds_mean": sum(r["latency_seconds"] for r in group) / len(group),
                "target_log_probability_mean": sum(r["target_log_probability"] for r in group) / len(group),
                "delta_vs_full_context_mean": sum(r["delta_vs_full_context"] for r in group) / len(group),
                "delta_vs_no_memory_mean": sum(r["delta_vs_no_memory"] for r in group) / len(group),
                "target_rank_mean": sum(r["target_rank"] for r in group) / len(group),
                "max_abs_logit_error_vs_full_max": max(r["max_abs_logit_error_vs_full"] for r in group),
                "case_count": len(group),
            }
        )
    return sorted(result, key=lambda row: (row["model_key"], row["consumer_layer_count"], row["candidate"]))


def select_profiles(aggregate_rows: list[dict]) -> dict:
    selected = {}
    for model_key in sorted({row["model_key"] for row in aggregate_rows}):
        rows = [row for row in aggregate_rows if row["model_key"] == model_key]
        quality = max(rows, key=lambda row: row["target_log_probability_mean"])
        tolerance = quality["target_log_probability_mean"] - 0.05
        sufficient = [
            row
            for row in rows
            if row["target_log_probability_mean"] >= tolerance
            and row["max_abs_logit_error_vs_full_max"] <= 0.1
        ]
        if not sufficient:
            # Partial topologies such as Gemma 3 cannot pass full-prefix logit
            # fidelity; retain the least-distorting measured profile.
            sufficient = [
                min(rows, key=lambda row: row["max_abs_logit_error_vs_full_max"])
            ]
        balanced = min(sufficient, key=lambda row: (row["active_layer_tokens_mean"], -row["target_log_probability_mean"]))
        economy = min(rows, key=lambda row: (row["active_layer_tokens_mean"], -row["target_log_probability_mean"]))
        correctness = max(rows, key=lambda row: row["consumer_layer_count"])
        selected[model_key] = {
            "reference_correctness": correctness["candidate"],
            "quality_max": quality["candidate"],
            "balanced": balanced["candidate"],
            "economy": economy["candidate"],
            "balanced_tolerance_nats": 0.05,
        }
    return selected


def write_contract_artifacts(rows: list[dict]) -> None:
    aggregate_rows = aggregate(rows)
    selections = select_profiles(aggregate_rows)
    _write_csv(RESULTS / "layer_calibration_candidates.csv", rows)
    _write_csv(RESULTS / "layer_calibration_pareto.csv", aggregate_rows)
    _write_json(RESULTS / "layer_calibration_selected_profiles.json", selections)
    for model_key in ("qwen", "llama", "gemma"):
        _write_csv(
            RESULTS / f"layer_profiles_{model_key}.csv",
            [row for row in aggregate_rows if row["model_key"] == model_key],
        )
    portability = []
    for objective in ("reference_correctness", "quality_max", "balanced", "economy"):
        portability.append(
            {
                "objective": objective,
                **{model: values.get(objective) for model, values in selections.items()},
                "portable_profile_name": len({values.get(objective) for values in selections.values()}) == 1,
            }
        )
    _write_csv(RESULTS / "layer_profile_portability.csv", portability)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PRA layer profile",
        "type": "object",
        "required": ["name", "objective", "consumption", "routing"],
        "properties": {
            "name": {"type": "string"},
            "objective": {"enum": ["reference_correctness", "quality_max", "balanced", "economy"]},
            "consumption": {"$ref": "#/$defs/layerSelection"},
            "routing": {"$ref": "#/$defs/layerSelection"},
        },
        "$defs": {
            "layerSelection": {
                "type": "object",
                "required": ["mode"],
                "properties": {
                    "mode": {"enum": ["all_layers", "last_n", "last_fraction", "evenly_spaced_n", "explicit"]},
                    "n": {"type": "integer", "minimum": 1},
                    "fraction": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "layers": {"type": "array", "items": {"type": "integer"}},
                },
            }
        },
    }
    _write_json(RESULTS / "pra_layer_profile_schema.json", schema)
    registry_path = SRC / "pra_hf/model_profiles/layer_profile_registry.json"
    _write_json(RESULTS / "pra_layer_profile_registry.json", json.loads(registry_path.read_text(encoding="utf-8")))
    _write_json(
        RESULTS / "detail_kv_encoding_policy_schema.json",
        {"enum": ["minimal", "profile_union", "all_layers", "explicit"], "default": "minimal"},
    )
    _write_json(
        RESULTS / "address_encoding_policy_schema.json",
        {"enum": ["routing_only", "all_candidate_routing_layers", "external_only", "explicit"], "default": "routing_only"},
    )
    storage_rows = [
        {
            "model_key": row["model_key"],
            "candidate": row["candidate"],
            "address_index_bytes_mean": row["address_index_bytes_mean"],
            "detail_kv_bytes_mean": row["detail_kv_bytes_mean"],
            "active_layer_tokens_mean": row["active_layer_tokens_mean"],
        }
        for row in aggregate_rows
    ]
    _write_csv(RESULTS / "layer_profile_storage_costs.csv", storage_rows)
    _write_csv(
        RESULTS / "layer_profile_switching_costs.csv",
        [
            {
                "model_key": model,
                "from_profile": "economy",
                "to_profile": "quality_max",
                "missing_detail_policy": "reencode_missing",
                "silent_downgrade_allowed": False,
            }
            for model in selections
        ],
    )
    _write_json(
        RESULTS / "layer_profile_detail_union.json",
        {
            model: {
                "profiles": values,
                "policy": "profile_union_requires_explicit_union_at_ingestion",
            }
            for model, values in selections.items()
        },
    )
    _write_csv(
        RESULTS / "partial_native_index_lifecycle.csv",
        [
            {"address_state": "NOT_BUILT", "detail_kv_state": "NOT_BUILT", "event": "created"},
            {"address_state": "BUILT", "detail_kv_state": "PARTIAL", "event": "address_then_partial_detail"},
            {"address_state": "BUILT", "detail_kv_state": "BUILT", "event": "requested_profile_ready"},
        ],
    )
    _write_json(
        RESULTS / "layer_calibration_manifest.json",
        {
            "models": sorted(selections),
            "case_count_per_model": len(SEMANTIC_CASES),
            "transport": "corrected_native_kv_post_position_shared_softmax",
            "routing_identity": "fixed_full_reference",
            "materialization": "full_selected_record",
            "selection_rule": "quality_max; balanced within 0.05 nats and max logit error <=0.1, then minimum active layer tokens",
            "limitations": "semantic smoke calibration; Paper 3 owns workload-scale scientific placement evidence",
        },
    )
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = {"qwen": "#2166ac", "llama": "#b2182b", "gemma": "#1b7837"}
    for model_key, color in colors.items():
        model_rows = [row for row in aggregate_rows if row["model_key"] == model_key]
        axis.scatter(
            [row["active_layer_tokens_mean"] for row in model_rows],
            [row["delta_vs_full_context_mean"] for row in model_rows],
            color=color,
            label=model_key,
            alpha=0.82,
        )
        for row in model_rows:
            if row["candidate"] in {
                selections[model_key]["reference_correctness"],
                selections[model_key]["balanced"],
                "last_8",
                "last_1",
            }:
                axis.annotate(
                    row["candidate"],
                    (row["active_layer_tokens_mean"], row["delta_vs_full_context_mean"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Mean active reference tokens across consumer layers")
    axis.set_ylabel("Target log probability minus visible-prefix control")
    axis.set_title("Layer-profile calibration uses fixed reference identity and width")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS / "layer_profile_quality_cost.png", dpi=180)
    figure.savefig(RESULTS / "layer_profile_quality_cost.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), action="append")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    models = args.model or list(MODEL_SPECS)
    rows = []
    if not args.finalize_only:
        for model_key in models:
            model_rows = run_model(model_key, args.device)
            _write_json(RESULTS / f"checkpoint_{model_key}.json", model_rows)
            rows.extend(model_rows)
    if set(models) != set(MODEL_SPECS):
        for model_key in MODEL_SPECS:
            checkpoint = RESULTS / f"checkpoint_{model_key}.json"
            if checkpoint.is_file() and model_key not in models:
                rows.extend(json.loads(checkpoint.read_text(encoding="utf-8")))
    if args.finalize_only:
        for model_key in MODEL_SPECS:
            checkpoint = RESULTS / f"checkpoint_{model_key}.json"
            if checkpoint.is_file():
                rows.extend(json.loads(checkpoint.read_text(encoding="utf-8")))
    if {row["model_key"] for row in rows} == set(MODEL_SPECS):
        write_contract_artifacts(rows)
    print(json.dumps({"rows": len(rows), "models": sorted({row['model_key'] for row in rows})}, indent=2))


if __name__ == "__main__":
    main()
